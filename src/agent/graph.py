"""
LangGraph Agent 工作流
======================
构建智能家居 Agent 的核心工作流图。

架构: ReAct (Reasoning + Acting) 循环

    ┌─────────────────────────────────────────────────────┐
    │                                                     │
    │  用户输入 → [Agent 节点] → 有工具调用? ──是──→ [Tool 节点] ──┘
    │                    │                                 │
    │                    否                                │
    │                    ↓                                 │
    │                  [END]                               │
    └─────────────────────────────────────────────────────┘

关键设计决策:
  1. 使用 ToolNode（LangGraph 预置），而非自己解析 tool_calls
  2. SystemMessage 每次追加在消息列表最前面（防止 LLM 遗忘角色）
  3. 使用 MemorySaver 实现多轮对话记忆
  4. 支持升级为 SqliteSaver 实现持久化记忆
"""

from typing import Literal
from loguru import logger

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# SqliteSaver 需要 langgraph-checkpoint-sqlite 包，可选安装
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    _HAS_SQLITE = True
except ImportError:
    SqliteSaver = None  # type: ignore[assignment]
    _HAS_SQLITE = False

from langchain_openai import ChatOpenAI

from .state import AgentState
from .prompts import build_system_prompt
from ..tools import get_all_tools
from ..devices.base import DeviceRegistry
from ..config import Settings
from ..middleware.interceptors import (
    LoggingInterceptor,
    RetryInterceptor,
)


def build_llm(settings: Settings) -> ChatOpenAI:
    """
    根据配置构建 LLM 实例。

    参数:
      settings: 应用配置（包含 API Key、模型名称等）

    返回:
      配置好的 ChatOpenAI 实例

    异常:
      ValueError: API Key 无效或未配置
    """
    logger.info(
        f"初始化 LLM | model={settings.model} | "
        f"base_url={settings.base_url}"
    )
    return ChatOpenAI(
        model=settings.model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        timeout=settings.llm_timeout,
        temperature=0.3,  # 低温度确保工具调用的稳定性和一致性
        max_retries=2,    # 失败自动重试 2 次
    )


def build_graph(
    registry: DeviceRegistry,
    settings: Settings,
) -> StateGraph:
    """
    构建 LangGraph Agent 工作流图。

    这是整个项目的核心 —— 定义 Agent 的"思考-行动"循环。

    参数:
      registry: 设备注册中心（工具函数需要它来操作设备）
      settings: 应用配置

    返回:
      编译好的 LangGraph 图（可调用 .invoke() / .stream() 运行）

    使用示例:
      from src.agent.graph import build_graph
      graph = build_graph(registry, settings)
      result = graph.invoke({"messages": [HumanMessage("打开客厅灯")]}, config)
    """
    # ---- 第 1 步: 初始化 LLM ----
    llm = build_llm(settings)

    # ---- 第 2 步: 获取工具列表并绑定到 LLM ----
    tools = get_all_tools()
    llm_with_tools = llm.bind_tools(tools)
    logger.info(f"已绑定 {len(tools)} 个工具到 LLM")

    # ---- 第 3 步: 生成系统提示词 ----
    system_prompt = build_system_prompt(registry)

    # ---- 第 4 步: 定义 Agent 节点 ----
    # 这个节点是工作流的"大脑"，负责:
    #   1. 读取当前消息历史
    #   2. 调用 LLM（带工具绑定）
    #   3. LLM 返回: 要么是文本回复，要么是 tool_calls
    def agent_node(state: AgentState) -> dict:
        """
        Agent 节点: 调用 LLM 进行推理。

        每次被调用时:
          1. 确保系统提示词在最前面（维护角色一致性）
          2. 将消息历史传给 LLM
          3. 返回 LLM 的响应（文本 或 tool_calls）
        """
        messages = list(state["messages"])

        # 确保系统提示词在消息列表最前面
        from langchain_core.messages import SystemMessage
        if not messages or not isinstance(messages[0], SystemMessage):
            messages.insert(0, SystemMessage(content=system_prompt))

        logger.debug(f"Agent: 发送 {len(messages)} 条消息给 LLM")

        # 调用 LLM
        response = llm_with_tools.invoke(messages)

        # 记录决策
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_names = [tc.get("name", "?") for tc in response.tool_calls]
            logger.info(f"Agent 决策: 调用工具 → {tool_names}")
        else:
            logger.info("Agent 决策: 直接文本回复")

        return {"messages": [response]}

    # ---- 第 5 步: 构建图结构 ----
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))

    # 入口: 从 agent 开始
    workflow.set_entry_point("agent")

    # ---- 第 6 步: 路由逻辑 ----
    #    从 agent 出来后:
    #      - 如果 LLM 发出了 tool_calls → 去 tools 节点执行
    #      - 否则 → 结束
    def router(state: AgentState) -> Literal["tools", "__end__"]:
        """路由函数: 检查是否需要执行工具"""
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        return "__end__"

    workflow.add_conditional_edges(
        "agent",
        router,
        {"tools": "tools", "__end__": END},
    )

    # tools 执行完毕 → 回到 agent 继续思考
    workflow.add_edge("tools", "agent")

    # ---- 第 7 步: 编译图（含检查点记忆）----
    # 检查点让 Agent 记住对话历史，实现多轮对话
    checkpointer = _build_checkpointer(settings)
    graph = workflow.compile(checkpointer=checkpointer)

    logger.info(
        f"Agent 图构建完成 | checkpointer={checkpointer.__class__.__name__}"
    )
    return graph


def _build_checkpointer(settings: Settings):
    """
    构建检查点存储。

    优先级:
      1. SQLite 文件（持久化，跨重启保留）
      2. 内存（临时，程序重启后丢失）

    参数:
      settings: 应用配置

    返回:
      MemorySaver 或 SqliteSaver 实例
    """
    db_path = settings.memory.db_path

    if db_path and _HAS_SQLITE:
        try:
            import os as _os
            import sqlite3
            _os.makedirs(_os.path.dirname(db_path), exist_ok=True)
            conn = sqlite3.connect(db_path, check_same_thread=False)
            checkpointer = SqliteSaver(conn)
            logger.info(f"使用 SQLite 检查点 | path={db_path}")
            return checkpointer
        except Exception as e:
            logger.warning(f"SQLite 检查点初始化失败，回退到内存模式: {e}")

    logger.info("使用内存检查点（重启后记忆丢失）")
    return MemorySaver()
