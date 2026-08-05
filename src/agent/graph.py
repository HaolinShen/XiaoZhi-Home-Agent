"""
LangGraph Agent 工作流
======================
构建智能家居 Agent 的核心工作流图。

架构: ReAct + Human-in-the-loop + Planner–Executor–Verifier

    普通请求 → ReAct Agent → ToolNode
    场景请求 → ReAct Agent → interrupt → ToolNode
    复杂任务 → Planner → interrupt → Executor → Verifier → retry/replan/final

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
from langgraph.types import interrupt
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from .state import AgentState
from .prompts import build_system_prompt
from ..tools import get_all_tools, set_memory_service
from ..devices.base import DeviceRegistry
from ..config import Settings
from ..memory import create_checkpointer
from ..memory import MemoryRepository, MemoryService
from ..memory.summarizer import build_compaction_update
from .context import SpaceDirectory
from .approval import (
    approval_is_granted,
    build_approval_request,
    rejection_tool_messages,
)
from .planning import (
    ExecutionPlan,
    expected_state_for_step,
    plan_approval_payload,
    planner_prompt,
    should_use_planner,
    verify_step,
)
from .routing import classify_intent, classify_intent_fallback
from .parallel import (
    build_device_query_subgraph,
    extract_query_targets,
    should_use_parallel_query,
)
from .multi_agent import agent_for_intent, role_prompt
from .reasoning import format_memory_decision, reason_about_memories
from .observability import emit_progress
from ..knowledge import KnowledgeBase, build_knowledge_rag_subgraph


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
    space_directory: SpaceDirectory | None = None,
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
    tools_by_name = {tool.name: tool for tool in tools}
    llm_with_tools = llm.bind_tools(tools)
    device_tool_names = {
        "control_light", "control_ac", "control_tv", "control_curtain", "get_device_status"
    }
    scene_tool_names = {"activate_scene", "list_scenes"}
    memory_tool_names = {
        "save_personal_memory", "save_home_rule", "list_personal_memories",
        "update_personal_memory", "delete_personal_memory", "list_preference_candidates",
        "confirm_preference_candidate", "reject_preference_candidate", "list_memory_versions",
    }
    specialised_llms = {
        "device": llm.bind_tools([tool for tool in tools if tool.name in device_tool_names]),
        "scene": llm.bind_tools([tool for tool in tools if tool.name in scene_tool_names]),
        "memory": llm.bind_tools([tool for tool in tools if tool.name in memory_tool_names]),
        "knowledge": llm,
        "chat": llm,
    }
    logger.info(f"已绑定 {len(tools)} 个工具到 LLM")

    # ---- 第 3 步: 生成系统提示词 ----
    system_prompt = build_system_prompt(registry)
    memory_service = None
    memory_repository = None
    if settings.memory.enable_long_term:
        memory_repository = MemoryRepository(settings.memory.long_term_db_path)
        memory_repository.cleanup_expired()
        memory_service = MemoryService(
            memory_repository,
            space_directory or SpaceDirectory.from_registry(registry, "default-home"),
        )
    set_memory_service(memory_service)
    device_query_subgraph = build_device_query_subgraph(registry)
    rag_config = getattr(settings, "rag", None)
    knowledge_base = KnowledgeBase(getattr(rag_config, "knowledge_path", "docs/knowledge"))
    knowledge_rag_subgraph = build_knowledge_rag_subgraph(
        knowledge_base,
        top_k=getattr(rag_config, "top_k", 3),
        max_rewrites=getattr(rag_config, "max_rewrites", 1),
    )

    # ---- 第 4 步: 定义 Agent 节点 ----
    # 这个节点是工作流的"大脑"，负责:
    #   1. 读取当前消息历史
    #   2. 调用 LLM（带工具绑定）
    #   3. LLM 返回: 要么是文本回复，要么是 tool_calls
    def sync_context_node(state: AgentState) -> dict:
        """Apply trusted request location before processing the current turn."""
        request_device_id = state.get("request_device_id")
        request_room_id = state.get("request_room_id")

        latest_text = ""
        if state.get("messages"):
            content = getattr(state["messages"][-1], "content", "")
            latest_text = content if isinstance(content, str) else ""
        explicit_room_id = (
            space_directory.resolve_room_mention(latest_text)
            if space_directory
            else None
        )

        result = {}
        if explicit_room_id:
            result.update({
                "active_room_id": explicit_room_id,
                "active_device_id": None,
            })
        elif request_device_id:
            inferred_room_id = (
                space_directory.room_for_device(request_device_id)
                if space_directory
                else request_room_id
            )
            result.update({
                "active_device_id": request_device_id,
                "active_room_id": request_room_id or inferred_room_id,
            })
        elif request_room_id:
            result.update({
                "active_room_id": request_room_id,
                "active_device_id": None,
            })

        if memory_service and state.get("request_home_id") and state.get("request_user_id"):
            context = _context_from_state(state, result)
            memory_service.extract_candidates_from_text(context, latest_text)
            records = memory_service.retrieve(
                context, latest_text,
                top_k=getattr(settings.memory, "retrieval_top_k", 6),
            )
            result["retrieved_memories"] = [record.model_dump(mode="json") for record in records]
            result["memory_context"] = "\n".join(
                f"- [{record.scope.value}/{record.memory_type.value}] "
                f"{record.memory_key}: {record.memory_value} "
                f"(confidence={record.confidence:.2f}, importance={record.importance:.2f})"
                for record in records
            ) or "（无可用长期记忆）"
        emit_progress("context_synced", intent_text=latest_text[:80])
        return result

    def memory_reasoner_node(state: AgentState) -> dict:
        decision = reason_about_memories(
            state.get("retrieved_memories", []),
            _latest_text(state),
        )
        emit_progress(
            "memory_reasoned",
            applicable_count=len(decision.applicable_memory_ids),
            needs_clarification=decision.needs_clarification,
        )
        return {
            "memory_decision": decision.model_dump(),
            "memory_context": (
                state.get("memory_context", "（无可用长期记忆）")
                + "\n显式记忆决策: " + format_memory_decision(decision)
            ),
        }

    def compact_context_node(state: AgentState) -> dict:
        """Bound checkpoint state and expose input-size statistics."""
        updates, summary, token_estimate = build_compaction_update(
            list(state["messages"]),
            state.get("conversation_summary", ""),
            max_messages=getattr(settings.memory, "context_max_messages", 12),
            max_tokens=getattr(settings.memory, "context_max_tokens", 2400),
            max_tool_result_chars=getattr(settings.memory, "tool_result_max_chars", 1200),
            max_summary_chars=getattr(settings.memory, "summary_max_chars", 1800),
        )
        kept_count = len(state["messages"]) - sum(
            1 for message in updates if message.__class__.__name__ == "RemoveMessage"
        )
        logger.debug(
            f"上下文规模 | messages={kept_count} | estimated_tokens={token_estimate}"
        )
        result = {
            "conversation_summary": summary,
            "context_message_count": kept_count,
            "context_token_estimate": token_estimate,
        }
        if updates:
            result["messages"] = updates
        return result

    def task_router_node(state: AgentState) -> dict:
        """Classify the request, then select planning or the normal ReAct path."""
        latest_text = ""
        if state.get("messages"):
            content = getattr(state["messages"][-1], "content", "")
            latest_text = content if isinstance(content, str) else ""
        emit_progress("supervisor_routing", request=latest_text[:80])
        routing_config = getattr(settings, "routing", None)
        routing_enabled = getattr(routing_config, "enabled", False)
        intent = (
            classify_intent(llm, latest_text)
            if routing_enabled
            else classify_intent_fallback(latest_text)
        )
        confidence_threshold = getattr(routing_config, "confidence_threshold", 0.6)
        planning_enabled = getattr(getattr(settings, "planning", None), "enabled", True)
        use_planner = planning_enabled and should_use_planner(latest_text)
        intent_route = "planner" if use_planner else "react"
        if (
            not use_planner
            and intent.intent == "device_knowledge"
            and getattr(rag_config, "enabled", False)
        ):
            intent_route = "knowledge_rag"
        if (
            not use_planner
            and intent.intent == "device_query"
            and should_use_parallel_query(latest_text, registry)
        ):
            intent_route = "parallel_query"
        if not use_planner and (
            intent.intent == "clarification" or intent.confidence < confidence_threshold
            or state.get("memory_decision", {}).get("needs_clarification", False)
        ):
            intent_route = "clarification"
        result = {
            "intent": intent.intent,
            "intent_confidence": intent.confidence,
            "intent_reason": intent.reason,
            "intent_route": intent_route,
        }
        multi_agent_enabled = getattr(getattr(settings, "multi_agent", None), "enabled", False)
        if multi_agent_enabled:
            result.update({
                "delegated_agent": agent_for_intent(intent.intent),
                "handoff_count": 1,
                "collaboration_status": "delegated",
            })
        if not use_planner:
            result["planning_active"] = False
            return result
        result.update({
            "planning_active": True,
            "planning_goal": latest_text,
            "plan": None,
            "plan_revision": 0,
            "current_step_index": 0,
            "step_retry_count": 0,
            "replan_count": 0,
            "planning_status": "planning",
            "planning_failure_feedback": "",
            "planning_results": [],
        })
        return result

    def planner_node(state: AgentState) -> dict:
        """Generate or revise a structured plan without executing tools."""
        structured_planner = llm.with_structured_output(ExecutionPlan)
        prompt = planner_prompt(
            state["planning_goal"],
            registry,
            state.get("memory_context", "（无可用长期记忆）"),
            state.get("planning_failure_feedback", ""),
        )
        plan = structured_planner.invoke(prompt)
        if not isinstance(plan, ExecutionPlan):
            plan = ExecutionPlan.model_validate(plan)
        max_steps = getattr(getattr(settings, "planning", None), "max_steps", 8)
        if len(plan.steps) > max_steps:
            plan = plan.model_copy(update={"steps": plan.steps[:max_steps]})
        revision = state.get("plan_revision", 0) + 1
        logger.info(
            "Planner 生成计划 | revision={} | steps={} | goal={}",
            revision,
            len(plan.steps),
            plan.goal,
        )
        emit_progress("plan_generated", revision=revision, step_count=len(plan.steps))
        return {
            "plan": plan.model_dump(),
            "plan_revision": revision,
            "current_step_index": 0,
            "step_retry_count": 0,
            "planning_status": "awaiting_approval",
            "last_execution": None,
            "last_verification": None,
        }

    def plan_approval_node(state: AgentState) -> dict:
        """Pause before executing a newly generated or revised plan."""
        request = plan_approval_payload(state["plan"])
        decision = interrupt(request)
        approved = approval_is_granted(decision)
        return {
            "approval_request": request,
            "approval_decision": "approved" if approved else "rejected",
            "planning_status": "executing" if approved else "cancelled",
        }

    def executor_node(state: AgentState, config: RunnableConfig) -> dict:
        """Execute exactly one plan step using the existing trusted tools."""
        step = state["plan"]["steps"][state.get("current_step_index", 0)]
        emit_progress("step_started", step_id=step["step_id"], description=step["description"])
        device_id, expected_state, preparation_error = expected_state_for_step(step, registry)
        tool = tools_by_name.get(step["tool_name"])
        try:
            if tool is None:
                tool_result = f"❌ 未注册工具 {step['tool_name']}"
            elif preparation_error:
                tool_result = f"❌ {preparation_error}"
            else:
                tool_result = str(tool.invoke(step["arguments"], config=config))
        except Exception as exc:
            logger.warning(
                "Executor 工具异常 | step={} | tool={} | error={}",
                step["step_id"], step["tool_name"], exc,
            )
            tool_result = f"❌ 工具执行异常: {exc}"
        return {
            "last_execution": {
                "step": step,
                "device_id": device_id,
                "expected_state": expected_state,
                "preparation_error": preparation_error,
                "tool_result": tool_result,
            },
            "planning_status": "executing",
            "approval_request": None,
        }

    def verifier_node(state: AgentState) -> dict:
        """Check the actual device state and select the next control action."""
        execution = state["last_execution"]
        verification = verify_step(
            registry,
            execution.get("device_id"),
            execution.get("expected_state", {}),
            execution.get("tool_result", ""),
            execution.get("preparation_error"),
        )
        emit_progress("step_verified", success=verification.success)
        index = state.get("current_step_index", 0)
        results = list(state.get("planning_results", []))
        results.append({
            "plan_revision": state.get("plan_revision", 1),
            "step_id": execution["step"]["step_id"],
            "description": execution["step"]["description"],
            "tool_result": execution["tool_result"],
            "verification": verification.model_dump(),
        })
        if verification.success:
            next_index = index + 1
            finished = next_index >= len(state["plan"]["steps"])
            return {
                "last_verification": verification.model_dump(),
                "planning_results": results,
                "current_step_index": next_index,
                "step_retry_count": 0,
                "planning_status": "completed" if finished else "executing",
            }

        retry_count = state.get("step_retry_count", 0) + 1
        max_retries = getattr(getattr(settings, "planning", None), "max_step_retries", 1)
        if retry_count <= max_retries:
            next_status = "executing"
            replan_count = state.get("replan_count", 0)
        else:
            replan_count = state.get("replan_count", 0) + 1
            max_replans = getattr(getattr(settings, "planning", None), "max_replans", 1)
            next_status = "planning" if replan_count <= max_replans else "failed"
        return {
            "last_verification": verification.model_dump(),
            "planning_results": results,
            "step_retry_count": retry_count,
            "replan_count": replan_count,
            "planning_status": next_status,
            "planning_failure_feedback": (
                f"步骤 {execution['step']['step_id']}（{execution['step']['description']}）失败："
                f"{verification.reason}。工具结果：{execution['tool_result']}"
            ),
        }

    def planning_finalize_node(state: AgentState) -> dict:
        """Produce a deterministic summary of the completed planning trajectory."""
        status = state.get("planning_status")
        results = state.get("planning_results", [])
        succeeded = sum(1 for item in results if item["verification"]["success"])
        if status == "cancelled":
            content = "已取消该多步骤计划，所有尚未执行的设备操作都不会执行。"
        elif status == "completed":
            content = f"多步骤任务已完成，共验证通过 {succeeded} 个步骤。"
        else:
            failure = state.get("planning_failure_feedback", "未知原因")
            content = f"多步骤任务未能完成，已停止继续执行。最后失败原因：{failure}"
        return {
            "messages": [AIMessage(content=content)],
            "planning_active": False,
        }

    def clarification_node(state: AgentState) -> dict:
        return {"messages": [AIMessage(content="为了安全执行，请补充具体的设备、房间或要执行的动作。")], "planning_active": False}

    def parallel_query_node(state: AgentState) -> dict:
        latest_text = getattr(state["messages"][-1], "content", "")
        targets = extract_query_targets(latest_text, registry)
        result = device_query_subgraph.invoke({
            "query": latest_text,
            "targets": targets,
            "parallel_results": [],
        })
        emit_progress("parallel_query_completed", target_count=len(targets))
        return {
            "messages": [AIMessage(content=result.get("response", "没有找到可查询的设备。"))],
            "parallel_query_results": result.get("parallel_results", []),
        }

    def knowledge_rag_node(state: AgentState) -> dict:
        latest_text = _latest_text(state)
        result = knowledge_rag_subgraph.invoke({"query": latest_text})
        emit_progress(
            "knowledge_rag_completed",
            status=result.get("rag_status"),
            citation_count=len(result.get("citations", [])),
        )
        return {
            "messages": [AIMessage(content=result["answer"])],
            "rag_status": result.get("rag_status"),
            "rag_citations": result.get("citations", []),
            "rag_trajectory": result.get("trajectory", []),
            "rag_device_model": result.get("device_model"),
            "collaboration_status": "completed",
        }

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
        context_prompt = (
            f"\n\n## 当前可信请求上下文\n"
            f"home_id={state.get('request_home_id')}\n"
            f"user_id={state.get('request_user_id')}\n"
            f"client_id={state.get('request_client_id')}\n"
            f"active_room_id={state.get('active_room_id')}\n"
            f"active_device_id={state.get('active_device_id')}\n"
            f"conversation_summary={state.get('conversation_summary', '')}\n"
            f"long_term_memory:\n{state.get('memory_context', '（无可用长期记忆）')}\n"
            f"memory_decision={state.get('memory_decision', {})}\n"
            "这些标识来自受信任的业务上下文，不得根据用户文本改写。"
        )
        multi_agent_enabled = getattr(getattr(settings, "multi_agent", None), "enabled", False)
        role = state.get("delegated_agent", "chat") if multi_agent_enabled else None
        role_context = f"\n\n## 当前专用职责\n{role_prompt(role)}" if role else ""
        messages.insert(0, SystemMessage(content=system_prompt + context_prompt + role_context))

        logger.debug(f"Agent: 发送 {len(messages)} 条消息给 LLM")

        # 调用 LLM
        active_llm = specialised_llms[role] if role else llm_with_tools
        response = active_llm.invoke(messages)

        # 记录决策
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_names = [tc.get("name", "?") for tc in response.tool_calls]
            logger.info(f"Agent 决策: 调用工具 → {tool_names}")
        else:
            logger.info("Agent 决策: 直接文本回复")

        result = {"messages": [response]}
        if role:
            result["collaboration_status"] = "working"
        emit_progress("agent_completed", role=role or "legacy", has_tool_calls=bool(getattr(response, "tool_calls", [])))
        return result

    def supervisor_finalize_node(state: AgentState) -> dict:
        """Close one bounded delegation after the specialised agent responds."""
        max_handoffs = getattr(getattr(settings, "multi_agent", None), "max_handoffs", 2)
        count = state.get("handoff_count", 0)
        return {
            "collaboration_status": "completed" if count <= max_handoffs else "stopped",
        }

    def approval_node(state: AgentState) -> dict:
        """Pause before executing a batch scene and wait for trusted approval."""
        last_msg = state["messages"][-1]
        tool_calls = getattr(last_msg, "tool_calls", [])
        request = build_approval_request(tool_calls)
        if request is None:
            return {"approval_request": None, "approval_decision": "approved"}

        decision = interrupt(request)
        approved = approval_is_granted(decision)
        logger.info(
            "批量设备操作确认结果 | approved={} | tools={}",
            approved,
            [call.get("name") for call in tool_calls],
        )
        return {
            "approval_request": request,
            "approval_decision": "approved" if approved else "rejected",
        }

    def reject_tools_node(state: AgentState) -> dict:
        """Represent rejected tool calls as results without touching devices."""
        last_msg = state["messages"][-1]
        tool_calls = getattr(last_msg, "tool_calls", [])
        return {
            "messages": rejection_tool_messages(tool_calls),
            "approval_request": None,
        }

    # ---- 第 5 步: 构建图结构 ----
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("sync_context", sync_context_node)
    workflow.add_node("task_router", task_router_node)
    workflow.add_node("memory_reasoner", memory_reasoner_node)
    workflow.add_node("compact_context", compact_context_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("approval", approval_node)
    workflow.add_node("reject_tools", reject_tools_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("planner", planner_node)
    workflow.add_node("plan_approval", plan_approval_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("verifier", verifier_node)
    workflow.add_node("planning_finalize", planning_finalize_node)
    workflow.add_node("clarification", clarification_node)
    workflow.add_node("device_query_subgraph", parallel_query_node)
    workflow.add_node("knowledge_rag", knowledge_rag_node)
    workflow.add_node("supervisor_finalize", supervisor_finalize_node)

    # 入口: 从 agent 开始
    workflow.set_entry_point("sync_context")
    workflow.add_edge("sync_context", "memory_reasoner")
    workflow.add_edge("memory_reasoner", "task_router")

    def route_task(state: AgentState) -> Literal[
        "planner", "compact_context", "clarification", "device_query_subgraph", "knowledge_rag"
    ]:
        if state.get("intent_route") == "clarification":
            return "clarification"
        if state.get("intent_route") == "parallel_query":
            return "device_query_subgraph"
        if state.get("intent_route") == "knowledge_rag":
            return "knowledge_rag"
        return "planner" if state.get("planning_active") else "compact_context"

    workflow.add_conditional_edges(
        "task_router",
        route_task,
        {
            "planner": "planner",
            "compact_context": "compact_context",
            "clarification": "clarification",
            "device_query_subgraph": "device_query_subgraph",
            "knowledge_rag": "knowledge_rag",
        },
    )
    workflow.add_edge("clarification", END)
    workflow.add_edge("device_query_subgraph", END)
    workflow.add_edge("knowledge_rag", END)
    workflow.add_edge("compact_context", "agent")

    workflow.add_edge("planner", "plan_approval")

    def route_after_plan_approval(state: AgentState) -> Literal["executor", "planning_finalize"]:
        return "executor" if state.get("planning_status") == "executing" else "planning_finalize"

    workflow.add_conditional_edges(
        "plan_approval",
        route_after_plan_approval,
        {"executor": "executor", "planning_finalize": "planning_finalize"},
    )
    workflow.add_edge("executor", "verifier")

    def route_after_verifier(
        state: AgentState,
    ) -> Literal["executor", "planner", "planning_finalize"]:
        status = state.get("planning_status")
        if status == "executing":
            return "executor"
        if status == "planning":
            return "planner"
        return "planning_finalize"

    workflow.add_conditional_edges(
        "verifier",
        route_after_verifier,
        {
            "executor": "executor",
            "planner": "planner",
            "planning_finalize": "planning_finalize",
        },
    )
    workflow.add_edge("planning_finalize", END)

    # ---- 第 6 步: 路由逻辑 ----
    #    从 agent 出来后:
    #      - 如果 LLM 发出了 tool_calls → 去 tools 节点执行
    #      - 否则 → 结束
    def router(state: AgentState) -> Literal["approval", "tools", "supervisor_finalize", "__end__"]:
        """路由函数: 检查是否需要执行工具"""
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            if build_approval_request(last_msg.tool_calls) is not None:
                return "approval"
            return "tools"
        if getattr(getattr(settings, "multi_agent", None), "enabled", False):
            return "supervisor_finalize"
        return "__end__"

    workflow.add_conditional_edges(
        "agent",
        router,
        {
            "approval": "approval",
            "tools": "tools",
            "supervisor_finalize": "supervisor_finalize",
            "__end__": END,
        },
    )
    workflow.add_edge("supervisor_finalize", END)

    def route_after_approval(state: AgentState) -> Literal["tools", "reject_tools"]:
        return "tools" if state.get("approval_decision") == "approved" else "reject_tools"

    workflow.add_conditional_edges(
        "approval",
        route_after_approval,
        {"tools": "tools", "reject_tools": "reject_tools"},
    )

    # tools 执行完毕 → 回到 agent 继续思考
    workflow.add_edge("tools", "compact_context")
    workflow.add_edge("reject_tools", "compact_context")

    # ---- 第 7 步: 编译图（含检查点记忆）----
    # 检查点让 Agent 记住对话历史，实现多轮对话
    checkpointer = create_checkpointer(settings.memory.db_path)
    graph = workflow.compile(checkpointer=checkpointer)
    # Expose owned resources for application shutdown and integration tests.
    graph.memory_service = memory_service
    graph.memory_repository = memory_repository

    logger.info(
        f"Agent 图构建完成 | checkpointer={checkpointer.__class__.__name__}"
    )
    return graph


def _context_from_state(state: AgentState, updates: dict | None = None):
    from .context import AgentContext
    return AgentContext(
        home_id=state["request_home_id"],
        user_id=state["request_user_id"],
        session_id=state.get("request_session_id", "state-session"),
        client_id=state.get("request_client_id", "unknown"),
        room_id=(updates or {}).get("active_room_id", state.get("request_room_id")),
        device_id=(updates or {}).get("active_device_id", state.get("request_device_id")),
        is_admin=state.get("request_is_admin", False),
    )


def _latest_text(state: AgentState) -> str:
    if not state.get("messages"):
        return ""
    content = getattr(state["messages"][-1], "content", "")
    return content if isinstance(content, str) else ""
