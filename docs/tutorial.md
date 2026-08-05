# 智能家居家电互联智能体 — 开发教程

> **适用版本**: v0.2.0（已包含长期记忆与 Human-in-the-loop）
> **适用人群**: 想要学习现代 AI Agent 开发的 Python 开发者  
> **前置知识**: Python 基础（类、装饰器、类型注解）、理解 LLM 的基本概念  
> **完成时间**: 阅读约 45 分钟，动手实践约 2 小时  

---

## 目录

1. [项目概述](#1-项目概述)
2. [环境准备](#2-环境准备)
3. [项目架构](#3-项目架构)
4. [核心模块详解](#4-核心模块详解)
5. [运行与调试](#5-运行与调试)
6. [扩展指南](#6-扩展指南)
7. [MCP 集成](#7-mcp-集成)
8. [部署建议](#8-部署建议)
9. [常见问题](#9-常见问题)
10. [学习资源](#10-学习资源)

---

## 1. 项目概述

### 1.1 这是什么？

一个基于 **LangGraph + MCP (Model Context Protocol)** 的智能家居 AI Agent。你可以用自然语言查询和控制设备，并学习状态图编排、工具调用、会话检查点、上下文压缩、长期记忆以及 Human-in-the-loop 可恢复执行。

当前项目已经具备：

- 灯光、空调、电视、窗帘等模拟设备控制；
- 回家、离家、睡眠、观影、起床等多设备场景；
- LangGraph ReAct 工具调用循环；
- 基于 `thread_id` 的短期会话记忆；
- 结构化长期记忆、候选确认、混合检索和版本追踪；
- 场景操作执行前的 `interrupt` 人工确认；
- 使用 `Command(resume=...)` 从原检查点批准或拒绝操作；
- MCP Server 工具暴露和外部 MCP Client 接入能力。

### 1.2 技术栈

| 技术 | 版本 | 作用 |
|------|------|------|
| **LangGraph** | ≥1.2 | Agent 工作流编排（状态图、节点、条件路由） |
| **LangChain** | ≥1.3 | LLM 调用封装、工具定义、消息管理 |
| **MCP (Model Context Protocol)** | ≥1.0 | 标准化工具暴露/消费协议 |
| **Pydantic v2** | ≥2.0 | 类型安全的数据模型 & 配置管理 |
| **Typer + Rich** | - | 现代化 CLI 终端界面 |
| **Loguru** | - | 结构化日志 |
| **阿里百炼** | - | 大模型 API（兼容 OpenAI 接口） |

### 1.3 Agent 工作流

```text
用户输入
  ↓
sync_context             同步可信身份、房间、设备和长期记忆
  ↓
task_router
  ├── 普通请求 / 预定义场景
  │      ↓
  │   compact_context → agent
  │      ├── 无工具调用 ───────────────────────────────→ END
  │      ├── 普通工具调用 ─────────────────────────────→ tools
  │      └── activate_scene → approval → tools / reject_tools
  │
  └── 自定义多步骤目标
         ↓
      planner → plan_approval（interrupt）
                    ├── rejected ──────────────────────→ finalize
                    └── approved
                           ↓
                        executor → verifier
                                      ├── 下一步 / retry → executor
                                      ├── replan         → planner
                                      └── 完成 / 失败     → finalize
```

图中的两类持久状态分别是：

- LangGraph Checkpoint：保存消息、摘要、当前位置以及中断点；
- 长期记忆 SQLite：保存跨会话用户偏好、家庭规则、候选和历史版本。

工具还可以通过 MCP Server 暴露给外部 AI 客户端。需要注意，当前 Human-in-the-loop 是 Agent 图的编排能力；外部客户端如果绕过图直接调用 MCP 工具，需要由外部客户端或 MCP 服务层另外实现确认策略。

---

## 2. 环境准备

### 2.1 创建 Conda 环境

```bash
# 创建 Python 3.12 环境
conda create -n langgraph python=3.12 -y
conda activate langgraph
```

### 2.2 安装依赖

```bash
# 进入项目目录
cd G:\大厂学习\minimind\langgraph

# 安装所有依赖
pip install langgraph langchain langchain-openai langgraph-checkpoint-sqlite \
            pydantic pydantic-settings python-dotenv \
            mcp typer rich loguru httpx
```

### 2.3 获取 API Key

1. 打开 [阿里百炼控制台](https://bailian.console.aliyun.com/)
2. 登录阿里云账号，进入 **模型广场**
3. 在左侧菜单找到 **API Key**，创建一个新的 Key
4. 复制 Key 备用

> 💰 **费用参考**: qwen-plus 约 ¥0.004/千 token，个人开发每月几块钱。

### 2.4 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 BAILIAN_API_KEY
```

---

## 3. 项目架构

```
langgraph/
├── pyproject.toml            # 项目元数据 & 依赖声明
├── .env.example              # 环境变量模板
├── .env                      # 实际环境变量（不提交 git）
├── .gitignore
│
├── src/                      # ★ 源代码
│   ├── __init__.py           # 包入口，版本声明
│   ├── config.py             # pydantic-settings 配置管理
│   ├── models.py             # Pydantic 设备数据模型
│   │
│   ├── devices/              # 设备层
│   │   ├── __init__.py
│   │   ├── base.py           # 抽象后端接口 + 设备注册中心
│   │   └── simulator.py      # 内存模拟器后端
│   │
│   ├── tools/                # 工具层
│   │   ├── __init__.py       # 工具注册 & 导出
│   │   ├── devices.py        # 设备控制工具 (control_light, ac, tv, curtain)
│   │   ├── scenes.py         # 场景模式工具 (activate_scene, list_scenes)
│   │   └── memory.py         # 长期记忆管理工具
│   │
│   ├── agent/                # Agent 层
│   │   ├── __init__.py
│   │   ├── context.py        # 可信请求身份和空间归属校验
│   │   ├── session.py        # session_id / thread_id 生命周期
│   │   ├── state.py          # Agent 状态定义 (AgentState)
│   │   ├── prompts.py        # 系统提示词模板
│   │   ├── approval.py       # 风险识别、中断数据和拒绝结果
│   │   ├── planning.py       # 结构化计划、预期状态和 Verifier
│   │   └── graph.py          # LangGraph 工作流图 (build_graph)
│   │
│   ├── mcp/                  # MCP 层
│   │   ├── __init__.py
│   │   ├── server.py         # MCP 服务器 (暴露工具给外部 AI)
│   │   └── client.py         # MCP 客户端 (消费外部 MCP 服务)
│   │
│   ├── memory/               # 记忆层
│   │   ├── __init__.py
│   │   ├── store.py          # Checkpoint (MemorySaver / SqliteSaver)
│   │   ├── summarizer.py     # 上下文窗口和滚动摘要
│   │   ├── models.py         # 长期记忆、候选、版本和冲突模型
│   │   ├── repository.py     # SQLite Repository 和迁移
│   │   ├── extractor.py      # 自然语言记忆候选抽取
│   │   └── service.py        # 权限、检索和生命周期规则
│   │
│   ├── middleware/           # 中间件层
│   │   ├── __init__.py
│   │   └── interceptors.py   # 日志拦截器 + 重试拦截器
│   │
│   └── main.py               # ★ CLI 主入口 (typer + rich)
│
├── tests/                    # 阶段一至阶段六自动化测试
│   ├── test_phase_one.py
│   ├── test_phase_two.py
│   ├── test_phase_three.py
│   ├── test_phase_four.py
│   ├── test_phase_five.py
│   ├── test_phase_six.py     # Human-in-the-loop 测试
│   └── test_phase_seven.py   # Planner–Executor–Verifier 测试
│
├── docs/                     # 文档
│   ├── tutorial.md           # 本教程
│   └── iterations/           # 各阶段设计与实现说明
│
└── data/                     # 运行时数据（自动创建）
    ├── checkpoints.db        # SQLite 会话检查点
    └── memories.db           # SQLite 长期结构化记忆
```

### 架构分层

```
┌─────────────────────────────────────────────────────┐
│  CLI / MCP Server                   表示与协议层       │
├─────────────────────────────────────────────────────┤
│  LangGraph + Human-in-the-loop       编排层           │
│  sync / compact / agent / approval / tools           │
├─────────────────────────────────────────────────────┤
│  @tool 设备、场景和记忆工具            工具层           │
├─────────────────────────────────────────────────────┤
│  DeviceRegistry / SpaceDirectory      领域与上下文层    │
├─────────────────────────────────────────────────────┤
│  Checkpoint / Long-term Memory        状态与记忆层      │
├─────────────────────────────────────────────────────┤
│  SimulatorBackend / SQLite / Config   基础设施层        │
└─────────────────────────────────────────────────────┘
```

---

## 4. 核心模块详解

### 4.1 配置管理 (`config.py`)

使用 `pydantic-settings` 实现类型安全的配置加载：

```python
from src.config import get_settings

settings = get_settings()
print(settings.bailian_model)   # "qwen-plus"
print(settings.mcp_server.port) # 8765
```

**特性**:
- 自动从 `.env` 加载，支持系统环境变量覆盖
- 字段级验证（API Key 不能为空/占位符）
- 嵌套配置对象（MCP、Memory 子配置）
- IDE 友好的类型提示

### 4.2 数据模型 (`models.py`)

所有设备类型使用 Pydantic v2 严格建模：

```python
from src.models import LightDevice, ACDevice, ACMode

# 创建灯光（自动验证: brightness 0-100）
light = LightDevice(
    device_id="living_room_light",
    name="客厅灯",
    brightness=80,
    color="暖白",
)

# 创建空调
ac = ACDevice(
    device_id="bedroom_ac",
    name="卧室空调",
    temperature=26,
    mode=ACMode.COOL,
)

# 操作设备
light.turn_on()
print(light.to_status_text())
# "客厅灯 (living_room_light): 🟢 开启 | 亮度: 80% | 色温: 暖白"
```

**扩展新设备类型的步骤**（以"加湿器"为例）:

1. 在 `models.py` 中定义 `HumidifierDevice(BaseDevice)`
2. 在 `DeviceType` 枚举中添加 `HUMIDIFIER = "humidifier"`
3. 在 `simulator.py` 中注册默认设备
4. 在 `tools/devices.py` 中创建 `control_humidifier` 工具
5. 在 `tools/__init__.py` 中注册

### 4.3 设备注册中心 (`devices/base.py`)

**Registry Pattern** — 设备查找和操作的中心枢纽：

#### `devices` 目录中的类关系

设备模块可以按“模型 → 后端 → 注册中心 → 工具”的方向理解：

```
models.py
  └─ BaseDevice
       ├─ LightDevice
       ├─ ACDevice
       ├─ TVDevice
       └─ CurtainDevice
              │  (AnyDevice 联合类型)
              ▼
DeviceBackend (抽象接口)
  └─ SimulatorBackend (内存字典实现，创建 8 个默认设备)
              │
              ▼
DeviceRegistry (查找、筛选、更新、状态摘要)
              │
              ▼
tools/devices.py 的 @tool 函数
  └─ control_light / control_ac / control_tv / control_curtain
     get_device_status
```

- **设备模型**（`BaseDevice` 及四个子类）只描述设备数据和状态文本。`device_id` 是程序内部的稳定标识，`name` 是用户输入时使用的中文名称，`device_type` 用 `DeviceType` 枚举区分类型。
- **`DeviceBackend`** 是抽象协议，统一定义 `get`、`get_all`、`get_by_type`、`update` 和 `get_status_summary`。上层不依赖具体存储方式。
- **`SimulatorBackend`** 实现该协议，用 `_devices: dict[str, AnyDevice]` 保存设备。更新时通过 Pydantic `model_copy(update=...)` 生成经过类型校验的新对象；进程重启后状态恢复为默认值。
- **`DeviceRegistry`** 持有一个 backend，负责精确 ID 查找、按类型筛选、中文名称模糊匹配，并把更新和状态查询委托给 backend。它是工具层与具体设备后端之间的唯一入口。
- **工具层** 在启动时由 `main.py` 调用 `set_registry(registry)` 注入同一个注册中心。工具先用 `registry.find()` 找设备，再用 `registry.update()` 修改状态，因此工具和 Agent 不需要知道设备存在哪里。

运行时调用链如下：

```
用户请求 → Agent 决定工具 → control_* → DeviceRegistry.find
         → DeviceBackend.get_by_type / update
         → SimulatorBackend 修改 _devices
         → 工具返回文本 → Agent 生成最终回复
```

启动时的组装代码位于 `main.py`：

```python
backend = SimulatorBackend()       # 可替换成真实 IoT 后端
registry = DeviceRegistry(backend)
set_tools_registry(registry)       # 所有设备工具共享此实例
```

因此，替换真实设备平台时只需实现新的 `DeviceBackend` 子类，并在启动入口替换 backend；设备模型、注册中心、工具层和 Agent 的调用方式保持不变。

```python
from src.devices import DeviceRegistry, SimulatorBackend
from src.models import DeviceType

backend = SimulatorBackend()
registry = DeviceRegistry(backend)

# 精确查找
device = registry.get("living_room_light")

# 模糊查找（用户输入友好）
device = registry.find("客厅灯", DeviceType.LIGHT)     # → living_room_light
device = registry.find("卧室的空调", DeviceType.AC)     # → bedroom_ac

# 更新设备
registry.update("living_room_light", power=True, brightness=70)

# 生成状态报告（给 LLM 看）
print(registry.get_status_summary())
```

**依赖倒置**: `DeviceBackend` 是抽象接口，`SimulatorBackend` 是内存实现。  
后续对接 Home Assistant 只需创建 `HomeAssistantBackend(DeviceBackend)`，工具层和 Agent 零修改。

### 4.4 工具层 (`tools/`)

每个工具是一个 `@tool` 装饰的 Python 函数。LangChain 自动：
- 从函数签名生成 JSON Schema
- 从 docstring 生成工具描述（LLM 读这个来决定何时调用）

```python
from langchain_core.tools import tool

@tool
def control_light(device_name: str, action: str, brightness: int = 50) -> str:
    """
    控制灯光设备。打开/关闭、调节亮度、调节色温。
    ...（LLM 阅读这部分决定何时调用此工具）
    """
    # 1. 模糊查找设备
    device = registry.find(device_name, DeviceType.LIGHT)
    # 2. 执行操作
    registry.update(device.device_id, power=True)
    # 3. 返回结果文本（作为 ToolMessage 还给 LLM）
    return f"✅ {device.name}已打开"
```

**工具调用流程**:

普通单设备操作会直接进入 `ToolNode`：

```text
用户：“打开客厅灯”
  → LLM 生成 control_light 工具调用
  → 风险判断：普通单设备操作，不需要确认
  → ToolNode 执行 control_light
  → ToolMessage 返回真实执行结果
  → LLM 生成最终回复
```

场景工具会在执行前进入确认节点：

```text
用户：“我要出门了”
  → LLM 生成 activate_scene(scene_name="离家模式")
  → approval 节点调用 interrupt
  → 用户批准后 ToolNode 才执行场景
  → 用户拒绝则生成取消 ToolMessage，不执行场景
```

### 4.5 Agent 图 (`agent/graph.py`)

LangGraph 的 `StateGraph` 定义了上下文准备、模型决策、人工确认和工具执行循环：

```python
from src.agent import build_graph

# 构建图
graph = build_graph(registry, settings)

# 运行
result = graph.invoke(
    state_input,
    context.to_config(),
)

# 获取回复
print(result["messages"][-1].content)
```

当前图包含以下节点：

| 节点 | 作用 |
| --- | --- |
| `sync_context` | 同步可信请求位置，抽取记忆候选并检索长期记忆 |
| `task_router` | 将明确的自定义多步骤目标路由到规划分支 |
| `compact_context` | 限制消息和 token 规模，维护滚动摘要 |
| `agent` | 调用绑定工具后的 LLM |
| `approval` | 对批量场景调用执行 `interrupt` |
| `tools` | 使用 `ToolNode` 执行已批准或无需确认的工具 |
| `reject_tools` | 拒绝时生成匹配工具调用 ID 的取消结果 |
| `planner` | 通过结构化输出生成或修订 `ExecutionPlan` |
| `plan_approval` | 在执行完整计划前暂停并展示步骤 |
| `executor` | 每次只执行当前计划中的一个原子工具步骤 |
| `verifier` | 读取真实设备状态，判断成功、重试或重新规划 |
| `planning_finalize` | 汇总完成、取消或失败结果 |

简化后的构图代码如下：

```python
workflow = StateGraph(AgentState)

workflow.add_node("sync_context", sync_context_node)
workflow.add_node("task_router", task_router_node)
workflow.add_node("compact_context", compact_context_node)
workflow.add_node("agent", agent_node)
workflow.add_node("approval", approval_node)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("reject_tools", reject_tools_node)
workflow.add_node("planner", planner_node)
workflow.add_node("plan_approval", plan_approval_node)
workflow.add_node("executor", executor_node)
workflow.add_node("verifier", verifier_node)
workflow.add_node("planning_finalize", planning_finalize_node)

workflow.set_entry_point("sync_context")
workflow.add_edge("sync_context", "task_router")

workflow.add_conditional_edges("task_router", route_task, {
    "planner": "planner",
    "compact_context": "compact_context",
})
workflow.add_edge("compact_context", "agent")

workflow.add_conditional_edges("agent", router, {
    "approval": "approval",
    "tools": "tools",
    "__end__": END,
})

workflow.add_conditional_edges("approval", route_after_approval, {
    "tools": "tools",
    "reject_tools": "reject_tools",
})

workflow.add_edge("tools", "compact_context")
workflow.add_edge("reject_tools", "compact_context")

workflow.add_edge("planner", "plan_approval")
workflow.add_conditional_edges("plan_approval", route_after_plan_approval, {
    "executor": "executor",
    "planning_finalize": "planning_finalize",
})
workflow.add_edge("executor", "verifier")
workflow.add_conditional_edges("verifier", route_after_verifier, {
    "executor": "executor",
    "planner": "planner",
    "planning_finalize": "planning_finalize",
})

graph = workflow.compile(checkpointer=create_checkpointer(db_path))
```

为什么拒绝后还要经过 `reject_tools`？因为模型已经生成了工具调用消息。每个工具调用都应该有匹配 `tool_call_id` 的 `ToolMessage`；否则下一次模型调用会收到不完整的消息协议。`reject_tools` 用“用户拒绝，工具未执行”闭合调用，然后再由 Agent 生成自然语言说明。

如果图在 `approval` 节点中断，第一次 `invoke` 的结果中会出现：

```python
result["__interrupt__"][0].value
```

调用方读取其中的 `question`、`summary` 和 `tool_calls`，取得用户决定后使用原配置恢复：

```python
from langgraph.types import Command

result = graph.invoke(
    Command(resume={"approved": True}),
    context.to_config(),
)
```

这里必须继续使用相同的 `thread_id`，因为中断位置保存在该线程对应的 Checkpoint 中。

### 4.6 中间件 (`middleware/interceptors.py`)

**LoggingInterceptor**: 记录每次 Agent 调用的输入/输出/耗时  
**RetryInterceptor**: LLM 调用失败自动重试（指数退避: 1s→2s→4s）

```python
from src.middleware import apply_all_middleware

# 包装任意函数，自动获得日志+重试能力
safe_func = apply_all_middleware(original_func)
```

### 4.7 MCP 集成 (`mcp/`)

MCP 是 Anthropic 提出的开放协议，让 AI 应用以标准方式暴露和消费工具。

**MCP Server** (`mcp/server.py`): 将智能家居工具暴露给外部 AI

```python
# 作为独立服务启动
python -m src.mcp.server --transport sse --port 8765

# 或在 Claude Desktop 配置文件中添加
# "smart-home": {
#     "command": "python",
#     "args": ["-m", "src.mcp.server"]
# }
```

**MCP Client** (`mcp/client.py`): 消费外部 MCP 服务（天气、日历等）

```python
# .env 中配置
EXTERNAL_MCP_SERVERS={"name":"weather","transport":"stdio","command":"python","args":["weather_mcp.py"]}
```

---

## 5. 运行与调试

### 5.1 启动交互式对话

```bash
conda activate langgraph
cd G:\大厂学习\minimind\langgraph
python -m src.main
```

```
🏠 智能家居管家 — 小智
模型: qwen-plus
框架: LangGraph + LangChain + MCP
平台: 阿里百炼 (Alibaba Bailian)

试着说: 打开客厅灯 / 空调调到25度 / 我要睡觉了 / 现在家里什么状态?

👤 你: 我回来了，有点热

🤖 小智: 准备激活回家模式。

┌────────────── ⚠️ 操作确认 · medium ──────────────┐
│ 即将执行批量设备操作：回家模式……是否继续？       │
└──────────────────────────────────────────────────┘
确认执行？[y/N]: y

🤖 小智: 已为你激活回家模式：
✅ 已激活「🏠 回家模式」
  · 客厅灯已打开（亮度 80%，暖白）
  · 客厅空调已开启（制冷 26°C）
  · 客厅窗帘已完全打开
🏠 欢迎回家！
```

如果输入 `n` 或直接回车，图会从中断点恢复到拒绝分支，不调用 `activate_scene`，并回复操作已经取消。普通的“打开客厅灯”等单设备请求不会弹出确认。

确认时 CLI 并不是重新发送一条“用户说同意”的普通聊天消息，而是使用：

```python
Command(resume={"approved": True})
```

因此模型不会重新生成计划，LangGraph 会从原来的 `approval` 节点继续。

### 5.2 常用命令

| 命令 | 说明 |
|------|------|
| `/status` | 查看所有设备状态 |
| `/scenes` | 列出可用场景模式 |
| `/reset`  | 重置对话记忆 |
| `/help`   | 显示使用指南 |
| `/quit`   | 退出 |

### 5.3 高级选项

```bash
# 使用更强的模型
python -m src.main --model qwen-max

# 调试模式（显示详细日志）
python -m src.main --debug

# 快速查看设备状态（不启动对话）
python -m src.main status
```

### 5.4 启动 MCP 服务器

```bash
# stdio 模式（供 Claude Desktop 连接）
python -m src.mcp.server

# SSE 模式（HTTP 长连接）
python -m src.mcp.server --transport sse --port 8765
```

### 5.5 运行自动化测试

使用项目现有的 `langgraph` Conda 环境运行：

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

当前测试覆盖阶段一至阶段七，共 50 个测试。Human-in-the-loop 相关测试位于 `tests/test_phase_six.py`，重点验证：

1. 中断返回时，批量场景尚未修改设备；
2. `approved=True` 后才执行真实场景；
3. `approved=False` 后设备保持原状；
4. 拒绝路径产生匹配工具调用 ID 的 `ToolMessage`；
5. 普通单设备工具不会触发中断；
6. SQLite Checkpoint 关闭并重新构建图后，仍可使用相同 `thread_id` 恢复。

Planner–Executor–Verifier 相关测试位于 `tests/test_phase_seven.py`，重点验证：

1. 复杂任务路由不会影响单设备和预定义场景；
2. 计划需要批准后才开始执行；
3. Executor 按步骤调用现有工具；
4. Verifier 根据真实设备状态判断结果；
5. 单步失败能够有限重试；
6. 重试耗尽后能够重新规划并再次确认；
7. SQLite Checkpoint 能在图重建后恢复已暂停的计划；
8. 重试和重新规划额度全部耗尽后任务会明确停止。

---

## 6. 扩展指南

### 6.1 添加新设备类型

以 **加湿器 (Humidifier)** 为例：

**第一步**: 在 `src/models.py` 中添加模型：

```python
class HumidifierDevice(BaseDevice):
    device_type: DeviceType = Field(default=DeviceType.HUMIDIFIER, frozen=True)
    humidity_target: int = Field(default=60, ge=30, le=90)
    water_level: int = Field(default=100, ge=0, le=100)

# 在 DeviceType 枚举中添加
class DeviceType(str, Enum):
    ...
    HUMIDIFIER = "humidifier"
```

**第二步**: 在 `src/devices/simulator.py` 中注册默认设备：

```python
HumidifierDevice(
    device_id="living_room_humidifier",
    name="客厅加湿器",
    location="客厅",
),
```

**第三步**: 在 `src/tools/devices.py` 中创建工具：

```python
@tool
def control_humidifier(device_name: str, action: str, humidity_target: int = 60) -> str:
    """控制加湿器..."""
    # 实现逻辑
```

**第四步**: 在 `src/tools/__init__.py` 中注册。

### 6.2 对接真实 IoT 平台

替换 `SimulatorBackend` 为真实后端：

```python
class HomeAssistantBackend(DeviceBackend):
    """对接 Home Assistant"""
    def __init__(self, ha_url: str, ha_token: str):
        self.base_url = ha_url
        self.headers = {"Authorization": f"Bearer {ha_token}"}

    def update(self, device_id: str, **kwargs) -> bool:
        # 调用 Home Assistant REST API
        response = requests.post(
            f"{self.base_url}/api/services/{domain}/turn_on",
            headers=self.headers,
            json={"entity_id": device_id, **kwargs},
        )
        return response.ok

# 在 main.py 中替换
# backend = SimulatorBackend()      # 旧: 模拟
backend = HomeAssistantBackend(     # 新: 真实设备
    ha_url="http://homeassistant:8123",
    ha_token=os.getenv("HA_TOKEN"),
)
registry = DeviceRegistry(backend)
```

### 6.3 添加自定义中间件

```python
from src.middleware.interceptors import LoggingInterceptor

class MetricsInterceptor:
    """收集 Agent 性能指标"""
    @staticmethod
    def wrap(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            # 发送到 Prometheus / Grafana
            return result
        return wrapper
```

### 6.4 持久化记忆

#### 6.4.1 先理解：项目中其实有两种“记忆”

大模型本身是无状态的。每次调用模型时，它只能看到本次传入的消息和提示词；如果程序不主动保存和重新提供信息，模型不会真正记得上一轮，更不会记得几天前用户说过什么。

本项目把记忆拆成两个层次，分别解决不同问题：

| 记忆层 | 解决的问题 | 典型内容 | 保存位置 |
|--------|------------|----------|----------|
| **短期会话记忆** | “这一轮对话之前聊了什么？” | 用户消息、AI 回复、工具调用结果、滚动摘要、当前关注的房间和设备 | LangGraph Checkpoint |
| **长期结构化记忆** | “这个用户长期喜欢什么、家里有什么固定规则？” | 灯光偏好、空调温度、设备别名、生活例程、家庭约束 | 独立 SQLite 数据库 |

举个例子：

```text
用户：打开客厅灯。
AI：已打开。
用户：调暗一点。
```

第二句中的“调暗一点”依赖刚才的对话，所以属于**短期会话记忆**。

```text
用户：我一般喜欢把客厅灯调成暖光，以后按这个习惯来。
```

“喜欢暖光”可能在未来的新会话中仍然有用，因此适合作为**长期记忆候选**。注意这里先生成候选，不会直接写入正式记忆；只有用户确认后才保存。

两层记忆的关系可以概括为：

```text
当前会话消息 ──> Checkpoint ──> 同一个 thread_id 恢复上下文

明确偏好或重复操作 ──> 候选记忆 ──> 用户确认 ──> 长期记忆
                                                    │
新会话中的问题 ──> 检索最相关记忆 <──────────────────┘
```

#### 6.4.2 短期会话记忆：Checkpoint 如何工作

LangGraph 在图编译时接收一个 checkpointer：

```python
checkpointer = create_checkpointer(settings.memory.db_path)
graph = workflow.compile(checkpointer=checkpointer)
```

每次图执行后，LangGraph 会按照 `thread_id` 保存 `AgentState`。再次使用相同的 `thread_id` 调用图时，之前的消息和状态会被恢复。

```text
thread_id="session-a"
  第 1 轮：[用户：打开客厅灯，AI：已打开]
  第 2 轮：[……，用户：再暗一点，AI：已调暗]

thread_id="session-b"
  另一段独立会话，不会读取 session-a 的消息历史
```

项目支持两种 Checkpoint 存储方式：

| 对比项 | `MemorySaver` | `SqliteSaver` |
|--------|---------------|---------------|
| 存储位置 | 当前 Python 进程内存 | `data/checkpoints.db` |
| 进程重启后 | 会话状态丢失 | 可以恢复原会话状态 |
| 适用场景 | 单元测试、临时调试 | 本地开发和持久化会话 |

`CHECKPOINT_DB_PATH` 为空时使用内存模式；配置数据库路径时使用 SQLite：

```dotenv
CHECKPOINT_DB_PATH=data/checkpoints.db
```

> 依赖以 `pyproject.toml` 为准。请先使用项目既有的 Conda 环境检查依赖；如果环境缺少所需包，应暂停运行并交由环境维护者配置，不要在教程步骤中擅自创建环境或安装依赖。

要恢复一段旧会话，不仅要保留数据库，还必须继续使用原来的 `thread_id`。仅仅重启程序后生成一个新的会话 ID，不会自动进入旧会话。

#### 6.4.3 为什么还要压缩会话上下文

Checkpoint 能保存消息，但不能让消息无限增长。长对话会带来三个问题：

1. 输入 token 越来越多，模型调用成本和延迟上升。
2. 很久以前的无关信息会干扰当前推理。
3. 工具返回内容可能很长，持续保存在状态中会让数据库膨胀。

因此图中在每次调用模型前都会经过 `compact_context` 节点：

```text
sync_context
    ↓
compact_context  ── 保留最近消息、生成滚动摘要、裁剪旧工具结果
    ↓
agent
    ↓ 有工具调用
tools ──────────────> compact_context
```

相关配置如下：

| 配置项 | 默认值 | 作用 |
|--------|-------:|------|
| `CHECKPOINT_CONTEXT_MAX_MESSAGES` | `12` | Checkpoint 中重点保留的最近消息数 |
| `CHECKPOINT_CONTEXT_MAX_TOKENS` | `2400` | 上下文估算 token 上限 |
| `CHECKPOINT_TOOL_RESULT_MAX_CHARS` | `1200` | 单条旧工具结果的字符上限 |
| `CHECKPOINT_SUMMARY_MAX_CHARS` | `1800` | 滚动摘要的最大长度 |
| `CHECKPOINT_SESSION_TTL_HOURS` | `168` | 会话过期时间，默认 7 天 |

压缩不是简单删除历史。系统会把较早对话归纳到 `conversation_summary`，保留最近的原始消息，并通过 LangGraph 的 `RemoveMessage` 从持久化状态中移除不再需要的旧消息。

#### 6.4.4 长期记忆保存什么，不保存什么

长期记忆只保存未来仍可能有用、并且能够结构化表达的信息。

适合保存：

- **偏好（preference）**：喜欢暖光、常用空调温度、电视音量偏好。
- **别名（alias）**：用户把某个设备称为“床头那盏灯”。
- **例程（routine）**：晚上 10 点后习惯使用安静模式。
- **约束（constraint）**：儿童房夜间音量不能超过某个值。

不适合保存：

- “今天有点冷”这类临时感受。
- “这次把空调调到 25 度”这类单次指令。
- 灯当前是否开启、空调当前温度等实时设备状态。
- 没有经过用户确认的模型推断。

实时设备状态应从 `DeviceBackend` 查询。把它当成长久偏好保存，会导致状态过时，还会混淆“设备现在是什么状态”和“用户长期喜欢什么”。

一条正式记忆的核心结构如下：

```python
MemoryRecord(
    scope="user",
    memory_type="preference",
    memory_key="lighting.color",
    memory_value={"color": "暖光"},
    confidence=0.84,
    importance=0.65,
    version=1,
    valid_from=...,
    valid_to=None,
)
```

这里故意使用 `memory_key + JSON memory_value`，而不是只保存一段自然语言。结构化数据更容易校验、去重、合并、检索和测试。

#### 6.4.5 作用域：这条记忆应该让谁看见

长期记忆支持四种作用域：

| 作用域 | 含义 | 示例 |
|--------|------|------|
| `user` | 只属于当前用户 | 用户 A 喜欢暖光 |
| `home` | 整个家庭共享 | 离家时关闭所有灯 |
| `room` | 某个房间共享 | 儿童房夜间保持安静 |
| `device` | 针对某台设备 | 客厅电视的默认音量限制 |

个人偏好与共享规则的权限不同：

- 普通用户可以创建、修改和删除自己的 `user` 记忆。
- `home`、`room`、`device` 属于共享记忆，写入和修改需要管理员权限。
- 所有 Repository 查询都必须带 `home_id`，避免不同家庭之间串数据。
- 个人记忆还要校验 `user_id`，用户 A 不能读取或修改用户 B 的个人偏好。

身份信息不会让模型自由填写。`home_id`、`user_id`、`room_id`、`device_id` 和 `is_admin` 都来自服务端构造的 `RunnableConfig`：

```python
config = {
    "configurable": {
        "thread_id": "session-a",
        "home_id": "home-a",
        "user_id": "user-a",
        "client_id": "phone-a",
        "room_id": "living_room",
        "is_admin": False,
    }
}
```

记忆工具的参数中没有 `home_id` 和 `user_id`。这样即使模型生成了错误工具参数，也不能借此访问另一个家庭或用户的数据。

#### 6.4.6 记忆如何产生：明确保存与候选确认

系统提供两条产生长期记忆的路径。

**路径一：用户明确要求记住**

例如用户说：“请记住，我喜欢暖光。”Agent 可以调用 `save_personal_memory`：

```python
save_personal_memory(
    memory_key="lighting.color",
    memory_value={"color": "暖光"},
    source="用户明确要求记住",
)
```

这是显式写入，因为保存动作来自用户明确指令。

**路径二：系统发现可能的习惯，先生成候选**

候选有两个来源：

1. **自然语言抽取**：识别“我喜欢”“我一般”“以后请”等稳定表达。
2. **重复操作统计**：同一个用户反复执行相同设置，达到阈值后形成候选。

自然语言抽取器位于 `src/memory/extractor.py`，当前采取高精度、低误报策略。例如：

```text
“我一般把空调设为 25 度”
    → 候选：ac.temperature = {"temperature": 25}

“我喜欢客厅用暖光”
    → 候选：lighting.color = {"color": "暖光"}

“今天有点冷，空调开 25 度”
    → 含有临时表达，不生成候选
```

重复操作的默认阈值是 3 次：

```text
成功控制设备
    ↓
record_operation()
    ↓
同一 user + memory_key + memory_value 累计达到 3 次
    ↓
PreferenceCandidate(status="pending")
```

无论候选来自哪条路径，都不会自动成为正式记忆：

```text
pending 候选
   ├─ 用户确认 ──> confirmed ──> 写入 MemoryRecord
   └─ 用户拒绝 ──> rejected  ──> 不写入长期记忆
```

这种“先候选、后确认”的设计可以显著降低误记忆风险。Agent 可以提出“我注意到你经常把空调设为 25 度，要记住这个偏好吗？”，但不能替用户做决定。

#### 6.4.7 检索：不是把所有记忆都塞进 Prompt

当长期记忆变多时，每轮把全部记录交给模型既浪费 token，也会引入无关信息。因此 `sync_context` 节点会根据用户当前问题检索 Top-K 记忆：

```python
memory_service.format_for_prompt(
    context,
    query=latest_user_text,
    top_k=settings.memory.retrieval_top_k,
)
```

默认 `CHECKPOINT_RETRIEVAL_TOP_K=6`。每条可访问记忆的分数由五部分组成：

```text
score =
    0.45 × 词项相关性
  + 0.20 × 置信度
  + 0.20 × 重要性
  + 0.10 × 时间新鲜度
  + 0.05 × 访问频率
```

- **相关性**：当前问题中的词是否出现在记忆 key 或 value 中。
- **置信度**：系统对这条记忆可靠程度的估计。
- **重要性**：这条记忆对未来决策的业务价值。
- **时间新鲜度**：较新的偏好通常优先级更高。
- **访问频率**：经常被用到的记忆获得少量加分，并使用对数归一化避免无限放大。

检索顺序非常重要：系统先按家庭、用户、房间和设备做权限过滤，再对当前用户可见的记录排序。不能先在全库召回，再指望模型自己忽略无权访问的内容。

命中的记忆会更新 `access_count` 和 `last_accessed_at`，为之后的排序提供访问统计。

当前实现没有引入 Embedding 或向量数据库，优点是依赖少、评分可解释、容易测试。`evaluate_vector_retrieval()` 会在记忆规模达到阈值时给出是否考虑向量召回的建议，但不会自动下载或初始化额外服务。

#### 6.4.8 更新、冲突与历史版本

用户偏好会改变。例如用户先说喜欢暖光，后来改成冷白光。如果直接覆盖数据库中的 JSON，系统只能看到最新结果，无法回答“什么时候发生了变化”。

因此每条记忆都有版本号和有效时间区间：

```text
版本 1：暖光
valid_from = 2026-07-01
valid_to   = 2026-08-01

版本 2：冷白光
valid_from = 2026-08-01
valid_to   = NULL        ← 当前仍然有效
```

更新流程如下：

1. 关闭旧版本，把旧版本的 `valid_to` 设为更新时间。
2. 写入新值并把 `version` 加 1。
3. 创建新的 `MemoryVersion`，其 `valid_to` 为空。
4. 当前查询只返回 active 的最新版本，历史版本仍可审计。

如果相同 key 收到不同 value，`MemoryService` 会执行结构化合并，并把旧值、输入值、最终值和解决方式写入 `memory_conflicts`。这使冲突处理可以追踪，而不是悄悄覆盖。

删除和过期也采用逻辑关闭：正式记录不再参与当前检索，同时关闭当前版本的有效区间。历史版本不会因为普通删除操作而消失。

可以通过工具查询版本：

```text
list_memory_versions(memory_id)
```

#### 6.4.9 SQLite 中保存了哪些表

Checkpoint 与长期记忆使用不同的数据库，职责不要混淆：

```text
data/checkpoints.db   LangGraph 会话状态
data/memories.db      结构化长期记忆
```

长期记忆数据库的主要表包括：

| 表 | 作用 |
|----|------|
| `memories` | 当前长期记忆及其状态、分数、访问统计和版本号 |
| `preference_observations` | 重复设备操作的原始计数 |
| `preference_candidates` | 等待用户确认或已经处理的候选 |
| `memory_conflicts` | 同一记忆收到不同值时的合并审计 |
| `memory_versions` | 每次变更对应的历史值和有效时间 |

Repository 启动时会检查旧数据库结构，并为阶段二至阶段四创建的数据库补充阶段五字段和版本记录，因此升级后不要求删除原数据库重建。

相关配置：

```dotenv
CHECKPOINT_DB_PATH=data/checkpoints.db
CHECKPOINT_LONG_TERM_DB_PATH=data/memories.db
ENABLE_LONG_TERM_MEMORY=true
CHECKPOINT_RETRIEVAL_TOP_K=6
```

当 `ENABLE_LONG_TERM_MEMORY=false` 时，Agent 仍可使用设备工具和短期 Checkpoint，但不会创建 MemoryRepository，也不会抽取或注入长期记忆。

#### 6.4.10 记忆模块的代码分层

为了避免 SQL、权限、抽取规则和 Agent 图互相耦合，记忆模块按职责拆分：

```text
src/memory/models.py       数据模型：MemoryRecord、Candidate、Version、Conflict
src/memory/repository.py   SQLite 表结构、迁移和基础 CRUD
src/memory/service.py      权限、作用域、候选、检索、冲突和生命周期规则
src/memory/extractor.py    自然语言候选抽取，只产出候选，不写数据库
src/memory/summarizer.py   会话摘要、token 估算和消息压缩
src/memory/store.py        Checkpoint 创建、关闭和会话清理
src/tools/memory.py        提供给 LLM 的记忆工具
```

调用链如下：

```text
用户消息
   ↓
LangGraph.sync_context
   ├─> extractor 生成候选
   └─> MemoryService 检索长期记忆
            ↓
       MemoryRepository 查询 SQLite
            ↓
格式化 Top-K 记忆并注入系统提示词
            ↓
LLM 根据当前消息 + 摘要 + 相关长期记忆进行决策
```

这里最重要的边界是：

- `Repository` 只负责可靠地读写数据。
- `Service` 负责“是否允许”和“应该怎样处理”。
- `Extractor` 不直接操作 Repository。
- `Tool` 不相信模型提供的身份，而是读取受信任配置。

#### 6.4.11 可供 Agent 调用的记忆工具

| 工具 | 用途 |
|------|------|
| `save_personal_memory` | 保存用户明确要求记住的个人偏好 |
| `save_home_rule` | 管理员保存家庭共享规则 |
| `list_personal_memories` | 查看当前上下文中可访问的记忆 |
| `update_personal_memory` | 修改当前用户拥有的个人记忆 |
| `delete_personal_memory` | 删除当前用户拥有的个人记忆 |
| `list_preference_candidates` | 查看等待确认的偏好候选 |
| `confirm_preference_candidate` | 确认候选并生成正式记忆 |
| `reject_preference_candidate` | 拒绝候选，不写入正式记忆 |
| `list_memory_versions` | 查看一条记忆的历史版本 |

业务代码也可以绕过 LLM，直接调用 `MemoryService`。例如保存一条个人偏好：

```python
from src.memory import MemoryScope, MemoryType, MemoryWrite

record = memory_service.save(
    context,
    MemoryWrite(
        scope=MemoryScope.USER,
        memory_type=MemoryType.PREFERENCE,
        memory_key="lighting.color",
        memory_value={"color": "暖光"},
        source="用户在设置页中明确保存",
        importance=0.7,
    ),
)
```

确认候选：

```python
candidates = memory_service.list_candidates(context)
record = memory_service.confirm_candidate(context, candidates[0].id)
```

查询历史版本：

```python
versions = memory_service.list_versions(context, record.id)
for version in versions:
    print(version.version, version.memory_value, version.valid_from, version.valid_to)
```

#### 6.4.12 如何验证记忆功能

先使用项目已配置的 `langgraph` 环境，不要新建环境或自行安装包：

```powershell
$python = 'F:\Software\Anaconda\envs\langgraph\python.exe'
& $python -m pytest -q
& $python -m compileall -q src tests
& $python -m pip check
```

建议重点验证以下场景：

1. 同一个 `thread_id` 能恢复上一轮消息，不同 `thread_id` 相互隔离。
2. 长对话会生成摘要并限制持久化消息规模。
3. 用户 A 无法读取和修改用户 B 的个人记忆。
4. 非管理员无法写入家庭、房间和设备共享规则。
5. “今天有点冷”不会被当成长期偏好。
6. 自然语言候选和重复操作候选都必须确认后才写入。
7. 与当前问题相关的记忆排在无关记忆之前，且只返回 Top-K。
8. 更新偏好后旧版本被关闭，新版本成为当前有效版本。
9. 阶段二至阶段四的旧 SQLite 数据库可以自动迁移。

只验证数据库文件存在还不够。最可靠的测试方式是断言业务行为和权限边界，而不是依赖肉眼查看表内容。

#### 6.4.13 常见问题

**Q：新会话为什么还能知道我的偏好？**
A：新 `thread_id` 不会继承旧消息，但 `sync_context` 会按当前可信身份重新检索 SQLite 长期记忆，所以跨会话偏好仍然可用。

**Q：Checkpoint 会保存模拟设备的实时状态吗？**
A：不会把设备后端状态自动持久化。Checkpoint 主要保存 AgentState；模拟设备的实时状态仍由 `SimulatorBackend` 管理，进程重启后是否保留取决于设备后端本身。

**Q：说一次“空调调到 25 度”会被永久记住吗？**
A：不会。它首先是一次设备操作。只有明确的稳定偏好表达，或相同操作重复达到阈值，才会生成候选；候选还必须由用户确认。

**Q：为什么候选不直接写入？**
A：自然语言可能有歧义，重复行为也可能只是短期需求。确认步骤把最终决定权留给用户，可以降低错误个性化和隐私风险。

**Q：修改偏好后旧值去哪了？**
A：当前记录会切换到新版本，旧值保存在 `memory_versions` 中，并带有完整的有效时间区间。

**Q：数据库会不会无限增长？**
A：会话消息通过压缩和 TTL 控制；长期记忆支持过期、逻辑删除和 Top-K 检索。生产环境仍应增加定期归档、备份、彻底删除和容量监控策略。

**Q：当前是否使用向量数据库？**
A：没有。当前使用结构化权限过滤和可解释的混合排序。家庭级小规模记忆足够使用，规模扩大后可再增加向量召回层。

### 6.5 智能体与 LangGraph 进阶扩展

前面的章节已经完成了一个基础智能体所需的主要能力：状态管理、条件路由、工具调用、会话检查点、上下文压缩和长期记忆。继续扩展项目时，不必急着增加更多设备类型，也可以把它作为一个 LangGraph 学习实验场，研究智能体如何暂停、规划、纠错、并行执行和协作。

下面这些方向主要服务于智能体原理与 LangGraph 框架学习，不以生产部署和工程优化为重点。

#### 6.5.1 Human-in-the-loop：让智能体暂停并等待用户确认

在加入 Human-in-the-loop 之前，图的主要执行路径是：

```text
sync_context → compact_context → agent → tools → agent
```

如果模型决定调用工具后立即执行，对于查询设备状态、打开普通灯光等低风险操作没有问题；但批量关闭设备、修改整个家庭场景或控制高风险设备时，更合理的流程是先暂停。当前项目已经采用下面的流程：

```text
用户请求
  ↓
识别操作和目标
  ↓
风险判断
  ├── 低风险 → 直接执行
  └── 中高风险 → interrupt 暂停
                       ↓
                  用户确认或拒绝
                       ↓
                  Command 恢复执行
```

这一实现集中使用了 LangGraph 的以下能力：

- `interrupt()`：在节点内部暂停图执行，并将待确认信息返回给调用方；
- `Command(resume=...)`：携带用户决定，从原检查点继续运行；
- Checkpoint：保存暂停前的状态，使应用不必重新执行前面的节点；
- durable execution：即使确认发生在下一次请求中，也能继续原任务。

例如，用户说“把家里所有设备都关掉”时，系统会生成待确认对象：

```python
class PendingApproval(TypedDict):
    operation: str
    target_ids: list[str]
    risk_level: str
    summary: str
```

确认节点调用 `interrupt`：

```python
from langgraph.types import interrupt

def approval_node(state):
    decision = interrupt({
        "question": "即将关闭 8 台设备，是否继续？",
        "operation": state["pending_operation"],
    })
    return {"approval_decision": decision}
```

应用收到用户确认后，再使用相同的 `thread_id` 恢复：

```python
from langgraph.types import Command

graph.invoke(
    Command(resume={"approved": True}),
    config=context.to_config(),
)
```

这个实验的重点不是增加一个确认弹窗，而是理解 LangGraph 图可以跨请求暂停和恢复，它不是一次性执行完毕的普通函数。

##### 当前项目中的实现

本项目已经将这一流程接入 Agent 图。当前策略是：

- `activate_scene` 会同时修改多台设备，因此执行前必须确认；
- `control_light`、`control_ac` 等普通单设备操作仍然直接执行；
- 用户拒绝后不会调用真实工具，也不会改变任何设备状态；
- 批准和拒绝都使用原来的 `session_id/thread_id` 恢复图；
- CLI 会自动识别 `__interrupt__`，显示确认面板并使用 `Command(resume=...)` 继续。

相关代码分工如下：

| 文件 | 职责 |
| --- | --- |
| `src/agent/approval.py` | 判断工具调用是否需要确认，构造中断数据，解析恢复决定 |
| `src/agent/graph.py` | 增加 `approval` 和 `reject_tools` 节点及条件边 |
| `src/agent/state.py` | 保存确认请求和确认结果 |
| `src/main.py` | CLI 展示确认信息，并通过 `Command` 恢复执行 |
| `tests/test_phase_six.py` | 验证批准、拒绝和普通操作三条路径 |

实际图结构为：

```text
agent
  ├── 没有工具调用 ───────────────────────────→ END
  ├── 普通工具调用 ───────────────────────────→ tools
  └── activate_scene ─→ approval
                          ├── approved ────────→ tools
                          └── rejected ────────→ reject_tools
                                                     ↓
compact_context ←────────────────────────────────────┘
       ↓
     agent
```

`approval_node` 不执行设备操作，只负责暂停：

```python
def approval_node(state: AgentState) -> dict:
    last_msg = state["messages"][-1]
    request = build_approval_request(last_msg.tool_calls)
    decision = interrupt(request)
    approved = approval_is_granted(decision)
    return {
        "approval_request": request,
        "approval_decision": "approved" if approved else "rejected",
    }
```

如果用户批准，条件边才会进入原来的 `ToolNode`。如果用户拒绝，`reject_tools` 会为每个被拒绝的工具调用生成对应的 `ToolMessage`：

```text
用户未批准该操作，工具没有执行，任何设备状态都未改变。
```

这一步很重要。模型已经生成了带 `tool_call_id` 的工具调用，如果直接跳回 Agent 而不给出对应工具结果，消息历史会成为不完整的工具调用协议。用拒绝结果闭合工具调用后，模型可以正常回复“操作已取消”。

CLI 中的恢复逻辑如下：

```python
result = graph.invoke(state_input, config)

while result.get("__interrupt__"):
    payload = result["__interrupt__"][0].value
    approved = ask_user(payload)
    result = graph.invoke(
        Command(resume={"approved": approved}),
        config,
    )
```

恢复时不能创建新的 `session_id`。`Command` 必须使用与发生中断时相同的配置，否则 LangGraph 无法找到暂停的检查点。

可以使用下面的指令体验流程：

```text
用户：我要出门了。
Agent：准备调用 activate_scene(scene_name="离家模式")。
系统：暂停并显示离家模式会修改的设备范围。
用户：y
系统：从检查点恢复，执行离家模式。
```

拒绝路径：

```text
用户：我要睡觉了。
系统：是否执行睡眠模式？
用户：n
Agent：已取消操作，设备状态没有变化。
```

自动化测试还会在暂停后读取模拟设备状态，确保设备只有在 `approved=True` 的恢复请求之后才发生变化。这样验证的是实际副作用，而不只是检查返回文字。

#### 6.5.2 显式意图识别与条件路由

阶段八已经在 `sync_context` 后增加结构化意图路由。Router 先识别请求属于哪类业务，再决定进入普通 ReAct、阶段七 Planner 或澄清节点；具体工具仍由后续节点选择：

```text
sync_context
  ↓
intent_router
  ├── device_query       → 设备查询分支
  ├── device_control     → 设备控制分支
  ├── scene_control      → 场景分支
  ├── memory_management  → 记忆分支
  ├── general_chat       → 通用对话分支
  └── clarification      → 澄清分支
```

路由结果使用结构化输出，而不是让代码解析模型生成的自然语言：

```python
class IntentResult(BaseModel):
    intent: Literal[
        "device_query",
        "device_control",
        "scene_control",
        "memory_management",
        "general_chat",
        "clarification",
    ]
    confidence: float
    reason: str
```

然后由条件边根据 `state["intent"]` 选择下一个节点：

```python
workflow.add_conditional_edges(
    "intent_router",
    route_by_intent,
    {
        "device_query": "device_query_agent",
        "device_control": "device_control_agent",
        "scene_control": "scene_agent",
        "memory_management": "memory_agent",
        "general_chat": "chat_agent",
        "clarification": "clarification_node",
    },
)
```

可以对比两种设计：

| 设计 | 优点 | 适合研究的问题 |
| --- | --- | --- |
| 单 Agent 自主选工具 | 图简单，模型自由度高 | ReAct、工具选择、提示词设计 |
| Router + 专用分支 | 路径清晰，状态容易约束 | 结构化输出、条件边、模块边界 |

不要默认认为节点越多越先进。这个实验真正需要观察的是：增加路由后，工具选择是否更准确，复杂度和模型调用次数是否值得。

#### 6.5.3 Planner–Executor：把规划和执行分开

阶段七已经实现 Planner–Executor 分支。项目现在保留两种执行方式：

- 普通单步请求和预定义场景继续使用原来的 ReAct 分支；
- 明确包含多个自定义设备动作的请求进入 Planner 分支。

例如：

```text
“关闭客厅灯，然后打开卧室空调到 25 度” → Planner
“关闭客厅灯”                           → 普通 ReAct
“我要出门了”                           → 预定义离家场景 + 场景确认
```

当前实现位于 `src/agent/routing.py`。`task_router` 会保存：

```python
intent: str
intent_confidence: float
intent_reason: str
intent_route: Literal["react", "planner", "clarification"]
```

正常运行时使用 `llm.with_structured_output(IntentResult)`。如果模型不支持结构化输出或调用异常，系统会回退到保守的关键词分类器，避免路由故障导致整个图不可用。低于 `ROUTING_CONFIDENCE_THRESHOLD`（默认 `0.6`）或明确分类为 `clarification` 的请求会直接询问用户补充设备、房间或动作，不会调用工具。

阶段八没有为六种意图立即创建六套 Agent。当前业务映射是：

```text
复杂多动作 device_control → Planner–Executor–Verifier
低置信度 / clarification   → clarification_node
其余已识别意图             → 现有 ReAct Agent
```

这样先把“识别意图”和“执行工具”拆开，同时保留阶段六场景确认、阶段七规划循环和已有记忆工具。下一阶段再把稳定的业务流程封装成子图。

路由器采用保守规则，只有检测到多个动作，并且涉及多种设备或明显连接词时才启用 Planner。这样可以避免所有请求都额外调用一次规划模型。

当前流程为：

```text
用户目标
  ↓
planner 生成计划
  ↓
plan_approval 展示完整计划并暂停
  ↓ approved
executor 执行当前步骤
  ↓
verifier 检查步骤结果
  ├── 成功且还有步骤 → executor
  ├── 全部完成         → final
  ├── 可以修复         → retry 当前步骤
  └── 重试耗尽         → planner 带失败信息重新规划
```

计划位于 `src/agent/planning.py`，使用 Pydantic 结构化输出：

```python
class PlanStep(BaseModel):
    step_id: int
    description: str
    tool_name: Literal[
        "control_light", "control_ac", "control_tv", "control_curtain"
    ]
    arguments: dict

class ExecutionPlan(BaseModel):
    goal: str
    rationale: str
    steps: list[PlanStep]
```

Planner 使用：

```python
structured_planner = llm.with_structured_output(ExecutionPlan)
plan = structured_planner.invoke(planner_prompt)
```

模型只能生成计划，不能在 Planner 节点中执行工具。工具名也被限制为四个原子设备工具，不允许在自定义计划中嵌套 `activate_scene`。

`AgentState` 已增加：

```python
planning_goal: str
plan: dict
plan_revision: int
current_step_index: int
step_retry_count: int
replan_count: int
planning_status: str
planning_results: list[dict]
planning_failure_feedback: str
```

例如用户输入：

```text
关闭客厅灯，然后打开卧室空调到 25 度。
```

Planner 可以生成：

```text
1. control_light(device_name="客厅灯", action="off")
2. control_ac(device_name="卧室空调", action="on", temperature=25, mode="cool")
```

在真正执行前，`plan_approval` 会调用 `interrupt`，CLI 展示完整步骤。用户拒绝后任何步骤都不会执行；用户批准后 Executor 从第一步开始。

当前限制如下：

- 计划长度默认最多 8 步；
- 每一步默认最多重试 1 次；
- 整个任务默认最多重新规划 1 次；
- 重新规划产生新计划后，需要再次经过用户确认；
- 当前按顺序执行，阶段八再研究并行步骤。

配置项位于 `.env`：

```text
PLANNING_ENABLED=true
PLANNING_MAX_STEPS=8
PLANNING_MAX_STEP_RETRIES=1
PLANNING_MAX_REPLANS=1
```

#### 6.5.4 Reflection 与 Verifier：让智能体检查执行结果

阶段七没有让模型对自己的操作进行纯文本“自我评价”，而是实现了确定性的 Verifier。工具返回“调用成功”不一定代表用户目标已经实现，因此 Verifier 会根据工具名和参数推导期望状态，再读取 `DeviceRegistry` 的真实设备状态进行比较。

```text
executor 调用原子工具
  ↓
verifier
  ├── success + 还有步骤 → executor
  ├── success + 全部完成 → planning_finalize
  ├── failure + 未达重试上限 → executor 重试当前步骤
  ├── failure + 可重新规划 → planner
  └── failure + 已达上限 → planning_finalize
```

验证结果结构化为：

```python
class VerificationResult(BaseModel):
    success: bool
    problem_type: Literal[
        "none",
        "device_not_found",
        "tool_error",
        "state_mismatch",
        "unsupported_action",
    ]
    reason: str
    actual_state: dict
    expected_state: dict
```

例如执行：

```python
control_ac(device_name="卧室空调", action="set_temp", temperature=25)
```

Verifier 推导期望值：

```python
{"power": True, "temperature": 25}
```

然后读取 `registry.get(device_id)`，只有真实状态匹配时才判定成功。工具文字声称成功但设备状态没有改变时，会得到 `state_mismatch`，触发重试。

当同一步骤重试耗尽时，Planner 会收到类似反馈：

```text
步骤 1（关闭客厅灯）失败：device state mismatch...工具结果：...
```

Planner 根据反馈生成修订版计划，`plan_revision` 加一，并再次请求用户批准。如果重新规划次数也耗尽，图会停止，不会无限循环。

目前 Reflection 主要体现在确定性的状态验证和失败反馈中。对于“环境是否舒适”“观影氛围是否合适”这类主观目标，后续可以再增加模型型 Verifier，但不应替代当前可验证设备状态检查。

#### 6.5.5 Evaluator–Optimizer：生成后评分并改进

Verifier 主要判断操作是否成功，Evaluator–Optimizer 更适合评价计划或最终回答的质量。例如场景计划生成后，可以在执行前检查：

- 是否覆盖了用户提出的全部目标；
- 是否选择了正确的房间和设备；
- 是否违反家庭约束；
- 是否使用了相关个人偏好；
- 是否包含多余操作；
- 是否需要用户确认。

评价节点输出：

```python
class PlanEvaluation(BaseModel):
    score: int
    missing_requirements: list[str]
    safety_problems: list[str]
    unnecessary_actions: list[str]
    revision_advice: str
```

图结构可以是：

```text
planner → evaluator
             ├── score 达标   → executor
             └── score 不达标 → planner 根据反馈修改
```

这种模式适合学习生成器与评价器的职责分离，但也要设置最大优化次数。否则模型可能反复润色一个已经可用的计划，增加调用成本却没有明显收益。

#### 6.5.6 使用 Subgraph 拆分复杂流程

随着节点增加，把所有逻辑放在同一个图中会越来越难理解。LangGraph 子图可以把一段完整流程封装起来：

阶段九已经把多设备状态查询封装为 `src/agent/parallel.py` 中的设备查询子图。主图只负责识别 `device_query` 并把请求交给子图；子图内部负责解析目标、并行读取状态和汇总回复。这使并行查询可以脱离完整 Agent 单独测试。

在学习顺序上，建议先理解 6.5.2 的结构化意图路由，再考虑如何把不同意图对应的流程封装成子图。结构化路由负责回答“这次请求应该进入哪条业务路径”，子图负责回答“这条路径内部如何组织节点”。前者是清晰划分边界的推荐前置，但不是使用 LangGraph `Subgraph` 的硬性技术前提；即使没有 Router，也可以先把一个已经稳定的流程抽成子图。

```text
主图
├── 设备控制子图
│   └── 定位目标 → 能力校验 → 风险判断 → 执行 → 验证
├── 场景规划子图
│   └── 生成计划 → 评价 → 执行 → 汇总
├── 记忆管理子图
│   └── 抽取候选 → 校验 → 冲突判断 → 确认 → 保存
└── 通用对话节点
```

学习子图时需要区分两种状态：

- 父图和子图共享的公共状态，例如 `messages`、`request_user_id`；
- 子图内部使用的私有状态，例如 `current_step`、`risk_result`。

子图适合解决“一个流程已经可以单独画成图”的问题。普通的单步工具函数没有必要为了使用 Subgraph 而强行包装成子图。

#### 6.5.7 动态并行：Fan-out 与结果聚合

智能家居任务经常包含多个互不依赖的操作。例如用户要求查询客厅、卧室和书房温度，可以并行分发：

当前阶段九实现的是无副作用的多设备状态查询。查询目标在运行时从设备名称或房间名称中解析，每个目标通过 `Send("query_device", ...)` 分发；`parallel_results` 使用 `Annotated[list[dict], operator.add]` reducer 合并，最后按设备 ID 排序后生成稳定回复。

动态并行通常放在路由和子图之后学习：先由结构化路由确定“这是设备查询还是设备控制”，再由对应子图决定哪些步骤可以并行。它们之间是推荐的分层关系，而不是严格的 API 依赖关系。真正决定能否并行的是任务之间是否相互独立，以及状态是否可以安全聚合。

```text
                 ┌→ 查询客厅温度 ─┐
解析查询范围 → fan-out → 查询卧室温度 ─→ aggregate → 最终回答
                 └→ 查询书房温度 ─┘
```

离家场景也可以将没有依赖关系的任务并行执行：

```text
关闭灯光 ───────┐
关闭空调 ───────┤
关闭影音设备 ────┼→ 汇总成功和失败结果
检查门窗状态 ────┘
```

这一方向可以学习：

- 使用动态 `Send` 根据运行时目标数量生成任务；
- 使用 reducer 合并多个并行分支写入的状态；
- 区分“全部成功才成功”和“允许部分成功”；
- 处理并行结果的顺序不确定性；
- 避免两个分支同时覆盖同一个普通状态字段。

例如为结果字段定义列表合并 reducer：

```python
import operator
from typing import Annotated

parallel_results: Annotated[list[dict], operator.add]
```

如果任务之间存在依赖，例如“先关闭燃气，再确认阀门状态”，就不应该为了并行而并行。

#### 6.5.8 多智能体：按职责协作，而不是堆叠角色

当设备控制、场景规划、记忆管理和安全审查已经形成清晰边界后，可以进一步研究多智能体：

阶段十已经实现 Supervisor 模式。阶段八的结构化 `task_router` 同时承担 Supervisor：根据意图设置 `delegated_agent`，再把请求交给只拥有对应工具集的专用 Agent。专用 Agent 最终回复后进入 `supervisor_finalize`，记录本轮协作已经完成。

```text
Supervisor
├── Device Agent：设备定位、查询和控制
├── Scene Agent：复杂目标拆解和多设备计划
├── Memory Agent：记忆检索、候选与冲突分析
└── Safety Agent：检查操作范围、约束和风险
```

常见协作方式有两种：

```text
Supervisor 模式：
用户 → Supervisor → 专用 Agent → Supervisor → 最终回答

Handoff 模式：
用户 → Supervisor → Device Agent → Safety Agent → 最终回答
```

多智能体学习的重点不是创建多个不同人格，而是回答这些问题：

- 每个 Agent 是否拥有明确且互斥的职责；
- Agent 之间传递完整对话，还是传递结构化任务结果；
- 谁负责最终回答；
- 如何限制转交次数，避免互相来回调用；
- 多 Agent 是否真的比单 Agent 加工具具有更高成功率。

如果职责尚未清楚，优先使用一个 Agent 加子图；当不同模块确实需要不同工具集、提示词和状态边界时，再拆成多个 Agent。

当前项目实际拆分为：

| Agent | 允许使用的工具 |
| --- | --- |
| Device Agent | 四类原子设备控制工具、设备状态查询 |
| Scene Agent | `activate_scene`、`list_scenes` |
| Memory Agent | 长期记忆、候选确认和版本管理工具 |
| Chat Agent | 不绑定工具 |

安全能力没有被包装成一个只输出文字的“角色”。场景副作用继续由阶段六 Human-in-the-loop 拦截，多步骤设备操作继续由阶段七 Verifier 检查真实状态。这些确定性节点共同构成 Safety 层。

协作状态包含：

```python
delegated_agent: Literal["device", "scene", "memory", "chat"]
handoff_count: int
collaboration_status: Literal["delegated", "working", "completed", "stopped"]
```

`MULTI_AGENT_MAX_HANDOFFS` 默认是 2。当前版本每轮只进行一次 Supervisor 委派，该限制为后续跨 Agent handoff 预留终止边界。

#### 6.5.9 让长期记忆显式参与推理

当前项目通过 `sync_context` 检索 Top-K 记忆，并将其格式化后放进系统提示词：

阶段十一已经增加 `memory_reasoner`。`sync_context` 会同时保留结构化 `retrieved_memories`，随后生成 `MemoryDecision`，明确哪些记忆适用、哪些被当前临时指令覆盖、哪些属于必须遵守的约束，以及是否存在冲突。

```text
检索记忆 → 拼接 Prompt → 模型自行判断如何使用
```

可以增加 `memory_reasoner` 节点，把“找到了什么记忆”和“本轮应该采用什么记忆”分开：

```text
用户请求
  ↓
检索相关记忆
  ↓
memory_reasoner
  ├── 适用偏好
  ├── 必须遵守的家庭约束
  ├── 被当前明确指令覆盖的记忆
  ├── 存在冲突的记忆
  └── 是否需要澄清
  ↓
planner / agent
```

结构化结果示例：

```python
class MemoryDecision(BaseModel):
    applicable_memory_ids: list[str]
    ignored_memory_ids: list[str]
    constraints: list[str]
    preferences: list[str]
    conflicts: list[str]
    needs_clarification: bool
```

这一实验可以研究重要的记忆推理原则：

```text
当前用户的明确指令
> 必须遵守的家庭安全约束
> 更具体作用域的已确认偏好
> 通用个人偏好
> 自动推断但尚未确认的候选
```

检索到一条记忆不代表一定要使用它。将适用性判断显式化，能够让规划和记忆模块的边界更容易观察。

#### 6.5.10 时间旅行与状态调试

Checkpoint 不只是保存聊天记录，还可以支持查看过去状态并从指定位置重新执行。可以为项目增加一个学习用调试入口：

阶段十一已经提供 `src/agent/time_travel.py`：`list_state_history` 查看历史，`fork_from_checkpoint` 使用完整历史快照配置调用 `graph.update_state` 创建分支。CLI 中可输入 `/history` 查看当前会话最近十个 Checkpoint。

```text
查看某个 thread_id 的状态历史
  ↓
选择一个 checkpoint
  ↓
检查当时的 messages、plan、memory_context 和设备目标
  ↓
修改某个状态字段或输入
  ↓
从该位置创建新的执行分支
```

例如，同一个“执行离家模式”请求可以形成两条轨迹：

```text
原始轨迹：规划 → 直接执行 → 完成
实验轨迹：规划 → 风险确认 → 用户修改范围 → 执行
```

通过比较两条轨迹，可以直观看到节点分别修改了哪些状态，以及条件边为何选择不同路径。这比只查看最终回复更有助于理解 LangGraph 的状态机和持久化执行模型。

#### 6.5.11 流式输出与自定义进度事件

智能体的“流式输出”不只包含模型逐字生成的 Token，还包括图执行进度和工具状态：

阶段十一已经在关键节点中通过 `get_stream_writer` 发出 `custom` 事件。调用方可以使用 `graph.stream(input, config, stream_mode="custom")` 获取上下文同步、记忆判断、Supervisor 路由、规划、步骤执行、验证、并行查询和 Agent 完成事件。

```text
正在识别房间和设备……
已找到客厅的 3 盏灯
正在执行关闭操作……
2 台成功，1 台离线
正在检查最终状态……
```

可以分别观察：

- 模型 Token 流：最终自然语言回答的增量内容；
- 状态更新流：每个节点对状态的局部修改；
- 完整状态流：节点执行后的完整状态快照；
- 自定义事件：工具主动报告执行进度。

学习时应避免把模型的隐藏推理过程直接展示给用户。可展示的是任务状态、工具进度和可验证结果，而不是模型内部思维文本。

#### 6.5.12 Agentic RAG：区分记忆、实时状态和外部知识

长期记忆保存用户偏好和家庭规则，但不适合保存设备说明书、故障代码和产品知识。可以增加设备文档 RAG 分支：

```text
用户：空调显示 E3 是什么意思？
  ↓
识别为设备知识查询
  ↓
根据当前设备获得品牌和型号
  ↓
检索对应说明书片段
  ↓
结合引用内容回答
```

项目中的信息来源应保持清晰边界：

| 信息 | 正确来源 |
| --- | --- |
| 当前开关、温度、在线状态 | 设备平台或模拟后端 |
| 用户喜欢 25°C | 长期记忆 |
| 本轮正在讨论卧室空调 | 会话状态 |
| 空调 E3 故障含义 | 说明书 RAG |
| 当前用户能否操作设备 | 可信权限上下文 |

Agentic RAG 与普通问答 RAG 的区别在于：智能体可能先调用设备工具取得型号，再检索知识库，还可能根据检索结果继续查询状态或给出下一步操作建议。

#### 6.5.13 推荐的进阶学习顺序

这些方向不建议一次全部加入当前图。可以按照对 LangGraph 核心能力的依赖关系分阶段学习：

```text
阶段六：人在回路与可恢复执行（已实现）
  interrupt、Command、Checkpoint 恢复、确认分支
        ↓
阶段七：规划、执行和反思（已实现）
  Planner、Executor、Verifier、重试与重新规划
        ↓
阶段八：结构化意图路由（已实现）
  IntentResult、置信度、澄清分支、条件边
        ↓
阶段九：子图与动态并行（已实现）
  Subgraph、Send、Reducer、结果聚合
        ↓
阶段十：多智能体协作（已实现）
  Supervisor、Handoff、专用 Agent、协作终止条件
        ↓
阶段十一：记忆显式参与推理、时间旅行与流式事件（已实现）
  MemoryReasoner、状态历史、可观测进度
        ↓
阶段十二：Agentic RAG 与轨迹评测
  知识检索、来源路由、执行轨迹比较
```

这里的阶段编号是学习路线编号，不等同于仓库中的历史迭代编号。当前项目已经完成：

- 阶段六：Human-in-the-loop（对应 6.5.1）；
- 阶段七：Planner–Executor，以及确定性的 Verifier、重试和重新规划（对应 6.5.3 和 6.5.4）。

阶段八至阶段十一均已完成：结构化 Router、动态并行查询子图、Supervisor 专用 Agent、显式记忆推理、Checkpoint 时间旅行和自定义进度事件已经形成一条可观察的完整执行链。下一步推荐进入阶段十二 Agentic RAG 与轨迹评测；多设备控制并行仍应等到依赖关系、审批范围和失败策略明确后再扩展。

### 6.6 家居领域模型训练：SFT、LoRA 与强化学习

前面的扩展主要改造 Agent 工作流：通过图结构、工具、记忆和检索提高任务完成能力。另一条学习路线是训练模型本身，使它更熟悉智能家居语言、工具协议、设备约束和多步骤任务。

这两条路线解决的问题不同：

| 层次 | 主要解决的问题 | 本项目中的例子 |
| --- | --- | --- |
| LangGraph 工作流 | 流程是否可控、可暂停、可恢复 | 路由、确认、规划、反思、工具执行 |
| RAG | 模型是否能获得外部知识 | 查询设备说明书和故障代码 |
| 长期记忆 | 模型是否获得当前家庭和用户信息 | 用户偏好、家庭规则、设备别名 |
| SFT | 模型是否学会期望的输入输出行为 | 正确理解指令、选择工具、生成参数 |
| LoRA/QLoRA | 如何用较少资源完成参数高效微调 | 只训练少量适配参数 |
| 偏好优化/RL | 多个可行行为中，哪个更符合目标 | 安全、少打扰、少操作、任务完成率高 |

训练模型不能替代工具权限、设备状态查询和高风险确认。即使经过充分训练，模型也不应该自行决定可信的 `home_id`、`user_id` 或管理员身份。

#### 6.6.1 先明确训练目标

不要一开始就把所有家居能力放进同一个训练任务。可以把目标拆成以下几类：

1. **领域语言理解**：理解“调暗一点”“有点闷”“睡眠模式”等家居表达；
2. **设备目标解析**：识别房间、设备类型、设备名称和指代对象；
3. **工具选择**：判断应该查询状态、控制设备、激活场景还是管理记忆；
4. **参数生成**：生成符合工具 Schema 的温度、亮度、模式等参数；
5. **澄清能力**：目标不唯一或参数缺失时主动询问；
6. **安全决策**：批量或高风险操作先确认，不伪造执行结果；
7. **多步骤规划**：将睡眠、离家、观影等目标拆成可执行步骤；
8. **结果总结**：根据真实 ToolMessage 说明成功、失败和未完成项目。

建议先训练边界清晰、容易自动判分的任务，例如设备解析和工具调用，再逐渐加入规划、偏好和安全决策。

#### 6.6.2 推荐的完整训练路线

一个适合本项目的学习顺序是：

```text
准备基础模型和 Tokenizer
        ↓
构造家居领域数据集
        ↓
SFT：学习领域表达、工具调用和基本规划
        ↓
LoRA/QLoRA：以参数高效方式完成 SFT
        ↓
离线评测和 LangGraph 集成测试
        ↓
偏好数据构造
        ↓
DPO 等偏好优化
        ↓
在 SimulatorBackend 中进行可验证 RL
        ↓
与未微调模型、纯 SFT 模型进行对照实验
```

这里需要避免一个常见误解：SFT 是训练目标，LoRA 是参数更新方式。可以进行“全参数 SFT”，也可以进行“LoRA SFT”；LoRA 并不是必须排在 SFT 之后的另一种能力训练。

#### 6.6.3 基础模型应该如何选择

选择基础模型时，优先考虑：

- 是否原生支持中文；
- 是否支持 Chat Template；
- 是否具备结构化输出和工具调用基础能力；
- 模型许可证是否允许研究和发布适配权重；
- 本地显存是否能够承担训练和推理；
- 是否能稳定输出项目需要的工具调用格式。

对于学习实验，小模型更适合快速迭代。训练目标应该是验证方法是否有效，而不是一开始追求最大的参数规模。先用小模型完成数据管线、评测和 Agent 集成，再决定是否扩大模型。

还要固定以下信息，保证实验可比较：

```text
基础模型版本
Tokenizer 版本
Chat Template
最大上下文长度
工具调用表示方式
训练数据版本
随机种子
```

不同模型的工具调用协议可能不同。训练数据必须使用目标模型所支持的模板，不能简单把一个模型的特殊 Token 和工具格式复制给另一个模型。

#### 6.6.4 如何从当前项目构造训练数据

本项目已经具备生成高质量数据的几个重要条件：

- `DeviceRegistry` 提供设备列表和能力；
- `SimulatorBackend` 可以执行操作并返回确定结果；
- 工具函数定义了参数 Schema；
- LangGraph 状态可以记录完整执行轨迹；
- 长期记忆提供用户偏好和家庭规则；
- 测试用例提供了正确行为的种子样本。

可以从以下来源构造数据：

| 数据来源 | 作用 | 注意事项 |
| --- | --- | --- |
| 人工编写 | 建立高质量核心样本 | 数量少但应覆盖关键边界 |
| 规则模板生成 | 扩充设备、房间和参数组合 | 避免句式过于单一 |
| 强模型生成 | 生成口语化表达和复杂场景 | 必须经过规则或人工校验 |
| Agent 运行轨迹 | 收集真实工具选择和执行过程 | 删除错误轨迹和敏感身份数据 |
| 单元测试案例 | 构造确定性正反例 | 适合作为评测集种子 |
| 用户反馈样本 | 学习真实错误和偏好 | 需要授权、脱敏和质量审查 |

不要把未经筛选的对话日志直接作为训练集。日志中可能包含错误工具调用、模型幻觉、重复内容、隐私数据和旧版本协议。

#### 6.6.5 定义统一的家居任务样本

为了让数据可追踪，可以先定义独立于具体训练框架的中间格式：

```json
{
  "sample_id": "control-0001",
  "task_type": "device_control",
  "context": {
    "room_id": "living_room",
    "available_devices": ["living_room_light", "living_room_ac"]
  },
  "user_input": "客厅有点暗，帮我把灯开到六成",
  "expected_action": {
    "tool": "control_light",
    "arguments": {
      "device_name": "客厅灯",
      "action": "set_brightness",
      "brightness": 60
    }
  },
  "expected_response": "已将客厅灯亮度调到 60%。",
  "safety_tags": [],
  "source": "human"
}
```

这个中间格式之后再转换成目标模型的 Chat Template。这样更换模型时不需要重写原始数据。

建议至少包含以下任务类型：

```text
device_query
device_control
scene_control
target_resolution
clarification
memory_save
memory_retrieval
plan_generation
tool_result_summary
safety_confirmation
general_chat
```

#### 6.6.6 SFT 数据应该覆盖什么

SFT，也就是监督微调，本质上是给模型示范“面对这种输入，应该输出什么”。在本项目中，SFT 数据不应该只有自然语言问答，还要覆盖完整的工具交互。

**一、直接回答样本**

```text
用户：你能控制哪些设备？
助手：我可以查询和控制当前家庭中已接入的灯光、空调、电视等设备……
```

**二、单次工具调用样本**

```text
用户：把卧室空调调到 25 度。
助手工具调用：control_ac(device_name="卧室空调", action="set_temp", temperature=25)
工具结果：设置成功，当前温度 25°C。
助手：已将卧室空调调到 25°C。
```

**三、需要澄清的样本**

```text
上下文：客厅有两盏灯。
用户：把灯关掉。
助手：客厅有主灯和落地灯，你想关闭哪一盏，还是全部关闭？
```

**四、拒绝伪造执行的样本**

```text
工具结果：设备离线，操作失败。
正确回答：卧室空调当前离线，暂时无法完成设置。
错误回答：已为你调到 25°C。
```

**五、多轮指代样本**

```text
用户：打开书房灯。
助手：已打开书房灯。
用户：再亮一点。
助手工具调用：对刚才的书房灯提高亮度。
```

**六、记忆适用性样本**

```text
长期偏好：用户一般喜欢空调 25°C。
用户：今天帮我调到 27°C。
正确行为：执行本轮明确的 27°C，不用历史偏好覆盖当前指令。
```

高质量 SFT 数据既要教模型“应该做什么”，也要教它“什么时候不应该调用工具”。

#### 6.6.7 正例、难例和反事实样本

如果训练集中全是完整、明确、可成功执行的请求，模型会倾向于任何时候都直接操作。应该主动加入难例：

- 不存在的设备；
- 同名或同类型设备不唯一；
- 温度、亮度等参数越界；
- 设备离线；
- 用户临时指令与长期偏好冲突；
- 家庭规则与个人偏好冲突；
- 用户只是在询问，并没有要求执行；
- 用户引用过去状态，但当前状态已经变化；
- ToolMessage 返回部分成功；
- 请求包含多个可以并行或具有依赖关系的步骤。

还可以为同一个请求构造反事实上下文：

```text
相同输入：“把灯关掉。”

上下文 A：当前房间只有一盏灯 → 可以直接执行
上下文 B：当前房间有三盏灯   → 应该澄清或确认全部关闭
上下文 C：没有可信房间上下文 → 应该询问房间
```

这能迫使模型学习使用上下文，而不是只记住表面句式。

#### 6.6.8 数据划分与防止泄漏

训练集、验证集和测试集不能只做随机句子切分。模板生成的数据高度相似，随机切分会让测试结果虚高。

更合理的划分方式包括：

- 按表达模板划分：测试集中使用训练时没出现的说法；
- 按场景划分：将某些复杂场景只放入测试集；
- 按设备组合划分：测试新的房间和设备组合；
- 按用户目标划分：测试未见过的多步骤目标；
- 按时间划分：较新的真实轨迹作为测试集。

评测数据应独立保存，不能继续被数据生成脚本作为提示示例，也不能在偏好优化阶段重新进入训练集。

#### 6.6.9 LoRA 和 QLoRA 在做什么

全参数微调会更新模型的全部参数，资源消耗较大。LoRA 冻结原模型参数，只在部分线性层旁增加低秩适配矩阵：

```text
原始权重 W 保持冻结
实际输出使用 W + ΔW
ΔW = B × A
```

其中 `A` 和 `B` 的维度远小于原始权重，因此需要训练和保存的参数更少。

QLoRA 通常进一步将冻结的基础模型量化，以降低显存占用，而 LoRA 适配参数仍以适合训练的精度更新。它适合资源有限的学习实验，但量化方式、计算精度和硬件支持需要匹配。

LoRA 常见配置概念包括：

| 配置 | 含义 | 观察重点 |
| --- | --- | --- |
| `r` | 低秩维度 | 越大容量越高，但参数和显存增加 |
| `lora_alpha` | LoRA 更新缩放 | 与 `r` 一起影响更新强度 |
| `lora_dropout` | 适配层 Dropout | 小数据集可用于抑制过拟合 |
| `target_modules` | 注入 LoRA 的模块 | 必须与模型结构匹配 |
| learning rate | 适配参数学习率 | 过大可能破坏原有能力 |

示意配置如下，具体模块名必须根据所选模型确认：

```python
lora_config = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
}
```

不要照抄参数作为固定答案。应该通过验证集比较不同配置，并观察：

- 工具选择是否提高；
- 参数 JSON 是否更稳定；
- 通用对话能力是否下降；
- 是否过度调用家居工具；
- 是否只记住训练模板。

#### 6.6.10 SFT 训练时的关键设置

训练框架不同，但核心概念基本一致：

```text
模型与 Tokenizer
Chat Template
训练/验证数据集
最大序列长度
批大小与梯度累积
学习率和调度器
训练轮数
混合精度
LoRA 配置
Checkpoint 与日志
```

对于包含工具调用的对话，应保证一次样本中的关键链路完整：

```text
system
→ user
→ assistant tool_call
→ tool result
→ assistant final answer
```

Loss Mask 需要根据训练目标设计。一般可以只对助手输出计算损失，避免要求模型学习预测用户消息和系统提示。若框架不能正确识别 Chat Template 中的角色边界，必须先检查格式化后的 Token，而不是直接开始长时间训练。

建议先做小规模过拟合实验：使用几十条样本训练，确认模型能够记住这些样本、Loss 正常下降并正确输出工具格式。这个检查通过后，再扩大数据规模。

#### 6.6.11 训练后如何接入当前 LangGraph

微调模型接入后，不需要推翻现有图结构。`build_llm()` 仍然负责创建模型，后续继续执行工具绑定：

```text
微调后的模型
  ↓ bind_tools
LangGraph agent 节点
  ↓
ToolNode
```

接入时重点验证：

1. 微调模型是否支持当前工具调用协议；
2. `bind_tools()` 生成的 Schema 与训练数据是否一致；
3. 模型服务是否返回标准 `tool_calls`；
4. ToolMessage 是否能被模型正确理解；
5. 上下文中的长期记忆和可信身份是否仍按原流程注入；
6. 模型是否在无需工具时保持正常回答能力。

不要让训练数据中的固定家庭 ID、用户 ID 和设备 ID 代替运行时上下文。模型应学习“使用给定上下文”，而不是记住某个实验家庭。

#### 6.6.12 先评测 SFT，再考虑强化学习

如果 SFT 模型仍然无法稳定生成合法工具参数，直接进入 RL 通常不会解决基础格式问题。先建立可复现评测：

| 指标 | 说明 |
| --- | --- |
| Intent Accuracy | 是否识别正确任务类型 |
| Tool Selection Accuracy | 是否选择正确工具 |
| Argument Exact Match | 参数是否完全正确 |
| Argument Valid Rate | 参数是否通过 Schema 校验 |
| Clarification Accuracy | 应澄清时是否澄清 |
| Unsafe Action Rate | 是否错误执行高风险操作 |
| Hallucinated Success Rate | 工具失败后是否声称成功 |
| Task Success Rate | 最终设备状态是否符合目标 |
| Average Tool Calls | 完成任务平均调用次数 |
| Unnecessary Action Rate | 是否执行了多余操作 |

文本相似度不是家居 Agent 最核心的指标。模型说法不同没有关系，真正重要的是工具、参数、权限、设备最终状态和任务是否完成。

#### 6.6.13 偏好数据与 DPO

SFT 告诉模型一个可接受答案，偏好学习则提供同一输入下的“更好”和“更差”两个回答：

```json
{
  "prompt": "用户：把灯关掉。上下文：客厅有主灯和落地灯。",
  "chosen": "客厅有主灯和落地灯，你想关闭哪一盏，还是全部关闭？",
  "rejected": "好的，已经全部关闭。"
}
```

适合构造偏好对的维度包括：

- 澄清优于猜测；
- 使用真实工具结果优于伪造成功；
- 满足当前明确指令优于机械套用历史偏好；
- 最少必要操作优于无关的批量操作；
- 安全确认优于直接执行高风险操作；
- 简洁准确的结果总结优于冗长但遗漏失败项；
- 正确作用域的记忆优于其他用户或房间的记忆。

DPO 一类方法直接使用 `chosen/rejected` 偏好对优化模型，不需要先训练一个显式奖励模型，也不需要在线与环境持续交互。对于首次学习偏好优化，它通常比完整 PPO 流程更容易建立和调试。

偏好数据不能只通过把错误答案写得非常荒谬来构造。更有价值的是“困难负例”，例如两个回答都基本合理，但其中一个进行了不必要操作，或遗漏了一项家庭约束。

#### 6.6.14 家居 Agent 的奖励应该如何设计

强化学习需要将任务质量转换为奖励。智能家居环境有一个优势：大量结果可以通过模拟器和规则自动验证，而不必完全依赖另一个大模型打分。

可以将总奖励拆为：

```text
总奖励
= 任务完成奖励
+ 参数正确奖励
+ 安全合规奖励
+ 澄清正确奖励
+ 回答质量奖励
- 无效工具惩罚
- 多余操作惩罚
- 越权或高风险操作惩罚
- 声称成功但状态未变化的惩罚
```

示意函数：

```python
def calculate_reward(trajectory, goal, final_device_state):
    reward = 0.0
    reward += 2.0 if goal.is_satisfied(final_device_state) else -1.0
    reward += 0.5 if trajectory.all_arguments_valid else -0.5
    reward -= 0.1 * trajectory.unnecessary_tool_calls
    reward -= 2.0 * trajectory.unsafe_actions
    reward -= 1.0 if trajectory.claimed_false_success else 0.0
    return reward
```

实际奖励权重必须通过实验调整。某个惩罚过强可能使模型为了避免犯错而拒绝所有操作；任务奖励过强又可能促使模型忽略安全确认。

#### 6.6.15 使用 SimulatorBackend 构造 RL 环境

不要直接让正在训练的策略在真实住宅中探索。当前 `SimulatorBackend` 可以扩展为家居 RL 环境：

```text
reset()
  → 随机生成初始设备状态、在线状态和用户目标

observation
  → 用户请求、可用工具、设备状态、房间上下文、相关记忆

action
  → 文本回答、工具选择和工具参数

step(action)
  → 执行模拟工具，返回新状态、工具结果和局部奖励

done
  → 目标完成、不可恢复失败或达到最大步数
```

例如训练任务：

```text
初始状态：客厅灯开启，电视开启，空调开启；卧室灯关闭。
用户目标：我要出门了。
家庭约束：门锁操作必须确认。

成功条件：关闭指定可关闭设备，保留不应操作的设备，门锁前请求确认。
```

环境应支持随机化：

- 房间和设备数量；
- 设备名称和别名；
- 初始开关状态；
- 设备离线或工具失败；
- 用户偏好和家庭约束；
- 指令清晰度；
- 单步与多步骤目标。

随机化可以降低模型只记住固定家庭布局的风险。

#### 6.6.16 DPO、奖励模型、PPO 和 GRPO 的区别

这些方法处于不同复杂度层次：

| 方法 | 主要数据或环境 | 特点 | 在本项目中的用途 |
| --- | --- | --- | --- |
| SFT | 标准答案轨迹 | 学习基本行为 | 工具调用、澄清、规划 |
| DPO 类方法 | chosen/rejected 偏好对 | 不需要在线环境，流程相对简单 | 安全与行为偏好 |
| Reward Model | 带偏好或分数的数据 | 学习给回答或轨迹打分 | 为在线 RL 提供奖励 |
| PPO | 策略采样 + 奖励 + 价值估计 | 经典但流程和资源要求较高 | 多步策略优化实验 |
| GRPO 类方法 | 同一问题的多组采样和相对奖励 | 可利用组内相对结果优化 | 可验证家居任务和规划 |

对本项目更稳妥的顺序是：

```text
SFT/LoRA
→ 自动评测
→ DPO
→ SimulatorBackend 中的可验证奖励优化
→ 最后再研究更完整的在线 RL
```

算法名称不是项目重点。更重要的是能否准确说明：状态是什么、动作是什么、奖励如何计算、环境如何重置、轨迹如何终止，以及训练是否真的提高了未见任务的成功率。

#### 6.6.17 轨迹级奖励与过程奖励

家居 Agent 经常需要多步工具调用，因此只评价最终回答可能不够。

**结果奖励**只关注最终设备状态：

```text
离家目标是否完成？
所有应关闭设备是否关闭？
不应操作的设备是否保持原状？
```

**过程奖励**关注中间行为：

```text
是否先查询了必要状态？
目标不唯一时是否澄清？
高风险步骤前是否确认？
工具失败后是否正确处理？
是否出现无意义循环？
```

结果奖励通常更客观，过程奖励可以提高学习效率，但过度规定过程会限制模型找到更好的执行策略。建议以最终状态验证为主，只对安全约束和明显无效行为增加必要的过程奖励。

#### 6.6.18 防止奖励投机

模型可能找到“获得高分但没有真正完成任务”的漏洞，这称为奖励投机。例如：

- 只生成“已完成”文本，不调用工具；
- 重复调用查询工具刷取局部奖励；
- 为避免失败而永远请求澄清；
- 修改不相关设备以满足错误的状态判断；
- 利用模拟器未覆盖的边界条件绕过安全检查。

因此奖励应基于环境中的真实最终状态和完整轨迹，而不是只检查最终文本。还应设置：

```text
最大工具调用次数
最大图循环次数
无状态变化操作惩罚
错误设备操作惩罚
必须满足的安全约束
任务终止条件
```

每次发现奖励漏洞，都应把对应轨迹加入回归评测集。

#### 6.6.19 模型训练与 Agent 图如何分工

训练后仍然建议保留确定性的 LangGraph 控制：

```text
模型适合学习：
- 理解自然语言
- 选择工具和参数
- 生成计划
- 总结执行结果
- 在模糊信息下提出澄清

图和代码负责保证：
- 身份与权限来自可信上下文
- 参数通过 Schema 校验
- 高风险操作必须确认
- 工具真实执行
- 循环次数受限
- Checkpoint 可恢复
- 长期记忆按作用域隔离
```

一个好的领域模型可以减少错误决策，但不能成为系统唯一的安全边界。

#### 6.6.20 建立对照实验

为了说明训练是否有效，至少比较以下版本：

```text
A. 基础模型 + 当前 LangGraph
B. SFT/LoRA 模型 + 相同 LangGraph
C. SFT + DPO 模型 + 相同 LangGraph
D. 经过模拟环境 RL 的模型 + 相同 LangGraph
```

所有版本使用相同的：

- 测试集；
- 工具 Schema；
- 系统提示词；
- 设备初始状态；
- 最大步骤数；
- 解码参数；
- LangGraph 流程。

然后比较：

```text
任务成功率
工具选择准确率
参数合法率
澄清准确率
高风险误操作率
平均工具调用数
平均完成步数
未见表达和未见设备组合的泛化能力
```

只有控制其他变量，才能判断提升来自模型训练，还是来自提示词、工具描述或图流程的变化。

#### 6.6.21 推荐的数据目录设计

如果后续真正开始训练，可以在项目中单独规划数据目录，但不要把原始私人对话和大体积模型权重直接提交到 Git：

```text
data_training/
├── raw/                 # 原始数据，只读保存
├── normalized/          # 统一中间格式
├── sft/                 # 转换后的 SFT 数据
├── preference/          # chosen/rejected 数据
├── evaluation/          # 独立评测集
└── manifests/           # 数据版本、来源和统计信息

training/
├── configs/             # 实验配置
├── scripts/             # 数据处理、训练和评测入口
└── reports/             # 指标与实验结论
```

每份数据集建议记录：

```text
版本号
样本数量
任务类型分布
数据来源
生成模型或规则版本
人工审查比例
去重方式
训练/验证/测试划分规则
```

#### 6.6.22 推荐的实践阶段

可以把模型训练路线安排为：

```text
阶段十一：家居 SFT 数据集
  统一 Schema、人工种子、模板扩充、难例、独立评测集
        ↓
阶段十二：LoRA/QLoRA 监督微调
  工具调用、参数生成、澄清、结果总结、对照评测
        ↓
阶段十三：偏好优化
  chosen/rejected、安全偏好、DPO、困难负例
        ↓
阶段十四：模拟环境强化学习
  环境随机化、可验证奖励、轨迹采样、奖励投机分析
        ↓
阶段十五：Agent 联合评测
  基础模型、SFT、DPO、RL 模型在同一 LangGraph 中比较
```

第一次实践建议只选一个小而完整的闭环：

```text
选择一个支持工具调用的中文小模型
→ 构造单设备控制与澄清数据
→ 使用 LoRA 完成 SFT
→ 接入当前 LangGraph
→ 在固定测试集上比较工具选择和参数准确率
```

完成这个闭环后，再加入多步骤规划、DPO 和模拟器 RL。这样每个阶段的收益和问题都能被单独观察，而不会把数据、训练、奖励和 Agent 图的错误混在一起。

---

## 7. MCP 集成

### 7.1 为什么需要 MCP

- **工具标准化**: 任何支持 MCP 的 AI 客户端（Claude Desktop、Cursor 等）都能发现和调用你的智能家居工具
- **互操作性**: 你的工具可以被多个 AI 应用复用
- **生态整合**: 可以消费社区的 MCP 服务（天气、新闻、日历等）

### 7.2 在 Claude Desktop 中配置

编辑 Claude Desktop 配置文件：

```json
{
  "mcpServers": {
    "smart-home": {
      "command": "python",
      "args": ["-m", "src.mcp.server"],
      "cwd": "G:\\大厂学习\\minimind\\langgraph"
    }
  }
}
```

配置后，Claude Desktop 会自动发现 6 个智能家居工具。

### 7.3 连接外部 MCP 服务

在 `.env` 中配置：

```ini
# 连接天气 MCP 服务
EXTERNAL_MCP_SERVERS={"name":"weather","transport":"stdio","command":"python","args":["weather_mcp.py"]}
```

Agent 会自动发现外部工具并在对话中使用。

---

## 8. 部署建议

### 8.1 开发阶段（当前）

- 使用 `SimulatorBackend` 模拟设备
- 内存检查点（快速开发迭代）
- DEBUG 日志级别

### 8.2 生产部署

- 替换为真实 IoT 后端（Home Assistant / MQTT）
- 启用 SQLite 检查点（持久化对话）
- INFO 日志级别 + 文件日志
- 可选: 用 Docker 容器化部署
- 可选: 对接语音助手（ASR + TTS）

---

## 9. 常见问题

### Q: 运行报 "BAILIAN_API_KEY 未配置"
```
A: 编辑 .env 文件，将 BAILIAN_API_KEY 设为真实的百炼 API Key
   不要加引号: BAILIAN_API_KEY=sk-abc123
```

### Q: LLM 调用了错误的工具
```
A: 正常现象，可尝试:
   1. 升级模型: --model qwen-max
   2. 在 tools 的 docstring 中写更详细的描述和示例
   3. 降低 temperature（config.py 中设为 0.1）
```

### Q: Windows 终端显示乱码
```
A: 在终端输入 chcp 65001 切换到 UTF-8，或使用 Windows Terminal
```

### Q: 如何添加语音控制
```
A: 集成 ASR（语音转文字）+ TTS（文字转语音）库即可
   常用方案: Azure Speech / 阿里云语音服务 / 本地 whisper
```

---

## 10. 学习资源

| 资源 | 链接 |
|------|------|
| LangGraph 官方文档 | https://langchain-ai.github.io/langgraph/ |
| LangChain 文档 | https://python.langchain.com/ |
| MCP 协议规范 | https://modelcontextprotocol.io/ |
| 阿里百炼文档 | https://help.aliyun.com/product/261167.html |
| Pydantic 文档 | https://docs.pydantic.dev/ |
| Typer 文档 | https://typer.tiangolo.com/ |

---

> 🎉 **恭喜完成！**  
> 这个项目是一个标准的现代 AI Agent 参考实现，涵盖了:  
> 状态管理、工具编排、MCP 协议、持久化记忆、中间件模式。  
> 随着你的深入，这些概念会变得越来越清晰。
