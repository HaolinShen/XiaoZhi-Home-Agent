# 智能家居家电互联智能体 — 开发教程

> **适用版本**: v0.1.0  
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

一个基于 **LangGraph + MCP (Model Context Protocol)** 的智能家居 AI Agent。你可以用自然语言控制智能设备，支持多轮对话记忆、场景模式、以及标准的 MCP 工具暴露。

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

```
┌──────────────────────────────────────────────────────────┐
│                    LangGraph Agent                        │
│                                                          │
│  用户输入                                                  │
│     │                                                    │
│     ▼                                                    │
│  ┌─────────┐    有 tool_calls?     ┌──────────┐         │
│  │ Agent   │ ───────────────────→  │  Tools   │         │
│  │ (LLM)   │                       │ (执行工具) │         │
│  │         │ ←───────────────────  │          │         │
│  └────┬────┘   返回执行结果          └──────────┘         │
│       │                                                  │
│       │ 没有 tool_calls                                   │
│       ▼                                                  │
│   最终回复 → 返回给用户                                     │
│                                                          │
│  记忆层: MemorySaver / SqliteSaver (跨轮次状态保持)         │
└──────────────────────────────────────────────────────────┘

同时，工具通过 MCP Server 暴露给外部 AI 客户端（如 Claude Desktop）
```

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
│   │   └── scenes.py         # 场景模式工具 (activate_scene, list_scenes)
│   │
│   ├── agent/                # Agent 层
│   │   ├── __init__.py
│   │   ├── state.py          # Agent 状态定义 (AgentState)
│   │   ├── prompts.py        # 系统提示词模板
│   │   └── graph.py          # LangGraph 工作流图 (build_graph)
│   │
│   ├── mcp/                  # MCP 层
│   │   ├── __init__.py
│   │   ├── server.py         # MCP 服务器 (暴露工具给外部 AI)
│   │   └── client.py         # MCP 客户端 (消费外部 MCP 服务)
│   │
│   ├── memory/               # 记忆层
│   │   ├── __init__.py
│   │   └── store.py          # 检查点存储 (MemorySaver / SqliteSaver)
│   │
│   ├── middleware/           # 中间件层
│   │   ├── __init__.py
│   │   └── interceptors.py   # 日志拦截器 + 重试拦截器
│   │
│   └── main.py               # ★ CLI 主入口 (typer + rich)
│
├── tests/                    # 测试
│   └── __init__.py
│
├── docs/                     # 文档
│   └── tutorial.md           # 本教程
│
└── data/                     # 运行时数据（自动创建）
    └── checkpoints.db        # SQLite 对话记忆
```

### 架构分层

```
┌─────────────────────────────────────────────┐
│  CLI / MCP Server  (表示层)                  │
├─────────────────────────────────────────────┤
│  Agent Graph        (编排层) ← LangGraph     │
│  Tools              (工具层) ← @tool         │
├─────────────────────────────────────────────┤
│  Device Registry    (领域层)                 │
│  Models             (数据层) ← Pydantic      │
├─────────────────────────────────────────────┤
│  Config             (配置层)                 │
│  Memory / Middleware (基础设施)              │
└─────────────────────────────────────────────┘
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

```
用户: "打开客厅灯"
  → LLM 分析: 需要调用 control_light
  → LLM 输出: tool_calls=[{"name":"control_light","args":{"device_name":"客厅","action":"on"}}]
  → ToolNode 执行: control_light("客厅", "on")
  → 工具返回: "✅ 客厅灯已打开"
  → 返回给 LLM → LLM 生成自然语言回复
```

### 4.5 Agent 图 (`agent/graph.py`)

LangGraph 的 `StateGraph` 定义了 Agent 的思考-行动循环：

```python
from src.agent import build_graph

# 构建图
graph = build_graph(registry, settings)

# 运行
result = graph.invoke(
    {"messages": [HumanMessage(content="打开客厅灯")]},
    config={"configurable": {"thread_id": "user-001"}}
)

# 获取回复
print(result["messages"][-1].content)
```

**图结构**:

```python
workflow = StateGraph(AgentState)

# 节点
workflow.add_node("agent", agent_node)   # LLM 推理
workflow.add_node("tools", ToolNode())   # 工具执行

# 边
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", router, {
    "tools": "tools",    # 有 tool_calls → 执行
    "__end__": END,      # 无 tool_calls → 结束
})
workflow.add_edge("tools", "agent")      # 执行完 → 回到 LLM

# 编译（带记忆）
graph = workflow.compile(checkpointer=MemorySaver())
```

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

🤖 小智: 欢迎回家！我帮你激活回家模式：
✅ 已激活「🏠 回家模式」
  · 客厅灯已打开（亮度 80%，暖白）
  · 客厅空调已开启（制冷 26°C）
  · 客厅窗帘已完全打开
🏠 欢迎回家！
```

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

#### 6.4.1 为什么 Agent 需要记忆？

大模型本身是**无状态**的 —— 每次调用 LLM，它都只看到你这次传入的消息，不记得之前说过什么。多轮对话所以能"记住"上下文，全靠我们在每一轮把历史消息重新喂给 LLM。

LangGraph 通过 **Checkpoint（检查点）** 机制解决这个问题：

- 每次节点执行完毕后，自动把图的状态（主要是 `messages` 消息列表）**存档**下来
- 调用 `graph.invoke(..., config)` 时带上一个 **`thread_id`**（线程 ID），LangGraph 就知道该从哪份存档里恢复上下文
- 整个 Agent 被设计成"有状态的"：同一 `thread_id` 下连续调用，历史消息自动累加

```
thread_id="user-abc123"  ──►  第1轮: [用户: 打开客厅灯]
                              ──►  第2轮: [用户: 打开客厅灯, AI: 已打开, 用户: 空调呢?]  ← 带上了历史!
```

这就是我们项目里 `graph.compile(checkpointer=...)` 传 `checkpointer` 参数的原因。

#### 6.4.2 两种存储模式对比

项目支持两种检查点存储，通过 `.env` 一键切换：

| 对比项 | 🧠 内存模式 (MemorySaver) | 💾 SQLite 持久化 (SqliteSaver) |
|--------|--------------------------|-------------------------------|
| 存储位置 | 程序内存 | `data/checkpoints.db` 文件 |
| 重启后记忆 | ❌ 丢失 | ✅ 保留 |
| 依赖 | 内置（LangGraph 自带） | 需额外安装 `langgraph-checkpoint-sqlite` |
| 性能 | 最快 | 略慢（磁盘 IO，可忽略） |
| 适用场景 | 开发调试、单元测试 | 生产部署、长期使用 |

#### 6.4.3 代码是如何决定用哪种的？

核心逻辑在 `src/agent/graph.py` 的 `_build_checkpointer()`：

```python
def _build_checkpointer(settings: Settings):
    db_path = settings.memory.db_path   # 来自 CHECKPOINT_DB_PATH 环境变量

    # 优先级: 有路径 + 装了 SQLite 支持包 → 用 SqliteSaver
    if db_path and _HAS_SQLITE:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        return SqliteSaver(conn)        # 💾 持久化

    # 否则回退到内存模式
    return MemorySaver()                # 🧠 临时
```

**三个关键细节：**

1. **`_HAS_SQLITE` 是环境探测**：`graph.py` 用 `try: from langgraph.checkpoint.sqlite import SqliteSaver` 探测，**装了这个包才为 `True`**。没装的话，即使配了路径也会静默回退到内存模式。
2. **目录自动创建**：`sqlite3.connect()` 前会 `os.makedirs(db_dir, exist_ok=True)`，`data/` 目录不存在时自动创建。
3. **单例配置**：`settings.memory` 来自 `src/config.py` 的 `MemoryConfig`，前缀是 `CHECKPOINT_`。

#### 6.4.4 启用 SQLite 持久化（完整步骤）

**第 1 步：安装 SQLite 支持包**

```bash
pip install langgraph-checkpoint-sqlite
```

> ⚠️ 这一步最容易漏。`pyproject.toml` 里没有把它列为硬依赖（保持内存模式开箱即用），需要手动安装。

**第 2 步：在 `.env` 中配置路径**

```ini
# .env
CHECKPOINT_DB_PATH=data/checkpoints.db
```

- 路径**留空**（`CHECKPOINT_DB_PATH=`）→ 回到内存模式
- 路径可以是任意位置，如 `C:/data/home/checkpoints.db`

**第 3 步：重启 Agent**

```bash
python -m src.main
```

#### 6.4.5 如何验证记忆真的生效了？

**方式一：检查数据库文件**

```bash
# 正常对话几轮后，data/ 目录下应出现数据库文件
ls -la data/
# 输出: checkpoints.db
```

**方式二：重启后验证**

1. 启动 Agent，输入 `把客厅灯调暗到 30%`，看到 Agent 执行成功
2. 按 `/quit` 退出，重新 `python -m src.main`
3. 输入 `/status` 或 `现在家里什么状态？`
4. **内存模式**：灯恢复默认亮度 80%（状态重置）
5. **SQLite 模式**：灯仍是 30%（状态保留）✅

**方式三：用 Python 直接查看数据库**

```python
import sqlite3
conn = sqlite3.connect("data/checkpoints.db")
# 查看有哪些线程（对话）存档
print(conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall())
```

#### 6.4.6 thread_id：对话如何被隔离

`src/main.py` 里每次启动都会生成一个**新的随机 thread_id**：

```python
thread_id = f"user-{uuid.uuid4().hex[:8]}"   # 如 user-a3f9c2d1
config = {"configurable": {"thread_id": thread_id}}
```

- **每次启动 = 新 thread_id = 全新对话**，从零开始
- 终端里的 **`/reset`** 命令做的事就是生成一个新 thread_id，从而清空当前对话记忆（旧存档仍在数据库里，只是不再被引用）
- 想让多轮调用共享上下文，手动固定同一个 thread_id 即可：

```python
config = {"configurable": {"thread_id": "my-fixed-session"}}
graph.invoke({"messages": [HumanMessage("打开客厅灯")]}, config)
graph.invoke({"messages": [HumanMessage("空调呢？")]}, config)   # 记得上一轮!
```

#### 6.4.7 长期记忆（规划中）

目前的记忆是**短期记忆**（对话上下文）。项目还为未来预留了**长期记忆**（用户偏好学习），配置项已就位：

```ini
# .env
ENABLE_LONG_TERM_MEMORY=true
```

规划中的能力：记住用户的习惯（"喜欢暖光"、"空调通常设 25°C"），跨会话、跨 `thread_id` 生效，用于个性化推荐。当前版本尚未实现，代码见 `src/memory/store.py` 的注释说明。

#### 6.4.8 常见问题

**Q: 配了 `CHECKPOINT_DB_PATH` 但重启后记忆还是丢？**
A: 99% 是没装 `langgraph-checkpoint-sqlite`，代码静默回退到了内存模式。执行第 6.4.4 节的第 1 步。

**Q: `checkpoints.db` 会被提交到 Git 吗？**
A: 不会。`.gitignore` 中 `data/` 目录已被排除，数据库文件属于本地运行数据。

**Q: 数据库会不会无限膨胀？**
A: 检查点只存消息和状态，量级很小（KB 级）。如需清理，删除 `data/checkpoints.db` 重启即可。

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
