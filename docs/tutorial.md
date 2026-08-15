# 智能家居家电互联智能体 — 开发教程

> **适用版本**: 阶段十二（已包含 Agentic RAG 与轨迹评测）
> **适用人群**: 想要学习现代 AI Agent 开发的 Python 开发者  
> **前置知识**: Python 基础（类、装饰器、类型注解）、理解 LLM 的基本概念  
> **完成时间**: 核心章节阅读约 2 小时，完整实践建议分阶段完成

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

一个基于 **LangGraph + MCP (Model Context Protocol)** 的智能家居 AI Agent。你可以用自然语言查询和控制设备，并学习状态图编排、工具调用、会话检查点、上下文压缩、长期记忆、Human-in-the-loop、规划执行、结构化路由、多智能体协作以及 Agentic RAG。

当前项目已经具备：

- 灯光、空调、电视、窗帘、加湿器等模拟设备控制；
- 温湿度与人体存在传感器（只读），读数会跟随执行器状态变化；
- 回家、离家、睡眠、观影、起床等多设备场景；
- LangGraph ReAct 工具调用循环；
- 基于 `thread_id` 的短期会话记忆；
- 结构化长期记忆、候选确认、混合检索和版本追踪；
- 场景操作执行前的 `interrupt` 人工确认；
- 使用 `Command(resume=...)` 从原检查点批准或拒绝操作；
- Planner–Executor–Verifier 规划、执行、验证、重试与重新规划；
- 结构化意图路由、设备查询子图和动态并行；
- Supervisor 按职责向设备、场景、记忆和聊天 Agent 委派；
- 显式记忆推理、Checkpoint 时间旅行和自定义进度事件；
- 基于本地设备文档的 Agentic RAG、来源引用与轨迹评测；
- MCP Server 工具暴露和外部 MCP Client 接入能力。

### 1.2 技术栈

| 技术 | 版本 | 作用 |
|------|------|------|
| **LangGraph** | ≥1.0 | Agent 工作流编排（状态图、节点、条件路由） |
| **LangChain** | ≥1.0 | LLM 调用封装、工具定义、消息管理 |
| **MCP (Model Context Protocol)** | ≥1.0 | 标准化工具暴露/消费协议 |
| **Pydantic v2** | ≥2.0 | 类型安全的数据模型 & 配置管理 |
| **Typer + Rich** | - | 现代化 CLI 终端界面 |
| **Loguru** | - | 结构化日志 |
| **阿里百炼** | - | 大模型 API（兼容 OpenAI 接口） |

### 1.3 Agent 工作流

```text
用户输入
  ↓
sync_context             同步可信位置，抽取候选并检索长期记忆
  ↓
memory_reasoner          判断记忆是否适用、冲突或被临时指令覆盖
  ↓
task_router
  ├── 信息不足 ───────────────────────────────→ clarification → END
  ├── 多设备状态查询 ────────────────→ device_query_subgraph → END
  ├── 设备知识或故障问题 ─────────────────────→ knowledge_rag → END
  ├── 自定义多步骤目标
  │      ↓
  │   planner → plan_approval（interrupt）
  │                 ├── rejected ─────────────────────→ finalize
  │                 └── approved
  │                        ↓
  │                     executor → verifier
  │                                   ├── 下一步 / retry → executor
  │                                   ├── replan         → planner
  │                                   └── 完成 / 失败     → finalize
  │
  └── 普通请求 / 预定义场景 / 专用 Agent
         ↓
      compact_context → agent
         ├── 普通工具调用 ────────────────────────────→ tools ─┐
         ├── 高风险或批量调用 → approval → tools / reject_tools ┤
         └── 无工具调用 → supervisor_finalize / END              │
                                                                  └→ compact_context
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

# 根据 pyproject.toml 安装项目及测试依赖
pip install -e ".[dev]"
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
# 编辑 .env，至少填写 LLM_API_KEY；模型和服务地址可按需调整
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
│   ├── models.py             # Pydantic 设备数据模型（执行器 + 只读传感器）
│   │
│   ├── devices/              # 设备层
│   │   ├── __init__.py
│   │   ├── base.py           # 抽象后端接口 + 设备注册中心
│   │   └── simulator.py      # 内存模拟器后端（含确定性环境推演）
│   │
│   ├── tools/                # 工具层
│   │   ├── __init__.py       # 工具注册 & 导出
│   │   ├── devices.py        # 设备控制工具 (control_light/ac/tv/curtain/humidifier) + read_sensor
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
│   │   ├── routing.py        # 结构化意图识别与确定性回退路由
│   │   ├── parallel.py       # 设备查询子图、Send 动态并行与聚合
│   │   ├── multi_agent.py    # Supervisor 角色和专用工具集
│   │   ├── reasoning.py      # 结构化长期记忆适用性判断
│   │   ├── time_travel.py    # Checkpoint 历史查看与分支恢复
│   │   ├── observability.py  # 自定义流式进度事件
│   │   └── graph.py          # 主 LangGraph 工作流图 (build_graph)
│   │
│   ├── mcp/                  # MCP 层
│   │   ├── __init__.py
│   │   ├── server.py         # MCP 服务器 (暴露工具给外部 AI)
│   │   ├── client.py         # MCP 客户端 (消费外部 MCP 服务)
│   │   └── weather_server.py  # 彩云天气 MCP（需 CAIYUN_WEATHER_TOKEN）
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
│   ├── knowledge/            # 设备知识与 Agentic RAG 子图
│   │   ├── base.py           # Markdown 文档加载、型号过滤和词法检索
│   │   └── rag.py            # 识别、检索、改写、回答与拒答流程
│   │
│   ├── evaluation/           # Agent 轨迹评测
│   │   └── trajectory.py     # 路由、状态、来源和拒答指标
│   │
│   ├── middleware/           # 中间件层
│   │   ├── __init__.py
│   │   └── interceptors.py   # 日志拦截器 + 重试拦截器
│   │
│   ├── progress_view.py      # 进度事件 → 终端渲染（Planner/Executor/Verifier 过程）
│   └── main.py               # ★ CLI 主入口 (typer + rich)
│
├── tests/                    # 阶段一至阶段十二 + 设备扩展自动化测试
│   ├── test_phase_one.py
│   ├── test_phase_two.py
│   ├── test_phase_three.py
│   ├── test_phase_four.py
│   ├── test_phase_five.py
│   ├── test_phase_six.py     # Human-in-the-loop 测试
│   ├── test_phase_seven.py   # Planner–Executor–Verifier 测试
│   ├── test_phase_eight.py   # 结构化意图路由测试
│   ├── test_phase_nine.py    # 子图与动态并行测试
│   ├── test_phase_ten.py     # Supervisor 多智能体测试
│   ├── test_phase_eleven.py  # 记忆推理、时间旅行与事件测试
│   ├── test_phase_twelve.py  # Agentic RAG 与轨迹评测测试
│   ├── test_weather_mcp.py   # 天气 MCP 发现、调用与结果格式测试
│   ├── test_humidifier.py    # 加湿器模型、工具、Planner 与状态测试
│   ├── test_sensors.py       # 传感器模型、环境推演、read_sensor 与只读约束测试
│   └── test_planning_progress.py  # 规划进度事件与终端渲染测试
│
├── docs/                     # 文档
│   ├── tutorial.md           # 本教程
│   ├── iterations/           # 各阶段设计与实现说明
│   └── knowledge/            # 型号目录与本地设备知识文档
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
│  LangGraph 主图 / 子图 / Supervisor    编排层           │
│  路由 / 规划 / 审批 / 并行 / RAG / 工具循环            │
├─────────────────────────────────────────────────────┤
│  @tool 设备、场景和记忆工具            工具层           │
├─────────────────────────────────────────────────────┤
│  DeviceRegistry / SpaceDirectory      领域与上下文层    │
├─────────────────────────────────────────────────────┤
│  Checkpoint / Long-term Memory / RAG  状态与知识层      │
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
print(settings.model)           # "qwen-plus"
print(settings.mcp_server.port) # 8765
print(settings.rag.top_k)       # 3
```

**特性**:
- 自动从 `.env` 加载，支持系统环境变量覆盖
- 字段级验证（API Key 不能为空/占位符）
- 嵌套配置对象（MCP、Memory、Planning、Routing、Multi-agent、RAG）
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

如果新增的是**只读传感器**，第 4 步不一样：不写 `control_xxx`，而是在
`read_sensor` 的类型映射里加一项；同时要把新类型加进 `SENSOR_DEVICE_TYPES`，
这样场景批量开关和规划器的工具白名单会自动把它排除在外。判断标准很简单——
这台设备的状态是我们命令出来的，还是环境本来就是那样。前者是执行器，后者是传感器。

### 4.3 设备注册中心 (`devices/base.py`)

**Registry Pattern** — 设备查找和操作的中心枢纽：

#### 4.3.1 先看整体：五层分别解决什么问题

设备模块可以按“模型 → 后端 → 注册中心 → 工具 → Agent”的方向理解。每层只解决一种问题：

```
models.py
  └─ BaseDevice
       ├─ LightDevice          ┐
       ├─ ACDevice             │
       ├─ TVDevice             ├─ 执行器：可读可写
       ├─ CurtainDevice        │
       ├─ HumidifierDevice     ┘
       ├─ TempHumiditySensor   ┐
       └─ PresenceSensor       ┘─ 传感器：只读
              │  (AnyDevice 联合类型)
              ▼
DeviceBackend (抽象接口)
  └─ SimulatorBackend (内存字典实现，创建 13 个默认设备)
              │
              ▼
DeviceRegistry (查找、筛选、更新、状态摘要、环境推演)
              │
              ▼
tools/devices.py 的 @tool 函数
  └─ control_light / control_ac / control_tv / control_curtain / control_humidifier
     read_sensor / get_device_status
              │
              ▼
LangGraph Agent / ToolNode（决定调用哪个工具并组织最终回复）
```

注意工具名的不对称：五个执行器各有 `control_xxx`，两个传感器只有一个 `read_sensor`。
这不是偷懒，而是让 LLM 从工具名就知道传感器改不了状态——它没有
`control_temp_humidity_sensor` 可用，也就不会试图“打开温湿度传感器”。

可以先用一句话记住各层：

| 层 | 回答的问题 | 不负责什么 |
| --- | --- | --- |
| 设备模型 | “一台设备有哪些字段，字段是否合法？” | 不保存全部设备，不连接硬件 |
| Backend | “设备状态实际保存在哪里，怎样读取和写入？” | 不理解用户自然语言 |
| Registry | “上层怎样用统一入口查找和操作设备？” | 不决定用户想调用哪个工具 |
| Tool | “一个明确动作怎样转换为设备操作？” | 不负责任务路由和多轮推理 |
| Agent / LangGraph | “用户想做什么，应该调用哪个工具？” | 不直接操作 `_devices` 或真实硬件 |

这种拆分的核心目的，是避免把“理解用户”“寻找设备”“保存状态”“连接硬件”全部写在一个函数里。

#### 4.3.2 第一层：设备模型是数据契约

`BaseDevice` 和各个子类位于 `src/models.py`。模型描述的是“一台设备现在长什么样”，例如所有设备都有：

```python
device_id: str       # 程序内部稳定 ID，例如 living_room_humidifier
name: str            # 用户可读名称，例如 客厅加湿器
device_type: DeviceType
power: bool
location: str
```

具体设备再增加自己的字段：

```text
LightDevice        → brightness、color
ACDevice           → temperature、mode、fan_speed
TVDevice           → volume、muted、channel
CurtainDevice      → position
HumidifierDevice   → target_humidity、mist_level、water_level
TempHumiditySensor → temperature、humidity、battery
PresenceSensor     → occupied、last_motion_at、timeout_minutes、battery
```

传感器这两行里的字段名值得多看一眼：它们记录的是**实测值**，
而执行器的 `temperature` 记录的是**目标值**。`ACDevice.temperature=24` 表示
“我要 24 度”，`TempHumiditySensor.temperature=27.0` 表示“现在实际 27 度”。
两者不一致是正常状态，正是这个差值让空调有事可做、让验证器有东西可验。

`PresenceSensor` 的 `occupied` 不是直接写入的，而是由 `last_motion_at` 加
`timeout_minutes` 推算出来的。原因是真实人体传感器只能感知“活动”，感知不到
静止不动的人，业界通用做法就是检测到活动置为有人、超过 N 分钟无新活动回落为无人。
把规则写进模拟器而不是随机生成，测试就能靠写一个时间戳精确控制传感器行为。

这里最重要的是“数据契约”四个字。Pydantic 会在创建或重新验证对象时保证字段满足约束，例如加湿器目标湿度只能是 30–80%。`to_status_text()` 则负责把结构化状态转换为适合用户或 LLM 阅读的文字。

设备模型本身不保存家庭中的全部设备，也不知道 Home Assistant、MQTT 或米家平台。下面的代码只是在内存中创建一个普通 Python 对象：

```python
device = HumidifierDevice(
    device_id="living_room_humidifier",
    name="客厅加湿器",
    location="客厅",
    target_humidity=60,
    water_level=100,
)
```

`AnyDevice` 是联合类型：

```python
AnyDevice = Union[
    LightDevice,
    ACDevice,
    TVDevice,
    CurtainDevice,
    HumidifierDevice,
    TempHumiditySensor,
    PresenceSensor,
]
```

它主要帮助类型检查器表达“这个位置可以存放任意一种受支持设备”，并不是一个可以直接实例化的新设备类。

同一个文件里还有一个 frozenset：

```python
SENSOR_DEVICE_TYPES = frozenset({
    DeviceType.TEMP_HUMIDITY_SENSOR,
    DeviceType.PRESENCE_SENSOR,
})
```

“这台设备不能被控制”这个判断在提示词生成、场景执行等多处都要用到。
集中定义一次，以后新增传感器只改这一处，不必去每个文件里补一遍类型列表。

#### 4.3.3 第二层：Backend 决定状态保存在哪里

`DeviceBackend` 是抽象接口，也可以理解为设备存储和控制协议。它规定所有后端都必须提供相同的方法：

```python
class DeviceBackend(ABC):
    def get(self, device_id: str) -> Optional[AnyDevice]: ...
    def get_all(self) -> dict[str, AnyDevice]: ...
    def get_by_type(self, device_type: DeviceType) -> dict[str, AnyDevice]: ...
    def update(self, device_id: str, **kwargs) -> bool: ...
    def get_status_summary(self) -> str: ...

    # 注意：这一个不是 @abstractmethod
    def tick_environment(self) -> None:
        return None
```

前五个方法是 `@abstractmethod`，任何后端都必须实现。最后一个 `tick_environment()`
是带默认空实现的普通方法，原因在“新增后端要不要被迫改代码”上：模拟器需要按执行器
状态推算传感器读数，所以会覆盖它；而真实后端（Home Assistant / MQTT）的传感器由硬件
自行上报，根本不需要推演，直接继承空实现即可。如果把它写成 `@abstractmethod`，
以后每个新后端都要写一个毫无意义的 `pass`。

抽象接口只说明“必须能做什么”，不规定“具体怎样做”。当前的 `SimulatorBackend` 使用：

```python
self._devices: dict[str, AnyDevice]
```

保存十三台模拟设备（九个执行器 + 四个传感器），因此：

- `get()` 是从字典按 ID 读取；
- `get_by_type()` 是按 `device_type` 过滤；
- `update()` 是合并旧状态，重新经过 Pydantic 验证，再替换字典中的对象；
- `get_status_summary()` 是按灯光、空调、电视、窗帘、加湿器、温湿度和人体存在分组生成状态文本；
- `tick_environment()` 按同房间执行器的状态推演一次传感器读数。

模拟器只存在于当前 Python 进程中。程序退出后状态会恢复默认值。如果以后接入 Home Assistant，可以实现：

```python
class HomeAssistantBackend(DeviceBackend):
    def get(self, device_id):
        # 调用 Home Assistant API
        ...

    def update(self, device_id, **kwargs):
        # 向 Home Assistant 发送真实控制请求
        ...
```

只要新后端遵守同一组方法，上层代码就不需要关心状态来自字典、数据库还是实际硬件。

#### 4.3.4 第三层：Registry 是稳定的业务入口

`DeviceRegistry` 容易被误解成“设备数据库”，但它自己并不保存设备字典。它只持有一个 Backend：

```python
class DeviceRegistry:
    def __init__(self, backend: DeviceBackend):
        self._backend = backend
```

Registry 的作用类似门面（Facade）：把底层 Backend 的能力整理成上层更容易使用的接口。

它提供两类查找方式：

```python
# 精确 ID 查找：代码已经知道稳定 ID
registry.get("living_room_humidifier")

# 面向用户表达的查找：名称可能不完全一致
registry.find("客厅的加湿器", DeviceType.HUMIDIFIER)
```

`find()` 当前会依次尝试：

1. 中文设备名精确匹配；
2. 名称字符匹配；
3. “灯”“空调”“加湿器”“温湿度”“人体感应”等类型关键词匹配；
4. 如果同类型设备存在多个且无法确定房间，返回 `None`，让 Agent 澄清而不是随便选择。

策略 2 比看上去更宽松，值得留意：它检查的是“输入里的每个字都出现在设备名中”，
所以 `find("湿度", TEMP_HUMIDITY_SENSOR)` 会命中`客厅温湿度传感器`——“湿”和“度”
都在里面。这不算错，但也说明它不适合承担指标类问题的解析。
`read_sensor` 因此完全不走 `find()`，改用 `get_by_type` + `location` 筛选：
“客厅湿度多少”里的“湿度”问的是一个指标，不是某台设备的名字。

Registry 的更新方法本身不修改字典，而是继续委托 Backend：

```python
def update(self, device_id: str, **kwargs) -> bool:
    return self._backend.update(device_id, **kwargs)
```

这样做看似多了一层，实际上提供了稳定边界。以后可以在 Registry 中统一增加权限检查、设备能力判断、审计日志或名称解析，而不需要修改每个工具。

Registry 还额外承担两件和传感器相关的事。第一是把环境推演转发给 Backend：

```python
def tick_environment(self) -> None:
    self._backend.tick_environment()
```

第二是在生成设备清单提示词时把执行器和传感器分成两组：

```text
可控制的设备列表:
  · 客厅灯（ID: living_room_light，类型: light）
  · 卧室灯（ID: bedroom_light，类型: light）
  ...
  · 客厅加湿器（ID: living_room_humidifier，类型: humidifier）
只读传感器列表（只能读取，不能控制）:
  · 客厅温湿度传感器（ID: living_room_th_sensor，类型: temp_humidity_sensor）
  · 卧室温湿度传感器（ID: bedroom_th_sensor，类型: temp_humidity_sensor）
  · 客厅人体传感器（ID: living_room_presence，类型: presence_sensor）
  · 玄关人体传感器（ID: entryway_presence，类型: presence_sensor）
```

分组之前这些设备混在一张清单里，规划器会理所当然地给温湿度传感器排一步“打开”。
清单里的一行字，比在提示词里反复叮嘱“不要控制传感器”有效得多——因为前者是模型
读到的事实，后者只是一句要求。

#### 4.3.4b 传感器读数为什么必须显式推演

传感器接进来以后，模拟器面临一个新问题：读数什么时候更新？

最省事的写法是在 `get()` / `get_all()` 里顺手推演一次，读的时候永远是最新值。
这个项目**没有**这样做，因为 `get()` 和 `get_all()` 被场景执行、计划验证等路径
大量调用。如果读取本身会改状态，“读一下看看”就变成了“读一下顺便改了”：
验证器多读一次，环境值就多走一步，读到的数字取决于代码里调了几次 `get()`。
这类 bug 极难复现，也极难解释。

所以 `tick_environment()` 是一个显式的公开方法，只有三个“读环境”入口会调用它：

| 入口 | 为什么它该推演 |
| --- | --- |
| `read_sensor` | 用户明确在问环境读数 |
| `get_device_status` | 一次显式的“看一眼家里现在什么状态” |
| 并行查询子图的 `dispatch` 节点 | 同上，且放在扇出**之前**——放进每个分支就会让并行度直接改变读数 |

设备控制、场景激活、计划验证一律不调用它。这条规则有测试守着：反复调用
`get_status_summary()`、`get()`、`get_all()` 之后，湿度必须仍然停在初始的 42%。

推演逻辑本身是确定性的，没有任何随机数：

```python
_TEMP_STEP = 0.5         # °C，空调把室温朝目标拉近的速度
_HUMIDITY_STEP = 2       # %，加湿器把湿度朝目标拉近的速度
_DRY_DRIFT = 1           # %，没有加湿器工作时湿度自然回落的速度
_BASELINE_HUMIDITY = 45  # %，无人工干预时房间的湿度基线
```

制冷把温度朝目标拉低、制热拉高、都不会越过目标；加湿器工作时湿度朝目标推进，
不工作时朝 45% 基线回落。固定步长的代价是不够“真实”，换来的是测试可以断言
精确值——`tick` 二十次之后湿度必须正好等于 60，而不是“大概接近 60”。

几个边界情况也是刻意处理的：水箱空了不加湿、`power=False` 的传感器完全跳过、
`last_motion_at` 格式非法时保留原状并打一条警告而不是抛异常。传感器离线时
`to_status_text()` 只报“⚠️ 离线”，不报最后那个可能已经过期很久的数值。

#### 4.3.5 第四层：Tool 把明确动作翻译成 Registry 操作

工具层位于 `src/tools/devices.py`。工具不是存储层，也不是设备对象；它是 LLM 可以调用的业务函数。例如：

```python
@tool
def control_humidifier(
    device_name: str,
    action: str,
    target_humidity: int = 60,
    mist_level: str = "auto",
) -> str:
    registry = _get_registry()
    device = registry.find(device_name, DeviceType.HUMIDIFIER)
    if device is None:
        return "❌ 找不到指定的加湿器设备"

    if action == "set_humidity":
        registry.update(
            device.device_id,
            target_humidity=target_humidity,
            power=True,
        )
        return f"✅ {device.name}目标湿度已设置"
```

工具主要负责：

- 定义 LLM 可见的参数 Schema；
- 将 `device_name` 定位为稳定 `device_id`；
- 校验 `action` 是否支持；
- 做必要的业务保护，例如水箱为空时禁止开启加湿器；
- 调用 Registry；
- 返回可以成为 `ToolMessage` 的执行结果。

工具不应该直接写 `_devices[device_id]`，否则它会与 `SimulatorBackend` 强绑定，未来换成真实平台时所有工具都要重写。

传感器的工具长得明显不一样。它不叫 `control_temp_humidity_sensor`，而是一个统一的
`read_sensor`，两种传感器共用：

```python
@tool
def read_sensor(sensor_type: str, location: str = "") -> str:
    """读取环境传感器的当前数值。控制设备前先用它了解实际情况。"""
    registry = _get_registry()
    device_type = {
        "temp_humidity": DeviceType.TEMP_HUMIDITY_SENSOR,
        "presence": DeviceType.PRESENCE_SENSOR,
    }.get(sensor_type)
    if device_type is None:
        return f"❌ 不支持的传感器类型「{sensor_type}」。支持: temp_humidity, presence"

    registry.tick_environment()          # 读之前推演一次
    sensors = registry.get_by_type(device_type)
    ...
```

三点和 `control_xxx` 不同，都是有意的：

- **没有 `action` 参数**：读取只有一个动作，多给一个参数就是多给一次出错机会；
- **不走 `find()`**：按 `get_by_type` + `location` 筛选。“客厅湿度多少”里的“湿度”
  问的是指标而不是某台设备，用名称模糊匹配去解析它是错的方向；
- **参数错误也返回可用选项**：类型写成 `"co2"` 会得到一条列出两种合法类型的
  ❌ 文本，房间写成“书房”会得到已安装传感器的房间列表。工具的错误返回是模型
  的下一轮输入，把可选项写清楚，模型能自己纠正；只回一句“不支持”，它只能猜。

docstring 里还专门写了“什么时候应该主动调用”：用户说“有点干”“有点热”这类主观
感受时先读数、执行离家模式前先确认没人。这类提示放在工具的 docstring 里比放在
系统提示词里更有效——它出现在模型正在挑工具的那一刻。

#### 4.3.6 第五层：Agent 决定使用哪个工具

Agent 并不会直接调用 `registry.update()`。构图时，工具通过 `llm.bind_tools(...)` 暴露给模型；模型根据用户请求生成结构化 `tool_calls`，再由 LangGraph 的 `ToolNode` 执行对应工具。

这意味着各层的输入逐渐从模糊变得明确：

```text
用户自然语言：“把客厅加湿器湿度设为 65%”
        ↓ Agent / LLM
结构化工具调用：control_humidifier(
    device_name="客厅加湿器",
    action="set_humidity",
    target_humidity=65,
)
        ↓ Tool
稳定设备操作：registry.update(
    "living_room_humidifier",
    target_humidity=65,
    power=True,
)
        ↓ Backend
保存或发送真实设备状态
```

#### 4.3.7 完整示例：“打开客厅加湿器”发生了什么

一次控制请求会经过下面的步骤：

1. 用户消息进入 LangGraph；
2. `task_router` 将它识别为 `device_control`；
3. Device Agent 得到设备控制工具列表；
4. LLM 产生 `control_humidifier(device_name="客厅加湿器", action="on")`；
5. `ToolNode` 调用 `control_humidifier`；
6. 工具通过 `_get_registry()` 得到启动时注入的 Registry；
7. `registry.find()` 委托 Backend 获取加湿器候选并完成名称匹配；
8. 工具检查水箱是否为空；
9. `registry.update()` 将更新委托给 `SimulatorBackend`；
10. Backend 重新验证 `HumidifierDevice` 并替换内存状态；
11. 工具返回“加湿器已开启”；
12. 结果作为 `ToolMessage` 回到 Agent，Agent 再生成最终自然语言回复。

对应调用链可以压缩为：

运行时调用链如下：

```
用户请求 → Agent 决定工具 → control_* → DeviceRegistry.find
         → DeviceBackend.get_by_type / update
         → SimulatorBackend 修改 _devices
         → 工具返回文本 → Agent 生成最终回复
```

查询请求也遵循同一边界，只是不修改状态：

```text
“查看所有设备状态”
  → get_device_status
  → registry.tick_environment()        ← 显式推演一次传感器读数
  → registry.get_status_summary()
  → backend.get_status_summary()
  → 每台设备.to_status_text()
  → 返回分组后的状态报告
```

还有一条闭环值得单独看，它是传感器存在的主要理由：

```text
“有点干”
  → read_sensor(sensor_type="temp_humidity", location="客厅")   # 先看数据
  → “温度 27.0°C，湿度 42%”
  → control_humidifier(device_name="客厅加湿器", action="set_humidity",
                       target_humidity=60)                      # 再动手
  → 下一次 read_sensor → 湿度 44% → 46% → …                     # 环境确实在变
```

没有传感器时，Agent 只能读到自己刚写下的目标值——“我设了 60，所以是 60”，
这种自证式的验证发现不了任何问题。有了传感器，验证读的是环境，
这是整条链路里唯一一处真正来自外部的反馈。

#### 4.3.8 启动时为什么要注入 Registry

启动时的组装代码位于 `main.py`：

```python
backend = SimulatorBackend()       # 可替换成真实 IoT 后端
registry = DeviceRegistry(backend)
set_tools_registry(registry)       # 所有设备工具共享此实例
```

这叫依赖注入：工具没有在模块导入时偷偷创建另一个模拟器，而是由启动入口明确告诉工具“本次运行使用这个 Registry”。因此 CLI、测试和 MCP Server 都可以传入不同实例，并控制它们是否共享状态。

如果每个工具各自创建一个 `SimulatorBackend()`，就会出现“打开灯使用一个字典，查询状态却读取另一个字典”的问题。共享同一个 Registry 可以保证所有工具看到的是同一份设备状态。

#### 4.3.9 可以直接使用 Registry 做什么

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

# 传感器：先推演环境，再读值
registry.tick_environment()
sensor = registry.get("living_room_th_sensor")
print(sensor.temperature, sensor.humidity)      # → 27.0 42

# 按类型取一组传感器（read_sensor 走的就是这条路）
registry.get_by_type(DeviceType.PRESENCE_SENSOR)

# 生成状态报告（给 LLM 看）
print(registry.get_status_summary())

# 生成设备清单提示词（执行器与只读传感器分成两组）
print(registry.get_device_list_prompt())
```

#### 4.3.10 常见误解

**Registry 是数据库吗？**

不是。Registry 是访问入口；当前真正保存状态的是 `SimulatorBackend._devices`。如果换成真实平台，状态可能保存在 Home Assistant 中。

**设备模型会控制真实硬件吗？**

不会。模型只保存和验证数据。真实硬件调用应该由 Backend 实现。

**为什么工具不直接调用 Home Assistant？**

如果工具直接依赖某个平台，就无法复用于模拟器和其他 IoT 平台。工具依赖 Registry，Registry 再依赖抽象 Backend，替换成本更低。

**为什么查找需要同时传名称和设备类型？**

设备类型可以缩小候选范围，避免“客厅设备”“打开一下”等模糊文字错误匹配到另一类设备。

**为什么更新要返回 `bool`？**

Backend 可能遇到设备不存在、字段不合法、设备离线或平台请求失败。布尔结果让 Registry 和工具有机会把失败转换为明确的业务反馈。

**依赖倒置是什么意思？**

高层的工具和 Agent 依赖 `DeviceBackend` 这套抽象能力，而不是依赖 `SimulatorBackend` 的内存字典。后续对接 Home Assistant 时，只需创建 `HomeAssistantBackend(DeviceBackend)` 并在启动入口替换 Backend；工具调用方式和 Agent 图可以保持不变。

**空调设了 24 度，为什么温湿度传感器读出来是 27 度？**

因为两个字段含义不同。`ACDevice.temperature` 是目标值（“我要 24 度”），
`TempHumiditySensor.temperature` 是实测值（“现在实际 27 度”）。两者不一致是正常
状态，正是这个差值让空调有事可做、让验证器有东西可验。

**为什么传感器没有 `control_temp_humidity_sensor`？**

因为真实传感器没有可写状态。给它造一个控制工具，只会让模型以为可以“把湿度调到
60%”——那实际上要控制的是加湿器。工具名本身就是给模型的约束。

**能不能在 `get()` 里自动推演传感器读数？**

不建议。`get()` 被控制、场景、验证等路径大量调用，一旦读取会改状态，读到的数字
就取决于代码里调了几次 `get()`，bug 无法复现。详见 4.3.4b。

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

只读工具走的还是第一条路径，但一次对话里往往会出现两轮：

```text
用户：“有点干”
  → LLM 生成 read_sensor(sensor_type="temp_humidity", location="客厅")
  → 风险判断：只读操作，不需要确认
  → ToolNode 执行 read_sensor（内部先 tick_environment 再读）
  → ToolMessage 返回“湿度 42%”
  → LLM 拿到数值，这一轮才生成 control_humidifier 调用
  → ToolNode 执行控制
  → LLM 生成最终回复
```

这就是 ReAct 循环真正发挥作用的形态：第一次工具调用的结果决定了第二次调什么。
如果没有传感器，模型面对“有点干”只能直接猜一个湿度值去设置。

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
| `memory_reasoner` | 判断检索到的记忆是否适用、冲突、属于约束或被临时指令覆盖 |
| `task_router` | 识别结构化意图，选择澄清、并行查询、RAG、规划或普通 Agent 分支 |
| `clarification` | 信息不足或路由置信度低时请求用户补充信息 |
| `device_query_subgraph` | 使用子图和 `Send` 并行查询多个设备并聚合结果 |
| `knowledge_rag` | 按设备型号检索本地知识，支持改写、引用和无依据拒答 |
| `compact_context` | 限制消息和 token 规模，维护滚动摘要 |
| `agent` | 运行普通 ReAct 或接收 Supervisor 委派后的专用 Agent |
| `approval` | 对批量场景调用执行 `interrupt` |
| `tools` | 使用 `ToolNode` 执行已批准或无需确认的工具 |
| `reject_tools` | 拒绝时生成匹配工具调用 ID 的取消结果 |
| `planner` | 通过结构化输出生成或修订 `ExecutionPlan` |
| `plan_approval` | 在执行完整计划前暂停并展示步骤 |
| `executor` | 每次只执行当前计划中的一个原子工具步骤 |
| `verifier` | 读取真实设备状态，判断成功、重试或重新规划 |
| `planning_finalize` | 汇总完成、取消或失败结果 |
| `supervisor_finalize` | 记录专用 Agent 委派完成和 handoff 轨迹 |

简化后的构图代码如下：

```python
workflow = StateGraph(AgentState)

workflow.add_node("sync_context", sync_context_node)
workflow.add_node("memory_reasoner", memory_reasoner_node)
workflow.add_node("task_router", task_router_node)
workflow.add_node("clarification", clarification_node)
workflow.add_node("device_query_subgraph", parallel_query_node)
workflow.add_node("knowledge_rag", knowledge_rag_node)
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
workflow.add_node("supervisor_finalize", supervisor_finalize_node)

workflow.set_entry_point("sync_context")
workflow.add_edge("sync_context", "memory_reasoner")
workflow.add_edge("memory_reasoner", "task_router")

workflow.add_conditional_edges("task_router", route_task, {
    "planner": "planner",
    "compact_context": "compact_context",
    "clarification": "clarification",
    "device_query_subgraph": "device_query_subgraph",
    "knowledge_rag": "knowledge_rag",
})
workflow.add_edge("clarification", END)
workflow.add_edge("device_query_subgraph", END)
workflow.add_edge("knowledge_rag", END)
workflow.add_edge("compact_context", "agent")

workflow.add_conditional_edges("agent", router, {
    "approval": "approval",
    "tools": "tools",
    "supervisor_finalize": "supervisor_finalize",
    "__end__": END,
})
workflow.add_edge("supervisor_finalize", END)

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

**MCP Client** (`mcp/client.py`): 启动时发现外部 MCP 工具，并将其转换为 LangChain 工具交给 Agent。当前项目附带一个基于**彩云天气**的天气 MCP：

```python
# .env.example 已提供此配置；python 会自动使用当前解释器
EXTERNAL_MCP_SERVERS=[{"name":"weather","transport":"stdio","command":"python","args":["-m","src.mcp.weather_server"]}]
WEATHER_DEFAULT_LOCATION=杭州
CAIYUN_WEATHER_TOKEN=你的彩云 token
```

启动后可直接询问：

```text
杭州现在天气怎么样？
北京未来三天天气如何？
```

天气 MCP 提供 `weather__current_weather` 和 `weather__weather_forecast` 两个只读工具，数据来自彩云天气。彩云 token 可在 <https://dashboard.caiyunapp.com> 免费领取；没配置时工具会返回明确的提示而不是报错（网络不可用时同理）。城市名到坐标的转换仍由免费的 Open-Meteo geocoding 完成，无需额外 Key。

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

传感器让“先看数据再动手”这类对话成立。同一个会话里接着说：

```
👤 你: 有点干

🤖 小智: 我先看一下客厅的湿度。
[调用 read_sensor(sensor_type="temp_humidity", location="客厅")]
📡 **温湿度传感器读数:**
  · 客厅温湿度传感器 (living_room_th_sensor): 温度 27.0°C | 湿度 42% | 电量 100%

🤖 小智: 客厅湿度只有 42%，确实偏干（舒适区间是 40-60%，偏下限）。
已帮你把加湿器目标湿度设为 60%。
✅ 客厅加湿器目标湿度已设为 60%。

👤 你: 现在呢

🤖 小智: 📡 **温湿度传感器读数:**
  · 客厅温湿度传感器 (living_room_th_sensor): 温度 27.0°C | 湿度 48% | 电量 100%
湿度已经从 42% 升到 48%，在朝 60% 走。
```

值得注意的是第二次读到的 48% 不是模型编的，也不是它把自己刚设的 60 念了一遍——
是模拟器按“加湿器正在工作”推演出来的中间值。这个数字来自环境而不是来自 Agent
自己的命令，所以它是整条链路里唯一能真正证明“操作生效了”的证据。

“家里有人吗”这类问题走同一个工具的另一个分支：

```
👤 你: 家里有人吗

🤖 小智: 📡 **人体存在传感器读数:**
  · 客厅人体传感器 (living_room_presence): 🕳️ 无人 | 电量 100%
  · 玄关人体传感器 (entryway_presence): 🕳️ 无人 | 电量 100%
客厅和玄关都没检测到活动。
```

### 5.2 常用命令

| 命令 | 说明 |
|------|------|
| `/status` | 查看所有设备状态 |
| `/scenes` | 列出可用场景模式 |
| `/reset`  | 重置对话记忆 |
| `/history` | 查看当前会话最近的 Checkpoint 状态历史 |
| `/plan` | 复盘最近一次多步骤计划：Planner 产出 + 逐步验证轨迹 |
| `/help`   | 显示使用指南 |
| `/quit`   | 退出 |

### 5.3 高级选项

```bash
# 使用更强的模型
python -m src.main --model qwen-max

# 调试模式（显示详细日志）
python -m src.main --debug

# 额外显示路由 / 记忆判断等诊断事件（规划过程默认就会显示）
python -m src.main --trace

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
python -m pytest -q

# 只运行某个阶段，例如阶段十二
python -m pytest -q tests/test_phase_twelve.py
```

当前共 123 个测试，覆盖阶段一至阶段十二、天气 MCP、加湿器设备闭环、环境传感器和规划过程可视化。测试不是只检查返回文本，还会验证权限边界、数据库状态、设备真实副作用、Checkpoint 恢复和 Agent 轨迹：

| 测试文件 | 主要验证内容 |
| --- | --- |
| `test_phase_one.py` | 可信请求上下文、住宅归属校验、稳定会话和 Checkpoint 模式 |
| `test_phase_two.py` | 长期记忆作用域、权限隔离、持久化和 Agent 工具上下文 |
| `test_phase_three.py` | 上下文压缩、滚动摘要、TTL 清理和会话结束策略 |
| `test_phase_four.py` | 记忆候选、确认、冲突、置信度衰减和行为观察 |
| `test_phase_five.py` | 自然语言抽取、混合排序、访问次数、版本与有效时间 |
| `test_phase_six.py` | Human-in-the-loop 批准、拒绝、无副作用中断和恢复 |
| `test_phase_seven.py` | Planner–Executor–Verifier、重试、重新规划和计划恢复 |
| `test_phase_eight.py` | 结构化意图、低置信度澄清和确定性回退分类 |
| `test_phase_nine.py` | 设备查询子图、动态 `Send` 并行和稳定结果聚合 |
| `test_phase_ten.py` | Supervisor 委派、专用 Agent 能力边界和工具隔离 |
| `test_phase_eleven.py` | 显式记忆决策、Checkpoint 时间旅行和自定义进度事件 |
| `test_phase_twelve.py` | Agentic RAG 型号过滤、引用、拒答和轨迹评测指标 |
| `test_weather_mcp.py` | 天气 MCP 配置解析、stdio 工具发现、同步调用和天气结果格式 |
| `test_humidifier.py` | 加湿器字段约束、状态汇总、控制工具、空水箱保护、场景关闭和 Planner 预期状态 |
| `test_sensors.py` | 传感器字段约束与离线展示、环境推演的确定性、`read_sensor` 的筛选与错误提示、只读约束不被场景与规划绕过 |
| `test_planning_progress.py` | 规划事件的发出顺序（计划先于执行、执行先于验证）、重试与重新规划事件、`PlanProgressView` 的渲染与容错 |

测试数量会随功能增加而变化，应以 `pytest --collect-only -q` 或实际测试输出为准；这里的 123 是规划过程可视化接入后的基线。

`test_sensors.py` 的结构值得单独说一下，它按四层组织，正好对应“新设备接进来要担心
哪四件事”：

1. **模型层**：湿度 120% 被拒、温度 99°C 被拒、离线传感器只报“离线”而不是上一次
   的旧读数；
2. **模拟器层**：环境推演必须确定性——`tick` 二十次湿度正好是 60，制冷正好停在
   24.0 不越过目标，空调只影响同房间的传感器，水箱空了不加湿，非法时间戳不抛异常；
3. **工具层**：`read_sensor` 的类型筛选、房间筛选、两种错误提示，以及一条完整闭环
   （开加湿器 → 反复读 → 湿度确实升到 60）；
4. **架构约束**：离家模式不会关掉传感器，`PlanStep(tool_name="read_sensor")` 直接
   被 Pydantic 拒绝，设备清单提示词里传感器只出现在只读那一组。

第 2 层和第 4 层是这批改动里最容易悄悄坏掉的部分。确定性一旦丢失（比如有人给推演
加了随机抖动），测试会从“断言精确值”退化成“断言大概范围”，然后就再也发现不了
偏差；只读约束一旦漏掉一处（比如新场景直接遍历所有设备做批量关闭），传感器就会
被关掉，而这种 bug 在对话里表现为“Agent 说家里没人”，很难联想到根因。

---

## 6. 扩展指南

### 6.1 添加新设备类型

当前项目已经完整接入 **加湿器 (Humidifier)**。新增设备不能只定义 Pydantic 模型，还要贯通状态展示、工具、规划验证和 MCP。

**第一步**: 在 `src/models.py` 中添加模型：

```python
class HumidifierDevice(BaseDevice):
    device_type: DeviceType = Field(default=DeviceType.HUMIDIFIER, frozen=True)
    target_humidity: int = Field(default=60, ge=30, le=80)
    mist_level: FanSpeed = Field(default=FanSpeed.AUTO)
    water_level: int = Field(default=100, ge=0, le=100)

    def to_status_text(self) -> str:
        ...

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
    target_humidity=60,
    mist_level=FanSpeed.AUTO,
    water_level=100,
),
```

**第三步**: 在 `src/tools/devices.py` 中创建工具：

```python
@tool
def control_humidifier(
    device_name: str,
    action: str,
    target_humidity: int = 60,
    mist_level: str = "auto",
) -> str:
    """支持 on、off、set_humidity 和 set_mist_level。"""
    # 实现逻辑
```

**第四步**: 在 `src/tools/__init__.py` 和 `agent/graph.py` 中注册，使普通 ReAct 与 Device Agent 都能获得工具。

**第五步**: 如果允许复杂任务规划，还要把工具加入 `agent/planning.py`，并定义每个 action 对应的可验证期望状态。

**第六步**: 在 `mcp/server.py` 暴露 MCP 工具，并用自动化测试验证状态变化，而不只是检查返回文字。

#### 6.1b 如果新设备是只读传感器

项目已经完整接入 **温湿度传感器** 和 **人体存在传感器**。它们走的是另一条路径，
差别集中在三步：

| 步骤 | 执行器（加湿器） | 只读传感器（温湿度） |
| --- | --- | --- |
| 模型 | 定义可写字段 + `DeviceType` 枚举项 | 同上，另外把类型加进 `SENSOR_DEVICE_TYPES` |
| 工具 | 新写一个 `control_xxx` | 在 `read_sensor` 的类型映射里加一项，**不写** `control_xxx` |
| 规划 | 加进 `PLANNING_TOOL_NAMES` 并定义期望状态 | **跳过**——读取不改状态，无法验证，也不该成为计划的一步 |
| 场景 | 需要考虑批量开关时怎么处理 | 无需处理，`SENSOR_DEVICE_TYPES` 已把它排除 |
| 模拟器 | 注册默认设备即可 | 注册默认设备，并在 `_tick_*` 里定义读数如何随执行器变化 |

只有传感器需要多写“读数怎么变”这一段。这段代码是可选的——完全可以让传感器返回
常量，但那样“开加湿器 → 湿度上升 → 验证通过”这条闭环就演示不出来，验证器读到的
永远是同一个数，等于没验。写这段推演时唯一的硬要求是**确定性**：固定步长、
不用随机数，否则测试只能断言范围，也就再也守不住行为。

`SENSOR_DEVICE_TYPES` 这个 frozenset 是这套区分的落点：

```python
SENSOR_DEVICE_TYPES = frozenset({
    DeviceType.TEMP_HUMIDITY_SENSOR,
    DeviceType.PRESENCE_SENSOR,
})
```

场景层、设备清单提示词和状态汇总都查它。加一个新传感器时忘了往这里加，症状是
离家模式会把它一起关掉、规划器会试图“打开”它——都不会报错，只会行为诡异。

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

注意 `tick_environment()` 在 `DeviceBackend` 里**不是** `@abstractmethod`，
而是一个返回 `None` 的具体方法：

```python
def tick_environment(self) -> None:
    """推演环境读数。真实后端不需要覆写。"""
    return None
```

这不是偷懒。环境推演是模拟器特有的问题——真实传感器自己会上报读数，
`HomeAssistantBackend` 无事可做。如果把它写成抽象方法，每个新后端都被迫写一个
毫无意义的 `pass`，而且那个 `pass` 会让读代码的人以为“这里本来应该做点什么”。
默认实现返回 `None` 表达的是“不需要做任何事”，语义准确。

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

| 记忆层 | 解决的问题 | 典型内容 | 保存位置 | 开关 |
|--------|------------|----------|----------|------|
| **短期会话记忆** | “这一轮对话之前聊了什么？” | 用户消息、AI 回复、工具调用结果、滚动摘要、当前关注的房间和设备 | `data/checkpoints.db` | `CHECKPOINT_DB_PATH` |
| **长期结构化记忆** | “这个用户长期喜欢什么、家里有什么固定规则？” | 灯光偏好、空调温度、设备别名、生活例程、家庭约束 | `data/memories.db` | `ENABLE_LONG_TERM_MEMORY` |

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

“喜欢暖光”在未来的新会话中仍然有用，属于**长期记忆**。

**先看代码分布**，后面每一节都会回到这张表：

| 文件 | 行数 | 职责 |
|------|-----:|------|
| `src/memory/models.py` | 128 | 六个 Pydantic 数据模型，定义“记忆长什么样” |
| `src/memory/repository.py` | 520 | SQLite 建表、迁移、CRUD。只管读写，不判断权限 |
| `src/memory/service.py` | 294 | 权限、作用域、候选、检索排序、冲突合并 |
| `src/memory/extractor.py` | 67 | 正则抽取自然语言候选。不碰数据库 |
| `src/memory/summarizer.py` | 90 | 会话摘要与 token 估算（服务于短期记忆） |
| `src/memory/store.py` | 117 | Checkpointer 创建与过期清理（服务于短期记忆） |
| `src/tools/memory.py` | — | 暴露给 LLM 的 9 个记忆工具 |

两层记忆的关系（注意**明确表述不经过候选**，这是三条路径中唯一直达的一条）：

```text
当前会话消息 ──> Checkpoint ──> 同一个 thread_id 恢复上下文


① 用户明确要求记住 ─────────────────────────────────> 长期记忆
                                                          ↑
② 同一操作重复 3 次 ──┐                                   │
                      ├─> 候选记忆 ──> 用户确认 ──────────┘
③ 对话中被动抽取 ─────┘         │
                                └─> 用户拒绝 ──> rejected（不写入）

新会话中的问题 ──> 按当前身份检索 Top-K ──> 注入系统提示词
```

> ⚠️ 路径 ① 直接写库，**没有确认步骤**。理由见 6.4.6：用户已经明说了“以后按这个来”，再弹一次确认属于多余。候选机制针对的是系统自己**猜出来**的偏好（路径 ②③）。

#### 6.4.2 短期会话记忆：Checkpoint 如何工作

**代码位置：`src/memory/store.py:37-79`**

这个函数做一件事：根据配置返回内存版还是 SQLite 版的 checkpointer。

```python
# src/memory/store.py:37
def create_checkpointer(db_path: Optional[str] = None):
    if db_path:
        if not _HAS_SQLITE:
            raise RuntimeError(
                "SQLite checkpointing is configured but "
                "langgraph-checkpoint-sqlite is not installed"
            )
        try:
            db_dir = os.path.dirname(db_path)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)      # 目录不存在就建

            conn = sqlite3.connect(db_path, check_same_thread=False)
            checkpointer = SqliteSaver(conn)
            logger.info(f"✅ SQLite 检查点已就绪 | path={db_path}")
            return checkpointer
        except Exception as exc:
            raise RuntimeError(
                f"failed to initialize SQLite checkpointer at {db_path!r}: {exc}"
            ) from exc

    logger.info("📝 使用内存检查点（会话记忆在重启后丢失）")
    return MemorySaver()
```

三个容易忽略但重要的设计：

1. **`check_same_thread=False`**（`store.py:69`）：LangGraph 可能在不同线程里执行节点，默认的 SQLite 线程检查会直接抛异常。
2. **配置了路径就必须成功**（`store.py:73-76`）：不会“SQLite 打不开就悄悄退回内存模式”。否则用户以为在持久化，重启后发现会话全丢。宁可启动就报错。
3. **`_HAS_SQLITE` 是可选依赖**（`store.py:29-34`）：`langgraph-checkpoint-sqlite` 没装时，只要不配置 `CHECKPOINT_DB_PATH` 就仍能跑内存模式。

编译图时把它交给 LangGraph（`src/agent/graph.py:730` 附近）：

```python
checkpointer = create_checkpointer(settings.memory.db_path)
graph = workflow.compile(checkpointer=checkpointer)
```

每次图执行后，LangGraph 按 `thread_id` 保存整个 `AgentState`。再次用相同 `thread_id` 调用时自动恢复：

```text
thread_id="session-a"
  第 1 轮：[用户：打开客厅灯，AI：已打开]
  第 2 轮：[……，用户：再暗一点，AI：已调暗]

thread_id="session-b"
  另一段独立会话，不会读取 session-a 的消息历史
```

两种存储方式的取舍：

| 对比项 | `MemorySaver` | `SqliteSaver` |
|--------|---------------|---------------|
| 触发条件 | `CHECKPOINT_DB_PATH` 为空 | 配置了路径 |
| 存储位置 | 当前 Python 进程内存 | `data/checkpoints.db` |
| 进程重启后 | 会话状态丢失 | 可以恢复原会话状态 |
| 适用场景 | 单元测试、临时调试 | 本地开发和持久化会话 |

**会话过期清理：`src/memory/store.py:89-117`**

`cleanup_expired_checkpoints()` 遍历所有 thread，取每个 thread 最新快照的时间戳，超过 TTL 的整个删除：

```python
# src/memory/store.py:98-108
for item in checkpointer.list(None):          # None = 不过滤，扫全部
    configurable = item.config.get("configurable", {})
    thread_id = configurable.get("thread_id")
    if not thread_id:
        continue
    timestamp = item.checkpoint.get("ts")
    if timestamp:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        previous = latest_by_thread.get(thread_id)
        if previous is None or parsed > previous:
            latest_by_thread[thread_id] = parsed   # 只留最新的一条
```

注意 `.replace("Z", "+00:00")`：Python 3.10 及更早的 `fromisoformat` 不认 `Z` 后缀，而 LangGraph 写入的时间戳用的正是 `Z`。判断过期时统一 `.astimezone(timezone.utc)`（`store.py:113`），避免带时区和不带时区的时间相减报错。

TTL 由 `CHECKPOINT_SESSION_TTL_HOURS` 控制，默认 168 小时（7 天）。这个函数需要**由调用方主动触发**，不是后台定时任务。

> 依赖以 `pyproject.toml` 为准。请先使用项目既有的 Conda 环境检查依赖；如果环境缺少所需包，应暂停运行并交由环境维护者配置，不要在教程步骤中擅自创建环境或安装依赖。

要恢复一段旧会话，不仅要保留数据库，还必须继续使用原来的 `thread_id`。仅仅重启程序后生成一个新的会话 ID，不会自动进入旧会话。

#### 6.4.3 为什么还要压缩会话上下文

Checkpoint 能保存消息，但不能让消息无限增长。长对话会带来三个问题：

1. 输入 token 越来越多，模型调用成本和延迟上升。
2. 很久以前的无关信息会干扰当前推理。
3. 工具返回内容可能很长，持续保存在状态中会让数据库膨胀。

因此每次调用模型前都会经过 `compact_context` 节点（图结构见 `src/agent/graph.py:608`、`:653`、`:725-726`）：

```text
sync_context → memory_reasoner → task_router
                                     ↓
                              compact_context  ── 保留最近消息、生成滚动摘要、裁剪旧工具结果
                                     ↓
                                   agent
                                     ↓ 有工具调用
                             tools ──┴──> compact_context（回到压缩，形成闭环）
```

注意 `tools` 执行完不是直接回 `agent`，而是**再过一次压缩**。因为工具结果本身就是上下文膨胀的主要来源。

**决定保留多少：`src/memory/summarizer.py:32-36`**

```python
# src/memory/summarizer.py:32
keep_from = max(0, len(messages) - max_messages)
while keep_from < len(messages) - 1 and estimate_tokens(messages[keep_from:]) > max_tokens:
    keep_from += 1
old = messages[:keep_from]
recent = messages[keep_from:]
```

两道闸门叠加：先按**条数**切出最近 12 条，如果这 12 条估算 token 仍超限，就继续往后推窗口。`keep_from < len(messages) - 1` 保证**至少留一条**消息——否则模型会收到空输入。

token 估算是刻意做成廉价且确定的（`summarizer.py:10-12`）：

```python
def estimate_tokens(messages: Iterable) -> int:
    return sum(max(1, (len(str(getattr(message, "content", ""))) + 1) // 2) for message in messages)
```

按“2 个字符 ≈ 1 token”粗算。它不追求准确，只用来做护栏；换成真正的 tokenizer 会引入依赖和耗时，而护栏并不需要那种精度。

**裁剪超长工具结果：`src/memory/summarizer.py:84-90`**

```python
def _truncate_message(message, max_chars: int):
    content = str(getattr(message, "content", ""))
    if max_chars <= 0 or len(content) <= max_chars:
        return message
    marker = "\n…（工具结果已裁剪）"
    kept = max(0, max_chars - len(marker))
    return message.model_copy(update={"content": content[:kept] + marker})
```

只裁 `ToolMessage`（`summarizer.py:26-31` 用 `isinstance` 判断），用户和 AI 的消息一律不动。留下明确的裁剪标记，模型才知道内容不完整，而不是误以为工具就返回了这么多。

**真正写回 Checkpoint：`src/memory/summarizer.py:64-74`**

```python
recent_ids = {message.id for message in recent if getattr(message, "id", None)}
removals = [
    RemoveMessage(id=message.id)
    for message in messages
    if getattr(message, "id", None) and message.id not in recent_ids
]
```

`RemoveMessage` 是 LangGraph 提供的删除指令。把它放进 `messages` 更新里，`add_messages` reducer 就会把对应消息从持久化状态中真正移除——不是只在这一轮不传给模型，而是数据库里也不再保留。这是“压缩”与“截断”的区别。

**节点里如何汇总：`src/agent/graph.py:249-270`**

```python
updates, summary, token_estimate = build_compaction_update(
    list(state["messages"]),
    state.get("conversation_summary", ""),      # 上一轮的摘要，滚动累积
    max_messages=getattr(settings.memory, "context_max_messages", 12),
    max_tokens=getattr(settings.memory, "context_max_tokens", 2400),
    ...
)
kept_count = len(state["messages"]) - sum(
    1 for message in updates if message.__class__.__name__ == "RemoveMessage"
)
```

摘要是**滚动合并**的（`summarizer.py:79-81`）：旧摘要与新归纳拼接后取末尾 `max_summary_chars` 个字符，所以越久远的内容会自然淡出。`conversation_summary` 最终被拼进系统提示词（`graph.py:535`），模型仍能看到早期对话的梗概。

相关配置（`src/config.py:44-70`，前缀 `CHECKPOINT_`）：

| 配置项 | 默认值 | 作用 |
|--------|-------:|------|
| `CHECKPOINT_CONTEXT_MAX_MESSAGES` | `12` | 保留的最近消息条数 |
| `CHECKPOINT_CONTEXT_MAX_TOKENS` | `2400` | 上下文估算 token 上限，超限则继续收窄窗口 |
| `CHECKPOINT_TOOL_RESULT_MAX_CHARS` | `1200` | 单条工具结果的字符上限 |
| `CHECKPOINT_SUMMARY_MAX_CHARS` | `1800` | 滚动摘要的最大长度 |
| `CHECKPOINT_SESSION_TTL_HOURS` | `168` | 会话检查点保留时长，默认 7 天 |

压缩过程还会把 `context_message_count` 和 `context_token_estimate` 写回状态（`graph.py:265-266`），配合 `graph.py:260-262` 的 `logger.debug`，可以直接观察上下文规模的变化趋势。

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

四种类型不是随手分的，它们在 `src/memory/models.py:23-27` 里是枚举，写库时由 Pydantic 卡住：

```python
# src/memory/models.py:23
class MemoryType(str, Enum):
    PREFERENCE = "preference"      # 偏好
    ALIAS = "alias"                # 别名
    ROUTINE = "routine"            # 例程
    CONSTRAINT = "constraint"      # 约束
```

**一条正式记忆长什么样：`src/memory/models.py:30-53`**

```python
class MemoryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")     # 多传字段直接报错

    id: str
    home_id: str                                  # 必填，家庭隔离的根据
    user_id: str | None = None                    # None 表示共享记忆
    room_id: str | None = None
    device_id: str | None = None
    scope: MemoryScope
    memory_type: MemoryType
    memory_key: str
    memory_value: dict[str, Any]
    confidence: float = Field(default=1.0, ge=0, le=1)    # 有多确信
    source: str | None = None                             # 来自哪里，可追溯
    status: str = "active"
    created_at: datetime
    updated_at: datetime
    expires_at: datetime | None = None
    importance: float = Field(default=0.5, ge=0, le=1)    # 有多重要
    access_count: int = Field(default=0, ge=0)            # 被检索命中几次
    last_accessed_at: datetime | None = None
    valid_from: datetime                                  # 版本生效时间
    valid_to: datetime | None = None                      # None = 当前有效版本
    version: int = Field(default=1, ge=1)
```

几个字段的设计意图：

- **`extra="forbid"`**：宁可在写入时报错，也不要让拼错的字段名被静默丢弃。所有六个模型都开了这个开关。
- **`user_id` 用 `None` 表达共享**，不是用空字符串或哨兵值。这样 6.4.5 的 SQL 权限过滤可以直接写 `user_id IS NULL OR user_id = ?`。
- **`confidence` 与 `importance` 分开**：“我有多确信这是真的”和“这条有多值得优先”是两件事。用户明说的偏好 confidence 高；儿童房音量约束 importance 高。检索排序会分别用到（6.4.7）。
- **`ge=0, le=1` 的约束**是有意义的护栏：分数只要越界，排序公式的结果就没法解释了。
- **`access_count` / `last_accessed_at`** 支撑检索的频次项和置信度衰减（6.4.7、6.4.8）。
- **`valid_from` / `valid_to` / `version`** 是版本机制的三件套。`valid_to is None` 即当前版本，见 6.4.8。

**写入用的是另一个模型：`src/memory/models.py:56-69`**

```python
class MemoryWrite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope: MemoryScope
    memory_type: MemoryType
    memory_key: str = Field(min_length=1)
    memory_value: dict[str, Any]
    room_id: str | None = None
    device_id: str | None = None
    source: str | None = None
    confidence: float = Field(default=1.0, ge=0, le=1)
    expires_at: datetime | None = None
    importance: float = Field(default=0.5, ge=0, le=1)
    valid_from: datetime | None = None
```

对比 `MemoryRecord` 会发现：`MemoryWrite` **没有 `home_id`、`user_id`、`id`、`version`、`access_count`**。这不是省略，而是刻意的边界——这些字段由服务端从可信上下文填入，调用方（包括 LLM）**没有位置**去指定它们。这是 6.4.5 权限模型能成立的前提。

`memory_key` 用 `min_length=1` 卡住空字符串，因为它参与唯一键判重（6.4.8）。

`src/memory/models.py` 里还有四个模型，各自服务一个环节：`MemoryVersion`（历史版本快照）、`ExtractedMemoryCandidate`（抽取器输出，见 6.4.6）、`PreferenceCandidate`（待确认候选）、`MemoryConflict`（冲突记录，见 6.4.8）。

**为什么是 `memory_key` + JSON `memory_value`，而不是一句自然语言？**

```python
memory_key="lighting.color", memory_value={"color": "暖光"}
```

用点号分段的 key 让同类偏好天然归类（`lighting.color`、`lighting.brightness`、`ac.temperature`），并且成为判重和合并的依据——如果存的是“我喜欢暖光”这句话，系统无法判断它和“灯光颜色设为暖白”是不是同一条。JSON value 则让合并可以逐字段进行（6.4.8 的 `_merge_values`），也让测试可以直接断言字典相等，而不是去匹配一段可能被模型改写的文本。

#### 6.4.5 作用域：这条记忆应该让谁看见

四种作用域定义在 `src/memory/models.py:16-20`：

| 作用域 | 含义 | 示例 | 写入权限 |
|--------|------|------|----------|
| `user` | 只属于当前用户 | 用户 A 喜欢暖光 | 本人 |
| `home` | 整个家庭共享 | 离家时关闭所有灯 | 管理员 |
| `room` | 某个房间共享 | 儿童房夜间保持安静 | 管理员 |
| `device` | 针对某台设备 | 客厅电视的默认音量限制 | 管理员 |

**可信身份从哪来：`src/agent/context.py:17-49`**

```python
class AgentContext(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    home_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    client_id: str = Field(min_length=1)
    room_id: str | None = None
    device_id: str | None = None
    is_admin: bool = False

    def to_config(self) -> dict:
        """Build the only supported LangGraph configurable payload."""
        return {"configurable": {"thread_id": self.session_id, **self.model_dump()}}
```

四个必填 ID 用 `min_length=1` 加 `reject_blank_required_ids` 校验器（`context.py:30-35`）双重卡空值。注意 `session_id` 直接充当 LangGraph 的 `thread_id`——身份和会话绑在一起，换会话不会换身份。

记忆工具的函数签名里**没有** `home_id` 和 `user_id`，它们只能从 `RunnableConfig` 里读。这样即使模型生成了越权的工具参数，也没有可以填写的位置（详见 6.4.11）。

**房间和设备归属校验：`src/agent/context.py:99-120`**

`SpaceDirectory.validate()` 是所有记忆操作的第一道门，`MemoryService` 的每个公开方法开头都调用它：

```python
def validate(self, context: AgentContext) -> AgentContext:
    rooms = self._rooms_by_home.get(context.home_id)
    if rooms is None:
        raise ContextValidationError(f"unknown home_id: {context.home_id}")

    if context.room_id and context.room_id not in rooms:
        raise ContextValidationError(...)            # 房间不属于这个家

    if context.device_id:
        location = self._devices.get(context.device_id)
        if location is None or location.home_id != context.home_id:
            raise ContextValidationError(...)        # 设备不属于这个家
        if context.room_id and location.room_id != context.room_id:
            raise ContextValidationError(...)        # 设备不在这个房间
```

三层递进：家 → 房间归属 → 设备归属，且设备和房间必须自洽。伪造一个别人家的 `device_id` 会在这里被挡住，根本到不了数据库。

**作用域自洽性检查：`src/memory/service.py:235-263`**

`_normalize_and_validate_scope()` 在写入前修正并校验字段组合：

```python
if item.scope == MemoryScope.HOME:
    if room_id or device_id:
        raise ValueError("home memory cannot specify room_id or device_id")
elif item.scope == MemoryScope.ROOM:
    if not room_id or device_id:
        raise ValueError("room memory requires room_id and cannot specify device_id")
elif item.scope == MemoryScope.DEVICE:
    if not device_id:
        raise ValueError("device memory requires device_id")
    inferred_room_id = self.spaces.room_for_device(device_id)
    room_id = room_id or inferred_room_id          # 设备记忆自动补房间
elif device_id and not room_id:                    # scope == USER
    room_id = self.spaces.room_for_device(device_id)
```

“全家规则”带着房间 ID 是自相矛盾的，直接拒绝；`device` 作用域则会**自动推断房间**（`room_for_device`，`context.py:122-126`），调用方不必手填，也不会填错。归一化后还要再 `validate` 一遍（`service.py:257-261`），因为推断出的 `room_id` 同样需要校验归属。

**共享作用域的管理员闸门：`src/memory/service.py:36-44`**

```python
shared_scope = item.scope in {MemoryScope.HOME, MemoryScope.ROOM, MemoryScope.DEVICE}
if shared_scope and not (is_admin or context.is_admin):
    raise MemoryPermissionError(
        "home, room, and device memories require administrator permission"
    )
owner = None if shared_scope else context.user_id     # 共享记忆没有 owner
```

最后一行是整套权限模型的枢纽：**共享记忆的 `user_id` 存 `NULL`，个人记忆存实际用户**。这个约定让读取权限可以完全交给 SQL。

**读取权限就是一条 SQL：`src/memory/repository.py:376-393`（`list_accessible`）**

```sql
SELECT * FROM memories
WHERE home_id = ? AND status = 'active'
  AND (user_id IS NULL OR user_id = ?)      -- 共享的 + 自己的
  AND (room_id IS NULL OR room_id = ?)      -- 全局的 + 当前房间的
  AND (device_id IS NULL OR device_id = ?)  -- 全局的 + 当前设备的
ORDER BY CASE scope WHEN 'home' THEN 0 WHEN 'room' THEN 1
                    WHEN 'device' THEN 2 ELSE 3 END, updated_at
```

三个 `IS NULL OR = ?` 同时表达了“共享内容都可见”和“专属内容只在对应上下文可见”。权限过滤放在 SQL 里而不是取回后用 Python 筛，意味着**不可能因为漏写一个 `if` 而泄漏数据**——越权的行根本不会被 SELECT 出来。

**单条读取的两种拒绝：`src/memory/service.py:224-233`**

```python
def _authorized_record(self, context, memory_id, *, is_admin) -> MemoryRecord:
    self.spaces.validate(context)
    record = self.repository.get(memory_id, context.home_id)   # home_id 已在 SQL 里
    if record is None:
        raise KeyError(memory_id)
    if record.user_id is not None and record.user_id != context.user_id:
        raise MemoryPermissionError("personal memory belongs to another user")
    if record.user_id is None and not is_admin:
        raise MemoryPermissionError("home shared memory requires administrator permission")
    return record
```

两个错误信息刻意分开：“这是别人的个人记忆”和“共享记忆需要管理员”是不同的失败原因，混成一句话会让排查变难。`update` 和 `delete` 都走这个方法（`service.py:194-207`），而只读的 `get`（`service.py:184-192`）不检查管理员——共享记忆本来就是给全家看的。

#### 6.4.6 记忆如何产生：明确保存与候选确认

对应 6.4.1 的三条路径：路径 ① 直接写库，路径 ②③ 先进候选池。

**路径 ①：用户明确要求记住 → `MemoryService.save()`（`src/memory/service.py:25-57`）**

用户说“请记住，我喜欢暖光”，Agent 调用 `save_personal_memory` 工具，最终落到 `save()`：

```python
def save(self, context, item: MemoryWrite, *, is_admin: bool = False) -> MemoryRecord:
    self.spaces.validate(context)                              # ① 空间归属
    item = self._normalize_and_validate_scope(context, item)   # ② 作用域自洽
    shared_scope = item.scope in {MemoryScope.HOME, MemoryScope.ROOM, MemoryScope.DEVICE}
    if shared_scope and not (is_admin or context.is_admin):    # ③ 管理员闸门
        raise MemoryPermissionError(...)
    owner = None if shared_scope else context.user_id
    existing = self.repository.find_by_key(...)                # ④ 是否已有同 key
    if existing:
        merged = _merge_values(existing.memory_value, item.memory_value)
        if existing.memory_value != item.memory_value:
            resolution = "merged" if merged != item.memory_value else "incoming_wins"
            self.repository.add_conflict(existing, item.memory_value, merged, resolution)
        item = item.model_copy(update={"memory_value": merged})
    return self.repository.upsert(context.home_id, owner, item)
```

**这里没有确认步骤**。用户已经明说了“记住”，再弹一次“要记住吗”是多余的交互。校验的是权限，不是意图。

第 ④ 步顺带解决了重复保存：同 key 不会写出两条记录，而是走合并 + 冲突记录（6.4.8）。

**路径 ②：重复操作统计 → `record_operation()`（`src/memory/service.py:59-75`）**

```python
def record_operation(self, context, memory_key, memory_value, *, minimum_repetitions: int = 3):
    """Aggregate a real operation and create, but never auto-save, a candidate."""
    self.spaces.validate(context)
    count = self.repository.observe_preference(
        context.home_id, context.user_id, memory_key, memory_value,
        context.room_id, context.device_id,
    )
    if count < minimum_repetitions:
        return None                                   # 不到 3 次，只累计不生成候选
    confidence = min(0.95, 0.5 + 0.1 * count)
    return self.repository.upsert_candidate(...)
```

docstring 里的 “but never auto-save” 是这个方法的全部要点：它写的是 `preference_candidates`，不是 `memories`。

置信度 `min(0.95, 0.5 + 0.1 * count)`：3 次得 0.8，4 次 0.9，5 次及以上封顶 0.95。**永远不到 1.0**——统计推断出的偏好不该和用户亲口说的享有同等确信度。

**路径 ③：自然语言抽取 → `extract_candidates_from_text()`（`src/memory/service.py:77-90`）**

它在 `sync_context` 节点里被调用（`src/agent/graph.py:213`），每轮对话都跑一次。真正的抽取逻辑在 `src/memory/extractor.py`，全文只有 67 行，但设计很讲究。

**先否决，再匹配：`src/memory/extractor.py:14-24`**

```python
_STABLE_MARKERS = ("我喜欢", "我偏好", "我习惯", "我通常", "我一般", "以后都", "以后请")
_TEMPORARY_MARKERS = ("今天", "这次", "现在", "有点", "暂时", "刚才")

def extract_memory_candidates(text: str) -> list[ExtractedMemoryCandidate]:
    normalized = text.strip()
    if not normalized or any(marker in normalized for marker in _TEMPORARY_MARKERS):
        return []                                     # ① 临时词一票否决
    if not any(marker in normalized for marker in _STABLE_MARKERS):
        return []                                     # ② 没有稳定词，不猜
```

两道过滤的顺序不能换。**临时标记优先于稳定标记**：“今天有点冷，我一般把空调开 25 度”同时含“今天/有点”和“我一般”，先否决就正确地不生成候选。反过来先匹配稳定词就会误记。

这是典型的**高精度、低召回**取舍：漏抽一条偏好，用户下次再说一遍就行；错记一条偏好，用户要主动发现并删除，代价高得多。

**四个正则抽取器：`src/memory/extractor.py:27-51`**

| memory_key | 触发词 + 取值 | 产出的 value | confidence | importance |
|------------|---------------|--------------|-----------:|-----------:|
| `ac.temperature` | 空调 / 温度 + 16–30 度 | `{"temperature": 25}` | 0.82 | 0.75 |
| `lighting.brightness` | 灯 / 亮度 + 0–100% | `{"brightness": 30}` | 0.80 | 0.65 |
| `lighting.color` | 暖白 / 暖黄 / 暖光 / 冷白光 / 白光 | `{"color": "暖光"}` | 0.84 | 0.65 |
| `routine.quiet_hours` | 时刻 + 安静 / 静音 / 低音量 | `{"after": "22:00"}` | 0.76 | 0.80 |

```python
# src/memory/extractor.py:27
temperature = re.search(r"(?:空调|温度).*?(1[6-9]|2\d|30)\s*(?:度|℃|°C)?", normalized, re.I)
```

温度范围 `1[6-9]|2\d|30` 直接写在正则里，而不是先抓任意数字再判断区间——“我一般把音量设成 50”里的 50 不会被误当成空调温度。

```python
# src/memory/extractor.py:39
color_match = re.search(r"(暖白|暖黄|暖光|冷白光|白光)", normalized)
if color_match and any(word in normalized for word in ("灯", "光", "色温")):
```

颜色词之外**额外要求**上下文出现“灯 / 光 / 色温”，否则“我喜欢暖黄色的墙纸”会被记成灯光偏好。

```python
# src/memory/extractor.py:45
quiet = re.search(r"(?:晚上|夜间)?\s*(\d{1,2})(?::(\d{2}))?\s*点?.*?(?:安静|静音|低音量)", normalized)
if quiet:
    hour = max(0, min(23, int(quiet.group(1))))     # 钳到合法小时
    minute = int(quiet.group(2) or 0)
```

`max(0, min(23, ...))` 把小时钳进 0–23，`quiet.group(2) or 0` 处理“10 点”这种没写分钟的情况。它的 confidence 最低（0.76）但 importance 最高（0.80）：时间表达更容易解析错，可“晚上要安静”一旦成立就很关键。

最后 `_deduplicate` 以 `(memory_key, repr(sorted(value.items())))` 为键去重（`extractor.py:63-67`），同一句话里重复提到同一偏好只留一条。用 `sorted(...items())` 而不是 `str(dict)`，因为字典字面顺序不同但内容相同的两个 value 应当算重复。

**抽取器不碰数据库**：它的返回类型是 `ExtractedMemoryCandidate`（纯数据），入库由 `service.py:84-89` 完成。这样抽取逻辑可以脱离 SQLite 单测——给一句话，断言返回的候选列表，仅此而已。

**确认与拒绝：`src/memory/service.py:96-121`**

```python
def confirm_candidate(self, context, candidate_id: str) -> MemoryRecord:
    candidate = self.repository.get_candidate(candidate_id, context.home_id)
    if candidate is None or candidate.user_id != context.user_id or candidate.status != "pending":
        raise KeyError(candidate_id)                  # 三个条件合并成同一个错误
    record = self.save(context, MemoryWrite(
        scope=MemoryScope.USER,                       # 恒为个人
        memory_type=MemoryType.PREFERENCE,            # 恒为偏好
        memory_key=candidate.memory_key,
        memory_value=candidate.memory_value,
        confidence=candidate.confidence,
        importance=candidate.importance,
        source=f"confirmed_candidate:{candidate.id}", # 可追溯到候选
    ))
    self.repository.resolve_candidate(
        candidate.id, context.home_id, context.user_id, "confirmed", record.id
    )
    return record
```

三处细节：

1. **`scope` 和 `memory_type` 写死**，不取自候选。系统猜出来的东西**永远只能变成个人偏好**，绝无可能升级成全家规则——那需要管理员显式操作。
2. **三个失败条件抛同一个 `KeyError`**：不存在、不是你的、已处理过，对外都是“找不到这个候选”。分开报错等于告诉调用方“这个 ID 存在但不属于你”。
3. **`source` 记录来源候选 ID**，事后可以追问“这条记忆当初是怎么来的”。

`reject_candidate()`（`service.py:117-121`）只把状态改成 `rejected`，候选行仍在库里——保留“用户拒绝过”这个事实，比删掉它更有价值。

```text
pending 候选
   ├─ confirm_candidate() ──> confirmed ──> 写入 memories 表 + 回填 confirmed_memory_id
   └─ reject_candidate()  ──> rejected  ──> 保留记录，不进 memories
```

#### 6.4.7 检索：不是把所有记忆都塞进 Prompt

当长期记忆变多时，每轮把全部记录交给模型既浪费 token，也会引入无关信息。因此 `sync_context` 节点会按当前问题检索 Top-K。

**调用点：`src/agent/graph.py:211-228`**

```python
if memory_service and state.get("request_home_id") and state.get("request_user_id"):
    context = _context_from_state(state, result)
    memory_service.extract_candidates_from_text(context, latest_text)   # 顺带抽候选
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
```

三点值得注意：

- 前置 `if` 要求 `home_id` 和 `user_id` 都存在。**没有可信身份就完全不检索**，而不是退化成“查全库”。
- 抽取候选和检索在同一个分支里完成，共用一次 `latest_text` 和一份 `context`。
- 格式化后的字符串写进 `memory_context`，最终拼进系统提示词（`graph.py:536`）。原始记录另存 `retrieved_memories`（`graph.py:219`），供调试和评估使用。

`src/memory/service.py:213-222` 有一个功能等价的 `format_for_prompt()`，输出格式与上面完全一致。图里选择内联，是因为它同时需要格式化文本**和**原始记录两份产物。

**排序公式：`src/memory/service.py:165-178`**

```python
@staticmethod
def _retrieval_score(record: MemoryRecord, query: str, now) -> float:
    searchable = (
        record.memory_key + " " + json.dumps(record.memory_value, ensure_ascii=False)
    ).lower()
    terms = _query_terms(query)
    relevance = sum(1 for term in terms if term in searchable) / max(1, len(terms))
    age_days = max(0.0, (now - record.updated_at).total_seconds() / 86400)
    recency = math.exp(-age_days / 90)
    frequency = min(1.0, math.log1p(record.access_count) / math.log(11))
    return (
        0.45 * relevance + 0.20 * record.confidence
        + 0.20 * record.importance + 0.10 * recency + 0.05 * frequency
    )
```

| 分项 | 权重 | 计算方式 | 为什么这么算 |
|------|-----:|----------|--------------|
| 词项相关性 | 0.45 | 命中词数 / 总词数 | 归一化到 0–1，长问题不会因为词多而占优 |
| 置信度 | 0.20 | 记录字段 | 系统对这条记忆有多确信 |
| 重要性 | 0.20 | 记录字段 | 业务价值，与确信度独立 |
| 时间新鲜度 | 0.10 | `exp(-天数/90)` | 90 天衰减到 1/e，平滑而非断崖 |
| 访问频率 | 0.05 | `log1p(n)/log(11)` | 10 次即接近 1.0，防止高频记忆霸榜 |

`ensure_ascii=False` 是必需的：否则中文 value 会变成 `\uXXXX` 转义，中文查询词永远匹配不上。`max(1, len(terms))` 防止空查询导致除零。`max(0.0, ...)` 兜住时钟回拨造成的负数天龄。

**同义词扩展：`src/memory/service.py:277-294`**

这一步是让中文查询能命中英文 key 的关键：

```python
def _query_terms(query: str) -> set[str]:
    lowered = query.lower()
    terms = set(re.findall(r"[a-z0-9_.]+|[一-鿿]{2,}", lowered))
    aliases = {
        "空调": {"ac", "temperature", "mode", "fan"},
        "温度": {"temperature", "ac"},
        "灯": {"lighting", "brightness", "color"},
        "亮度": {"brightness", "lighting"},
        ...
    }
    for marker, expansions in aliases.items():
        if marker in lowered:
            terms.update(expansions)
    return terms
```

没有这张表，“空调调到多少度”里一个中文词都匹配不上 `ac.temperature` 这个英文 key，relevance 恒为 0。

正则 `[a-z0-9_.]+|[一-鿿]{2,}` 分两类切词：英文串连点号一起保留（`ac.temperature` 可整体匹配），中文取**连续 2 字以上**——单个汉字噪声太大。注意 alias 匹配用的是 `marker in lowered`（在原始字符串里找子串），所以单字键“灯”依然能生效，绕开了 2 字下限。

**顺序不能颠倒：`src/memory/service.py:147-163`**

```python
def retrieve(self, context, query: str, *, top_k: int = 6) -> list[MemoryRecord]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    records = self.list(context)                      # ① 先做权限过滤（SQL）
    if not records:
        return []
    now = utc_now()
    ranked = sorted(records, key=lambda r: self._retrieval_score(r, query, now),
                    reverse=True)[:top_k]             # ② 再对可见集排序
    self.repository.record_accesses(context.home_id, [r.id for r in ranked])
    return ranked
```

**先权限过滤，再排序**。`list()`（`service.py:140-145`）走的是 6.4.5 那条 SQL，无权访问的行根本不会进入排序候选集。不能反过来全库召回再指望模型忽略——那不是权限控制。

最后一行 `record_accesses` 让被选中的记录 `access_count + 1`，喂给下一次检索的 frequency 项，形成正反馈。默认 `top_k` 为 6，由 `CHECKPOINT_RETRIEVAL_TOP_K` 配置（`src/config.py`，`Field(default=6, ge=1, le=20)`）。

**为什么不用向量检索：`src/memory/service.py:131-138`**

```python
def evaluate_vector_retrieval(self, home_id=None, *, threshold: int = 500) -> dict:
    count = self.repository.active_count(home_id)
    return {
        "active_memory_count": count,
        "threshold": threshold,
        "recommend_vector_retrieval": count >= threshold,
        "reason": "memory_scale_threshold_reached" if count >= threshold
                  else "structured_filters_sufficient",
    }
```

它只**返回建议**，不会自动下载模型或起服务。家庭场景的记忆量通常是几十条，结构化过滤加可解释打分完全够用；上向量库要付出依赖、索引维护和“为什么召回了这条”难以解释的代价。500 条是给出建议的阈值，不是自动切换的开关。

#### 6.4.8 更新、冲突与历史版本

用户偏好会改变。例如用户先说喜欢暖光，后来改成冷白光。如果直接覆盖数据库里的 JSON，系统就只剩最新结果，无法回答“什么时候变的”。

因此每条记忆都带版本号和有效区间：

```text
版本 1：暖光
valid_from = 2026-07-01
valid_to   = 2026-08-01

版本 2：冷白光
valid_from = 2026-08-01
valid_to   = NULL        ← 当前仍然有效
```

**同一条记忆如何被识别：`src/memory/repository.py:56-60`**

```sql
CREATE UNIQUE INDEX IF NOT EXISTS uq_memories_business_key
    ON memories(
        home_id, COALESCE(user_id, ''), COALESCE(room_id, ''),
        COALESCE(device_id, ''), scope, memory_type, memory_key
    );
```

七个字段构成业务唯一键。`COALESCE(x, '')` 是必需的：SQLite 的唯一索引里 **`NULL` 互不相等**，不做转换的话共享记忆（`user_id IS NULL`）会被允许重复插入无数条。

**版本推进：`src/memory/repository.py:149-199`（`upsert`）**

```python
existing = self.find_by_key(...)
memory_id = str(uuid.uuid4())
version = existing.version + 1 if existing else 1
if existing:
    memory_id = existing.id                       # 复用同一 id，不新建行
    self.connection.execute(
        "UPDATE memory_versions SET valid_to=? WHERE memory_id=? AND version=?",
        (valid_from, existing.id, existing.version),
    )                                             # 关闭上一版本的区间
```

关键在 `memory_id = existing.id`：更新走的是同一行，`memories` 表里永远只有一条**当前**记录，历史沉到 `memory_versions`。旧版本的 `valid_to` 被设为新版本的 `valid_from`，两段区间首尾相接、不留空隙。

```sql
-- src/memory/repository.py:179-189
ON CONFLICT DO UPDATE SET
    memory_value=excluded.memory_value,
    ...
    status='active',          -- 已删除的记忆再次写入会复活
    valid_to=NULL,            -- 新版本尚未结束
    version=excluded.version
```

`status='active'` 让“删掉后又重新保存同一 key”正常工作，不会留下一条 `deleted` 挡住唯一索引。写完主表后 `_save_version()`（`repository.py:201-212`）落一条版本快照，`UNIQUE(memory_id, version)` 保证同一版本不会重复记录。

**冲突合并：`src/memory/service.py:266-274`**

```python
def _merge_values(previous: dict, incoming: dict) -> dict:
    """Preserve complementary fields while explicit incoming fields take precedence."""
    merged = dict(previous)
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_values(merged[key], value)      # 嵌套字典递归合并
        else:
            merged[key] = value                                  # 新值覆盖
    return merged
```

递归合并而不是整体替换。已有 `{"color": "暖光", "brightness": 60}`，新值只给 `{"color": "冷白光"}`，结果保留 `brightness=60`——用户只说改颜色，不该顺带丢掉亮度。

合并结果决定冲突的记录方式（`service.py:52-56`）：

```python
merged = _merge_values(existing.memory_value, item.memory_value)
if existing.memory_value != item.memory_value:
    resolution = "merged" if merged != item.memory_value else "incoming_wins"
    self.repository.add_conflict(existing, item.memory_value, merged, resolution)
```

`merged != item.memory_value` 说明旧值贡献了字段，记为 `merged`；否则新值完全覆盖，记为 `incoming_wins`。`memory_conflicts` 表同时保存 previous / incoming / resolved 三份值（`repository.py:332-344`），事后可以完整复盘一次覆盖到底发生了什么，而不是只看到结果。

**删除是逻辑关闭：`src/memory/repository.py:434-446`**

```python
def delete(self, memory_id: str, home_id: str) -> bool:
    now = utc_now().isoformat()
    cursor = self.connection.execute(
        """UPDATE memories SET status='deleted', updated_at=?, valid_to=?
           WHERE id=? AND home_id=? AND status='active'""",
        (now, now, memory_id, home_id),
    )
    self.connection.execute(
        "UPDATE memory_versions SET valid_to=? WHERE memory_id=? AND valid_to IS NULL",
        (now, memory_id),
    )
```

没有 `DELETE FROM`。改状态 + 关闭区间，`memory_versions` 里的历史一条不少。`WHERE ... status='active'` 让重复删除返回 `False`（`rowcount == 0`）而不是假装成功。过期清理 `cleanup_expired()`（`repository.py:467-492`）用同一套逻辑，只是状态写 `'expired'`；它在每次 `list_accessible` 开头被调用（`repository.py:380`），所以过期记忆不需要定时任务就会自动退出检索。

**置信度随时间衰减：`src/memory/repository.py:356-363`**

```sql
UPDATE memories SET confidence=MAX(?, confidence * ?), updated_at=?
WHERE status='active' AND updated_at<? AND confidence>?
```

`MemoryService.decay_stale_confidence()`（`service.py:123-129`）的默认参数是 90 天、系数 0.9、下限 0.2。`MAX(floor, ...)` 保证衰减不会归零——很久没用到只说明优先级降低，不代表这条偏好错了。`confidence>?` 让已经触底的行不再被更新，避免每次调用都白改一遍 `updated_at`。

查询历史版本用 `list_memory_versions(memory_id)` 工具，底层是 `repository.py:407-413`，按 `version` 升序返回。`MemoryService.list_versions()`（`service.py:180-182`）会先调 `self.get(...)` 做一次可见性校验——**不能通过查版本绕过权限读到别人的记忆**。

#### 6.4.9 SQLite 中保存了哪些表

长期记忆的建表语句全在 `src/memory/repository.py:26-99` 一个 `executescript` 里，五张表加四个索引：

| 表 | 定义位置 | 作用 |
|----|---------:|------|
| `memories` | `:29-51` | 当前长期记忆及其状态、分数、访问统计、版本号 |
| `preference_observations` | `:61-68` | 重复操作的原始计数（路径 ②） |
| `preference_candidates` | `:69-78` | 等待确认或已处理的候选（路径 ②③） |
| `memory_conflicts` | `:82-87` | 同一记忆收到不同值时的合并审计 |
| `memory_versions` | `:88-95` | 每次变更的历史值和有效区间 |

`memories` 与 `preference_observations` 的主键设计差别值得看一眼：

```sql
-- src/memory/repository.py:67
PRIMARY KEY (home_id, user_id, memory_key, value_fingerprint)
```

观测表用**复合主键**而不是自增 ID，因为它的语义就是“这个组合被观察了几次”，天然唯一。`value_fingerprint` 是 `json.dumps(..., sort_keys=True)` 的结果（`repository.py:249`）——`sort_keys=True` 保证 `{"a":1,"b":2}` 和 `{"b":2,"a":1}` 产生同一个指纹，否则同一个偏好会被拆成两条计数，永远攒不够 3 次。

```sql
-- src/memory/repository.py:79-81
CREATE UNIQUE INDEX IF NOT EXISTS uq_pending_candidate
    ON preference_candidates(home_id, user_id, memory_key)
    WHERE status='pending';
```

这是**部分索引**（partial index）：约束只作用于 `pending` 行。效果是同一个 key 最多有一条待确认候选（不会反复打扰用户），但历史上的 `confirmed` / `rejected` 记录可以任意累积。

四个索引各有分工：`idx_memories_home_scope` 和 `idx_memories_user` 服务 6.4.5 那条权限过滤 SQL，`uq_memories_business_key` 保证判重（6.4.8），`idx_memory_versions_memory` 服务版本回溯。

**原地迁移：`src/memory/repository.py:103-147`（`_migrate_schema`）**

```python
memory_columns = {row[1] for row in self.connection.execute("PRAGMA table_info(memories)")}
additions = {
    "importance": "REAL NOT NULL DEFAULT 0.5",
    "access_count": "INTEGER NOT NULL DEFAULT 0",
    "last_accessed_at": "TEXT",
    "valid_from": "TEXT", "valid_to": "TEXT",
    "version": "INTEGER NOT NULL DEFAULT 1",
}
for name, definition in additions.items():
    if name not in memory_columns:
        self.connection.execute(f"ALTER TABLE memories ADD COLUMN {name} {definition}")
self.connection.execute("UPDATE memories SET valid_from=COALESCE(valid_from, created_at)")
```

先用 `PRAGMA table_info` 读出现有列，缺什么补什么。新增的 `valid_from` 对老数据是 `NULL`，用 `created_at` 回填——老记忆的“生效时间”就是它的创建时间，这是唯一合理的推断。

迁移的最后一步给缺失版本快照的记忆补建 `memory_versions`（`repository.py:131-147`）：

```sql
SELECT * FROM memories m WHERE NOT EXISTS (
    SELECT 1 FROM memory_versions v WHERE v.memory_id=m.id AND v.version=m.version
)
```

`_migrate_schema` 在 `_create_schema` 内部被调用（`repository.py:100`），也就是**每次构造 `MemoryRepository` 都会跑一遍**。它设计成幂等的：所有操作都先判断“是否已经做过”，重复执行无副作用。所以升级版本不需要删库重建。

**两个数据库分开：**

```text
data/checkpoints.db   LangGraph 会话状态（由 SqliteSaver 管理表结构）
data/memories.db      结构化长期记忆（由本项目管理表结构）
```

职责不同，生命周期也不同：checkpoints 按 TTL 过期（7 天），memories 长期保留。混在一个文件里会让备份和清理策略互相牵制。

相关配置：

```dotenv
CHECKPOINT_DB_PATH=data/checkpoints.db
CHECKPOINT_LONG_TERM_DB_PATH=data/memories.db
ENABLE_LONG_TERM_MEMORY=true
CHECKPOINT_RETRIEVAL_TOP_K=6
```

`ENABLE_LONG_TERM_MEMORY=false` 时，Agent 仍能用设备工具和短期 Checkpoint，但不会创建 `MemoryRepository`，也不会抽取或注入长期记忆——对应 `graph.py:211` 那个 `if memory_service and ...` 判断。

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

**依赖方向是单向的**，看每个文件的 import 就能确认：

```text
tools/memory.py  ──> memory/service.py ──> memory/repository.py ──> memory/models.py
                                      └──> memory/extractor.py  ──> memory/models.py
                                      └──> agent/context.py
```

- `repository.py` 只 import `models`（`repository.py:11-14`），**没有** import `service`。
- `extractor.py` 只 import `models`（`extractor.py:11`），既不认识 Repository 也不认识 Service。
- `service.py` 同时依赖 `repository`、`extractor` 和 `agent/context`（`service.py:10-13`）——它是唯一知道全局的一层。

这个方向不能反过来。一旦 Repository 开始 import Service 去查权限，两层就锁死了，Repository 再也无法脱离权限体系单测。

调用链：

```text
用户消息
   ↓
LangGraph.sync_context（graph.py:211-228）
   ├─> MemoryService.extract_candidates_from_text ──> extractor 抽取 ──> Repository 存候选
   └─> MemoryService.retrieve ──> Repository.list_accessible（SQL 权限过滤）
                                        ↓
                                  Service 打分排序 Top-K
                                        ↓
                    格式化写入 memory_context ──> 拼进系统提示词（graph.py:536）
                                        ↓
                    LLM 根据当前消息 + 摘要 + 相关长期记忆决策
```

四条边界，各自对应代码里的具体做法：

| 边界 | 含义 | 代码证据 |
|------|------|----------|
| Repository 只管读写 | 不判断权限，不做业务决策 | 权限体现为 SQL 的 `WHERE`（`repository.py:385-387`），而非 Python 的 `if` |
| Service 管“允许吗”和“怎么处理” | 所有方法开头 `self.spaces.validate(context)` | `service.py:33, 64, 81, 93, 97, 118, 141` |
| Extractor 不碰数据库 | 返回纯数据对象 | 返回 `list[ExtractedMemoryCandidate]`，入库在 `service.py:84-89` |
| Tool 不信任模型给的身份 | 身份只从 `RunnableConfig` 取 | `tools/memory.py:22-31` 的 `_context()` |

分层带来的直接好处是可测性：测 Extractor 不需要数据库，测 Repository 不需要构造权限上下文，测 Service 的权限逻辑不需要跑 LLM。

#### 6.4.11 可供 Agent 调用的记忆工具

九个工具全部定义在 `src/tools/memory.py`：

| 工具 | 定义位置 | 用途 |
|------|---------:|------|
| `save_personal_memory` | `:52-72` | 保存用户明确要求记住的个人偏好 |
| `save_home_rule` | `:75-97` | 管理员保存家庭共享规则 |
| `list_personal_memories` | `:100-109` | 查看当前上下文可访问的记忆 |
| `update_personal_memory` | `:112-122` | 修改当前用户拥有的个人记忆 |
| `delete_personal_memory` | `:125-131` | 删除当前用户拥有的记忆 |
| `list_preference_candidates` | `:134-144` | 查看等待确认的偏好候选 |
| `confirm_preference_candidate` | `:147-153` | 确认候选并生成正式记忆 |
| `reject_preference_candidate` | `:156-163` | 拒绝候选，不写入正式记忆 |
| `list_memory_versions` | `:166-181` | 查看一条记忆的历史版本 |

**身份注入：`src/tools/memory.py:22-36`**

这是整个工具层最关键的十几行：

```python
def _context(config: RunnableConfig) -> AgentContext:
    configurable = config.get("configurable", {})
    return AgentContext(
        home_id=configurable["home_id"],          # 中括号，缺了就 KeyError
        user_id=configurable["user_id"],
        session_id=configurable["thread_id"],
        client_id=configurable["client_id"],
        room_id=configurable.get("room_id"),      # 可选，用 .get()
        device_id=configurable.get("device_id"),
    )

def _is_admin(config: RunnableConfig) -> bool:
    """Read authorization only from trusted server-side configuration."""
    return config.get("configurable", {}).get("is_admin") is True
```

必填 ID 用 `configurable["..."]` 而非 `.get()`：**没有身份就必须失败**，不能退化成匿名调用。`is_admin` 用 `is True` 严格比较，`"true"`、`1`、`"admin"` 这些值一律不算管理员。

`RunnableConfig` 是 LangChain 注入的参数——写在工具签名里，LangChain 会自动传入，但**不会出现在生成给 LLM 的 JSON Schema 中**。模型看到的 `save_personal_memory` 只有 `memory_key`、`memory_value`、`source` 三个参数，根本没有位置去指定 `home_id`。

**看一个完整工具：`src/tools/memory.py:52-72`**

```python
@tool
def save_personal_memory(
    memory_key: str, memory_value: dict[str, Any], source: str, config: RunnableConfig,
) -> str:
    """Save an explicitly requested personal preference. Identity is injected by the server."""
    if _service is None:
        return "长期记忆未启用"                    # 未启用时返回提示，不抛异常
    context = _context(config)
    record = _service.save(context, MemoryWrite(
        scope=MemoryScope.USER,                    # 写死
        memory_type=MemoryType.PREFERENCE,         # 写死
        memory_key=memory_key,
        memory_value=memory_value,
        room_id=context.room_id,                   # 来自 config，不是模型
        device_id=context.device_id,
        source=source,
    ))
    return f"已保存个人记忆 {record.memory_key}（id={record.id}）"
```

`scope` 和 `memory_type` 写死为 `USER` / `PREFERENCE`。模型**不能**通过这个工具写家庭规则——那要走 `save_home_rule`，而后者会带上 `is_admin=_is_admin(config)`（`memory.py:95`）交给 Service 校验。两个工具分开，权限差异就体现在工具本身，而不是靠一个模型可控的参数来区分。

九个工具开头都有同一句 `if _service is None: return "长期记忆未启用"`。返回文本而不是抛异常，是因为这是**配置状态**而非错误：`ENABLE_LONG_TERM_MEMORY=false` 时模型该知道这条路走不通，然后换别的方式回答。

**路径 ② 的触发点：`src/tools/memory.py:39-49`**

```python
def record_preference_operation(
    config: RunnableConfig, device_id: str, memory_key: str, memory_value: dict[str, Any],
) -> None:
    """Record a successful device setting without exposing identity to the model."""
    if _service is None:
        return
    context = _context(config).model_copy(update={"room_id": None, "device_id": device_id})
    _service.record_operation(context, memory_key, memory_value)
```

注意它**没有 `@tool` 装饰器**——这是给设备控制工具在成功后内部调用的，模型看不见也调不到。重复操作统计必须来自真实执行结果，如果做成工具，模型就能凭空"刷"出候选。

`room_id` 被显式置空、`device_id` 用真实操作的设备覆盖：偏好统计要归到**实际被控制的那台设备**，而不是 App 当前所在的房间。

**绕过 LLM 直接调用 Service**

业务代码（如设置页面）不需要经过模型：

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

权限校验在 Service 层，不在工具层。所以绕过工具直连 Service **不会绕过权限**——`context` 依然要通过 `SpaceDirectory.validate()`，共享作用域依然要求 `is_admin`。工具层只负责“别让模型碰身份”，真正的守卫在下面一层。

#### 6.4.12 如何验证记忆功能

先使用项目已配置的 `langgraph` 环境，不要新建环境或自行安装包：

```powershell
$python = 'F:\Software\Anaconda\envs\langgraph\python.exe'
& $python -m pytest -q
& $python -m compileall -q src tests
& $python -m pip check
```

> ⚠️ 直接敲 `python -m pytest` 很可能命中 base 环境，报 `ModuleNotFoundError: No module named 'loguru'`。这不是代码问题，是解释器选错了。用上面的完整路径。

**记忆相关的测试分布在三个文件里**，每条断言都对应前面某一节：

| 测试文件 | 覆盖的章节 | 关键用例 |
|----------|-----------|----------|
| `tests/test_phase_two.py` | 6.4.5 作用域与权限 | 13 个用例，见下表 |
| `tests/test_phase_three.py` | 6.4.2 / 6.4.3 短期记忆与压缩 | 6 个用例 |
| `tests/test_phase_four.py` | 6.4.6 候选机制 | 9 个用例 |
| `tests/test_phase_five.py` | 6.4.7 / 6.4.8 检索与版本 | 5 个用例 |

单独跑记忆相关的部分：

```powershell
& $python -m pytest tests/test_phase_two.py tests/test_phase_three.py `
                    tests/test_phase_four.py tests/test_phase_five.py -q
```

**权限边界：`tests/test_phase_two.py`**

```python
test_homes_and_users_are_isolated                          # :130
test_shared_rules_are_visible_but_personal_preferences_are_private  # :143
test_shared_memory_write_requires_admin                    # :159
test_room_device_and_personal_combinations_are_filtered    # :171
test_invalid_scope_combinations_are_rejected               # :231
test_view_update_and_delete_respect_ownership              # :254
test_tools_use_identity_from_runnable_config               # :286
test_home_rule_tool_requires_trusted_admin_flag            # :316
```

这一组是**安全断言**，不是功能断言。`test_tools_use_identity_from_runnable_config` 验证的是 6.4.11 那个设计：即使模型试图传身份也无效。`test_upsert_survives_repository_restart`（`:116`）单独验证数据真的落盘了，不是只活在内存里。

**候选机制：`tests/test_phase_four.py`**

```python
test_repeated_operations_create_candidate_but_not_memory   # :39  —— 3 次只出候选
test_successful_device_settings_feed_candidate_observations # :53  —— 只统计成功操作
test_confirm_candidate_saves_memory_and_reject_does_not    # :71
test_candidates_are_isolated_by_user_and_tools_require_trusted_context  # :87
test_conflicts_are_recorded_and_complementary_values_are_merged  # :105 —— 对应 _merge_values
test_stale_confidence_decays_with_floor                    # :124 —— 验证下限 0.2
test_vector_retrieval_is_only_recommended_after_scale_threshold  # :138
```

第一个用例的名字就是 6.4.6 的核心约定：**create candidate but not memory**。

**检索与版本：`tests/test_phase_five.py`**

```python
test_natural_language_extractor_is_conservative            # :34 —— "今天有点冷"不入库
test_extracted_text_only_creates_pending_candidate         # :40
test_hybrid_retrieval_uses_top_k_and_tracks_access_count   # :48 —— Top-K + access_count
test_updates_create_version_and_close_previous_validity    # :60 —— valid_to 被关闭
test_graph_injects_only_relevant_top_k_memory_and_extracts_candidate  # :72
```

最后一个是**端到端**用例：跑真实的图，断言无关记忆没有被注入提示词。

**短期记忆：`tests/test_phase_three.py`**

```python
test_message_and_token_window_is_bounded_with_rolling_summary  # :23
test_tool_results_are_trimmed_and_checkpoint_update_removes_old_messages  # :32
test_graph_persists_a_bounded_recent_window_and_context_statistics  # :52
test_checkpoint_cleanup_deletes_only_expired_threads        # :92 —— 注意 "only"
test_session_end_removes_memory_checkpoint                 # :119
test_expired_long_term_memories_are_cleaned_globally       # :139
```

`deletes_only_expired_threads` 里的 `only` 是重点：清理逻辑要删对，更要**不删错**。只断言"过期的被删了"会漏掉把活跃会话一起删掉的 bug。

**手工验证跨会话记忆**

```powershell
& $python -m src.main --home-id demo-home --user-id user-001 --session-id s1
# 输入：以后我睡觉时空调设为 25 度
# 输入：/quit

& $python -m src.main --home-id demo-home --user-id user-001 --session-id s2
# 输入：我要睡了 —— 应当体现 25 度
```

换 `--session-id` 意味着新 `thread_id`、空消息历史，但长期记忆按身份重新检索，所以偏好仍然生效。这正好把 6.4.1 那张两层表验证了一遍。

只验证数据库文件存在远远不够。可靠的测试断言的是**业务行为和权限边界**，而不是肉眼看表内容——上面每个用例名都能读成一句可判真假的陈述，这是有意为之。

#### 6.4.13 常见问题

**Q：新会话为什么还能知道我的偏好？**
A：新 `thread_id` 不继承旧消息，但 `sync_context` 会按当前可信身份重新检索 SQLite 长期记忆（`graph.py:211-228`）。短期记忆按会话隔离，长期记忆按**身份**检索——这是 6.4.1 两层设计的直接结果。

**Q：Checkpoint 会保存模拟设备的实时状态吗？**
A：不会。Checkpoint 保存的是 `AgentState`（`src/agent/state.py`）。设备实时状态由 `SimulatorBackend` 管理，重启后是否保留取决于设备后端本身。这也是 6.4.4 强调"实时状态不进长期记忆"的原因：状态该问设备，不该问记忆。

**Q：说一次"空调调到 25 度"会被永久记住吗？**
A：不会。它首先是一次设备操作。`record_operation()` 会累计观测，但 `count < 3` 时直接返回 `None`（`service.py:69-70`）；到 3 次也只生成候选，仍需确认。

**Q：那我说"以后空调都开 25 度"呢？**
A：这句含稳定标记"以后都"，抽取器会产出 `ac.temperature` 候选（`extractor.py:14`、`:27`）——**仍然是候选**。但如果你调用的是"请记住"这类明确指令并触发 `save_personal_memory` 工具，就走 6.4.6 的路径 ①，直接写库。区别在于是否有明确的保存指令，而不是措辞的强弱。

**Q：为什么候选不直接写入，明确保存却可以？**
A：候选是系统**猜**的——自然语言有歧义，重复行为也可能只是短期需求，猜错的代价由用户承担（要主动发现并删除）。明确保存是用户**说**的，意图已经确定，再确认一次纯属多余。所以 `save()` 里没有确认步骤，只有权限校验（`service.py:25-57`）。

**Q：确认候选会不会被提升成全家规则？**
A：不会。`confirm_candidate()` 把 `scope` 和 `memory_type` 写死为 `USER` / `PREFERENCE`（`service.py:102-103`），不读候选里的值。系统猜出来的东西永远只能成为个人偏好；全家规则必须由管理员显式创建。

**Q：修改偏好后旧值去哪了？**
A：`memories` 表里的行被原地更新（复用同一 `id`），`version + 1`；旧值以快照形式留在 `memory_versions`，并把 `valid_to` 设为新版本的 `valid_from`（`repository.py:158-163`）。用 `list_memory_versions` 可以完整回看。

**Q：删除记忆是真的删了吗？**
A：不是。`delete()` 只把 `status` 改成 `'deleted'` 并关闭有效区间（`repository.py:434-446`），`memory_versions` 里的历史一条不少。检索走 `status='active'` 过滤，所以用户感知上确实消失了。要物理删除需要另外的运维操作。

**Q：数据库会不会无限增长？**
A：会话消息由压缩（6.4.3）和 TTL（`CHECKPOINT_SESSION_TTL_HOURS`）控制；长期记忆有过期清理、逻辑删除和唯一索引判重——同一个 key 不会堆出多行。但 `memory_versions`、`memory_conflicts`、`preference_observations` 是**只增不减**的审计表。生产环境应增加定期归档、备份、物理删除和容量监控。

**Q：`cleanup_expired_checkpoints` 会自动跑吗？**
A：不会。它是普通函数（`store.py:89-117`），需要调用方主动触发。长期记忆的 `cleanup_expired` 不同，它挂在每次 `list_accessible` 开头（`repository.py:380`），所以过期记忆会自动退出检索。

**Q：当前是否使用向量数据库？**
A：没有。用的是结构化权限过滤 + 可解释的五项加权排序（6.4.7）。好处是零额外依赖、每条记忆的得分都能拆开解释、测试可以直接断言顺序。`evaluate_vector_retrieval()` 会在活跃记忆达到 500 条时**建议**考虑向量召回，但不会自动启用任何服务。

**Q：为什么不用 LLM 来抽取偏好，正则不是太弱了吗？**
A：正则确实召回低，但它**确定、可测、零成本**——每轮对话都跑一次，用 LLM 抽取意味着每轮多一次调用和一次不可预测的输出。而且 6.4.6 的取舍是漏抽比错记好，正则的高精度倾向正好对上。真要提升召回，可以在候选生成后加一层 LLM 复核，而不是替换掉这一层。

#### 6.4.14 通俗总结：把整个记忆模块说成一件事

前面十三节把每个零件都拆开讲过了。这一节不引入新东西，只把它们串成一个普通人能听懂的故事——如果你只读一节，读这节。

**一句话概括**

> 大模型每次醒来都失忆。这个模块的全部工作，就是在它开口之前，把"刚才聊了什么"和"这个人长期是什么样"这两份材料准备好递过去。

**打个比方：新来的管家**

想象你家请了一位管家，他能力很强，但有个毛病——**每天下班后会忘掉当天所有事**。为了让他能正常工作，你得给他配两样东西：

| 东西 | 对应本项目 | 解决什么 |
|------|-----------|---------|
| **手边的便签本**，记着今天这一趟的来龙去脉，下班就撕掉 | 短期会话记忆（Checkpoint，`data/checkpoints.db`） | "刚才你说的调暗一点，是指哪盏灯？" |
| **抽屉里的住户手册**，写着这家人长期的习惯和规矩，一直留着 | 长期结构化记忆（`data/memories.db`） | "这家男主人一向喜欢暖光" |

这两样东西的差别不在于存多久，而在于**取的方式**：便签本按"这一趟"取（同一个 `session_id` / `thread_id`），住户手册按"这个人是谁"取（`home_id` + `user_id`）。所以你换个会话重新进来，便签本是空的，但手册照样翻得到——这就是 6.4.1 讲的两层，也是 6.4.13 第一问的答案。

**便签本会写满：为什么要压缩**

一趟活干久了，便签本会写满。管家不可能每说一句话都把整本从头念一遍——费时间，而且真正要紧的就最近几条。

所以每次开口前会做一次整理（`compact_context` 节点，`graph.py:247`）：

```text
最近 12 条 → 原样留着（细节要准）
更早的       → 每条压成一句话，并进"摘要"（大意留着就行）
超长的工具结果 → 截断，加一句"（已裁剪）"
```

管家最终看到的是「一段摘要 + 最近几条原话」，而不是全本流水账。好处很实在：数据库不会无限膨胀，每次调用模型的费用也被卡住了上限。代价也很实在：**被压缩掉的细节真的没了**，只剩摘要里那一句概括。所以压缩窗口开多大，本质上是在花钱买记性——这就是 6.4.3 讲的取舍。

**住户手册怎么写进去：三条路，两种态度**

这是整个模块最值得理解的地方。系统对"要不要记住一件事"，态度取决于**是谁提出的**：

```text
① 你亲口说"以后都按这个来"  ──────────────────> 直接写进手册
② 同一个操作你重复做了 3 次  ──┐
                               ├─> 先写在"待办便签"上 ──> 问过你 ──> 写进手册
③ 聊天时被系统听出苗头      ──┘                    └─> 你说不用 ──> 丢掉
```

路径 ① 不问，路径 ②③ 一定要问。理由很朴素：**你说的算数，系统猜的不算数**。猜错了要你自己去发现、去删除，成本落在你身上；而你都已经明说了"以后按这个来"，再弹个框问"确定吗"就是烦人。

而且系统猜的东西有个天花板：候选被确认后，只能变成"**你个人的偏好**"，永远变不成"**全家的规矩**"（`service.py:102-103` 把作用域写死了）。想立全家规矩，必须管理员显式来定。系统再有把握，也没资格替全家做决定。

至于路径 ③ 的"听出苗头"，用的是**正则匹配**（`extractor.py`），不是让模型去理解。规则很死板：句子里必须出现"我喜欢/我习惯/以后都"这类稳定信号，而且不能出现"今天/这次/暂时"这类临时信号。所以"我今天想把空调开 26 度"绝对抽不出来——**宁可漏记，也不错记**。

**手册怎么翻开：不是全塞给管家**

住户手册可能有几百条，全念一遍既浪费又干扰判断。所以每轮对话开始时（`sync_context` 节点），系统只挑最相关的 6 条塞进提示词，挑选分两步：

第一步，**SQL 层面先筛掉不该看的**。别人的私人偏好、别的房间的规则，压根不会被查出来——权限写在 `WHERE` 里，不是查出来再用 `if` 过滤掉。这个区别很关键：过滤发生在数据离开数据库之前，代码里根本没机会失误。

第二步，**给剩下的打分排序**，五个因素加权：

| 因素 | 权重 | 通俗说法 |
|------|-----:|---------|
| 相关性 | 0.45 | 跟这句话有关系吗 |
| 置信度 | 0.20 | 这条有多可靠 |
| 重要性 | 0.20 | 这条有多要紧 |
| 新鲜度 | 0.10 | 是不是最近才更新的 |
| 使用频率 | 0.05 | 平时常不常用上 |

没有用向量数据库。用的是"关键词命中 + 手写权重"，土，但**每一条为什么排在前面都能拆开算给你看**，测试里可以直接断言顺序。记忆条数真到 500 x条以上时，系统会**建议**你考虑上向量检索，但不会自己偷偷启用（6.4.7）。

**改了和删了：其实什么都没丢**

手册里的条目改动时，旧值不是被覆盖掉，而是留了一份快照进 `memory_versions`，条目本身版本号加一。删除也不是真删，只是把状态标成 `deleted`，检索时过滤掉——**你感觉它消失了，审计记录里它还在**。

这个设计的用处在出问题的时候才显出来：用户说"我明明设的是暖光，怎么变冷白了"，你能把这条记忆的每一次变更、每一次的来源翻出来对账。代价是那几张审计表只增不减，生产环境得自己加归档和容量监控（6.4.13 最后几问）。

**权限：三道锁**

记忆里存的是个人习惯，串了就是隐私事故。所以有三道独立的锁：

1. **身份不听模型的**。`home_id` / `user_id` 只从 `RunnableConfig` 取（`tools/memory.py` 的 `_context()`），用户在对话里说"我是管理员"没有任何效果——那只是一段文本。
2. **看不见就是看不见**。跨用户、跨房间的隔离做在 SQL 的 `WHERE` 里。
3. **写共享的要管理员**。个人偏好自己就能存；房间规则、全家约束必须管理员。

**为什么拆成六个文件**

一句话：**让每一层都能单独测**。

```text
models.py      记忆长什么样        —— 只有数据定义
repository.py  怎么存怎么取        —— 只管读写，不判断权限
service.py     允不允许、怎么处理  —— 唯一知道全局的一层
extractor.py   从话里听出苗头      —— 碰不到数据库
summarizer.py  便签本怎么压缩      —— 纯函数，没有副作用
store.py       便签本本身          —— Checkpointer 的创建和清理
```

依赖方向严格单向：上面的认识下面的，下面的绝不回头认识上面的。一旦 `repository.py` 为了查权限去 import `service.py`，两层就锁死了，从此测存取必须先构造一整套权限上下文。现在的好处是：测抽取不用开数据库，测存取不用造用户身份，测权限不用调模型。

**最后：这个模块的一贯态度**

回头看，六个文件、十几个函数，背后其实是同一套判断反复出现：

- **能确定的，绝不去猜**——正则而不是模型抽取，SQL 权限而不是 Python 判断，手写权重而不是黑盒向量。
- **要猜的，一定问过再算数**——候选机制，而且系统猜的永远只能是个人偏好。
- **删掉的，留一份底**——版本快照、逻辑删除、冲突记录。
- **说好持久化，就必须真持久化**——配了 SQLite 路径却打不开，宁可启动直接报错，也不悄悄退回内存模式让你以为存上了（`store.py:73-76`）。

这几条不是记忆模块独有的，它们是整个项目在**面对"不确定"时的一致选择**：宁可能力弱一点，也要行为可预测。

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

**先说清楚它解决什么问题**

没有路由的时候，图只有一条路：不管用户说什么，都塞给同一个 agent 节点，由模型自己面对全部工具做决定。这条路能跑通，但有三件事做不了：

- 一句"客厅、卧室、书房的空调各是多少度"，模型只能一个一个查，串行三次；
- 一句"帮我把灯打开、空调调到 25 度、窗帘拉上"，模型可能漏掉其中一步，而且没人检查它做没做到；
- 一句"这个故障码 E3 什么意思"，模型会凭记忆编，因为它手里没有说明书。

这些请求本质上需要**不同的处理流程**，不是换个提示词就能解决的。于是加一个岔路口：先花一次判断确定这句话属于哪一类，再决定走哪条流程。这个岔路口就是 `task_router`（`graph.py:272`）。

**它在图里的位置**

```text
sync_context      ← 确定说的是哪个房间/设备，检索长期记忆
  ↓
memory_reasoner   ← 判断哪些记忆用得上
  ↓
task_router       ← 岔路口在这里
  ↓
  ├─→ planner              多步骤任务，先规划再逐步执行验证
  ├─→ knowledge_rag        设备知识问答，查本地文档
  ├─→ device_query_subgraph 多设备查询，并行发出
  ├─→ clarification        信息不够，直接反问
  └─→ compact_context      其余全部走原来的 ReAct 环
```

注意最后一条：**路由不是取代 ReAct，而是在它前面加了几个专用出口**。大部分日常请求（开灯、关空调、聊天）仍然走 `compact_context → agent → tools` 那条老路。路由只把"老路处理不好"的那几类请求摘出去。

**第一步：判断意图**

分类结果不是让模型自由发挥再拿代码去解析文本，而是用结构化输出锁死格式（`routing.py:22-25`）：

```python
Intent = Literal[
    "device_query",       # 查状态、温度、开关
    "device_control",     # 控制设备，但不属于预定义场景
    "scene_control",      # 启用/取消预定义场景
    "memory_management",  # 记住、修改、删除偏好
    "device_knowledge",   # 说明书、故障码、维护方法
    "general_chat",       # 闲聊、一般问题
    "clarification",      # 信息不足，无法安全判断
]

class IntentResult(BaseModel):
    intent: Intent
    confidence: float = Field(ge=0, le=1)   # 0-1，越低越不确定
    reason: str = ""                        # 为什么这么判，方便排查
```

`Literal` 保证 `intent` 只能是这七个值之一，`ge=0, le=1` 保证 `confidence` 落在合法区间——模型返回别的东西会当场校验失败，而不是带着脏数据往下走。多出来的 `reason` 字段不参与任何逻辑，纯粹是给人看的：路由判错时，你能直接看到它当时是怎么想的。

**第二步：LLM 判不了就用关键词**

`classify_intent`（`routing.py:68`）的实现只有几行，但结构值得注意：

```python
def classify_intent(llm, text: str) -> IntentResult:
    """Use structured LLM output when available; never let routing break the agent."""
    fallback = classify_intent_fallback(text)     # ① 先把兜底算出来
    try:
        structured = llm.with_structured_output(IntentResult)
        result = structured.invoke(intent_router_prompt(text))
        return result if isinstance(result, IntentResult) else IntentResult.model_validate(result)
    except Exception:
        return fallback                           # ② 出任何问题都退回兜底
```

docstring 那句 `never let routing break the agent` 是整段代码的目的：**路由是辅助功能，它自己坏了不能把整个 Agent 拖下水**。API 超时、模型输出不合 schema、网络断了，统统退回关键词匹配继续跑。

兜底层（`classify_intent_fallback`，`routing.py:44`）是纯字符串子串匹配，没有分词、没有正则、没有模型：

```python
if any(word in value for word in memory_words):
    return IntentResult(intent="memory_management", confidence=0.92, reason="包含记忆或偏好操作词")
```

五组关键词按固定顺序依次判断，先命中先返回：

```text
记忆词 → 知识词 → 场景词 → 查询词 → 控制词 → (太短)澄清 → 通用对话
```

`confidence` 是手写常量，不是算出来的。这层"土"得很明显，但它**确定、可测、零成本**——和 6.4 记忆模块里正则抽取的取舍是同一套思路。

顺序是人工排的，有对也有错。`scene_words` 排在 `control_words` 前面是对的，所以"关闭睡眠场景"能正确进场景分支；但 `query_words` 里有"温度"且排在控制词前面，于是"把温度调到 25 度"会被判成 `device_query`——一句明确的控制指令被当成了查询。正常情况下踩不到，因为默认走 LLM；一旦模型侧静默降级，这类误判就会浮现。这是兜底方案要付的价。

**第三步：意图 ≠ 路径**

这是最容易看错的一步。上面七个意图，和图里五个出口，**不是一一对应的**。中间隔着一层判定，顺序有讲究（`graph.py:288-306`）：

```python
use_planner = planning_enabled and should_use_planner(latest_text)
intent_route = "planner" if use_planner else "react"

if not use_planner and intent.intent == "device_knowledge" and rag_enabled:
    intent_route = "knowledge_rag"
if not use_planner and intent.intent == "device_query" and should_use_parallel_query(...):
    intent_route = "parallel_query"
if not use_planner and (intent.intent == "clarification"
                        or intent.confidence < confidence_threshold
                        or memory_decision.needs_clarification):
    intent_route = "clarification"
```

三个关键点：

**① Planner 优先级最高。** 每个后续分支都带 `not use_planner` 前置条件，所以复杂多步任务不会被降级成 RAG 或澄清。而且判断多步任务用的**不是 LLM 意图**，而是独立的规则函数 `should_use_planner`（`planning.py:63`）：数一句话里有几个动作词、涉及几类设备、有没有"然后/同时/并且"这类连接词，`动作 ≥ 2 且（设备种类 ≥ 2 或有连接词）` 才成立。它还专门排除了预定义场景——"我要睡了"归场景分支，不进 Planner。

**② 意图对了还得条件成立。** `device_knowledge` 只有在 `RAG_ENABLED=true` 时才走 RAG，否则回落 ReAct；`device_query` 只有在**真的涉及 2 个以上设备**时才走并行子图（`parallel.py:38`），查单个设备走并行反而是浪费。

**③ 澄清有三个触发口。** 模型明确判为 `clarification`、置信度低于阈值（默认 0.6）、或者记忆推理阶段认为需要追问——任一成立就反问用户。第二个口是关键：**模型说不准的时候，宁可多问一句，也不要猜着去动设备。**

**第四步：条件边**

前面算出的 `intent_route` 存进 state，`route_task` 只是把它翻译成节点名（`graph.py:628-649`）：

```python
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

workflow.add_conditional_edges("task_router", route_task, {
    "planner": "planner",
    "compact_context": "compact_context",
    "clarification": "clarification",
    "device_query_subgraph": "device_query_subgraph",
    "knowledge_rag": "knowledge_rag",
})
```

**决策和跳转是分开的**：`task_router` 节点负责想，`route_task` 函数负责跳，中间靠 state 传递。这样拆的好处是路由决策会进 checkpoint——事后你能查到"这一轮为什么走了 RAG 分支"，而不是只看到结果。

返回值标注成 `Literal` 也不是摆设：这几个字符串必须和 `add_conditional_edges` 的映射键对上，标了之后写错节点名类型检查器当场报错，不用等运行到那条边才抛异常；同时 LangGraph 会读这个标注来推断图的边，画出来的 mermaid 图才有正确的分支（见 `tests/visualize_graph.ipynb`）。

**这个项目到底是哪种设计**

回到最初的对比：

| 设计 | 优点 | 适合研究的问题 |
| --- | --- | --- |
| 单 Agent 自主选工具 | 图简单，模型自由度高 | ReAct、工具选择、提示词设计 |
| Router + 专用分支 | 路径清晰，状态容易约束 | 结构化输出、条件边、模块边界 |

**本项目两种都有，是"外层 Router 分流 + 内层 ReAct"的混合结构，骨架是 Router 式的。**

- **外层**：`task_router` 决定走哪条业务流程，路径清晰、状态可约束——只有 planner 分支才有 `plan`、`step_retry_count` 这些字段。
- **内层**：落到 ReAct 分支后，agent 节点面对完整工具列表，调哪个工具、调几次全由模型决定。**Router 只选路径，不选工具**（`routing.py:3-4` 的注释明确写了这一点）。
- **中间还有一层**：`MULTI_AGENT_ENABLED=true` 时，Router 会把意图映射成角色（`multi_agent.py:9-18`），agent 节点按角色换一组**预绑定的工具**和一份角色提示词。Device Agent 摸不到记忆工具，Memory Agent 不得控制设备——比纯 ReAct 多了约束，比硬编码分支灵活。

**这个实验该怎么做**

不要默认节点越多越先进。加路由是有代价的：**每轮对话多一次 LLM 调用**，多一层可能判错的逻辑，图也从一条线变成了五个分支。

真正的验证方法是做对照：

```powershell
# 关掉路由，全部走 ReAct
$env:ROUTING_ENABLED = "false"
& $python -m src.main --home-id demo-home --user-id user-001

# 开启路由（默认）
$env:ROUTING_ENABLED = "true"
```

用同一组请求跑两遍，对比三件事：

| 观察项 | 怎么看 |
|--------|--------|
| 工具选择是否更准 | 日志里 `Agent 决策: 调用工具 → [...]` 的序列对不对 |
| 多花的调用值不值 | 路由多花 1 次，但并行查询可能省下 N-1 次串行 |
| 判错时是否可控 | 关键词兜底会不会把控制指令判成查询 |

结论很可能是"看情况"：单设备操作路由纯属额外开销；多设备查询和多步任务，路由带来的并行和规划能力是 ReAct 给不了的。**这个"看情况"本身就是实验结果**——你要找的不是哪种设计更好，而是哪类请求值得为它多付一次调用。

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
intent_route: Literal[
    "react", "planner", "clarification", "parallel_query", "knowledge_rag"
]
delegated_agent: Literal["device", "scene", "memory", "knowledge", "chat"]
```

正常运行时使用 `llm.with_structured_output(IntentResult)`。如果模型不支持结构化输出或调用异常，系统会回退到保守的关键词分类器，避免路由故障导致整个图不可用。低于 `ROUTING_CONFIDENCE_THRESHOLD`（默认 `0.6`）或明确分类为 `clarification` 的请求会直接询问用户补充设备、房间或动作，不会调用工具。

阶段八最初没有为每种意图创建一套 Agent，后续阶段在保持路由与执行解耦的基础上逐步增加了稳定分支。当前业务映射是：

```text
复杂多动作 device_control → Planner–Executor–Verifier
多目标 device_query        → 设备查询子图与动态并行
device_knowledge           → Agentic RAG 子图
低置信度 / clarification   → clarification 节点
其余已识别意图             → ReAct；启用多智能体时使用对应专用工具集
```

这样既保留阶段六场景确认和阶段七规划循环，也让阶段九的子图、阶段十的 Supervisor 与阶段十二的 RAG 共用同一份结构化路由结果。

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
        "control_light", "control_ac", "control_tv", "control_curtain",
        "control_humidifier"
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

模型只能生成计划，不能在 Planner 节点中执行工具。工具名也被限制为五个原子设备工具，不允许在自定义计划中嵌套 `activate_scene`。

这个 `Literal` 里没有 `read_sensor`，同样是刻意的。计划的每一步都要能被 Verifier
验证，而验证的方式是比对“执行后的状态”和“期望状态”。读取不改变任何状态，
也就没有期望状态可比——把它排成一步，Verifier 只能空转通过，等于在计划里插了一个
永远成功的步骤。传感器该出现的位置是 ReAct 循环：模型想看数据就随时读，
读完再决定计划里放什么。

这条约束在类型层面就生效了，`PlanStep(tool_name="read_sensor", ...)` 会直接被
Pydantic 拒绝，不需要靠提示词提醒模型别这么干。测试里也有一条守着它。

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
- Planner 中的有副作用控制步骤仍按顺序执行；阶段九的动态并行目前只用于相互独立、无副作用的多设备状态查询。

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

这里有一个必须说清楚的界限：Verifier 读的是**执行器自己的字段**，不是传感器读数。
上面那个例子里 `temperature: 25` 是空调的目标温度，验证它等于确认“命令写进去了”。
它不能验证“房间真的到了 25 度”——那要看温湿度传感器，而空调需要几分钟才能把
室温拉过去，当场去读只会一直 `state_mismatch`。

所以两种“验证”是分开的：

| | 验证对象 | 时效 | 谁在做 |
| --- | --- | --- | --- |
| 命令是否生效 | 执行器字段（目标值） | 立即 | `verifier_node`，确定性比对 |
| 环境是否达标 | 传感器读数（实测值） | 需要时间 | 模型在 ReAct 循环里隔一会儿再读一次 |

把第二种塞进 Verifier 会让计划因为“物理世界还没跟上”而误判失败并重试，
而重试的动作（再设一次 25 度）对结果毫无帮助。传感器该做的是给模型提供事实，
不是给计划提供通过条件——这也是 `read_sensor` 不进 `PLANNING_TOOL_NAMES` 的另一个理由。

当同一步骤重试耗尽时，Planner 会收到类似反馈：

```text
步骤 1（关闭客厅灯）失败：device state mismatch...工具结果：...
```

Planner 根据反馈生成修订版计划，`plan_revision` 加一，并再次请求用户批准。如果重新规划次数也耗尽，图会停止，不会无限循环。

目前 Reflection 主要体现在确定性的状态验证和失败反馈中。对于“环境是否舒适”“观影氛围是否合适”这类主观目标，后续可以再增加模型型 Verifier，但不应替代当前可验证设备状态检查。

#### 6.5.5 Evaluator–Optimizer：生成后评分并改进（扩展方向）

当前项目尚未增加独立的生成质量优化循环。Verifier 主要判断操作是否成功，Evaluator–Optimizer 更适合评价计划或最终回答的质量。例如场景计划生成后，可以在执行前检查：

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

当主图只有三五个节点时，把所有逻辑写在一起通常没有问题。随着确认、规划、验证、并行查询和知识检索不断加入，主图会同时承担两类职责：

- 决定请求应该进入哪条业务路径；
- 描述每条业务路径内部的所有执行步骤。

这两类职责混在一起后，主图会越来越长，也很难单独测试某一条路径。子图（Subgraph）的作用，就是把一段已经具有明确输入、输出和内部步骤的流程，封装成一张可以独立编译、调用和测试的小图。

可以把它理解成“图版本的函数”：

```text
普通函数：输入参数 → 函数内部步骤 → 返回值
子图：    输入状态 → 多个节点和边 → 输出状态
```

需要注意，LangGraph 并没有要求使用一个名为 `Subgraph` 的特殊类。通常仍然使用 `StateGraph` 构建流程，再通过 `compile()` 得到可调用的小图；当这个小图被另一张图使用时，我们才称它为子图。

##### 当前项目真正拆出的子图

阶段九只把“多设备状态查询”拆成了子图，代码位于 `src/agent/parallel.py`。它的内部结构是：

```text
dispatch
   │
   ├── 根据 targets 动态创建多个 query_device 分支
   │       ├── query_device(device_1)
   │       ├── query_device(device_2)
   │       └── query_device(device_n)
   │
   └──────────────────────────────→ aggregate → END
```

三个节点分别负责：

- `dispatch`：在查询前统一推演一次模拟环境，并准备扇出；
- `query_device`：每个分支只查询一个设备；
- `aggregate`：合并全部分支结果，排序后生成最终文本。

它使用独立的 `QueryState`，不需要知道完整 Agent 的记忆、规划、审批等状态：

```python
class QueryState(TypedDict):
    query: str
    targets: list[str]
    device_id: NotRequired[str]
    parallel_results: Annotated[list[dict], operator.add]
    response: NotRequired[str]
```

构建和编译方式与普通 LangGraph 相同：

```python
graph = StateGraph(QueryState)
graph.add_node("dispatch", dispatch_node)
graph.add_node("query_device", query_device)
graph.add_node("aggregate", aggregate)

graph.set_entry_point("dispatch")
graph.add_conditional_edges("dispatch", fan_out, ["query_device"])
graph.add_edge("query_device", "aggregate")
graph.add_edge("aggregate", END)

return graph.compile()
```

所以这里不是把一段代码简单移动到另一个文件，而是真的构建并编译了一张可以独立运行的小图。

##### 子图如何接入主图

当前项目采用“包装节点调用子图”的接入方式，而不是把编译后的子图直接注册成主图节点。真实调用链如下：

```text
用户消息
  → task_router 判断为 device_query
  → should_use_parallel_query 判断目标数量不少于 2
  → intent_route = "parallel_query"
  → 主图进入 parallel_query_node
  → parallel_query_node 调用 device_query_subgraph.invoke(...)
  → 查询结果写回 AgentState
  → END
```

主图中的注册代码是：

```python
workflow.add_node("device_query_subgraph", parallel_query_node)
```

这里注册的 `parallel_query_node` 是普通节点函数。它负责连接父图和子图：

```python
def parallel_query_node(state: AgentState) -> dict:
    latest_text = getattr(state["messages"][-1], "content", "")
    targets = extract_query_targets(latest_text, registry)

    result = device_query_subgraph.invoke({
        "query": latest_text,
        "targets": targets,
        "parallel_results": [],
    })

    return {
        "messages": [AIMessage(content=result["response"])],
        "parallel_query_results": result["parallel_results"],
    }
```

可以把 `parallel_query_node` 理解为一个“状态转换器”：

```text
父图 AgentState
  messages[-1].content
          │
          ▼
包装节点提取 query 和 targets
          │
          ▼
子图 QueryState
  query、targets、parallel_results
          │
          ▼
子图输出 response 和 parallel_results
          │
          ▼
包装节点写回父图
  messages、parallel_query_results
```

这种方式的好处是父图和子图可以使用不同的状态结构。子图只接收完成查询所需的最少字段，边界清晰，也不会意外依赖父图里的用户身份、长期记忆或规划状态。

另一种常见方式是直接把编译后的图作为节点注册：

```python
workflow.add_node("device_query_subgraph", device_query_subgraph)
```

直接注册更简洁，但通常要求父图和子图拥有能够兼容或直接映射的状态字段。当前项目的 `AgentState` 和 `QueryState` 职责差异较大，因此使用包装节点显式转换状态，更容易理解和维护。

##### 路由与子图分别解决什么问题

结构化路由和子图经常一起出现，但职责不同：

```text
结构化路由：这次请求应该走哪条业务路径？
子图：      进入这条路径后，内部应该按什么步骤执行？
```

例如“查询客厅和卧室设备状态”先由路由识别为 `device_query`，再进入设备查询子图；子图不再判断这是控制、规划还是知识问答，只专注于完成多设备查询。

因此，建议先学习 6.5.2 的结构化意图路由，再学习子图。不过 Router 并不是使用子图的技术前提：只要某段流程边界稳定，即使没有 Router，也可以先把它抽成独立子图。

##### 如何验证子图确实在运行

项目已经提供三层测试，位于 `tests/test_phase_nine.py`：

```powershell
# 运行阶段九的全部测试
python -m pytest -q tests/test_phase_nine.py
```

测试分别验证：

1. `extract_query_targets` 能正确解析房间和设备目标，并判断是否需要并行查询；
2. 查询子图可以独立 `invoke`，多个分支结果能够被合并并稳定排序；
3. 完整主图会进入 `parallel_query` 路径，并且不会调用普通 ReAct LLM。

只运行独立子图测试：

```powershell
python -m pytest -q tests/test_phase_nine.py::PhaseNineSubgraphParallelTests::test_subgraph_fanout_aggregates_sorted_results
```

如果想直接观察节点事件和每个并行分支，可以执行：

```powershell
$env:PYTHONIOENCODING = "utf-8"

@'
from src.agent.parallel import build_device_query_subgraph
from src.devices.base import DeviceRegistry
from src.devices.simulator import SimulatorBackend

registry = DeviceRegistry(SimulatorBackend())
graph = build_device_query_subgraph(registry)

inputs = {
    "query": "查询设备",
    "targets": ["bedroom_ac", "living_room_light"],
    "parallel_results": [],
}

for event in graph.stream(inputs, stream_mode="updates"):
    print(event)
'@ | python -
```

输出中应该依次出现一次 `dispatch`、多次 `query_device` 和一次 `aggregate`。例如：

```text
{'dispatch': None}
{'query_device': {'parallel_results': [{'device_id': 'bedroom_ac', ...}]}}
{'query_device': {'parallel_results': [{'device_id': 'living_room_light', ...}]}}
{'aggregate': {'response': '...'}}
```

这比只检查最终回答更有价值，因为它同时证明了子图节点顺序、动态分支和结果聚合都实际发生了。

##### 哪些内容尚未拆成子图

当前项目已经实现的是“多设备状态查询子图”，不是完整的“设备控制子图”“场景规划子图”和“记忆管理子图”。下面是未来可以继续演进的架构示意，不代表项目当前已经全部完成：

```text
主图
├── 设备控制子图（扩展方向）
│   └── 定位目标 → 能力校验 → 风险判断 → 执行 → 验证
├── 场景规划子图（扩展方向）
│   └── 生成计划 → 评价 → 执行 → 汇总
├── 记忆管理子图（扩展方向）
│   └── 抽取候选 → 校验 → 冲突判断 → 确认 → 保存
├── 多设备查询子图（当前已实现）
│   └── 准备 → 动态并行查询 → 汇总
└── 通用对话节点
```

判断一段逻辑是否值得拆成子图，可以问三个问题：

1. 它是否包含多个节点和明确的内部流程？
2. 它是否有相对稳定的输入、输出和职责边界？
3. 它是否值得脱离完整 Agent 单独测试或复用？

如果三个答案大多是“是”，通常适合拆成子图。普通的单步工具函数只有一次输入和一次输出，没有独立工作流，就不必为了使用 Subgraph 而强行包装。

#### 6.5.7 动态并行：Fan-out 与结果聚合

智能家居任务经常包含多个互不依赖的操作。例如用户要求查询客厅、卧室和书房温度，可以并行分发：

当前阶段九实现的是无副作用的多设备状态查询。查询目标在运行时从设备名称或房间名称中解析，每个目标通过 `Send("query_device", ...)` 分发；`parallel_results` 使用 `Annotated[list[dict], operator.add]` reducer 合并，最后按设备 ID 排序后生成稳定回复。

传感器接进来后，这里多了一处需要小心的地方。`extract_query_targets` 按“设备名或
房间名出现在句子里”匹配，所以“查询客厅和卧室的设备状态”会同时命中两个房间的
执行器**和**传感器——这是对的，读数本来就属于“状态”的一部分，而且读取是只读操作。
但环境推演必须放在 `dispatch` 节点，也就是扇出**之前**：

```python
def dispatch_node(state: QueryState):
    registry.tick_environment()      # 一次查询只推演一次
    return {}

def fan_out(state: QueryState):
    return [Send("query_device", {"device_id": d}) for d in state["targets"]]
```

如果把 `tick_environment()` 写进 `query_device`，每个并行分支都会推演一次，
于是“查两个房间”和“查所有设备”读到的湿度会不一样——**并行度直接改变了读数**。
这类 bug 在单设备测试里永远不会出现，只在扇出宽度变化时才显形，
是并行改造里很容易踩的一脚。

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

多智能体并不是“创建几个不同性格的聊天机器人，让它们轮流说话”。真正有价值的拆分是：每个 Agent 拥有不同的职责说明、工具权限和终止边界，由一个 Supervisor 决定本轮应该把任务交给谁。

例如，管理记忆的 Agent 不应该同时拥有开关空调的工具；普通聊天 Agent 也不应该因为一句含糊的话就能启用离家场景。多智能体首先解决的是权限和职责隔离，其次才是协作。

##### 当前项目是哪一种多智能体实现

阶段十实现的是一次有界的 Supervisor 委派：`task_router` 同时承担 Supervisor，根据结构化意图设置 `delegated_agent`；普通 ReAct 路径进入共享的 `agent` 节点后，再根据这个字段选择对应的职责提示词和工具集。

这里有一个非常容易误解的地方：Device Agent、Scene Agent、Memory Agent 等角色并不是五个独立的 LangGraph 节点。当前主图中只有一个名为 `agent` 的节点，角色是在运行时动态选择的。

```text
静态图看到的结构：

task_router → compact_context → agent → tools → compact_context → agent
                                  │                              │
                                  └──────────────→ supervisor_finalize → END

运行时 agent 节点内部：

delegated_agent == "device"    → Device 职责提示词    + 设备工具集
delegated_agent == "scene"     → Scene 职责提示词     + 场景工具集
delegated_agent == "memory"    → Memory 职责提示词    + 记忆工具集
delegated_agent == "knowledge" → Knowledge 职责提示词 + 无控制工具
delegated_agent == "chat"      → Chat 职责提示词      + 只读外部工具
```

这种设计可以理解为“共享执行节点，运行时切换能力边界”。它比为每个角色复制一套 `agent → tools → agent` 节点更紧凑，也避免五套几乎相同的循环。

##### Supervisor 如何选择角色

角色映射位于 `src/agent/multi_agent.py`：

```python
def agent_for_intent(intent: str) -> AgentRole:
    if intent in {"device_query", "device_control"}:
        return "device"
    if intent == "scene_control":
        return "scene"
    if intent == "memory_management":
        return "memory"
    if intent == "device_knowledge":
        return "knowledge"
    return "chat"
```

`task_router_node` 将分类结果和协作状态一起写入 `AgentState`：

```python
{
    "intent": "device_control",
    "intent_route": "react",
    "delegated_agent": "device",
    "handoff_count": 1,
    "collaboration_status": "delegated",
}
```

这些字段分别回答不同问题：

| 字段 | 回答的问题 | 示例 |
| --- | --- | --- |
| `intent` | 用户想做什么 | `device_control` |
| `intent_route` | 主图实际进入哪条路径 | `react`、`planner`、`parallel_query` |
| `delegated_agent` | 由哪个职责角色处理 | `device` |
| `handoff_count` | 已发生几次委派 | `1` |
| `collaboration_status` | 当前协作进行到哪里 | `delegated`、`working`、`completed` |

不要把 `intent`、`intent_route` 和 `delegated_agent` 当成同一个概念。例如“查询客厅和卧室所有设备状态”的角色仍然是 `device`，但实际路径是 `parallel_query`，请求会直接进入查询子图，而不是进入共享 `agent` 节点。

##### 工具权限如何隔离

构图时会提前为不同角色绑定不同的工具集合：

| Agent | 职责 | 实际可用工具 |
| --- | --- | --- |
| Device Agent | 单设备查询和控制 | `control_light`、`control_ac`、`control_tv`、`control_curtain`、`control_humidifier`、`read_sensor`、`get_device_status` |
| Scene Agent | 查询和启用预定义场景 | `activate_scene`、`list_scenes`、`read_sensor` |
| Memory Agent | 管理长期记忆和偏好候选 | 保存、查询、更新、删除、候选确认和版本工具 |
| Knowledge Agent | 根据设备文档回答问题 | 不绑定设备控制工具；启用 RAG 时通常直接进入 `knowledge_rag` 子图 |
| Chat Agent | 普通对话和生活信息查询 | 只绑定天气等外部只读 MCP 工具；没有外部工具时不绑定工具 |

运行 `agent_node` 时，先追加对应角色的职责提示词，再选择已经绑定好工具的模型：

```python
role = state.get("delegated_agent", "chat")
role_context = f"\n\n## 当前专用职责\n{role_prompt(role)}"
messages.insert(0, SystemMessage(
    content=system_prompt + context_prompt + role_context
))

active_llm = specialised_llms[role]
response = active_llm.invoke(messages)
```

因此，工具隔离不是只靠提示词要求模型“不要调用”。Memory Agent 实际拿不到 `control_light`，即使模型想生成该工具调用，也没有对应工具 Schema 可供选择。

##### 一次设备控制怎样走完整流程

以“打开客厅灯”为例，典型轨迹是：

```text
START
  ↓
sync_context
  ↓
memory_reasoner
  ↓
task_router / Supervisor
  ├── intent = device_control
  ├── intent_route = react
  ├── delegated_agent = device
  └── collaboration_status = delegated
  ↓
compact_context
  ↓
agent（使用 Device Agent 提示词和设备工具集）
  ├── collaboration_status = working
  └── tool_call = control_light(...)
  ↓
tools（真实执行设备工具）
  ↓
compact_context
  ↓
agent（读取 ToolMessage，生成最终回答）
  ↓
supervisor_finalize
  ├── collaboration_status = completed
  └── 检查 handoff_count 是否越界
  ↓
END
```

如果专用 Agent 不需要工具，第一次进入 `agent` 后就会生成文本，然后直接进入 `supervisor_finalize`。如果它生成工具调用，则继续使用原来的 ReAct 循环，直到生成不带工具调用的最终回答。

场景操作还可能经过 Human-in-the-loop：

```text
Scene Agent → approval
                  ├── 批准 → tools → Scene Agent → supervisor_finalize
                  └── 拒绝 → reject_tools → Scene Agent → supervisor_finalize
```

##### 并非所有请求都进入专用 Agent 节点

Supervisor 先设置角色，主图随后还会根据 `intent_route` 选择更合适的专用工作流：

| 请求示例 | `delegated_agent` | `intent_route` | 实际执行路径 |
| --- | --- | --- | --- |
| 打开客厅灯 | `device` | `react` | Device Agent → tools → finalize |
| 查询客厅灯状态 | `device` | `react` | Device Agent 查询 |
| 查询客厅和卧室设备状态 | `device` | `parallel_query` | 多设备查询子图 → END |
| 启用观影模式 | `scene` | `react` | Scene Agent → approval → tools |
| 记住我喜欢暖光 | `memory` | `react` | Memory Agent → memory tool |
| 加湿器怎么清洗 | `knowledge` | `knowledge_rag` | 知识 RAG 子图 → END |
| 你好 | `chat` | `react` | Chat Agent → finalize |
| 关闭全屋设备并把卧室调到睡眠状态 | 取决于意图分类 | `planner` | Planner → Executor → Verifier |

这张表很重要。多智能体角色表示职责归属，子图和 Planner 表示具体执行机制，两者可以同时存在。

##### Safety 为什么不是一个只会评论的 Agent

当前项目没有额外创建 Safety Agent。安全约束由确定性节点负责：

- 场景等批量副作用由 `approval` 节点暂停并等待用户确认；
- 多步骤设备操作由 Verifier 查询真实状态；
- 用户身份和家庭范围来自可信 `AgentContext`，不能由模型从文本中改写；
- Agent 的工具集合在构图时已经限制。

这种 Safety 层比增加一句“你是安全审查员，请谨慎”更可靠，因为它能够真正阻止工具执行或发现状态不一致。

##### 三层可视化：从可能路径到实际路径

要看懂多智能体流程，最好不要只依赖一张总图。可以分三层观察。

第一层是静态拓扑图，用来回答“主图可能走哪些节点”。项目中的 `tests/visualize_graph.ipynb` 已经可以绘制完整主图，也可以直接获取 Mermaid：

```python
print(graph.get_graph().draw_mermaid())
```

静态图会显示 `task_router → agent → tools → agent → supervisor_finalize`，但不会把 Device Agent 和 Memory Agent画成两个节点，因为它们共用 `agent` 节点。静态图适合看结构，不适合判断某一次请求实际选择了哪个角色。

第二层是语义诊断事件，用来回答“Supervisor 委派给了谁、Agent 做了什么、何时结束”。启动 CLI 时增加 `--trace`：

```powershell
python -m src.main --trace
```

输入“打开客厅灯”后，可以看到类似诊断信息：

```text
· supervisor_routing request=打开客厅灯 intent=device_control confidence=0.95
  intent_route=react delegated_agent=device handoff_count=1
· agent_completed role=device has_tool_calls=True tool_names=['control_light']
· agent_completed role=device has_tool_calls=False tool_names=[]
· supervisor_finalized role=device handoff_count=1 max_handoffs=2 status=completed
```

这组事件表达的是业务语义，但为了避免输出过多，它不会列出每个普通图节点。

第三层是 `updates` 节点更新流，用来回答“这一次实际经过了哪些 LangGraph 节点”。调试代码可以同时订阅 `custom` 和 `updates`：

```python
important = {
    "intent", "intent_route", "delegated_agent",
    "handoff_count", "collaboration_status",
}

for mode, chunk in graph.stream(
    payload,
    config,
    stream_mode=["custom", "updates"],
):
    if mode == "custom":
        print("事件:", chunk)
        continue

    node_name, update = next(iter(chunk.items()))
    summary = {
        key: value
        for key, value in (update or {}).items()
        if key in important
    }
    print(f"节点: {node_name:20} 状态: {summary}")
```

一次不调用工具的 Device Agent 请求可能输出：

```text
节点: sync_context         状态: {}
节点: memory_reasoner      状态: {}
节点: task_router          状态: {'intent': 'device_control',
                                  'intent_route': 'react',
                                  'delegated_agent': 'device',
                                  'handoff_count': 1,
                                  'collaboration_status': 'delegated'}
节点: compact_context      状态: {}
节点: agent                状态: {'collaboration_status': 'working'}
节点: supervisor_finalize 状态: {'collaboration_status': 'completed'}
```

如果 Agent 调用了工具，中间还会出现 `tools → compact_context → agent`。这种运行时轨迹比给静态图上的边加颜色更可靠，因为它来自本次执行真实产生的节点更新。

三层信息可以这样配合：

```text
静态 Mermaid：有哪些可能路径？
custom 事件： 每个阶段在业务上做了什么？
updates 流：  这次请求实际经过了哪些节点？
```

##### 如何测试职责隔离和运行轨迹

阶段十测试位于 `tests/test_phase_ten.py`：

```powershell
python -m pytest -q tests/test_phase_ten.py
```

测试不仅检查最终回答，还检查以下行为：

- `device_control` 是否映射到 Device Agent；
- Device Agent 是否拿不到场景和记忆工具；
- Memory Agent 是否拿不到设备控制工具；
- Chat Agent 在没有外部工具时是否为空工具集；
- `supervisor_routing` 是否记录实际意图、路径和委派角色；
- `supervisor_finalized` 是否记录协作正常结束。

只测试运行时轨迹事件：

```powershell
python -m pytest -q tests/test_phase_ten.py::PhaseTenMultiAgentTests::test_runtime_trace_exposes_delegation_and_completion
```

##### 当前实现的边界

配置项为：

```text
MULTI_AGENT_ENABLED=true
MULTI_AGENT_MAX_HANDOFFS=2
```

当前版本每轮只由 Supervisor 委派一次，并没有实现 Device Agent 再主动转交给 Safety Agent、Memory Agent 再转交给 Chat Agent 这样的连续 handoff。`handoff_count` 和 `MULTI_AGENT_MAX_HANDOFFS` 目前主要提供状态记录和终止边界，为后续真正的跨 Agent 转交预留空间。

因此，当前实现更准确的名称是“Supervisor 选择专用能力边界”，而不是“多个 Agent 自由对话”。这种受控设计更适合智能家居：路径容易测试、工具权限清晰，也更容易从运行轨迹中判断任务究竟交给了谁。

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

这里有一个容易踩的坑，它同时也是“为什么规划过程一度完全看不见”的答案：`get_stream_writer()` 只有在 `stream` 模式下才拿到真正的写入器；用 `graph.invoke()` 跑图时 LangGraph 会给节点一个空写入器，于是每一次 `emit_progress` 都被静默丢弃，不报错、不警告。CLI 早期就是用 `invoke` 消费图的，所以 Planner / Executor / Verifier 明明是三个独立节点，用户看到的却只有“已生成 N 步执行计划”和最后一句“任务已完成”。

修好它的关键不是往图里加日志，而是换消费方式并补一个渲染层：

```python
# src/main.py —— 同时要进度事件和中断，所以传一个列表
for mode, chunk in graph.stream(payload, config, stream_mode=["custom", "updates"]):
    if mode == "custom":
        view.handle(chunk)                    # 进度事件 → 终端
    elif mode == "updates":
        interrupts = chunk.get("__interrupt__")
        if interrupts:
            pending = interrupts[0].value     # 审批中断照旧工作
```

多模式 `stream` 产出的是 `(mode, chunk)` 二元组；中断以 `updates` 分支里的 `{"__interrupt__": (Interrupt(...),)}` 形式到达，所以人在回路的审批不受影响。事件已经边跑边打印，最终状态就不必再从返回值里取，直接 `graph.get_state(config).values` 读 Checkpoint 更可靠。

分层上刻意分成三处，谁都不越界：

| 位置 | 职责 | 依赖 |
| --- | --- | --- |
| `src/agent/observability.py` | 声明事件名（`PLANNING_EVENTS` / `TRACE_EVENTS`）并发出事件 | 只依赖 LangGraph |
| `src/agent/graph.py` | 在节点里按阶段发事件 | 不知道有终端 |
| `src/progress_view.py` | 把事件渲染成表格和彩色行 | 依赖 rich，不知道有图 |

事件的字段划分也照抄 Executor / Verifier 的职责边界：`step_executed` 只报告“工具说了什么”（`tool_result`），不带任何结论；`success`、`problem_type` 和“期望 vs 实测”只出现在 `step_verified` 里。这样终端上呈现的分工就不是解说词，而是数据结构本身的形状。

于是同一条多动作请求在运行时长这样（`plan_generated` 一定早于第一个 `step_started`，也就是说参数全部定下来时设备还一点没动）：

```text
🧭 Planner 分支   目标 关掉客厅灯，然后把卧室空调调到25度
📋 计划 v1（2 步 · 此刻尚未触碰任何设备）
  1  关闭客厅灯      control_light  device_name=客厅灯, action=off
  2  打开卧室空调     control_ac     device_name=卧室空调, action=on, temperature=25
▶ 计划 v1 已批准，开始逐步执行
⚙ Executor 步骤 1/2 · 关闭客厅灯
    调用 control_light(device_name=客厅灯, action=off)
    工具返回 ✅ 客厅灯已关闭。
✔ Verifier 步骤 1/2 通过 · 期望 power=False ≡ 实测 power=False
...
🏁 规划结束 · completed · 验证通过 2 次 / 共 2 次尝试 · 最终计划 v2
```

失败路径同样看得见：`↻ 重试步骤 1（第 2/2 次尝试）` 之后若仍失败，就是 `⟲ 重试额度已用尽，把失败原因交回 Planner 重新规划`，接着 v2 计划表格重新出现并再次等待审批。诊断类事件（路由、记忆判断）默认折叠，加 `--trace` 才显示，避免把规划过程淹掉。

进度事件是“流过去就没了”，所以另外提供 `/plan` 命令，从 Checkpoint 里把同一份轨迹再取出来复盘：Planner 产出一张表，Executor + Verifier 的每次尝试一行，重试和被放弃的旧版本都在里面。

可以分别观察：

- 模型 Token 流：最终自然语言回答的增量内容；
- 状态更新流：每个节点对状态的局部修改；
- 完整状态流：节点执行后的完整状态快照；
- 自定义事件：工具主动报告执行进度。

学习时应避免把模型的隐藏推理过程直接展示给用户。可展示的是任务状态、工具进度和可验证结果，而不是模型内部思维文本。上面这套事件正是按这个界线设计的：只有“第几步、调了哪个工具、期望值与实测值是否一致”，没有一个字段承载模型的推理文本。

#### 6.5.12 Agentic RAG：区分记忆、实时状态和外部知识

长期记忆保存用户偏好和家庭规则，但不适合保存设备说明书、故障代码和产品知识。阶段十二已经实现设备文档 RAG 分支：

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

当前实现使用 `docs/knowledge/catalog.json` 保存型号与 Markdown 文件的结构化映射，`src/knowledge/base.py` 使用标准库完成可解释词法检索，不需要额外安装向量数据库。`src/knowledge/rag.py` 构建如下子图：

```text
identify_device
  ↓
retrieve
  ├── 命中 → answer + citations
  ├── 未命中且可重试 → rewrite_query → retrieve
  └── 仍未命中 → refuse
```

结构化 Router 新增 `device_knowledge` 意图。知识请求进入 RAG 子图，不绑定设备控制工具。故障代码查询还会要求文档中出现完全相同的代码，例如知识库只有 E3 时，询问 E9 不会返回“相似答案”，而是明确拒答。

RAG 状态会保存：

```python
rag_status: Literal["answered", "refused"]
rag_citations: list[str]
rag_trajectory: list[dict]
rag_device_model: str | None
```

`src/evaluation/trajectory.py` 可以离线计算路由准确、回答/拒答状态、来源正确性、是否发生检索、查询改写次数和引用数量。评测关注的是整条轨迹，而不只是最终回答是否流畅。

#### 6.5.13 推荐的进阶学习顺序

下面阶段六至十二的能力目前都已进入项目，但学习时仍不建议一次全部展开。可以按照它们对 LangGraph 核心能力的依赖关系分阶段理解：

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
阶段十二：Agentic RAG 与轨迹评测（已实现）
  知识检索、来源路由、执行轨迹比较
```

这里的阶段编号是教程中的 Agent 学习路线编号，不等同于 `docs/iterations/` 文件名前缀。章节对应关系为：阶段六对应 6.5.1；阶段七对应 6.5.3 和 6.5.4；阶段八至十二依次对应 6.5.2、6.5.6–6.5.7、6.5.8、6.5.9–6.5.11 和 6.5.12。6.5.5 的独立 Evaluator–Optimizer 循环仍是扩展方向。

阶段六至阶段十二均已完成。项目现在覆盖人在回路、规划验证、结构化路由、子图并行、Supervisor 多智能体、显式记忆推理、时间旅行、进度事件以及带来源的 Agentic RAG。下一步更适合从评测数据出发补齐薄弱场景，或进入 6.6 的家居领域 SFT、LoRA 与偏好优化路线，而不是继续无目标增加节点。

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

#### 6.6.22 推荐的模型训练步骤

为避免与阶段六至十二的 Agent 开发路线重名，模型训练使用步骤字母编号：

```text
训练步骤 A：家居 SFT 数据集
  统一 Schema、人工种子、模板扩充、难例、独立评测集
        ↓
训练步骤 B：LoRA/QLoRA 监督微调
  工具调用、参数生成、澄清、结果总结、对照评测
        ↓
训练步骤 C：偏好优化
  chosen/rejected、安全偏好、DPO、困难负例
        ↓
训练步骤 D：模拟环境强化学习
  环境随机化、可验证奖励、轨迹采样、奖励投机分析
        ↓
训练步骤 E：Agent 联合评测
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

配置后，Claude Desktop 会自动发现 8 个智能家居工具：五个执行器控制（灯光、空调、
电视、窗帘、加湿器）、传感器读取、状态查询和场景激活。注意暴露给外部的也是
`read_sensor` 而不是某个传感器控制工具——只读约束在协议边界上同样成立。

### 7.3 配置本项目附带的天气 MCP

复制 `.env.example` 后，保留或填写：

```ini
EXTERNAL_MCP_SERVERS=[{"name":"weather","transport":"stdio","command":"python","args":["-m","src.mcp.weather_server"]}]
WEATHER_DEFAULT_LOCATION=杭州
CAIYUN_WEATHER_TOKEN=你的彩云 token
```

`CAIYUN_WEATHER_TOKEN` 在 <https://dashboard.caiyunapp.com> 免费领取。留空时天气工具仍能被发现和调用，只会返回“未配置彩云天气 token”的提示。免费额度下预报最多 3 天，请求过密会触发限流提示。

运行 `python -m src.main` 时，Agent 会自动发现天气工具并把它们加入普通聊天 Agent。设备控制、场景、记忆和知识 RAG Agent 不会获得天气工具，避免职责混用。

如果要接入其他天气、日历或新闻 MCP，只需将 `EXTERNAL_MCP_SERVERS` 改成 JSON 数组；支持 `stdio`、`sse` 和 `streamable_http` 三种传输方式。外部服务连接失败会记录 warning，主 Agent 仍可使用内置能力。

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
