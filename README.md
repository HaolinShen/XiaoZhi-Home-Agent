# 🏠 智能家居家电互联智能体 (Smart Home Agent)

> 用自然语言控制你的智能家居 —— 基于 **LangGraph + MCP + 阿里百炼** 的 AI Agent

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-%E2%89%A51.0-green)](https://github.com/langchain-ai/langgraph)
[![MCP](https://img.shields.io/badge/MCP-%E2%89%A51.0-purple)](https://modelcontextprotocol.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## ✨ 项目简介

这是一个完整可运行的智能家居 AI Agent 示例项目，面向想学习现代 **AI Agent 开发** 的开发者。你只需要对着终端说一句自然语言，Agent 就会自动理解意图、调用工具、控制设备。

```
"打开客厅灯，空调调到 25 度"  →  🤖 Agent 自动执行设备控制
"我要睡了"                    →  🌙 一键激活睡眠场景
"现在家里什么状态？"          →  📊 查询全部设备状态
```

### 🎯 设计目标

- **教学友好**：代码注释详细，专为理解 Agent 工作原理而设计
- **真实可跑**：内置设备模拟器，无需任何硬件即可运行和体验
- **协议标准**：基于 LangGraph 官方工作流 + MCP 标准协议
- **易于扩展**：新增设备类型、场景模式、真实硬件后端都很简单

---

## 🚀 功能特性

| 特性 | 说明 |
|------|------|
| 🧠 **ReAct 智能体** | LangGraph 状态图驱动的「思考-行动」循环，自动决策何时调用工具 |
| 🗺️ **结构化意图路由** | 模型分类 + 确定性兜底，动态选择 ReAct / 规划 / 并行查询 / RAG / 澄清五条路径 |
| 📋 **Planner–Executor–Verifier** | 多步骤任务先出计划、逐步执行、按真实设备状态验证，失败自动重试/重新规划 |
| 💡 **多设备控制** | 灯光、空调、电视、窗帘、加湿器、热水器、门锁、烧水壶，能力声明单一数据源自动生成工具 |
| 🌡️ **环境传感器** | 温湿度 + 人体存在传感器（只读），读数随执行器状态变化，让「先看数据再动手」成为可能 |
| 🎬 **场景模式** | 回家 / 离家 / 睡眠 / 观影 / 起床，一句话一键执行多个设备操作 |
| ⏰ **事件驱动自动化** | 定时 / 起床 / 车辆 ETA 例程持久化调度，动作在图外执行并验证 |
| 🤝 **多智能体协作** | 6 个角色按工具集隔离（device / scene / memory / automation / knowledge / chat） |
| 📚 **Agentic RAG（混合检索）** | 39 份设备说明书，BM25 词法通道 + 向量语义通道 RRF 融合；引用由代码拼接，答不出时明确拒绝。口语查询召回 3/30 → 23/30，Recall@1 51.8% → 87.5%（见 `evals/`） |
| 💬 **多轮对话记忆** | 基于 LangGraph Checkpoint，默认 SQLite 持久化，重启不丢上下文 |
| 🧠 **结构化长期记忆** | SQLite 保存家庭规则与个人偏好，支持范围隔离、查看、修改和删除 |
| 🔌 **MCP 集成** | 通过 Model Context Protocol 将工具暴露给 Claude Desktop 等外部 AI |
| 🛡️ **人工审批** | 敏感动作（解锁门锁、批量场景、自动化例程）执行前需用户确认 |
| 📈 **可观测性** | 进度事件双写（stream + 结构化日志），LLM 调用级 token/延迟采集 |
| 🖥️ **现代化 CLI** | Typer + Rich 构建，支持 Markdown 渲染、彩色面板、交互式命令 |
| 📦 **类型安全** | Pydantic v2 严格数据模型，枚举约束杜绝魔法字符串 Bug |

---

## 🛠️ 技术栈

| 技术 | 版本 | 作用 |
|------|------|------|
| **LangGraph** | ≥ 1.0 | Agent 工作流编排（状态图、节点、条件路由） |
| **LangChain** | ≥ 1.0 | LLM 调用封装、工具定义、消息管理 |
| **MCP** | ≥ 1.0 | 标准化工具暴露 / 消费协议 |
| **Pydantic v2** | ≥ 2.0 | 类型安全的数据模型 & 配置管理 |
| **阿里百炼** | — | 大模型 API（Qwen 系列，兼容 OpenAI 接口） |
| **Typer + Rich** | — | 现代化 CLI 终端界面 |
| **Loguru** | — | 结构化日志 |

---

## 🧠 Agent 工作原理

单次请求经过三段前置处理后，按意图进入**五条互斥业务路径**之一：

```
sync_context → memory_reasoner → task_router ─┬→ planner ⇄ 审批 ⇄ executor ⇄ verifier   （多步任务）
                                              ├→ compact_context → agent ⇄ 审批 ⇄ tools  （ReAct 主路）
                                              ├→ device_query_subgraph                   （多设备并行查询）
                                              ├→ knowledge_rag                           （设备知识 RAG）
                                              └→ clarification                           （信息不足反问）
```

**关键设计：**
1. 意图分类 = 模型结构化输出 + 确定性兜底；「未来时间 + 自动化词」等硬信号直接短路，不让模型误判
2. Planner–Executor–Verifier 是显式状态机：`planning_status` 驱动全部条件边，
   验证读**注册中心的真实设备状态**，不靠 LLM 自述成败
3. 身份永远来自 `RunnableConfig["configurable"]`，绝不接受模型生成的 `home_id` / `user_id`
4. 多智能体开启时 6 个角色各自 `bind_tools` 一个工具子集（工具集隔离）
5. 自动化子系统运行在图外：持久化例程由后台调度器执行，动作同样经 `verify_step` 验证
6. 完整的架构演进见 [docs/tutorial.md](docs/tutorial.md) 与 [docs/iterations](docs/iterations)

---

## 📦 快速开始

### 1. 环境准备

```bash
# 创建并激活 Python 3.12 环境（推荐）
conda create -n langgraph python=3.12 -y
conda activate langgraph
```

### 2. 安装依赖

```bash
cd langgraph
pip install -e ".[dev]"
# 或手动安装：
# pip install langgraph langchain langchain-openai langgraph-checkpoint-sqlite \
#             pydantic pydantic-settings python-dotenv mcp typer rich loguru httpx
```

### 3. 配置 API Key

1. 打开 [阿里百炼控制台](https://bailian.console.aliyun.com/)，登录阿里云账号
2. 进入 **模型广场**，在左侧菜单创建 **API Key**
3. 复制 Key 并配置：

```bash
cp .env.example .env
# 编辑 .env，填入你的 API Key
```

`.env` 支持两种命名风格（通用名优先）：

```dotenv
# 通用名（推荐，兼容多种 LLM 提供商）
LLM_MODEL_ID=qwen-plus
LLM_API_KEY=sk-your-api-key-here
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# 百炼专用名（备选）
# BAILIAN_API_KEY=sk-xxx
# BAILIAN_MODEL=qwen-plus
```

> 💰 **费用参考**：qwen-plus 约 ¥0.004 / 千 token，个人开发每月几块钱。

### 4. 启动对话

```bash
# 启动交互式对话
python -m src.main

# 指定模型 / 调试模式
python -m src.main --model qwen-max
python -m src.main --debug

# 额外显示路由 / 记忆判断等诊断事件（规划过程默认就会显示）
python -m src.main --trace

# 使用稳定业务会话，并提供 App 当前房间上下文
python -m src.main --home-id demo-home --user-id user-001 \
  --session-id session-001 --client-id phone-001 --room-id living_room

# 本地演示家庭共享规则管理（生产环境应由业务后端授予权限）
python -m src.main --home-id demo-home --user-id admin-001 --admin
```

`session_id` 会直接用作 LangGraph `thread_id`。复用同一 `session_id` 可恢复会话，
`/reset` 会创建新会话；`room_id` 和 `device_id` 会在进入 Agent 前校验住宅归属。

### 长期记忆

长期记忆默认保存到 `data/memories.db`。用户明确表达“记住我喜欢暖光”或
“以后睡觉时空调设为 25 度”时，Agent 可调用记忆工具保存个人偏好；单次控制指令、
临时感受和实时设备状态不会保存。

支持的范围包括：

- 家庭共享：`home_id`
- 房间共享：`home_id + room_id`
- 设备共享：`home_id + device_id`
- 个人及个人空间/设备：`home_id + user_id`，可附带 `room_id` 或 `device_id`

所有查询都强制包含 `home_id`。个人记忆只对所属用户可见；家庭、房间和设备共享
记忆的写入、修改和删除需要可信业务上下文中的管理员权限。模型工具参数不能指定
任意 `home_id` 或 `user_id`。

---

## 💬 使用指南

### 支持的设备

**执行器**（可读可写，有对应的 `control_xxx` 工具）

| 设备 | 示例指令 |
|------|---------|
| 💡 灯光 | `打开客厅灯` · `把卧室灯调暗到 30%` · `灯调成白光` |
| ❄️ 空调 | `打开客厅空调` · `空调调到 25 度` · `风速调高` · `切到制热` |
| 📺 电视 | `打开电视` · `音量调到 50` · `静音` · `切换到 HDMI 2` |
| 🪟 窗帘 | `打开窗帘` · `关上窗帘` · `窗帘打开一半` |
| 💧 加湿器 | `开加湿器` · `湿度设到 60%` · `雾量调低` |
| 🚿 电热水器 | `打开热水器` · `热水器调到 50 度` |
| 🔒 门锁 | `把门锁上` · `解锁门锁`（需人工审批） |
| ☕ 烧水壶 | `把水烧开` · `烧水到 80 度` |

> 每个设备的工具、合法动作、参数 Schema、Planner 词表、模拟器默认实例全部从
> `src/devices/capabilities.py` 的能力声明**自动派生**——新增设备只需在那一处声明，
> 一致性由 `tests/test_capabilities.py` 在测试阶段兜底（详见迭代 011）。

**传感器**（只读，只有 `read_sensor` 工具）

| 设备 | 示例指令 |
|------|---------|
| 🌡️ 温湿度 | `屋里多少度` · `客厅湿度怎么样` · `有点干`（先读数再决定开多大） |
| 👤 人体存在 | `家里有人吗` · `玄关有人经过吗` |

传感器故意没有 `control_xxx` 工具，Agent 从工具名就知道它改不了状态：
它读到的值来自环境而非自己的命令，所以是验证环节唯一真正的外部反馈。
模拟器会按同房间执行器的状态推演读数——开加湿器，湿度就会朝目标爬升。

### 场景模式

| 场景 | 触发语 | 效果 |
|------|--------|------|
| 🏠 回家 | `我回来了` | 开客厅灯 + 空调 + 窗帘 |
| 👋 离家 | `我出门了` | 关闭所有电器，节能安全 |
| 🌙 睡眠 | `我要睡了` | 关灯关电视，卧室空调低风速 |
| 🎬 观影 | `我要看电影` | 氛围灯 + 开电视 + 关窗帘 |
| 🌅 起床 | `起床了` | 开窗帘 + 关空调 + 渐亮灯光 |

### 特殊命令

| 命令 | 说明 |
|------|------|
| `/status` | 查看所有设备状态（不经过 LLM） |
| `/scenes` | 列出所有可用场景 |
| `/reset` | 重置对话记忆 |
| `/history` | 查看当前会话最近的 Checkpoint 状态历史 |
| `/plan` | 复盘最近一次多步骤计划：Planner 产出 + 逐步验证轨迹 |
| `/routines` | 查看定时起床和车辆回家例程及任务状态 |
| `/help` | 显示帮助 |
| `/quit` | 退出 |

### 看得见的规划过程

多动作请求（如「关掉客厅灯，然后把卧室空调调到 25 度」）会走 Planner → Executor →
Verifier，三个阶段在运行时逐步显示，不需要额外开关：

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
🏁 规划结束 · completed · 验证通过 2 次 / 共 2 次尝试 · 最终计划 v1
```

计划表格在任何设备被操作之前就已完整列出工具名和参数——这就是「Planner 只写不做」
的直接证据。失败时能看到 `↻ 重试步骤 …`，重试额度用尽则是 `⟲ 把失败原因交回 Planner
重新规划`，随后 v2 计划重新出现并再次等待确认。

路由、记忆判断这类诊断事件默认折叠，加 `--trace` 才显示，避免淹没规划过程。

### 事件驱动自动化

项目支持由 Automation Agent 动态规划的持久化例程，不要求命中固定场景模板：

- `今天下午5点打球回到家，提前准备洗澡水和客厅降温`：Agent 生成热水器与空调动作，并分别安排在目标时间之前；
- `明天早上6点叫我起床，提前准备洗澡和冲牛奶的热水`：动态生成音响、热水器、烧水壶、窗帘或灯光动作；
- `车辆到家前准备热水、空调和窗帘`：Agent 生成车辆 ETA 相对动作，ETA 更新时调整尚未执行任务。

创建例程会先要求确认。任务保存在 `data/automation.db`，CLI 后台调度器负责执行；输入 `/routines` 可以查看每个例程的待执行、完成、失败和取消数量。真实音响和汽车厂商 API 尚未绑定，当前通过 `SimulatorSpeakerBackend` 和 `VehicleSimulator` 完成本地闭环。

---

## 🔌 MCP 集成

通过 [Model Context Protocol](https://modelcontextprotocol.io/)，可将智能家居工具暴露给 Claude Desktop 等外部 AI 客户端。

```bash
# 方式 1: stdio 模式（由 Claude Desktop 等 MCP 客户端启动）
python -m src.mcp.server

# 方式 2: SSE 模式（独立 HTTP 服务，端口 8765）
python -m src.mcp.server --transport sse --port 8765
```

**在 Claude Desktop 中配置（`claude_desktop_config.json`）：**

```json
{
  "mcpServers": {
    "smart-home": {
      "command": "python",
      "args": ["-m", "src.mcp.server"]
    }
  }
}
```

项目还内置了一个基于**彩云天气**的天气 MCP。`.env.example` 默认通过当前 Python 环境启动它：

```dotenv
EXTERNAL_MCP_SERVERS=[{"name":"weather","transport":"stdio","command":"python","args":["-m","src.mcp.weather_server"]}]
WEATHER_DEFAULT_LOCATION=杭州
CAIYUN_WEATHER_TOKEN=你的彩云 token
```

彩云 token 可在 <https://dashboard.caiyunapp.com> 免费领取；没配置时天气工具会返回明确的提示而不是报错。彩云只接受经纬度，城市名到坐标的转换仍由免费的 Open-Meteo geocoding 完成，无需额外 Key。

启动主 Agent 后可以询问“杭州今天天气怎么样”或“北京未来三天天气如何”。天气 MCP 只提供实时天气和预报查询（免费额度下预报最多 3 天），不拥有任何设备控制权限。

---

## 📁 项目结构

```
langgraph/
├── pyproject.toml              # 项目元数据 & 依赖 & ruff/mypy 配置
├── .env.example                # 环境变量模板
├── docs/
│   ├── tutorial.md             # 45 分钟开发教程（强烈推荐阅读）
│   └── iterations/             # 迭代方案与实现记录（001-011）
│
├── src/
│   ├── main.py                 # CLI 入口（Typer + Rich 交互界面）
│   ├── config.py               # 配置管理（pydantic-settings）
│   ├── models.py               # Pydantic v2 设备数据模型
│   │
│   ├── agent/
│   │   ├── context.py          # 可信请求身份与空间归属校验
│   │   ├── graph.py            # ★ LangGraph 工作流（Agent 核心）
│   │   ├── routing.py          # 结构化意图路由（模型 + 确定性兜底）
│   │   ├── planning.py         # Planner–Executor–Verifier 规划侧
│   │   ├── heuristics.py       # 确定性启发式判定（路由/规划/自动化共用词表）
│   │   ├── telemetry.py        # LLM 调用级 token / 延迟采集
│   │   ├── observability.py    # 进度事件（stream + 结构化日志双写）
│   │   ├── multi_agent.py      # 多智能体角色与工具集隔离
│   │   ├── parallel.py         # 设备并行查询子图
│   │   ├── approval.py         # 敏感动作人工审批判定
│   │   ├── reasoning.py        # 记忆适用性推理
│   │   ├── session.py          # 会话创建、恢复与结束
│   │   ├── state.py            # Agent 状态定义
│   │   └── prompts.py          # 系统提示词
│   │
│   ├── devices/
│   │   ├── capabilities.py     # ★ 设备能力单一数据源（工具/规划/场景由此派生）
│   │   ├── base.py             # 设备后端抽象接口 + 注册中心
│   │   └── simulator.py        # 内存模拟器（默认后端）
│   │
│   ├── tools/
│   │   ├── __init__.py         # build_all_tools 工厂（显式依赖注入）
│   │   ├── devices.py          # 设备工具工厂（从能力声明生成 control_xxx）
│   │   ├── scenes.py           # 场景模式工具工厂
│   │   ├── memory.py           # 长期记忆工具工厂
│   │   └── automation.py       # 自动化例程工具工厂
│   │
│   ├── automation/             # 图外的持久化调度子系统
│   │   ├── runtime.py          # 运行时组装（store / executor / scheduler）
│   │   ├── executor.py         # 例程动作执行 + 验证（显式关闭偏好观察）
│   │   ├── scheduler.py        # 可注入虚拟时间的确定性调度
│   │   ├── store.py            # SQLite 持久化（data/automation.db）
│   │   └── ...
│   │
│   ├── knowledge/              # 说明书语料索引 + 混合检索 + Agentic RAG 子图
│   ├── evaluation/             # 离线轨迹评测 + 说明书召回评测
│   │
│   ├── memory/
│   │   ├── models.py           # 长期记忆数据模型
│   │   ├── repository.py       # SQLite Repository
│   │   ├── service.py          # 隔离、权限与范围规则
│   │   └── store.py            # 检查点记忆（内存 / SQLite）
│   │
│   ├── mcp/
│   │   ├── server.py           # MCP 服务器（复用图内同一份工具实现）
│   │   ├── client.py           # MCP 客户端
│   │   └── weather_server.py   # 彩云天气 MCP
│   │
│   └── middleware/
│       └── interceptors.py     # 中间件（日志 + 重试）—— 教学演示，未接入运行路径
│
└── tests/                      # 全量 unittest 回归（capabilities/heuristics/telemetry 等）
```

---

## 🧩 扩展指南

### 新增一个设备类型

P0 改造后只需 **1 步**：在 `src/devices/capabilities.py` 的 `CAPABILITIES` 里加一条
能力声明（设备类型、工具名、合法 action 及其副作用实现、期望状态、参数、类型关键词、
默认实例、场景归属）。

其余全部自动派生：

- 控制工具的 JSON Schema / docstring（`src/tools/devices.py`）
- Planner 合法 action 词表与 `PlanStep.tool_name` 的枚举（`src/agent/planning.py`）
- `registry.find()` 的类型关键词（`src/devices/base.py`）
- 模拟器默认设备（`src/devices/simulator.py`）
- 场景批量开关的设备类型集合（`src/tools/scenes.py`）
- 自动化例程允许的工具名（`src/automation/planning.py`）
- MCP 服务器暴露的控制工具（`src/mcp/server.py`）

仍需要手工做的只有两件（与能力声明本身无关）：在 `src/models.py` 定义设备数据模型；
若动作对外敏感（如解锁），在 `src/agent/approval.py` 加审批判定。
`tests/test_capabilities.py` 会把所有派生点逐一钉住，漏任何一处都在测试阶段失败。

> 改造前这条路径要手工同步 9 处（含两处无法反射的副本），漏改的表现是
> 「Planner 第一版计划稳定失败，且不报错」。

### 接入真实设备

1. 新建 `RealDeviceBackend(DeviceBackend)`，实现相同的接口方法（`get` / `update` 等）
2. 在 `src/main.py` 中将 `SimulatorBackend()` 替换为 `RealDeviceBackend()`
3. **工具层和 Agent 层无需任何修改** —— 面向接口编程的好处

---

## ⚙️ 环境变量说明

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LLM_MODEL_ID` / `BAILIAN_MODEL` | `qwen-plus` | 模型名称 |
| `LLM_API_KEY` / `BAILIAN_API_KEY` | — | API Key（必填） |
| `LLM_BASE_URL` / `BAILIAN_BASE_URL` | 百炼默认地址 | 服务地址 |
| `LLM_TIMEOUT` | `60` | LLM 请求超时（秒） |
| `LOG_LEVEL` | `INFO` | 日志级别 |
| `MCP_SERVER_ENABLED` | `true` | 是否内置启动 MCP 服务器 |
| `MCP_SERVER_PORT` | `8765` | MCP SSE 模式端口 |
| `EXTERNAL_MCP_SERVERS` | 天气 MCP 配置 | 启动时发现并加载外部 MCP 工具 |
| `WEATHER_DEFAULT_LOCATION` | 空 | 天气查询未提供城市时使用的默认城市 |
| `CHECKPOINT_DB_PATH` | `data/checkpoints.db` | 记忆持久化路径（留空=内存模式） |
| `ENABLE_LONG_TERM_MEMORY` | `true` | 是否启用结构化长期记忆 |
| `CHECKPOINT_LONG_TERM_DB_PATH` | `data/memories.db` | 长期记忆 SQLite 路径 |
| `CHECKPOINT_CONTEXT_MAX_MESSAGES` | `12` | 模型输入保留的最近消息上限 |
| `CHECKPOINT_CONTEXT_MAX_TOKENS` | `2400` | 模型输入的估算 token 上限 |
| `CHECKPOINT_TOOL_RESULT_MAX_CHARS` | `1200` | 单条工具结果保留字符上限 |
| `CHECKPOINT_SUMMARY_MAX_CHARS` | `1800` | 滚动摘要字符上限 |
| `CHECKPOINT_SESSION_TTL_HOURS` | `168` | 无活动会话检查点保留小时数 |
| `PLANNING_ENABLED` / `PLANNING_MAX_STEPS` 等 | `true` / `8` | Planner–Executor–Verifier 开关与步数/重试上限 |
| `ROUTING_ENABLED` / `ROUTING_CONFIDENCE_THRESHOLD` | `true` / `0.6` | 结构化意图路由开关与置信阈值 |
| `MULTI_AGENT_ENABLED` / `MULTI_AGENT_MAX_HANDOFFS` | `true` / `2` | 多智能体协作开关与交接上限 |
| `RAG_ENABLED` / `RAG_TOP_K` | `true` / `3` | 本地知识 RAG 开关与检索条数 |
| `RAG_BM25_WEIGHT` / `RAG_DENSE_WEIGHT` | `0.5` / `0.5` | 混合检索两个通道的权重，设 0 表示该通道不参与 |
| `RAG_EMBEDDING_MODEL_ID` | `text-embedding-v4` | 语义通道的 embedding 型号；留空则退化为纯 BM25（地址与 Key 默认回落到 `LLM_*`） |
| `RAG_MIN_SCORE` / `RAG_REWRITTEN_MIN_SCORE` / `RAG_RELATIVE_FLOOR` | `0.35` / `0.42` / `0.7` | 命中准入的两档绝对下限与引用的相对截断（实测标定，换语料需 `--sweep` 重标） |
| `AUTOMATION_ENABLED` / `AUTOMATION_DB_PATH` | `true` / `data/automation.db` | 事件驱动自动化开关与持久化路径 |

---

## 📚 学习资源

- [📖 开发教程](docs/tutorial.md) — 从零讲解项目架构与 Agent 开发（约 45 分钟）
- [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/) — 工作流与状态图
- [MCP 规范](https://modelcontextprotocol.io/) — 工具标准化协议
- [阿里百炼控制台](https://bailian.console.aliyun.com/) — 获取 API Key 与模型广场

---

## 📄 License

[MIT](LICENSE)

---

<p align="center">
  <sub>用 ❤️ 和 LangGraph 构建 · 有问题欢迎提交 Issue</sub>
</p>
