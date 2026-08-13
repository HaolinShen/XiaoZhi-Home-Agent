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
| 💡 **多设备控制** | 灯光、空调、电视、窗帘、加湿器的完整控制能力 |
| 🌡️ **环境传感器** | 温湿度 + 人体存在传感器（只读），读数随执行器状态变化，让「先看数据再动手」成为可能 |
| 🎬 **场景模式** | 回家 / 离家 / 睡眠 / 观影 / 起床，一句话一键执行多个设备操作 |
| 💬 **多轮对话记忆** | 基于 LangGraph Checkpoint，默认 SQLite 持久化，重启不丢上下文 |
| 🧠 **结构化长期记忆** | SQLite 保存家庭规则与个人偏好，支持范围隔离、查看、修改和删除 |
| 🔌 **MCP 集成** | 通过 Model Context Protocol 将工具暴露给 Claude Desktop 等外部 AI |
| 🛡️ **中间件体系** | 日志记录 + 失败自动重试（指数退避），装饰器模式可自由组合 |
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

项目采用经典的 **ReAct（Reasoning + Acting）循环**：

```
┌──────────────────────────────────────────────────────────┐
│                   LangGraph Agent 状态图                   │
│                                                          │
│   用户输入                                                │
│      │                                                   │
│      ▼                                                   │
│  ┌─────────┐  有 tool_calls?   ┌──────────┐             │
│  │  Agent  │ ────────────────→ │  Tools   │             │
│  │  (LLM)  │                   │ (执行工具) │             │
│  │         │ ←──────────────── │          │             │
│  └────┬────┘   返回执行结果     └──────────┘             │
│       │                                                   │
│       │  没有 tool_calls                                   │
│       ▼                                                   │
│    最终回复 → 返回给用户                                    │
│                                                          │
│   记忆层: MemorySaver / SqliteSaver（跨轮次状态保持）       │
└──────────────────────────────────────────────────────────┘
```

**关键设计：**
1. 使用 LangGraph 预置的 `ToolNode`，无需手动解析 tool_calls
2. `SystemMessage` 每次追加在消息列表最前，防止 LLM 遗忘角色
3. 检查点机制实现多轮对话记忆（内存 / SQLite 可切换）
4. 工具函数通过类型注解 + docstring 自动生成 LLM 可见的 JSON Schema

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
| `/help` | 显示帮助 |
| `/quit` | 退出 |

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
├── pyproject.toml              # 项目元数据 & 依赖声明
├── .env.example                # 环境变量模板
├── docs/
│   └── tutorial.md             # 45 分钟开发教程（强烈推荐阅读）
│
├── src/
│   ├── main.py                 # CLI 入口（Typer + Rich 交互界面）
│   ├── config.py               # 配置管理（pydantic-settings）
│   ├── models.py               # Pydantic v2 设备数据模型
│   │
│   ├── agent/
│   │   ├── context.py          # 可信请求身份与空间归属校验
│   │   ├── graph.py            # ★ LangGraph 工作流（Agent 核心）
│   │   ├── session.py          # 会话创建、恢复与结束
│   │   ├── state.py            # Agent 状态定义
│   │   └── prompts.py          # 系统提示词
│   │
│   ├── devices/
│   │   ├── base.py             # 设备后端抽象接口
│   │   └── simulator.py        # 内存模拟器（默认后端）
│   │
│   ├── tools/
│   │   ├── devices.py          # 设备控制工具（灯光/空调/电视/窗帘/加湿器）+ 传感器读取
│   │   └── scenes.py           # 场景模式工具
│   │
│   ├── mcp/
│   │   ├── server.py           # MCP 服务器（stdio / SSE）
│   │   └── client.py           # MCP 客户端
│   │
│   ├── memory/
│   │   ├── models.py           # 长期记忆数据模型
│   │   ├── repository.py       # SQLite Repository
│   │   ├── service.py          # 隔离、权限与范围规则
│   │   └── store.py            # 检查点记忆（内存 / SQLite）
│   ├── tools/
│   │   └── memory.py           # 受可信上下文约束的记忆工具
│   │
│   └── middleware/
│       └── interceptors.py     # 中间件（日志 + 重试）
│
└── tests/
    └── visualize_graph.ipynb   # LangGraph 图可视化 Notebook
```

---

## 🧩 扩展指南

### 新增一个设备类型

只需 4 步（以「智能门锁」为例）：

1. **`src/models.py`**：定义 `LockDevice(BaseDevice)` 数据模型
2. **`src/devices/simulator.py`**：注册默认设备实例
3. **`src/tools/devices.py`**：添加 `@tool def control_lock(...)` 工具函数
4. **`src/tools/__init__.py`**：在 `get_all_tools()` 中注册

> LLM 会自动通过工具名称和 docstring 学会何时调用、怎么传参。

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
