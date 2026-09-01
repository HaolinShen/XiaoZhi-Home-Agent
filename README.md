# 🏠 智能家居家电互联智能体 (Smart Home Agent)

> 用自然语言控制你的智能家居 —— 基于 **LangGraph + MCP + 阿里百炼** 的 AI Agent

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-%E2%89%A51.0-green)](https://github.com/langchain-ai/langgraph)
[![MCP](https://img.shields.io/badge/MCP-%E2%89%A51.0-purple)](https://modelcontextprotocol.io/)
[![Tests](https://img.shields.io/badge/tests-264%20passed-brightgreen)](#-测试与静态检查)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## ✨ 项目简介

这是一个完整可运行的智能家居 AI Agent，内置设备模拟器，不需要任何硬件就能跑起来。
对着终端说一句自然语言，Agent 会自己判断意图、选择执行范式、调用工具、核对真实设备状态。

```
"打开客厅灯，空调调到 25 度"      →  📋 Planner 先出计划，逐步执行并按真实状态验证
"我要睡了"                       →  🌙 一键激活睡眠场景（需确认）
"客厅空调和卧室空调都什么状态？"  →  ⚡ Send 动态扇出并行查询
"卧室空调显示 E4 是什么意思"      →  📚 说明书混合检索，答不出就明确拒绝
"明天早上 6 点叫我起床，提前烧好水" →  ⏰ 生成持久化例程，图外调度执行
```

### 🎯 设计目标

- **真实可跑**：内置设备模拟器与传感器推演，无需硬件即可完整体验
- **约束优先**：能不能做由代码判定，不由模型自述——验证读真实设备状态，身份来自可信上下文
- **单一数据源**：新增设备只在一处声明，工具 Schema / 规划词表 / 场景归属 / MCP 工具全部自动派生
- **协议标准**：LangGraph 状态图 + MCP 标准协议，工具可被 Claude Desktop 等外部客户端消费

---

## 🚀 功能特性

| 特性 | 说明 |
|------|------|
| 🧠 **ReAct 智能体** | LangGraph 状态图驱动的「思考-行动」循环，自动决策何时调用工具 |
| 🗺️ **结构化意图路由** | 模型分类 + 确定性兜底，8 类意图动态选择 ReAct / 规划 / 并行查询 / RAG / 澄清五条路径 |
| 📋 **Planner–Executor–Verifier** | 多步骤任务先出计划（2–8 步）、逐步执行、按真实设备状态验证，失败自动重试或重新规划 |
| 💡 **多设备控制** | 8 类执行器共 40 个合法动作，工具与词表由能力声明单一数据源自动生成 |
| 🌡️ **环境传感器** | 温湿度 + 人体存在（只读），读数随同房间执行器状态推演，让「先看数据再动手」成立 |
| 🎬 **场景模式** | 回家 / 离家 / 睡眠 / 观影 / 起床，一句话执行多设备编排 |
| ⏰ **事件驱动自动化** | 定时 / 起床 / 车辆 ETA 例程持久化调度，动作在图外执行并同样经验证 |
| 🤝 **多智能体协作** | 6 个角色按工具集隔离（device / scene / memory / automation / knowledge / chat） |
| 📚 **Agentic RAG（混合检索）** | 39 份说明书切 124 个 chunk，BM25 词法 + 向量语义双通道 RRF 融合；引用由代码拼接，答不出明确拒答。口语查询召回 3/30 → 23/30，Recall@1 51.8% → 87.5%，拒答准确率 100% |
| 💬 **多轮对话记忆** | LangGraph Checkpoint，默认 SQLite 持久化，重启不丢上下文；支持时间旅行回看 |
| 🧠 **结构化长期记忆** | SQLite 保存家庭规则与个人偏好，范围隔离 + 管理员权限 + 版本历史 + 偏好候选确认 |
| 🔌 **MCP 集成** | 对外暴露 11 个工具；同时作为客户端消费外部 MCP（内置彩云天气） |
| 🛡️ **人工审批** | 解锁门锁（高危）、批量场景、创建自动化例程执行前需用户确认 |
| 📈 **可观测性** | 进度事件双写（stream + 结构化日志），LLM 调用级 token / 延迟采集，节点计时 |
| 🧪 **回归与评测** | 264 个测试 + 383 subtests；说明书检索有独立召回评测（63 条标注用例） |
| 🖥️ **现代化 CLI** | Typer + Rich，Markdown 渲染、彩色面板、运行时规划可视化 |

---

## 🛠️ 技术栈

| 技术 | 版本 | 作用 |
|------|------|------|
| **LangGraph** | ≥ 1.0 | Agent 工作流编排（状态图、条件边、Send、Checkpoint、interrupt） |
| **LangChain** | ≥ 1.0 | LLM 调用封装、工具定义、消息管理 |
| **MCP** | ≥ 1.0 | 标准化工具暴露 / 消费协议 |
| **Pydantic v2** | ≥ 2.0 | 类型安全的数据模型 & 配置管理 |
| **rank-bm25 + jieba** | — | 说明书检索的词法通道（BM25Okapi + 中文分词） |
| **NumPy** | ≥ 1.24 | 向量通道的归一化与余弦相似度 |
| **阿里百炼** | — | 大模型与 embedding API（Qwen 系列，兼容 OpenAI 接口） |
| **Typer + Rich** | — | 现代化 CLI 终端界面 |
| **Loguru** | — | 结构化日志（channel 分流） |

---

## 🧠 Agent 工作原理

`src/agent/graph.py:build_graph()` 编译出**单个** `StateGraph`（17 个节点）。入口固定是三段前置处理，
然后按意图进入**五条互斥业务路径**之一：

```
sync_context → memory_reasoner → task_router ─┬→ planner ⇄ plan_approval ⇄ executor ⇄ verifier → planning_finalize → END
                                              ├→ compact_context → agent ⇄ [approval] ⇄ tools              → END
                                              ├→ device_query_subgraph  （Send 动态扇出并行查询）          → END
                                              ├→ knowledge_rag          （说明书 Agentic RAG 子图）        → END
                                              └→ clarification          （信息不足直接反问）               → END
```

两处容易看错的细节：

- ReAct 环的回边落在 **`compact_context`** 而不是 `agent`——工具结果先过一次上下文压缩再回模型，
  所以长对话里的工具输出不会无限堆积。
- 多智能体开启时，`agent` 还有一条出边 **`supervisor_finalize`**，负责按 `max_handoffs` 收口协作状态。

**关键设计：**

1. **意图分类 = 模型结构化输出 + 确定性兜底。** `automation_management` 命中确定性信号时直接短路返回，
   不给模型改判机会——否则「明天 6 点回家前开空调」会被误判成预定义的「回家模式」。
2. **Planner–Executor–Verifier 是显式状态机。** `planning_status` 驱动全部条件边，`verifier` 读**注册中心的
   真实设备状态**跟 `expected_state` 比对，不靠 LLM 自述成败。失败后的分支取决于 `problem_type`：
   `unsupported_action` / `device_not_found` 是确定性错误，跳过重试直接 replan（同样参数重放不可能成功）。
3. **身份永远来自 `RunnableConfig["configurable"]`**，绝不接受模型生成的 `home_id` / `user_id`；
   `room_id` / `device_id` 由 `SpaceDirectory.validate()` 校验住宅归属。
4. **工具通过工厂显式注入依赖。** `build_all_tools(registry, *, memory_service, automation_runtime,
   external_tools, enable_preference_tracking)` 闭包持有依赖，没有模块级单例。自动化未启用时
   自动化工具根本不出现在模型面前。
5. **多智能体是工具集隔离**，不是多个图。6 个角色各 `bind_tools` 一个子集（见下表）。
6. **自动化子系统运行在图外**：持久化例程由后台调度器执行，动作同样经 `verify_step` 按真实状态验证。

### 工具与角色

自动化启用时模型面前共 **27 个工具**（未启用时 21 个），外部 MCP 工具追加在末尾：

| 组 | 数量 | 工具 |
|---|---|---|
| 设备 | 10 | 8 个 `control_*` + `read_sensor` + `get_device_status` |
| 场景 | 2 | `activate_scene`、`list_scenes` |
| 记忆 | 9 | 保存 / 列出 / 修改 / 删除个人记忆、保存家庭规则、偏好候选的列出 / 确认 / 拒绝、版本历史 |
| 自动化 | 6 | 定时例程、车辆到家例程、起床例程、启用车辆例程、列出、取消 |

多智能体的 6 个角色及其工具子集（`src/agent/multi_agent.py` + `graph.py`）：

| 角色 | 触发意图 | 工具集 |
|---|---|---|
| `device` | `device_query` / `device_control` | 10 个设备工具（从 `CONTROL_TOOL_NAMES` 派生，新增设备自动进入） |
| `scene` | `scene_control` | `activate_scene`、`list_scenes`、`read_sensor`（离家前要先确认无人） |
| `memory` | `memory_management` | 9 个记忆工具 |
| `automation` | `automation_management` | 6 个自动化工具 |
| `knowledge` | `device_knowledge` | 无工具（裸 LLM，检索由 RAG 子图完成） |
| `chat` | 其余 | 仅外部 MCP 只读工具 |

> `automation` 角色额外有一层强制：识别出「未来触发信号 + 设备动作信号」同时出现时锁定必须调用的
> 创建工具，模型没调就补一条纠正 SystemMessage 重试一次。查询类与取消类请求绝不会被判成创建请求。

---

## 📦 快速开始

### 1. 环境准备

```bash
# 推荐 Python 3.12（项目要求 ≥ 3.11）
conda create -n langgraph python=3.12 -y
conda activate langgraph
```

### 2. 安装依赖

```bash
cd langgraph
pip install -e ".[dev]"
```

`[dev]` 额外带 `pytest` / `pytest-asyncio` / `ruff` / `mypy`。

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

> 占位值（`sk-your-api-key-here` / `your-api-key` / 空）会在启动时直接抛 `ValueError`，
> 而不是等到第一次调用才报鉴权失败。
>
> 💰 **费用参考**：qwen-plus 约 ¥0.004 / 千 token，个人开发每月几块钱。说明书检索的
> embedding 会按内容哈希落盘缓存到 `data/embeddings`，同一份语料只算一次钱。

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

# 以家庭管理员身份运行（才能写家庭共享记忆；生产环境应由业务后端授权）
python -m src.main --home-id demo-home --user-id admin-001 --admin

# 不进对话，只打印一次设备状态
python -m src.main status
```

`session_id` 会直接用作 LangGraph `thread_id`。复用同一 `session_id` 可恢复会话，
`/reset` 会创建新会话；`room_id` 和 `device_id` 会在进入 Agent 前校验住宅归属。

> ⚠️ **必须在项目根目录启动**：说明书语料路径（`docs/knowledge`）和评测数据集路径
> （`evals/knowledge_recall.json`）都是相对 cwd 解析的。

---

## 💬 使用指南

### 支持的设备

**执行器**（可读可写，各有对应的 `control_xxx` 工具）

| 设备 | 合法动作数 | 示例指令 |
|------|-----------|---------|
| 💡 灯光 | 4 | `打开客厅灯` · `把卧室灯调暗到 30%` · `灯调成白光` |
| ❄️ 空调 | 5 | `打开客厅空调` · `空调调到 25 度` · `风速调高` · `切到制热` |
| 📺 电视 | 5 | `打开电视` · `音量调到 50` · `静音` · `切换到 HDMI 2` |
| 🪟 窗帘 | 3 | `打开窗帘` · `关上窗帘` · `窗帘打开一半` |
| 💧 加湿器 | 4 | `开加湿器` · `湿度设到 60%` · `雾量调低`（水箱空时前三个动作会被 precheck 拦下） |
| 🚿 电热水器 | 3 | `打开热水器` · `热水器调到 50 度` |
| 🔒 门锁 | 2 | `把门锁上` · `解锁门锁`（需人工审批；离线时被 precheck 拦下） |
| ☕ 烧水壶 | 4 | `把水烧开` · `烧水到 80 度` |

> 每个设备的工具、合法动作、参数 Schema、期望状态、Planner 词表、`registry.find()` 关键词、
> 模拟器默认实例、场景归属、自动化允许的工具名、MCP 工具，全部从
> `src/devices/capabilities.py` 的能力声明**自动派生**。一致性由 `tests/test_capabilities.py`
> 的生成式断言兜底：漏一处是测试失败，不是运行期静默出错。

**传感器**（只读，只有 `read_sensor` 工具）

| 设备 | 示例指令 |
|------|---------|
| 🌡️ 温湿度 | `屋里多少度` · `客厅湿度怎么样` · `有点干`（先读数再决定开多大） |
| 👤 人体存在 | `家里有人吗` · `玄关有人经过吗` |

传感器故意没有 `control_xxx` 工具，`PlanStep.tool_name` 的枚举也不含 `read_sensor` ——
Agent 从工具名就知道它改不了状态。它读到的值来自环境而非自己的命令，
所以是验证环节唯一真正的外部反馈。模拟器会按同房间执行器的状态推演读数：
开加湿器，湿度就会朝目标爬升。

### 场景模式

| 场景 | 触发语 | 实际效果 |
|------|--------|---------|
| 🏠 回家模式 | `我回来了` · `到家了` | 客厅灯开（亮度 80、暖白）+ 客厅空调制冷 26℃ + 客厅窗帘全开 |
| 👋 离家模式 | `我要出门了` · `走了` | 6 类电器全部断电 + 窗帘全关 + 门锁全部上锁 |
| 🌙 睡眠模式 | `我要睡了` · `困了` | 关灯 / 电视 / 加湿器 + 窗帘全关 + 卧室空调 26℃ 低风速 + **客厅空调关闭** |
| 🎬 观影模式 | `我要看电影` · `看剧` | 客厅灯降至 10% 暖黄、其余灯关 + 电视开 + 窗帘全关 + **客厅空调 25℃ 低风速** |
| 🌅 起床模式 | `起床了` · `早上好` | 卧室窗帘全开 + 卧室空调关 + 卧室灯亮度 50 暖白 |

`activate_scene` 只认场景全名（精确匹配），触发语到场景名的映射由工具 docstring 和系统提示词
交给模型完成；命中场景关键词时**一律不走 Planner**，避免把预定义编排拆成逐步计划。
离家模式**不会**遍历所有设备去关——那会把传感器一起关掉，表现为「Agent 说家里没人」。

### 特殊命令

| 命令 | 说明 |
|------|------|
| `/status` | 查看所有设备状态（不经过 LLM） |
| `/scenes` | 列出所有可用场景 |
| `/reset` | 重置对话记忆（新建会话） |
| `/history` | 查看当前会话最近的 Checkpoint 状态历史（时间旅行） |
| `/plan` | 复盘最近一次多步骤计划：Planner 产出 + 逐步验证轨迹 |
| `/routines` | 查看定时起床和车辆回家例程及任务状态 |
| `/help`（`/h`） | 显示帮助 |
| `/quit`（`/exit`、`/q`） | 退出 |

### 看得见的规划过程

多动作请求（≥2 个动作词，且 ≥2 类设备或出现连接词）会走 Planner → Executor → Verifier，
三个阶段在运行时逐步显示，不需要额外开关：

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

例程由 Automation Agent **动态规划**，不要求命中固定场景模板：

- `今天下午5点打球回到家，提前准备洗澡水和客厅降温` → 生成热水器与空调动作，分别安排在目标时间之前
- `明天早上6点叫我起床，提前准备洗澡和冲牛奶的热水` → 动态生成音响、热水器、烧水壶、窗帘或灯光动作
- `车辆到家前准备热水、空调和窗帘` → 生成车辆 ETA 相对动作，ETA 更新时只移动**尚未执行**的任务

创建例程一律先要求确认，且自动化动作**不含门锁解锁**。任务按 `dedupe_key` 去重，
持久化在 `data/automation.db`，CLI 后台调度线程负责执行；`/routines` 可查看每个例程的
待执行 / 完成 / 失败 / 取消数量。真实音响与汽车厂商 API 尚未绑定，当前通过
`SimulatorSpeakerBackend` 和 `VehicleSimulator` 完成本地闭环。

### 长期记忆

长期记忆默认保存到 `data/memories.db`。用户明确表达「记住我喜欢暖光」或
「以后睡觉时空调设为 25 度」时，Agent 可调用记忆工具保存偏好；单次控制指令、
临时感受和实时设备状态不会保存。除显式保存外，系统还会在设备操作中**观察**出偏好候选，
经用户确认后才落库（`list_preference_candidates` / `confirm_preference_candidate` /
`reject_preference_candidate`），修改留有版本历史（`list_memory_versions`）。

支持的范围：

- 家庭共享：`home_id`
- 房间共享：`home_id + room_id`
- 设备共享：`home_id + device_id`
- 个人及个人空间 / 设备：`home_id + user_id`，可附带 `room_id` 或 `device_id`

所有查询都强制包含 `home_id`。个人记忆只对所属用户可见；家庭、房间和设备共享记忆的
写入、修改和删除需要可信业务上下文中的管理员权限（`--admin`）。模型工具参数不能指定
任意 `home_id` 或 `user_id`。

> 偏好观察是**构造期的显式选择**：图路径开启观察，缺身份直接 `RuntimeError` 快速失败；
> 后台例程执行器与 MCP 服务器显式关闭观察（它们没有交互式用户身份）。

---

## 📚 说明书检索与混合检索

`docs/knowledge/` 是产品数据而非文档：**39 份说明书**覆盖 13 个型号（每个型号
`-errors` / `-maintenance` / `-symptoms` 三份），按 `## 小节` 切成 **124 个 chunk**。
`catalog.json` 登记了却找不到文件会在构造期抛 `FileNotFoundError`，不静默跳过。

### 检索流程

```
identify（实体消解）→ retrieve（硬过滤 → 双通道 → RRF）─┬→ rewrite（回环，最多 1 次）
                                                        ├→ self_check → answer（强制引用）
                                                        └→ refuse（答不出就明确拒答）
```

- **实体消解四态**：`resolved` / `ambiguous` / `no_model` / `unknown`。只有 `resolved` 放行检索，
  其余三态**不兜底成全库检索**——设备没登记型号就拒答，而不是猜。
- **两道硬过滤在打分之前**：型号逐字相等、查询里的错误码必须是该小节错误码的子集。
  这不是打分项而是准入条件——E4 的语义近邻天然是 E5/E7，降级成「相似度的一部分」
  就会拿 E7 的步骤回答 E4。
- **两套分数各管一件事**：`rrf`（k=60）只决定**名次**，`confidence`（[0,1]）只决定**放不放行**。
  绝不能拿 RRF 守门——它是纯名次的，第一名恒为满分，拒答分支会永远走不到。
- **弱信号先归零再谈合成**：两通道各有实测噪声基线（BM25 原始分 3.5、余弦 0.42），
  低于基线的既不进 `confidence` 也**不进名次**。RRF 奖励「在两个通道都出现」，而 BM25 的
  「出现」极其廉价，只在 confidence 处过滤会让噪声靠双通道在场击败正确答案。
- **自证分流**：排查清单每条末尾标注 `<!--check:xxx-->`（系统可用真实设备状态自证）或
  `<!--manual-->`（必须人工到现场）。引用未声明的 check id 会让 `KnowledgeBase` 构造直接失败。

### 实测召回（63 条标注用例）

```bash
python -m src.evaluation.recall            # legacy / bm25 / dense / hybrid 四种配置对比
python -m src.evaluation.recall --offline   # 只跑不需要 embedding 接口的两种
python -m src.evaluation.recall --sweep     # 权重 × 分数下限网格，换语料后重新标定
```

数据集 `evals/knowledge_recall.json`：口语 30 / 说明书原词 10 / 错误码 10 / 同码异义 6 / 应当拒答 7。

| 配置 | Recall@1 | Recall@3 | MRR | 拒答准确 | 口语类 | 说明书原词 |
|------|---------|---------|-----|---------|-------|-----------|
| legacy（纯词法覆盖率） | 51.8% | 57.1% | 0.539 | 7/7 | 3/30 | 10/10 |
| 仅 BM25 词法通道 | 58.9% | 62.5% | 0.604 | 7/7 | 7/30 | 10/10 |
| 仅向量通道 | 87.5% | **94.6%** | **0.908** | 7/7 | **24/30** | 9/10 |
| **混合（生产配置）** | **87.5%** | 89.3% | 0.881 | 7/7 | 23/30 | **10/10** |

**诚实说明**：Recall@1 从 51.8% 提到 87.5% 的功劳主要属于**向量通道**——纯向量单通道就已达到
87.5%，且 Recall@3 与 MRR 比混合更好。仍然选混合的理由是：说明书原词类混合保持 10/10
而纯向量掉到 9/10；离线（无 embedding）时 BM25 通道仍能工作；以及拒答闸门的可靠性来自
BM25 的硬性词法证据，而不是余弦阈值调得准（正例余弦中位 0.653 与困难负例 0.568 大幅重叠）。

**阈值全部是实测标定的，不是手调**：首轮准入 0.35、重写后 0.42、引用相对截断 0.7，
带错误码时下限为 0（这不是参数而是规则本身，刻意不可配置）。拒答准确率在下限
0.30 → 0.35 这一步从 85.7% 跳到 100%，但最高假阳性是 0.336，**离 0.35 只有 4% 余量**。
改语料、分词、打分或权重之后必须跑 `--sweep` 重新标定。

> 语料里有**两处刻意保留的缺口，不要顺手补全**：客厅灯不登记型号（守 `no_model` 拒答路径）、
> FrostLine-AC310 的症状手册没有噪音章节（守「本型号说明书查不到就必须拒答」）。两处都有测试钉着。

---

## 🔌 MCP 集成

### 作为服务端：把设备工具暴露给外部 AI

通过 [Model Context Protocol](https://modelcontextprotocol.io/) 对外暴露 **11 个工具**
（8 个设备控制 + `read_sensor` + `get_device_status` + `activate_scene`），复用图内同一份工具实现：

```bash
# 方式 1: stdio 模式（由 Claude Desktop 等 MCP 客户端启动）
python -m src.mcp.server

# 方式 2: SSE 模式（独立 HTTP 服务）
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

### 作为客户端：消费外部 MCP

支持 `stdio` / `sse` / `streamable_http` 三种传输。项目内置一个基于**彩云天气**的天气 MCP，
`.env.example` 默认通过当前 Python 环境启动它：

```dotenv
EXTERNAL_MCP_SERVERS=[{"name":"weather","transport":"stdio","command":"python","args":["-m","src.mcp.weather_server"]}]
WEATHER_DEFAULT_LOCATION=杭州
CAIYUN_WEATHER_TOKEN=你的彩云 token
```

彩云 token 可在 <https://dashboard.caiyunapp.com> 免费领取；没配置时天气工具返回明确提示
而不是报错。彩云只接受经纬度，城市名到坐标的转换由免费的 Open-Meteo geocoding 完成，无需额外 Key。

启动后可以问「杭州今天天气怎么样」或「北京未来三天天气如何」。天气 MCP 只提供实时天气和
预报查询（免费额度下预报最多 3 天），**不拥有任何设备控制权限**——这也是 `chat` 角色唯一能拿到的工具。

---

## 🧪 测试与静态检查

全部测试是 `unittest.TestCase`，pytest 与 unittest 都能跑，**不需要 `.env` 或 API Key**
（用 `SimpleNamespace` 手搓 settings、`FakeLLM` 替掉真实 LLM、`StubEmbeddings` 替掉向量通道）。

```bash
# 全量回归：当前基线 264 passed + 383 subtests
PYTHONIOENCODING=utf-8 python -m pytest -q

# 单文件 / 单测
python -m pytest -q tests/test_phase_seven.py
python -m pytest -q tests/test_automation_routines.py -k "wake_routine"

# unittest 亦可
python -m unittest tests.test_sensors -v

# 静态检查（当前基线全绿）
ruff check src tests --exclude "*.ipynb"
mypy src
```

22 个测试文件 / 53 个 TestCase / 264 个测试方法。测试验证的是**权限边界、数据库状态、
设备真实副作用、Checkpoint 恢复和事件顺序**，不是返回文本。

> Windows 上运行任何会打印设备名或 emoji 的命令都要加 `PYTHONIOENCODING=utf-8`，否则 `UnicodeEncodeError`。
>
> `tests/test_weather_mcp.py` 的 stdio 用例在受限沙箱里会因子进程被拦截而失败，真实环境可全绿。

RAG 相关改动的验证分三层，缺一层会漏掉真实缺陷：**单元测试**（确定性、无 Key、向量用 stub）→
**召回评测**（真实 embedding、无 LLM）→ **端到端手动跑**（真实 embedding + 真实 LLM + 完整主图）。
查询重写的 prompt 缺陷就只有第三层能暴露。

---

## 📁 项目结构

```
langgraph/
├── pyproject.toml              # 项目元数据 & 依赖 & ruff/mypy 配置
├── .env.example                # 环境变量模板
│
├── src/
│   ├── main.py                 # CLI 入口（Typer + Rich），status / mcp-server 两个子命令
│   ├── config.py               # 配置管理（pydantic-settings，7 个子配置段）
│   ├── models.py               # Pydantic v2 设备数据模型 + DeviceType 枚举
│   ├── progress_view.py        # 运行时进度事件的终端渲染
│   │
│   ├── agent/
│   │   ├── graph.py            # ★ LangGraph 工作流（17 节点 / 五条路径）
│   │   ├── context.py          # 可信请求身份与空间归属校验
│   │   ├── routing.py          # 结构化意图路由（8 类意图，模型 + 确定性兜底）
│   │   ├── planning.py         # Planner–Executor–Verifier 规划侧
│   │   ├── heuristics.py       # 确定性启发式判定（路由 / 规划 / 自动化共用词表）
│   │   ├── multi_agent.py      # 6 个角色定义与意图映射
│   │   ├── parallel.py         # 设备并行查询子图（dispatch → Send → aggregate）
│   │   ├── approval.py         # 敏感动作人工审批判定
│   │   ├── reasoning.py        # 记忆适用性推理
│   │   ├── session.py          # 会话创建、恢复与结束
│   │   ├── time_travel.py      # Checkpoint 历史回看
│   │   ├── observability.py    # 进度事件（stream + 结构化日志双写）
│   │   ├── telemetry.py        # LLM 调用级 token / 延迟采集 + 节点计时
│   │   ├── state.py            # Agent 状态定义
│   │   └── prompts.py          # 系统提示词
│   │
│   ├── devices/
│   │   ├── capabilities.py     # ★ 设备能力单一数据源（工具 / 规划 / 场景 / MCP 由此派生）
│   │   ├── base.py             # 设备后端抽象接口 + 注册中心 + 环境推演
│   │   └── simulator.py        # 内存模拟器（默认后端）
│   │
│   ├── tools/
│   │   ├── __init__.py         # build_all_tools 工厂（显式依赖注入）
│   │   ├── devices.py          # 设备工具工厂（从能力声明生成 control_xxx）
│   │   ├── scenes.py           # 场景模式工具工厂
│   │   ├── memory.py           # 长期记忆工具工厂 + 偏好观察
│   │   └── automation.py       # 自动化例程工具工厂
│   │
│   ├── knowledge/              # 说明书 Agentic RAG
│   │   ├── base.py             # catalog 加载、分节、清单解析、两道硬过滤
│   │   ├── retrieval.py        # BM25 + 向量双通道 + 加权 RRF
│   │   ├── embeddings.py       # embedding provider 协议 / 远程实现 / 显式空实现
│   │   ├── tokenizer.py        # jieba 分词 + 错误码正则（唯一数据源）
│   │   ├── resolution.py       # 实体消解四态
│   │   ├── selfcheck.py        # 排查清单的系统自证分流
│   │   └── rag.py              # RAG 子图 + 三档准入阈值
│   │
│   ├── automation/             # 图外的持久化调度子系统
│   │   ├── runtime.py          # 运行时组装（store / executor / scheduler / arrivals）
│   │   ├── executor.py         # 例程动作执行 + 验证（显式关闭偏好观察）
│   │   ├── scheduler.py        # 可注入虚拟时间的确定性调度
│   │   ├── store.py            # SQLite 持久化（data/automation.db）
│   │   ├── planning.py         # 自动化动作的合法性约束
│   │   ├── vehicle.py          # 车辆 ETA 模拟
│   │   └── speaker.py          # 音响后端模拟
│   │
│   ├── memory/
│   │   ├── models.py           # 长期记忆数据模型
│   │   ├── repository.py       # SQLite Repository
│   │   ├── service.py          # 隔离、权限与范围规则
│   │   ├── extractor.py        # 从对话抽取候选偏好
│   │   ├── summarizer.py       # 滚动摘要与上下文压缩
│   │   └── store.py            # 检查点记忆（内存 / SQLite）
│   │
│   ├── evaluation/
│   │   ├── recall.py           # 说明书召回评测（四配置对比 + 网格扫参）
│   │   └── trajectory.py       # 离线轨迹评测
│   │
│   ├── mcp/
│   │   ├── server.py           # MCP 服务器（11 个工具，复用图内实现）
│   │   ├── client.py           # MCP 客户端（stdio / sse / streamable_http）
│   │   └── weather_server.py   # 彩云天气 MCP
│   │
│   └── middleware/
│       └── interceptors.py     # 中间件示例 —— 刻意未接入运行路径，见「已知边界」
│
├── docs/knowledge/             # ★ 运行时资产：39 份说明书 + catalog.json（不是文档）
├── evals/knowledge_recall.json # 召回评测的 63 条标注用例
└── tests/                      # 264 个测试 + 383 subtests
```

> 仓库只保留跑起来所必需的内容。教程长文、迭代设计记录（001–013）、差距分析与可视化
> notebook 都只在开发者本地保留，不进云端——它们体积上会淹没仓库主体，且与运行无关。

---

## 🧩 扩展指南

### 新增一种设备：1 处声明 + 2 处手工

在 `src/devices/capabilities.py` 的 `CAPABILITIES` 加一条 `DeviceCapability` 声明
（设备类型、工具名、合法 action 及其副作用实现、期望状态、参数、类型关键词、
默认实例、场景归属）。其余**全部自动派生**：

- 控制工具的 JSON Schema / docstring（`src/tools/devices.py`）
- Planner 合法 action 词表与 `PlanStep.tool_name` 的 `Literal`（`src/agent/planning.py`）
- `registry.find()` 的类型关键词（`src/devices/base.py`）
- 模拟器默认设备（`src/devices/simulator.py`）
- 场景批量开关的设备类型集合（`src/tools/scenes.py`）
- 自动化例程允许的工具名（`src/automation/planning.py`）
- MCP 服务器暴露的控制工具（`src/mcp/server.py`）
- 多智能体 `device` 角色的工具集（从 `CONTROL_TOOL_NAMES` 派生）

仍需手工的只有两件（与能力声明本身无关）：

1. `src/models.py` — 设备数据模型 + `DeviceType` 枚举 + 字段范围约束
2. `src/agent/approval.py` — 若动作对外敏感（如解锁），加审批判定

`tests/test_capabilities.py` 会把所有派生点逐一钉住，漏任何一处都在测试阶段失败。

> 改造前这条路径要手工同步 9 处（含两处无法反射的副本），漏改的表现是
> 「Planner 第一版计划稳定失败，且不报错」——很难联想到根因。

### 新增一份说明书：3 处声明

1. `docs/knowledge/catalog.json` 登记（`id` / `title` / `model` / `file`），Markdown 用 `## 小节名` 分节。
   **`title` 必须与文件第一行 H1 逐字相同**；`file` 指向的文件不存在会在构造期抛 `FileNotFoundError`。
2. `src/devices/capabilities.py` 的默认实例里给对应设备填 `model`。**型号的唯一数据源是
   `BaseDevice.model`**，知识模块里不许再放型号映射表。不填 `model` 是合法的，表示「没登记说明书」。
3. 排查清单条目末尾挂 `<!--check:xxx-->`（`xxx` 必须是 `selfcheck.py:SELF_CHECKS` 里的 id）
   或 `<!--manual-->`。引用未声明的 id 会让构造抛 `ValueError`；反方向也有校验——
   声明了却没有任何语料引用的 check id 会让测试失败（那是加了检查却没人用的死代码）。

写正文的硬性约定：**一律用书面语**（写「雾量明显偏小」而不是「不怎么出雾」），口语留给查询侧，
否则「语义通道能不能跨过同义不同字」就无从验证；**不要出现 3 位以上的数字**（如「220 V」「180 秒」），
它们会被错误码正则当成假错误码，触发精确键过滤。

### 接入真实设备

1. 新建 `RealDeviceBackend(DeviceBackend)`，实现相同的接口方法（`get` / `update` 等）
2. 在 `src/main.py` 中将 `SimulatorBackend()` 替换为 `RealDeviceBackend()`
3. **工具层和 Agent 层无需任何修改** —— 面向接口编程的好处

---

## ⚙️ 环境变量说明

以下默认值全部与 `src/config.py` 一致。

### LLM 与全局

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LLM_MODEL_ID` / `BAILIAN_MODEL` | `qwen-plus` | 模型名称（通用名优先） |
| `LLM_API_KEY` / `BAILIAN_API_KEY` | — | API Key（**必填**，占位值会启动失败） |
| `LLM_BASE_URL` / `BAILIAN_BASE_URL` | 百炼兼容模式地址 | 服务地址 |
| `LLM_TIMEOUT` | `60` | LLM 请求超时（秒） |
| `LOG_LEVEL` | `INFO` | 日志级别 |

### 记忆与上下文（前缀 `CHECKPOINT_`）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `CHECKPOINT_DB_PATH` | `data/checkpoints.db` | 短期记忆持久化路径（留空 = 内存模式） |
| `ENABLE_LONG_TERM_MEMORY` | `true` | 是否启用结构化长期记忆（注意此项**不带前缀**） |
| `CHECKPOINT_LONG_TERM_DB_PATH` | `data/memories.db` | 长期记忆 SQLite 路径 |
| `CHECKPOINT_CONTEXT_MAX_MESSAGES` | `12` | 模型输入保留的最近消息上限 |
| `CHECKPOINT_CONTEXT_MAX_TOKENS` | `2400` | 模型输入的估算 token 上限 |
| `CHECKPOINT_TOOL_RESULT_MAX_CHARS` | `1200` | 单条工具结果保留字符上限 |
| `CHECKPOINT_SUMMARY_MAX_CHARS` | `1800` | 滚动摘要字符上限 |
| `CHECKPOINT_SESSION_TTL_HOURS` | `168` | 无活动会话检查点保留小时数 |
| `CHECKPOINT_RETRIEVAL_TOP_K` | `6` | 每轮检索注入的长期记忆条数上限 |

### 执行范式开关

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PLANNING_ENABLED` | `true` | Planner–Executor–Verifier 总开关 |
| `PLANNING_MAX_STEPS` | `8` | 单个计划最大步数（2–12） |
| `PLANNING_MAX_STEP_RETRIES` | `1` | 单步重试上限（0–3） |
| `PLANNING_MAX_REPLANS` | `1` | 重新规划上限（0–3） |
| `ROUTING_ENABLED` | `true` | 结构化意图路由开关 |
| `ROUTING_CONFIDENCE_THRESHOLD` | `0.6` | 低于此置信度转澄清 |
| `MULTI_AGENT_ENABLED` | `true` | 多智能体协作开关 |
| `MULTI_AGENT_MAX_HANDOFFS` | `2` | 角色交接上限（1–5） |

### 说明书 RAG（前缀 `RAG_`）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `RAG_ENABLED` | `true` | 是否走 RAG 路径（语料仍在启动期加载） |
| `RAG_KNOWLEDGE_PATH` | `docs/knowledge` | 语料目录（相对 cwd） |
| `RAG_TOP_K` | `3` | 检索条数（1–10） |
| `RAG_MAX_REWRITES` | `1` | 查询重写次数上限（0–3） |
| `RAG_BM25_WEIGHT` / `RAG_DENSE_WEIGHT` | `0.5` / `0.5` | 两通道权重，设 0 表示该通道**名次与置信度都不参与** |
| `RAG_MIN_SCORE` | `0.35` | 首轮命中准入下限（实测标定） |
| `RAG_REWRITTEN_MIN_SCORE` | `0.42` | 重写后命中准入下限 |
| `RAG_RELATIVE_FLOOR` | `0.7` | 引用的相对截断（不是拒答闸门，第一名永远保留） |
| `RAG_EMBEDDING_MODEL_ID` | `text-embedding-v4` | 语义通道型号；**留空则退化为纯 BM25** |
| `RAG_EMBEDDING_BASE_URL` / `RAG_EMBEDDING_API_KEY` | 空 | 留空回落到 `LLM_BASE_URL` / `LLM_API_KEY` |
| `RAG_EMBEDDING_DIMENSION` | `1024` | 向量维度（64–4096） |
| `RAG_EMBEDDING_CACHE_PATH` | `data/embeddings` | 按内容哈希落盘的向量缓存 |

> 阈值默认值必须与 `src/knowledge/rag.py` 的 `DEFAULT_*` 常量保持一致——两处不一致会让
> 「测试跑的」和「生产跑的」是两套阈值，而测试全绿。

### 自动化（前缀 `AUTOMATION_`）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `AUTOMATION_ENABLED` | `true` | 事件驱动自动化开关（关闭时自动化工具不出现在模型面前） |
| `AUTOMATION_DB_PATH` | `data/automation.db` | 例程与任务持久化路径 |
| `AUTOMATION_TIMEZONE` | `Asia/Shanghai` | 例程时间解析时区 |
| `AUTOMATION_POLL_SECONDS` | `1.0` | 调度线程轮询间隔（0.1–60） |

### MCP

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `EXTERNAL_MCP_SERVERS` | 空 | 外部 MCP 服务 JSON 数组，启动时发现并加载工具 |
| `WEATHER_DEFAULT_LOCATION` | 空 | 天气查询未提供城市时的默认城市 |
| `CAIYUN_WEATHER_TOKEN` | 空 | 彩云天气 token，缺失时天气工具返回明确提示 |
| `MCP_SERVER_ENABLED` / `MCP_SERVER_PORT` | `true` / `8765` | ⚠️ **预留但未接线**，见「已知边界」 |

---

## 🗺️ 架构演进

功能是分 13 次迭代长出来的，每一次都对应一个具体的失效场景：

| # | 主题 | 解决的问题 |
|---|------|-----------|
| 001 | 记忆与空间定位 MVP | 会话重启丢上下文；「这个房间」无从解析 |
| 002 | Human-in-the-loop 审批 | 批量设备操作和解锁不该由模型单方面决定 |
| 003 | Planner–Executor–Verifier | 多步任务在 ReAct 里会中途放弃或谎报成功 |
| 004 | 结构化意图路由 | 单一 ReAct 路径处理不了性质不同的请求 |
| 005 | 子图与动态并行 | 多设备查询串行调用，延迟随设备数线性增长 |
| 006 | Supervisor 多智能体 | 全部工具一起 bind，模型在 27 个工具里选错 |
| 007 | 记忆推理 / 时间旅行 / 进度事件 | 记忆该不该用需要判断；出错后无法回看 |
| 008 | Agentic RAG 与轨迹评测 | 设备故障问题模型只会编造 |
| 009 | 规划过程运行时可见 | 「Planner 只写不做」无法被用户观察到 |
| 010 | 事件驱动自动化 | 「明天 6 点叫我起床」需要图外的持久化调度 |
| 011 | 工程收口 | 设备能力散落 9 处；模块级单例让依赖隐式 |
| 012 | 说明书 RAG 升级 | 跨型号串答；排查清单里系统能自证的项也甩给用户 |
| 013 | 混合检索 | 口语提问命中率只有 3/30——词法通道跨不过同义不同字 |

完整的迭代方案文档与配套教程长文只在开发者本地保留（见「项目结构」末尾的说明）。

---

## ⚠️ 已知边界

诚实列出当前没做完或刻意搁置的部分：

- **`src/middleware/interceptors.py` 未接入运行路径**，是刻意的。三条理由写在文件头：
  无条件 `except Exception` 会吞掉 `GraphInterrupt` 破坏审批语义；叠加在
  `ChatOpenAI(max_retries=2)` 上会放大确定性错误；可观测性已由 loguru +
  `observability.py:emit_progress()` 承担。除包内 `__init__.py` 的 re-export 外没有调用方。
- **`MCP_SERVER_ENABLED` / `MCP_SERVER_PORT` 在代码里没有消费方**。Agent 启动时不会后台
  拉起 MCP 服务器；`mcp-server` 子命令用的是自己的 Typer 选项默认值。这两个配置是预留。
- **系统提示词的工具表只列了 5 个 `control_*` 工具**（`prompts.py`），
  `control_water_heater` / `control_lock` / `control_kettle` 靠 `bind_tools` 的 Schema 暴露而非提示词表。
- **拒答阈值余量只有 4%**：最高假阳性 0.336，下限 0.35。语料一变就可能越过，所以换语料必须重跑 `--sweep`。
- **真实硬件后端未接**：设备、音响、车辆 ETA 全部由模拟器闭环。接入路径见「扩展指南」。
- **传感器只读约束靠多处协同维持**，没有单点强制。`registry.tick_environment()` 只应由
  `read_sensor` 调用——别处调用会让同一轮对话里的读数随调用次数漂移。

---

## 📄 License

[MIT](LICENSE)

---

<p align="center">
  <sub>用 ❤️ 和 LangGraph 构建 · 有问题欢迎提交 Issue</sub>
</p>
