# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 开发环境

- 必须使用已有的 Conda 环境 `langgraph`，解释器路径 `F:\Software\Anaconda\envs\langgraph\python.exe`。不要用 `uv run`、不要创建 `.venv` 或其它虚拟环境，也不要在用户未明确要求时安装/升级/卸载包。
- Windows 下运行任何会打印设备名或 emoji 的命令都要加 `PYTHONIOENCODING=utf-8`，否则 `UnicodeEncodeError`。

## 常用命令

```bash
# 全量测试（当前基线 158 项，含 subtests）
PYTHONIOENCODING=utf-8 "F:/Software/Anaconda/envs/langgraph/python.exe" -m pytest -q

# 单文件 / 单个测试
... -m pytest -q tests/test_phase_seven.py
... -m pytest -q tests/test_automation_routines.py -k "wake_routine"

# unittest 亦可（测试全部是 unittest.TestCase）
... -m unittest tests.test_sensors -v

# 启动 CLI（需要 .env 里的 LLM_API_KEY）
python -m src.main
python -m src.main --trace          # 额外显示路由/记忆判断等诊断事件
python -m src.main --admin          # 以家庭管理员身份运行，才能写家庭共享记忆

# MCP 服务器
python -m src.mcp.server                              # stdio
python -m src.mcp.server --transport sse --port 8765  # SSE
```

没有 lint / format / typecheck 配置，也没有 `conftest.py`；`pyproject.toml` 里除 pytest 依赖外无 pytest 配置节。

## 架构大局

`README.md` 描述的是阶段十二之前的形态。实际实现已经到阶段十三（事件驱动自动化），迭代方案记录在 `docs/iterations/NNN-*.md`，`docs/tutorial.md` 是最完整的架构长文。

### 一次请求走过的图

`src/agent/graph.py:build_graph()` 是单个 `StateGraph`，编译出的图里有五条互斥业务路径。入口固定是三段前置处理：

```
sync_context → memory_reasoner → task_router → ┬→ planner ⇄ plan_approval ⇄ executor ⇄ verifier → planning_finalize
                                               ├→ compact_context → agent ⇄ (approval) ⇄ tools     ← ReAct 主路
                                               ├→ device_query_subgraph   （Send 动态并行查询）
                                               ├→ knowledge_rag           （本地 Agentic RAG）
                                               └→ clarification           （信息不足直接反问）
```

- `sync_context`（graph.py:246）落定本轮的可信空间（`active_room_id` / `active_device_id`），并检索长期记忆写入 `memory_context`。
- `task_router`（graph.py:343）先用 `src/agent/routing.py` 分类意图，再决定走哪条路。**注意 `classify_intent()` 会在确定性信号命中 `automation_management` 时直接返回 fallback，不问 LLM** —— 这是防止模型把「明天 6 点回家前开空调」误判成预定义「回家模式」的硬约束。
- `should_use_planner()`（`src/agent/planning.py:245`）是纯正则判断：≥2 个动作词 且（≥2 类设备 或 出现连接词）。命中预定义场景关键词则一律不走 Planner。

### Planner–Executor–Verifier 是显式状态机

`planning_status` 字段（`src/agent/state.py`）驱动全部条件边，不靠 LLM 自述成败：

- `verifier`（graph.py:536）用 `verify_step()` 读**注册中心的真实设备状态**跟 `expected_state` 比对。
- 失败后的分支取决于 `problem_type`：`unsupported_action` / `device_not_found` 是确定性错误，**跳过重试直接 replan**（同样参数重放不可能成功）；`tool_error` / `state_mismatch` 才消耗 `max_step_retries`。
- replan 时 `planning_failure_feedback` 会带上合法 action 列表和设备清单回喂 Planner。
- Planner 走 `llm.with_structured_output(ExecutionPlan)` 而**不是** `bind_tools`，所以工具 docstring 到不了模型面前 —— 合法 action 必须由 `planning.py:TOOL_ACTIONS` 显式喂进 prompt。

### 可信身份边界（安全核心）

身份**永远**来自 `RunnableConfig["configurable"]`，绝不接受 LLM 生成的 `home_id` / `user_id`：

- `AgentContext.to_config()`（`src/agent/context.py:42`）把 `session_id` 同时用作 LangGraph `thread_id`，并把身份塞进 `configurable`。
- 工具从 `config` 反解身份：`src/tools/automation.py:_identity()`、`src/tools/memory.py:_context()`。
- `SpaceDirectory.validate()` 校验 `room_id` / `device_id` 的住宅归属；`MemoryService` 另有作用域与管理员权限规则（家庭/房间/设备共享记忆需 `is_admin`）。
- **陷阱**：后台自动化执行器直接 `tool.invoke(arguments)` 时不带身份，但 LangChain 仍会注入一个 `configurable` 为空的 config，`if config is not None` 拦不住。凡是从 config 取身份的**尽力而为型**副作用（如 `record_preference_operation`）必须逐键检查后安静跳过，否则会把整个定时动作判成失败。

### 工具通过模块级单例拿依赖

工具函数不接收 registry / service 参数，靠启动时注入：`set_registry()`（`src/tools/__init__.py:44`，同时注入 devices 和 scenes）、`set_memory_service()`、`set_automation_runtime()`。测试里改完必须在 `tearDown` 复位（尤其 `set_automation_runtime(None)`）。

### 自动化子系统运行在图之外

`src/automation/` 是独立的持久化调度子系统，不是图的一部分：

- `AutomationRuntime`（`runtime.py`）组装 store / executor / scheduler / arrivals，由 `src/main.py` 创建并启动后台线程。
- `RoutineScheduler.tick(now=...)` 可注入虚拟时间，所以调度测试是确定性的、不睡眠。
- 任务按 `dedupe_key` 去重；车辆 ETA 更新只移动**尚未执行**的任务。
- 数据落 `data/automation.db`（`data/` 与 `*.db` 均被 gitignore）。
- 创建例程一律走人工审批，且自动化动作**不含门锁解锁**。

### 多智能体是工具集隔离

`multi_agent.enabled` 时，`graph.py:201` 为 6 个角色（device / scene / memory / automation / knowledge / chat）各 `bind_tools` 一个子集，`agent_node` 按 `delegated_agent` 选用。**新增工具若不加进对应角色的名字集合，该角色就永远调不到它。**

`automation` 角色额外有一层强制：`_required_automation_tool()`（graph.py:98）在识别出「未来触发信号 + 设备动作信号」同时出现时锁定必须调用的创建工具，模型没调就补一条纠正 SystemMessage 重试一次。该函数默认必须返回 `None` —— 查询类和取消类请求绝不能被判成创建请求。

## 改代码时容易漏的同步点

### 新增一种设备要改 9 处

`README.md` 说 4 步，实际上（参照 `git show bcbf8b9` 新增热水器/门锁/烧水壶）需要：

1. `src/models.py` — 设备模型 + `DeviceType` 枚举 + 字段范围约束
2. `src/devices/simulator.py` — 注册默认实例
3. `src/devices/base.py` — `DeviceRegistry.find()` 的 `keywords_map` 加关键词
4. `src/tools/devices.py` — `@tool def control_xxx`
5. `src/tools/__init__.py` — 导入 + `get_all_tools()` + `__all__`
6. `src/agent/graph.py` — `device_tool_names` 集合（否则 device 角色拿不到）
7. `src/agent/planning.py` — `DEVICE_ACTION_SPECS` 加一个 `ToolSpec` 条目，外加 `PlanStep.tool_name` 的 `Literal`（`PLANNING_TOOL_NAMES`、`TOOL_ACTIONS`、`expected_state_for_step()` 都从声明派生，不用动）
8. `src/mcp/server.py` — 对应 MCP 工具
9. `src/tools/scenes.py` — 相关场景（如离家模式该不该关它）

对外敏感动作（如解锁）还要在 `src/agent/approval.py` 加审批判定。

### action 枚举还剩两处副本

`DEVICE_ACTION_SPECS`（`planning.py:75`）是规划侧的单一数据源，`TOOL_ACTIONS`、`PLANNING_TOOL_NAMES`、`expected_state_for_step()` 全从它派生。剩下两处仍需手工对齐：

1. **工具实现的 `if/elif`**（`src/tools/devices.py`）—— 带各自的副作用文本和前置检查，无法反射。
2. **`PlanStep.tool_name` 的 `Literal`** —— structured output 靠它生成 JSON Schema 的 enum，无法从 dict 派生。

`tests/test_phase_seven.py` 用一致性用例双向钉住这两处：实际调用工具看是否回「不支持的操作」，并用 `get_args()` 比对 `Literal` 与声明的键集。漏一处的表现是 Planner 第一版计划稳定失败，且不报错。

### 进度事件只在 stream 模式存在

`emit_progress()`（`src/agent/observability.py`）走 `get_stream_writer()`。`graph.invoke()` 下 LangGraph 给一个空写入器，事件被静默丢弃。CLI 因此必须用 `graph.stream(..., stream_mode=["custom", "updates"])`（`src/main.py:_stream_segment`）。事件分两级：`PLANNING_EVENTS` 默认显示，`TRACE_EVENTS` 需 `--trace`。

### 传感器是只读的，这个约束靠多处协同

传感器故意没有 `control_xxx` 工具；`PlanStep.tool_name` 的 `Literal` 不含 `read_sensor`；`get_device_list_prompt()` 把传感器单列成只读组；场景批量操作不能遍历所有设备去关（会关掉传感器，表现为「Agent 说家里没人」，很难联想到根因）。`registry.tick_environment()` 只应由 `read_sensor` 调用 —— 别处调用会让同一轮对话里的读数随调用次数漂移。

## 测试约定

- 全部是 `unittest.TestCase`，用 pytest 或 unittest 都能跑。
- 不需要 `.env` / API Key：测试用 `types.SimpleNamespace` 手搓 settings（绕开 `Settings` 的 API Key 校验），用 `patch("src.agent.graph.build_llm", return_value=FakeLLM())` 替掉真实 LLM。FakeLLM 需实现 `bind_tools` / `invoke`，走 Planner 的还要实现 `with_structured_output`。
- **Windows 资源清理**：`TemporaryDirectory.cleanup()` 遇到未关闭的 SQLite 连接会抛 `PermissionError: [WinError 32]`，测试本体断言通过也会判失败。凡在临时目录里建过 `AutomationStore` / `MemoryRepository` / `build_graph()`（它内部会建 `MemoryRepository`，通过 `graph.memory_repository` 暴露），都要在目录删除前 `close()`。
  - **顺序陷阱**：unittest 先跑完整个 `tearDown()` 才轮到 `doCleanups()`，所以把 `temp_dir.cleanup()` 写在 `tearDown` 里会早于测试方法内 `addCleanup(repo.close)` 注册的关闭动作。正确做法是在 `setUp` 里 `self.addCleanup(self.temp_dir.cleanup)` —— `doCleanups` 是 LIFO，最先注册就最后执行（见 `tests/test_automation_routines.py:50`）。
- **不要硬编码未来日期**。`AutomationRuntime.schedule_wake()` / `create_scheduled_routine()` 会校验「目标时间必须晚于当前时间」，写死的日期一过就让用例永久失败。用 `datetime.now(...) + timedelta(...)` 构造锚点。反之，`scheduler.schedule(..., now=...)` + `scheduler.tick(虚拟时间)` 这条路径不做该校验，固定日期在那里是安全且刻意的（保证调度断言确定）。
- 测试验证的是权限边界、数据库状态、设备真实副作用、Checkpoint 恢复和事件顺序，不是返回文本。新增功能照这个标准补用例。

## 提交与文档约定

- 提交信息用中文，`feat:` / `docs:` 前缀，body 说明**根因**而非现象（参见 `git log`）。修 bug 时找根因改，不做表面兜底。
- 新迭代方案写 `docs/iterations/NNN-主题.md`，序号递增不复用，并同步 `docs/iterations/README.md` 的索引表。
- 代码注释用中文，写「为什么这样」而不是「这是什么」；曾经踩过的坑就地记在注释里（如 `planning.py:62` 解释为何要单独维护一份 action 声明、为何把三处副本合成一处）。
