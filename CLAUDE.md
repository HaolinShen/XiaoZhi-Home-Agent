# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 开发环境

- 必须使用已有的 Conda 环境 `langgraph`，解释器路径 `F:\Software\Anaconda\envs\langgraph\python.exe`。不要用 `uv run`、不要创建 `.venv` 或其它虚拟环境，也不要在用户未明确要求时安装/升级/卸载包。
- Windows 下运行任何会打印设备名或 emoji 的命令都要加 `PYTHONIOENCODING=utf-8`，否则 `UnicodeEncodeError`。

## 常用命令

```bash
# 全量测试（当前基线 189 项 + 137 subtests；tests/test_weather_mcp.py 的 stdio 用例
# 在受限沙箱里会因子进程被拦截而失败，真实环境可全绿）
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

`pyproject.toml` 里有 `[tool.ruff]` / `[tool.mypy]` 配置（011 起，渐进式不 strict；环境未安装这两者时先 `pip install -e ".[dev]"` 再 `ruff check src tests` / `mypy src`）。没有 `conftest.py`。

## 架构大局

实际实现已到阶段十三（事件驱动自动化）+ 011 工程收口（能力单一数据源/显式注入/可观测性），迭代方案记录在 `docs/iterations/NNN-*.md`，`docs/tutorial.md` 是最完整的架构长文。

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
- `should_use_planner()`（`src/agent/heuristics.py`，`planning.py` 保留再导出）是纯正则判断：≥2 个动作词 且（≥2 类设备 或 出现连接词）。命中预定义场景关键词则一律不走 Planner。

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
- **011 起**：偏好观察是**构造期的显式选择**。图路径 `build_device_tools(registry, service)` 开启观察，缺身份直接 `RuntimeError`（fail-fast）；后台执行器 / MCP 用 `enable_preference_tracking=False` 显式关闭。旧陷阱（空 configurable 的 config 让 `if config is not None` 恒为真）已随模块级单例一并移除，不要再写「逐键检查后安静跳过」的兜底。

### 工具通过工厂显式注入依赖（011 起）

工具不读模块级单例：`src/tools/__init__.py:build_all_tools(registry, *, memory_service, automation_runtime, external_tools, enable_preference_tracking)` 按依赖构建全部工具并闭包持有。`build_graph` 新增 `automation_runtime` 参数；`automation_runtime=None`（自动化未启用）时自动化工具不出现在 Agent 面前。`set_registry` / `set_memory_service` / `set_automation_runtime` 已删除——测试别再 import 它们。

### 自动化子系统运行在图之外

`src/automation/` 是独立的持久化调度子系统，不是图的一部分：

- `AutomationRuntime`（`runtime.py`）组装 store / executor / scheduler / arrivals，由 `src/main.py` 创建并启动后台线程。
- `RoutineScheduler.tick(now=...)` 可注入虚拟时间，所以调度测试是确定性的、不睡眠。
- 任务按 `dedupe_key` 去重；车辆 ETA 更新只移动**尚未执行**的任务。
- 数据落 `data/automation.db`（`data/` 与 `*.db` 均被 gitignore）。
- 创建例程一律走人工审批，且自动化动作**不含门锁解锁**。
- `RoutineExecutor` 构造自己的设备工具集（偏好观察关闭）；动作仍经 `verify_step` 按真实设备状态验证。

### 多智能体是工具集隔离

`multi_agent.enabled` 时，`graph.py` 为 6 个角色（device / scene / memory / automation / knowledge / chat）各 `bind_tools` 一个子集，`agent_node` 按 `delegated_agent` 选用。device 角色的工具名从 `capabilities.CONTROL_TOOL_NAMES` 派生（新增设备自动进入）；**新增非设备工具若不加进对应角色的名字集合，该角色就永远调不到它。**

`automation` 角色额外有一层强制：`required_automation_tool()`（`src/agent/heuristics.py`，graph.py 里保留 `_required_automation_tool` 别名）在识别出「未来触发信号 + 设备动作信号」同时出现时锁定必须调用的创建工具，模型没调就补一条纠正 SystemMessage 重试一次。该函数默认必须返回 `None` —— 查询类和取消类请求绝不能被判成创建请求。

## 改代码时容易漏的同步点

### 新增一种设备：1 处声明 + 2 处手工（011 起）

旧规则「要改 9 处」已作废。现在只在 `src/devices/capabilities.py` 的 `CAPABILITIES`
加一条 `DeviceCapability` 声明（action 的 handler/expected/参数/关键词/默认实例/
scene_exit），工具 Schema、Planner 词表、`PlanStep` Literal、`registry.find` 关键词、
模拟器默认实例、场景批量类型、自动化工具名、MCP 工具全部自动派生。
仍需手工的只有两件：

1. `src/models.py` — 设备数据模型 + `DeviceType` 枚举 + 字段范围约束
2. `src/agent/approval.py` — 对外敏感动作（如解锁）的审批判定

一致性由 `tests/test_capabilities.py` 的生成式断言兜底；漏一处 = 测试失败，而非运行期静默。

### action 枚举已无副本（011 起）

`DEVICE_ACTION_SPECS`、`TOOL_ACTIONS`、`PLANNING_TOOL_NAMES`、`expected_state_for_step()`、`PlanStep.tool_name` 的 `Literal`、工具实现、MCP 工具现在**同源于** `devices/capabilities.py`。旧的两处手写副本（工具 if/elif、Literal）已不存在；`tests/test_capabilities.py` + `tests/test_phase_seven.py` 的双向反射用例继续钉住「声明 ↔ 工具实现」的一致性。

### 进度事件双写 + token/延迟度量（011 起）

`emit_progress()`（`src/agent/observability.py`）双写：stream 通道（CLI 实时渲染不变，CLI 仍用 `graph.stream(..., stream_mode=["custom", "updates"])`）+ loguru 结构化日志（`channel=graph_progress`，DEBUG 级），invoke/后台路径也有痕迹。事件分两级：`PLANNING_EVENTS` 默认显示，`TRACE_EVENTS` 需 `--trace`。

LLM 调用级 token/延迟由 `src/agent/telemetry.py:UsageTracer` 采集（挂在 `build_llm`，落 `channel=llm_usage` 日志）；`traced_node` 装饰器给关键节点计时（`channel=node_latency`）。

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
- 代码注释用中文，写「为什么这样」而不是「这是什么」；曾经踩过的坑就地记在注释里（如 `capabilities.py` 顶部解释为何要把设备能力收敛成单一数据源、`tools/memory.py:make_preference_recorder` 解释旧「逐键检查后安静跳过」的陷阱为何被构造期显式选择取代）。
