# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 开发环境

- 必须使用已有的 Conda 环境 `langgraph`，解释器路径 `F:\Software\Anaconda\envs\langgraph\python.exe`。不要用 `uv run`、不要创建 `.venv` 或其它虚拟环境，也不要在用户未明确要求时安装/升级/卸载包。
  - 013 起新增三个已装依赖（用户明确许可后装的）：`rank-bm25`、`jieba`、`numpy`。`numpy` 以前只是被别的包顺带装上、未在 `pyproject.toml` 声明，013 起显式声明——靠传递依赖存在的包，换个环境就没有。
- Windows 下运行任何会打印设备名或 emoji 的命令都要加 `PYTHONIOENCODING=utf-8`，否则 `UnicodeEncodeError`。

## 常用命令

```bash
# 全量测试（当前基线 264 项 + 383 subtests；tests/test_weather_mcp.py 的 stdio 用例
# 在受限沙箱里会因子进程被拦截而失败，真实环境可全绿）
PYTHONIOENCODING=utf-8 "F:/Software/Anaconda/envs/langgraph/python.exe" -m pytest -q

# 单文件 / 单个测试
... -m pytest -q tests/test_phase_seven.py
... -m pytest -q tests/test_automation_routines.py -k "wake_routine"

# unittest 亦可（测试全部是 unittest.TestCase）
... -m unittest tests.test_sensors -v

# 说明书检索的召回评测（013 起）。改了语料、分词、打分或阈值都要跑
python -m src.evaluation.recall              # legacy / bm25 / dense / hybrid 四种配置对比
python -m src.evaluation.recall --offline     # 只跑不需要 embedding 接口的两种
python -m src.evaluation.recall --sweep       # 权重 × 分数下限网格，换语料后重新标定

# 启动 CLI（需要 .env 里的 LLM_API_KEY）
python -m src.main
python -m src.main --trace          # 额外显示路由/记忆判断等诊断事件
python -m src.main --admin          # 以家庭管理员身份运行，才能写家庭共享记忆

# MCP 服务器
python -m src.mcp.server                              # stdio
python -m src.mcp.server --transport sse --port 8765  # SSE
```

`pyproject.toml` 里有 `[tool.ruff]` / `[tool.mypy]` 配置（011 起，渐进式不 strict）。环境已安装这两个工具（用户许可后装的），**当前基线全绿**：`ruff check src tests` 与 `mypy src` 均零错误——改完代码要保持这个状态再提交。注意 `tests/visualize_graph.ipynb` 会被 ruff 扫到（可用 `--exclude "*.ipynb"` 跳过，notebook 里的 E402/I001 不算项目代码问题）。`mypy` 的 `python_version` 配置是 `3.12`（环境真实版本；写 3.11 会因 numpy stub 用了 `type` 语句而无法解析）。没有 `conftest.py`。

## 架构大局

实际实现已到阶段十三（事件驱动自动化）+ 011 工程收口（能力单一数据源/显式注入/可观测性）+ 012 说明书 RAG 升级（实体消解/自证分流/强制引用）+ 013 混合检索（BM25 与向量双通道/名次与准入分离/召回评测），迭代方案记录在 `docs/iterations/NNN-*.md`，`docs/tutorial.md` 是最完整的架构长文。

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

### 说明书 RAG 的声明点（012 起，013 扩展）

新增一份说明书要动三处，缺一处都会在构造期或测试里失败，不会静默：

1. `docs/knowledge/catalog.json` 登记文档（`id` / `title` / `model` / `file`），Markdown 用 `## 小节名` 分节。**`title` 必须与文件第一行的 H1 逐字相同**，`file` 指向的文件不存在会在构造期抛 `FileNotFoundError`（013 起；012 时代是静默跳过）。
2. `src/devices/capabilities.py` 的默认实例里给对应设备填 `model`。**型号的唯一数据源是 `BaseDevice.model`**，知识模块里不许再放型号映射表。设备不填 `model` 是合法的，表示「没登记说明书」，检索会拒答而不是猜。
3. 排查清单条目末尾挂行内标注：`<!--check:xxx-->`（系统可自证，`xxx` 必须是 `src/knowledge/selfcheck.py:SELF_CHECKS` 里的 id）或 `<!--manual-->`（必须人工到现场）。**引用未声明的 id 会让 `KnowledgeBase` 构造直接抛 `ValueError`**；反方向也有校验 —— **声明了却没有任何语料引用的 check id 会让测试失败**（那是死代码，加了检查却没人用，"诊断没变强"不会自己报错）。

写新说明书的硬性约定：**正文一律用书面语**（写「雾量明显偏小」而不是「不怎么出雾」）。口语留给查询侧，否则「语义通道能不能跨过同义不同字」就无从验证。正文里**不要出现 3 位以上的数字**（如「220 V」「180 秒」），它们会被错误码正则当成假错误码，触发精确键过滤。

**两处刻意保留的语料缺口，不要顺手"补全"**：客厅灯不登记 `model`（守 `no_model` 拒答路径）、FrostLine-AC310 的症状手册没有噪音章节（守「本型号说明书查不到就必须拒答」）。两处都有测试钉着。

### 混合检索的三条不变量（013 起）

`src/knowledge/` 从"零依赖词法检索"变成双通道混合检索，新增 `tokenizer.py`（jieba 分词 + 错误码正则唯一数据源）、`embeddings.py`（provider 协议 + 远程实现 + 显式的"无语义通道"）、`retrieval.py`（BM25 + 向量 + 加权 RRF）。`base.py` 现在只做 catalog 加载、分节、清单解析和**两道硬过滤**，打分全部委托出去。改这一块前先读三个模块的 docstring，里面写了每个常量为什么是这个值。

1. **两套分数不能混。** `rrf` 只决定名次，`confidence`（[0,1]）只决定放不放行，三档下限套在后者身上。**绝不能拿 RRF 去守门** —— 它是纯名次的，第一名恒为满分，跟像不像无关，拒答分支会永远走不到。
2. **硬过滤在两个通道之前。** 型号相等、错误码 issubset 是准入条件不是打分项。向量检索让这条更要紧：E4 的语义近邻天然是 E5/E7，降级成"相似度的一部分"就会拿 E7 的步骤回答 E4。
3. **弱信号要先归零，再谈合成。** 两个通道各有噪声基线（BM25 原始分 3.5 / 余弦 0.42），低于基线既不进 `confidence` 也**不进名次**。只在 confidence 处过滤是不够的：RRF 奖励"在两个通道都出现"，而 BM25 的"出现"极其廉价，噪声会靠双通道在场击败只有一个通道支持的正确答案。

阈值与常量**全部是实测标定的**，不是手调：改语料、分词、打分或权重之后必须 `python -m src.evaluation.recall --sweep` 重标，并更新 `docs/iterations/013-hybrid-retrieval.md` 里的数字。BM25 噪声基线依赖语料规模（IDF 与平均文档长度都是语料级统计量），小语料上照抄这个值会让所有候选都进不了名次。

`RAG_MIN_SCORE` / `RAG_REWRITTEN_MIN_SCORE` / `RAG_RELATIVE_FLOOR` 在 `RAGConfig` 里，**默认值必须和 `rag.py` 的 `DEFAULT_*` 常量一致** —— 两处不一致会让"测试跑的"和"生产跑的"是两套阈值，而测试全绿。第三档「带错误码时下限为 0」刻意不可配置：那不是参数，是规则本身。

`relative_floor`（默认 0.7）是**引用精度**的相对截断：置信度不到第一名 0.7 倍的候选不进引用。它管的是"同一份手册的兄弟小节一起被召回"（同一台电器的不同症状语义本来就近，绝对下限提高会连正确答案一起砍）。**它不是拒答闸门** —— 参照点是第一名，第一名永远保留。

**查询重写的 LLM prompt 必须留"都不符合"的出口。** `_llm_rewrite` 让模型在该型号真实存在的小节标题里挑一个，但强制单选会在正确答案不在清单里时逼模型挑一个最像的，而校验只查"标题真实存在"——存在不等于相关，于是重写后的查询轻松过阈值，拿着不相关的章节权威作答。模型回「无」时要保持原句去查一次然后拒答，**连词表兜底都不走**（词表只看用户措辞，不知道这个型号有没有对应章节）。这个漏洞 012 就有，因为所有单元测试都传 `llm=None`，只有跑真实 LLM 的端到端验证才暴露。

其它别踩的坑：`KnowledgeBase.search()` 的 `model` 是**关键字必填**参数（全库检索必须是调用方的显式选择，不能是漏传参数的副产品）；实体消解四态里只有 `resolved` 放行检索，**其余三态不许兜底成全库检索**；带错误码的查询永不重写；自证检查只能拿到 `CheckContext`，碰不到 `registry`，更不能调 `tick_environment()`；测试里的向量通道用 `StubEmbeddings` 注入，**不要在测试里打真实 embedding 接口**（测试不需要 API Key，且换模型不能让断言全变）。

**验证要覆盖配置的真实组合，不只是单元测试。** 013 的验证分三层：单元测试（确定性、无 Key、向量通道用 stub）、召回评测（真实 embedding、无 LLM）、端到端手动跑（真实 embedding + 真实 LLM + 完整主图）。上面那个重写漏洞前两层都测不到。改完 RAG 相关代码后，至少手动跑一遍主图的知识查询。

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
