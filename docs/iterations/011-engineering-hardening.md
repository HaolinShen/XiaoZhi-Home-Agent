# 011 · 工程收口：单一能力数据源、显式依赖注入与统一可观测性

## 背景与根因

项目迭代到阶段十三时，结构上积累了四类互相关联的技术债，全部指向同一个根因：
**信息与依赖分散在多份手工副本里，漏改不报错，错误延迟到运行期才爆发**。

1. **新增设备要手工同步 9 处**（CLAUDE.md 记录，实测还有第 10、11 处）。
   其中工具实现的 `if action == ...` 与 `PlanStep.tool_name` 的 `Literal` 无法反射，
   漏改一处的表现是「Planner 第一版计划稳定失败，且不报错」——这是最坏的失败模式。
2. **模块级可变单例**（`set_registry` / `set_memory_service` / `set_automation_runtime`）
   让测试必须记得复位、后台执行器的无身份调用只能靠「逐键检查后安静跳过」兜底，
   把 bug 藏进静默分支（曾因此把定时热水器动作整个判成失败）。
3. **可观测性只有 stream 模式**：`emit_progress` 走 `get_stream_writer()`，
   `graph.invoke()` 下事件被静默丢弃，后台自动化执行器完全无观测；
   全项目没有 token/延迟采集。
4. **同类正则/关键词表散落三处**（routing 兜底、Planner 判定、自动化强制工具），
   「打开/关闭」至少出现三次，改一处容易漏另一处。

本迭代按依赖顺序完成四项收口（P0–P4）。

## 方案

### P0：设备能力单一数据源（消除 9+2 处手工同步）

新增 `src/devices/capabilities.py`，用 `DeviceCapability` 声明每个可控制设备的全部信息：

- 设备类型、工具名（`control_xxx`）、中文类别名、docstring 素材；
- 每个 action 的：喂 Planner 的签名文本、docstring 说明、**期望状态函数**、
  **副作用实现 handler**（含 clamp、水箱/离线前置检查）、偏好观察声明；
- 工具级参数（进 JSON Schema）、类型关键词（`registry.find` 策略 3）、
  模拟器默认实例、离家/睡眠批量行为（`scene_exit`）。

派生视图全部从声明生成，禁止手写：

| 派生点 | 原实现 | 现实现 |
|---|---|---|
| 控制工具（Schema/docstring/副作用） | `tools/devices.py` 手写 669 行 if/elif | `build_device_tools()` 从声明生成 |
| Planner 词表 `DEVICE_ACTION_SPECS` / `TOOL_ACTIONS` | `agent/planning.py` 手写 | 从声明派生 |
| `PlanStep.tool_name` 的 `Literal` | 手写 + 一致性用例钉住 | `Literal[tuple(PLANNING_TOOL_NAMES)]` |
| `registry.find` 类型关键词 | `devices/base.py` 手写 `keywords_map` | `TYPE_KEYWORDS` |
| 模拟器默认设备 | `devices/simulator.py` 手写 140 行 | `CAPABILITIES` + `SENSOR_DEFAULT_DEVICES` |
| 场景批量关闭类型 | `tools/scenes.py` 手写 DeviceType 元组 | `scene_exit` 分组派生 |
| `AutomationToolName`（第 11 处副本） | `automation/planning.py` 手抄清单 | `Literal[tuple((*PLANNING_TOOL_NAMES, "set_alarm"))]` |
| 多智能体 device 角色工具集 | `graph.py` 手写 `device_tool_names` | `CONTROL_TOOL_NAMES` + 只读工具 |
| MCP 控制工具（第 10 处副本） | `mcp/server.py` 把 8 个 if/elif 再抄一遍 | 工厂工具的薄包装，副作用同源 |

`tests/test_capabilities.py` 用生成式断言把每条派生链路钉住：漏任何一处都在
测试阶段失败，而不是运行期静默。

### P1：移除模块级单例，改显式依赖注入

- `src/tools/__init__.py` 提供 `build_all_tools(registry, *, memory_service, automation_runtime, external_tools, enable_preference_tracking)`；
  `set_registry` / `set_memory_service` / `set_automation_runtime` 全部删除。
- `build_graph` 新增 `automation_runtime` 参数；长期记忆服务先于工具构建，
  工具工厂以闭包持有依赖。
- 身份语义从「调用期逐键检查后安静跳过」改成**构造期显式选择**：
  - 图路径：偏好观察开启，缺身份 `RuntimeError` fail-fast；
  - 后台执行器（`automation/executor.py`）与 MCP 服务器：构造期显式关闭
    偏好观察——机器触发的动作本就不该计入「重复手动操作」。
  - 自动化未启用（runtime=None）时自动化工具根本不出现在 Agent 面前，
    比旧行为「调用时报尚未初始化」更安全。

### P2：统一可观测性（invoke / 后台路径 + token / 延迟）

- `emit_progress` 双写：stream 通道（保留 CLI 渲染）+ loguru 结构化日志
  （`channel=graph_progress`）；`get_stream_writer()` 在图外会抛异常，已容错。
- 新增 `src/agent/telemetry.py`：
  - `UsageTracer`（LangChain 回调）挂在 `build_llm`，随 `bind_tools` /
    `with_structured_output` 传播，兼容 `usage_metadata` 与旧 `token_usage`
    两代字段名，落 `channel=llm_usage` 结构化日志；
  - `traced_node` 装饰器给关键节点计时，覆盖没有 LLM 调用的节点。

### P3：启发式判定收敛到单一模块

新增 `src/agent/heuristics.py`，收拢三处判定：

- `should_use_planner`（原 `planning.py`）
- 路由兜底关键词表（原 `routing.py`）
- `required_automation_tool`（原 `graph.py`）

真正相同的词表只定义一次（`ACTION_CORE`），各判定在其上做**有语义的扩展**
并注释差异原因（路由要宽线索、Planner 要精确动作、自动化要定时语境动作）。
行为与迁移前逐字一致，由 `tests/test_heuristics.py` 钉住；旧导入路径
（`planning.should_use_planner`、`graph._required_automation_tool`）保留再导出。

### P4：工程卫生

- `pyproject.toml` 增加 `[tool.ruff]` / `[tool.mypy]` 配置与 dev 依赖声明
  （渐进式，不 strict；安装需用户许可，配置先行）。
- `README.md` 同步到阶段十三 + 本迭代：五路业务路径图、完整设备清单、
  项目结构、环境变量表；「新增设备 4 步（实为 9 处）」改为「1 处声明」。

## 行为兼容性

- 所有设备工具的入参 Schema、返回文本、clamp 边界、前置检查与改造前一致；
  三个启发式判定逐字一致；场景批量行为一致（类型集合改由声明派生，成员相同）。
- 有意的行为变化仅两处，均已用新用例显式钉住：
  1. 图路径偏好观察缺身份时由「静默跳过」改为 `RuntimeError`；
  2. 自动化未启用时自动化工具从 Agent 的工具集消失。

## 测试

- 新增 `tests/test_capabilities.py`（P0 派生一致性 + P1 注入语义）、
  `tests/test_heuristics.py`（P3 判定行为）、`tests/test_telemetry.py`（P2 采集）。
- 既有用例按工厂模式更新（`set_registry` 等已删除的接口不再使用）。
- 全量：`PYTHONIOENCODING=utf-8 python -m pytest -q` → 189 passed + 137 subtests。

## 验证注意事项（DSH 沙箱环境）

在 DSH 沙箱内跑本套件需要两个环境适配（**非代码问题，真实环境不需要**）：

1. 沙箱拦截 `tempfile.mkdtemp`（0o700 ACL）目录内的 SQLite 写入，表现为
   `sqlite3.OperationalError: unable to open database file`。绕过：把 `TMP`/`TEMP`
   指向工作区目录，并在启动 pytest 前补丁 `tempfile.mkdtemp` 使用默认 mode；
2. 沙箱禁止带管道 stdio 的子进程，`tests/test_weather_mcp.py::test_stdio_mcp_...`
   会因 `WinError 5` 失败（`src/mcp/client.py` 拉起天气 MCP 子进程），与本次改动无关。

## 后续方向

- 结构收口后，拟真语料回归集（intent/route/planner 判定的准确率）才有稳定的底座；
- token 度量落盘后，可验证「多智能体工具集隔离是否省 token」这一悬空假设。
