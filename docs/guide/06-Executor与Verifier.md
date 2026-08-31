[← 第 5 章 Planner：让 Agent 先说清要做什么](05-Planner规划器.md) · [目录](README.md) · [第 7 章 人在回路：让暂停活过进程重启 →](07-人在回路审批.md)

---

# 第 6 章 Executor 与 Verifier：不听模型自述，去查真实状态

## 6.1 要解决什么问题

第 5 章末尾留下的问题很具体：**工具返回的字符串不是事实。**

回顾第 3 章 3.5 的坑二——所有 handler 都忽略 `registry.update()` 的返回值：

```python
# capabilities.py:157-159 里 handler 的真实写法
def on(registry, device, args):
    registry.update(device.device_id, power=True)   # 返回值没人看
    return f"✅ {device.name}已打开，当前亮度 {device.brightness}%，色温 {device.color}。", None
```

更新失败了？工具照样回 ✅。ReAct 分支没有任何机制发现这件事——它只有工具返回的文本。

第二个问题是**失败之后怎么办**。「怎么办」有两个层次：

- 这次失败是**瞬时的**（超时、状态竞态）还是**确定性的**（计划本身写错了）？
- 如果是计划写错，重试同样的参数有意义吗？

这两个问题的答案完全不同，而 ReAct 把它们混在一起交给模型「自己想想」。

第三个问题是**控制流靠什么驱动**。ReAct 里模型说「我做完了」，循环就结束。如果模型判断错了呢？

本章讲的三样东西正好各解一个：

| 问题 | 解法 |
|---|---|
| 工具返回的不是事实 | `verify_step()` 执行后读注册中心真实状态对账 |
| 失败原因分不清 | `problem_type` 五分类，确定性错误跳过重试直接 replan |
| 控制流靠模型自述 | `planning_status` 显式状态机驱动全部条件边 |

## 6.2 代码怎么写的

### 6.2.1 `planning_status`：一个字段驱动全部条件边

`src/agent/state.py:66-68`：

```python
planning_status: NotRequired[
    Literal["planning", "awaiting_approval", "executing", "completed", "failed", "cancelled"]
]
```

六个取值。`CLAUDE.md:58` 一句话概括了它的地位：

> `planning_status` 字段驱动全部条件边，**不靠 LLM 自述成败**。

完整的状态流转：

```
task_router (graph.py:311)
  │  should_use_planner → planning_status="planning"          [graph.py:384]
  ▼
planner (graph.py:390)
  │  with_structured_output(ExecutionPlan)                     [graph.py:393]
  │  → "awaiting_approval", current_step_index=0, revision+=1  [graph.py:433-438]
  ▼ 固定边（无条件）                                            [graph.py:852]
plan_approval (graph.py:441)
  │  interrupt(request)  ⏸ 图暂停，状态落 checkpoint            [graph.py:444]
  ├─ 批准 → "executing"                                        [graph.py:450]
  └─ 拒绝 → "cancelled" ────────────────────────┐              [graph.py:450]
  ▼                                              │
executor (graph.py:453)                          │
  │  执行 1 步，status 保持 "executing"           │            [graph.py:502]
  ▼ 固定边                                        │            [graph.py:862]
verifier (graph.py:506)
  │  verify_step() 读真实设备状态                              [graph.py:510]
  │
  ├─ 成功 且还有下一步 → index+1, retry=0, "executing" → 回 executor   [:538-545]
  ├─ 成功 且是最后一步 → "completed" ──────────┐                       [:545]
  └─ 失败 → deterministic?                     │                       [:554]
       ├─ 否 且 retry ≤ max → "executing" → 回 executor                [:556]
       └─ 是 或额度耗尽 → replan_count+=1                              [:566]
            ├─ ≤ max_replans → "planning" → 回 planner（带失败反馈）    [:568]
            └─ > max_replans → "failed" ──┐                            [:568]
  ▼                                        ▼
planning_finalize (graph.py:596) ← ────────┘
  │  planning_active=False                                             [:617]
  ▼
END                                                                    [:883]
```

三条条件边的位置：

| 路由函数 | 定义 | 读哪个字段 |
|---|---|---|
| `route_task` | `graph.py:825-834` | `planning_active`（不读 status） |
| `route_after_plan_approval` | `graph.py:854-855` | `planning_status` |
| `route_after_verifier` | `graph.py:864-872` | `planning_status` |

**注意 `"awaiting_approval"` 这个值没有任何条件边读它**——`planner → plan_approval` 是固定边（`graph.py:852`）。它纯粹是给 `/plan` 命令和外部观察者看的语义标记。

**整个循环里 LLM 只被调用一次**（`graph.py:400`，生成计划）。执行、验证、路由、重试、终止全是确定性代码。这是本章最值得记住的一句话。

### 6.2.2 Executor：一步只调一个工具

`executor_node` 在 `graph.py:453-504`。骨架很朴素：

```python
step = steps[state["current_step_index"]]
device_id, expected_state, preparation_error = expected_state_for_step(step, registry)  # :469

try:                                     # :471
    if tool is None:
        tool_result = f"❌ 未注册工具 {step['tool_name']}"
    elif preparation_error:              # :474-475
        tool_result = f"❌ {preparation_error}"     # 工具根本不被调用
    else:
        tool_result = str(tool.invoke(step["arguments"], config=config))
except Exception as exc:
    tool_result = f"❌ 工具执行异常: {exc}"

return {"last_execution": {...}, "planning_status": "executing"}   # :494-502
```

两个细节：

**细节一：`expected_state_for_step()` 在 `try` 之外**（`:469` vs `:471`）。这就是 5.5 坑一的位置。

**细节二：`preparation_error` 会让工具根本不被调用**（`:474-475`）。计划写错的那类失败**在碰设备之前就被拦下**——零副作用地失败，然后重规划。

Executor 自己**不判断成败**。它只把「执行了什么、拿到什么」写进 `last_execution` 字段（`:494-501`），成败交给下一个节点。

### 6.2.3 Verifier：对账的三行核心

`verify_step()` 在 `src/agent/planning.py:197-256`。docstring 第一句（`:204`）：

> Verify execution using the actual registry state, **not model self-report**.

判定顺序严格短路，**顺序本身就是设计**：

| 序 | 条件 | 行 | `problem_type` |
|---|---|---|---|
| 1 | `device_id is None`（设备名解析不出来） | `:206-211` | `device_not_found` |
| 2 | `preparation_error` 非空 | `:212-218` | `unsupported_action` |
| 3 | `tool_result` 以 `❌` 开头 | `:219-225` | `tool_error` |
| 4 | 执行后 `registry.get(device_id)` 为 None | `:227-234` | `device_not_found` |
| 5 | 逐字段比对有差异 | `:237-249` | `state_mismatch` |
| 6 | 全部一致 | `:250-256` | `none` |

对账的核心是这三行（`planning.py:235-241`）：

```python
actual = {name: _plain_value(getattr(device, name, None)) for name in expected_state}
expected = {name: _plain_value(value) for name, value in expected_state.items()}
mismatches = {
    name: {"expected": expected[name], "actual": actual[name]}
    for name in expected
    if actual[name] != expected[name]
}
```

三个要点：

**1. 只比 `expected_state` 里出现的字段**（`for name in expected_state`）。

这是「部分验证」。`set_volume` 的期望状态只声明了 `{"volume": ...}`（`capabilities.py:543`），所以只查音量，不管电源；而 `set_brightness` 声明了 `{"power": True, "brightness": ...}`（`capabilities.py:428-431`），会连电源一起查。

**期望状态里的字段集就是验证的范围**——写在第 3 章那条声明里。

**2. `getattr(device, name, None)`** —— 期望状态的 key 必须和设备对象的属性名同名。这是一条隐式契约，**写错 key 会静默拿到 `None` 然后判 mismatch**（不会报错说「你 key 写错了」）。

**3. `_plain_value()`**（`planning.py:259-260`）：`getattr(value, "value", value)`，把枚举拆成裸值再比。不做这一步，`DeviceType.AC` 和 `"ac"` 会被判不等。

`mismatches` 会原样进 `reason`（`:246`），所以 replan 反馈里能看到「期望 X，实测 Y」的逐字段明细。

### 6.2.4 期望状态从哪来：第 3 章那条链的另一半

第 3 章讲了 `ActionSpec` 有 `handler` 和 `expected` 两个 lambda。现在看 `expected` 这一半怎么用。

`capabilities.py:59-62` 的类型注释说清了两者的分工：

> expected 签名：(arguments, device) -> 期望状态字典。与 handler 的差异：handler 写真实副作用，expected 只描述"做完之后设备应该是什么样"，**Verifier 拿它跟注册中心的真实状态比对。mute 这类翻转语义必须读 device。**

几个实例：

| action | 期望状态 | 位置 |
|---|---|---|
| `control_light` `on` / `off` | `{"power": True}` / `{"power": False}` | `capabilities.py:422-423` |
| `control_light` `set_brightness` | `{"power": True, "brightness": clamp(0,100)}` ← **顺带断言开机** | `:428-431` |
| `control_curtain` `open` / `close` | `{"position": 100}` / `{"position": 0}` | `:585-586` |
| `control_tv` `set_volume` | `{"volume": clamp(0,100)}` ← 只验音量 | `:543` |
| `control_tv` `mute` | `{"muted": not device.muted}` ← **唯一读 device 的翻转语义** | `:552` |

`expected_state_for_step()`（`planning.py:174-194`）负责把声明变成实际的期望字典，返回三元组 `(device_id, expected_state, error)`。五个分支：

| 分支 | 行 | 返回 |
|---|---|---|
| 工具名不认识 | `:180-182` | `(None, {}, "unsupported tool")` |
| 设备解析不出来 | `:183-185` | `(None, {}, "device not found or ambiguous")` |
| action 不在合法集里 | `:187-190` | `(device_id, {}, _unsupported_action(...))` |
| 参数解析失败 | `:191-194` | `(device_id, {}, str(exc))` |
| 正常 | `:192` | `(device_id, action_spec.expected(args, device), None)` |

### 6.2.5 失败分流：这是整章最精妙的十行

`graph.py:548-556`。**注释比代码长，而且注释就是设计文档**：

```python
retry_count = state.get("step_retry_count", 0) + 1
max_retries = getattr(getattr(settings, "planning", None), "max_step_retries", 1)
# unsupported_action / device_not_found 是确定性错误：计划本身写错了，
# 用同一批参数原样重放不可能成功，只会白白耗掉重试额度、拖慢自愈。
# 直接跳到 replan 分支，把失败原因（已带合法值列表）交回 Planner 重写。
# tool_error / state_mismatch 可能是瞬时的（超时、状态竞态），仍然可重试。
deterministic = verification.problem_type in ("unsupported_action", "device_not_found")
if retry_count <= max_retries and not deterministic:
```

这十行背后是一个**归因分层**：

| 错误类别 | 谁错了 | 该在哪一层修 | 重试有意义吗 |
|---|---|---|---|
| `unsupported_action` / `device_not_found` | **计划错了**（Planner 的输出错） | Planner → replan | **无。** 同参数重放必然同样失败 |
| `tool_error` / `state_mismatch` | **执行环境有问题**（超时、竞态、设备离线） | Executor → retry | 有。可能是瞬时的 |

注意 `and not` 的写法（`:555`）：确定性错误**连一次重试都不做**。`retry_count` 虽然被 +1 写回（`:590`），但不影响本次分流。

`docs/gap-analysis.md` 把这个设计列为「已经做对的、改动时不要回退」的第二条：

> 识别出「同样参数重放不可能成功」这一点，比无脑重试高一个层次。

### 6.2.6 replan 反馈：让重试有信息增量

失败后回喂给 Planner 的不是一句「失败了」。`graph.py:577-586`：

```python
feedback = (
    f"步骤 {execution['step']['step_id']}（{execution['step']['description']}）失败："
    f"{verification.reason}。工具结果：{execution['tool_result']}"
)
if verification.problem_type == "device_not_found":
    feedback += (
        "\n设备名未能解析。请仅从下列可用设备中选择，"
        "device_name 必须与设备名称逐字一致（不要添加'的''那台'等修饰）：\n"
        + registry.get_device_list_prompt()
    )
```

三层信息：

1. **无条件带上**：失败步骤的 id、原文 description、`verification.reason`（对 `state_mismatch` 而言就是逐字段的 expected/actual 明细）、工具返回的原始文本。
2. **`device_not_found` 额外追加**：完整可用设备清单 + 一条「逐字一致」的格式指令。
3. **`unsupported_action` 走另一条路**（更早注入）：`_unsupported_action()` 在 `expected_state_for_step` 阶段就把合法值拼进 `preparation_error`（`planning.py:169-171`），随后成为 `reason`，再被拼进 feedback。

回喂链路：`graph.py:593` 写进 state → `graph.py:398` 读出 → `planning.py:117` 形参 → `planning.py:132-133` 注入 prompt 的「上一次执行失败信息：」段。

测试钉住了这条链路（`tests/test_phase_seven.py:344`）：

```python
self.assertIn("device not found", fake.planner_prompts[-1])
```

**replan 时被清空和被保留的字段不一样**（画状态图时要注意）：

| 字段 | replan 时 | 为什么 |
|---|---|---|
| `current_step_index` | 归零（`:434`） | 新计划从第一步开始 |
| `step_retry_count` | 归零（`:435`） | 新计划的重试额度重置 |
| `plan_revision` | +1（`:433`） | v1 → v2 |
| `replan_count` | +1（`:566`） | 消耗全局额度 |
| `planning_results` | **跨版本保留**（`:529-536` 累加） | `/plan` 要能看到两版计划的完整轨迹 |

### 6.2.7 三个预算常量

`src/config.py:73-79`（`env_prefix="PLANNING_"`）：

```python
enabled: bool = Field(default=True)
max_steps: int = Field(default=8, ge=2, le=12)
max_step_retries: int = Field(default=1, ge=0, le=3)
max_replans: int = Field(default=1, ge=0, le=3)
```

默认值意味着：**每步最多执行 2 次**（1 + 1 retry），**整个任务最多 2 版计划**（1 + 1 replan），**每版最多 8 步**。最坏情况 32 次工具调用 + 2 次 LLM 规划调用，**有界**。

读取处全部用双层 `getattr` 兜底（`graph.py:403`、`:549`、`:567`），settings 缺字段也不崩——这是为了让测试能用 `SimpleNamespace` 手搓 settings（第 17 章讲）。

## 6.3 关键设计决策

### 决策一：为什么值得多一次 LLM 调用

`docs/defense-script.md:330`：

> 换来三件 ReAct 给不了的东西：执行前人能看到**完整**动作清单（ReAct 是走一步才知道下一步）、每步有客观验证、失败有预算化的重试和重新规划。所以我只在'多设备多动作'时付这个代价……而且故意写得保守。

「故意写得保守」对应的就是第 5 章那个判定函数——它宁可漏判，不肯多判。

### 决策二：为什么不让 LLM 自评成败

已经引过一次，值得再看一遍（`docs/defense-script.md:333`）：

> 因为 LLM 自评会说谎 —— 它会把'我调用了工具'当成'设备状态正确'。……**唯一的事实来源是设备状态，不是模型的自述。** 反思的产物是结构化的 `VerificationResult`，里面带 `problem_type`，这个字段直接决定控制流走重试还是重新规划。

最后半句是关键：**`problem_type` 不是给人看的日志，它是控制流的输入。** 一个模型自述的字符串没法驱动条件边，一个五值枚举可以。

### 决策三：`state_mismatch` 这个分类的价值

五个 `problem_type` 里，`state_mismatch` 是最能体现整套设计价值的那个：**工具报告成功，但设备真的没变。**

在 ReAct 分支里，这种情况**完全不可见**——工具回了 ✅，模型信了，用户也信了。只有「执行后再读一次真实状态」这个动作能把它揪出来。

第 3 章 3.5 坑二里那个「`registry.update()` 返回值被忽略」的问题，在 Planner 路径上被 Verifier 兜住了，在 ReAct 路径上没有。**这是两条路径真实存在的可靠性差异。**

## 6.4 动手试一试

### 实验 A：看到 Verifier 的对账账本

```bash
PYTHONIOENCODING=utf-8 "F:/Software/Anaconda/envs/langgraph/python.exe" -m src.main --trace
```

输入 `关闭客厅灯，然后打开卧室空调到25度`，批准计划，然后盯住 `step_verified` 事件里的两个字段（`graph.py:525-526`）：

```
step_verified  step_id=1  success=true
               expected_state={'power': False}
               actual_state={'power': False}
```

这两个字典就是账本。左边来自 `capabilities.py` 的声明，右边来自 `registry.get()` 的真实读取。

执行完输入 `/plan`，从 checkpoint 里把同一份轨迹再取一遍。`main.py:169-170` 的注释解释了为什么要有这个命令：

> 进度事件是'流过去就没了'，这个命令从 checkpoint 里把同一份轨迹再取出来，用于事后复盘：哪一步重试过、Verifier 比对的期望值和实测值分别是什么。

### 实验 B：制造一次 replan（`device_not_found` 路线，最容易复现）

**为什么这条最容易**：默认有 3 盏灯（客厅灯 / 卧室灯 / 厨房灯，`capabilities.py:447-449`），所以任何不存在的灯名都会被 `registry.find` 的策略 3 拒绝猜测（第 3 章讲过：多候选时拒绝猜测）。

输入：

```
关闭书房灯，然后打开卧室空调
```

必走 Planner（2 动作 + 2 设备类 + 连接词「然后」），且必失败（「书房灯」三级策略全落空）。批准后观察事件流：

```
step_started      step_id=1
step_executed     tool_result=❌ device not found or ambiguous
                  ↑ 工具根本没被调用，这是 preparation_error 变成的文本
step_verified     success=false  problem_type=device_not_found
（注意：没有 step_retry 事件！）
replan_requested  replan_count=1  max_replans=1  accepted=true
plan_generated    ← v2 计划，设备名应该已被纠正
（第二次审批面板）
```

**「没有 step_retry 事件」就是「确定性错误跳过重试」的可观测证据**（`graph.py:554-555`）。

批准 v2 后 `/plan` 复盘：`plan_revision` = 2，`replan_count` = 1。

### 实验 C：把额度调成 0，看明确失败

```bash
PLANNING_MAX_REPLANS=0 PYTHONIOENCODING=utf-8 "F:/Software/Anaconda/envs/langgraph/python.exe" -m src.main --trace
```

重跑实验 B。这次第一次失败之后直接 `planning_status="failed"`，输出确定性文案（`graph.py:607`）：

```
多步骤任务未能完成，已停止继续执行。最后失败原因：...
```

注意这句话不是模型生成的，是代码写死的。**终止路径上不调 LLM**——失败时最不该依赖的就是可能也在出问题的那个组件。

### 实验 D：看到唯一能触发 `step_retry` 的场景

`state_mismatch` 在模拟器里很难自然发生（内存字典不会写失败）。测试用打桩造出来了（`tests/test_phase_seven.py:290-314`）：

```python
with patch.object(self.registry, "update", side_effect=flaky_update):
    # 第一次 update 返回 False，第二次正常
```

跑它并看断言：

```bash
PYTHONIOENCODING=utf-8 "F:/Software/Anaconda/envs/langgraph/python.exe" -m pytest tests/test_phase_seven.py -k "failed_step_retries" -v
```

断言是「step 1 出现**两条**结果记录，第一条失败第二条成功，最终 `completed`」——这是 `max_step_retries=1` 生效的样子。

## 6.5 踩坑与局限

**坑一：`mute` 的期望状态不幂等。**

`{"muted": not device.muted}`（`capabilities.py:552`）读的是**执行前**的状态。所以同一步重试时，期望值会**跟着翻转**。

作者自陈（`docs/defense-script.md:348`）：

> 有一个 —— 静音是取反语义……期望值依赖执行前的状态，所以**同一步重试时期望值会跟着翻转**。目前静音不在多设备规划的常见路径上，但这是设计上的真实局限，正确做法是把取反类动作在规划阶段就固化成绝对目标值。

后台自动化执行器有同款问题（`max_attempts=2` 的重试假设动作幂等）。

**坑二：`handler` 和 `expected` 是两个独立 lambda，靠人对齐。**

第 3 章末尾提过这一点，这里能看到具体后果：空调 `on` 的 `expected` 不含 `fan_speed`，但 `handler` 会写 `fan_speed`。**结果是验证器不校验风速**——风速写错了，Verifier 不会发现。

同一条声明里两个函数，没有任何机制保证它们描述的是同一件事。

**坑三：期望状态的 key 写错不会报错。**

`getattr(device, name, None)` 在属性不存在时返回 `None`，所以 key 打错（比如 `"powre"`）的表现是**永久 mismatch**，而不是「你 key 写错了」。debug 时会一直怀疑设备，不怀疑声明。

**坑四：ReAct 路径没有 Verifier。**

这不是 bug，是范围。但值得明确写下来：**只有 Planner 路径有对账**。单步请求「打开客厅灯」走 ReAct，工具回 ✅ 就是 ✅，没有第二次确认。

两条路径的可靠性不同，这件事对使用者是不可见的。

**这一章的局限**：状态机让执行过程可控、可验证、可收敛了。但有一个环节我们一直跳过没讲——`plan_approval` 那个「⏸ 图暂停」到底是怎么实现的？暂停期间程序在干什么？如果这时候进程挂了会怎样？

**下一章的问题**：怎么让一次「等用户点确认」的暂停，在进程重启之后还能接着跑？

---

[← 第 5 章 Planner：让 Agent 先说清要做什么](05-Planner规划器.md) · [目录](README.md) · [第 7 章 人在回路：让暂停活过进程重启 →](07-人在回路审批.md)
