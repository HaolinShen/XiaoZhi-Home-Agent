# 009 规划过程的运行时可见性

## 1. 问题

003 已经把 Planner / Executor / Verifier 拆成了三个独立节点，数据结构上分得很干净：
计划先完整产出（`plan`），审批后才逐步执行（`last_execution`），判断由 Verifier 单独
给出（`last_verification`）。但**在实际运行时**用户看不出这个分工，一次多动作请求在
终端上只有两行：

```text
已生成 2 步执行计划，是否开始执行？
多步骤任务已完成，共验证通过 2 个步骤。
```

中间谁在规划、谁在调工具、谁在比对状态，全都不可见。

根因不在图里，而在消费方式：节点里的 `emit_progress` 依赖 `get_stream_writer()`，
而 `get_stream_writer()` 只有在 `graph.stream(...)` 下才拿到真正的写入器 ——
`graph.invoke()` 会给一个空写入器，事件被静默丢弃，不报错也不警告。CLI 一直用
`invoke`，所以 007 加的那批 `custom` 事件从来没有到过终端。

## 2. 改动

三层，各自不越界：

| 位置 | 职责 | 依赖 |
| --- | --- | --- |
| `src/agent/observability.py` | 声明事件词表 `PLANNING_EVENTS` / `TRACE_EVENTS`，发出事件 | 只依赖 LangGraph |
| `src/agent/graph.py` | 在各阶段节点里发事件，并补齐载荷 | 不知道有终端存在 |
| `src/progress_view.py` | `PlanProgressView` 把事件渲染成表格和彩色行 | 依赖 rich，不知道有图 |
| `src/main.py` | 改用 `stream` 消费图，把事件交给 view | 表现层 |

### 2.1 事件载荷按职责边界切分

事件字段刻意照抄 Executor / Verifier 的分工，让终端上的分工是数据结构的形状，而不是
解说词：

- `plan_generated` 带上每步的 `tool_name` 和 `arguments`。它必然早于第一个
  `step_started`，也就是说参数全部定下来时设备一点没动 —— 这是「Planner 只写不做」
  最直接的证据；
- `step_executed` 只报告 `tool_result`（工具说了什么），不含任何结论；
- `success` / `problem_type` / `expected_state` / `actual_state` 只出现在
  `step_verified` 里。

新增事件：`planning_selected`（进入规划分支及原因）、`plan_decision`（批准或拒绝）、
`step_executed`、`step_retry`、`replan_requested`。原有 007 的事件名一个没改，
`tests/test_phase_eleven.py` 的断言继续成立。

### 2.2 CLI 改用多模式 stream

```python
for mode, chunk in graph.stream(payload, config, stream_mode=["custom", "updates"]):
    if mode == "custom":
        view.handle(chunk)                    # 进度事件 → 终端
    elif mode == "updates":
        interrupts = chunk.get("__interrupt__")
        if interrupts:
            pending = interrupts[0].value     # 审批中断照旧
```

多模式 `stream` 产出 `(mode, chunk)` 二元组；审批中断以 `updates` 分支里的
`{"__interrupt__": (Interrupt(...),)}` 到达，`Command(resume=...)` 在同一
`thread_id` 上继续，人在回路不受影响。事件已经边跑边打印，最终状态改为从
Checkpoint 读（`graph.get_state(config).values`），不再依赖返回值。

`console.status` 的「思考中」保持在底部，等 LLM 时仍有反馈；`🤖 小智:` 标签移到跑完
之后打印，否则规划过程会插在标签和回复中间。

### 2.3 渲染层的两个约束

- 未知事件一律忽略：以后图里新增事件，旧 CLI 不会崩；
- `TRACE_EVENTS`（路由、记忆判断、并行查询）默认折叠，`--trace` 才显示，避免淹没
  规划过程。

## 3. 运行效果

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
✘ Verifier 步骤 1/2 未通过 · device_not_found
↻ 重试步骤 1（第 2/2 次尝试）
⟲ 重试额度已用尽，把失败原因交回 Planner 重新规划（第 1/1 次）
🏁 规划结束 · completed · 验证通过 2 次 / 共 2 次尝试 · 最终计划 v2
```

结束行写成「验证通过 X 次 / 共 Y 次尝试」而不是「X/Y 步」：`planning_results` 记录的
是每次尝试，失败重试和被放弃的旧版本计划都在里面，按步数读会对不上。

## 4. `/plan` 复盘命令

进度事件是流过去就没了。`/plan` 从 Checkpoint 里把同一份轨迹重新取出来：一张
Planner 产出表（执行前就已确定），一张 Executor + Verifier 轨迹表（每次尝试一行，
带计划版本号）。适合事后回答「哪一步重试过」「当时期望值和实测值分别是什么」。

## 5. 测试

`tests/test_planning_progress.py`，分两组：

图侧（`stream_mode="custom"`，不依赖终端）

1. `plan_generated` 早于任何 `step_started`，且已带齐工具名与参数，此时设备未变；
2. 批准后每步严格是 `step_started → step_executed → step_verified`，`step_executed`
   不含 `success`，期望/实测只在 `step_verified`；
3. 失败路径依次发出两次失败验证、`step_retry`、`replan_requested`，并产出 v2 计划；
4. 拒绝计划时不发 `step_started`，结束状态为 `cancelled`，设备未变。

终端侧（`Console(file=StringIO())`，不跑图）

5. 计划表格含工具名、参数和「尚未触碰任何设备」；
6. Verifier 通过/未通过两条路径都展示期望与实测；
7. `TRACE_EVENTS` 默认不输出，`--trace` 才输出，且不影响 `planning_seen`；
8. 未知事件和空字典都被忽略；
9. `PLANNING_EVENTS` 里每个名字都有对应的 `_on_*` 渲染方法，两张词表不重叠；
10. 格式化函数处理空值与超长值截断。

全量：123 passed。
