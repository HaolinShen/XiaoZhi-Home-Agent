# 003 Planner–Executor–Verifier 规划执行循环

## 1. 目标

为自定义多设备目标增加显式规划、逐步执行、真实状态验证、有限重试和重新规划能力，同时保留已有 ReAct 和 Human-in-the-loop 分支。

## 2. 路由边界

```text
单设备操作             → ReAct
预定义场景             → ReAct + 场景确认
自定义多动作设备任务     → Planner–Executor–Verifier
```

阶段七使用保守规则识别多步骤任务：请求必须包含多个动作，并涉及多种设备或明显的顺序/并列连接词。

## 3. 图结构

```text
sync_context
  ↓
task_router
  ├── ReAct branch
  └── planner
        ↓
      plan_approval (interrupt)
        ↓ approved
      executor → verifier
                   ├── next/retry → executor
                   ├── replan     → planner
                   └── finish     → planning_finalize
```

每次重新规划都会生成一个新版本计划，并再次暂停等待用户确认。

## 4. 组件职责

- Planner：通过 `with_structured_output(ExecutionPlan)` 生成 2 至 8 个原子步骤；
- Executor：每次只调用一个已有设备工具；
- Verifier：根据工具参数推导预期状态，并读取 `DeviceRegistry` 的实际状态；
- Retry：当前步骤验证失败时有限重试；
- Replan：重试耗尽后，将失败原因反馈给 Planner；
- Finalizer：生成完成、取消或失败结果。

## 5. 循环限制

```text
PLANNING_MAX_STEPS=8
PLANNING_MAX_STEP_RETRIES=1
PLANNING_MAX_REPLANS=1
```

达到限制后任务明确结束，避免无限工具调用和无限重新规划。

## 6. 测试

`tests/test_phase_seven.py` 验证：

1. 只有自定义多动作请求进入 Planner；
2. 计划批准前设备不发生变化；
3. 批准后逐步执行并验证实际状态；
4. 拒绝计划后所有步骤都不执行；
5. 状态不匹配时重试当前步骤；
6. 重试耗尽后重新规划并再次确认；
7. 图重建后可通过 SQLite Checkpoint 恢复已批准计划；
8. 重试和重新规划额度耗尽后任务明确结束。
