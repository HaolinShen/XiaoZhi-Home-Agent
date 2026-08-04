# 阶段八：结构化意图路由

## 目标

在 `sync_context` 后增加结构化 `IntentResult`，把设备查询、设备控制、场景控制、记忆管理、普通对话和需要澄清的请求区分开。路由只负责选择业务路径，不直接执行工具。

## 实现

- `src/agent/routing.py`：定义 `IntentResult`、结构化路由提示词和无模型时的确定性回退分类器。
- `AgentState`：保存 `intent`、`intent_confidence`、`intent_reason` 和 `intent_route`。
- `task_router`：优先使用 `llm.with_structured_output(IntentResult)`；调用失败时回退规则分类。
- 低于置信度阈值（默认 `0.6`）或分类为 `clarification` 时，进入澄清节点，不调用设备工具。
- 复杂多动作请求仍优先进入阶段七 Planner 分支，保持原有确认、执行、验证和重规划行为。

## 配置

```text
ROUTING_ENABLED=true
ROUTING_CONFIDENCE_THRESHOLD=0.6
```

## 边界

当前结构化路由已经记录意图和置信度，但设备查询、设备控制和记忆管理仍复用同一个 ReAct Agent。下一阶段再将这些业务路径拆成子图，并在独立任务中引入动态并行。
