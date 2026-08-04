# 阶段九：子图与动态并行

## 目标

将已经稳定的设备查询流程封装为可独立测试的子图，并使用 LangGraph `Send` 根据运行时设备数量动态 fan-out，再用 reducer 聚合结果。

## 实现

- `src/agent/parallel.py`：定义查询子图状态、目标解析、`Send` fan-out 和结果聚合。
- `src/agent/graph.py`：当结构化路由识别为多设备查询时，进入 `device_query_subgraph`，不调用普通 ReAct。
- `AgentState`：保存 `parallel_query_results`，便于观察并行分支结果。
- 查询结果使用 `Annotated[list[dict], operator.add]` 合并，最终按设备 ID 排序，避免并行完成顺序影响用户看到的结果。

## 当前边界

阶段九只并行化无副作用的设备状态查询。多设备控制仍由阶段七 Planner 顺序执行，并保留计划确认、Verifier 和重试/重规划；预定义场景仍由阶段六 Human-in-the-loop 保护。后续可在明确依赖关系和失败策略后，扩展到安全的独立控制任务。
