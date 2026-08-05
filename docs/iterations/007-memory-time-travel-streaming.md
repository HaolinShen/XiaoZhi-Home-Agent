# 阶段十一：记忆推理、时间旅行与进度事件

## 1. 显式记忆推理

`sync_context` 不再只拼接记忆文本，还会保存结构化 `retrieved_memories`。随后 `memory_reasoner` 输出 `MemoryDecision`：适用、忽略、约束、偏好、冲突以及是否需要澄清。当前明确的临时指令可以覆盖一般偏好，但不会绕过家庭约束。

## 2. Checkpoint 时间旅行

`src/agent/time_travel.py` 提供：

- `list_state_history`：查看 checkpoint ID、时间、下一节点、元数据和状态字段；
- `fork_from_checkpoint`：通过 `graph.update_state` 从历史 checkpoint 创建实验分支，不覆盖原始轨迹。

## 3. 自定义进度事件

关键节点通过 `get_stream_writer` 发出 `custom` 事件，例如：

- `context_synced`
- `memory_reasoned`
- `supervisor_routing`
- `plan_generated`
- `step_started`
- `step_verified`
- `parallel_query_completed`
- `agent_completed`

调用方可使用 `graph.stream(..., stream_mode="custom")` 获取这些事件。事件只描述可验证的任务进度，不包含模型隐藏推理文本。
