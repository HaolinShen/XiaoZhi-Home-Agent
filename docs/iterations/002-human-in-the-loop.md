# 002 Human-in-the-loop 批量设备操作确认

## 1. 目标

为智能家居 Agent 增加可恢复的人工确认流程。模型提出批量设备操作后，LangGraph 在真实工具执行前暂停；只有收到同一会话中的明确批准才继续执行。

## 2. 当前确认策略

当前所有 `activate_scene` 调用都需要确认，因为场景会同时修改多台设备。单设备工具和只读查询不需要确认。

```text
agent
  ├── 普通工具 → tools
  └── 场景工具 → approval
                   ├── 批准 → tools
                   └── 拒绝 → reject_tools
```

后续加入门锁、燃气和报警器等设备时，可以在 `src/agent/approval.py` 中扩展风险规则。

## 3. 中断数据

中断返回可序列化的确认请求：

```python
{
    "kind": "tool_approval",
    "question": "即将执行批量设备操作……是否继续？",
    "risk_level": "medium",
    "summary": "离家模式：关闭所有灯光、空调……",
    "tool_calls": [
        {
            "id": "scene-call-id",
            "name": "activate_scene",
            "args": {"scene_name": "离家模式"},
        }
    ],
}
```

## 4. 恢复协议

调用方必须保留原来的 `thread_id`，然后使用：

```python
graph.invoke(
    Command(resume={"approved": True}),
    original_config,
)
```

`approved` 只有严格为 `True` 时才视为批准，缺失、格式错误或其他值都按拒绝处理。

## 5. 拒绝处理

拒绝后不执行真实工具。图会为被拒绝的工具调用生成匹配 `tool_call_id` 的 `ToolMessage`，使消息协议保持完整，再由 Agent 生成面向用户的取消说明。

## 6. 验证范围

`tests/test_phase_six.py` 覆盖：

1. 场景操作在中断发生时尚未修改设备；
2. `approved=True` 后场景才执行；
3. `approved=False` 后设备保持不变；
4. 拒绝路径会生成完整 ToolMessage；
5. 普通单设备操作不会触发确认；
6. 关闭并重新构建图后，SQLite Checkpoint 仍能使用相同 `thread_id` 恢复执行。
