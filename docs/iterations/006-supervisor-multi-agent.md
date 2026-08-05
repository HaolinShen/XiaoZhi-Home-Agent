# 阶段十：Supervisor 多智能体协作

## 目标

将单个全工具 Agent 拆成职责互斥的专用 Agent，由阶段八结构化路由充当 Supervisor 完成一次有界委派。

## 角色

- Device Agent：设备查询和单设备控制，只绑定设备工具。
- Scene Agent：预定义场景查询和启用，只绑定场景工具。
- Memory Agent：长期记忆、候选和版本管理，只绑定记忆工具。
- Chat Agent：普通对话，不绑定任何工具。

阶段六 Human-in-the-loop 和阶段七 Verifier 继续充当安全执行层，并没有额外创建一个只会输出文字的 Safety Agent。

## 流程

```text
sync_context → Supervisor(task_router)
                    ↓ delegated_agent
               专用 Agent → 工具/审批循环 → supervisor_finalize → END
```

状态保存 `delegated_agent`、`handoff_count` 和 `collaboration_status`。`MULTI_AGENT_MAX_HANDOFFS` 默认是 2，用于限制未来的跨 Agent 转交扩展。

## 配置

```text
MULTI_AGENT_ENABLED=true
MULTI_AGENT_MAX_HANDOFFS=2
```
