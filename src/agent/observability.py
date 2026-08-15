"""Public helpers for custom graph progress events.

这些事件是"图在运行时对外说的话"。调用方用
``graph.stream(..., stream_mode="custom")`` 接收；用 ``graph.invoke()`` 时
LangGraph 给一个空写入器，事件被静默丢弃 —— 这就是 CLI 曾经看不到
Planner / Executor / Verifier 分工的原因。

事件只描述可验证的任务进度（第几步、调了哪个工具、期望值与实测值是否一致），
不包含模型的隐藏推理文本。
"""

from langgraph.config import get_stream_writer

# 规划链路的事件：默认就该显示给用户，因为它们正是"规划与执行分开"的证据。
PLANNING_EVENTS = (
    "planning_selected",
    "plan_generated",
    "plan_decision",
    "step_started",
    "step_executed",
    "step_verified",
    "step_retry",
    "replan_requested",
    "planning_finished",
)

# 路由 / 记忆 / 上下文类事件：属于诊断信息，默认折叠，避免淹没规划过程。
TRACE_EVENTS = (
    "context_synced",
    "memory_reasoned",
    "supervisor_routing",
    "parallel_query_completed",
    "knowledge_rag_completed",
    "agent_completed",
    "supervisor_finalized",
)


def emit_progress(event: str, **payload) -> None:
    get_stream_writer()({"event": event, **payload})
