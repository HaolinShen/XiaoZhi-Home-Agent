"""Public helpers for custom graph progress events.

这些事件是"图在运行时对外说的话"。调用方用
``graph.stream(..., stream_mode="custom")`` 接收；用 ``graph.invoke()`` 时
LangGraph 给一个空写入器，事件被静默丢弃 —— 这就是 CLI 曾经看不到
Planner / Executor / Verifier 分工的原因。

P2 改造: `emit_progress` 现在是双写 ——
  1. stream writer（保持 CLI 的实时渲染不变）；
  2. loguru 结构化日志（channel=graph_progress，DEBUG 级），
     让 invoke / 后台自动化执行器路径也有完整的事件痕迹可查，
     不再依赖"边跑边看"的 stream 模式。

事件只描述可验证的任务进度（第几步、调了哪个工具、期望值与实测值是否一致），
不包含模型的隐藏推理文本。
"""

import json

from langgraph.config import get_stream_writer
from loguru import logger

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
    """发一个进度事件：stream 模式走 custom 通道，其余路径写结构化日志。

    invoke 模式（及后台执行器）下 LangGraph 给空写入器，事件会被静默丢弃；
    直接在图上下文之外调用时 `get_stream_writer()` 还会抛 RuntimeError ——
    所以流式通道必须容错，日志侧才是所有路径都能依赖的痕迹。
    payload 里可能有嵌套 dict，序列化用 default=str 兜底，度量日志绝不该把业务打断。
    """
    try:
        writer = get_stream_writer()
    except Exception:
        writer = None
    if writer is not None:
        writer({"event": event, **payload})
    logger.bind(channel="graph_progress").debug(
        "{event} | {payload}",
        event=event,
        payload=json.dumps(payload, ensure_ascii=False, default=str),
    )
