"""
Agent 状态定义
==============
LangGraph 的状态管理核心。定义 Agent 在图流转过程中携带的数据结构。

State 设计原则:
  - messages: 对话历史（核心字段），使用 add_messages 合并器
  - 可以添加额外字段（如 device_context, user_profile 等）用于增强 Agent 能力
"""

from typing import Annotated, Any, Literal
from typing_extensions import NotRequired, TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    Agent 工作流的状态对象。

    fields:
      messages: 对话消息列表。使用 add_messages 作为合并器，
                新消息会自动追加而非覆盖旧消息。

    扩展示例（后续版本可添加）:
      device_context: dict  # 当前关注的设备上下文
      user_profile: dict    # 用户偏好（如"喜欢暖光"）
      conversation_id: str  # 对话 ID（用于多用户场景）
    """
    messages: Annotated[list, add_messages]
    active_room_id: NotRequired[str | None]
    active_device_id: NotRequired[str | None]
    request_home_id: NotRequired[str]
    request_user_id: NotRequired[str]
    request_client_id: NotRequired[str]
    request_session_id: NotRequired[str]
    request_room_id: NotRequired[str | None]
    request_device_id: NotRequired[str | None]
    request_is_admin: NotRequired[bool]
    intent: NotRequired[str]
    intent_confidence: NotRequired[float]
    intent_reason: NotRequired[str]
    intent_route: NotRequired[Literal["react", "planner", "clarification", "parallel_query"]]
    parallel_query_results: NotRequired[list[dict[str, Any]]]
    conversation_summary: NotRequired[str]
    memory_context: NotRequired[str]
    context_message_count: NotRequired[int]
    context_token_estimate: NotRequired[int]
    approval_request: NotRequired[dict[str, Any] | None]
    approval_decision: NotRequired[Literal["approved", "rejected"] | None]
    planning_active: NotRequired[bool]
    planning_goal: NotRequired[str]
    plan: NotRequired[dict[str, Any] | None]
    plan_revision: NotRequired[int]
    current_step_index: NotRequired[int]
    step_retry_count: NotRequired[int]
    replan_count: NotRequired[int]
    planning_status: NotRequired[
        Literal["planning", "awaiting_approval", "executing", "completed", "failed", "cancelled"]
    ]
    last_execution: NotRequired[dict[str, Any] | None]
    last_verification: NotRequired[dict[str, Any] | None]
    planning_failure_feedback: NotRequired[str]
    planning_results: NotRequired[list[dict[str, Any]]]
