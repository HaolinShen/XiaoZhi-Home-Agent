"""Structured intent routing with a deterministic fallback.

The router is deliberately small: it selects a business path, while the
specialised Agent/Planner nodes still decide the concrete tool calls.

P3 改造: 兜底分类器的关键词表已迁到 `heuristics.py`（ROUTING_* 常量），
本模块只保留分类流程本身。语义与迁移前逐字一致。
"""

from typing import Literal

from pydantic import BaseModel, Field

from .heuristics import (
    ROUTING_AUTOMATION_WORDS,
    ROUTING_CONTROL_WORDS,
    ROUTING_KNOWLEDGE_WORDS,
    ROUTING_MEMORY_WORDS,
    ROUTING_QUERY_WORDS,
    ROUTING_SCENE_WORDS,
    has_future_time,
)

Intent = Literal[
    "device_query",
    "device_control",
    "scene_control",
    "memory_management",
    "automation_management",
    "device_knowledge",
    "general_chat",
    "clarification",
]


class IntentResult(BaseModel):
    intent: Intent
    confidence: float = Field(ge=0, le=1)
    reason: str = ""


def intent_router_prompt(text: str) -> str:
    return f"""你是智能家居请求分类器。只输出结构化结果，不调用工具。
将用户请求分类为以下意图之一：
- device_query：查询设备状态、温度、开关或在线状态
- device_control：控制单个或多个设备，但不属于预定义场景
- scene_control：启用、取消或询问预定义场景模式
- memory_management：记住、修改、删除或查询用户偏好/家庭规则
- automation_management：创建、查看或取消任何未来时间/提前执行的自动化，
  包括固定时间回家准备、定时起床、闹钟和车辆回家联动
- device_knowledge：设备说明书、故障代码、维护方法或产品能力知识
- general_chat：闲聊、解释或一般问题
- clarification：信息不足，无法安全判断目标设备或动作
confidence 必须是 0 到 1 之间的小数；低于 0.6 时优先选择 clarification。

用户请求：{text}
"""


def classify_intent_fallback(text: str) -> IntentResult:
    value = text.strip().lower()
    if not value:
        return IntentResult(intent="clarification", confidence=0.1, reason="请求为空")
    if has_future_time(value) or any(word in value for word in ROUTING_AUTOMATION_WORDS):
        return IntentResult(intent="automation_management", confidence=0.9, reason="包含定时或事件自动化词")
    if any(word in value for word in ROUTING_MEMORY_WORDS):
        return IntentResult(intent="memory_management", confidence=0.92, reason="包含记忆或偏好操作词")
    if any(word in value for word in ROUTING_KNOWLEDGE_WORDS):
        return IntentResult(intent="device_knowledge", confidence=0.9, reason="包含设备知识或故障咨询词")
    if any(word in value for word in ROUTING_SCENE_WORDS):
        return IntentResult(intent="scene_control", confidence=0.88, reason="包含场景或模式词")
    if any(word in value for word in ROUTING_QUERY_WORDS):
        return IntentResult(intent="device_query", confidence=0.86, reason="包含设备查询词")
    if any(word in value for word in ROUTING_CONTROL_WORDS):
        return IntentResult(intent="device_control", confidence=0.82, reason="包含设备控制词")
    if len(value) < 2:
        return IntentResult(intent="clarification", confidence=0.2, reason="请求信息不足")
    return IntentResult(intent="general_chat", confidence=0.75, reason="未命中明确设备意图")


def classify_intent(llm, text: str) -> IntentResult:
    """Use structured LLM output when available; never let routing break the agent."""
    fallback = classify_intent_fallback(text)
    # Explicit future times and automation phrases are deterministic business
    # signals. Do not let a model over-weight words such as "回家" and route a
    # fixed-time routine into the predefined scene branch.
    if fallback.intent == "automation_management":
        return fallback
    try:
        structured = llm.with_structured_output(IntentResult)
        result = structured.invoke(intent_router_prompt(text))
        result = result if isinstance(result, IntentResult) else IntentResult.model_validate(result)
        return result
    except Exception:
        return fallback
