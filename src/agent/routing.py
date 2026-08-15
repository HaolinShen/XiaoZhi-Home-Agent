"""Structured intent routing with a deterministic fallback.

The router is deliberately small: it selects a business path, while the
specialised Agent/Planner nodes still decide the concrete tool calls.
"""

import re
from typing import Literal
from pydantic import BaseModel, Field


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
    memory_words = ("记住", "忘记", "删除记忆", "偏好", "喜欢", "家庭规则", "记忆")
    automation_words = (
        "定时", "闹钟", "车辆回家", "汽车回家", "到家前", "回家前",
        "取消例程", "自动化", "提前准备", "提前打开",
    )
    knowledge_words = ("故障", "错误代码", "说明书", "怎么清洗", "怎么维护", "支持什么", "是什么意思")
    scene_words = ("场景", "模式", "睡眠", "离家", "回家", "观影", "起床")
    query_words = ("查询", "状态", "温度", "开着吗", "在线", "有哪些设备")
    control_words = ("打开", "关闭", "开启", "关掉", "调到", "设置", "调高", "调低")
    has_future_time = bool(re.search(
        r"(?:今天|明天|后天|周[一二三四五六日天]|星期[一二三四五六日天]).{0,10}"
        r"(?:上午|下午|晚上|早上|凌晨|\d{1,2}\s*[点时])",
        value,
    ))
    if has_future_time or any(word in value for word in automation_words):
        return IntentResult(intent="automation_management", confidence=0.9, reason="包含定时或事件自动化词")
    if any(word in value for word in memory_words):
        return IntentResult(intent="memory_management", confidence=0.92, reason="包含记忆或偏好操作词")
    if any(word in value for word in knowledge_words):
        return IntentResult(intent="device_knowledge", confidence=0.9, reason="包含设备知识或故障咨询词")
    if any(word in value for word in scene_words):
        return IntentResult(intent="scene_control", confidence=0.88, reason="包含场景或模式词")
    if any(word in value for word in query_words):
        return IntentResult(intent="device_query", confidence=0.86, reason="包含设备查询词")
    if any(word in value for word in control_words):
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
