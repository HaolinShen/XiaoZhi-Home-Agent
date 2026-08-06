"""Supervisor delegation and specialised agent boundaries."""

from typing import Literal


AgentRole = Literal["device", "scene", "memory", "knowledge", "chat"]


def agent_for_intent(intent: str) -> AgentRole:
    if intent in {"device_query", "device_control"}:
        return "device"
    if intent == "scene_control":
        return "scene"
    if intent == "memory_management":
        return "memory"
    if intent == "device_knowledge":
        return "knowledge"
    return "chat"


ROLE_INSTRUCTIONS: dict[AgentRole, str] = {
    "device": "你是 Device Agent，只负责设备定位、状态查询和单设备控制。不要管理长期记忆或主动启用场景。",
    "scene": "你是 Scene Agent，只负责预定义场景的查询和启用。场景执行必须服从系统的人机确认流程。",
    "memory": "你是 Memory Agent，只负责长期记忆、候选偏好、版本和家庭规则，不得控制设备。",
    "knowledge": "你是 Knowledge Agent，只根据设备文档回答知识问题，必须给出来源，不得猜测。",
    "chat": "你是 Chat Agent，只负责普通对话和生活信息查询。只能调用分配给你的只读外部工具，不得控制设备或管理记忆。",
}


def role_prompt(role: AgentRole) -> str:
    return ROLE_INSTRUCTIONS[role]
