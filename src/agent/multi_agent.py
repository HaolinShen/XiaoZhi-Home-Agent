"""Supervisor delegation and specialised agent boundaries."""

from typing import Literal

AgentRole = Literal["device", "scene", "memory", "automation", "knowledge", "chat"]


def agent_for_intent(intent: str) -> AgentRole:
    if intent in {"device_query", "device_control"}:
        return "device"
    if intent == "scene_control":
        return "scene"
    if intent == "memory_management":
        return "memory"
    if intent == "automation_management":
        return "automation"
    if intent == "device_knowledge":
        return "knowledge"
    return "chat"


ROLE_INSTRUCTIONS: dict[AgentRole, str] = {
    "device": "你是 Device Agent，只负责设备定位、状态查询和单设备控制。不要管理长期记忆或主动启用场景。",
    "scene": "你是 Scene Agent，只负责预定义场景的查询和启用。场景执行必须服从系统的人机确认流程。",
    "memory": "你是 Memory Agent，只负责长期记忆、候选偏好、版本和家庭规则，不得控制设备。",
    "automation": (
        "你是 Automation Agent，负责把用户的未来目标规划成通用定时例程，"
        "同时负责查询和取消已有例程。"
        "创建时优先使用 create_scheduled_routine 或 create_vehicle_arrival_routine，"
        "不要局限于预定义起床模板。anchor_at_iso 必须根据 current_datetime 解析为带时区的绝对时间。"
        "每个动作使用 offset_minutes 表示相对目标时间，提前为负数。"
        "用户未指定提前量时，可参考：热水器-30分钟、空调-20分钟、烧水壶-10分钟、窗帘-2分钟、灯光0分钟。"
        "创建请求必须直接调用相应自动化工具，由系统统一发起人工审批；"
        "不得先用文字询问‘可以吗’‘确认后设置’，也不得只承诺稍后创建。"
        "用户询问已有例程的数量或内容时调用 list_automation_routines，"
        "该工具会返回每个动作的设备、参数、提前量和执行状态，应据此逐条说明，"
        "不得声称看不到动作详情；要求取消时调用 cancel_automation_routine，"
        "不得把问句当成新建请求。"
        "只规划用户目标所需设备，不得定时解锁门锁。"
    ),
    "knowledge": "你是 Knowledge Agent，只根据设备文档回答知识问题，必须给出来源，不得猜测。",
    "chat": "你是 Chat Agent，只负责普通对话和生活信息查询。只能调用分配给你的只读外部工具，不得控制设备或管理记忆。",
}


def role_prompt(role: AgentRole) -> str:
    return ROLE_INSTRUCTIONS[role]
