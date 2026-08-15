"""Human-in-the-loop approval helpers for risky smart-home actions."""

from __future__ import annotations

import json
from typing import Any, Literal, TypedDict

from langchain_core.messages import ToolMessage

from ..tools.scenes import SCENE_META


class ApprovalRequest(TypedDict):
    """Serializable payload exposed by LangGraph ``interrupt``."""

    kind: Literal["tool_approval"]
    question: str
    risk_level: Literal["medium", "high"]
    summary: str
    tool_calls: list[dict[str, Any]]


def _is_unlock_call(call: dict[str, Any]) -> bool:
    """Return True when the model proposes unlocking a door lock.

    解锁是"对外"的敏感动作（屋里有没有人取决于它），所以和批量场景一样
    需要人工确认后才能执行。上锁（lock）是收拢安全边界，不需要审批。
    """
    return (
        call.get("name") == "control_lock"
        and str(call.get("args", {}).get("action", "")) == "unlock"
    )


def _is_automation_call(call: dict[str, Any]) -> bool:
    return call.get("name") in {
        "create_scheduled_routine",
        "create_vehicle_arrival_routine",
        "schedule_wake_routine",
        "enable_vehicle_arrival_routine",
    }


def _automation_actions(value: Any) -> list[dict[str, Any]]:
    """Normalize model-produced action payloads before tool validation runs."""
    for _ in range(2):
        if not isinstance(value, str):
            break
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        if isinstance(item, str):
            try:
                item = json.loads(item)
            except json.JSONDecodeError:
                continue
        if isinstance(item, dict):
            result.append(item)
        elif hasattr(item, "model_dump"):
            result.append(item.model_dump())
    return result


def build_approval_request(tool_calls: list[dict[str, Any]]) -> ApprovalRequest | None:
    """Return an approval request for batch scene actions or lock unlocks.

    两类操作共用一个 interrupt 通道，但文案和风险等级分开：
      · activate_scene → medium，场景名 + 官方描述
      · control_lock(unlock) → high，设备名
    """
    risky_calls = [
        call for call in tool_calls
        if call.get("name") == "activate_scene" or _is_unlock_call(call) or _is_automation_call(call)
    ]
    if not risky_calls:
        return None

    descriptions = []
    risk_levels: list[str] = []
    for call in risky_calls:
        if call.get("name") == "activate_scene":
            scene_name = str(call.get("args", {}).get("scene_name", "未知场景"))
            meta = SCENE_META.get(scene_name, {})
            description = meta.get("description", "将同时修改多台家居设备")
            descriptions.append(f"{scene_name}：{description}")
            risk_levels.append("medium")
        elif _is_unlock_call(call):
            device_name = str(call.get("args", {}).get("device_name", "门锁"))
            descriptions.append(f"解锁{device_name}（对外敏感动作）")
            risk_levels.append("high")
        else:
            if call.get("name") in {"schedule_wake_routine", "create_scheduled_routine"}:
                args = call.get("args", {})
                target = args.get("wake_at_iso") or args.get("anchor_at_iso") or "未指定时间"
                name = args.get("name", "起床自动化")
                actions = _automation_actions(args.get("actions", []))
                action_text = "；".join(
                    f"{item.get('offset_minutes', 0):+}分钟 {item.get('description') or item.get('tool_name', '')}"
                    for item in actions
                )
                suffix = f"：{action_text}" if action_text else ""
                descriptions.append(f"创建定时自动化「{name}」（{target}）{suffix}")
            else:
                args = call.get("args", {})
                vehicle_id = args.get("vehicle_id", "未指定车辆")
                actions = _automation_actions(args.get("actions", []))
                action_text = "；".join(
                    f"{item.get('offset_minutes', 0):+}分钟 {item.get('description') or item.get('tool_name', '')}"
                    for item in actions
                )
                suffix = f"：{action_text}" if action_text else ""
                descriptions.append(f"启用车辆 {vehicle_id} 的回家自动化{suffix}")
            risk_levels.append("medium")

    summary = "；".join(descriptions)
    risk_level: Literal["medium", "high"] = "high" if "high" in risk_levels else "medium"
    return {
        "kind": "tool_approval",
        "question": f"即将执行需要确认的操作：{summary}。是否继续？",
        "risk_level": risk_level,
        "summary": summary,
        "tool_calls": [
            {
                "id": call.get("id"),
                "name": call.get("name"),
                "args": call.get("args", {}),
            }
            for call in tool_calls
        ],
    }


def approval_is_granted(value: Any) -> bool:
    """Normalize CLI/API resume payloads to a strict approval decision.
    将 CLI/API 恢复有效负载规范化，使其符合严格的审批决策。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        return value.get("approved") is True
    if isinstance(value, str):
        return value.strip().lower() in {"y", "yes", "true", "确认", "同意", "继续", "执行", "确定", "好"}
    return False


def rejection_tool_messages(tool_calls: list[dict[str, Any]]) -> list[ToolMessage]:
    """Close every proposed tool call without executing it after rejection."""
    return [
        ToolMessage(
            content="用户未批准该操作，工具没有执行，任何设备状态都未改变。",
            tool_call_id=str(call.get("id", "unknown-tool-call")),
            name=call.get("name"),
            status="error",
        )
        for call in tool_calls
    ]
