"""Human-in-the-loop approval helpers for risky smart-home actions."""

from __future__ import annotations

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


def build_approval_request(tool_calls: list[dict[str, Any]]) -> ApprovalRequest | None:
    """Return an approval request for batch scene actions or lock unlocks.

    两类操作共用一个 interrupt 通道，但文案和风险等级分开：
      · activate_scene → medium，场景名 + 官方描述
      · control_lock(unlock) → high，设备名
    """
    risky_calls = [
        call for call in tool_calls
        if call.get("name") == "activate_scene" or _is_unlock_call(call)
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
        else:
            device_name = str(call.get("args", {}).get("device_name", "门锁"))
            descriptions.append(f"解锁{device_name}（对外敏感动作）")
            risk_levels.append("high")

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
