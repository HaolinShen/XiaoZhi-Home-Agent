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


def build_approval_request(tool_calls: list[dict[str, Any]]) -> ApprovalRequest | None:
    """Return an approval request when a model proposes a batch scene action."""
    risky_calls = [call for call in tool_calls if call.get("name") == "activate_scene"]
    if not risky_calls:
        return None

    descriptions = []
    for call in risky_calls:
        scene_name = str(call.get("args", {}).get("scene_name", "未知场景"))
        meta = SCENE_META.get(scene_name, {})
        description = meta.get("description", "将同时修改多台家居设备")
        descriptions.append(f"{scene_name}：{description}")

    summary = "；".join(descriptions)
    return {
        "kind": "tool_approval",
        "question": f"即将执行批量设备操作：{summary}。是否继续？",
        "risk_level": "medium",
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
    """Normalize CLI/API resume payloads to a strict approval decision."""
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        return value.get("approved") is True
    if isinstance(value, str):
        return value.strip().lower() in {"y", "yes", "true", "确认", "同意", "继续"}
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
