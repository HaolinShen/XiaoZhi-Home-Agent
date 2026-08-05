"""Explicit, auditable reasoning over retrieved long-term memories."""

from typing import Any
from pydantic import BaseModel, Field


class MemoryDecision(BaseModel):
    applicable_memory_ids: list[str] = Field(default_factory=list)
    ignored_memory_ids: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    preferences: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    needs_clarification: bool = False
    reason: str = ""


def reason_about_memories(records: list[dict[str, Any]], request: str) -> MemoryDecision:
    """Apply deterministic precedence rules and return an inspectable decision."""
    explicit_override = any(marker in request for marker in ("这次", "临时", "不要按", "不用记住的偏好"))
    applicable, ignored, constraints, preferences = [], [], [], []
    seen_keys: dict[str, str] = {}
    conflicts: list[str] = []
    for record in records:
        memory_id = record.get("id", "")
        key = record.get("memory_key", "")
        value = record.get("memory_value", {})
        if key in seen_keys and seen_keys[key] != repr(value):
            conflicts.append(key)
        seen_keys[key] = repr(value)
        if explicit_override and record.get("memory_type") == "preference":
            ignored.append(memory_id)
            continue
        applicable.append(memory_id)
        text = f"{key}={value}"
        if record.get("memory_type") == "constraint":
            constraints.append(text)
        else:
            preferences.append(text)
    needs_clarification = bool(conflicts)
    return MemoryDecision(
        applicable_memory_ids=applicable,
        ignored_memory_ids=ignored,
        constraints=constraints,
        preferences=preferences,
        conflicts=conflicts,
        needs_clarification=needs_clarification,
        reason=("存在同键不同值的记忆，需要确认" if conflicts else "按当前指令和记忆作用域完成适用性判断"),
    )


def format_memory_decision(decision: MemoryDecision) -> str:
    return (
        f"applicable={decision.applicable_memory_ids}; "
        f"ignored={decision.ignored_memory_ids}; "
        f"constraints={decision.constraints}; preferences={decision.preferences}; "
        f"conflicts={decision.conflicts}; needs_clarification={decision.needs_clarification}"
    )
