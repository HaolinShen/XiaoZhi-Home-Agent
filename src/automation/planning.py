"""Schemas and deterministic validation for LLM-generated automation plans."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from ..agent.planning import PLANNING_TOOL_NAMES, expected_state_for_step
from ..devices.base import DeviceRegistry

# 自动化允许的工具名 = 规划控制工具 + 闹钟（P0：从 PLANNING_TOOL_NAMES 派生，
# 以前这里手抄一份 control_xxx 清单，是第 11 处需要同步的副本）。
AutomationToolName = Literal[tuple((*PLANNING_TOOL_NAMES, "set_alarm"))]  # type: ignore[valid-type]


# Defaults are planning guidance, not hidden execution rules. The Automation
# Agent can choose different offsets when the user gives an explicit value.
DEFAULT_LEAD_MINUTES = {
    "control_water_heater": 30,
    "control_ac": 20,
    "control_kettle": 10,
    "control_curtain": 2,
    "control_light": 0,
    "set_alarm": 0,
}


AUTOMATION_ACTION_ALIASES = {
    "turn_on": "on",
    "turn_off": "off",
}


class ScheduledActionInput(BaseModel):
    offset_minutes: int = Field(
        ge=-1440,
        le=1440,
        description="相对目标时间的分钟偏移；提前执行用负数，例如提前30分钟为-30",
    )
    tool_name: AutomationToolName  # type: ignore[valid-type]  # 动态 Literal，运行期由 pydantic 求值
    arguments: dict[str, Any] = Field(default_factory=dict)
    description: str = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def parse_json_action(cls, value):
        return _decode_json(value, "action 必须是 JSON 对象")

    @field_validator("arguments", mode="before")
    @classmethod
    def parse_json_arguments(cls, value):
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError("arguments 必须是 JSON 对象") from exc
        return value

    @model_validator(mode="after")
    def reject_scheduled_unlock(self):
        if self.tool_name == "control_lock" and self.arguments.get("action") == "unlock":
            raise ValueError("定时自动化禁止解锁门锁")
        return self


class ScheduledRoutineInput(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    anchor_at_iso: str = Field(
        description="带日期和时区的 ISO 8601 目标时间，例如 2026-08-15T17:00:00+08:00"
    )
    actions: list[ScheduledActionInput] = Field(min_length=1, max_length=8)

    @field_validator("actions", mode="before")
    @classmethod
    def parse_json_actions(cls, value):
        return _parse_actions(value)


class VehicleRoutineInput(BaseModel):
    vehicle_id: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=80)
    actions: list[ScheduledActionInput] = Field(min_length=1, max_length=8)

    @field_validator("actions", mode="before")
    @classmethod
    def parse_json_actions(cls, value):
        return _parse_actions(value)


def _parse_actions(value):
    value = _decode_json(value, "actions 必须是 JSON 数组")
    if isinstance(value, dict):
        value = [value]
    return value


def _decode_json(value, error_message: str):
    # Some models stringify a JSON array twice. Decode at most twice so a
    # malformed payload fails deterministically instead of looping forever.
    for _ in range(2):
        if not isinstance(value, str):
            break
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(error_message) from exc
    return value


def normalize_and_validate_automation_actions(
    actions: list[ScheduledActionInput], registry: DeviceRegistry
) -> list[ScheduledActionInput]:
    """Canonicalize model-generated arguments and reject invalid actions.

    General device tools deliberately use Chinese ``device_name`` values and
    short actions such as ``on``/``off``. Some models still emit registry IDs
    and Home Assistant-style ``turn_on``/``turn_off`` inside nested automation
    actions, so normalize those narrow aliases before persisting the routine.
    """
    normalized_actions: list[ScheduledActionInput] = []
    for index, action in enumerate(actions, start=1):
        if action.tool_name == "set_alarm":
            if not action.arguments.get("speaker_name"):
                raise ValueError(f"步骤 {index} 缺少 speaker_name")
            normalized_actions.append(action)
            continue
        if action.tool_name not in PLANNING_TOOL_NAMES:
            raise ValueError(f"步骤 {index} 使用了不允许的工具 {action.tool_name}")

        arguments = dict(action.arguments)
        raw_action = arguments.get("action")
        if raw_action is not None:
            arguments["action"] = AUTOMATION_ACTION_ALIASES.get(raw_action, raw_action)
        device_id = arguments.pop("device_id", None)
        if not arguments.get("device_name") and device_id:
            device = registry.get(str(device_id))
            if device is None:
                raise ValueError(f"步骤 {index} 无法执行: device not found or ambiguous")
            arguments["device_name"] = device.name

        normalized_action = action.model_copy(update={"arguments": arguments})
        step = {
            "step_id": index,
            "description": normalized_action.description,
            "tool_name": normalized_action.tool_name,
            "arguments": normalized_action.arguments,
        }
        _, _, error = expected_state_for_step(step, registry)
        if error:
            raise ValueError(f"步骤 {index} 无法执行: {error}")
        normalized_actions.append(normalized_action)
    return normalized_actions
