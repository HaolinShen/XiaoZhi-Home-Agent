"""Deterministic routine execution using the existing trusted tools."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ..agent.planning import expected_state_for_step, verify_step
from ..devices.base import DeviceRegistry
from .speaker import SpeakerBackend


class RoutineExecutor:
    def __init__(self, registry: DeviceRegistry, speaker: SpeakerBackend):
        self.registry = registry
        self.speaker = speaker
        # Import only after the tools package has finished initializing. Automation
        # tools themselves import this runtime, so a module-level import cycles.
        from ..tools import get_all_tools
        self.tools = {tool.name: tool for tool in get_all_tools()}

    def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        tool_name = payload["tool_name"]
        arguments = dict(payload.get("arguments", {}))

        if tool_name == "set_alarm":
            alarm_at = datetime.fromisoformat(arguments["alarm_at"])
            alarm_id = self.speaker.set_alarm(
                arguments.get("speaker_name", "卧室音响"),
                alarm_at,
                arguments.get("label", "起床"),
                payload.get("routine_id"),
            )
            return {
                "success": True,
                "tool_name": tool_name,
                "tool_result": f"闹钟已设置，alarm_id={alarm_id}",
                "verification": {"success": True, "alarm_id": alarm_id},
            }

        step = {
            "tool_name": tool_name,
            "arguments": arguments,
            "step_id": payload.get("step_id", 1),
            "description": payload.get("description", "自动化例程动作"),
        }
        device_id, expected, preparation_error = expected_state_for_step(step, self.registry)
        tool = self.tools.get(tool_name)
        if tool is None:
            tool_result = f"❌ 未注册工具 {tool_name}"
        elif preparation_error:
            tool_result = f"❌ {preparation_error}"
        else:
            try:
                tool_result = str(tool.invoke(arguments))
            except Exception as exc:
                tool_result = f"❌ 工具执行异常: {exc}"

        verification = verify_step(
            self.registry, device_id, expected, tool_result, preparation_error
        )
        return {
            "success": verification.success,
            "tool_name": tool_name,
            "tool_result": tool_result,
            "device_id": device_id,
            "verification": verification.model_dump(),
        }
