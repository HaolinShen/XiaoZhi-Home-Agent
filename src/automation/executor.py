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
        # P1: 执行器构造自己的工具集（闭包注入 registry），并且**显式关闭
        # 偏好观察**——机器触发的动作不计入"重复手动操作"，否则会凭空造出
        # 用户从未设过的偏好。这也是对旧 bug（无身份调用下 record_preference
        # 抛 KeyError 把整个定时动作判成失败）的根因修复：现在"无身份"是
        # 构造期的显式选择，而不是调用期靠逐键检查去猜。
        from ..tools import build_device_tools
        self.tools = {
            tool.name: tool
            for tool in build_device_tools(registry, enable_preference_tracking=False)
        }

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
