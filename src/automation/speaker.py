"""Speaker/alarm adapter boundary with a deterministic simulator."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from uuid import uuid4


class SpeakerBackend(ABC):
    @abstractmethod
    def set_alarm(
        self, speaker_name: str, alarm_at: datetime, label: str, routine_id: str | None = None
    ) -> str:
        ...

    @abstractmethod
    def cancel_alarm(self, alarm_id: str) -> bool:
        ...

    @abstractmethod
    def list_alarms(self) -> list[dict]:
        ...

    @abstractmethod
    def cancel_routine_alarms(self, routine_id: str) -> int:
        ...


class SimulatorSpeakerBackend(SpeakerBackend):
    def __init__(self):
        self._alarms: dict[str, dict] = {}

    def set_alarm(
        self, speaker_name: str, alarm_at: datetime, label: str, routine_id: str | None = None
    ) -> str:
        alarm_id = uuid4().hex
        self._alarms[alarm_id] = {
            "alarm_id": alarm_id,
            "speaker_name": speaker_name,
            "alarm_at": alarm_at,
            "label": label,
            "enabled": True,
            "routine_id": routine_id,
        }
        return alarm_id

    def cancel_alarm(self, alarm_id: str) -> bool:
        alarm = self._alarms.get(alarm_id)
        if alarm is None:
            return False
        alarm["enabled"] = False
        return True

    def list_alarms(self) -> list[dict]:
        return list(self._alarms.values())

    def cancel_routine_alarms(self, routine_id: str) -> int:
        count = 0
        for alarm in self._alarms.values():
            if alarm.get("routine_id") == routine_id and alarm["enabled"]:
                alarm["enabled"] = False
                count += 1
        return count
