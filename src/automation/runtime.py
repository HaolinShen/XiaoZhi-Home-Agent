"""Application-facing automation runtime."""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from ..devices.base import DeviceRegistry
from .executor import RoutineExecutor
from .planning import ScheduledActionInput, normalize_and_validate_automation_actions
from .routines import (
    build_arrival_routine,
    build_scheduled_routine,
    build_vehicle_routine,
    build_wake_routine,
)
from .scheduler import RoutineScheduler
from .speaker import SimulatorSpeakerBackend, SpeakerBackend
from .store import AutomationStore
from .vehicle import ArrivalOrchestrator


class AutomationRuntime:
    def __init__(
        self,
        registry: DeviceRegistry,
        *,
        db_path: str = "data/automation.db",
        timezone_name: str = "Asia/Shanghai",
        speaker: SpeakerBackend | None = None,
        event_sink=None,
    ):
        self.timezone = ZoneInfo(timezone_name)
        self.registry = registry
        self.store = AutomationStore(db_path)
        self.speaker = speaker or SimulatorSpeakerBackend()
        self.executor = RoutineExecutor(registry, self.speaker)
        self.scheduler = RoutineScheduler(self.store, self.executor, event_sink=event_sink)
        self.arrivals = ArrivalOrchestrator(self.store, self.scheduler)

    def schedule_wake(self, home_id: str, user_id: str, wake_at: datetime):
        if wake_at.tzinfo is None:
            wake_at = wake_at.replace(tzinfo=self.timezone)
        if wake_at <= datetime.now(UTC):
            raise ValueError("起床时间必须晚于当前时间")
        routine = self.store.save_routine(build_wake_routine(home_id, user_id))
        tasks = self.scheduler.schedule(
            routine,
            anchor_at=wake_at,
            trigger_key=f"wake:{wake_at.isoformat()}",
        )
        return routine, tasks

    def create_scheduled_routine(
        self,
        home_id: str,
        user_id: str,
        name: str,
        anchor_at: datetime,
        actions: list[ScheduledActionInput],
    ):
        if anchor_at.tzinfo is None:
            anchor_at = anchor_at.replace(tzinfo=self.timezone)
        if anchor_at <= datetime.now(UTC):
            raise ValueError("目标时间必须晚于当前时间")
        actions = normalize_and_validate_automation_actions(actions, self.registry)
        routine = self.store.save_routine(
            build_scheduled_routine(home_id, user_id, name, actions)
        )
        tasks = self.scheduler.schedule(
            routine,
            anchor_at=anchor_at,
            trigger_key=f"scheduled:{anchor_at.isoformat()}",
        )
        return routine, tasks

    def enable_vehicle_arrival(self, home_id: str, user_id: str, vehicle_id: str):
        routine = build_arrival_routine(home_id, user_id)
        routine.metadata["vehicle_id"] = vehicle_id
        return self.store.save_routine(routine)

    def create_vehicle_arrival_routine(
        self,
        home_id: str,
        user_id: str,
        vehicle_id: str,
        name: str,
        actions: list[ScheduledActionInput],
    ):
        actions = normalize_and_validate_automation_actions(actions, self.registry)
        return self.store.save_routine(
            build_vehicle_routine(home_id, user_id, vehicle_id, name, actions)
        )

    def cancel(self, routine_id: str, home_id: str, user_id: str) -> int:
        cancelled_tasks = self.store.cancel_routine(routine_id, home_id, user_id)
        self.speaker.cancel_routine_alarms(routine_id)
        return cancelled_tasks

    def close(self) -> None:
        self.scheduler.stop()
        self.store.close()
