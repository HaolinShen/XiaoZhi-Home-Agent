"""Time-aware scheduler with a synchronous testable tick and optional worker."""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from typing import Callable

from .executor import RoutineExecutor
from .models import Routine, RoutineRun, ScheduledTask
from .store import AutomationStore


class RoutineScheduler:
    def __init__(
        self,
        store: AutomationStore,
        executor: RoutineExecutor,
        *,
        max_attempts: int = 2,
        poll_seconds: float = 1.0,
        event_sink: Callable[[dict], None] | None = None,
    ):
        self.store = store
        self.executor = executor
        self.max_attempts = max_attempts
        self.poll_seconds = poll_seconds
        self.event_sink = event_sink or (lambda event: None)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def schedule(
        self,
        routine: Routine,
        *,
        anchor_at: datetime,
        trigger_key: str,
        now: datetime | None = None,
    ) -> list[ScheduledTask]:
        now = now or datetime.now(timezone.utc)
        run = RoutineRun(routine_id=routine.id, trigger_key=trigger_key, anchor_at=anchor_at)
        self.store.create_run(run)
        scheduled: list[ScheduledTask] = []
        for index, action in enumerate(routine.actions, start=1):
            if not action.enabled:
                continue
            arguments = dict(action.arguments)
            due_at = anchor_at + timedelta(minutes=action.offset_minutes)
            if action.tool_name == "set_alarm":
                # The alarm must be armed when the routine is created, not at wake time.
                due_at = now
                arguments["alarm_at"] = anchor_at.isoformat()
            task = ScheduledTask(
                routine_id=routine.id,
                home_id=routine.home_id,
                user_id=routine.user_id,
                action_id=action.id,
                due_at=due_at,
                dedupe_key=f"{routine.id}:{trigger_key}:{action.id}",
                payload={
                    "routine_id": routine.id,
                    "step_id": index,
                    "description": action.description,
                    "tool_name": action.tool_name,
                    "arguments": arguments,
                },
            )
            self.store.add_task(task)
            scheduled.append(task)
        self.event_sink({
            "event": "routine_scheduled",
            "routine_id": routine.id,
            "trigger_key": trigger_key,
            "anchor_at": anchor_at.isoformat(),
            "task_count": len(scheduled),
        })
        return scheduled

    def tick(self, now: datetime | None = None) -> list[dict]:
        now = now or datetime.now(timezone.utc)
        results: list[dict] = []
        for task in self.store.due_tasks(now):
            task.status = "running"
            task.attempts += 1
            self.store.update_task(task)
            self.event_sink({
                "event": "routine_action_started",
                "routine_id": task.routine_id,
                "task_id": task.id,
                "tool_name": task.payload.get("tool_name"),
                "due_at": task.due_at.isoformat(),
            })
            result = self.executor.execute(task.payload)
            if result["success"]:
                task.status = "completed"
                task.executed_at = now
                task.error = None
            elif task.attempts < self.max_attempts:
                task.status = "pending"
                task.error = result["verification"].get("reason", result["tool_result"])
            else:
                task.status = "failed"
                task.executed_at = now
                task.error = result["verification"].get("reason", result["tool_result"])
            self.store.update_task(task)
            event = {
                "event": "routine_action_finished",
                "routine_id": task.routine_id,
                "task_id": task.id,
                "status": task.status,
                **result,
            }
            self.event_sink(event)
            results.append(event)
        return results

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="routine-scheduler", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.poll_seconds):
            self.tick()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=max(2.0, self.poll_seconds * 2))
