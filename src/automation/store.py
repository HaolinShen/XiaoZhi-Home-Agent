"""SQLite persistence for routines, runs and scheduled actions."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from .models import Routine, RoutineRun, ScheduledTask


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


class AutomationStore:
    def __init__(self, path: str = "data/automation.db"):
        self.path = path
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS routines (
                id TEXT PRIMARY KEY, home_id TEXT NOT NULL, user_id TEXT NOT NULL,
                payload TEXT NOT NULL, enabled INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS routine_runs (
                id TEXT PRIMARY KEY, routine_id TEXT NOT NULL, trigger_key TEXT NOT NULL,
                anchor_at TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL,
                UNIQUE(routine_id, trigger_key)
            );
            CREATE TABLE IF NOT EXISTS scheduled_tasks (
                id TEXT PRIMARY KEY, routine_id TEXT NOT NULL, home_id TEXT NOT NULL,
                user_id TEXT NOT NULL, action_id TEXT NOT NULL, due_at TEXT NOT NULL,
                payload TEXT NOT NULL, status TEXT NOT NULL, dedupe_key TEXT UNIQUE NOT NULL,
                attempts INTEGER NOT NULL, error TEXT, created_at TEXT NOT NULL,
                executed_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_tasks_due ON scheduled_tasks(status, due_at);
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def save_routine(self, routine: Routine) -> Routine:
        self.connection.execute(
            """INSERT INTO routines(id, home_id, user_id, payload, enabled)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET payload=excluded.payload, enabled=excluded.enabled""",
            (routine.id, routine.home_id, routine.user_id, routine.model_dump_json(), routine.enabled),
        )
        self.connection.commit()
        return routine

    def get_routine(self, routine_id: str) -> Routine | None:
        row = self.connection.execute("SELECT payload FROM routines WHERE id=?", (routine_id,)).fetchone()
        return Routine.model_validate_json(row["payload"]) if row else None

    def list_routines(self, home_id: str, *, enabled_only: bool = False) -> list[Routine]:
        query = "SELECT payload FROM routines WHERE home_id=?"
        params: list[object] = [home_id]
        if enabled_only:
            query += " AND enabled=1"
        rows = self.connection.execute(query, params).fetchall()
        return [Routine.model_validate_json(row["payload"]) for row in rows]

    def create_run(self, run: RoutineRun) -> bool:
        cursor = self.connection.execute(
            """INSERT OR IGNORE INTO routine_runs
               (id, routine_id, trigger_key, anchor_at, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (run.id, run.routine_id, run.trigger_key, _iso(run.anchor_at), run.status, _iso(run.created_at)),
        )
        self.connection.commit()
        return cursor.rowcount == 1

    def list_runs(self, routine_id: str) -> list[RoutineRun]:
        """Return every run of a routine, oldest anchor first."""
        rows = self.connection.execute(
            "SELECT * FROM routine_runs WHERE routine_id=? ORDER BY anchor_at", (routine_id,)
        ).fetchall()
        return [
            RoutineRun(
                id=row["id"],
                routine_id=row["routine_id"],
                trigger_key=row["trigger_key"],
                anchor_at=datetime.fromisoformat(row["anchor_at"]),
                status=row["status"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    def add_task(self, task: ScheduledTask) -> bool:
        cursor = self.connection.execute(
            """INSERT INTO scheduled_tasks
               (id, routine_id, home_id, user_id, action_id, due_at, payload,
                status, dedupe_key, attempts, error, created_at, executed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(dedupe_key) DO UPDATE SET
                 due_at=excluded.due_at,
                 payload=excluded.payload
               WHERE scheduled_tasks.status='pending'""",
            (task.id, task.routine_id, task.home_id, task.user_id, task.action_id,
             _iso(task.due_at), json.dumps(task.payload, ensure_ascii=False), task.status,
             task.dedupe_key, task.attempts, task.error, _iso(task.created_at),
             _iso(task.executed_at) if task.executed_at else None),
        )
        self.connection.commit()
        return cursor.rowcount == 1

    def due_tasks(self, now: datetime) -> list[ScheduledTask]:
        rows = self.connection.execute(
            "SELECT * FROM scheduled_tasks WHERE status='pending' AND due_at<=? ORDER BY due_at, created_at",
            (_iso(now),),
        ).fetchall()
        return [self._task(row) for row in rows]

    def _task(self, row: sqlite3.Row) -> ScheduledTask:
        return ScheduledTask(
            id=row["id"], routine_id=row["routine_id"], home_id=row["home_id"],
            user_id=row["user_id"], action_id=row["action_id"],
            due_at=datetime.fromisoformat(row["due_at"]), payload=json.loads(row["payload"]),
            status=row["status"], dedupe_key=row["dedupe_key"], attempts=row["attempts"],
            error=row["error"], created_at=datetime.fromisoformat(row["created_at"]),
            executed_at=datetime.fromisoformat(row["executed_at"]) if row["executed_at"] else None,
        )

    def update_task(self, task: ScheduledTask) -> None:
        self.connection.execute(
            """UPDATE scheduled_tasks SET status=?, attempts=?, error=?, executed_at=? WHERE id=?""",
            (task.status, task.attempts, task.error, _iso(task.executed_at) if task.executed_at else None, task.id),
        )
        self.connection.commit()

    def cancel_routine(self, routine_id: str, home_id: str, user_id: str) -> int:
        cursor = self.connection.execute(
            """UPDATE scheduled_tasks SET status='cancelled'
               WHERE routine_id=? AND home_id=? AND user_id=? AND status='pending'""",
            (routine_id, home_id, user_id),
        )
        self.connection.commit()
        return cursor.rowcount

    def list_tasks(self, routine_id: str | None = None) -> list[ScheduledTask]:
        if routine_id:
            rows = self.connection.execute(
                "SELECT * FROM scheduled_tasks WHERE routine_id=? ORDER BY due_at", (routine_id,)
            ).fetchall()
        else:
            rows = self.connection.execute("SELECT * FROM scheduled_tasks ORDER BY due_at").fetchall()
        return [self._task(row) for row in rows]
