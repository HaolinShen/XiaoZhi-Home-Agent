"""SQLite repository for structured long-term memories."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime

from .models import (
    MemoryConflict, MemoryRecord, MemoryScope, MemoryType, MemoryWrite,
    PreferenceCandidate, utc_now,
)


class MemoryRepository:
    def __init__(self, db_path: str) -> None:
        directory = os.path.dirname(db_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self.connection = sqlite3.connect(db_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                home_id TEXT NOT NULL,
                user_id TEXT,
                room_id TEXT,
                device_id TEXT,
                scope TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                memory_key TEXT NOT NULL,
                memory_value TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 1.0,
                source TEXT,
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                expires_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_memories_home_scope
                ON memories(home_id, scope, status);
            CREATE INDEX IF NOT EXISTS idx_memories_user
                ON memories(home_id, user_id, status);
            CREATE UNIQUE INDEX IF NOT EXISTS uq_memories_business_key
                ON memories(
                    home_id, COALESCE(user_id, ''), COALESCE(room_id, ''),
                    COALESCE(device_id, ''), scope, memory_type, memory_key
                );
            CREATE TABLE IF NOT EXISTS preference_observations (
                home_id TEXT NOT NULL, user_id TEXT NOT NULL,
                room_id TEXT, device_id TEXT, memory_key TEXT NOT NULL,
                value_fingerprint TEXT NOT NULL, memory_value TEXT NOT NULL,
                observation_count INTEGER NOT NULL DEFAULT 1,
                first_observed_at TEXT NOT NULL, last_observed_at TEXT NOT NULL,
                PRIMARY KEY (home_id, user_id, memory_key, value_fingerprint)
            );
            CREATE TABLE IF NOT EXISTS preference_candidates (
                id TEXT PRIMARY KEY, home_id TEXT NOT NULL, user_id TEXT NOT NULL,
                room_id TEXT, device_id TEXT, memory_key TEXT NOT NULL,
                memory_value TEXT NOT NULL, observation_count INTEGER NOT NULL,
                confidence REAL NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                confirmed_memory_id TEXT
            );
            CREATE UNIQUE INDEX IF NOT EXISTS uq_pending_candidate
                ON preference_candidates(home_id, user_id, memory_key)
                WHERE status='pending';
            CREATE TABLE IF NOT EXISTS memory_conflicts (
                id TEXT PRIMARY KEY, memory_id TEXT NOT NULL, home_id TEXT NOT NULL,
                previous_value TEXT NOT NULL, incoming_value TEXT NOT NULL,
                resolved_value TEXT NOT NULL, resolution TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    def upsert(self, home_id: str, user_id: str | None, item: MemoryWrite) -> MemoryRecord:
        now = utc_now().isoformat()
        memory_id = str(uuid.uuid4())
        values = (
            memory_id, home_id, user_id, item.room_id, item.device_id,
            item.scope.value, item.memory_type.value, item.memory_key,
            json.dumps(item.memory_value, ensure_ascii=False), item.confidence,
            item.source, now, now,
            item.expires_at.isoformat() if item.expires_at else None,
        )
        self.connection.execute(
            """
            INSERT INTO memories (
                id, home_id, user_id, room_id, device_id, scope, memory_type,
                memory_key, memory_value, confidence, source, created_at,
                updated_at, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT DO UPDATE SET
                memory_value=excluded.memory_value,
                confidence=excluded.confidence,
                source=excluded.source,
                status='active',
                updated_at=excluded.updated_at,
                expires_at=excluded.expires_at
            """,
            values,
        )
        self.connection.commit()
        return self.get_by_key(
            home_id, user_id, item.room_id, item.device_id,
            item.scope, item.memory_type, item.memory_key,
        )

    def get_by_key(
        self, home_id: str, user_id: str | None, room_id: str | None,
        device_id: str | None, scope: MemoryScope, memory_type: MemoryType,
        memory_key: str,
    ) -> MemoryRecord:
        row = self.connection.execute(
            """
            SELECT * FROM memories WHERE home_id=? AND user_id IS ?
              AND room_id IS ? AND device_id IS ? AND scope=?
              AND memory_type=? AND memory_key=?
            """,
            (home_id, user_id, room_id, device_id, scope.value, memory_type.value, memory_key),
        ).fetchone()
        if row is None:
            raise KeyError(memory_key)
        return self._to_record(row)

    def find_by_key(
        self, home_id: str, user_id: str | None, room_id: str | None,
        device_id: str | None, scope: MemoryScope, memory_type: MemoryType,
        memory_key: str,
    ) -> MemoryRecord | None:
        try:
            record = self.get_by_key(
                home_id, user_id, room_id, device_id, scope, memory_type, memory_key
            )
        except KeyError:
            return None
        return record if record.status == "active" else None

    def observe_preference(
        self, home_id: str, user_id: str, memory_key: str,
        memory_value: dict, room_id: str | None, device_id: str | None,
    ) -> int:
        now = utc_now().isoformat()
        value_json = json.dumps(memory_value, ensure_ascii=False, sort_keys=True)
        self.connection.execute(
            """INSERT INTO preference_observations (
                   home_id,user_id,room_id,device_id,memory_key,value_fingerprint,
                   memory_value,observation_count,first_observed_at,last_observed_at
               ) VALUES (?,?,?,?,?,?,?,1,?,?)
               ON CONFLICT(home_id,user_id,memory_key,value_fingerprint) DO UPDATE SET
                   observation_count=observation_count+1,
                   room_id=excluded.room_id, device_id=excluded.device_id,
                   last_observed_at=excluded.last_observed_at""",
            (home_id, user_id, room_id, device_id, memory_key, value_json,
             value_json, now, now),
        )
        self.connection.commit()
        row = self.connection.execute(
            """SELECT observation_count FROM preference_observations
               WHERE home_id=? AND user_id=? AND memory_key=? AND value_fingerprint=?""",
            (home_id, user_id, memory_key, value_json),
        ).fetchone()
        return int(row[0])

    def upsert_candidate(
        self, home_id: str, user_id: str, memory_key: str,
        memory_value: dict, observation_count: int, confidence: float,
        room_id: str | None, device_id: str | None,
    ) -> PreferenceCandidate:
        now = utc_now().isoformat()
        existing = self.connection.execute(
            """SELECT id FROM preference_candidates WHERE home_id=? AND user_id=?
               AND memory_key=? AND status='pending'""",
            (home_id, user_id, memory_key),
        ).fetchone()
        candidate_id = existing[0] if existing else str(uuid.uuid4())
        if existing:
            self.connection.execute(
                """UPDATE preference_candidates SET memory_value=?, observation_count=?,
                   confidence=?, room_id=?, device_id=?, updated_at=? WHERE id=?""",
                (json.dumps(memory_value, ensure_ascii=False), observation_count,
                 confidence, room_id, device_id, now, candidate_id),
            )
        else:
            self.connection.execute(
                """INSERT INTO preference_candidates (
                   id,home_id,user_id,room_id,device_id,memory_key,memory_value,
                   observation_count,confidence,status,created_at,updated_at,confirmed_memory_id
                   ) VALUES (?,?,?,?,?,?,?,?,?,'pending',?,?,NULL)""",
                (candidate_id, home_id, user_id, room_id, device_id, memory_key,
                 json.dumps(memory_value, ensure_ascii=False), observation_count,
                 confidence, now, now),
            )
        self.connection.commit()
        return self.get_candidate(candidate_id, home_id)  # type: ignore[return-value]

    def get_candidate(self, candidate_id: str, home_id: str) -> PreferenceCandidate | None:
        row = self.connection.execute(
            "SELECT * FROM preference_candidates WHERE id=? AND home_id=?",
            (candidate_id, home_id),
        ).fetchone()
        return self._to_candidate(row) if row else None

    def list_candidates(self, home_id: str, user_id: str) -> list[PreferenceCandidate]:
        rows = self.connection.execute(
            """SELECT * FROM preference_candidates WHERE home_id=? AND user_id=?
               AND status='pending' ORDER BY updated_at DESC""",
            (home_id, user_id),
        ).fetchall()
        return [self._to_candidate(row) for row in rows]

    def resolve_candidate(
        self, candidate_id: str, home_id: str, user_id: str, status: str,
        confirmed_memory_id: str | None = None,
    ) -> bool:
        cursor = self.connection.execute(
            """UPDATE preference_candidates SET status=?, confirmed_memory_id=?,
               updated_at=? WHERE id=? AND home_id=? AND user_id=? AND status='pending'""",
            (status, confirmed_memory_id, utc_now().isoformat(), candidate_id, home_id, user_id),
        )
        self.connection.commit()
        return cursor.rowcount > 0

    def add_conflict(
        self, memory: MemoryRecord, incoming: dict, resolved: dict, resolution: str,
    ) -> MemoryConflict:
        conflict_id, now = str(uuid.uuid4()), utc_now().isoformat()
        self.connection.execute(
            "INSERT INTO memory_conflicts VALUES (?,?,?,?,?,?,?,?)",
            (conflict_id, memory.id, memory.home_id,
             json.dumps(memory.memory_value, ensure_ascii=False),
             json.dumps(incoming, ensure_ascii=False),
             json.dumps(resolved, ensure_ascii=False), resolution, now),
        )
        self.connection.commit()
        return self.list_conflicts(memory.home_id, memory.id)[-1]

    def list_conflicts(self, home_id: str, memory_id: str | None = None) -> list[MemoryConflict]:
        sql = "SELECT * FROM memory_conflicts WHERE home_id=?"
        params: list[str] = [home_id]
        if memory_id:
            sql += " AND memory_id=?"
            params.append(memory_id)
        sql += " ORDER BY created_at"
        rows = self.connection.execute(sql, params).fetchall()
        return [self._to_conflict(row) for row in rows]

    def decay_confidence(self, older_than: datetime, factor: float, floor: float) -> int:
        cursor = self.connection.execute(
            """UPDATE memories SET confidence=MAX(?, confidence * ?), updated_at=?
               WHERE status='active' AND updated_at<? AND confidence>?""",
            (floor, factor, utc_now().isoformat(), older_than.isoformat(), floor),
        )
        self.connection.commit()
        return cursor.rowcount

    def active_count(self, home_id: str | None = None) -> int:
        if home_id is None:
            row = self.connection.execute(
                "SELECT COUNT(*) FROM memories WHERE status='active'"
            ).fetchone()
        else:
            row = self.connection.execute(
                "SELECT COUNT(*) FROM memories WHERE status='active' AND home_id=?", (home_id,)
            ).fetchone()
        return int(row[0])

    def list_accessible(
        self, home_id: str, user_id: str, room_id: str | None = None,
        device_id: str | None = None,
    ) -> list[MemoryRecord]:
        self.cleanup_expired(home_id)
        rows = self.connection.execute(
            """
            SELECT * FROM memories
            WHERE home_id=? AND status='active'
              AND (user_id IS NULL OR user_id=?)
              AND (room_id IS NULL OR room_id=?)
              AND (device_id IS NULL OR device_id=?)
            ORDER BY CASE scope WHEN 'home' THEN 0 WHEN 'room' THEN 1
                                WHEN 'device' THEN 2 ELSE 3 END, updated_at
            """,
            (home_id, user_id, room_id, device_id),
        ).fetchall()
        return [self._to_record(row) for row in rows]

    def get(self, memory_id: str, home_id: str) -> MemoryRecord | None:
        row = self.connection.execute(
            "SELECT * FROM memories WHERE id=? AND home_id=? AND status='active'",
            (memory_id, home_id),
        ).fetchone()
        return self._to_record(row) if row else None

    def update_value(self, memory_id: str, home_id: str, value: dict) -> MemoryRecord | None:
        self.connection.execute(
            "UPDATE memories SET memory_value=?, updated_at=? WHERE id=? AND home_id=? AND status='active'",
            (json.dumps(value, ensure_ascii=False), utc_now().isoformat(), memory_id, home_id),
        )
        self.connection.commit()
        return self.get(memory_id, home_id)

    def delete(self, memory_id: str, home_id: str) -> bool:
        cursor = self.connection.execute(
            "UPDATE memories SET status='deleted', updated_at=? WHERE id=? AND home_id=? AND status='active'",
            (utc_now().isoformat(), memory_id, home_id),
        )
        self.connection.commit()
        return cursor.rowcount > 0

    def delete_user_memories(self, home_id: str, user_id: str) -> int:
        cursor = self.connection.execute(
            "UPDATE memories SET status='deleted', updated_at=? WHERE home_id=? AND user_id=? AND status='active'",
            (utc_now().isoformat(), home_id, user_id),
        )
        self.connection.commit()
        return cursor.rowcount

    def cleanup_expired(self, home_id: str | None = None) -> int:
        home_filter = " AND home_id=?" if home_id is not None else ""
        parameters = [utc_now().isoformat()]
        if home_id is not None:
            parameters.append(home_id)
        parameters.append(utc_now().isoformat())
        cursor = self.connection.execute(
            f"""UPDATE memories SET status='expired', updated_at=?
                WHERE status='active'{home_filter}
                  AND expires_at IS NOT NULL AND expires_at<=?""",
            parameters,
        )
        self.connection.commit()
        return cursor.rowcount

    def close(self) -> None:
        self.connection.close()

    @staticmethod
    def _to_record(row: sqlite3.Row) -> MemoryRecord:
        data = dict(row)
        data["memory_value"] = json.loads(data["memory_value"])
        return MemoryRecord.model_validate(data)

    @staticmethod
    def _to_candidate(row: sqlite3.Row) -> PreferenceCandidate:
        data = dict(row)
        data["memory_value"] = json.loads(data["memory_value"])
        return PreferenceCandidate.model_validate(data)

    @staticmethod
    def _to_conflict(row: sqlite3.Row) -> MemoryConflict:
        data = dict(row)
        for key in ("previous_value", "incoming_value", "resolved_value"):
            data[key] = json.loads(data[key])
        return MemoryConflict.model_validate(data)
