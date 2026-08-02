"""SQLite repository for structured long-term memories."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime

from .models import MemoryRecord, MemoryScope, MemoryType, MemoryWrite, utc_now


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
