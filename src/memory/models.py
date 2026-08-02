"""Structured long-term memory models."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MemoryScope(str, Enum):
    HOME = "home"
    ROOM = "room"
    DEVICE = "device"
    USER = "user"


class MemoryType(str, Enum):
    PREFERENCE = "preference"
    ALIAS = "alias"
    ROUTINE = "routine"
    CONSTRAINT = "constraint"


class MemoryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    home_id: str
    user_id: str | None = None
    room_id: str | None = None
    device_id: str | None = None
    scope: MemoryScope
    memory_type: MemoryType
    memory_key: str
    memory_value: dict[str, Any]
    confidence: float = Field(default=1.0, ge=0, le=1)
    source: str | None = None
    status: str = "active"
    created_at: datetime
    updated_at: datetime
    expires_at: datetime | None = None


class MemoryWrite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope: MemoryScope
    memory_type: MemoryType
    memory_key: str = Field(min_length=1)
    memory_value: dict[str, Any]
    room_id: str | None = None
    device_id: str | None = None
    source: str | None = None
    confidence: float = Field(default=1.0, ge=0, le=1)
    expires_at: datetime | None = None


class PreferenceCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    home_id: str
    user_id: str
    room_id: str | None = None
    device_id: str | None = None
    memory_key: str
    memory_value: dict[str, Any]
    observation_count: int = Field(ge=1)
    confidence: float = Field(ge=0, le=1)
    status: str
    created_at: datetime
    updated_at: datetime
    confirmed_memory_id: str | None = None


class MemoryConflict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    memory_id: str
    home_id: str
    previous_value: dict[str, Any]
    incoming_value: dict[str, Any]
    resolved_value: dict[str, Any]
    resolution: str
    created_at: datetime
