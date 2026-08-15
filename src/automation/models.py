"""Persistent models for scheduled and vehicle-triggered routines."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RoutineAction(BaseModel):
    """One action relative to a routine's trigger time.

    ``offset_minutes`` is relative to the anchor event. For a wake routine,
    the anchor is wake-up time. For an arrival routine, it is the geofence ETA.
    """

    id: str = Field(default_factory=lambda: uuid4().hex)
    offset_minutes: int = 0
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    description: str = ""
    requires_confirmation: bool = False
    enabled: bool = True


class Routine(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    home_id: str
    user_id: str
    name: str
    trigger_type: Literal["fixed_time", "vehicle_eta", "vehicle_geofence", "manual"]
    timezone: str = "Asia/Shanghai"
    enabled: bool = True
    actions: list[RoutineAction] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScheduledTask(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    routine_id: str
    home_id: str
    user_id: str
    action_id: str
    due_at: datetime
    payload: dict[str, Any] = Field(default_factory=dict)
    status: Literal["pending", "running", "completed", "failed", "cancelled"] = "pending"
    dedupe_key: str
    attempts: int = 0
    error: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    executed_at: datetime | None = None


class RoutineRun(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    routine_id: str
    trigger_key: str
    anchor_at: datetime
    status: Literal["scheduled", "running", "completed", "partial", "failed", "cancelled"] = "scheduled"
    created_at: datetime = Field(default_factory=utc_now)


class VehicleEvent(BaseModel):
    vehicle_id: str
    home_id: str
    event_type: Literal["location", "eta_update", "geofence_enter", "ignition_on", "ignition_off"]
    latitude: float
    longitude: float
    eta_minutes: int | None = Field(default=None, ge=0)
    occurred_at: datetime = Field(default_factory=utc_now)
    event_id: str = Field(default_factory=lambda: uuid4().hex)
    metadata: dict[str, Any] = Field(default_factory=dict)
