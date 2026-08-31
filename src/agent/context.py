"""Trusted request context and smart-home space validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..devices.base import DeviceRegistry


class ContextValidationError(ValueError):
    """Raised when request identity or spatial context is invalid."""


class AgentContext(BaseModel):
    """Identity and optional App location supplied by the trusted backend."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    home_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    client_id: str = Field(min_length=1)
    room_id: str | None = None
    device_id: str | None = None
    is_admin: bool = False

    @field_validator("home_id", "user_id", "session_id", "client_id")
    @classmethod
    def reject_blank_required_ids(cls, value: str) -> str:
        if not value:
            raise ValueError("identifier must not be blank")
        return value

    @field_validator("room_id", "device_id")
    @classmethod
    def normalize_optional_ids(cls, value: str | None) -> str | None:
        return value or None

    def to_config(self) -> dict:
        """Build the only supported LangGraph configurable payload."""
        return {
            "configurable": {
                "thread_id": self.session_id,
                **self.model_dump(),
            }
        }

    def to_state_input(self) -> dict:
        """Expose request location to the context synchronization node."""
        return {
            "request_home_id": self.home_id,
            "request_user_id": self.user_id,
            "request_client_id": self.client_id,
            "request_session_id": self.session_id,
            "request_room_id": self.room_id,
            "request_device_id": self.device_id,
            "request_is_admin": self.is_admin,
        }


@dataclass(frozen=True)
class DeviceLocation:
    home_id: str
    room_id: str


class SpaceDirectory:
    """Trusted mapping used to validate room and device ownership."""

    def __init__(
        self,
        rooms_by_home: Mapping[str, set[str]],
        devices: Mapping[str, DeviceLocation],
    ) -> None:
        self._rooms_by_home = {
            home_id: frozenset(room_ids)
            for home_id, room_ids in rooms_by_home.items()
        }
        self._devices = dict(devices)

    @classmethod
    def from_registry(
        cls,
        registry: DeviceRegistry,
        home_id: str,
    ) -> SpaceDirectory:
        """Create a directory for the single-home simulator backend."""
        devices: dict[str, DeviceLocation] = {}
        rooms: set[str] = set()
        for device_id, device in registry.get_all().items():
            room_id = location_to_room_id(device.location)
            rooms.add(room_id)
            devices[device_id] = DeviceLocation(home_id=home_id, room_id=room_id)
        return cls({home_id: rooms}, devices)

    def validate(self, context: AgentContext) -> AgentContext:
        rooms = self._rooms_by_home.get(context.home_id)
        if rooms is None:
            raise ContextValidationError(f"unknown home_id: {context.home_id}")

        if context.room_id and context.room_id not in rooms:
            raise ContextValidationError(
                f"room_id {context.room_id!r} does not belong to home {context.home_id!r}"
            )

        if context.device_id:
            location = self._devices.get(context.device_id)
            if location is None or location.home_id != context.home_id:
                raise ContextValidationError(
                    f"device_id {context.device_id!r} does not belong to home {context.home_id!r}"
                )
            if context.room_id and location.room_id != context.room_id:
                raise ContextValidationError(
                    f"device_id {context.device_id!r} does not belong to room {context.room_id!r}"
                )

        return context

    def room_for_device(self, device_id: str | None) -> str | None:
        if device_id is None:
            return None
        location = self._devices.get(device_id)
        return location.room_id if location else None

    def resolve_room_mention(self, text: str) -> str | None:
        """Resolve an explicit room mention without asking the model to set IDs."""
        aliases = {
            "客厅": "living_room",
            "主卧": "bedroom",
            "卧室": "bedroom",
            "厨房": "kitchen",
        }
        for alias, room_id in aliases.items():
            if alias in text:
                return room_id
        return None


def location_to_room_id(location: str) -> str:
    """Convert simulator location labels to stable business identifiers."""
    aliases = {
        "客厅": "living_room",
        "卧室": "bedroom",
        "厨房": "kitchen",
        "玄关": "entryway",
    }
    return aliases.get(location, location.strip().lower().replace(" ", "_"))
