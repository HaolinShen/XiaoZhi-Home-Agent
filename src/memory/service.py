"""Authorization and business rules for long-term memory."""

from __future__ import annotations

from .models import MemoryRecord, MemoryScope, MemoryWrite
from .repository import MemoryRepository
from ..agent.context import AgentContext, SpaceDirectory


class MemoryPermissionError(PermissionError):
    pass


class MemoryService:
    def __init__(self, repository: MemoryRepository, spaces: SpaceDirectory) -> None:
        self.repository = repository
        self.spaces = spaces

    def save(
        self,
        context: AgentContext,
        item: MemoryWrite,
        *,
        is_admin: bool = False,
    ) -> MemoryRecord:
        """Save one memory after validating ownership, scope, and space."""
        self.spaces.validate(context)
        item = self._normalize_and_validate_scope(context, item)

        shared_scope = item.scope in {
            MemoryScope.HOME,
            MemoryScope.ROOM,
            MemoryScope.DEVICE,
        }
        if shared_scope and not (is_admin or context.is_admin):
            raise MemoryPermissionError(
                "home, room, and device memories require administrator permission"
            )

        owner = None if shared_scope else context.user_id
        return self.repository.upsert(context.home_id, owner, item)

    def list(self, context: AgentContext) -> list[MemoryRecord]:
        self.spaces.validate(context)
        room_id = context.room_id or self.spaces.room_for_device(context.device_id)
        return self.repository.list_accessible(
            context.home_id, context.user_id, room_id, context.device_id
        )

    def get(self, context: AgentContext, memory_id: str) -> MemoryRecord:
        """Return one memory only when it is visible to the current user."""
        self.spaces.validate(context)
        record = self.repository.get(memory_id, context.home_id)
        if record is None:
            raise KeyError(memory_id)
        if record.user_id is not None and record.user_id != context.user_id:
            raise MemoryPermissionError("personal memory belongs to another user")
        return record

    def update(self, context: AgentContext, memory_id: str, value: dict, *, is_admin: bool = False) -> MemoryRecord:
        record = self._authorized_record(
            context, memory_id, is_admin=is_admin or context.is_admin
        )
        updated = self.repository.update_value(record.id, context.home_id, value)
        if updated is None:
            raise KeyError(memory_id)
        return updated

    def delete(self, context: AgentContext, memory_id: str, *, is_admin: bool = False) -> bool:
        self._authorized_record(
            context, memory_id, is_admin=is_admin or context.is_admin
        )
        return self.repository.delete(memory_id, context.home_id)

    def clear_personal(self, context: AgentContext) -> int:
        self.spaces.validate(context)
        return self.repository.delete_user_memories(context.home_id, context.user_id)

    def format_for_prompt(self, context: AgentContext) -> str:
        records = self.list(context)
        if not records:
            return "（无可用长期记忆）"
        return "\n".join(
            f"- [{record.scope.value}/{record.memory_type.value}] "
            f"{record.memory_key}: {record.memory_value}"
            for record in records
        )

    def _authorized_record(self, context: AgentContext, memory_id: str, *, is_admin: bool) -> MemoryRecord:
        self.spaces.validate(context)
        record = self.repository.get(memory_id, context.home_id)
        if record is None:
            raise KeyError(memory_id)
        if record.user_id is not None and record.user_id != context.user_id:
            raise MemoryPermissionError("personal memory belongs to another user")
        if record.user_id is None and not is_admin:
            raise MemoryPermissionError("home shared memory requires administrator permission")
        return record

    def _normalize_and_validate_scope(
        self,
        context: AgentContext,
        item: MemoryWrite,
    ) -> MemoryWrite:
        room_id = item.room_id
        device_id = item.device_id

        if item.scope == MemoryScope.HOME:
            if room_id or device_id:
                raise ValueError("home memory cannot specify room_id or device_id")
        elif item.scope == MemoryScope.ROOM:
            if not room_id or device_id:
                raise ValueError("room memory requires room_id and cannot specify device_id")
        elif item.scope == MemoryScope.DEVICE:
            if not device_id:
                raise ValueError("device memory requires device_id")
            inferred_room_id = self.spaces.room_for_device(device_id)
            room_id = room_id or inferred_room_id
        elif device_id and not room_id:
            room_id = self.spaces.room_for_device(device_id)

        if room_id or device_id:
            self.spaces.validate(context.model_copy(update={
                "room_id": room_id,
                "device_id": device_id,
            }))

        return item.model_copy(update={"room_id": room_id, "device_id": device_id})
