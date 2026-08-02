"""Authorization and business rules for long-term memory."""

from __future__ import annotations

from datetime import timedelta
import json
import math
import re

from .models import MemoryRecord, MemoryScope, MemoryType, MemoryWrite, PreferenceCandidate, utc_now
from .extractor import extract_memory_candidates
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
        existing = self.repository.find_by_key(
            context.home_id, owner, item.room_id, item.device_id,
            item.scope, item.memory_type, item.memory_key,
        )
        if existing:
            merged = _merge_values(existing.memory_value, item.memory_value)
            if existing.memory_value != item.memory_value:
                resolution = "merged" if merged != item.memory_value else "incoming_wins"
                self.repository.add_conflict(existing, item.memory_value, merged, resolution)
            item = item.model_copy(update={"memory_value": merged})
        return self.repository.upsert(context.home_id, owner, item)

    def record_operation(
        self, context: AgentContext, memory_key: str, memory_value: dict,
        *, minimum_repetitions: int = 3,
    ) -> PreferenceCandidate | None:
        """Aggregate a real operation and create, but never auto-save, a candidate."""
        self.spaces.validate(context)
        count = self.repository.observe_preference(
            context.home_id, context.user_id, memory_key, memory_value,
            context.room_id, context.device_id,
        )
        if count < minimum_repetitions:
            return None
        confidence = min(0.95, 0.5 + 0.1 * count)
        return self.repository.upsert_candidate(
            context.home_id, context.user_id, memory_key, memory_value,
            count, confidence, context.room_id, context.device_id,
        )

    def extract_candidates_from_text(
        self, context: AgentContext, text: str,
    ) -> list[PreferenceCandidate]:
        """Extract conservative natural-language candidates without saving memories."""
        self.spaces.validate(context)
        results = []
        for extracted in extract_memory_candidates(text):
            results.append(self.repository.upsert_candidate(
                context.home_id, context.user_id, extracted.memory_key,
                extracted.memory_value, 1, extracted.confidence,
                context.room_id, context.device_id,
                importance=extracted.importance, source_text=extracted.source_text,
            ))
        return results

    def list_candidates(self, context: AgentContext) -> list[PreferenceCandidate]:
        self.spaces.validate(context)
        return self.repository.list_candidates(context.home_id, context.user_id)

    def confirm_candidate(self, context: AgentContext, candidate_id: str) -> MemoryRecord:
        self.spaces.validate(context)
        candidate = self.repository.get_candidate(candidate_id, context.home_id)
        if candidate is None or candidate.user_id != context.user_id or candidate.status != "pending":
            raise KeyError(candidate_id)
        record = self.save(context, MemoryWrite(
            scope=MemoryScope.USER,
            memory_type=MemoryType.PREFERENCE,
            memory_key=candidate.memory_key,
            memory_value=candidate.memory_value,
            room_id=candidate.room_id,
            device_id=candidate.device_id,
            confidence=candidate.confidence,
            importance=candidate.importance,
            source=f"confirmed_candidate:{candidate.id}",
        ))
        self.repository.resolve_candidate(
            candidate.id, context.home_id, context.user_id, "confirmed", record.id
        )
        return record

    def reject_candidate(self, context: AgentContext, candidate_id: str) -> bool:
        self.spaces.validate(context)
        return self.repository.resolve_candidate(
            candidate_id, context.home_id, context.user_id, "rejected"
        )

    def decay_stale_confidence(
        self, *, stale_after: timedelta = timedelta(days=90), factor: float = 0.9,
        floor: float = 0.2,
    ) -> int:
        if not 0 < factor < 1 or not 0 <= floor <= 1:
            raise ValueError("invalid confidence decay settings")
        return self.repository.decay_confidence(utc_now() - stale_after, factor, floor)

    def evaluate_vector_retrieval(self, home_id: str | None = None, *, threshold: int = 500) -> dict:
        count = self.repository.active_count(home_id)
        return {
            "active_memory_count": count,
            "threshold": threshold,
            "recommend_vector_retrieval": count >= threshold,
            "reason": "memory_scale_threshold_reached" if count >= threshold else "structured_filters_sufficient",
        }

    def list(self, context: AgentContext) -> list[MemoryRecord]:
        self.spaces.validate(context)
        room_id = context.room_id or self.spaces.room_for_device(context.device_id)
        return self.repository.list_accessible(
            context.home_id, context.user_id, room_id, context.device_id
        )

    def retrieve(
        self, context: AgentContext, query: str, *, top_k: int = 6,
    ) -> list[MemoryRecord]:
        """Rank accessible memories by relevance, confidence, importance and use."""
        if top_k < 1:
            raise ValueError("top_k must be positive")
        records = self.list(context)
        if not records:
            return []
        now = utc_now()
        ranked = sorted(
            records,
            key=lambda record: self._retrieval_score(record, query, now),
            reverse=True,
        )[:top_k]
        self.repository.record_accesses(context.home_id, [record.id for record in ranked])
        return ranked

    @staticmethod
    def _retrieval_score(record: MemoryRecord, query: str, now) -> float:
        searchable = (
            record.memory_key + " " + json.dumps(record.memory_value, ensure_ascii=False)
        ).lower()
        terms = _query_terms(query)
        relevance = sum(1 for term in terms if term in searchable) / max(1, len(terms))
        age_days = max(0.0, (now - record.updated_at).total_seconds() / 86400)
        recency = math.exp(-age_days / 90)
        frequency = min(1.0, math.log1p(record.access_count) / math.log(11))
        return (
            0.45 * relevance + 0.20 * record.confidence
            + 0.20 * record.importance + 0.10 * recency + 0.05 * frequency
        )

    def list_versions(self, context: AgentContext, memory_id: str):
        self.get(context, memory_id)
        return self.repository.list_versions(memory_id, context.home_id)

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

    def format_for_prompt(self, context: AgentContext, query: str = "", *, top_k: int = 6) -> str:
        records = self.retrieve(context, query, top_k=top_k)
        if not records:
            return "（无可用长期记忆）"
        return "\n".join(
            f"- [{record.scope.value}/{record.memory_type.value}] "
            f"{record.memory_key}: {record.memory_value} "
            f"(confidence={record.confidence:.2f}, importance={record.importance:.2f})"
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


def _merge_values(previous: dict, incoming: dict) -> dict:
    """Preserve complementary fields while explicit incoming fields take precedence."""
    merged = dict(previous)
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_values(merged[key], value)
        else:
            merged[key] = value
    return merged


def _query_terms(query: str) -> set[str]:
    lowered = query.lower()
    terms = set(re.findall(r"[a-z0-9_.]+|[\u4e00-\u9fff]{2,}", lowered))
    aliases = {
        "空调": {"ac", "temperature", "mode", "fan"},
        "温度": {"temperature", "ac"},
        "灯": {"lighting", "brightness", "color"},
        "亮度": {"brightness", "lighting"},
        "色温": {"color", "lighting"},
        "电视": {"tv", "volume", "channel"},
        "音量": {"volume", "tv"},
        "窗帘": {"curtain", "position"},
        "安静": {"quiet", "routine"},
    }
    for marker, expansions in aliases.items():
        if marker in lowered:
            terms.update(expansions)
    return terms
