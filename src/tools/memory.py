"""Long-term memory tools whose identity always comes from RunnableConfig."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from ..agent.context import AgentContext
from ..memory import MemoryScope, MemoryService, MemoryType, MemoryWrite

_service: MemoryService | None = None


def set_memory_service(service: MemoryService | None) -> None:
    global _service
    _service = service


def _context(config: RunnableConfig) -> AgentContext:
    configurable = config.get("configurable", {})
    return AgentContext(
        home_id=configurable["home_id"],
        user_id=configurable["user_id"],
        session_id=configurable["thread_id"],
        client_id=configurable["client_id"],
        room_id=configurable.get("room_id"),
        device_id=configurable.get("device_id"),
    )


def _is_admin(config: RunnableConfig) -> bool:
    """Read authorization only from trusted server-side configuration."""
    return config.get("configurable", {}).get("is_admin") is True


def record_preference_operation(
    config: RunnableConfig,
    device_id: str,
    memory_key: str,
    memory_value: dict[str, Any],
) -> None:
    """Record a successful device setting without exposing identity to the model."""
    if _service is None:
        return
    context = _context(config).model_copy(update={"room_id": None, "device_id": device_id})
    _service.record_operation(context, memory_key, memory_value)


@tool
def save_personal_memory(
    memory_key: str,
    memory_value: dict[str, Any],
    source: str,
    config: RunnableConfig,
) -> str:
    """Save an explicitly requested personal preference. Identity is injected by the server."""
    if _service is None:
        return "长期记忆未启用"
    context = _context(config)
    record = _service.save(context, MemoryWrite(
        scope=MemoryScope.USER,
        memory_type=MemoryType.PREFERENCE,
        memory_key=memory_key,
        memory_value=memory_value,
        room_id=context.room_id,
        device_id=context.device_id,
        source=source,
    ))
    return f"已保存个人记忆 {record.memory_key}（id={record.id}）"


@tool
def save_home_rule(
    memory_key: str,
    memory_value: dict[str, Any],
    source: str,
    config: RunnableConfig,
) -> str:
    """Save an explicit home-wide rule; trusted configuration must grant admin access."""
    if _service is None:
        return "长期记忆未启用"
    context = _context(config)
    record = _service.save(
        context,
        MemoryWrite(
            scope=MemoryScope.HOME,
            memory_type=MemoryType.CONSTRAINT,
            memory_key=memory_key,
            memory_value=memory_value,
            source=source,
        ),
        is_admin=_is_admin(config),
    )
    return f"已保存家庭规则 {record.memory_key}（id={record.id}）"


@tool
def list_personal_memories(config: RunnableConfig) -> str:
    """List shared and personal memories accessible in the current request scope."""
    if _service is None:
        return "长期记忆未启用"
    records = _service.list(_context(config))
    return json.dumps([
        {"id": r.id, "scope": r.scope.value, "key": r.memory_key, "value": r.memory_value}
        for r in records
    ], ensure_ascii=False)


@tool
def update_personal_memory(
    memory_id: str,
    memory_value: dict[str, Any],
    config: RunnableConfig,
) -> str:
    """Update one personal memory visible to the current trusted user."""
    if _service is None:
        return "长期记忆未启用"
    record = _service.update(_context(config), memory_id, memory_value)
    return f"已更新个人记忆 {record.memory_key}"


@tool
def delete_personal_memory(memory_id: str, config: RunnableConfig) -> str:
    """Delete one memory owned by the current user; shared rules require an admin API."""
    if _service is None:
        return "长期记忆未启用"
    _service.delete(_context(config), memory_id)
    return "记忆已删除"


@tool
def list_preference_candidates(config: RunnableConfig) -> str:
    """查看系统根据重复操作生成、等待用户确认的偏好候选。"""
    if _service is None:
        return "长期记忆未启用"
    return json.dumps([
        {"id": c.id, "key": c.memory_key, "value": c.memory_value,
         "observations": c.observation_count, "confidence": c.confidence}
        for c in _service.list_candidates(_context(config))
    ], ensure_ascii=False)


@tool
def confirm_preference_candidate(candidate_id: str, config: RunnableConfig) -> str:
    """用户明确确认一个偏好候选后，将其保存为个人长期记忆。"""
    if _service is None:
        return "长期记忆未启用"
    record = _service.confirm_candidate(_context(config), candidate_id)
    return f"已确认并保存偏好 {record.memory_key}（id={record.id}）"


@tool
def reject_preference_candidate(candidate_id: str, config: RunnableConfig) -> str:
    """拒绝一个偏好候选，不会写入长期记忆。"""
    if _service is None:
        return "长期记忆未启用"
    if not _service.reject_candidate(_context(config), candidate_id):
        raise KeyError(candidate_id)
    return "已拒绝该偏好候选"
