"""Long-term memory tools whose identity always comes from RunnableConfig.

P1 改造: 工具不再读模块级 `_service` 单例，而是由 `build_memory_tools(service)`
工厂以闭包持有；`service=None` 时工具照常存在并返回"长期记忆未启用"，
和旧行为一致，但依赖关系从隐式单例变成构造期显式传入。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool

from ..agent.context import AgentContext
from ..memory import MemoryScope, MemoryService, MemoryType, MemoryWrite

_OPERATION_IDENTITY_KEYS = ("home_id", "user_id", "thread_id", "client_id")


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


def make_preference_recorder(service, enabled: bool):
    """构造设备工具的"行为观察"记录器。

    enabled=False 或 service=None 时返回 no-op：这是**构造期的显式选择**
    （后台自动化执行器、MCP 这类机器触发的调用方不参与偏好学习），而不是
    以前"逐键检查后安静跳过"的隐式兜底。

    否则身份必须完整：缺任一键直接 raise，让"定时动作因缺身份被判失败"
    这类 bug 立刻暴露。曾经的实现靠 `if config is not None` 守卫，LangChain
    总会注入空 configurable 的 config，那个判断恒为真、一个字都拦不住。
    """
    if not enabled or service is None:
        def _noop(config, device_id, memory_key, memory_value) -> None:
            return None

        return _noop

    def recorder(config, device_id, memory_key, memory_value) -> None:
        configurable = (config or {}).get("configurable", {})
        missing = [key for key in _OPERATION_IDENTITY_KEYS if not configurable.get(key)]
        if missing:
            raise RuntimeError(f"记录设备偏好缺少可信身份: {missing}")
        context = _context(config).model_copy(update={"room_id": None, "device_id": device_id})
        service.record_operation(context, memory_key, memory_value)

    return recorder


def _build_memory_tools(service: MemoryService | None) -> list[StructuredTool]:
    def save_personal_memory(
        memory_key: str,
        memory_value: dict[str, Any],
        source: str,
        config: RunnableConfig,
    ) -> str:
        """Save an explicitly requested personal preference. Identity is injected by the server."""
        if service is None:
            return "长期记忆未启用"
        context = _context(config)
        record = service.save(context, MemoryWrite(
            scope=MemoryScope.USER,
            memory_type=MemoryType.PREFERENCE,
            memory_key=memory_key,
            memory_value=memory_value,
            room_id=context.room_id,
            device_id=context.device_id,
            source=source,
        ))
        return f"已保存个人记忆 {record.memory_key}（id={record.id}）"

    def save_home_rule(
        memory_key: str,
        memory_value: dict[str, Any],
        source: str,
        config: RunnableConfig,
    ) -> str:
        """Save an explicit home-wide rule; trusted configuration must grant admin access."""
        if service is None:
            return "长期记忆未启用"
        context = _context(config)
        record = service.save(
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

    def list_personal_memories(config: RunnableConfig) -> str:
        """List shared and personal memories accessible in the current request scope."""
        if service is None:
            return "长期记忆未启用"
        records = service.list_memories(_context(config))
        return json.dumps([
            {"id": r.id, "scope": r.scope.value, "key": r.memory_key, "value": r.memory_value}
            for r in records
        ], ensure_ascii=False)

    def update_personal_memory(
        memory_id: str,
        memory_value: dict[str, Any],
        config: RunnableConfig,
    ) -> str:
        """Update one personal memory visible to the current trusted user."""
        if service is None:
            return "长期记忆未启用"
        record = service.update(_context(config), memory_id, memory_value)
        return f"已更新个人记忆 {record.memory_key}"

    def delete_personal_memory(memory_id: str, config: RunnableConfig) -> str:
        """Delete one memory owned by the current user; shared rules require an admin API."""
        if service is None:
            return "长期记忆未启用"
        service.delete(_context(config), memory_id)
        return "记忆已删除"

    def list_preference_candidates(config: RunnableConfig) -> str:
        """查看系统根据重复操作生成、等待用户确认的偏好候选。"""
        if service is None:
            return "长期记忆未启用"
        return json.dumps([
            {"id": c.id, "key": c.memory_key, "value": c.memory_value,
             "observations": c.observation_count, "confidence": c.confidence,
             "importance": c.importance, "source_text": c.source_text}
            for c in service.list_candidates(_context(config))
        ], ensure_ascii=False)

    def confirm_preference_candidate(candidate_id: str, config: RunnableConfig) -> str:
        """用户明确确认一个偏好候选后，将其保存为个人长期记忆。"""
        if service is None:
            return "长期记忆未启用"
        record = service.confirm_candidate(_context(config), candidate_id)
        return f"已确认并保存偏好 {record.memory_key}（id={record.id}）"

    def reject_preference_candidate(candidate_id: str, config: RunnableConfig) -> str:
        """拒绝一个偏好候选，不会写入长期记忆。"""
        if service is None:
            return "长期记忆未启用"
        if not service.reject_candidate(_context(config), candidate_id):
            raise KeyError(candidate_id)
        return "已拒绝该偏好候选"

    def list_memory_versions(memory_id: str, config: RunnableConfig) -> str:
        """查看一条当前可访问记忆的历史版本和有效时间区间。"""
        if service is None:
            return "长期记忆未启用"
        versions = service.list_versions(_context(config), memory_id)
        return json.dumps([
            {
                "version": v.version,
                "value": v.memory_value,
                "valid_from": v.valid_from.isoformat(),
                "valid_to": v.valid_to.isoformat() if v.valid_to else None,
                "source": v.source,
            }
            for v in versions
        ], ensure_ascii=False)

    # 显式标注：不写的话列表元素类型被推断成第一个函数的具体签名，
    # 后面 from_function 对其余函数全部报错。
    functions: list[Callable[..., str]] = [
        save_personal_memory,
        save_home_rule,
        list_personal_memories,
        update_personal_memory,
        delete_personal_memory,
        list_preference_candidates,
        confirm_preference_candidate,
        reject_preference_candidate,
        list_memory_versions,
    ]
    # 不标注的话，列表类型会被推断成第一个函数的具体签名，from_function 随之报错
    return [
        StructuredTool.from_function(fn, name=fn.__name__, description=fn.__doc__)
        for fn in functions
    ]


def build_memory_tools(service: MemoryService | None) -> list[StructuredTool]:
    """构建全部长期记忆工具（service=None 时工具返回"长期记忆未启用"）。"""
    return _build_memory_tools(service)


__all__ = [
    "build_memory_tools",
    "make_preference_recorder",
]
