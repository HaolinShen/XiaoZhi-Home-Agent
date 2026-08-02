"""Deterministic conversation window and rolling-summary helpers."""

from __future__ import annotations

from typing import Iterable

from langchain_core.messages import RemoveMessage, ToolMessage


def estimate_tokens(messages: Iterable) -> int:
    """Return a cheap, deterministic token estimate suitable for guardrails."""
    return sum(max(1, (len(str(getattr(message, "content", ""))) + 1) // 2) for message in messages)


def compact_messages(
    messages: list,
    *,
    max_messages: int = 12,
    max_tokens: int = 2400,
    max_tool_result_chars: int = 1200,
    max_summary_chars: int = 1800,
) -> tuple[list, str]:
    """Keep a bounded recent window and summarize messages that fall out."""
    if not messages:
        return [], ""
    messages = [
        _truncate_message(message, max_tool_result_chars)
        if isinstance(message, ToolMessage)
        else message
        for message in messages
    ]
    keep_from = max(0, len(messages) - max_messages)
    while keep_from < len(messages) - 1 and estimate_tokens(messages[keep_from:]) > max_tokens:
        keep_from += 1
    old = messages[:keep_from]
    recent = messages[keep_from:]
    summary_parts = []
    for message in old:
        role = getattr(message, "type", "message")
        content = str(getattr(message, "content", ""))
        if content:
            summary_parts.append(f"{role}: {content[:240]}")
    summary = "\n".join(summary_parts)[-max_summary_chars:]
    return recent, summary


def build_compaction_update(
    messages: list,
    existing_summary: str = "",
    *,
    max_messages: int = 12,
    max_tokens: int = 2400,
    max_tool_result_chars: int = 1200,
    max_summary_chars: int = 1800,
) -> tuple[list, str, int]:
    """Build an ``add_messages`` update that removes compacted checkpoint data."""
    recent, generated = compact_messages(
        messages,
        max_messages=max_messages,
        max_tokens=max_tokens,
        max_tool_result_chars=max_tool_result_chars,
        max_summary_chars=max_summary_chars,
    )
    recent_ids = {message.id for message in recent if getattr(message, "id", None)}
    removals = [
        RemoveMessage(id=message.id)
        for message in messages
        if getattr(message, "id", None) and message.id not in recent_ids
    ]
    replacements = [
        message
        for original, message in zip(messages[-len(recent):], recent)
        if getattr(message, "content", None) != getattr(original, "content", None)
    ] if recent else []
    merged_summary = _merge_summary(existing_summary, generated, max_summary_chars)
    return removals + replacements, merged_summary, estimate_tokens(recent)


def _merge_summary(existing: str, generated: str, max_chars: int) -> str:
    parts = [part for part in (existing.strip(), generated.strip()) if part]
    return "\n".join(parts)[-max_chars:]


def _truncate_message(message, max_chars: int):
    content = str(getattr(message, "content", ""))
    if max_chars <= 0 or len(content) <= max_chars:
        return message
    marker = "\n…（工具结果已裁剪）"
    kept = max(0, max_chars - len(marker))
    return message.model_copy(update={"content": content[:kept] + marker})
