"""Deterministic conversation window and rolling-summary helpers."""

from __future__ import annotations

from collections.abc import Iterable

from langchain_core.messages import RemoveMessage, ToolMessage


def estimate_tokens(messages: Iterable) -> int:
    """Return a cheap, deterministic token estimate suitable for guardrails."""
    return sum(max(1, (_billable_chars(message) + 1) // 2) for message in messages)


def _billable_chars(message) -> int:
    """content 字符数 + tool_calls 载荷字符数。

    坑：带 tool_calls 的 AIMessage 其 `content` 通常是空串——工具名和参数都躺在
    `tool_calls` 字段里，不在 content 里。只数 content 会让整个调用载荷被
    `max(1, ...)` 兜底成 1 token，于是预算护栏恰好在最需要它的地方
    （工具密集的长对话）失效。注意这只是缓解：完整请求体还包含 system prompt
    与全部工具 JSON Schema，两者都不在这里计数（见 docs/gap-analysis.md 3.1）。
    """
    chars = len(str(getattr(message, "content", "")))
    for call in getattr(message, "tool_calls", None) or []:
        if isinstance(call, dict):
            chars += len(str(call.get("name", ""))) + len(str(call.get("args", "")))
        else:
            chars += len(str(call))
    return chars


def _align_window_start(messages: list, keep_from: int) -> int:
    """把窗口起点推到合法边界：窗口首条不能是 ToolMessage。

    为什么必须有这一步：`keep_from` 只由「消息条数」和「估算字符长度」算出来，
    这两个量跟「哪条消息是哪条调用的结果」毫无关系。刀口一旦落在 ToolMessage 上，
    它的父 AIMessage(tool_calls=[...]) 就被切到窗口外，还会被 RemoveMessage
    从 checkpoint 物理删除，于是发往模型的请求体第一条就是「有答案、找不到问题」
    的孤儿。OpenAI 兼容协议对此不宽容——直接 400 拒绝整个请求
    （`messages with role 'tool' must be a response to a preceding message
    with 'tool_calls'`），而 CLI 会把这句英文原文打给用户，排查方向极易被带偏
    到 API Key 或网络。实测默认参数下约半数长对话会踩到，且下一轮孤儿把自己
    挤出窗口后问题自愈，表现为「偶发一次报错，重问就好」的间歇性故障。

    优先向后前进（丢掉孤儿结果，只会让窗口更小，预算天然仍满足）；只有当
    keep_from 之后全是 ToolMessage、前进会把窗口清空时（超小预算 + 一次多工具
    调用），才后退去带上父消息——宁可略微超出预算，也不能发出非法请求体。
    """
    index = keep_from
    while index < len(messages) and isinstance(messages[index], ToolMessage):
        index += 1
    if index >= len(messages):
        index = keep_from
        while index > 0 and isinstance(messages[index], ToolMessage):
            index -= 1
    return index


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
    # 条数与 token 两道闸门都算完之后再对齐边界：这两道闸门只看长度，
    # 不认识 tool_calls ↔ ToolMessage 的配对，必须由对齐兜住协议不变量。
    keep_from = _align_window_start(messages, keep_from)
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
