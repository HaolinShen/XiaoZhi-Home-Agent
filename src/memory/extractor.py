"""Conservative natural-language memory candidate extraction.

The extractor never writes long-term memory. It only returns candidates that
must be confirmed by the user through MemoryService.
"""

from __future__ import annotations

import re

from .models import ExtractedMemoryCandidate

_STABLE_MARKERS = ("我喜欢", "我偏好", "我习惯", "我通常", "我一般", "以后都", "以后请")
_TEMPORARY_MARKERS = ("今天", "这次", "现在", "有点", "暂时", "刚才")


def extract_memory_candidates(text: str) -> list[ExtractedMemoryCandidate]:
    """Extract high-precision preference/routine candidates from one utterance."""
    normalized = text.strip()
    if not normalized or any(marker in normalized for marker in _TEMPORARY_MARKERS):
        return []
    if not any(marker in normalized for marker in _STABLE_MARKERS):
        return []

    candidates: list[ExtractedMemoryCandidate] = []
    temperature = re.search(r"(?:空调|温度).*?(1[6-9]|2\d|30)\s*(?:度|℃|°C)?", normalized, re.I)
    if temperature:
        candidates.append(_candidate(
            "ac.temperature", {"temperature": int(temperature.group(1))}, normalized, 0.82, 0.75
        ))

    brightness = re.search(r"(?:灯|亮度).*?(\d{1,3})\s*%", normalized)
    if brightness and 0 <= int(brightness.group(1)) <= 100:
        candidates.append(_candidate(
            "lighting.brightness", {"brightness": int(brightness.group(1))}, normalized, 0.8, 0.65
        ))

    color_match = re.search(r"(暖白|暖黄|暖光|冷白光|白光)", normalized)
    if color_match and any(word in normalized for word in ("灯", "光", "色温")):
        candidates.append(_candidate(
            "lighting.color", {"color": color_match.group(1)}, normalized, 0.84, 0.65
        ))

    quiet = re.search(r"(?:晚上|夜间)?\s*(\d{1,2})(?::(\d{2}))?\s*点?.*?(?:安静|静音|低音量)", normalized)
    if quiet:
        hour = max(0, min(23, int(quiet.group(1))))
        minute = int(quiet.group(2) or 0)
        candidates.append(_candidate(
            "routine.quiet_hours", {"after": f"{hour:02d}:{minute:02d}"}, normalized, 0.76, 0.8
        ))

    return _deduplicate(candidates)


def _candidate(key, value, source, confidence, importance):
    return ExtractedMemoryCandidate(
        memory_key=key, memory_value=value, confidence=confidence,
        importance=importance, source_text=source,
    )


def _deduplicate(items: list[ExtractedMemoryCandidate]) -> list[ExtractedMemoryCandidate]:
    result = {}
    for item in items:
        result[(item.memory_key, repr(sorted(item.memory_value.items())))] = item
    return list(result.values())
