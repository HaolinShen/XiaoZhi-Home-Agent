"""Public helpers for custom graph progress events."""

from langgraph.config import get_stream_writer


def emit_progress(event: str, **payload) -> None:
    get_stream_writer()({"event": event, **payload})
