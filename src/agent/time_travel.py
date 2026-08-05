"""Safe helpers for inspecting and branching LangGraph checkpoints."""

from typing import Any


def list_state_history(graph, config: dict, *, limit: int = 20) -> list[dict[str, Any]]:
    if limit < 1:
        raise ValueError("limit must be positive")
    history = []
    for snapshot in graph.get_state_history(config, limit=limit):
        checkpoint = snapshot.config.get("configurable", {}).get("checkpoint_id")
        history.append({
            "checkpoint_id": checkpoint,
            "created_at": snapshot.created_at,
            "next": list(snapshot.next),
            "metadata": dict(snapshot.metadata),
            "state_keys": sorted(snapshot.values.keys()),
        })
    return history


def fork_from_checkpoint(graph, config: dict, checkpoint_id: str, updates: dict) -> dict:
    """Fork a historical state by applying updates through LangGraph's API."""
    for snapshot in graph.get_state_history(config):
        current_id = snapshot.config.get("configurable", {}).get("checkpoint_id")
        if current_id == checkpoint_id:
            return graph.update_state(snapshot.config, updates)
    raise KeyError(f"checkpoint not found: {checkpoint_id}")
