"""Deterministic metrics for Agentic RAG execution trajectories."""


def evaluate_rag_trajectory(state: dict, *, expected_status: str, expected_source: str | None = None) -> dict:
    citations = state.get("rag_citations", [])
    trajectory = state.get("rag_trajectory", [])
    actual_status = state.get("rag_status")
    route_correct = state.get("intent") == "device_knowledge"
    source_correct = expected_source is None or any(expected_source in item for item in citations)
    return {
        "route_accuracy": 1.0 if route_correct else 0.0,
        "status_accuracy": 1.0 if actual_status == expected_status else 0.0,
        "source_accuracy": 1.0 if source_correct else 0.0,
        "has_retrieval": any(item.get("step") == "retrieve" for item in trajectory),
        "rewrite_count": sum(1 for item in trajectory if item.get("step") == "rewrite"),
        "citation_count": len(citations),
        "refusal_correct": actual_status == "refused" and not citations,
    }
