"""Agentic RAG subgraph: identify, retrieve, rewrite, answer or refuse."""

from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph

from .base import KnowledgeBase


DEVICE_PROFILES = {
    "living_room_ac": {"model": "SmartCool-AC2024", "label": "客厅空调"},
    "bedroom_ac": {"model": "SmartCool-AC2024", "label": "卧室空调"},
    "living_room_tv": {"model": "VisionTV-V1", "label": "客厅电视"},
}


class RagState(TypedDict, total=False):
    query: str
    rewritten_query: str
    device_id: str | None
    device_model: str | None
    hits: list[dict]
    answer: str
    citations: list[str]
    rag_status: str
    rewrite_count: int
    trajectory: list[dict]


def resolve_device_profile(query: str) -> tuple[str | None, str | None]:
    explicit = [(device_id, profile) for device_id, profile in DEVICE_PROFILES.items() if profile["label"] in query]
    if explicit:
        device_id, profile = explicit[0]
        return device_id, profile["model"]
    if "空调" in query:
        return None, "SmartCool-AC2024"
    if "电视" in query:
        return "living_room_tv", "VisionTV-V1"
    return None, None


def rewrite_knowledge_query(query: str) -> str:
    rewritten = query.replace("是什么意思", "故障 原因 处理").replace("怎么办", "原因 处理步骤")
    return rewritten + " 说明书 故障排查"


def build_knowledge_rag_subgraph(knowledge: KnowledgeBase, *, top_k: int = 3, max_rewrites: int = 1):
    def identify(state: RagState):
        device_id, model = resolve_device_profile(state["query"])
        return {
            "device_id": device_id, "device_model": model, "rewrite_count": 0,
            "trajectory": [{"step": "identify", "device_id": device_id, "model": model}],
        }

    def retrieve(state: RagState):
        query = state.get("rewritten_query") or state["query"]
        hits = knowledge.search(query, model=state.get("device_model"), top_k=top_k)
        serialized = [hit.model_dump() for hit in hits if hit.score >= 0.12]
        trajectory = list(state.get("trajectory", []))
        trajectory.append({"step": "retrieve", "query": query, "hit_count": len(serialized)})
        return {"hits": serialized, "trajectory": trajectory}

    def after_retrieve(state: RagState):
        if state.get("hits"):
            return "answer"
        if state.get("rewrite_count", 0) < max_rewrites:
            return "rewrite"
        return "refuse"

    def rewrite(state: RagState):
        rewritten = rewrite_knowledge_query(state["query"])
        trajectory = list(state.get("trajectory", []))
        trajectory.append({"step": "rewrite", "query": rewritten})
        return {"rewritten_query": rewritten, "rewrite_count": state.get("rewrite_count", 0) + 1, "trajectory": trajectory}

    def answer(state: RagState):
        hits = state["hits"]
        citations = [f"{hit['chunk']['source']}#{hit['chunk']['section']}" for hit in hits]
        excerpts = [hit["chunk"]["content"] for hit in hits]
        response = "根据设备文档：\n\n" + "\n\n".join(excerpts)
        response += "\n\n来源：\n" + "\n".join(f"- {citation}" for citation in citations)
        trajectory = list(state.get("trajectory", []))
        trajectory.append({"step": "answer", "citation_count": len(citations)})
        return {"answer": response, "citations": citations, "rag_status": "answered", "trajectory": trajectory}

    def refuse(state: RagState):
        trajectory = list(state.get("trajectory", []))
        trajectory.append({"step": "refuse", "reason": "no_supported_document"})
        return {
            "answer": "当前知识库中没有找到足以支持答案的对应设备文档，我不能可靠确认。请提供设备型号或补充说明书。",
            "citations": [], "rag_status": "refused", "trajectory": trajectory,
        }

    graph = StateGraph(RagState)
    graph.add_node("identify", identify)
    graph.add_node("retrieve", retrieve)
    graph.add_node("rewrite", rewrite)
    graph.add_node("answer", answer)
    graph.add_node("refuse", refuse)
    graph.set_entry_point("identify")
    graph.add_edge("identify", "retrieve")
    graph.add_conditional_edges("retrieve", after_retrieve, {"answer": "answer", "rewrite": "rewrite", "refuse": "refuse"})
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("answer", END)
    graph.add_edge("refuse", END)
    return graph.compile()
