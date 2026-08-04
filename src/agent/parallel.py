"""Subgraph and dynamic fan-out for independent device status queries."""

import operator
from typing import Annotated
from typing_extensions import NotRequired, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.types import Send

from ..devices.base import DeviceRegistry


class QueryState(TypedDict):
    query: str
    targets: list[str]
    device_id: NotRequired[str]
    parallel_results: Annotated[list[dict], operator.add]
    response: NotRequired[str]


def extract_query_targets(query: str, registry: DeviceRegistry) -> list[str]:
    """Resolve room/device mentions without guessing ambiguous device types."""
    text = query.strip()
    devices = registry.get_all()
    if any(word in text for word in ("所有设备", "全部设备", "家里设备")):
        return list(devices)
    exact_targets = [device_id for device_id, device in devices.items() if device.name in text]
    if exact_targets:
        return exact_targets
    targets = []
    for device_id, device in devices.items():
        if device.name in text or device.location in text:
            targets.append(device_id)
    # Deduplicate while retaining registry order.
    return list(dict.fromkeys(targets))


def should_use_parallel_query(query: str, registry: DeviceRegistry) -> bool:
    return len(extract_query_targets(query, registry)) >= 2


def build_device_query_subgraph(registry: DeviceRegistry):
    """Build a reusable subgraph: prepare → Send fan-out → aggregate."""
    def dispatch_node(state: QueryState):
        return {}

    def fan_out(state: QueryState):
        return [Send("query_device", {"device_id": device_id}) for device_id in state["targets"]]

    def query_device(state: QueryState):
        device = registry.get(state["device_id"])
        if device is None:
            result = {"device_id": state["device_id"], "ok": False, "text": "设备不存在"}
        else:
            result = {"device_id": device.device_id, "ok": True, "text": device.to_status_text()}
        return {"parallel_results": [result]}

    def aggregate(state: QueryState):
        results = sorted(state.get("parallel_results", []), key=lambda item: item["device_id"])
        lines = [item["text"] for item in results]
        return {"response": "\n".join(lines) if lines else "没有找到可查询的设备。"}

    graph = StateGraph(QueryState)
    graph.add_node("dispatch", dispatch_node)
    graph.add_node("query_device", query_device)
    graph.add_node("aggregate", aggregate)
    graph.set_entry_point("dispatch")
    graph.add_conditional_edges("dispatch", fan_out, ["query_device"])
    graph.add_edge("query_device", "aggregate")
    graph.add_edge("aggregate", END)
    return graph.compile()
