import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import HumanMessage

from src.agent.context import AgentContext, SpaceDirectory
from src.agent.graph import build_graph
from src.agent.parallel import (
    build_device_query_subgraph,
    extract_query_targets,
    should_use_parallel_query,
)
from src.devices.base import DeviceRegistry
from src.devices.simulator import SimulatorBackend


class PhaseNineSubgraphParallelTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.registry = DeviceRegistry(SimulatorBackend())
        self.directory = SpaceDirectory.from_registry(self.registry, "home-a")
        self.context = AgentContext(
            home_id="home-a", user_id="user-a", session_id="parallel-session", client_id="test"
        )
        self.settings = SimpleNamespace(
            memory=SimpleNamespace(
                enable_long_term=False, long_term_db_path=str(Path(self.temp_dir.name) / "memory.db"),
                db_path="", context_max_messages=12, context_max_tokens=2400,
                tool_result_max_chars=1200, summary_max_chars=1800, retrieval_top_k=6,
            ),
            planning=SimpleNamespace(enabled=True, max_steps=8, max_step_retries=1, max_replans=1),
            routing=SimpleNamespace(enabled=False, confidence_threshold=0.6),
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_target_extraction_and_parallel_decision(self):
        targets = extract_query_targets("查询客厅和卧室的设备状态", self.registry)
        # 传感器也在结果里：查"状态"时读数属于状态的一部分，且读取是只读操作。
        # 玄关人体传感器不在其中，因为用户没有提到玄关。
        self.assertEqual(targets, [
            "living_room_light", "bedroom_light", "living_room_ac", "bedroom_ac",
            "living_room_tv", "living_room_curtain", "bedroom_curtain",
            "living_room_humidifier",
            "living_room_th_sensor", "bedroom_th_sensor", "living_room_presence",
        ])
        self.assertTrue(should_use_parallel_query("查询客厅和卧室的设备状态", self.registry))
        self.assertFalse(should_use_parallel_query("查询客厅灯状态", self.registry))

    def test_subgraph_fanout_aggregates_sorted_results(self):
        graph = build_device_query_subgraph(self.registry)
        result = graph.invoke({
            "query": "查询设备",
            "targets": ["bedroom_ac", "living_room_light"],
            "parallel_results": [],
        })
        self.assertEqual([item["device_id"] for item in result["parallel_results"]], ["bedroom_ac", "living_room_light"])
        self.assertIn("卧室空调", result["response"])
        self.assertIn("客厅灯", result["response"])

    def test_main_graph_uses_parallel_query_subgraph_without_react(self):
        class UnusedLLM:
            def bind_tools(self, tools):
                return self

            def invoke(self, messages):
                raise AssertionError("parallel query should not invoke ReAct")

        with patch("src.agent.graph.build_llm", return_value=UnusedLLM()):
            graph = build_graph(self.registry, self.settings, self.directory)
            result = graph.invoke(
                {"messages": [HumanMessage(content="查询客厅和卧室的设备状态")], **self.context.to_state_input()},
                self.context.to_config(),
            )
        self.assertEqual(result["intent"], "device_query")
        self.assertEqual(result["intent_route"], "parallel_query")
        self.assertGreaterEqual(len(result["parallel_query_results"]), 2)
        self.assertIn("客厅灯", result["messages"][-1].content)


if __name__ == "__main__":
    unittest.main()
