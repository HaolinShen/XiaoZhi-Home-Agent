import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import HumanMessage

from src.agent.context import AgentContext, SpaceDirectory
from src.agent.graph import build_graph
from src.devices.base import DeviceRegistry
from src.devices.simulator import SimulatorBackend
from src.evaluation import evaluate_rag_trajectory
from src.knowledge import KnowledgeBase, build_knowledge_rag_subgraph


class PhaseTwelveAgenticRAGTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.registry = DeviceRegistry(SimulatorBackend())
        self.directory = SpaceDirectory.from_registry(self.registry, "home-a")
        self.context = AgentContext(
            home_id="home-a", user_id="user-a", session_id="rag-session", client_id="test"
        )
        self.settings = SimpleNamespace(
            memory=SimpleNamespace(
                enable_long_term=False, long_term_db_path=str(Path(self.temp_dir.name) / "memory.db"),
                db_path="", context_max_messages=12, context_max_tokens=2400,
                tool_result_max_chars=1200, summary_max_chars=1800, retrieval_top_k=6,
            ),
            planning=SimpleNamespace(enabled=True, max_steps=8, max_step_retries=1, max_replans=1),
            routing=SimpleNamespace(enabled=False, confidence_threshold=0.6),
            multi_agent=SimpleNamespace(enabled=False, max_handoffs=2),
            rag=SimpleNamespace(enabled=True, knowledge_path="docs/knowledge", top_k=3, max_rewrites=1),
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_rag_answers_with_model_filtered_citation(self):
        kb = KnowledgeBase("docs/knowledge")
        graph = build_knowledge_rag_subgraph(kb)
        result = graph.invoke({"query": "客厅空调显示 E3 是什么意思"})
        self.assertEqual(result["rag_status"], "answered")
        self.assertIn("温度传感器", result["answer"])
        self.assertTrue(any("smartcool-ac2024-errors.md" in item for item in result["citations"]))
        self.assertEqual(result["trajectory"][0]["step"], "identify")

    def test_rag_refuses_unsupported_error_code(self):
        graph = build_knowledge_rag_subgraph(KnowledgeBase("docs/knowledge"))
        result = graph.invoke({"query": "空调显示 E9 是什么意思"})
        self.assertEqual(result["rag_status"], "refused")
        self.assertEqual(result["citations"], [])
        self.assertIn("不能可靠确认", result["answer"])

    def test_main_graph_routes_knowledge_query_to_rag(self):
        class UnusedLLM:
            def bind_tools(self, tools):
                return self

            def invoke(self, messages):
                raise AssertionError("knowledge query should not invoke ReAct")

        with patch("src.agent.graph.build_llm", return_value=UnusedLLM()):
            graph = build_graph(self.registry, self.settings, self.directory)
            result = graph.invoke(
                {"messages": [HumanMessage(content="客厅空调显示 E3 是什么意思")], **self.context.to_state_input()},
                self.context.to_config(),
            )
        self.assertEqual(result["intent"], "device_knowledge")
        self.assertEqual(result["intent_route"], "knowledge_rag")
        self.assertEqual(result["rag_status"], "answered")

    def test_trajectory_evaluator_reports_retrieval_and_citation_metrics(self):
        state = {
            "intent": "device_knowledge", "rag_status": "answered",
            "rag_citations": ["smartcool-ac2024-errors.md#E3"],
            "rag_trajectory": [{"step": "retrieve", "hit_count": 1}],
        }
        metrics = evaluate_rag_trajectory(
            state, expected_status="answered", expected_source="smartcool-ac2024-errors.md"
        )
        self.assertEqual(metrics["route_accuracy"], 1.0)
        self.assertEqual(metrics["source_accuracy"], 1.0)
        self.assertTrue(metrics["has_retrieval"])


if __name__ == "__main__":
    unittest.main()
