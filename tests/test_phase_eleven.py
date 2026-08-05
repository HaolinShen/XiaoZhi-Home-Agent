import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from src.agent.context import AgentContext, SpaceDirectory
from src.agent.graph import build_graph
from src.agent.reasoning import reason_about_memories
from src.agent.time_travel import fork_from_checkpoint, list_state_history
from src.devices.base import DeviceRegistry
from src.devices.simulator import SimulatorBackend
from src.memory import MemoryScope, MemoryType, MemoryWrite
from src.tools import set_registry


class EchoLLM:
    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        return AIMessage(content="处理完成")


class PhaseElevenMemoryTimeTravelStreamingTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.registry = DeviceRegistry(SimulatorBackend())
        set_registry(self.registry)
        self.directory = SpaceDirectory.from_registry(self.registry, "home-a")
        self.context = AgentContext(
            home_id="home-a", user_id="user-a", session_id="phase-eleven", client_id="test"
        )
        self.settings = SimpleNamespace(
            memory=SimpleNamespace(
                enable_long_term=True,
                long_term_db_path=str(Path(self.temp_dir.name) / "memory.db"),
                db_path="", context_max_messages=12, context_max_tokens=2400,
                tool_result_max_chars=1200, summary_max_chars=1800, retrieval_top_k=6,
            ),
            planning=SimpleNamespace(enabled=False, max_steps=8, max_step_retries=1, max_replans=1),
            routing=SimpleNamespace(enabled=False, confidence_threshold=0.6),
            multi_agent=SimpleNamespace(enabled=False, max_handoffs=2),
        )
        self.graphs = []

    def tearDown(self):
        for graph in self.graphs:
            if graph.memory_repository is not None:
                graph.memory_repository.close()
        self.temp_dir.cleanup()

    def _build(self):
        with patch("src.agent.graph.build_llm", return_value=EchoLLM()):
            graph = build_graph(self.registry, self.settings, self.directory)
        self.graphs.append(graph)
        return graph

    def test_memory_reasoner_respects_explicit_temporary_override(self):
        records = [{
            "id": "m1", "memory_type": "preference", "memory_key": "ac.temperature",
            "memory_value": {"temperature": 25},
        }, {
            "id": "m2", "memory_type": "constraint", "memory_key": "quiet.hours",
            "memory_value": {"after": "23:00"},
        }]
        decision = reason_about_memories(records, "这次空调调到27度")
        self.assertIn("m1", decision.ignored_memory_ids)
        self.assertIn("m2", decision.applicable_memory_ids)
        self.assertEqual(len(decision.constraints), 1)

    def test_graph_stores_explicit_memory_decision(self):
        graph = self._build()
        record = graph.memory_service.save(self.context, MemoryWrite(
            scope=MemoryScope.USER, memory_type=MemoryType.PREFERENCE,
            memory_key="ac.temperature", memory_value={"temperature": 25},
        ))
        result = graph.invoke(
            {"messages": [HumanMessage(content="打开卧室空调")], **self.context.to_state_input()},
            self.context.to_config(),
        )
        self.assertIn(record.id, result["memory_decision"]["applicable_memory_ids"])
        self.assertIn("显式记忆决策", result["memory_context"])

    def test_checkpoint_history_can_be_inspected_and_forked(self):
        graph = self._build()
        config = self.context.to_config()
        graph.invoke(
            {"messages": [HumanMessage(content="你好")], **self.context.to_state_input()}, config
        )
        history = list_state_history(graph, config, limit=10)
        self.assertGreater(len(history), 1)
        checkpoint_id = history[-1]["checkpoint_id"]
        fork_config = fork_from_checkpoint(
            graph, config, checkpoint_id, {"conversation_summary": "实验分支摘要"}
        )
        snapshot = graph.get_state(fork_config)
        self.assertEqual(snapshot.values["conversation_summary"], "实验分支摘要")

    def test_custom_stream_exposes_progress_events(self):
        graph = self._build()
        events = list(graph.stream(
            {"messages": [HumanMessage(content="你好")], **self.context.to_state_input()},
            self.context.to_config(), stream_mode="custom",
        ))
        names = [event["event"] for event in events]
        self.assertIn("context_synced", names)
        self.assertIn("memory_reasoned", names)
        self.assertIn("supervisor_routing", names)
        self.assertIn("agent_completed", names)


if __name__ == "__main__":
    unittest.main()
