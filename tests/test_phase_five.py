import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from src.agent.context import AgentContext, SpaceDirectory
from src.agent.graph import build_graph
from src.devices.base import DeviceRegistry
from src.devices.simulator import SimulatorBackend
from src.memory import (
    MemoryRepository,
    MemoryScope,
    MemoryService,
    MemoryType,
    MemoryWrite,
    extract_memory_candidates,
)


class PhaseFiveTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repository = MemoryRepository(str(Path(self.temp_dir.name) / "memory.db"))
        self.directory = SpaceDirectory.from_registry(DeviceRegistry(SimulatorBackend()), "home-a")
        self.service = MemoryService(self.repository, self.directory)
        self.context = AgentContext(
            home_id="home-a", user_id="user-a", session_id="session-a", client_id="phone"
        )

    def tearDown(self):
        self.repository.close()
        self.temp_dir.cleanup()

    def test_natural_language_extractor_is_conservative(self):
        candidates = extract_memory_candidates("我通常喜欢把空调设为25度")
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].memory_key, "ac.temperature")
        self.assertEqual(extract_memory_candidates("今天有点冷，空调调到25度"), [])

    def test_extracted_text_only_creates_pending_candidate(self):
        candidates = self.service.extract_candidates_from_text(
            self.context, "以后请把灯光调成暖光"
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].status, "pending")
        self.assertEqual(self.service.list_memories(self.context), [])

    def test_hybrid_retrieval_uses_top_k_and_tracks_access_count(self):
        for index, key in enumerate(("lighting.color", "ac.temperature", "tv.volume")):
            self.service.save(self.context, MemoryWrite(
                scope=MemoryScope.USER, memory_type=MemoryType.PREFERENCE,
                memory_key=key, memory_value={"value": index}, importance=0.5 + index * 0.1,
            ))
        records = self.service.retrieve(self.context, "空调温度", top_k=1)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].memory_key, "ac.temperature")
        refreshed = self.service.get(self.context, records[0].id)
        self.assertEqual(refreshed.access_count, 1)

    def test_updates_create_version_and_close_previous_validity(self):
        record = self.service.save(self.context, MemoryWrite(
            scope=MemoryScope.USER, memory_type=MemoryType.PREFERENCE,
            memory_key="lighting.color", memory_value={"color": "暖光"},
        ))
        updated = self.service.update(self.context, record.id, {"color": "冷白光"})
        self.assertEqual(updated.version, 2)
        versions = self.service.list_versions(self.context, record.id)
        self.assertEqual([version.version for version in versions], [1, 2])
        self.assertIsNotNone(versions[0].valid_to)
        self.assertIsNone(versions[1].valid_to)

    def test_graph_injects_only_relevant_top_k_memory_and_extracts_candidate(self):
        self.service.save(self.context, MemoryWrite(
            scope=MemoryScope.USER, memory_type=MemoryType.PREFERENCE,
            memory_key="ac.temperature", memory_value={"temperature": 25}, importance=0.9,
        ))
        self.service.save(self.context, MemoryWrite(
            scope=MemoryScope.USER, memory_type=MemoryType.PREFERENCE,
            memory_key="lighting.color", memory_value={"color": "暖光"}, importance=0.4,
        ))
        captured = []

        class FakeLLM:
            def bind_tools(self, tools):
                return self

            def invoke(self, messages):
                captured.extend(messages)
                return AIMessage(content="好的")

        settings = SimpleNamespace(memory=SimpleNamespace(
            enable_long_term=True, long_term_db_path=str(Path(self.temp_dir.name) / "memory.db"),
            db_path="", context_max_messages=12, context_max_tokens=2400,
            tool_result_max_chars=1200, summary_max_chars=1800, retrieval_top_k=1,
        ))
        registry = DeviceRegistry(SimulatorBackend())
        with patch("src.agent.graph.build_llm", return_value=FakeLLM()):
            graph = build_graph(registry, settings, self.directory)
            graph.invoke(
                {"messages": [HumanMessage(content="以后我通常把空调设为26度")], **self.context.to_state_input()},
                self.context.to_config(),
            )
        self.assertIn("ac.temperature", captured[0].content)
        self.assertNotIn("lighting.color", captured[0].content)
        self.assertEqual(len(graph.memory_service.list_candidates(self.context)), 1)
        graph.memory_repository.close()


if __name__ == "__main__":
    unittest.main()
