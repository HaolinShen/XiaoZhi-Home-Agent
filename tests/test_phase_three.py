import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from src.agent.context import AgentContext, SpaceDirectory
from src.agent.graph import build_graph
from src.agent.session import SessionManager
from src.devices.base import DeviceRegistry
from src.devices.simulator import SimulatorBackend
from src.memory.models import MemoryScope, MemoryType, MemoryWrite
from src.memory.repository import MemoryRepository
from src.memory.store import cleanup_expired_checkpoints
from src.memory.summarizer import build_compaction_update, compact_messages, estimate_tokens


class PhaseThreeTests(unittest.TestCase):
    def test_message_and_token_window_is_bounded_with_rolling_summary(self):
        messages = [HumanMessage(content=f"第{i}条" + "内容" * 30, id=f"m{i}") for i in range(8)]
        recent, summary = compact_messages(messages, max_messages=3, max_tokens=80)

        self.assertLessEqual(len(recent), 3)
        self.assertLessEqual(estimate_tokens(recent), 80)
        self.assertIn("第0条", summary)
        self.assertEqual(recent[-1].content, messages[-1].content)

    def test_tool_results_are_trimmed_and_checkpoint_update_removes_old_messages(self):
        messages = [
            HumanMessage(content="查询", id="human-1"),
            ToolMessage(content="x" * 500, tool_call_id="call-1", id="tool-1"),
            AIMessage(content="完成", id="ai-1"),
        ]
        updates, summary, token_estimate = build_compaction_update(
            messages,
            max_messages=2,
            max_tokens=1000,
            max_tool_result_chars=80,
        )

        self.assertIn("human: 查询", summary)
        self.assertTrue(any(getattr(item, "type", "") == "remove" for item in updates))
        trimmed = next(item for item in updates if getattr(item, "id", None) == "tool-1")
        self.assertLessEqual(len(trimmed.content), 80)
        self.assertIn("已裁剪", trimmed.content)
        self.assertLessEqual(token_estimate, 1000)

    def test_graph_persists_a_bounded_recent_window_and_context_statistics(self):
        captured_sizes = []

        class FakeLLM:
            def bind_tools(self, tools):
                return self

            def invoke(self, messages):
                captured_sizes.append(len(messages))
                return AIMessage(content="好的")

        memory = SimpleNamespace(
            enable_long_term=False,
            db_path="",
            context_max_messages=4,
            context_max_tokens=1000,
            tool_result_max_chars=100,
            summary_max_chars=300,
        )
        settings = SimpleNamespace(memory=memory)
        registry = DeviceRegistry(SimulatorBackend())
        directory = SpaceDirectory.from_registry(registry, "home-a")
        context = AgentContext(
            home_id="home-a", user_id="user-a", session_id="session-a", client_id="phone-a"
        )
        with patch("src.agent.graph.build_llm", return_value=FakeLLM()):
            graph = build_graph(registry, settings, directory)
            for i in range(8):
                graph.invoke(
                    {"messages": [HumanMessage(content=f"消息{i}")], **context.to_state_input()},
                    context.to_config(),
                )

        state = graph.get_state(context.to_config()).values
        self.assertLessEqual(len(state["messages"]), 5)
        self.assertTrue(state["conversation_summary"])
        self.assertLessEqual(state["context_message_count"], 4)
        self.assertLessEqual(state["context_token_estimate"], 1000)
        self.assertLessEqual(max(captured_sizes), 5)  # includes the system prompt

    def test_checkpoint_cleanup_deletes_only_expired_threads(self):
        now = datetime(2026, 8, 2, tzinfo=timezone.utc)

        class FakeCheckpointer:
            def __init__(self):
                self.deleted = []

            def list(self, config):
                def item(thread_id, timestamp):
                    return SimpleNamespace(
                        config={"configurable": {"thread_id": thread_id}},
                        checkpoint={"ts": timestamp.isoformat()},
                    )
                return iter([
                    item("fresh", now - timedelta(hours=2)),
                    item("fresh", now - timedelta(days=9)),
                    item("expired", now - timedelta(days=8)),
                ])

            def delete_thread(self, thread_id):
                self.deleted.append(thread_id)

        saver = FakeCheckpointer()
        deleted = cleanup_expired_checkpoints(saver, timedelta(days=7), now=now)
        self.assertEqual(deleted, 1)
        self.assertEqual(saver.deleted, ["expired"])

    def test_session_end_removes_memory_checkpoint(self):
        saver = MemorySaver()
        manager = SessionManager(SpaceDirectory({}, {}), saver)
        context = AgentContext(
            home_id="home-a",
            user_id="user-a",
            session_id="session-a",
            client_id="phone-a",
        )
        checkpoint_config = context.to_config()
        checkpoint_config["configurable"]["checkpoint_ns"] = ""
        saver.put(
            checkpoint_config,
            {"v": 1, "ts": datetime.now(timezone.utc).isoformat(), "id": "cp-1", "channel_values": {}, "channel_versions": {}, "versions_seen": {}, "updated_channels": []},
            {},
            {},
        )
        manager.end(context)
        self.assertIsNone(saver.get(checkpoint_config))

    def test_expired_long_term_memories_are_cleaned_globally(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = MemoryRepository(str(Path(temp_dir) / "memories.db"))
            expired = datetime.now(timezone.utc) - timedelta(minutes=1)
            for home_id in ("home-a", "home-b"):
                repository.upsert(
                    home_id,
                    "user-a",
                    MemoryWrite(
                        scope=MemoryScope.USER,
                        memory_type=MemoryType.PREFERENCE,
                        memory_key="temporary.preference",
                        memory_value={"value": home_id},
                        expires_at=expired,
                    ),
                )

            self.assertEqual(repository.cleanup_expired(), 2)
            statuses = repository.connection.execute(
                "SELECT DISTINCT status FROM memories"
            ).fetchall()
            self.assertEqual([row[0] for row in statuses], ["expired"])
            repository.close()


if __name__ == "__main__":
    unittest.main()
