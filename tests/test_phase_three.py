import itertools
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
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
        # 注意 fixture 里必须带上 tool_calls 的父 AIMessage：这个用例原本用
        # [Human, Tool, AI] + max_messages=2，窗口首条恰好是那条孤儿 ToolMessage
        # ——它无意间把「切断配对」的错误行为当成了预期（见 docs/gap-analysis.md 1.1）。
        # 现在窗口起点会对齐到合法边界，所以改用带父消息的拟真历史。
        messages = [
            HumanMessage(content="很久以前的闲聊", id="stale-1"),
            HumanMessage(content="查询", id="human-1"),
            AIMessage(
                content="",
                tool_calls=[{"name": "read_sensor", "args": {}, "id": "call-1"}],
                id="ai-call-1",
            ),
            ToolMessage(content="x" * 500, tool_call_id="call-1", id="tool-1"),
            AIMessage(content="完成", id="ai-1"),
        ]
        updates, summary, token_estimate = build_compaction_update(
            messages,
            max_messages=3,
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
        now = datetime(2026, 8, 2, tzinfo=UTC)

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
            {"v": 1, "ts": datetime.now(UTC).isoformat(), "id": "cp-1", "channel_values": {}, "channel_versions": {}, "versions_seen": {}, "updated_channels": []},
            {},
            {},
        )
        manager.end(context)
        self.assertIsNone(saver.get(checkpoint_config))

    def test_expired_long_term_memories_are_cleaned_globally(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = MemoryRepository(str(Path(temp_dir) / "memories.db"))
            expired = datetime.now(UTC) - timedelta(minutes=1)
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


class ToolCallPairingTests(unittest.TestCase):
    """docs/gap-analysis.md 1.1 的回归守卫：裁剪窗口不得切断工具调用配对。

    OpenAI 兼容协议要求 `tool` 角色的消息必须紧跟在带对应 `tool_calls` 的助手
    消息之后。窗口首条一旦是父消息已被切掉的 ToolMessage，整个请求体会被服务端
    400 拒绝；而 `RemoveMessage` 已经把父消息从 checkpoint 物理删除，补不回来。
    这里不断言返回文本，只断言协议不变量本身。
    """

    @staticmethod
    def _orphan_ids(window: list) -> list[str]:
        """返回窗口内「有结果但找不到调用」的 tool_call_id。"""
        provided: set[str] = set()
        orphans = []
        for message in window:
            for call in getattr(message, "tool_calls", None) or []:
                provided.add(call["id"])
            if isinstance(message, ToolMessage) and message.tool_call_id not in provided:
                orphans.append(message.tool_call_id)
        return orphans

    @classmethod
    def _conversation(cls, tool_counts) -> list:
        """按「每轮的工具调用次数」构造拟真混合历史。"""
        messages = []
        call_seq = 0
        for round_index, tool_count in enumerate(tool_counts):
            messages.append(HumanMessage(
                content=f"第{round_index}轮请求" * (round_index % 4 + 1),
                id=f"h{round_index}",
            ))
            if tool_count:
                calls = []
                for _ in range(tool_count):
                    call_seq += 1
                    calls.append({
                        "name": "control_light",
                        "args": {"device_name": f"灯{call_seq}", "action": "turn_on"},
                        "id": f"c{call_seq}",
                    })
                messages.append(AIMessage(content="", tool_calls=calls, id=f"a{round_index}"))
                for call in calls:
                    messages.append(ToolMessage(
                        content="已执行" * (call_seq % 7 + 1),
                        tool_call_id=call["id"],
                        id=f"t{call['id']}",
                    ))
            messages.append(AIMessage(content=f"第{round_index}轮完成", id=f"z{round_index}"))
        return messages

    def test_documented_orphan_case_is_fixed(self):
        """gap-analysis 1.1 里走查的那段 14 条历史：keep_from 原本正好落在 Tool c1 上。"""
        messages = self._conversation([1, 1, 2]) + [HumanMessage(content="现在几度", id="h9")]
        self.assertEqual(len(messages), 14)  # 与文档中的走查一致

        recent, _ = compact_messages(messages, max_messages=12, max_tokens=2400)

        self.assertNotIsInstance(recent[0], ToolMessage)
        self.assertEqual(self._orphan_ids(recent), [])

    def test_no_orphan_across_generated_corpus(self):
        """穷举 0/1/2 次工具调用的 5 轮组合，再对每个前缀长度各裁剪一次。"""
        checked = 0
        offenders = []
        for spec in itertools.product((0, 1, 2), repeat=5):
            full = self._conversation(spec)
            for cut in range(1, len(full) + 1):
                recent, _ = compact_messages(full[:cut], max_messages=12, max_tokens=2400)
                checked += 1
                if recent and (isinstance(recent[0], ToolMessage) or self._orphan_ids(recent)):
                    offenders.append((spec, cut, self._orphan_ids(recent)))
        self.assertGreater(checked, 1000)
        self.assertEqual(offenders, [], f"{len(offenders)}/{checked} 个裁剪结果仍有孤儿")

    def test_tiny_budget_retreats_to_include_the_parent_call(self):
        """极端预算下前进会清空窗口，此时必须后退带上父消息，宁可超预算也不发孤儿。"""
        messages = [
            HumanMessage(content="都关掉", id="h0"),
            AIMessage(content="", id="a0", tool_calls=[
                {"name": "control_light", "args": {}, "id": "c1"},
                {"name": "control_ac", "args": {}, "id": "c2"},
                {"name": "control_curtain", "args": {}, "id": "c3"},
            ]),
            ToolMessage(content="ok", tool_call_id="c1", id="t1"),
            ToolMessage(content="ok", tool_call_id="c2", id="t2"),
            ToolMessage(content="ok", tool_call_id="c3", id="t3"),
        ]

        recent, _ = compact_messages(messages, max_messages=2, max_tokens=10)

        self.assertIs(recent[0], messages[1])  # 后退到了父 AIMessage
        self.assertEqual(self._orphan_ids(recent), [])

    def test_estimate_counts_tool_call_payload(self):
        """带 tool_calls 的 AIMessage 其 content 为空，载荷不能被算成 1 token。"""
        empty = AIMessage(content="", id="a0")
        with_calls = AIMessage(content="", id="a1", tool_calls=[
            {"name": "control_light", "args": {"device_name": "客厅灯", "action": "turn_on"}, "id": "c1"},
        ])
        self.assertEqual(estimate_tokens([empty]), 1)
        self.assertGreater(estimate_tokens([with_calls]), estimate_tokens([empty]))


if __name__ == "__main__":
    unittest.main()
