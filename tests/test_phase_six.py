import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from src.agent.context import AgentContext, SpaceDirectory
from src.agent.graph import build_graph
from src.devices.base import DeviceRegistry
from src.devices.simulator import SimulatorBackend
from src.memory.store import close_checkpointer


class ApprovalAwareFakeLLM:
    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        tool_results = [message for message in messages if isinstance(message, ToolMessage)]
        if tool_results:
            if any("未批准" in str(message.content) for message in tool_results):
                return AIMessage(content="已取消操作，设备状态没有变化。")
            return AIMessage(content="离家模式已经执行完成。")
        return AIMessage(
            content="",
            tool_calls=[{
                "name": "activate_scene",
                "args": {"scene_name": "离家模式"},
                "id": "scene-call-1",
                "type": "tool_call",
            }],
        )


class PhaseSixHumanInTheLoopTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.registry = DeviceRegistry(SimulatorBackend())
        self.directory = SpaceDirectory.from_registry(self.registry, "home-a")
        self.context = AgentContext(
            home_id="home-a",
            user_id="user-a",
            session_id="approval-session",
            client_id="test",
        )
        self.settings = SimpleNamespace(memory=SimpleNamespace(
            enable_long_term=False,
            long_term_db_path=str(Path(self.temp_dir.name) / "memory.db"),
            db_path="",
            context_max_messages=12,
            context_max_tokens=2400,
            tool_result_max_chars=1200,
            summary_max_chars=1800,
            retrieval_top_k=6,
        ))

    def tearDown(self):
        self.temp_dir.cleanup()

    def _build_graph(self):
        with patch("src.agent.graph.build_llm", return_value=ApprovalAwareFakeLLM()):
            return build_graph(self.registry, self.settings, self.directory)

    def _start(self, graph):
        return graph.invoke(
            {"messages": [HumanMessage(content="我要出门了")], **self.context.to_state_input()},
            self.context.to_config(),
        )

    def test_scene_waits_for_approval_before_changing_devices(self):
        graph = self._build_graph()
        self.registry.update("living_room_light", power=True)
        self.registry.update("living_room_tv", power=True)

        interrupted = self._start(graph)

        self.assertIn("__interrupt__", interrupted)
        payload = interrupted["__interrupt__"][0].value
        self.assertEqual(payload["kind"], "tool_approval")
        self.assertEqual(payload["risk_level"], "medium")
        self.assertIn("离家模式", payload["summary"])
        self.assertTrue(self.registry.get("living_room_light").power)
        self.assertTrue(self.registry.get("living_room_tv").power)

        completed = graph.invoke(
            Command(resume={"approved": True}),
            self.context.to_config(),
        )

        self.assertFalse(self.registry.get("living_room_light").power)
        self.assertFalse(self.registry.get("living_room_tv").power)
        self.assertIn("执行完成", completed["messages"][-1].content)

    def test_rejection_closes_tool_call_without_changing_devices(self):
        graph = self._build_graph()
        self.registry.update("living_room_light", power=True)
        self.registry.update("living_room_tv", power=True)
        self._start(graph)

        completed = graph.invoke(
            Command(resume={"approved": False}),
            self.context.to_config(),
        )

        self.assertTrue(self.registry.get("living_room_light").power)
        self.assertTrue(self.registry.get("living_room_tv").power)
        self.assertIn("取消", completed["messages"][-1].content)
        rejection_results = [
            message for message in completed["messages"]
            if isinstance(message, ToolMessage) and "未批准" in str(message.content)
        ]
        self.assertEqual(len(rejection_results), 1)

    def test_single_device_control_does_not_require_approval(self):
        class SingleDeviceLLM:
            def bind_tools(self, tools):
                return self

            def invoke(self, messages):
                if any(isinstance(message, ToolMessage) for message in messages):
                    return AIMessage(content="客厅灯已经关闭。")
                return AIMessage(content="", tool_calls=[{
                    "name": "control_light",
                    "args": {"device_name": "客厅灯", "action": "off"},
                    "id": "light-call-1",
                    "type": "tool_call",
                }])

        self.registry.update("living_room_light", power=True)
        with patch("src.agent.graph.build_llm", return_value=SingleDeviceLLM()):
            graph = build_graph(self.registry, self.settings, self.directory)

        result = graph.invoke(
            {"messages": [HumanMessage(content="关闭客厅灯")], **self.context.to_state_input()},
            self.context.to_config(),
        )

        self.assertNotIn("__interrupt__", result)
        self.assertFalse(self.registry.get("living_room_light").power)

    def test_sqlite_checkpoint_resumes_after_graph_is_rebuilt(self):
        checkpoint_path = str(Path(self.temp_dir.name) / "approval-checkpoints.db")
        self.settings.memory.db_path = checkpoint_path
        self.registry.update("living_room_light", power=True)

        first_graph = self._build_graph()
        interrupted = self._start(first_graph)
        self.assertIn("__interrupt__", interrupted)
        self.assertTrue(self.registry.get("living_room_light").power)
        close_checkpointer(first_graph.checkpointer)

        second_graph = self._build_graph()
        completed = second_graph.invoke(
            Command(resume={"approved": True}),
            self.context.to_config(),
        )

        self.assertFalse(self.registry.get("living_room_light").power)
        self.assertIn("执行完成", completed["messages"][-1].content)
        close_checkpointer(second_graph.checkpointer)


if __name__ == "__main__":
    unittest.main()
