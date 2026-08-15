import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from src.agent.context import AgentContext, SpaceDirectory
from src.agent.graph import build_graph
from src.agent.multi_agent import agent_for_intent
from src.devices.base import DeviceRegistry
from src.devices.simulator import SimulatorBackend
from src.tools import set_registry


class BoundAgent:
    def __init__(self, owner, tool_names):
        self.owner = owner
        self.tool_names = set(tool_names)

    def invoke(self, messages):
        self.owner.invoked_tool_sets.append(self.tool_names)
        return AIMessage(content="专用 Agent 已完成处理。")


class MultiAgentFakeLLM:
    def __init__(self):
        self.invoked_tool_sets = []

    def bind_tools(self, tools):
        return BoundAgent(self, [tool.name for tool in tools])

    def invoke(self, messages):
        self.invoked_tool_sets.append(set())
        return AIMessage(content="Chat Agent 已完成处理。")


class PhaseTenMultiAgentTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.registry = DeviceRegistry(SimulatorBackend())
        set_registry(self.registry)
        self.directory = SpaceDirectory.from_registry(self.registry, "home-a")
        self.context = AgentContext(
            home_id="home-a", user_id="user-a", session_id="multi-agent-session", client_id="test"
        )
        self.settings = SimpleNamespace(
            memory=SimpleNamespace(
                enable_long_term=False, long_term_db_path=str(Path(self.temp_dir.name) / "memory.db"),
                db_path="", context_max_messages=12, context_max_tokens=2400,
                tool_result_max_chars=1200, summary_max_chars=1800, retrieval_top_k=6,
            ),
            planning=SimpleNamespace(enabled=True, max_steps=8, max_step_retries=1, max_replans=1),
            routing=SimpleNamespace(enabled=False, confidence_threshold=0.6),
            multi_agent=SimpleNamespace(enabled=True, max_handoffs=2),
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _invoke(self, text):
        fake = MultiAgentFakeLLM()
        with patch("src.agent.graph.build_llm", return_value=fake):
            graph = build_graph(self.registry, self.settings, self.directory)
            result = graph.invoke(
                {"messages": [HumanMessage(content=text)], **self.context.to_state_input()},
                self.context.to_config(),
            )
        return result, fake

    def test_supervisor_maps_intents_to_specialised_agents(self):
        self.assertEqual(agent_for_intent("device_control"), "device")
        self.assertEqual(agent_for_intent("scene_control"), "scene")
        self.assertEqual(agent_for_intent("memory_management"), "memory")
        self.assertEqual(agent_for_intent("general_chat"), "chat")

    def test_device_agent_only_receives_device_tools(self):
        result, fake = self._invoke("打开客厅灯")
        self.assertEqual(result["delegated_agent"], "device")
        self.assertEqual(result["collaboration_status"], "completed")
        self.assertEqual(result["handoff_count"], 1)
        selected = fake.invoked_tool_sets[-1]
        self.assertIn("control_light", selected)
        self.assertIn("get_device_status", selected)
        self.assertNotIn("activate_scene", selected)
        self.assertNotIn("save_personal_memory", selected)

    def test_memory_and_chat_agents_have_separate_capabilities(self):
        memory_result, memory_fake = self._invoke("记住我喜欢暖光")
        self.assertEqual(memory_result["delegated_agent"], "memory")
        self.assertIn("save_personal_memory", memory_fake.invoked_tool_sets[-1])
        self.assertNotIn("control_light", memory_fake.invoked_tool_sets[-1])

        chat_result, chat_fake = self._invoke("你好")
        self.assertEqual(chat_result["delegated_agent"], "chat")
        self.assertEqual(chat_fake.invoked_tool_sets[-1], set())

    def test_runtime_trace_exposes_delegation_and_completion(self):
        fake = MultiAgentFakeLLM()
        with patch("src.agent.graph.build_llm", return_value=fake):
            graph = build_graph(self.registry, self.settings, self.directory)

        events = list(graph.stream(
            {
                "messages": [HumanMessage(content="打开客厅灯")],
                **self.context.to_state_input(),
            },
            self.context.to_config(),
            stream_mode="custom",
        ))

        routing = next(event for event in events if event["event"] == "supervisor_routing")
        self.assertEqual(routing["intent"], "device_control")
        self.assertEqual(routing["intent_route"], "react")
        self.assertEqual(routing["delegated_agent"], "device")
        self.assertEqual(routing["handoff_count"], 1)

        agent = next(event for event in events if event["event"] == "agent_completed")
        self.assertEqual(agent["role"], "device")
        self.assertEqual(agent["tool_names"], [])

        finalized = next(event for event in events if event["event"] == "supervisor_finalized")
        self.assertEqual(finalized["role"], "device")
        self.assertEqual(finalized["status"], "completed")


if __name__ == "__main__":
    unittest.main()
