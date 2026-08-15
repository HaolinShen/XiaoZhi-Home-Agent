import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from src.agent.context import AgentContext, SpaceDirectory
from src.agent.graph import build_graph
from src.agent.routing import IntentResult, classify_intent, classify_intent_fallback
from src.devices.base import DeviceRegistry
from src.devices.simulator import SimulatorBackend
from src.tools import set_registry


class PhaseEightStructuredRoutingTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.registry = DeviceRegistry(SimulatorBackend())
        set_registry(self.registry)
        self.directory = SpaceDirectory.from_registry(self.registry, "home-a")
        self.context = AgentContext(
            home_id="home-a", user_id="user-a", session_id="routing-session", client_id="test"
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

    def test_fallback_classifies_six_intents(self):
        cases = {
            "客厅灯现在开着吗": "device_query",
            "打开客厅灯": "device_control",
            "开启睡眠模式": "scene_control",
            "记住我喜欢25度": "memory_management",
            "你好": "general_chat",
            "": "clarification",
        }
        for text, expected in cases.items():
            result = classify_intent_fallback(text)
            self.assertIsInstance(result, IntentResult)
            self.assertEqual(result.intent, expected, text)

    def test_low_information_request_returns_clarification_without_tools(self):
        class UnusedLLM:
            def bind_tools(self, tools):
                return self

            def invoke(self, messages):
                raise AssertionError("clarification branch must not invoke ReAct")

        with patch("src.agent.graph.build_llm", return_value=UnusedLLM()):
            graph = build_graph(self.registry, self.settings, self.directory)
        result = graph.invoke(
            {"messages": [HumanMessage(content=" ")], **self.context.to_state_input()},
            self.context.to_config(),
        )
        self.assertEqual(result["intent"], "clarification")
        self.assertEqual(result["intent_route"], "clarification")
        self.assertIn("补充具体", result["messages"][-1].content)

    def test_structured_router_result_is_used_when_enabled(self):
        class FakeStructured:
            def invoke(self, prompt):
                return IntentResult(intent="memory_management", confidence=0.97, reason="明确要求保存偏好")

        class FakeLLM:
            def bind_tools(self, tools):
                return self

            def with_structured_output(self, schema):
                return FakeStructured()

            def invoke(self, messages):
                return AIMessage(content="已识别为记忆管理请求。")

        self.settings.routing.enabled = True
        with patch("src.agent.graph.build_llm", return_value=FakeLLM()):
            graph = build_graph(self.registry, self.settings, self.directory)
            result = graph.invoke(
                {"messages": [HumanMessage(content="请记住这个偏好")], **self.context.to_state_input()},
                self.context.to_config(),
            )
        self.assertEqual(result["intent"], "memory_management")
        self.assertEqual(result["intent_confidence"], 0.97)
        self.assertEqual(result["intent_route"], "react")

    def test_future_home_preparation_cannot_be_misrouted_as_a_scene(self):
        class WrongStructuredRouter:
            def invoke(self, prompt):
                return IntentResult(
                    intent="scene_control", confidence=0.96, reason="看到了回家关键词"
                )

        class FakeLLM:
            def with_structured_output(self, schema):
                return WrongStructuredRouter()

        result = classify_intent(
            FakeLLM(),
            "我今天下午5点打球回到家，帮我提前准备洗澡水，同时提前打开客厅空调降温",
        )
        self.assertEqual(result.intent, "automation_management")
        self.assertIn("定时或事件自动化", result.reason)

    def test_low_confidence_structured_result_routes_to_clarification(self):
        class FakeStructured:
            def invoke(self, prompt):
                return IntentResult(intent="device_control", confidence=0.35, reason="缺少设备")

        class FakeLLM:
            def bind_tools(self, tools):
                return self

            def with_structured_output(self, schema):
                return FakeStructured()

            def invoke(self, messages):
                raise AssertionError("low-confidence route must not invoke ReAct")

        self.settings.routing.enabled = True
        with patch("src.agent.graph.build_llm", return_value=FakeLLM()):
            graph = build_graph(self.registry, self.settings, self.directory)
            result = graph.invoke(
                {"messages": [HumanMessage(content="帮我调一下")], **self.context.to_state_input()},
                self.context.to_config(),
            )
        self.assertEqual(result["intent"], "device_control")
        self.assertEqual(result["intent_route"], "clarification")
        self.assertIn("补充具体", result["messages"][-1].content)


if __name__ == "__main__":
    unittest.main()
