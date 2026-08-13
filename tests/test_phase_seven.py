import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from src.agent.context import AgentContext, SpaceDirectory
from src.agent.graph import build_graph
from src.agent.planning import (
    TOOL_ACTIONS,
    ExecutionPlan,
    PlanStep,
    expected_state_for_step,
    should_use_planner,
)
from src.devices.base import DeviceRegistry
from src.devices.simulator import SimulatorBackend
from src.memory.store import close_checkpointer
from src.tools import set_registry


class StructuredPlanner:
    def __init__(self, owner):
        self.owner = owner

    def invoke(self, prompt):
        self.owner.planner_prompts.append(str(prompt))
        index = min(self.owner.plan_index, len(self.owner.plans) - 1)
        plan = self.owner.plans[index]
        self.owner.plan_index += 1
        return plan


class PlanningFakeLLM:
    def __init__(self, plans):
        self.plans = plans
        self.plan_index = 0
        self.planner_prompts = []

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema):
        self.schema = schema
        return StructuredPlanner(self)

    def invoke(self, messages):
        raise AssertionError("complex planning branch should not invoke the ReAct agent")


def plan(goal="关闭客厅灯并打开卧室空调"):
    return ExecutionPlan(
        goal=goal,
        rationale="按用户给出的顺序执行",
        steps=[
            PlanStep(
                step_id=1,
                description="关闭客厅灯",
                tool_name="control_light",
                arguments={"device_name": "客厅灯", "action": "off"},
            ),
            PlanStep(
                step_id=2,
                description="打开卧室空调并设置 25 度",
                tool_name="control_ac",
                arguments={
                    "device_name": "卧室空调",
                    "action": "on",
                    "temperature": 25,
                    "mode": "cool",
                },
            ),
        ],
    )


class PhaseSevenPlannerExecutorVerifierTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.registry = DeviceRegistry(SimulatorBackend())
        set_registry(self.registry)
        self.directory = SpaceDirectory.from_registry(self.registry, "home-a")
        self.context = AgentContext(
            home_id="home-a",
            user_id="user-a",
            session_id="planning-session",
            client_id="test",
        )
        self.settings = SimpleNamespace(
            memory=SimpleNamespace(
                enable_long_term=False,
                long_term_db_path=str(Path(self.temp_dir.name) / "memory.db"),
                db_path="",
                context_max_messages=12,
                context_max_tokens=2400,
                tool_result_max_chars=1200,
                summary_max_chars=1800,
                retrieval_top_k=6,
            ),
            planning=SimpleNamespace(
                enabled=True, max_steps=8, max_step_retries=1, max_replans=1
            ),
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _build(self, plans):
        fake = PlanningFakeLLM(plans)
        with patch("src.agent.graph.build_llm", return_value=fake):
            graph = build_graph(self.registry, self.settings, self.directory)
        return graph, fake

    def _start(self, graph, text="关闭客厅灯，然后打开卧室空调到25度"):
        return graph.invoke(
            {"messages": [HumanMessage(content=text)], **self.context.to_state_input()},
            self.context.to_config(),
        )

    def test_router_only_selects_explicit_custom_multi_action_requests(self):
        self.assertTrue(should_use_planner("关闭客厅灯，然后打开卧室空调到25度"))
        self.assertFalse(should_use_planner("关闭客厅灯"))
        self.assertFalse(should_use_planner("我要出门了"))
        self.assertFalse(should_use_planner("开启离家模式"))

    def test_tool_actions_stay_in_sync_with_expected_state_for_step(self):
        """TOOL_ACTIONS 词表（喂给 Planner）必须和 expected_state_for_step 的
        if/elif（执行前校验）逐一对上：文档里写的每个合法 action 都不能被判成
        unsupported，否则 Planner 照做反而失败。三处枚举靠手写同步，用这条钉住。
        """
        import re

        # 每个工具挑一个真实存在的设备名，让 device 能被解析（否则会先撞 device_not_found）。
        device_for_tool = {
            "control_light": "客厅灯",
            "control_ac": "客厅空调",
            "control_tv": "客厅电视",
            "control_curtain": "客厅窗帘",
            "control_humidifier": "客厅加湿器",
        }
        for tool_name, spec in TOOL_ACTIONS.items():
            # spec 形如 "on / off / set_temp(temperature) / ..."；按 " / " 切段，
            # 每段去掉括号里的参数说明，剩下的裸词就是合法 action 名。
            for segment in spec.split("/"):
                action = re.sub(r"\(.*?\)", "", segment).strip()
                self.assertTrue(action, f"{tool_name} 的 spec 段为空：{segment!r}")
                step = {
                    "tool_name": tool_name,
                    "arguments": {"device_name": device_for_tool[tool_name], "action": action},
                }
                _, _, error = expected_state_for_step(step, self.registry)
                self.assertIsNone(
                    error,
                    f"{tool_name} 的合法 action {action!r} 被判成了错误：{error}",
                )

    def test_invalid_action_is_rejected_with_valid_choices_listed(self):
        """turn_off 这类其他平台命名必须被判 unsupported，且错误信息带上合法值列表，
        这样重新规划时 Planner 能直接照着改，而不是再猜一轮。
        """
        step = {
            "tool_name": "control_light",
            "arguments": {"device_name": "客厅灯", "action": "turn_off"},
        }
        _, _, error = expected_state_for_step(step, self.registry)
        self.assertIsNotNone(error)
        self.assertIn("unsupported action", error)
        self.assertIn("turn_off", error)
        # 合法值列表要出现在反馈里，供 Planner 参考。
        self.assertIn("on / off", error)

    def test_approved_plan_executes_each_step_and_verifies_actual_state(self):
        graph, _ = self._build([plan()])
        self.registry.update("living_room_light", power=True)

        interrupted = self._start(graph)

        payload = interrupted["__interrupt__"][0].value
        self.assertEqual(payload["kind"], "plan_approval")
        self.assertEqual(len(payload["plan"]["steps"]), 2)
        self.assertTrue(self.registry.get("living_room_light").power)

        completed = graph.invoke(
            Command(resume={"approved": True}), self.context.to_config()
        )

        self.assertFalse(self.registry.get("living_room_light").power)
        bedroom_ac = self.registry.get("bedroom_ac")
        self.assertTrue(bedroom_ac.power)
        self.assertEqual(bedroom_ac.temperature, 25)
        self.assertIn("任务已完成", completed["messages"][-1].content)
        self.assertEqual(completed["planning_status"], "completed")
        self.assertEqual(len(completed["planning_results"]), 2)
        self.assertTrue(all(
            result["verification"]["success"]
            for result in completed["planning_results"]
        ))

    def test_rejected_plan_does_not_execute_any_step(self):
        graph, _ = self._build([plan()])
        self.registry.update("living_room_light", power=True)
        self._start(graph)

        completed = graph.invoke(
            Command(resume={"approved": False}), self.context.to_config()
        )

        self.assertTrue(self.registry.get("living_room_light").power)
        self.assertFalse(self.registry.get("bedroom_ac").power)
        self.assertEqual(completed["planning_status"], "cancelled")
        self.assertIn("取消", completed["messages"][-1].content)

    def test_failed_step_retries_then_succeeds(self):
        graph, _ = self._build([plan()])
        self.registry.update("living_room_light", power=True)
        original_update = self.registry.update
        attempts = {"living_room_light": 0}

        def flaky_update(device_id, **kwargs):
            if device_id == "living_room_light" and attempts[device_id] == 0:
                attempts[device_id] += 1
                return False
            return original_update(device_id, **kwargs)

        with patch.object(self.registry, "update", side_effect=flaky_update):
            self._start(graph)
            completed = graph.invoke(
                Command(resume={"approved": True}), self.context.to_config()
            )

        first_step_attempts = [
            item for item in completed["planning_results"] if item["step_id"] == 1
        ]
        self.assertEqual(len(first_step_attempts), 2)
        self.assertFalse(first_step_attempts[0]["verification"]["success"])
        self.assertTrue(first_step_attempts[1]["verification"]["success"])
        self.assertEqual(completed["planning_status"], "completed")

    def test_exhausted_retries_triggers_replan_and_second_approval(self):
        bad_plan = ExecutionPlan(
            goal="调整两个设备",
            steps=[
                PlanStep(
                    step_id=1,
                    description="关闭不存在的书房灯",
                    tool_name="control_light",
                    arguments={"device_name": "书房灯", "action": "off"},
                ),
                PlanStep(
                    step_id=2,
                    description="打开卧室空调",
                    tool_name="control_ac",
                    arguments={"device_name": "卧室空调", "action": "on"},
                ),
            ],
        )
        graph, fake = self._build([bad_plan, plan("调整两个设备")])
        first_interrupt = self._start(graph, "关闭客厅灯，然后打开卧室空调")
        self.assertEqual(first_interrupt["__interrupt__"][0].value["kind"], "plan_approval")

        second_interrupt = graph.invoke(
            Command(resume={"approved": True}), self.context.to_config()
        )

        self.assertEqual(second_interrupt["__interrupt__"][0].value["kind"], "plan_approval")
        self.assertEqual(fake.plan_index, 2)
        self.assertIn("device not found", fake.planner_prompts[-1])

        completed = graph.invoke(
            Command(resume={"approved": True}), self.context.to_config()
        )
        self.assertEqual(completed["planning_status"], "completed")
        self.assertEqual(completed["replan_count"], 1)
        self.assertEqual(completed["plan_revision"], 2)

    def test_sqlite_checkpoint_resumes_approved_plan_after_graph_rebuild(self):
        self.settings.memory.db_path = str(
            Path(self.temp_dir.name) / "planning-checkpoints.db"
        )
        first_graph, _ = self._build([plan()])
        interrupted = self._start(first_graph)
        self.assertEqual(
            interrupted["__interrupt__"][0].value["kind"], "plan_approval"
        )
        close_checkpointer(first_graph.checkpointer)

        second_graph, _ = self._build([plan()])
        completed = second_graph.invoke(
            Command(resume={"approved": True}), self.context.to_config()
        )

        self.assertEqual(completed["planning_status"], "completed")
        self.assertFalse(self.registry.get("living_room_light").power)
        self.assertTrue(self.registry.get("bedroom_ac").power)
        close_checkpointer(second_graph.checkpointer)

    def test_task_stops_after_retry_and_replan_limits_are_exhausted(self):
        self.settings.planning.max_replans = 0
        impossible = ExecutionPlan(
            goal="控制不存在的设备",
            steps=[
                PlanStep(
                    step_id=1,
                    description="关闭书房灯",
                    tool_name="control_light",
                    arguments={"device_name": "书房灯", "action": "off"},
                ),
                PlanStep(
                    step_id=2,
                    description="打开卧室空调",
                    tool_name="control_ac",
                    arguments={"device_name": "卧室空调", "action": "on"},
                ),
            ],
        )
        graph, fake = self._build([impossible])
        self._start(graph, "关闭书房灯，然后打开卧室空调")

        completed = graph.invoke(
            Command(resume={"approved": True}), self.context.to_config()
        )

        self.assertEqual(completed["planning_status"], "failed")
        self.assertEqual(completed["replan_count"], 1)
        self.assertEqual(fake.plan_index, 1)
        # device_not_found 是确定性错误：不再浪费重试额度，第一次失败就直接走 replan，
        # 因此只记录一次尝试（而不是重试后的两次）。
        self.assertEqual(len(completed["planning_results"]), 1)
        self.assertIn("未能完成", completed["messages"][-1].content)


if __name__ == "__main__":
    unittest.main()
