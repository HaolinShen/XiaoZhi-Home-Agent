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
    DEVICE_ACTION_SPECS,
    PLANNING_TOOL_NAMES,
    TOOL_ACTIONS,
    ExecutionPlan,
    PlanStep,
    expected_state_for_step,
    should_use_planner,
)
from src.devices.base import DeviceRegistry
from src.devices.simulator import SimulatorBackend
from src.memory.store import close_checkpointer


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

    def _sample_device_name(self, device_type) -> str:
        """从模拟器里取一台该类型的真实设备名。

        故意不写死映射表：新增设备类型时这些一致性用例应该自动覆盖到，
        而不是又多一处需要手工同步的清单。
        """
        devices = self.registry.get_by_type(device_type)
        self.assertTrue(devices, f"模拟器里没有注册 {device_type} 设备")
        return next(iter(devices.values())).name

    def test_planning_literal_and_derived_views_match_the_action_specs(self):
        """PlanStep.tool_name 的 Literal 必须和 DEVICE_ACTION_SPECS 的键集相等。

        P0 改造后 Literal 用 Literal[tuple(PLANNING_TOOL_NAMES)] 从能力声明派生，
        不再手写；这个用例退化为"派生链路的最终防线"——能力声明、规划词表、
        Literal 三者的键集必须一致，漏任何一环都在测试阶段失败而不是运行期静默。
        """
        from typing import get_args

        literal_tools = set(get_args(PlanStep.model_fields["tool_name"].annotation))
        self.assertEqual(literal_tools, set(DEVICE_ACTION_SPECS))
        # 这两个是派生视图，顺手确认它们没有被手写覆盖过。
        self.assertEqual(set(PLANNING_TOOL_NAMES), set(DEVICE_ACTION_SPECS))
        self.assertEqual(set(TOOL_ACTIONS), set(DEVICE_ACTION_SPECS))

    def test_every_declared_action_passes_expected_state_resolution(self):
        """DEVICE_ACTION_SPECS 里声明的每个 action 都不能被判成 unsupported。

        喂给 Planner 的合法值列表就是从这份声明派生的，所以只要有一个对不上，
        Planner 照着做反而失败。
        """
        for tool_name, spec in DEVICE_ACTION_SPECS.items():
            device_name = self._sample_device_name(spec.device_type)
            for action in spec.actions:
                with self.subTest(tool=tool_name, action=action):
                    step = {
                        "tool_name": tool_name,
                        "arguments": {"device_name": device_name, "action": action},
                    }
                    _, expected, error = expected_state_for_step(step, self.registry)
                    self.assertIsNone(error, f"{tool_name}/{action} 被判成了错误：{error}")
                    self.assertTrue(expected, f"{tool_name}/{action} 没有给出任何期望状态")

    def test_action_specs_match_what_the_tools_actually_accept(self):
        """声明的 action 集合必须和工具实现双向一致。

        P0 改造后工具实现直接从同一份能力声明生成，理论上不会漂移；这个用例
        退化成运行时反射的双向钉住：声明里有的工具必须认，工具不认的声明里
        也不能有。以前只有"喂 Planner 的词表 ↔ expected_state_for_step"对得上，
        工具实现的 if/elif 副本漏改不会被任何用例发现。
        """
        from src.tools import build_device_tools

        tools_by_name = {
            tool.name: tool
            for tool in build_device_tools(self.registry, enable_preference_tracking=False)
        }
        rejection_marker = "不支持的操作"

        for tool_name, spec in DEVICE_ACTION_SPECS.items():
            tool = tools_by_name[tool_name]
            device_name = self._sample_device_name(spec.device_type)
            for action in spec.actions:
                with self.subTest(tool=tool_name, action=action):
                    result = str(tool.invoke({"device_name": device_name, "action": action}))
                    self.assertNotIn(
                        rejection_marker, result,
                        f"{tool_name} 声明支持 {action!r}，但实现拒绝了它：{result}",
                    )
            # 反向：其他平台的命名必须被实现拒绝，也不能悄悄出现在声明里。
            for foreign_action in ("turn_on", "turn_off", "toggle", "enable"):
                with self.subTest(tool=tool_name, action=foreign_action):
                    self.assertNotIn(foreign_action, spec.actions)
                    result = str(
                        tool.invoke({"device_name": device_name, "action": foreign_action})
                    )
                    self.assertIn(rejection_marker, result)

    def test_unparsable_numeric_argument_becomes_feedback_instead_of_crashing(self):
        """参数写成非数字时必须转成 preparation_error 回喂 Planner。

        expected_state_for_step 在 executor 的 try 块之外被调用，裸 int() 抛出的
        ValueError 会掀翻整张图；静默套用默认值又会把"调到很亮"执行成 50% 再报成功。
        """
        step = {
            "tool_name": "control_light",
            "arguments": {
                "device_name": self._sample_device_name(
                    DEVICE_ACTION_SPECS["control_light"].device_type
                ),
                "action": "set_brightness",
                "brightness": "很亮",
            },
        }
        device_id, expected, error = expected_state_for_step(step, self.registry)
        self.assertIsNotNone(device_id)
        self.assertEqual(expected, {})
        self.assertIn("invalid argument", error)
        self.assertIn("brightness", error)
        # 字符串形式的数字仍应照常接受，不能把容错做成一刀切的拒绝。
        step["arguments"]["brightness"] = "30"
        _, expected, error = expected_state_for_step(step, self.registry)
        self.assertIsNone(error)
        self.assertEqual(expected["brightness"], 30)

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
