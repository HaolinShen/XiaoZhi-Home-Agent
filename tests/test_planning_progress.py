"""规划过程的运行时可见性测试。

这里守住两件事:
  1. 图在跑计划时，通过 ``stream_mode="custom"`` 真的把 Planner / Executor /
     Verifier 各阶段的事件发出来了，而且 plan_generated 在任何工具被调用之前
     就带上了完整的 步骤 + 工具名 + 参数 —— 这正是"规划与执行分开"的证据。
  2. 渲染层 ``PlanProgressView`` 能独立于图工作：喂它事件字典就出终端文本，
     不需要跑图，也不需要真的终端。
"""

import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import HumanMessage
from langgraph.types import Command
from rich.console import Console

from src.agent.context import AgentContext, SpaceDirectory
from src.agent.graph import build_graph
from src.agent.observability import PLANNING_EVENTS, TRACE_EVENTS
from src.agent.planning import ExecutionPlan, PlanStep
from src.devices.base import DeviceRegistry
from src.devices.simulator import SimulatorBackend
from src.progress_view import PlanProgressView, format_arguments, format_state
from src.tools import set_registry


class StructuredPlanner:
    def __init__(self, owner):
        self.owner = owner

    def invoke(self, prompt):
        index = min(self.owner.plan_index, len(self.owner.plans) - 1)
        plan = self.owner.plans[index]
        self.owner.plan_index += 1
        return plan


class PlanningFakeLLM:
    def __init__(self, plans):
        self.plans = plans
        self.plan_index = 0

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema):
        return StructuredPlanner(self)

    def invoke(self, messages):
        raise AssertionError("规划分支不应该走到 ReAct agent")


def good_plan(goal="关闭客厅灯并打开卧室空调"):
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
                arguments={"device_name": "卧室空调", "action": "on", "temperature": 25},
            ),
        ],
    )


def unreachable_plan(goal="调整两个设备"):
    """第一步指向不存在的设备，用来触发 重试 → 重新规划 这条路径。

    ExecutionPlan 至少要 2 步（少于 2 步就不该走规划分支），所以第二步随便给一个
    合法设备 —— 第一步就会卡住，它根本轮不到执行。
    """
    return ExecutionPlan(
        goal=goal,
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


class PlanningProgressStreamTests(unittest.TestCase):
    """事件是否真的从图里流出来（图侧，不依赖终端）。"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.registry = DeviceRegistry(SimulatorBackend())
        set_registry(self.registry)
        self.directory = SpaceDirectory.from_registry(self.registry, "home-a")
        self.context = AgentContext(
            home_id="home-a", user_id="user-a",
            session_id="progress-session", client_id="test",
        )
        self.settings = SimpleNamespace(
            memory=SimpleNamespace(
                enable_long_term=False,
                long_term_db_path=str(Path(self.temp_dir.name) / "memory.db"),
                db_path="", context_max_messages=12, context_max_tokens=2400,
                tool_result_max_chars=1200, summary_max_chars=1800, retrieval_top_k=6,
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
        return graph

    def _stream(self, graph, payload):
        """跑一段图，只收集 custom 事件（进度事件）。"""
        return list(graph.stream(payload, self.context.to_config(), stream_mode="custom"))

    def _start(self, graph, text="关闭客厅灯，然后打开卧室空调到25度"):
        return self._stream(graph, {
            "messages": [HumanMessage(content=text)],
            **self.context.to_state_input(),
        })

    def test_plan_is_fully_described_before_any_tool_runs(self):
        """plan_generated 出现在 step_started 之前，且已带齐工具名和参数。"""
        graph = self._build([good_plan()])
        self.registry.update("living_room_light", power=True)

        events = self._start(graph)
        names = [event["event"] for event in events]

        self.assertIn("planning_selected", names)
        self.assertIn("plan_generated", names)
        # 计划已经产出，但审批还没通过，所以一步都没执行、设备也没动。
        self.assertNotIn("step_started", names)
        self.assertTrue(self.registry.get("living_room_light").power)

        generated = events[names.index("plan_generated")]
        self.assertEqual(generated["revision"], 1)
        self.assertEqual(
            [(step["step_id"], step["tool_name"]) for step in generated["steps"]],
            [(1, "control_light"), (2, "control_ac")],
        )
        self.assertEqual(
            generated["steps"][0]["arguments"],
            {"device_name": "客厅灯", "action": "off"},
        )

    def test_approved_run_emits_executor_then_verifier_for_each_step(self):
        graph = self._build([good_plan()])
        self.registry.update("living_room_light", power=True)
        self._start(graph)

        events = self._stream(graph, Command(resume={"approved": True}))
        names = [event["event"] for event in events]

        self.assertEqual(names[0], "plan_decision")
        self.assertTrue(events[0]["approved"])
        # 每一步都是 先执行、后验证，顺序不能颠倒。
        self.assertEqual(
            [name for name in names if name.startswith("step_")],
            ["step_started", "step_executed", "step_verified"] * 2,
        )

        executed = [event for event in events if event["event"] == "step_executed"]
        verified = [event for event in events if event["event"] == "step_verified"]
        # Executor 只报告工具说了什么，不给结论。
        self.assertNotIn("success", executed[0])
        self.assertTrue(executed[0]["tool_result"])
        # 结论和"期望 vs 实测"的对比属于 Verifier。
        self.assertTrue(all(event["success"] for event in verified))
        self.assertEqual(verified[0]["expected_state"], {"power": False})
        self.assertEqual(verified[0]["actual_state"], {"power": False})

        finished = events[-1]
        self.assertEqual(finished["event"], "planning_finished")
        self.assertEqual(finished["status"], "completed")
        self.assertEqual(finished["succeeded"], 2)
        self.assertEqual(finished["plan_revision"], 1)

    def test_deterministic_failure_skips_retry_and_replans(self):
        """device_not_found 是确定性错误：不重试，一次失败就直接重新规划。"""
        graph = self._build([unreachable_plan(), good_plan("调整两个设备")])
        self.registry.update("living_room_light", power=True)
        self._start(graph, "关闭书房灯，然后打开卧室空调")

        events = self._stream(graph, Command(resume={"approved": True}))
        names = [event["event"] for event in events]

        failed = [
            event for event in events
            if event["event"] == "step_verified" and not event["success"]
        ]
        # 只失败一次：原样重放同一批参数不可能成功，没必要浪费重试额度。
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["problem_type"], "device_not_found")
        # 因此没有 step_retry，直接走 replan。
        self.assertNotIn("step_retry", names)

        replan = events[names.index("replan_requested")]
        self.assertTrue(replan["accepted"])
        self.assertEqual(replan["replan_count"], 1)
        # 重新规划后又产出了 v2，并再次停下来等审批。
        self.assertEqual(names[-1], "plan_generated")
        self.assertEqual(events[-1]["revision"], 2)

    def test_transient_failure_emits_retry_then_replan_events(self):
        """state_mismatch 可能是瞬时的（超时、状态竞态），仍然要重试后再重新规划。"""
        graph = self._build([good_plan(), good_plan("客厅灯与卧室空调")])
        self.registry.update("living_room_light", power=True)
        self._start(graph)

        original_update = self.registry.update

        def always_fail_first_step(device_id, **kwargs):
            # 第一步（关客厅灯）始终"执行成功但状态没变"，制造 state_mismatch；
            # 这类失败被当作瞬时错误，先重试，重试也不过才走重新规划。
            if device_id == "living_room_light":
                return False
            return original_update(device_id, **kwargs)

        with patch.object(
            self.registry, "update", side_effect=always_fail_first_step
        ):
            events = self._stream(graph, Command(resume={"approved": True}))
        names = [event["event"] for event in events]

        failed = [
            event for event in events
            if event["event"] == "step_verified" and not event["success"]
        ]
        self.assertEqual(len(failed), 2)  # 首次 + 重试各失败一次
        self.assertEqual(failed[0]["problem_type"], "state_mismatch")

        retry = events[names.index("step_retry")]
        self.assertEqual(retry["attempt"], 2)
        self.assertEqual(retry["max_attempts"], 2)

        replan = events[names.index("replan_requested")]
        self.assertTrue(replan["accepted"])
        self.assertEqual(replan["replan_count"], 1)
        # 重新规划后又产出了 v2，并再次停下来等审批。
        self.assertEqual(names[-1], "plan_generated")
        self.assertEqual(events[-1]["revision"], 2)

    def test_rejected_plan_reports_cancelled_without_executing(self):
        graph = self._build([good_plan()])
        self.registry.update("living_room_light", power=True)
        self._start(graph)

        events = self._stream(graph, Command(resume={"approved": False}))
        names = [event["event"] for event in events]

        self.assertFalse(events[names.index("plan_decision")]["approved"])
        self.assertNotIn("step_started", names)
        self.assertEqual(events[-1]["status"], "cancelled")
        self.assertTrue(self.registry.get("living_room_light").power)


class PlanProgressViewRenderTests(unittest.TestCase):
    """渲染层能不能独立工作（终端侧，不跑图）。"""

    def _render(self, events, show_trace=False):
        buffer = io.StringIO()
        # width 固定，避免不同终端宽度把表格换行位置改掉。
        console = Console(file=buffer, width=100, no_color=True, force_terminal=False)
        view = PlanProgressView(console, show_trace=show_trace)
        for event in events:
            view.handle(event)
        return buffer.getvalue(), view

    def test_plan_table_shows_tool_and_arguments_before_execution(self):
        text, view = self._render([
            {"event": "planning_selected", "goal": "关灯并开空调", "reason": "多动作请求"},
            {
                "event": "plan_generated", "revision": 1, "step_count": 1,
                "goal": "关灯并开空调", "rationale": "按顺序执行",
                "steps": [{
                    "step_id": 1, "description": "关闭客厅灯",
                    "tool_name": "control_light",
                    "arguments": {"device_name": "客厅灯", "action": "off"},
                }],
            },
        ])
        self.assertTrue(view.planning_seen)
        self.assertIn("Planner", text)
        self.assertIn("尚未触碰任何设备", text)
        self.assertIn("control_light", text)
        self.assertIn("客厅灯", text)
        self.assertIn("规划理由", text)

    def test_verifier_shows_expected_versus_actual_on_both_outcomes(self):
        passed, _ = self._render([{
            "event": "step_verified", "success": True,
            "step_id": 1, "step_index": 1, "step_total": 2,
            "expected_state": {"power": False}, "actual_state": {"power": False},
        }])
        self.assertIn("Verifier", passed)
        self.assertIn("通过", passed)
        self.assertIn("power=False", passed)

        failed, _ = self._render([{
            "event": "step_verified", "success": False,
            "step_id": 1, "step_index": 1, "step_total": 2,
            "problem_type": "device_not_found", "reason": "找不到书房灯",
            "expected_state": {}, "actual_state": {},
        }])
        self.assertIn("未通过", failed)
        self.assertIn("device_not_found", failed)
        self.assertIn("找不到书房灯", failed)

    def test_trace_events_are_hidden_unless_requested(self):
        trace_event = {"event": "supervisor_routing", "request": "打开客厅灯"}
        hidden, view = self._render([trace_event])
        self.assertEqual(hidden.strip(), "")
        # 诊断事件不算"走过规划分支"，否则 CLI 会多打一条无意义的分隔。
        self.assertFalse(view.planning_seen)

        shown, _ = self._render([trace_event], show_trace=True)
        self.assertIn("supervisor_routing", shown)

    def test_unknown_events_are_ignored(self):
        """图以后新增事件时，旧版 CLI 不应该崩。"""
        text, view = self._render([{"event": "some_future_event", "x": 1}, {}])
        self.assertEqual(text.strip(), "")
        self.assertFalse(view.planning_seen)

    def test_every_declared_planning_event_has_a_renderer(self):
        view = PlanProgressView(Console(file=io.StringIO()))
        for name in PLANNING_EVENTS:
            self.assertTrue(
                hasattr(view, f"_on_{name}"),
                f"PLANNING_EVENTS 里声明了 {name}，但渲染层没有对应的 _on_{name}",
            )
        self.assertFalse(set(PLANNING_EVENTS) & set(TRACE_EVENTS))

    def test_formatters_handle_empty_and_overlong_values(self):
        self.assertEqual(format_arguments(None), "—")
        self.assertEqual(format_state({}), "—")
        self.assertEqual(format_state({"power": True, "brightness": 60}),
                         "power=True brightness=60")
        long = format_arguments({"note": "很长的说明" * 20})
        self.assertLessEqual(len(long), 6 + 40)
        self.assertTrue(long.endswith("…"))


if __name__ == "__main__":
    unittest.main()



