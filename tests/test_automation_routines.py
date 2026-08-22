import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.types import Command

from src.agent.approval import build_approval_request
from src.agent.context import AgentContext, SpaceDirectory
from src.agent.graph import build_graph, _required_automation_tool
from src.agent.multi_agent import agent_for_intent
from src.agent.routing import classify_intent_fallback
from src.automation.executor import RoutineExecutor
from src.automation.routines import build_arrival_routine, build_wake_routine
from src.automation.runtime import AutomationRuntime
from src.automation.planning import ScheduledActionInput
from src.automation.scheduler import RoutineScheduler
from src.automation.speaker import SimulatorSpeakerBackend
from src.automation.store import AutomationStore
from src.automation.vehicle import ArrivalOrchestrator, VehicleEvent, VehicleSimulator
from src.devices.base import DeviceRegistry
from src.devices.simulator import SimulatorBackend
from src.memory import MemoryRepository, MemoryService
from src.tools import build_automation_tools, build_device_tools


UTC = timezone.utc


class AutomationRoutineTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        # 目录删除必须排在所有 SQLite 连接关闭之后，否则 Windows 会抛
        # PermissionError: [WinError 32]，测试断言全过也判失败。
        # unittest 先跑完整个 tearDown 才轮到 doCleanups，所以放在 tearDown 里
        # 会早于测试方法内 addCleanup 注册的 close；而 doCleanups 是 LIFO，
        # 在 setUp 最先注册就能保证它最后执行。
        self.addCleanup(self.temp_dir.cleanup)
        self.registry = DeviceRegistry(SimulatorBackend())
        self.store = AutomationStore(str(Path(self.temp_dir.name) / "automation.db"))
        self.speaker = SimulatorSpeakerBackend()
        self.events = []
        self.scheduler = RoutineScheduler(
            self.store,
            RoutineExecutor(self.registry, self.speaker),
            event_sink=self.events.append,
        )

    def tearDown(self):
        self.scheduler.stop()
        self.store.close()

    @staticmethod
    def automation_tools(runtime):
        """P1: 自动化工具由工厂显式构建（闭包持有 runtime，无模块级单例）。"""
        return {tool.name: tool for tool in build_automation_tools(runtime)}

    def test_wake_routine_arms_alarm_and_executes_relative_actions(self):
        wake_at = datetime(2026, 8, 16, 6, 0, tzinfo=UTC)
        created_at = wake_at - timedelta(hours=8)
        routine = self.store.save_routine(build_wake_routine("home-a", "user-a"))
        self.scheduler.schedule(
            routine, anchor_at=wake_at, trigger_key="wake-1", now=created_at
        )

        self.scheduler.tick(created_at)
        alarms = self.speaker.list_alarms()
        self.assertEqual(len(alarms), 1)
        self.assertEqual(alarms[0]["alarm_at"], wake_at)

        self.scheduler.tick(wake_at - timedelta(minutes=30))
        heater = self.registry.get("bathroom_water_heater")
        self.assertTrue(heater.power)
        self.assertEqual(heater.target_temp, 45)

        self.scheduler.tick(wake_at - timedelta(minutes=10))
        kettle = self.registry.get("kitchen_kettle")
        self.assertTrue(kettle.power)
        self.assertEqual(kettle.target_temp, 80)

        self.scheduler.tick(wake_at)
        self.assertEqual(self.registry.get("bedroom_curtain").position, 100)
        self.assertTrue(self.registry.get("bedroom_light").power)
        self.assertTrue(all(task.status == "completed" for task in self.store.list_tasks(routine.id)))

    def test_routine_actions_survive_missing_identity_and_skip_preference_learning(self):
        """长期记忆开启时，后台执行的例程动作仍须成功。

        回归 `KeyError: 'home_id'`：执行器走 `tool.invoke(arguments)`，不携带可信身份，
        但 LangChain 仍会注入一个 `configurable` 为空的 config。当时 devices.py 每个
        写偏好的分支都有一道 `if config is not None` 守卫，看着像在防这种无身份调用，
        实际恒为真、一个字都拦不住，`_context()` 的下标访问照样抛错，把热水器、烧水壶
        和灯光全判成 failed。窗帘的 `open` 分支不记录偏好，所以当时唯独它成功 ——
        这个"只有一个动作活下来"的形态就是本 bug 的指纹。

        P1 根因修复后：执行器在**构造期**显式关闭偏好观察
        （build_device_tools(..., enable_preference_tracking=False)），
        "无身份"不再是调用期靠逐键检查去猜的情况；而图路径的工具开启偏好观察，
        缺身份会直接 raise 而不是静默吞掉。本用例钉住两侧的行为。

        上面那个起床测试没能抓到，是因为它从不给设备工具注入记忆服务，
        记录器在构造期就是 no-op，绕过了出错的那一行。
        """
        repository = MemoryRepository(str(Path(self.temp_dir.name) / "memory.db"))
        self.addCleanup(repository.close)
        service = MemoryService(
            repository,
            SpaceDirectory.from_registry(self.registry, "home-a"),
        )
        observed: list[tuple[str, dict]] = []
        real_record = service.record_operation

        def counting_record(context, memory_key, memory_value, **kwargs):
            observed.append((memory_key, memory_value))
            return real_record(context, memory_key, memory_value, **kwargs)

        service.record_operation = counting_record
        # 图路径语义：偏好观察开启，身份来自 RunnableConfig。
        device_tools = {
            tool.name: tool
            for tool in build_device_tools(self.registry, service)
        }

        wake_at = datetime(2026, 8, 16, 6, 0, tzinfo=UTC)
        routine = self.store.save_routine(build_wake_routine("home-a", "user-a"))
        self.scheduler.schedule(
            routine, anchor_at=wake_at, trigger_key="wake-1", now=wake_at - timedelta(hours=8)
        )
        # 单次 tick 追平全部到期任务，正是应用停机后重启的补偿执行形态。
        self.scheduler.tick(wake_at)

        tasks = self.store.list_tasks(routine.id)
        self.assertEqual(
            [task.status for task in tasks],
            ["completed"] * len(tasks),
            msg=[(task.payload.get("tool_name"), task.error) for task in tasks],
        )
        heater = self.registry.get("bathroom_water_heater")
        self.assertTrue(heater.power)
        self.assertEqual(heater.target_temp, 45)
        self.assertEqual(self.registry.get("kitchen_kettle").target_temp, 80)
        self.assertEqual(self.registry.get("bedroom_light").brightness, 40)
        self.assertEqual(self.registry.get("bedroom_curtain").position, 100)

        # 自动化是机器触发的，计入"重复手动操作"会凭空造出用户没设过的偏好。
        self.assertEqual(observed, [])

        # 但同一个工具走正常对话路径（带可信身份）时必须照常记录，
        # 否则这个修复就退化成"把偏好学习关掉了"。
        device_tools["control_light"].invoke(
            {"device_name": "卧室灯", "action": "set_brightness", "brightness": 70},
            config={"configurable": {
                "home_id": "home-a", "user_id": "user-a",
                "thread_id": "session-a", "client_id": "phone",
            }},
        )
        self.assertEqual(observed, [("lighting.brightness", {"brightness": 70})])

    def test_vehicle_eta_schedules_and_executes_arrival_preparation(self):
        routine = build_arrival_routine("home-a", "user-a")
        routine.metadata["vehicle_id"] = "car-1"
        routine = self.store.save_routine(routine)
        simulator = VehicleSimulator("car-1", "home-a")
        orchestrator = ArrivalOrchestrator(self.store, self.scheduler)
        occurred_at = datetime(2026, 8, 15, 17, 0, tzinfo=UTC)
        event = simulator.eta_event(20, trip_id="trip-1").model_copy(
            update={"occurred_at": occurred_at}
        )

        self.assertEqual(orchestrator.handle(event), 1)
        self.scheduler.tick(occurred_at + timedelta(minutes=5))
        self.assertTrue(self.registry.get("bathroom_water_heater").power)

        self.scheduler.tick(occurred_at + timedelta(minutes=10))
        ac = self.registry.get("living_room_ac")
        self.assertTrue(ac.power)
        self.assertEqual(ac.temperature, 25)

        self.scheduler.tick(occurred_at + timedelta(minutes=18))
        self.assertEqual(self.registry.get("living_room_curtain").position, 100)

        self.scheduler.tick(occurred_at + timedelta(minutes=20))
        self.assertTrue(self.registry.get("living_room_light").power)
        self.assertEqual(len(self.store.list_tasks(routine.id)), 4)

    def test_eta_updates_move_only_pending_tasks_without_duplication(self):
        routine = build_arrival_routine("home-a", "user-a")
        routine.metadata["vehicle_id"] = "car-1"
        routine = self.store.save_routine(routine)
        orchestrator = ArrivalOrchestrator(self.store, self.scheduler)
        start = datetime(2026, 8, 15, 17, 0, tzinfo=UTC)
        first = VehicleEvent(
            vehicle_id="car-1", home_id="home-a", event_type="eta_update",
            latitude=0, longitude=0, eta_minutes=20, occurred_at=start,
            metadata={"trip_id": "trip-2"},
        )
        orchestrator.handle(first)
        self.scheduler.tick(start + timedelta(minutes=5))

        update = first.model_copy(update={
            "event_id": "event-2",
            "eta_minutes": 25,
            "occurred_at": start + timedelta(minutes=1),
        })
        orchestrator.handle(update)
        tasks = self.store.list_tasks(routine.id)
        self.assertEqual(len(tasks), 4)
        completed = [task for task in tasks if task.status == "completed"]
        pending = [task for task in tasks if task.status == "pending"]
        self.assertEqual(len(completed), 1)
        self.assertEqual(len(pending), 3)
        self.assertEqual(pending[0].due_at, start + timedelta(minutes=16))

    def test_cancel_prevents_pending_actions(self):
        wake_at = datetime(2026, 8, 16, 6, 0, tzinfo=UTC)
        routine = self.store.save_routine(build_wake_routine("home-a", "user-a"))
        self.scheduler.schedule(
            routine, anchor_at=wake_at, trigger_key="wake-cancel",
            now=wake_at - timedelta(hours=8),
        )
        self.assertEqual(self.store.cancel_routine(routine.id, "home-a", "user-a"), 5)
        self.assertEqual(self.scheduler.tick(wake_at), [])
        self.assertFalse(self.registry.get("bathroom_water_heater").power)

    def test_runtime_cancel_also_disables_an_already_armed_alarm(self):
        runtime = AutomationRuntime(
            self.registry,
            db_path=str(Path(self.temp_dir.name) / "cancel-alarm.db"),
            speaker=SimulatorSpeakerBackend(),
        )
        wake_at = datetime.now(UTC) + timedelta(hours=8)
        try:
            routine, _ = runtime.schedule_wake("home-a", "user-a", wake_at)
            runtime.scheduler.tick(datetime.now(UTC) + timedelta(seconds=1))
            self.assertTrue(runtime.speaker.list_alarms()[0]["enabled"])
            runtime.cancel(routine.id, "home-a", "user-a")
            self.assertFalse(runtime.speaker.list_alarms()[0]["enabled"])
        finally:
            runtime.close()

    def test_vehicle_and_cancel_are_scoped_to_trusted_owners(self):
        routine = build_arrival_routine("home-a", "user-a")
        routine.metadata["vehicle_id"] = "car-1"
        routine = self.store.save_routine(routine)
        orchestrator = ArrivalOrchestrator(self.store, self.scheduler)
        foreign_car = VehicleSimulator("car-2", "home-a").eta_event(10, trip_id="foreign")
        self.assertEqual(orchestrator.handle(foreign_car), 0)
        self.assertEqual(self.store.list_tasks(routine.id), [])

        wake = self.store.save_routine(build_wake_routine("home-a", "user-a"))
        wake_at = datetime(2026, 8, 16, 6, 0, tzinfo=UTC)
        self.scheduler.schedule(wake, anchor_at=wake_at, trigger_key="owned", now=wake_at)
        self.assertEqual(self.store.cancel_routine(wake.id, "home-a", "user-b"), 0)
        self.assertTrue(any(task.status == "pending" for task in self.store.list_tasks(wake.id)))

    def test_routing_role_and_approval_cover_automation(self):
        intent = classify_intent_fallback("明天早上6点设置闹钟并准备热水")
        self.assertEqual(intent.intent, "automation_management")
        self.assertEqual(agent_for_intent(intent.intent), "automation")

        request = build_approval_request([{
            "id": "call-1",
            "name": "schedule_wake_routine",
            "args": {"wake_at_iso": "2026-08-16T06:00:00+08:00"},
        }])
        self.assertIsNotNone(request)
        self.assertEqual(request["risk_level"], "medium")
        self.assertIn("起床自动化", request["summary"])

        generic = build_approval_request([{
            "id": "call-2",
            "name": "create_scheduled_routine",
            "args": {
                "name": "打球回家准备",
                "anchor_at_iso": "2026-08-15T17:00:00+08:00",
                "actions": [{
                    "offset_minutes": -30,
                    "tool_name": "control_water_heater",
                    "description": "准备洗澡热水",
                    "arguments": {},
                }],
            },
        }])
        self.assertIsNotNone(generic)
        self.assertIn("打球回家准备", generic["summary"])
        self.assertIn("-30分钟", generic["summary"])

    def test_required_tool_only_locks_real_creation_requests(self):
        """查询和取消类自动化请求不得被强制要求调用创建工具。"""
        # 这些问句此前会被判定为"必须创建"，导致 Agent 连续两轮失败后对用户报错。
        for text in (
            "当前有多少个定时任务",
            "现在有几个定时任务",
            "帮我看看有哪些自动化",
            "列出所有定时例程",
            "查询一下明天的定时任务",
            "现在有没有定时任务",
        ):
            with self.subTest(text=text):
                self.assertIsNone(_required_automation_tool(text))

        for text in ("取消明天的起床计划", "删除那个定时任务", "停用车辆回家例程"):
            with self.subTest(text=text):
                self.assertIsNone(_required_automation_tool(text))

        # 真正的创建请求仍然必须锁定到对应的创建工具。
        self.assertEqual(
            _required_automation_tool(
                "我今天下午5点打球回到家，帮我提前准备洗澡水，同时提前打开客厅空调降温"
            ),
            "create_scheduled_routine",
        )
        self.assertEqual(
            _required_automation_tool("明天早上7点叫我起床"),
            "schedule_wake_routine",
        )
        self.assertEqual(
            _required_automation_tool("车辆到家前提前打开空调"),
            "create_vehicle_arrival_routine",
        )
        # 只有触发词没有动作时不强制，让 Agent 自己澄清或查询。
        self.assertIsNone(_required_automation_tool("定时任务"))
        self.assertIsNone(_required_automation_tool(""))

    def test_list_routines_exposes_every_action_detail(self):
        """查看定时任务时必须能拿到每个动作的设备、参数、时间和执行状态。"""
        runtime = AutomationRuntime(
            self.registry,
            db_path=str(Path(self.temp_dir.name) / "list-automation.db"),
            speaker=SimulatorSpeakerBackend(),
        )
        tools = self.automation_tools(runtime)
        trusted = {"configurable": {"home_id": "home-a", "user_id": "user-a"}}
        try:
            # 用相对时间，避免硬编码日期在未来某天变成"目标时间早于当前时间"。
            anchor = datetime.now(UTC) + timedelta(hours=3)
            runtime.create_scheduled_routine(
                "home-a",
                "user-a",
                "下午5点回家准备",
                anchor,
                [
                    ScheduledActionInput(
                        offset_minutes=-30,
                        tool_name="control_water_heater",
                        description="提前30分钟打开热水器准备洗澡水",
                        arguments={
                            "device_name": "卫生间电热水器",
                            "action": "on",
                            "target_temp": 45,
                        },
                    ),
                    ScheduledActionInput(
                        offset_minutes=-20,
                        tool_name="control_ac",
                        description="提前20分钟打开客厅空调制冷降温",
                        arguments={
                            "device_name": "客厅空调",
                            "action": "on",
                            "temperature": 25,
                        },
                    ),
                ],
            )

            payload = json.loads(tools["list_automation_routines"].invoke({}, config=trusted))
            self.assertEqual(len(payload), 1)
            entry = payload[0]
            self.assertEqual(entry["name"], "下午5点回家准备")
            self.assertEqual(entry["action_count"], 2)
            self.assertIsNotNone(entry["target_time"])

            heater, ac = entry["actions"]
            self.assertEqual(heater["timing"], "提前30分钟")
            self.assertEqual(heater["tool_name"], "control_water_heater")
            self.assertEqual(heater["arguments"]["device_name"], "卫生间电热水器")
            self.assertEqual(heater["arguments"]["target_temp"], 45)
            self.assertEqual(heater["status"], "待执行")
            self.assertIsNotNone(heater["scheduled_at"])
            self.assertEqual(ac["timing"], "提前20分钟")
            self.assertEqual(ac["arguments"]["device_name"], "客厅空调")
            self.assertEqual(ac["arguments"]["temperature"], 25)

            # 到期执行后，明细里要能看到已完成状态和实际执行时间。
            runtime.scheduler.tick(anchor - timedelta(minutes=29))
            refreshed = json.loads(tools["list_automation_routines"].invoke({}, config=trusted))
            done, pending = refreshed[0]["actions"]
            self.assertEqual(done["status"], "已完成")
            self.assertIsNotNone(done["executed_at"])
            self.assertEqual(pending["status"], "待执行")
        finally:
            runtime.close()

    def test_list_routines_labels_alarm_and_unscheduled_actions(self):
        """闹钟动作和未排期的车辆例程都要有可读的时间说明。"""
        runtime = AutomationRuntime(
            self.registry,
            db_path=str(Path(self.temp_dir.name) / "list-alarm.db"),
            speaker=SimulatorSpeakerBackend(),
        )
        tools = self.automation_tools(runtime)
        trusted = {"configurable": {"home_id": "home-a", "user_id": "user-a"}}
        try:
            runtime.schedule_wake(
                "home-a", "user-a", datetime.now(UTC) + timedelta(hours=10)
            )
            runtime.enable_vehicle_arrival("home-a", "user-a", "car-1")

            payload = json.loads(tools["list_automation_routines"].invoke({}, config=trusted))
            by_name = {item["name"]: item for item in payload}
            self.assertEqual(set(by_name), {"起床准备", "车辆回家准备"})

            wake = by_name["起床准备"]
            self.assertEqual(wake["action_count"], 5)
            alarm = wake["actions"][0]
            self.assertEqual(alarm["tool_name"], "set_alarm")
            self.assertIn("闹钟", alarm["timing"])
            # 闹钟在创建时就已 arm，due_at 不是响铃时刻，因此不暴露 offset_minutes。
            self.assertNotIn("offset_minutes", alarm)
            self.assertEqual(alarm["arguments"]["speaker_name"], "卧室音响")
            self.assertEqual(wake["actions"][1]["timing"], "提前30分钟")
            self.assertEqual(wake["actions"][3]["timing"], "准点执行")

            # 车辆例程尚未收到 ETA，动作应标为未排期而不是缺失状态。
            vehicle = by_name["车辆回家准备"]
            self.assertEqual(vehicle["trigger_type"], "vehicle_eta")
            self.assertIsNone(vehicle["target_time"])
            self.assertEqual(len(vehicle["actions"]), 4)
            self.assertTrue(
                all(action["status"] == "未排期" for action in vehicle["actions"])
            )
            self.assertTrue(
                all("scheduled_at" not in action for action in vehicle["actions"])
            )
        finally:
            runtime.close()

    def test_generic_plan_handles_ball_game_return_without_a_template(self):
        request = "我今天下午5点打球回到家，帮我提前准备洗澡水，同时提前打开客厅空调降温"
        intent = classify_intent_fallback(request)
        self.assertEqual(intent.intent, "automation_management")

        anchor_at = datetime.now(UTC) + timedelta(hours=5)
        runtime = AutomationRuntime(
            self.registry,
            db_path=str(Path(self.temp_dir.name) / "generic-plan.db"),
            speaker=SimulatorSpeakerBackend(),
        )
        tools = self.automation_tools(runtime)
        try:
            result = tools["create_scheduled_routine"].invoke(
                {
                    "name": "打球回家准备",
                    "anchor_at_iso": anchor_at.isoformat(),
                    "actions": [
                        {
                            "offset_minutes": -30,
                            "tool_name": "control_water_heater",
                            "description": "提前准备洗澡热水",
                            "arguments": {
                                "device_name": "卫生间电热水器",
                                "action": "set_temp",
                                "target_temp": 45,
                            },
                        },
                        {
                            "offset_minutes": -20,
                            "tool_name": "control_ac",
                            "description": "提前降低客厅温度",
                            "arguments": {
                                "device_name": "客厅空调",
                                "action": "on",
                                "temperature": 25,
                                "mode": "cool",
                            },
                        },
                    ],
                },
                config={"configurable": {"home_id": "home-a", "user_id": "user-a"}},
            )
            self.assertIn("已创建定时例程", result)
            routine = runtime.store.list_routines("home-a")[0]
            tasks = runtime.store.list_tasks(routine.id)
            self.assertEqual([task.payload["tool_name"] for task in tasks], [
                "control_water_heater", "control_ac",
            ])

            runtime.scheduler.tick(anchor_at - timedelta(minutes=30))
            self.assertTrue(self.registry.get("bathroom_water_heater").power)
            self.assertFalse(self.registry.get("living_room_ac").power)

            runtime.scheduler.tick(anchor_at - timedelta(minutes=20))
            self.assertTrue(self.registry.get("living_room_ac").power)
            self.assertEqual(self.registry.get("living_room_ac").temperature, 25)
        finally:
            runtime.close()

    def test_generic_plan_normalizes_device_ids_and_turn_on_actions(self):
        """Match the nested arguments emitted by the real automation model."""
        anchor_at = datetime.now(UTC) + timedelta(hours=5)
        runtime = AutomationRuntime(
            self.registry,
            db_path=str(Path(self.temp_dir.name) / "model-style-plan.db"),
            speaker=SimulatorSpeakerBackend(),
        )
        tools = self.automation_tools(runtime)
        try:
            result = tools["create_scheduled_routine"].invoke(
                {
                    "name": "下午5点回家准备",
                    "anchor_at_iso": anchor_at.isoformat(),
                    "actions": [
                        {
                            "offset_minutes": -30,
                            "tool_name": "control_water_heater",
                            "arguments": {
                                "device_id": "bathroom_water_heater",
                                "action": "turn_on",
                            },
                            "description": "提前准备洗澡热水",
                        },
                        {
                            "offset_minutes": -20,
                            "tool_name": "control_ac",
                            "arguments": {
                                "device_id": "living_room_ac",
                                "action": "turn_on",
                                "temperature": 25,
                                "mode": "cool",
                            },
                            "description": "提前打开客厅空调降温",
                        },
                    ],
                },
                config={"configurable": {"home_id": "home-a", "user_id": "user-a"}},
            )
            self.assertIn("已创建定时例程", result)

            routine = runtime.store.list_routines("home-a")[0]
            self.assertEqual(
                routine.actions[0].arguments,
                {"action": "on", "device_name": "卫生间电热水器"},
            )
            self.assertEqual(
                routine.actions[1].arguments,
                {
                    "action": "on",
                    "temperature": 25,
                    "mode": "cool",
                    "device_name": "客厅空调",
                },
            )

            runtime.scheduler.tick(anchor_at - timedelta(minutes=30))
            self.assertTrue(self.registry.get("bathroom_water_heater").power)
            runtime.scheduler.tick(anchor_at - timedelta(minutes=20))
            self.assertTrue(self.registry.get("living_room_ac").power)
            self.assertEqual(self.registry.get("living_room_ac").temperature, 25)
        finally:
            runtime.close()

    def test_generic_plan_rejects_scheduled_unlock(self):
        with self.assertRaises(ValueError):
            ScheduledActionInput(
                offset_minutes=0,
                tool_name="control_lock",
                description="自动解锁",
                arguments={"device_name": "玄关门锁", "action": "unlock"},
            )

    def test_stringified_actions_are_safe_in_approval_and_tool_validation(self):
        actions = [{
            "offset_minutes": -30,
            "tool_name": "control_water_heater",
            "description": "准备洗澡热水",
            "arguments": json.dumps({
                "device_name": "卫生间电热水器",
                "action": "set_temp",
                "target_temp": 45,
            }, ensure_ascii=False),
        }]
        request = build_approval_request([{
            "id": "string-actions",
            "name": "create_scheduled_routine",
            "args": {
                "name": "字符串动作计划",
                "anchor_at_iso": (datetime.now(UTC) + timedelta(hours=4)).isoformat(),
                "actions": json.dumps(actions, ensure_ascii=False),
            },
        }])
        self.assertIsNotNone(request)
        self.assertIn("-30分钟", request["summary"])

        runtime = AutomationRuntime(
            self.registry,
            db_path=str(Path(self.temp_dir.name) / "string-actions.db"),
            speaker=SimulatorSpeakerBackend(),
        )
        tools = self.automation_tools(runtime)
        try:
            result = tools["create_scheduled_routine"].invoke(
                {
                    "name": "字符串动作计划",
                    "anchor_at_iso": (datetime.now(UTC) + timedelta(hours=4)).isoformat(),
                    "actions": json.dumps(actions, ensure_ascii=False),
                },
                config={"configurable": {"home_id": "home-a", "user_id": "user-a"}},
            )
            self.assertIn("已创建定时例程", result)
            self.assertEqual(len(runtime.store.list_tasks()), 1)
        finally:
            runtime.close()

    def test_full_graph_plans_waits_for_approval_then_persists_generic_routine(self):
        anchor_at = datetime.now(UTC) + timedelta(hours=5)
        runtime = AutomationRuntime(
            self.registry,
            db_path=str(Path(self.temp_dir.name) / "graph-generic.db"),
            speaker=SimulatorSpeakerBackend(),
        )

        class BoundAutomationAgent:
            def __init__(self, tool_names):
                self.tool_names = set(tool_names)

            def invoke(self, messages):
                if any(isinstance(message, ToolMessage) for message in messages):
                    return AIMessage(content="定时计划已经创建。")
                if "create_scheduled_routine" not in self.tool_names:
                    return AIMessage(content="无可用自动化工具。")
                if not any(
                    isinstance(message, SystemMessage)
                    and "上一份回答不是可执行结果" in message.content
                    for message in messages
                ):
                    return AIMessage(content=(
                        "我已经整理好定时准备计划，请问这个安排可以吗？"
                        "确认后我马上帮您设置。"
                    ))
                return AIMessage(content="", tool_calls=[{
                    "id": "schedule-call",
                    "name": "create_scheduled_routine",
                    "args": {
                        "name": "打球回家准备",
                        "anchor_at_iso": anchor_at.isoformat(),
                        "actions": [
                            {
                                "offset_minutes": -30,
                                "tool_name": "control_water_heater",
                                "description": "准备洗澡热水",
                                "arguments": {
                                    "device_id": "bathroom_water_heater",
                                    "action": "turn_on",
                                },
                            },
                            {
                                "offset_minutes": -20,
                                "tool_name": "control_ac",
                                "description": "降低客厅温度",
                                "arguments": {
                                    "device_id": "living_room_ac",
                                    "action": "turn_on",
                                    "temperature": 25,
                                    "mode": "cool",
                                },
                            },
                        ],
                    },
                }])

        class AutomationFakeLLM:
            def bind_tools(self, tools):
                return BoundAutomationAgent([tool.name for tool in tools])

            def invoke(self, messages):
                return AIMessage(content="普通回复")

        settings = SimpleNamespace(
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
            planning=SimpleNamespace(enabled=True, max_steps=8, max_step_retries=1, max_replans=1),
            routing=SimpleNamespace(enabled=False, confidence_threshold=0.6),
            multi_agent=SimpleNamespace(enabled=True, max_handoffs=2),
            rag=SimpleNamespace(enabled=False, knowledge_path="docs/knowledge", top_k=3, max_rewrites=1),
            automation=SimpleNamespace(timezone="Asia/Shanghai"),
        )
        context = AgentContext(
            home_id="home-a", user_id="user-a", session_id="generic-graph",
            client_id="test",
        )
        try:
            with patch("src.agent.graph.build_llm", return_value=AutomationFakeLLM()):
                # P1: 自动化运行时经构造参数显式注入图，取代模块级单例。
                graph = build_graph(
                    self.registry,
                    settings,
                    SpaceDirectory.from_registry(self.registry, "home-a"),
                    automation_runtime=runtime,
                )
            interrupted = graph.invoke(
                {
                    "messages": [HumanMessage(content=(
                        "我今天下午5点打球回到家，帮我提前准备洗澡水，"
                        "同时提前打开客厅空调降温"
                    ))],
                    **context.to_state_input(),
                },
                context.to_config(),
            )
            self.assertIn("__interrupt__", interrupted)
            self.assertEqual(runtime.store.list_routines("home-a"), [])

            completed = graph.invoke(
                Command(resume={"approved": True}), context.to_config()
            )
            self.assertEqual(completed["delegated_agent"], "automation")
            routines = runtime.store.list_routines("home-a")
            self.assertEqual(len(routines), 1)
            self.assertEqual(len(runtime.store.list_tasks(routines[0].id)), 2)
            self.assertEqual(
                routines[0].actions[0].arguments,
                {"action": "on", "device_name": "卫生间电热水器"},
            )
            self.assertEqual(routines[0].actions[1].arguments["action"], "on")
            self.assertEqual(routines[0].actions[1].arguments["device_name"], "客厅空调")
        finally:
            runtime.close()

    def test_agent_tool_creates_persistent_wake_routine_from_trusted_context(self):
        runtime = AutomationRuntime(
            self.registry,
            db_path=str(Path(self.temp_dir.name) / "tool-automation.db"),
            speaker=SimulatorSpeakerBackend(),
        )
        tools = self.automation_tools(runtime)
        # 这里必须用相对时间：schedule_wake 会拒绝早于当前时间的目标，原先硬编码的
        # 2026-08-16 在该日过去后就让本用例永久失败。保留"带时区的 ISO 8601 + 早上
        # 6 点"这个真实入参形态（工具 docstring 要求的格式），只把日期取成次日。
        zone = timezone(timedelta(hours=8))
        wake_at = (datetime.now(zone) + timedelta(days=1)).replace(
            hour=6, minute=0, second=0, microsecond=0
        )
        try:
            result = tools["schedule_wake_routine"].invoke(
                {"wake_at_iso": wake_at.isoformat()},
                config={"configurable": {"home_id": "home-a", "user_id": "user-a"}},
            )
            self.assertIn("已创建起床例程", result)
            routines = runtime.store.list_routines("home-a")
            self.assertEqual(len(routines), 1)
            self.assertEqual(len(runtime.store.list_tasks(routines[0].id)), 5)
        finally:
            runtime.close()


if __name__ == "__main__":
    unittest.main()
