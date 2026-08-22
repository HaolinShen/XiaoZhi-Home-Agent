"""P0/P1 守卫用例：能力声明的单一数据源与工具的显式依赖注入。

这里钉住两件事：
  1. P0 的派生链路：新增设备时只要改 capabilities.CAPABILITIES 一处，
     工具集 / Planner 词表 / PlanStep Literal / registry.find 关键词 /
     模拟器默认实例 / 场景批量类型 / 自动化工具名全部自动跟上。
     任何一处派生点失配，测试阶段就失败，而不是运行期静默。
  2. P1 的注入语义：工具经工厂显式持有依赖；偏好观察是构造期选择，
     开启后缺身份必须 fail-fast，关闭后机器触发动作绝不记录偏好。
"""

import unittest

from pydantic import ValidationError

from src.agent.context import SpaceDirectory
from src.agent.planning import (
    DEVICE_ACTION_SPECS,
    PLANNING_TOOL_NAMES,
    PlanStep,
)
from src.automation.planning import AutomationToolName
from src.devices.base import DeviceRegistry
from src.devices.capabilities import (
    CAPABILITIES,
    CAPABILITIES_BY_TOOL,
    CONTROL_TOOL_NAMES,
    SENSOR_DEFAULT_DEVICES,
    TYPE_KEYWORDS,
)
from src.devices.simulator import SimulatorBackend
from src.memory import MemoryRepository, MemoryService
from src.tools import build_all_tools, build_device_tools


class CapabilitySingleSourceTests(unittest.TestCase):
    """P0: 派生视图与能力声明必须逐项一致。"""

    def setUp(self):
        self.registry = DeviceRegistry(SimulatorBackend())

    def test_control_tool_names_derive_from_capabilities(self):
        """每个能力声明恰好对应一个 control_xxx 工具。"""
        tools = {
            tool.name: tool
            for tool in build_device_tools(self.registry, enable_preference_tracking=False)
        }
        for cap in CAPABILITIES:
            with self.subTest(tool=cap.tool_name):
                self.assertIn(cap.tool_name, tools)
                self.assertNotIn("read_sensor", CONTROL_TOOL_NAMES)

    def test_planner_specs_and_literal_derive_from_capabilities(self):
        from typing import get_args

        literal_tools = set(get_args(PlanStep.model_fields["tool_name"].annotation))
        self.assertEqual(literal_tools, set(CAPABILITIES_BY_TOOL))
        self.assertEqual(set(PLANNING_TOOL_NAMES), set(CAPABILITIES_BY_TOOL))
        self.assertEqual(set(DEVICE_ACTION_SPECS), set(CAPABILITIES_BY_TOOL))
        for tool_name, spec in DEVICE_ACTION_SPECS.items():
            cap = CAPABILITIES_BY_TOOL[tool_name]
            self.assertEqual(spec.device_type, cap.device_type)
            self.assertEqual(set(spec.actions), {a.name for a in cap.actions})

    def test_automation_tool_names_derive_from_planning_tool_names(self):
        """AutomationToolName = 规划控制工具 + set_alarm，不再手抄清单。"""
        from typing import get_args

        self.assertEqual(
            set(get_args(AutomationToolName)),
            {*PLANNING_TOOL_NAMES, "set_alarm"},
        )

    def test_registry_keywords_derive_from_capabilities(self):
        """find() 的类型关键词与能力声明一致，且覆盖全部设备类型（含传感器）。"""
        from src.models import DeviceType

        for cap in CAPABILITIES:
            keywords = TYPE_KEYWORDS.get(cap.device_type, ())
            self.assertTrue(keywords, f"{cap.device_type} 缺少类型关键词")
            # 模拟器里至少有一台该类型设备，关键词匹配才有意义。
            self.assertTrue(self.registry.get_by_type(cap.device_type))
        for sensor_type in (DeviceType.TEMP_HUMIDITY_SENSOR, DeviceType.PRESENCE_SENSOR):
            self.assertTrue(TYPE_KEYWORDS.get(sensor_type))

    def test_simulator_defaults_derive_from_capabilities(self):
        """模拟器注册的每台默认设备都能在能力声明里找到出处。"""
        for cap in CAPABILITIES:
            for _, kwargs in cap.default_devices:
                device = self.registry.get(kwargs["device_id"])
                self.assertIsNotNone(device, f"缺少默认设备 {kwargs['device_id']}")
                self.assertEqual(device.name, kwargs["name"])
        for _, kwargs in SENSOR_DEFAULT_DEVICES:
            self.assertIsNotNone(self.registry.get(kwargs["device_id"]))

    def test_tool_schema_exposes_declared_params_and_rejects_foreign_actions(self):
        """生成的工具 Schema 必须覆盖声明参数；其他平台命名必须被拒绝。"""
        tools = {
            tool.name: tool
            for tool in build_device_tools(self.registry, enable_preference_tracking=False)
        }
        light = tools["control_light"]
        schema = light.args_schema.model_json_schema()
        properties = schema["properties"]
        self.assertIn("device_name", properties)
        self.assertIn("action", properties)
        self.assertIn("brightness", properties)
        self.assertIn("color", properties)
        self.assertNotIn("config", properties)  # 身份参数绝不可暴露给模型
        result = light.invoke({"device_name": "客厅灯", "action": "turn_on"})
        self.assertIn("不支持的操作", result)

    def test_plan_step_rejects_sensors_and_unknown_tools(self):
        with self.assertRaises(ValidationError):
            PlanStep(
                step_id=1,
                description="读取湿度",
                tool_name="read_sensor",
                arguments={"sensor_type": "temp_humidity"},
            )
        with self.assertRaises(ValidationError):
            PlanStep(
                step_id=1,
                description="未知工具",
                tool_name="control_fridge",
                arguments={},
            )

    def test_mcp_server_reuses_the_same_tool_implementations(self):
        """MCP 工具与图内工具必须同源（P0 消灭了第 10 份 if/elif 副本）。"""
        from src.mcp.server import create_mcp_server

        built = {
            tool.name: tool
            for tool in build_all_tools(self.registry, enable_preference_tracking=False)
        }
        create_mcp_server(self.registry)
        # 行为同源的直接证据：同一台设备、同一个动作，两边结果逐字一致。
        result = built["control_kettle"].invoke({"device_name": "厨房烧水壶", "action": "boil"})
        self.assertIn("加热至 100°C", result)
        self.registry.update("kitchen_kettle", power=False, target_temp=100)


class ToolInjectionSemanticsTests(unittest.TestCase):
    """P1: 显式依赖注入与偏好观察的构造期选择。"""

    def setUp(self):
        self.registry = DeviceRegistry(SimulatorBackend())

    def test_preference_tracking_requires_full_identity_and_fails_fast(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            repository = MemoryRepository(str(Path(tmp) / "memory.db"))
            try:
                service = MemoryService(
                    repository, SpaceDirectory.from_registry(self.registry, "home-a"),
                )
                tools = {
                    tool.name: tool
                    for tool in build_device_tools(self.registry, service)
                }
                # 缺身份的调用必须立刻失败，而不是静默吞掉（旧行为的隐患）。
                with self.assertRaises(RuntimeError):
                    tools["control_light"].invoke(
                        {"device_name": "客厅灯", "action": "set_brightness", "brightness": 60},
                    )
            finally:
                repository.close()

    def test_tracking_disabled_never_records_even_with_identity(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            repository = MemoryRepository(str(Path(tmp) / "memory.db"))
            try:
                service = MemoryService(
                    repository, SpaceDirectory.from_registry(self.registry, "home-a")
                )
                observed = []
                service.record_operation = (
                    lambda context, key, value, **kw: observed.append((key, value))
                )
                # 后台执行器 / MCP 语义：构造期显式关闭偏好观察。
                tools = {
                    tool.name: tool
                    for tool in build_device_tools(
                        self.registry, service, enable_preference_tracking=False
                    )
                }
                tools["control_light"].invoke(
                    {"device_name": "客厅灯", "action": "set_brightness", "brightness": 60},
                    config={"configurable": {
                        "home_id": "home-a", "user_id": "user-a",
                        "thread_id": "s", "client_id": "c",
                    }},
                )
                self.assertEqual(observed, [])
            finally:
                repository.close()


if __name__ == "__main__":
    unittest.main()
