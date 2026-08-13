"""新设备（电热水器 / 门锁 / 电热水壶）的模型、工具与审批行为测试。"""

import unittest

from pydantic import ValidationError

from src.agent.approval import build_approval_request
from src.agent.planning import expected_state_for_step
from src.devices.base import DeviceRegistry
from src.devices.simulator import SimulatorBackend
from src.models import KettleDevice, LockDevice, WaterHeaterDevice
from src.tools import (
    activate_scene,
    control_kettle,
    control_lock,
    control_water_heater,
    set_registry,
)


class WaterHeaterDeviceTests(unittest.TestCase):
    def setUp(self):
        self.backend = SimulatorBackend()
        self.registry = DeviceRegistry(self.backend)
        set_registry(self.registry)

    def test_model_accepts_in_range_and_rejects_out_of_range_temp(self):
        device = WaterHeaterDevice(
            device_id="test_wh", name="测试热水器", target_temp=50,
        )
        self.assertEqual(device.target_temp, 50)
        for bad in (20, 90):
            with self.assertRaises(ValidationError):
                WaterHeaterDevice(
                    device_id=f"bad_wh_{bad}", name="非法热水器", target_temp=bad,
                )

    def test_control_tool_on_off_and_set_temp(self):
        on_result = control_water_heater.invoke({
            "device_name": "卫生间电热水器", "action": "on",
        })
        self.assertTrue(on_result.startswith("✅"))
        self.assertTrue(self.registry.get("bathroom_water_heater").power)

        temp_result = control_water_heater.invoke({
            "device_name": "卫生间电热水器", "action": "set_temp", "target_temp": 50,
        })
        self.assertIn("50°C", temp_result)
        self.assertEqual(self.registry.get("bathroom_water_heater").target_temp, 50)

        off_result = control_water_heater.invoke({
            "device_name": "卫生间电热水器", "action": "off",
        })
        self.assertFalse(self.registry.get("bathroom_water_heater").power)

    def test_invalid_action_is_rejected(self):
        result = control_water_heater.invoke({
            "device_name": "卫生间电热水器", "action": "boil",
        })
        self.assertTrue(result.startswith("❌"))


class LockDeviceTests(unittest.TestCase):
    def setUp(self):
        self.backend = SimulatorBackend()
        self.registry = DeviceRegistry(self.backend)
        set_registry(self.registry)

    def test_factory_default_is_locked(self):
        lock = self.registry.get("entryway_lock")
        self.assertTrue(lock.locked)
        self.assertTrue(lock.power)

    def test_control_tool_switches_locked_state(self):
        self.assertTrue(self.registry.get("entryway_lock").locked)
        unlock_result = control_lock.invoke({
            "device_name": "玄关门锁", "action": "unlock",
        })
        self.assertTrue(unlock_result.startswith("✅"))
        self.assertFalse(self.registry.get("entryway_lock").locked)

        lock_result = control_lock.invoke({
            "device_name": "玄关门锁", "action": "lock",
        })
        self.assertTrue(lock_result.startswith("✅"))
        self.assertTrue(self.registry.get("entryway_lock").locked)

    def test_unlock_requires_approval_but_lock_does_not(self):
        unlock_call = {"name": "control_lock", "args": {"device_name": "玄关门锁", "action": "unlock"}, "id": "u1"}
        lock_call = {"name": "control_lock", "args": {"device_name": "玄关门锁", "action": "lock"}, "id": "l1"}
        light_call = {"name": "control_light", "args": {"device_name": "客厅灯", "action": "off"}, "id": "g1"}

        unlock_req = build_approval_request([unlock_call])
        self.assertIsNotNone(unlock_req)
        self.assertEqual(unlock_req["risk_level"], "high")
        self.assertIn("解锁", unlock_req["summary"])

        self.assertIsNone(build_approval_request([lock_call]))
        self.assertIsNone(build_approval_request([light_call]))


class KettleDeviceTests(unittest.TestCase):
    def setUp(self):
        self.backend = SimulatorBackend()
        self.registry = DeviceRegistry(self.backend)
        set_registry(self.registry)

    def test_model_accepts_in_range_and_rejects_out_of_range_temp(self):
        device = KettleDevice(device_id="test_k", name="测试水壶", target_temp=60)
        self.assertEqual(device.target_temp, 60)
        for bad in (20, 110):
            with self.assertRaises(ValidationError):
                KettleDevice(
                    device_id=f"bad_k_{bad}", name="非法水壶", target_temp=bad,
                )

    def test_boil_is_a_single_step_composite_action(self):
        result = control_kettle.invoke({"device_name": "厨房烧水壶", "action": "boil"})
        self.assertTrue(result.startswith("✅"))
        kettle = self.registry.get("kitchen_kettle")
        self.assertTrue(kettle.power)
        self.assertEqual(kettle.target_temp, 100)

    def test_set_temp_and_off(self):
        control_kettle.invoke({
            "device_name": "厨房烧水壶", "action": "set_temp", "target_temp": 80,
        })
        self.assertEqual(self.registry.get("kitchen_kettle").target_temp, 80)
        self.assertTrue(self.registry.get("kitchen_kettle").power)

        control_kettle.invoke({"device_name": "厨房烧水壶", "action": "off"})
        self.assertFalse(self.registry.get("kitchen_kettle").power)


class NewDevicesPlanningTests(unittest.TestCase):
    def setUp(self):
        self.backend = SimulatorBackend()
        self.registry = DeviceRegistry(self.backend)
        set_registry(self.registry)

    def test_expected_state_for_step_resolves_new_devices(self):
        cases = [
            ("control_water_heater", "卫生间电热水器", "set_temp", {"power": True, "target_temp": 45}),
            ("control_lock", "玄关门锁", "unlock", {"locked": False}),
            ("control_lock", "玄关门锁", "lock", {"locked": True}),
            ("control_kettle", "厨房烧水壶", "boil", {"power": True, "target_temp": 100}),
        ]
        for tool, dev, action, want in cases:
            device_id, expected, error = expected_state_for_step(
                {
                    "tool_name": tool,
                    "arguments": {"device_name": dev, "action": action},
                },
                self.registry,
            )
            self.assertIsNone(error, f"{tool}/{action} 不应判错: {error}")
            self.assertEqual(expected, want)

    def test_unsupported_action_lists_valid_choices(self):
        _, _, error = expected_state_for_step(
            {
                "tool_name": "control_kettle",
                "arguments": {"device_name": "厨房烧水壶", "action": "mute"},
            },
            self.registry,
        )
        self.assertIsNotNone(error)
        self.assertIn("unsupported action", error)
        self.assertIn("boil", error)


class NewDevicesSceneTests(unittest.TestCase):
    def setUp(self):
        self.backend = SimulatorBackend()
        self.registry = DeviceRegistry(self.backend)
        set_registry(self.registry)

    def test_leaving_scene_turns_off_new_actuators_and_locks_door(self):
        self.registry.update("bathroom_water_heater", power=True)
        self.registry.update("kitchen_kettle", power=True)
        self.registry.update("entryway_lock", locked=False)

        result = activate_scene.invoke({"scene_name": "离家模式"})

        self.assertIn("热水器和烧水壶已关闭", result)
        self.assertIn("门锁已上锁", result)
        self.assertFalse(self.registry.get("bathroom_water_heater").power)
        self.assertFalse(self.registry.get("kitchen_kettle").power)
        self.assertTrue(self.registry.get("entryway_lock").locked)


if __name__ == "__main__":
    unittest.main()
