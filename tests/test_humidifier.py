import unittest

from pydantic import ValidationError

from src.agent.planning import expected_state_for_step
from src.devices.base import DeviceRegistry
from src.devices.simulator import SimulatorBackend
from src.models import DeviceType, FanSpeed, HumidifierDevice
from src.tools import activate_scene, control_humidifier, set_registry


class HumidifierDeviceTests(unittest.TestCase):
    def setUp(self):
        self.backend = SimulatorBackend()
        self.registry = DeviceRegistry(self.backend)
        set_registry(self.registry)

    def test_model_validates_fields_and_formats_status(self):
        device = HumidifierDevice(
            device_id="test_humidifier",
            name="测试加湿器",
            power=True,
            target_humidity=65,
            mist_level=FanSpeed.MID,
            water_level=80,
        )
        self.assertEqual(DeviceType.HUMIDIFIER.label_cn, "加湿器")
        self.assertIn("目标湿度: 65%", device.to_status_text())
        self.assertIn("雾量: 中", device.to_status_text())
        self.assertIn("水箱: 80%", device.to_status_text())
        with self.assertRaises(ValidationError):
            HumidifierDevice(
                device_id="invalid_humidifier",
                name="非法加湿器",
                target_humidity=90,
            )

    def test_simulator_lists_humidifier_and_rejects_invalid_update(self):
        device = self.registry.find("客厅的加湿器", DeviceType.HUMIDIFIER)
        self.assertIsNotNone(device)
        self.assertIn("💧 加湿器", self.registry.get_status_summary())
        self.assertIn("目标湿度: 60%", self.registry.get_status_summary())
        self.assertFalse(self.registry.update(device.device_id, target_humidity=90))
        self.assertEqual(self.registry.get(device.device_id).target_humidity, 60)

    def test_control_tool_updates_power_humidity_and_mist_level(self):
        on_result = control_humidifier.invoke({
            "device_name": "客厅加湿器",
            "action": "on",
        })
        self.assertTrue(on_result.startswith("✅"))
        self.assertTrue(self.registry.get("living_room_humidifier").power)

        humidity_result = control_humidifier.invoke({
            "device_name": "客厅加湿器",
            "action": "set_humidity",
            "target_humidity": 75,
        })
        self.assertIn("75%", humidity_result)
        self.assertEqual(self.registry.get("living_room_humidifier").target_humidity, 75)

        mist_result = control_humidifier.invoke({
            "device_name": "客厅加湿器",
            "action": "set_mist_level",
            "mist_level": "high",
        })
        self.assertIn("高", mist_result)
        self.assertEqual(
            self.registry.get("living_room_humidifier").mist_level,
            FanSpeed.HIGH,
        )

    def test_empty_water_tank_blocks_power_on(self):
        self.assertTrue(self.registry.update("living_room_humidifier", water_level=0))
        result = control_humidifier.invoke({
            "device_name": "客厅加湿器",
            "action": "on",
        })
        self.assertTrue(result.startswith("❌"))
        self.assertFalse(self.registry.get("living_room_humidifier").power)
        humidity_result = control_humidifier.invoke({
            "device_name": "客厅加湿器",
            "action": "set_humidity",
            "target_humidity": 70,
        })
        self.assertTrue(humidity_result.startswith("❌"))
        self.assertEqual(self.registry.get("living_room_humidifier").target_humidity, 60)

    def test_leaving_scene_turns_humidifier_off(self):
        self.assertTrue(self.registry.update("living_room_humidifier", power=True))
        result = activate_scene.invoke({"scene_name": "离家模式"})
        self.assertIn("加湿器、热水器和烧水壶已关闭", result)
        self.assertFalse(self.registry.get("living_room_humidifier").power)

    def test_planner_can_prepare_verifiable_humidifier_step(self):
        device_id, expected, error = expected_state_for_step(
            {
                "tool_name": "control_humidifier",
                "arguments": {
                    "device_name": "客厅加湿器",
                    "action": "set_humidity",
                    "target_humidity": 70,
                },
            },
            self.registry,
        )
        self.assertIsNone(error)
        self.assertEqual(device_id, "living_room_humidifier")
        self.assertEqual(expected, {"power": True, "target_humidity": 70})


if __name__ == "__main__":
    unittest.main()
