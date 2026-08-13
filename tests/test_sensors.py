"""温湿度 / 人体存在传感器测试

覆盖四层:
  1. 模型层: 字段约束与状态文本
  2. 模拟器层: 环境推演的确定性（温度、湿度、占用超时）
  3. 工具层: read_sensor 的类型/房间筛选与错误提示
  4. 架构约束: 传感器只读 —— 不进场景批量开关、不进规划工具白名单
"""

import unittest
from datetime import datetime, timedelta, timezone

from pydantic import ValidationError

from src.agent.planning import PLANNING_TOOL_NAMES, PlanStep
from src.devices.base import DeviceRegistry
from src.devices.simulator import SimulatorBackend
from src.models import (
    SENSOR_DEVICE_TYPES,
    DeviceType,
    PresenceSensor,
    TempHumiditySensor,
)
from src.tools import activate_scene, control_humidifier, read_sensor, set_registry


def _iso_minutes_ago(minutes: int) -> str:
    """生成一个"N 分钟前"的 ISO 8601 时间戳（带 UTC 时区）"""
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()


class SensorModelTests(unittest.TestCase):
    """第 1 层: Pydantic 模型的约束与展示"""

    def test_temp_humidity_labels_and_status_text(self):
        sensor = TempHumiditySensor(
            device_id="test_th",
            name="测试温湿度传感器",
            temperature=23.5,
            humidity=55,
            battery=88,
        )
        self.assertEqual(DeviceType.TEMP_HUMIDITY_SENSOR.label_cn, "温湿度传感器")
        text = sensor.to_status_text()
        self.assertIn("温度 23.5°C", text)  # 状态文本保留一位小数
        self.assertIn("湿度 55%", text)
        self.assertIn("电量 88%", text)

    def test_temp_humidity_rejects_out_of_range_humidity(self):
        with self.assertRaises(ValidationError):
            TempHumiditySensor(
                device_id="bad_th", name="非法温湿度传感器", humidity=120
            )

    def test_temp_humidity_rejects_impossible_temperature(self):
        with self.assertRaises(ValidationError):
            TempHumiditySensor(
                device_id="bad_th2", name="非法温湿度传感器", temperature=99.0
            )

    def test_offline_sensor_reports_offline_instead_of_stale_values(self):
        sensor = TempHumiditySensor(
            device_id="offline_th", name="离线传感器", power=False, temperature=31.0
        )
        self.assertIn("离线", sensor.to_status_text())
        self.assertNotIn("31", sensor.to_status_text())

    def test_presence_status_text_distinguishes_occupied(self):
        empty = PresenceSensor(device_id="p1", name="空房间传感器")
        self.assertIn("无人", empty.to_status_text())

        busy = PresenceSensor(
            device_id="p2",
            name="有人传感器",
            occupied=True,
            last_motion_at=_iso_minutes_ago(1),
        )
        self.assertIn("有人", busy.to_status_text())

    def test_sensor_types_are_registered_as_readonly(self):
        self.assertIn(DeviceType.TEMP_HUMIDITY_SENSOR, SENSOR_DEVICE_TYPES)
        self.assertIn(DeviceType.PRESENCE_SENSOR, SENSOR_DEVICE_TYPES)
        self.assertNotIn(DeviceType.LIGHT, SENSOR_DEVICE_TYPES)


class SensorSimulatorTests(unittest.TestCase):
    """第 2 层: 环境推演必须是确定性的，测试才能断言"""

    def setUp(self):
        self.backend = SimulatorBackend()
        self.registry = DeviceRegistry(self.backend)
        set_registry(self.registry)

    def test_default_devices_include_both_sensor_kinds(self):
        self.assertEqual(len(self.registry.get_by_type(DeviceType.TEMP_HUMIDITY_SENSOR)), 2)
        self.assertEqual(len(self.registry.get_by_type(DeviceType.PRESENCE_SENSOR)), 2)
        summary = self.registry.get_status_summary()
        self.assertIn("🌡️ 温湿度传感器", summary)
        self.assertIn("👤 人体存在传感器", summary)

    def test_humidifier_raises_humidity_step_by_step(self):
        """开加湿器 → 湿度朝目标爬升，这是 verifier 能验的真实环境反馈"""
        start = self.registry.get("living_room_th_sensor").humidity
        self.assertLess(start, 60)  # 初始 42%，低于加湿器目标 60%

        self.registry.update("living_room_humidifier", power=True, target_humidity=60)

        self.registry.tick_environment()
        after_one = self.registry.get("living_room_th_sensor").humidity
        self.assertGreater(after_one, start)

        for _ in range(20):
            self.registry.tick_environment()
        self.assertEqual(self.registry.get("living_room_th_sensor").humidity, 60)

    def test_humidity_falls_back_to_baseline_without_humidifier(self):
        self.registry.update("living_room_th_sensor", humidity=70)
        for _ in range(40):
            self.registry.tick_environment()
        self.assertEqual(self.registry.get("living_room_th_sensor").humidity, 45)

    def test_empty_water_tank_does_not_raise_humidity(self):
        self.registry.update(
            "living_room_humidifier", power=True, target_humidity=60, water_level=0
        )
        self.registry.update("living_room_th_sensor", humidity=42)
        self.registry.tick_environment()
        # 水箱空 → 不加湿，湿度按自然回落处理（42 已低于基线 45，保持不动）
        self.assertEqual(self.registry.get("living_room_th_sensor").humidity, 42)

    def test_cooling_ac_lowers_temperature_without_overshooting(self):
        self.registry.update("living_room_th_sensor", temperature=30.0)
        self.registry.update(
            "living_room_ac", power=True, temperature=24, mode="cool"
        )
        for _ in range(30):
            self.registry.tick_environment()
        self.assertEqual(self.registry.get("living_room_th_sensor").temperature, 24.0)

    def test_heating_ac_raises_temperature(self):
        self.registry.update("bedroom_th_sensor", temperature=18.0)
        self.registry.update("bedroom_ac", power=True, temperature=25, mode="heat")
        for _ in range(30):
            self.registry.tick_environment()
        self.assertEqual(self.registry.get("bedroom_th_sensor").temperature, 25.0)

    def test_ac_only_affects_sensors_in_the_same_room(self):
        before = self.registry.get("bedroom_th_sensor").temperature
        self.registry.update("living_room_th_sensor", temperature=30.0)
        self.registry.update("living_room_ac", power=True, temperature=20, mode="cool")
        for _ in range(10):
            self.registry.tick_environment()
        self.assertEqual(self.registry.get("bedroom_th_sensor").temperature, before)
        self.assertLess(self.registry.get("living_room_th_sensor").temperature, 30.0)

    def test_recent_motion_marks_room_occupied(self):
        self.registry.update("living_room_presence", last_motion_at=_iso_minutes_ago(2))
        self.registry.tick_environment()
        self.assertTrue(self.registry.get("living_room_presence").occupied)

    def test_stale_motion_times_out_to_unoccupied(self):
        self.registry.update(
            "living_room_presence", occupied=True, last_motion_at=_iso_minutes_ago(30)
        )
        self.registry.tick_environment()
        self.assertFalse(self.registry.get("living_room_presence").occupied)

    def test_timeout_boundary_respects_per_sensor_setting(self):
        """玄关超时 5 分钟，客厅 15 分钟；同样 10 分钟前的活动结论应相反"""
        stamp = _iso_minutes_ago(10)
        self.registry.update("entryway_presence", last_motion_at=stamp)
        self.registry.update("living_room_presence", last_motion_at=stamp)
        self.registry.tick_environment()
        self.assertFalse(self.registry.get("entryway_presence").occupied)
        self.assertTrue(self.registry.get("living_room_presence").occupied)

    def test_offline_sensor_is_never_advanced(self):
        self.registry.update(
            "living_room_th_sensor", power=False, temperature=30.0, humidity=42
        )
        self.registry.update("living_room_humidifier", power=True, target_humidity=60)
        self.registry.tick_environment()
        sensor = self.registry.get("living_room_th_sensor")
        self.assertEqual(sensor.temperature, 30.0)
        self.assertEqual(sensor.humidity, 42)

    def test_malformed_timestamp_leaves_state_untouched(self):
        self.registry.update(
            "living_room_presence", occupied=True, last_motion_at="not-a-timestamp"
        )
        self.registry.tick_environment()
        self.assertTrue(self.registry.get("living_room_presence").occupied)

    def test_reading_does_not_drift_without_explicit_tick(self):
        """控制/验证路径只读快照，不能因为多读几次就让环境值漂移"""
        self.registry.update("living_room_humidifier", power=True, target_humidity=60)
        for _ in range(5):
            self.registry.get_status_summary()
            self.registry.get("living_room_th_sensor")
            self.registry.get_all()
        self.assertEqual(self.registry.get("living_room_th_sensor").humidity, 42)


class ReadSensorToolTests(unittest.TestCase):
    """第 3 层: read_sensor 工具的筛选行为与错误提示"""

    def setUp(self):
        self.backend = SimulatorBackend()
        self.registry = DeviceRegistry(self.backend)
        set_registry(self.registry)

    def test_reads_all_temp_humidity_sensors_by_default(self):
        result = read_sensor.invoke({"sensor_type": "temp_humidity"})
        self.assertIn("客厅温湿度传感器", result)
        self.assertIn("卧室温湿度传感器", result)

    def test_filters_by_location(self):
        result = read_sensor.invoke(
            {"sensor_type": "temp_humidity", "location": "客厅"}
        )
        self.assertIn("客厅温湿度传感器", result)
        self.assertNotIn("卧室温湿度传感器", result)

    def test_reads_presence_sensors(self):
        self.registry.update("entryway_presence", last_motion_at=_iso_minutes_ago(1))
        result = read_sensor.invoke({"sensor_type": "presence", "location": "玄关"})
        self.assertIn("玄关人体传感器", result)
        self.assertIn("有人", result)

    def test_unknown_sensor_type_is_rejected_with_options(self):
        result = read_sensor.invoke({"sensor_type": "co2"})
        self.assertTrue(result.startswith("❌"))
        self.assertIn("temp_humidity", result)
        self.assertIn("presence", result)

    def test_unknown_location_lists_available_places(self):
        result = read_sensor.invoke(
            {"sensor_type": "temp_humidity", "location": "书房"}
        )
        self.assertTrue(result.startswith("❌"))
        self.assertIn("客厅", result)

    def test_tool_advances_environment_before_reporting(self):
        """读取是唯一会推进环境的入口，所以读到的值应当已经反映加湿器在工作"""
        self.registry.update("living_room_humidifier", power=True, target_humidity=60)
        read_sensor.invoke({"sensor_type": "temp_humidity", "location": "客厅"})
        self.assertGreater(self.registry.get("living_room_th_sensor").humidity, 42)

    def test_control_then_read_shows_the_closed_loop(self):
        """完整闭环: 开加湿器 → 反复读取 → 湿度确实升到目标"""
        control_humidifier.invoke({"device_name": "客厅加湿器", "action": "on"})
        control_humidifier.invoke(
            {
                "device_name": "客厅加湿器",
                "action": "set_humidity",
                "target_humidity": 60,
            }
        )
        for _ in range(20):
            read_sensor.invoke({"sensor_type": "temp_humidity", "location": "客厅"})
        self.assertEqual(self.registry.get("living_room_th_sensor").humidity, 60)


class SensorReadOnlyContractTests(unittest.TestCase):
    """第 4 层: 只读约束不能被场景层或规划层绕过"""

    def setUp(self):
        self.backend = SimulatorBackend()
        self.registry = DeviceRegistry(self.backend)
        set_registry(self.registry)

    def test_leaving_scene_does_not_switch_sensors_off(self):
        activate_scene.invoke({"scene_name": "离家模式"})
        for sensor in self.registry.get_by_type(DeviceType.TEMP_HUMIDITY_SENSOR).values():
            self.assertTrue(sensor.power, "离家模式不应该关闭温湿度传感器")
        for sensor in self.registry.get_by_type(DeviceType.PRESENCE_SENSOR).values():
            self.assertTrue(sensor.power, "离家模式不应该关闭人体传感器")

    def test_planner_cannot_schedule_a_sensor_read_as_a_step(self):
        """计划步骤必须可验证，而"读一下"不改变任何状态，所以不进白名单"""
        self.assertNotIn("read_sensor", PLANNING_TOOL_NAMES)
        with self.assertRaises(ValidationError):
            PlanStep(
                step_id=1,
                description="读取客厅湿度",
                tool_name="read_sensor",
                arguments={"sensor_type": "temp_humidity"},
            )

    def test_device_list_prompt_separates_controllable_from_readonly(self):
        prompt = self.registry.get_device_list_prompt()
        self.assertIn("只读传感器列表", prompt)
        controllable, readonly = prompt.split("只读传感器列表", 1)
        self.assertIn("客厅灯", controllable)
        self.assertNotIn("客厅温湿度传感器", controllable)
        self.assertIn("客厅温湿度传感器", readonly)
        self.assertIn("客厅人体传感器", readonly)

    def test_registry_find_supports_sensor_names_but_is_not_the_read_path(self):
        """find() 认识传感器名字，但读取传感器不走 find()

        read_sensor 用 get_by_type + location 筛选，不做名称模糊匹配——
        因为"客厅湿度多少"里的"湿度"是问指标，不是指某台设备。
        find() 这里只需要保证：给出完整名字能定位到正确的那一台。
        """
        found = self.registry.find("客厅温湿度传感器", DeviceType.TEMP_HUMIDITY_SENSOR)
        self.assertIsNotNone(found)
        self.assertEqual(found.device_id, "living_room_th_sensor")

        found_bedroom = self.registry.find("卧室温湿度", DeviceType.TEMP_HUMIDITY_SENSOR)
        self.assertIsNotNone(found_bedroom)
        self.assertEqual(found_bedroom.device_id, "bedroom_th_sensor")

    def test_registry_find_refuses_type_keyword_when_multiple_candidates(self):
        """只说"人体感应"而家里有两个人体传感器时，不该替用户挑一个"""
        self.assertIsNone(self.registry.find("人体感应", DeviceType.PRESENCE_SENSOR))


if __name__ == "__main__":
    unittest.main()
