"""
内存设备模拟器
==============
基于内存字典的智能家居设备模拟后端。

用途:
  - 开发阶段: 无需真实硬件即可测试 Agent 逻辑
  - 演示: 展示 Agent 能力，不依赖外部硬件
  - 单元测试: 快速、可预测的设备行为

切换为真实设备的步骤:
  1. 创建新类 RealDeviceBackend(DeviceBackend)
  2. 实现相同的接口方法
  3. 在 main.py 中将 SimulatorBackend 替换为 RealDeviceBackend
  4. 工具层和 Agent 层无需任何修改

内置设备:
  - 3 个灯光: 客厅灯、卧室灯、厨房灯
  - 2 个空调: 客厅空调、卧室空调
  - 1 个电视: 客厅电视
  - 2 个窗帘: 客厅窗帘、卧室窗帘
  - 1 个加湿器: 客厅加湿器
  - 1 个电热水器: 卫生间电热水器
  - 1 个门锁: 玄关门锁
  - 1 个电热水壶: 厨房烧水壶
  - 2 个温湿度传感器: 客厅温湿度传感器、卧室温湿度传感器
  - 2 个人体存在传感器: 客厅人体传感器、玄关人体传感器

传感器为什么会"动"?
  真实传感器读到的是环境值，会随设备运行而变化。如果模拟器永远返回常量，
  "开加湿器 → 湿度上升 → 验证成功"这条闭环就演示不出来。
  所以这里提供 tick_environment()，按同房间执行器的状态推演温湿度和占用。
  推演是确定性的（固定步长，无随机数），测试才能断言。

  注意它是**显式调用**的：只有 read_sensor / get_device_status 这两个
  "读环境"的入口会调用它，get()/get_all() 不会。原因见 tick_environment
  的文档字符串——让读取本身改状态会导致验证器读到的值随调用次数漂移。
"""

from datetime import datetime, timedelta, timezone
from typing import Optional
from loguru import logger

from .base import DeviceBackend
from ..models import (
    AnyDevice,
    DeviceType,
    LightDevice,
    ACDevice,
    TVDevice,
    CurtainDevice,
    ACMode,
    FanSpeed,
    HumidifierDevice,
    WaterHeaterDevice,
    KettleDevice,
    LockDevice,
    TempHumiditySensor,
    PresenceSensor,
)


# 每次推演环境时读数的变化步长。
# 取小值是为了让多次读取能看出趋势，而不是一步跳到目标。
_TEMP_STEP = 0.5      # °C，空调把室温朝目标温度拉近的速度
_HUMIDITY_STEP = 2    # %，加湿器把湿度朝目标拉近的速度
_DRY_DRIFT = 1        # %，没有加湿器工作时湿度自然回落的速度
_BASELINE_HUMIDITY = 45  # %，无人工干预时房间的湿度基线


class SimulatorBackend(DeviceBackend):
    """
    基于内存字典的模拟设备后端。

    所有设备状态存储在 self._devices 字典中。
    程序重启后状态会重置为默认值。
    """

    def __init__(self):
        """初始化模拟设备，创建默认设备列表"""
        self._devices: dict[str, AnyDevice] = {}
        self._init_default_devices()
        logger.info(
            f"SimulatorBackend 已初始化 | 设备数量: {len(self._devices)}"
        )

    # ---- 接口实现 ----

    def get(self, device_id: str) -> Optional[AnyDevice]:
        return self._devices.get(device_id)

    def get_all(self) -> dict[str, AnyDevice]:
        return self._devices

    def get_by_type(self, device_type: DeviceType) -> dict[str, AnyDevice]:
        return {
            k: v for k, v in self._devices.items()
            if v.device_type == device_type
        }

    def update(self, device_id: str, **kwargs) -> bool:
        device = self._devices.get(device_id)
        if device is None:
            return False

        # 合并旧状态后重新构造模型，确保所有字段约束仍然生效。
        try:
            data = device.model_dump()
            data.update(kwargs)
            updated = type(device).model_validate(data)
            self._devices[device_id] = updated
            logger.debug(f"设备已更新 | {device_id}: {kwargs}")
            return True
        except Exception as e:
            logger.error(f"设备更新异常 | {device_id}: {e}")
            return False

    def get_status_summary(self) -> str:
        """生成格式化设备状态报告（给 LLM 和用户阅读）"""
        lines = ["🏠 **当前所有设备状态:**", ""]

        type_order = [
            (DeviceType.LIGHT, "💡 灯光"),
            (DeviceType.AC, "❄️ 空调"),
            (DeviceType.TV, "📺 电视"),
            (DeviceType.CURTAIN, "🪟 窗帘"),
            (DeviceType.HUMIDIFIER, "💧 加湿器"),
            (DeviceType.WATER_HEATER, "🚿 电热水器"),
            (DeviceType.LOCK, "🔒 门锁"),
            (DeviceType.KETTLE, "☕ 电热水壶"),
            (DeviceType.TEMP_HUMIDITY_SENSOR, "🌡️ 温湿度传感器"),
            (DeviceType.PRESENCE_SENSOR, "👤 人体存在传感器"),
        ]

        for dev_type, label in type_order:
            devices = self.get_by_type(dev_type)
            if not devices:
                continue
            lines.append(f"**{label}**")
            for dev_id, dev in devices.items():
                lines.append(f"  · {dev.to_status_text()}")

        return "\n".join(lines)

    # ---- 环境推演 ----

    def tick_environment(self) -> None:
        """
        按同房间执行器的状态推进一次传感器读数。

        为什么是显式调用，而不是在 get() 里自动跑？
          get()/get_all() 被场景模式、验证器等大量调用，如果每次读取都改状态，
          "读一下看看" 就会变成 "读一下顺便改了"，验证器读到的值也会随调用
          次数漂移，非常难排查。所以推演做成显式动作：只有传感器读取工具
          （read_sensor）会主动调用它，其他读取路径拿到的是稳定快照。

        推演规则（全部确定性，无随机数）:
          温度: 同房间空调开着 → 每次朝目标温度靠近 _TEMP_STEP
                制冷只降不升，制热只升不降，避免来回抖动
          湿度: 同房间加湿器开着 → 朝目标湿度靠近 _HUMIDITY_STEP
                否则朝 _BASELINE_HUMIDITY 回落 _DRY_DRIFT
          人体: 从 last_motion_at 与 timeout_minutes 推算，
                超时未见活动即回落为无人
        """
        for device_id, sensor in list(self._devices.items()):
            if sensor.device_type == DeviceType.TEMP_HUMIDITY_SENSOR:
                self._tick_temp_humidity(device_id, sensor)
            elif sensor.device_type == DeviceType.PRESENCE_SENSOR:
                self._tick_presence(device_id, sensor)

    def _tick_temp_humidity(self, device_id: str, sensor: AnyDevice) -> None:
        """推进一个温湿度传感器的读数"""
        if not sensor.power:
            return

        temperature = sensor.temperature
        humidity = sensor.humidity

        # ---- 温度: 跟随同房间正在运行的空调 ----
        ac = self._first_running(sensor.location, DeviceType.AC)
        if ac is not None:
            target = float(ac.temperature)
            if ac.mode == ACMode.COOL and temperature > target:
                temperature = max(target, temperature - _TEMP_STEP)
            elif ac.mode == ACMode.HEAT and temperature < target:
                temperature = min(target, temperature + _TEMP_STEP)

        # ---- 湿度: 加湿器工作则上升，否则自然回落 ----
        humidifier = self._first_running(sensor.location, DeviceType.HUMIDIFIER)
        if humidifier is not None and humidifier.water_level > 0:
            target_humidity = humidifier.target_humidity
            if humidity < target_humidity:
                humidity = min(target_humidity, humidity + _HUMIDITY_STEP)
            elif humidity > target_humidity:
                humidity = max(target_humidity, humidity - _HUMIDITY_STEP)
        elif humidity > _BASELINE_HUMIDITY:
            humidity = max(_BASELINE_HUMIDITY, humidity - _DRY_DRIFT)

        if temperature != sensor.temperature or humidity != sensor.humidity:
            self.update(device_id, temperature=round(temperature, 1), humidity=humidity)

    def _tick_presence(self, device_id: str, sensor: AnyDevice) -> None:
        """按超时规则推算一个人体传感器的占用状态"""
        if not sensor.power:
            return

        if sensor.last_motion_at is None:
            if sensor.occupied:
                self.update(device_id, occupied=False)
            return

        try:
            last_motion = datetime.fromisoformat(sensor.last_motion_at)
        except ValueError:
            logger.warning(
                f"人体传感器时间格式无法解析 | {device_id}: {sensor.last_motion_at}"
            )
            return

        if last_motion.tzinfo is None:
            last_motion = last_motion.replace(tzinfo=timezone.utc)

        deadline = last_motion + timedelta(minutes=sensor.timeout_minutes)
        occupied = datetime.now(timezone.utc) <= deadline
        if occupied != sensor.occupied:
            self.update(device_id, occupied=occupied)

    def _first_running(
        self, location: str, device_type: DeviceType
    ) -> Optional[AnyDevice]:
        """找到指定房间里第一台正在运行的该类型设备，没有则返回 None"""
        for device in self._devices.values():
            if (
                device.device_type == device_type
                and device.location == location
                and device.power
            ):
                return device
        return None

    # ---- 初始化默认设备 ----

    def _init_default_devices(self) -> None:
        """创建默认的智能家居设备列表"""
        default_devices: list[AnyDevice] = [
            # ===== 灯光 =====
            LightDevice(
                device_id="living_room_light",
                name="客厅灯",
                location="客厅",
                brightness=80,
                color="暖白",
            ),
            LightDevice(
                device_id="bedroom_light",
                name="卧室灯",
                location="卧室",
                brightness=60,
                color="暖白",
            ),
            LightDevice(
                device_id="kitchen_light",
                name="厨房灯",
                location="厨房",
                brightness=100,
                color="白光",
            ),

            # ===== 空调 =====
            ACDevice(
                device_id="living_room_ac",
                name="客厅空调",
                location="客厅",
                temperature=26,
                mode=ACMode.COOL,
                fan_speed=FanSpeed.AUTO,
            ),
            ACDevice(
                device_id="bedroom_ac",
                name="卧室空调",
                location="卧室",
                temperature=26,
                mode=ACMode.COOL,
                fan_speed=FanSpeed.AUTO,
            ),

            # ===== 电视 =====
            TVDevice(
                device_id="living_room_tv",
                name="客厅电视",
                location="客厅",
                volume=30,
                channel="HDMI 1",
            ),

            # ===== 窗帘 =====
            CurtainDevice(
                device_id="living_room_curtain",
                name="客厅窗帘",
                location="客厅",
                position=0,
            ),
            CurtainDevice(
                device_id="bedroom_curtain",
                name="卧室窗帘",
                location="卧室",
                position=0,
            ),

            # ===== 加湿器 =====
            HumidifierDevice(
                device_id="living_room_humidifier",
                name="客厅加湿器",
                location="客厅",
                target_humidity=60,
                mist_level=FanSpeed.AUTO,
                water_level=100,
            ),

            # ===== 电热水器（洗澡用）=====
            WaterHeaterDevice(
                device_id="bathroom_water_heater",
                name="卫生间电热水器",
                location="卫生间",
                power=False,
                target_temp=45,
            ),

            # ===== 智能门锁 =====
            # 出厂即锁（locked=True），解锁是对外动作，会走人工审批。
            LockDevice(
                device_id="entryway_lock",
                name="玄关门锁",
                location="玄关",
                locked=True,
                battery=90,
            ),

            # ===== 电热水壶 =====
            KettleDevice(
                device_id="kitchen_kettle",
                name="厨房烧水壶",
                location="厨房",
                power=False,
                target_temp=100,
            ),

            # ===== 温湿度传感器（只读）=====
            # 初始湿度低于加湿器目标湿度 60%，这样"开加湿器 → 湿度上升"
            # 的闭环一开机就能演示出来。
            TempHumiditySensor(
                device_id="living_room_th_sensor",
                name="客厅温湿度传感器",
                location="客厅",
                temperature=27.0,
                humidity=42,
            ),
            TempHumiditySensor(
                device_id="bedroom_th_sensor",
                name="卧室温湿度传感器",
                location="卧室",
                temperature=26.0,
                humidity=48,
            ),

            # ===== 人体存在传感器（只读）=====
            # last_motion_at 留空表示开机时没有任何活动记录，
            # 因此 occupied 为 False。测试可以写入具体时间来控制状态。
            PresenceSensor(
                device_id="living_room_presence",
                name="客厅人体传感器",
                location="客厅",
                timeout_minutes=15,
            ),
            PresenceSensor(
                device_id="entryway_presence",
                name="玄关人体传感器",
                location="玄关",
                timeout_minutes=5,
            ),
        ]

        for device in default_devices:
            self._devices[device.device_id] = device
