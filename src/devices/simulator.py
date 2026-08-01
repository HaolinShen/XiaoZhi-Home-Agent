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
"""

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
)


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

        # 使用 Pydantic 的 model_copy + update 确保类型验证
        try:
            updated = device.model_copy(update=kwargs)
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
        ]

        for dev_type, label in type_order:
            devices = self.get_by_type(dev_type)
            if not devices:
                continue
            lines.append(f"**{label}**")
            for dev_id, dev in devices.items():
                lines.append(f"  · {dev.to_status_text()}")

        return "\n".join(lines)

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
        ]

        for device in default_devices:
            self._devices[device.device_id] = device
