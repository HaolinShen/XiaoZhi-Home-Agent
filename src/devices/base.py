"""
设备注册中心
============
管理所有智能家居设备的生命周期和状态。

设计模式: Registry Pattern（注册表模式）
  - 所有设备在启动时注册到 Registry
  - 工具层通过 Registry 查找和操作设备
  - 不关心设备是模拟的还是真实的（依赖倒置）

扩展方式:
  - 当前: 内存字典存储（SimulatorBackend）
  - 未来: Home Assistant / MQTT / 涂鸦 API / 小米米家 SDK
    只需实现相同的接口，替换 backend 即可，工具层零修改
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable

from loguru import logger

from ..models import (
    SENSOR_DEVICE_TYPES,
    AnyDevice,
    DeviceType,
)
from .capabilities import TYPE_KEYWORDS

# 设备名模糊匹配时忽略的助词与量词。
# 让"客厅的灯""那台空调"能匹配到"客厅灯""客厅空调"，
# 而不影响策略 3"多候选拒绝猜测"的安全规则。
# 这里只放纯虚词/量词，绝不放可能作为设备名区分字的实义字（如"室""厅"）。
_MATCH_STOPWORDS = frozenset("的之那这台个只盏")


# ============================================================
# 抽象后端接口
# ============================================================

class DeviceBackend(ABC):
    """
    设备后端抽象基类。

    定义了设备存储和操作的标准接口。
    所有后端（模拟器、Home Assistant、MQTT 等）必须实现这些方法。

    这是"依赖倒置原则"的体现:
      - 高层模块（Tools、Agent）依赖此抽象接口
      - 低层模块（Simulator、HA Client）实现此接口
    """

    @abstractmethod
    def get(self, device_id: str) -> AnyDevice | None:
        """根据 ID 获取设备"""
        ...

    @abstractmethod
    def get_all(self) -> dict[str, AnyDevice]:
        """获取所有设备"""
        ...

    @abstractmethod
    def get_by_type(self, device_type: DeviceType) -> dict[str, AnyDevice]:
        """按类型筛选设备"""
        ...

    @abstractmethod
    def update(self, device_id: str, **kwargs) -> bool:
        """更新设备属性，返回是否成功"""
        ...

    @abstractmethod
    def get_status_summary(self) -> str:
        """生成所有设备状态的摘要文本（给 LLM 阅读）"""
        ...

    def tick_environment(self) -> None:
        """
        推进一次环境推演（可选实现）。

        模拟器需要按执行器状态推算传感器读数，所以会覆盖这个方法。
        真实后端（Home Assistant / MQTT）的传感器由硬件自行上报，
        不需要任何推演，直接继承这个空实现即可——所以它是普通方法
        而不是 @abstractmethod，新增后端不会因此被强制改代码。
        """
        return None


# ============================================================
# 设备注册中心
# ============================================================

class DeviceRegistry:
    """
    设备注册中心。

    职责:
      1. 管理所有设备实例
      2. 提供设备查找（精确/模糊匹配）
      3. 委托后端执行实际的设备操作

    使用方式:
      from src.devices.base import DeviceRegistry
      registry = DeviceRegistry(backend=simulator_backend)
      device = registry.find("客厅灯", device_type=DeviceType.LIGHT)
    """

    def __init__(self, backend: DeviceBackend):
        """
        初始化注册中心。

        参数:
          backend: 设备后端实现（如 SimulatorBackend）
        """
        self._backend = backend
        logger.info(f"DeviceRegistry 已初始化 | backend={backend.__class__.__name__}")

    # ---- 后端属性（只读）----
    @property
    def backend(self) -> DeviceBackend:
        return self._backend

    # ---- 设备查找 ----

    def get(self, device_id: str) -> AnyDevice | None:
        """精确 ID 查找"""
        return self._backend.get(device_id)

    def get_all(self) -> dict[str, AnyDevice]:
        """获取所有设备"""
        return self._backend.get_all()

    def get_by_type(self, device_type: DeviceType) -> dict[str, AnyDevice]:
        """按类型获取设备"""
        return self._backend.get_by_type(device_type)

    def find(self, user_input: str, device_type: DeviceType) -> AnyDevice | None:
        """
        模糊查找设备。

        根据用户输入的自然语言描述，匹配最合适的设备。
        匹配策略（按优先级）:
          1. 名称精确匹配: 用户输入 == 设备中文名
          2. 字符包含匹配: 用户输入的每个字都在设备名中出现
          3. 类型关键词匹配: 用户提到了"灯"/"空调"等关键词 → 返回该类型第一个设备

        参数:
          user_input:  用户输入的中文描述（如 "客厅灯"、"卧室的空调"、"电视"）
          device_type: 设备类型，只在该类型的设备中查找

        返回:
          匹配到的设备，找不到返回 None
        """
        type_devices = self._backend.get_by_type(device_type)

        if not type_devices:
            return None

        # ---- 策略 1: 精确匹配 ----
        for device in type_devices.values():
            if device.name == user_input:
                return device

        # ---- 策略 2: 模糊匹配（字符包含，忽略助词与量词）----
        # 过滤掉"的/那/台"等虚词后再比对，这样"客厅的灯"也能命中"客厅灯"。
        for device in type_devices.values():
            chars = [
                c for c in user_input
                if c.strip() and c not in _MATCH_STOPWORDS
            ]
            if chars and all(c in device.name for c in chars):
                return device

        # ---- 策略 3: 类型关键词匹配 ----
        # 关键词表来自 capabilities.TYPE_KEYWORDS（P0 单一数据源）：
        # 新增设备类型时在那里声明一次，find 的匹配、工具 docstring、模拟器
        # 默认实例全部自动跟上，不用再手改本文件。
        # 注解写成 Iterable：get() 的默认值是 list 而表里存的是 tuple，
        # 让两者都合法，不为了类型把默认值改成空 tuple。
        keywords: Iterable[str] = TYPE_KEYWORDS.get(device_type, ())
        for kw in keywords:
            if kw in user_input:
                # 多候选时拒绝猜测，让 Agent 向用户澄清。
                if len(type_devices) == 1:
                    return next(iter(type_devices.values()))
                return None

        return None

    # ---- 设备操作 ----

    def update(self, device_id: str, **kwargs) -> bool:
        """
        更新设备属性。

        示例:
          registry.update("living_room_light", power=True, brightness=80)
        """
        success = self._backend.update(device_id, **kwargs)
        if not success:
            logger.warning(f"设备更新失败 | device_id={device_id} | kwargs={kwargs}")
        return success

    def get_status_summary(self) -> str:
        """获取所有设备状态的文本摘要"""
        return self._backend.get_status_summary()

    def tick_environment(self) -> None:
        """
        推进一次环境推演，让传感器读数反映执行器的当前状态。

        只应该由传感器读取工具调用。设备控制、场景激活和计划验证
        都不该触发它——否则同一次对话里读到的环境值会随调用次数漂移。
        """
        self._backend.tick_environment()

    def get_device_list_prompt(self) -> str:
        """
        生成可用设备列表（用于系统提示词）。

        执行器和传感器分开列出，因为它们对 LLM 的含义完全不同：
        执行器可以被 control_xxx 控制，传感器只能被 read_sensor 读取。
        如果混在一张清单里，规划器会试图给温湿度传感器下"打开"指令。
        """
        controllable: list[str] = []
        readonly: list[str] = []
        for dev_id, dev in self._backend.get_all().items():
            entry = f"  · {dev.name}（ID: {dev_id}，类型: {dev.device_type.value}）"
            if dev.device_type in SENSOR_DEVICE_TYPES:
                readonly.append(entry)
            else:
                controllable.append(entry)

        lines = ["可控制的设备列表:", *controllable]
        if readonly:
            lines.append("只读传感器列表（只能读取，不能控制）:")
            lines.extend(readonly)
        return "\n".join(lines)
