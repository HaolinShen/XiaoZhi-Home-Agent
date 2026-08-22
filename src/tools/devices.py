"""
设备控制工具
============
将 DeviceRegistry 的操作封装为 LangChain Tool。

每个工具函数遵循同样的设计模式:
  1. 使用 @tool 装饰器标记
  2. 参数类型注解 → 自动生成 LLM 可见的 JSON Schema
  3. docstring → LLM 阅读的工具描述（决定了何时调用、怎么传参）
  4. 返回值 → 工具执行结果文本（会作为 ToolMessage 返回给 LLM）

扩展新设备类型只需:
  1. 在 models.py 中定义 Pydantic 模型
  2. 在 devices/simulator.py 中注册默认设备
  3. 在此文件中添加对应的 @tool 函数
  4. 在 get_all_tools() 中注册

执行器 vs 传感器:
  执行器（灯/空调/电视/窗帘/加湿器）→ control_xxx 工具，可读可写
  传感器（温湿度/人体存在）        → read_sensor 工具，只读
  传感器故意不做成 control_xxx，这样 LLM 从工具名就知道它改不了状态。

关于 config 参数:
  `config: RunnableConfig = None` 的默认值是给类型标注用的，运行时拿不到 None ——
  LangChain 总会注入一个 config，后台自动化执行器那种 `tool.invoke(arguments)`
  的无身份调用注入的是 `configurable` 为空的 config。所以**不要**在这里写
  `if config is not None:` 来"保护"依赖身份的副作用，那个判断恒为真，一个字都拦
  不住（曾因此把定时热水器动作整个判成失败）。身份校验的责任在被调用方内部，
  见 `record_preference_operation()`。
"""

from typing import Optional
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from loguru import logger

from ..models import DeviceType, ACMode, FanSpeed
from ..devices.base import DeviceRegistry
from .memory import record_preference_operation


# ============================================================
# 全局注册中心引用（由 main.py 在启动时注入）
# ============================================================
_registry: Optional[DeviceRegistry] = None


def set_registry(registry: DeviceRegistry) -> None:
    """
    设置全局设备注册中心。

    在 main.py 启动时调用一次，之后所有工具函数共享同一个注册中心。
    这种"依赖注入"模式避免了全局变量和循环导入问题。
    """
    global _registry
    _registry = registry
    logger.info("工具层已注入 DeviceRegistry")


def _get_registry() -> DeviceRegistry:
    """获取注册中心（带检查）"""
    if _registry is None:
        raise RuntimeError(
            "DeviceRegistry 尚未初始化。请在 main.py 中先调用 set_registry()。"
        )
    return _registry


# ============================================================
# 灯光控制工具
# ============================================================

@tool
def control_light(
    device_name: str,
    action: str,
    brightness: int = 50,
    color: str = "暖白",
    config: RunnableConfig = None,
) -> str:
    """
    控制灯光设备。支持打开/关闭、调节亮度、调节色温。

    使用场景:
      "打开客厅灯"              → device_name="客厅灯", action="on"
      "把卧室灯关掉"            → device_name="卧室灯", action="off"
      "把灯光调暗到30%"         → action="set_brightness", brightness=30
      "把灯调成白光"            → action="set_color", color="白光"

    参数:
        device_name: 设备名称，如"客厅灯"、"卧室灯"、"厨房灯"
        action: 操作类型:
                - "on": 打开灯光
                - "off": 关闭灯光
                - "set_brightness": 调节亮度（需配合 brightness 参数）
                - "set_color": 调节色温（需配合 color 参数）
        brightness: 亮度百分比（0-100），默认 50。仅 action="set_brightness" 时有效
        color: 色温描述，如"暖白"、"白光"、"暖黄"。仅 action="set_color" 时有效

    返回:
        执行结果的文本描述。
    """
    registry = _get_registry()

    # 1. 模糊查找设备
    device = registry.find(device_name, DeviceType.LIGHT)
    if device is None:
        return (
            f"❌ 找不到名为「{device_name}」的灯光设备。"
            f"当前可用的灯光有: 客厅灯、卧室灯、厨房灯。"
        )

    # 2. 执行操作
    if action == "on":
        registry.update(device.device_id, power=True)
        return f"✅ {device.name}已打开，当前亮度 {device.brightness}%，色温 {device.color}。"

    elif action == "off":
        registry.update(device.device_id, power=False)
        return f"✅ {device.name}已关闭。"

    elif action == "set_brightness":
        brightness = max(0, min(100, brightness))
        registry.update(device.device_id, brightness=brightness, power=True)
        record_preference_operation(
            config, device.device_id, "lighting.brightness", {"brightness": brightness}
        )
        return f"✅ {device.name}亮度已调至 {brightness}%。"

    elif action == "set_color":
        registry.update(device.device_id, color=color, power=True)
        record_preference_operation(
            config, device.device_id, "lighting.color", {"color": color}
        )
        return f"✅ {device.name}色温已调至「{color}」。"

    else:
        return f"❌ 不支持的操作「{action}」。灯光支持: on / off / set_brightness / set_color"


# ============================================================
# 空调控制工具
# ============================================================

@tool
def control_ac(
    device_name: str,
    action: str,
    temperature: int = 26,
    mode: str = "cool",
    fan_speed: str = "auto",
    config: RunnableConfig = None,
) -> str:
    """
    控制空调设备。支持打开/关闭、调温、切换模式、调节风速。

    使用场景:
      "打开客厅空调，制冷25度"        → action="on", temperature=25, mode="cool"
      "把卧室空调关了"               → action="off"
      "空调温度调到28度"             → action="set_temp", temperature=28
      "客厅空调换到制热模式"          → action="set_mode", mode="heat"
      "风速调高一点"                 → action="set_fan", fan_speed="high"

    参数:
        device_name: 设备名称，如"客厅空调"、"卧室空调"
        action: 操作类型:
                - "on": 打开空调
                - "off": 关闭空调
                - "set_temp": 设置温度（需配合 temperature 参数）
                - "set_mode": 设置模式（需配合 mode 参数）
                - "set_fan": 设置风速（需配合 fan_speed 参数）
        temperature: 目标温度 16-30°C，默认 26。仅 action="set_temp" 或 "on" 时有效
        mode: 运行模式: cool(制冷), heat(制热), fan(送风), dry(除湿)。默认 cool
        fan_speed: 风速: auto(自动), low(低), mid(中), high(高)。默认 auto

    返回:
        执行结果的文本描述。
    """
    registry = _get_registry()
    device = registry.find(device_name, DeviceType.AC)

    if device is None:
        return "❌ 找不到指定的空调设备。当前可用的空调有: 客厅空调、卧室空调。"

    if action == "on":
        valid_modes = {"cool", "heat", "fan", "dry"}
        mode = mode if mode in valid_modes else "cool"
        registry.update(device.device_id, power=True, temperature=temperature, mode=mode, fan_speed=fan_speed)
        mode_cn = {"cool": "制冷", "heat": "制热", "fan": "送风", "dry": "除湿"}
        return f"✅ {device.name}已开启，{mode_cn.get(mode, mode)}模式，目标温度 {temperature}°C。"

    elif action == "off":
        registry.update(device.device_id, power=False)
        return f"✅ {device.name}已关闭。"

    elif action == "set_temp":
        temperature = max(16, min(30, temperature))
        registry.update(device.device_id, temperature=temperature, power=True)
        record_preference_operation(
            config, device.device_id, "ac.temperature", {"temperature": temperature}
        )
        return f"✅ {device.name}温度已设为 {temperature}°C。"

    elif action == "set_mode":
        valid_modes = {"cool", "heat", "fan", "dry"}
        if mode not in valid_modes:
            return f"❌ 无效的模式「{mode}」。支持: cool(制冷), heat(制热), fan(送风), dry(除湿)"
        registry.update(device.device_id, mode=mode, power=True)
        record_preference_operation(
            config, device.device_id, "ac.mode", {"mode": mode}
        )
        mode_cn = {"cool": "制冷", "heat": "制热", "fan": "送风", "dry": "除湿"}
        return f"✅ {device.name}已切换至{mode_cn[mode]}模式。"

    elif action == "set_fan":
        valid_speeds = {"auto", "low", "mid", "high"}
        if fan_speed not in valid_speeds:
            return f"❌ 无效的风速「{fan_speed}」。支持: auto(自动), low(低), mid(中), high(高)"
        registry.update(device.device_id, fan_speed=fan_speed)
        record_preference_operation(
            config, device.device_id, "ac.fan_speed", {"fan_speed": fan_speed}
        )
        speed_cn = {"auto": "自动", "low": "低", "mid": "中", "high": "高"}
        return f"✅ {device.name}风速已设为{speed_cn[fan_speed]}。"

    else:
        return f"❌ 不支持的操作「{action}」。空调支持: on / off / set_temp / set_mode / set_fan"


# ============================================================
# 电视控制工具
# ============================================================

@tool
def control_tv(
    device_name: str,
    action: str,
    volume: int = 30,
    channel: str = "HDMI 1",
    config: RunnableConfig = None,
) -> str:
    """
    控制电视设备。支持打开/关闭、调节音量、静音、切换输入源。

    使用场景:
      "打开电视"              → action="on"
      "电视音量调到50"        → action="set_volume", volume=50
      "电视静音"              → action="mute"
      "切换到HDMI 2"          → action="set_channel", channel="HDMI 2"

    参数:
        device_name: 设备名称，如"客厅电视"
        action: 操作类型:
                - "on": 打开电视
                - "off": 关闭电视
                - "set_volume": 调节音量（需配合 volume 参数）
                - "mute": 切换静音状态
                - "set_channel": 切换输入源（需配合 channel 参数）
        volume: 音量百分比 0-100，默认 30。仅 action="set_volume" 时有效
        channel: 输入源名称。仅 action="set_channel" 时有效

    返回:
        执行结果的文本描述。
    """
    registry = _get_registry()
    device = registry.find(device_name, DeviceType.TV)

    if device is None:
        return "❌ 找不到指定的电视设备。当前可用的电视有: 客厅电视。"

    if action == "on":
        registry.update(device.device_id, power=True)
        return f"✅ {device.name}已打开，音量 {device.volume}%，输入源 {device.channel}。"

    elif action == "off":
        registry.update(device.device_id, power=False)
        return f"✅ {device.name}已关闭。"

    elif action == "set_volume":
        volume = max(0, min(100, volume))
        registry.update(device.device_id, volume=volume)
        record_preference_operation(
            config, device.device_id, "tv.volume", {"volume": volume}
        )
        return f"✅ {device.name}音量已调至 {volume}%。"

    elif action == "mute":
        current_muted = device.muted
        registry.update(device.device_id, muted=not current_muted)
        return f"✅ {device.name}已{'取消静音' if current_muted else '静音'}。"

    elif action == "set_channel":
        registry.update(device.device_id, channel=channel, power=True)
        record_preference_operation(
            config, device.device_id, "tv.channel", {"channel": channel}
        )
        return f"✅ {device.name}已切换至 {channel}。"

    else:
        return f"❌ 不支持的操作「{action}」。电视支持: on / off / set_volume / mute / set_channel"


# ============================================================
# 窗帘控制工具
# ============================================================

@tool
def control_curtain(
    device_name: str,
    action: str,
    percentage: int = 100,
    config: RunnableConfig = None,
) -> str:
    """
    控制窗帘设备。支持完全打开、完全关闭、或调节到指定开合度。

    使用场景:
      "打开客厅窗帘"          → action="open"
      "把卧室窗帘关上"        → action="close"
      "窗帘打开一半"          → action="set_position", percentage=50

    参数:
        device_name: 设备名称，如"客厅窗帘"、"卧室窗帘"
        action: 操作类型:
                - "open": 完全打开窗帘
                - "close": 完全关闭窗帘
                - "set_position": 调节到指定开合度（需配合 percentage 参数）
        percentage: 开合度 0-100（0=全关, 100=全开），默认 100

    返回:
        执行结果的文本描述。
    """
    registry = _get_registry()
    device = registry.find(device_name, DeviceType.CURTAIN)

    if device is None:
        return "❌ 找不到指定的窗帘设备。当前可用的窗帘有: 客厅窗帘、卧室窗帘。"

    if action == "open":
        registry.update(device.device_id, position=100)
        return f"✅ {device.name}已完全打开。"

    elif action == "close":
        registry.update(device.device_id, position=0)
        return f"✅ {device.name}已完全关闭。"

    elif action == "set_position":
        percentage = max(0, min(100, percentage))
        registry.update(device.device_id, position=percentage)
        record_preference_operation(
            config, device.device_id, "curtain.position", {"percentage": percentage}
        )
        if percentage == 0:
            desc = "完全关闭"
        elif percentage == 100:
            desc = "完全打开"
        else:
            desc = f"打开至 {percentage}%"
        return f"✅ {device.name}已{desc}。"

    else:
        return f"❌ 不支持的操作「{action}」。窗帘支持: open / close / set_position"


# ============================================================
# 加湿器控制工具
# ============================================================

@tool
def control_humidifier(
    device_name: str,
    action: str,
    target_humidity: int = 60,
    mist_level: str = "auto",
    config: RunnableConfig = None,
) -> str:
    """控制加湿器的开关、目标湿度和雾量档位。

    参数:
        device_name: 设备名称，如“客厅加湿器”
        action: on / off / set_humidity / set_mist_level
        target_humidity: 目标湿度 30-80%，仅 set_humidity 时使用
        mist_level: auto / low / mid / high，仅 set_mist_level 时使用
    """
    registry = _get_registry()
    device = registry.find(device_name, DeviceType.HUMIDIFIER)
    if device is None:
        return "❌ 找不到指定的加湿器设备。当前可用的加湿器有: 客厅加湿器。"

    if action in {"on", "set_humidity", "set_mist_level"} and device.water_level <= 0:
        return f"❌ {device.name}水箱已空，请加水后再开启。"

    if action == "on":
        registry.update(device.device_id, power=True)
        return (
            f"✅ {device.name}已开启，目标湿度 {device.target_humidity}%，"
            f"雾量{device.mist_level.label_cn}。"
        )

    if action == "off":
        registry.update(device.device_id, power=False)
        return f"✅ {device.name}已关闭。"

    if action == "set_humidity":
        target_humidity = max(30, min(80, target_humidity))
        registry.update(device.device_id, target_humidity=target_humidity, power=True)
        record_preference_operation(
            config,
            device.device_id,
            "humidifier.target_humidity",
            {"target_humidity": target_humidity},
        )
        return f"✅ {device.name}目标湿度已设为 {target_humidity}%。"

    if action == "set_mist_level":
        valid_levels = {"auto", "low", "mid", "high"}
        if mist_level not in valid_levels:
            return "❌ 无效的雾量档位。支持: auto(自动), low(低), mid(中), high(高)"
        registry.update(device.device_id, mist_level=mist_level, power=True)
        record_preference_operation(
            config,
            device.device_id,
            "humidifier.mist_level",
            {"mist_level": mist_level},
        )
        level_cn = {"auto": "自动", "low": "低", "mid": "中", "high": "高"}
        return f"✅ {device.name}雾量已设为{level_cn[mist_level]}。"

    return "❌ 不支持的操作。加湿器支持: on / off / set_humidity / set_mist_level"


# ============================================================
# 电热水器控制工具
# ============================================================

@tool
def control_water_heater(
    device_name: str,
    action: str,
    target_temp: int = 45,
    config: RunnableConfig = None,
) -> str:
    """控制电热水器的开关和目标水温。

    使用场景:
      "打开热水器"        → device_name="卫生间电热水器", action="on"
      "把热水器调到50度"  → action="set_temp", target_temp=50
      "关掉热水器"        → action="off"

    参数:
        device_name: 设备名称，如"卫生间电热水器"
        action: on / off / set_temp
        target_temp: 目标水温 35-75°C，仅 set_temp 或 on 时使用
    """
    registry = _get_registry()
    device = registry.find(device_name, DeviceType.WATER_HEATER)
    if device is None:
        return "❌ 找不到指定的电热水器。当前可用的电热水器有: 卫生间电热水器。"

    if action == "on":
        registry.update(device.device_id, power=True)
        return f"✅ {device.name}已开启，目标水温 {device.target_temp}°C。"
    elif action == "off":
        registry.update(device.device_id, power=False)
        return f"✅ {device.name}已关闭。"
    elif action == "set_temp":
        target_temp = max(35, min(75, target_temp))
        registry.update(device.device_id, target_temp=target_temp, power=True)
        record_preference_operation(
            config,
            device.device_id,
            "water_heater.target_temp",
            {"target_temp": target_temp},
        )
        return f"✅ {device.name}目标水温已设为 {target_temp}°C。"
    return "❌ 不支持的操作。电热水器支持: on / off / set_temp"


# ============================================================
# 智能门锁控制工具
# ============================================================

@tool
def control_lock(
    device_name: str,
    action: str,
    config: RunnableConfig = None,
) -> str:
    """控制智能门锁的上锁与解锁。

    安全提示: 解锁属于对外敏感动作，会触发人工确认，需要用户批准后才真正执行。

    使用场景:
      "把门锁上"      → device_name="玄关门锁", action="lock"
      "解锁门锁"      → action="unlock"（需人工审批）

    参数:
        device_name: 设备名称，如"玄关门锁"
        action: lock(上锁) / unlock(解锁)
    """
    registry = _get_registry()
    device = registry.find(device_name, DeviceType.LOCK)
    if device is None:
        return "❌ 找不到指定的门锁。当前可用的门锁有: 玄关门锁。"
    if not device.power:
        return f"❌ {device.name}离线，无法操作。"

    if action == "lock":
        registry.update(device.device_id, locked=True)
        return f"✅ {device.name}已上锁。"
    elif action == "unlock":
        registry.update(device.device_id, locked=False)
        return f"✅ {device.name}已解锁。"
    return "❌ 不支持的操作。门锁支持: lock / unlock"


# ============================================================
# 电热水壶控制工具
# ============================================================

@tool
def control_kettle(
    device_name: str,
    action: str,
    target_temp: int = 100,
    config: RunnableConfig = None,
) -> str:
    """控制电热水壶的开关、目标水温和"烧开"动作。

    使用场景:
      "把水烧开"      → action="boil"
      "烧水到80度"    → action="set_temp", target_temp=80
      "打开烧水壶"    → action="on"
      "关掉烧水壶"    → action="off"

    参数:
        device_name: 设备名称，如"厨房烧水壶"
        action: on / off / set_temp / boil（boil 一键开机并加热到 100°C）
        target_temp: 目标水温 40-100°C
    """
    registry = _get_registry()
    device = registry.find(device_name, DeviceType.KETTLE)
    if device is None:
        return "❌ 找不到指定的电热水壶。当前可用的烧水壶有: 厨房烧水壶。"

    if action == "boil":
        registry.update(device.device_id, power=True, target_temp=100)
        return f"✅ {device.name}已开始烧水，加热至 100°C。"
    elif action == "on":
        registry.update(device.device_id, power=True)
        return f"✅ {device.name}已开启，目标水温 {device.target_temp}°C。"
    elif action == "off":
        registry.update(device.device_id, power=False)
        return f"✅ {device.name}已关闭。"
    elif action == "set_temp":
        target_temp = max(40, min(100, target_temp))
        registry.update(device.device_id, target_temp=target_temp, power=True)
        record_preference_operation(
            config,
            device.device_id,
            "kettle.target_temp",
            {"target_temp": target_temp},
        )
        return f"✅ {device.name}目标水温已设为 {target_temp}°C。"
    return "❌ 不支持的操作。烧水壶支持: on / off / set_temp / boil"


# ============================================================
# 传感器读取工具（只读）
# ============================================================
#
# 传感器只有"读"这一个动作，所以不做成 control_xxx，而是单独一个
# read_sensor。这样 LLM 从工具名就能看出它改不了任何东西，
# 也不会误以为可以"打开温湿度传感器"。

@tool
def read_sensor(sensor_type: str, location: str = "") -> str:
    """
    读取环境传感器的当前数值。控制设备前先用它了解实际情况。

    使用场景:
      "现在屋里多少度"          → sensor_type="temp_humidity"
      "客厅湿度怎么样"          → sensor_type="temp_humidity", location="客厅"
      "家里有人吗"              → sensor_type="presence"
      "玄关有人经过吗"          → sensor_type="presence", location="玄关"

    什么时候应该主动调用:
      · 用户说"有点干"、"有点热"这类主观感受 → 先读数值再决定开什么、开多大
      · 执行离家模式这类批量操作前 → 先确认家里没人
      · 用户问"要不要开加湿器" → 先读湿度再给建议

    参数:
        sensor_type: 传感器类型:
                     - "temp_humidity": 温湿度传感器（温度和湿度）
                     - "presence": 人体存在传感器（有人/无人）
        location: 可选房间名，如"客厅"、"卧室"、"玄关"。留空则返回该类型全部传感器

    返回:
        传感器读数的文本描述。传感器不存在时返回可用房间提示。
    """
    registry = _get_registry()

    type_map = {
        "temp_humidity": DeviceType.TEMP_HUMIDITY_SENSOR,
        "presence": DeviceType.PRESENCE_SENSOR,
    }
    device_type = type_map.get(sensor_type)
    if device_type is None:
        return (
            f"❌ 不支持的传感器类型「{sensor_type}」。"
            f"支持: temp_humidity(温湿度), presence(人体存在)"
        )

    # 读取前推进一次环境推演，让读数反映执行器的当前状态。
    # 只有"读环境"的入口才该这么做（这里、get_device_status、并行查询子图的
    # dispatch）；控制和计划验证路径都不该触发它，否则同一次对话里读到的值
    # 会随调用次数漂移。
    registry.tick_environment()

    sensors = registry.get_by_type(device_type)
    if not sensors:
        return f"❌ 家里没有安装{device_type.label_cn}。"

    wanted = location.strip()
    if wanted:
        sensors = {
            dev_id: dev for dev_id, dev in sensors.items()
            if wanted in dev.location or wanted in dev.name
        }
        if not sensors:
            available = "、".join(
                dev.location or dev.name
                for dev in registry.get_by_type(device_type).values()
            )
            return (
                f"❌ 「{wanted}」没有{device_type.label_cn}。"
                f"已安装的位置: {available}。"
            )

    logger.debug(f"读取传感器 | type={sensor_type} | location={location}")
    lines = [f"📡 **{device_type.label_cn}读数:**"]
    lines.extend(f"  · {dev.to_status_text()}" for dev in sensors.values())
    return "\n".join(lines)


# ============================================================
# 设备状态查询工具
# ============================================================

@tool
def get_device_status(query: str = "") -> str:
    """
    查询所有智能家居设备的当前状态（含传感器读数）。

    无需指定参数即可查看全部设备状态。
    也可以指定类型关键词来筛选，如"灯光"、"空调"。

    参数:
        query: 可选筛选词（如"灯光"只查看灯光状态）。留空返回全部设备。

    返回:
        格式化设备状态报告。
    """
    registry = _get_registry()
    _ = query  # 保留参数给未来扩展（按类型筛选）
    # 这是一次显式的"看一眼环境"，所以先推演传感器读数。
    registry.tick_environment()
    logger.debug(f"查询设备状态 | query={query}")
    return registry.get_status_summary()
