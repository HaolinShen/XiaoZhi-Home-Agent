"""设备能力的单一数据源（Single Source of Truth）。

为什么要有这个模块
====================
以前"新增一种设备"要手工改 9 处（models、simulator 默认实例、base 的 keywords_map、
tools/devices 的 if/elif、tools/__init__、graph 的 device_tool_names、planning 的
DEVICE_ACTION_SPECS 与 PlanStep 的 Literal、mcp/server、scenes 的类型清单），其中
工具实现的 if/elif 和 PlanStep 的 Literal 无法反射，漏改一处的表现是
"Planner 第一版计划稳定失败，且不报错"。

现在所有派生视图都从这里生成，新增设备 = 在 CAPABILITIES 里加一条声明：

  - 控制工具的 JSON Schema / docstring / 参数（tools/devices.build_device_tools）
  - Planner 合法 action 词表与期望状态（agent/planning 的 DEVICE_ACTION_SPECS）
  - PlanStep.tool_name 的 Literal（agent/planning，用 Literal[tuple(...)] 派生）
  - registry.find 的类型关键词（devices/base.py）
  - 模拟器默认设备（devices/simulator.py）
  - 场景批量关闭的设备类型集合（tools/scenes.py 的 scene_exit）
  - 自动化允许的工具名（automation/planning 的 AutomationToolName）

一致性由 tests/test_capabilities.py 的生成式断言兜底：只要声明和派生点
任何一处失配，测试阶段就会失败，而不是运行期静默。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..models import (
    ACDevice,
    ACMode,
    CurtainDevice,
    DeviceType,
    FanSpeed,
    HumidifierDevice,
    KettleDevice,
    LightDevice,
    LockDevice,
    PresenceSensor,
    TempHumiditySensor,
    TVDevice,
    WaterHeaterDevice,
)

# ============================================================
# 声明结构
# ============================================================

# handler 的返回值：(结果文本, 生效后的参数)。
# 第二个元素只在动作成功后非空，用于偏好观察（拿到 clamp 后的真实值，
# 而不是模型写的原始值）；失败路径返回 None，绝不记录偏好。
HandlerResult = tuple[str, dict[str, Any] | None]

# handler 签名：(registry, device, args) -> HandlerResult
#   args 是"工具级参数名 → 值"的字典，值已经过 schema 校验。
Handler = Callable[..., HandlerResult]

# expected 签名：(arguments, device) -> 期望状态字典。
# 与 handler 的差异：handler 写真实副作用，expected 只描述"做完之后设备应该是什么样"，
# Verifier 拿它跟注册中心的真实状态比对。mute 这类翻转语义必须读 device。
Expected = Callable[[dict[str, Any], Any], dict[str, Any]]


@dataclass(frozen=True)
class ParamSpec:
    """工具级参数声明，自动进入工具 JSON Schema 与 docstring。"""

    name: str
    annotation: type
    default: Any
    description: str


@dataclass(frozen=True)
class PreferenceSpec:
    """动作成功后要记录的行为观察（重复操作 → 偏好候选）。"""

    memory_key: str
    value_from: Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class ActionSpec:
    """一个 action 的完整声明：喂 Planner 的文本、工具执行逻辑、期望状态、偏好观察。"""

    name: str
    signature: str  # 喂给 Planner 的合法值文本，括号内为该 action 需附带的参数
    doc: str  # 工具 docstring 里对该 action 的一行说明
    expected: Expected
    handler: Handler
    precheck: Callable[[Any, dict[str, Any]], str | None] | None = None
    preference: PreferenceSpec | None = None


@dataclass(frozen=True)
class DeviceCapability:
    """一个可控制设备类型的全部声明。"""

    device_type: DeviceType
    tool_name: str  # control_xxx
    device_label: str  # 中文类别名（"灯光"），用于错误提示与 docstring
    tool_summary: str  # docstring 首行
    usage_examples: tuple[str, ...]
    device_examples: str  # 设备名示例，写进 device_name 参数说明
    not_found_text: str  # 找不到设备时的提示，{device_name} 为占位符
    common_params: tuple[ParamSpec, ...]
    actions: tuple[ActionSpec, ...]
    default_devices: tuple[tuple[type, dict], ...]  # (模型类, kwargs)，模拟器据此注册
    # 离家/睡眠批量操作时该类型设备怎么处理；None 表示不参与批量开关。
    scene_exit: str | None  # "power_off" | "curtain_close" | "lock"


# ============================================================
# 工具函数（供 action handler 使用）
# ============================================================


def _clamp(value: Any, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


def _int_arg(args: dict[str, Any], name: str, default: int, low: int, high: int) -> int:
    """读取整数参数并夹到合法区间，无法解析时报错而不是崩溃。

    这里必须容错：`expected_state_for_step()` 在 executor 的 try 块之外被调用，
    模型只要写出 brightness="很亮" 这种值，裸 int() 抛出的 ValueError 就会掀翻
    整张图。转成 preparation_error 后会被判成确定性错误直接 replan —— 既不崩，
    也不会静默套用默认值把"调到很亮"悄悄执行成"调到 50%"再报告成功。
    """
    raw = args.get(name, default)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise InvalidPlanArgument(
            f"invalid argument: {name} 需要 {low}-{high} 之间的整数，收到 {raw!r}"
        ) from None
    return max(low, min(high, value))


class InvalidPlanArgument(ValueError):
    """计划里的参数无法解析成设备可接受的值。

    单独立一个异常类型，是为了把"参数写错"和真正的程序 bug 分开：前者应该变成
    preparation_error 回喂给 Planner 重写，后者才该往上抛。
    """


# ============================================================
# 各设备的 action handler
# ============================================================

_LABEL_CN = {"cool": "制冷", "heat": "制热", "fan": "送风", "dry": "除湿"}


def _light_handlers() -> dict[str, Handler]:
    def on(registry, device, args):
        registry.update(device.device_id, power=True)
        return f"✅ {device.name}已打开，当前亮度 {device.brightness}%，色温 {device.color}。", None

    def off(registry, device, args):
        registry.update(device.device_id, power=False)
        return f"✅ {device.name}已关闭。", None

    def set_brightness(registry, device, args):
        brightness = _clamp(args.get("brightness", 50), 0, 100)
        registry.update(device.device_id, brightness=brightness, power=True)
        return f"✅ {device.name}亮度已调至 {brightness}%。", {"brightness": brightness}

    def set_color(registry, device, args):
        color = args.get("color", "暖白")
        registry.update(device.device_id, color=color, power=True)
        return f"✅ {device.name}色温已调至「{color}」。", {"color": color}

    return {"on": on, "off": off, "set_brightness": set_brightness, "set_color": set_color}


_LIGHT_HANDLERS = _light_handlers()


def _ac_handlers() -> dict[str, Handler]:
    valid_modes = {"cool", "heat", "fan", "dry"}
    valid_speeds = {"auto", "low", "mid", "high"}

    def on(registry, device, args):
        mode = args.get("mode", "cool")
        mode = mode if mode in valid_modes else "cool"
        registry.update(
            device.device_id,
            power=True,
            temperature=args.get("temperature", 26),
            mode=mode,
            fan_speed=args.get("fan_speed", "auto"),
        )
        return (
            f"✅ {device.name}已开启，{_LABEL_CN.get(mode, mode)}模式，"
            f"目标温度 {args.get('temperature', 26)}°C。",
            None,
        )

    def off(registry, device, args):
        registry.update(device.device_id, power=False)
        return f"✅ {device.name}已关闭。", None

    def set_temp(registry, device, args):
        temperature = _clamp(args.get("temperature", 26), 16, 30)
        registry.update(device.device_id, temperature=temperature, power=True)
        return f"✅ {device.name}温度已设为 {temperature}°C。", {"temperature": temperature}

    def set_mode(registry, device, args):
        mode = args.get("mode", "cool")
        if mode not in valid_modes:
            return (
                f"❌ 无效的模式「{mode}」。支持: cool(制冷), heat(制热), fan(送风), dry(除湿)",
                None,
            )
        registry.update(device.device_id, mode=mode, power=True)
        return f"✅ {device.name}已切换至{_LABEL_CN[mode]}模式。", {"mode": mode}

    def set_fan(registry, device, args):
        fan_speed = args.get("fan_speed", "auto")
        if fan_speed not in valid_speeds:
            return f"❌ 无效的风速「{fan_speed}」。支持: auto(自动), low(低), mid(中), high(高)", None
        registry.update(device.device_id, fan_speed=fan_speed)
        speed_cn = {"auto": "自动", "low": "低", "mid": "中", "high": "高"}
        return f"✅ {device.name}风速已设为{speed_cn[fan_speed]}。", {"fan_speed": fan_speed}

    return {"on": on, "off": off, "set_temp": set_temp, "set_mode": set_mode, "set_fan": set_fan}


_AC_HANDLERS = _ac_handlers()


def _tv_handlers() -> dict[str, Handler]:
    def on(registry, device, args):
        registry.update(device.device_id, power=True)
        return f"✅ {device.name}已打开，音量 {device.volume}%，输入源 {device.channel}。", None

    def off(registry, device, args):
        registry.update(device.device_id, power=False)
        return f"✅ {device.name}已关闭。", None

    def set_volume(registry, device, args):
        volume = _clamp(args.get("volume", 30), 0, 100)
        registry.update(device.device_id, volume=volume)
        return f"✅ {device.name}音量已调至 {volume}%。", {"volume": volume}

    def mute(registry, device, args):
        current_muted = device.muted
        registry.update(device.device_id, muted=not current_muted)
        return f"✅ {device.name}已{'取消静音' if current_muted else '静音'}。", None

    def set_channel(registry, device, args):
        channel = args.get("channel", "HDMI 1")
        registry.update(device.device_id, channel=channel, power=True)
        return f"✅ {device.name}已切换至 {channel}。", {"channel": channel}

    return {"on": on, "off": off, "set_volume": set_volume, "mute": mute, "set_channel": set_channel}


_TV_HANDLERS = _tv_handlers()


def _curtain_handlers() -> dict[str, Handler]:
    def open_(registry, device, args):
        registry.update(device.device_id, position=100)
        return f"✅ {device.name}已完全打开。", None

    def close(registry, device, args):
        registry.update(device.device_id, position=0)
        return f"✅ {device.name}已完全关闭。", None

    def set_position(registry, device, args):
        percentage = _clamp(args.get("percentage", 100), 0, 100)
        registry.update(device.device_id, position=percentage)
        if percentage == 0:
            desc = "完全关闭"
        elif percentage == 100:
            desc = "完全打开"
        else:
            desc = f"打开至 {percentage}%"
        return f"✅ {device.name}已{desc}。", {"percentage": percentage}

    return {"open": open_, "close": close, "set_position": set_position}


_CURTAIN_HANDLERS = _curtain_handlers()


def _humidifier_handlers() -> dict[str, Callable[..., Any]]:
    valid_levels = {"auto", "low", "mid", "high"}

    def empty_tank(device, args) -> str | None:
        if device.water_level <= 0:
            return f"❌ {device.name}水箱已空，请加水后再开启。"
        return None

    def on(registry, device, args):
        registry.update(device.device_id, power=True)
        return (
            f"✅ {device.name}已开启，目标湿度 {device.target_humidity}%，"
            f"雾量{device.mist_level.label_cn}。",
            None,
        )

    def off(registry, device, args):
        registry.update(device.device_id, power=False)
        return f"✅ {device.name}已关闭。", None

    def set_humidity(registry, device, args):
        target_humidity = _clamp(args.get("target_humidity", 60), 30, 80)
        registry.update(device.device_id, target_humidity=target_humidity, power=True)
        return f"✅ {device.name}目标湿度已设为 {target_humidity}%。", {"target_humidity": target_humidity}

    def set_mist_level(registry, device, args):
        mist_level = args.get("mist_level", "auto")
        if mist_level not in valid_levels:
            return "❌ 无效的雾量档位。支持: auto(自动), low(低), mid(中), high(高)", None
        registry.update(device.device_id, mist_level=mist_level, power=True)
        level_cn = {"auto": "自动", "low": "低", "mid": "中", "high": "高"}
        return f"✅ {device.name}雾量已设为{level_cn[mist_level]}。", {"mist_level": mist_level}

    return {
        "on": on, "off": off, "set_humidity": set_humidity, "set_mist_level": set_mist_level,
        "empty_tank": empty_tank,
    }


_HUMIDIFIER_HANDLERS = _humidifier_handlers()


def _water_heater_handlers() -> dict[str, Handler]:
    def on(registry, device, args):
        registry.update(device.device_id, power=True)
        return f"✅ {device.name}已开启，目标水温 {device.target_temp}°C。", None

    def off(registry, device, args):
        registry.update(device.device_id, power=False)
        return f"✅ {device.name}已关闭。", None

    def set_temp(registry, device, args):
        target_temp = _clamp(args.get("target_temp", 45), 35, 75)
        registry.update(device.device_id, target_temp=target_temp, power=True)
        return f"✅ {device.name}目标水温已设为 {target_temp}°C。", {"target_temp": target_temp}

    return {"on": on, "off": off, "set_temp": set_temp}


_WATER_HEATER_HANDLERS = _water_heater_handlers()


def _lock_handlers() -> dict[str, Callable[..., Any]]:
    def offline(device, args) -> str | None:
        if not device.power:
            return f"❌ {device.name}离线，无法操作。"
        return None

    def lock(registry, device, args):
        registry.update(device.device_id, locked=True)
        return f"✅ {device.name}已上锁。", None

    def unlock(registry, device, args):
        registry.update(device.device_id, locked=False)
        return f"✅ {device.name}已解锁。", None

    return {"lock": lock, "unlock": unlock, "offline": offline}


_LOCK_HANDLERS = _lock_handlers()


def _kettle_handlers() -> dict[str, Handler]:
    def boil(registry, device, args):
        registry.update(device.device_id, power=True, target_temp=100)
        return f"✅ {device.name}已开始烧水，加热至 100°C。", None

    def on(registry, device, args):
        registry.update(device.device_id, power=True)
        return f"✅ {device.name}已开启，目标水温 {device.target_temp}°C。", None

    def off(registry, device, args):
        registry.update(device.device_id, power=False)
        return f"✅ {device.name}已关闭。", None

    def set_temp(registry, device, args):
        target_temp = _clamp(args.get("target_temp", 100), 40, 100)
        registry.update(device.device_id, target_temp=target_temp, power=True)
        return f"✅ {device.name}目标水温已设为 {target_temp}°C。", {"target_temp": target_temp}

    return {"boil": boil, "on": on, "off": off, "set_temp": set_temp}


_KETTLE_HANDLERS = _kettle_handlers()


# ============================================================
# 能力清单（新增设备只改这里）
# ============================================================

CAPABILITIES: tuple[DeviceCapability, ...] = (
    DeviceCapability(
        device_type=DeviceType.LIGHT,
        tool_name="control_light",
        device_label="灯光",
        tool_summary="控制灯光设备。支持打开/关闭、调节亮度、调节色温。",
        usage_examples=(
            '  "打开客厅灯"              → device_name="客厅灯", action="on"',
            '  "把卧室灯关掉"            → device_name="卧室灯", action="off"',
            '  "把灯光调暗到30%"         → action="set_brightness", brightness=30',
            '  "把灯调成白光"            → action="set_color", color="白光"',
        ),
        device_examples='"客厅灯"、"卧室灯"、"厨房灯"',
        not_found_text=(
            "❌ 找不到名为「{device_name}」的灯光设备。"
            "当前可用的灯光有: 客厅灯、卧室灯、厨房灯。"
        ),
        common_params=(
            ParamSpec("brightness", int, 50, "亮度百分比（0-100），默认 50。仅 action=set_brightness 时有效"),
            ParamSpec("color", str, "暖白", '色温描述，如"暖白"、"白光"、"暖黄"。仅 action=set_color 时有效'),
        ),
        actions=(
            ActionSpec("on", "on", "打开灯光", lambda args, device: {"power": True}, _LIGHT_HANDLERS["on"]),
            ActionSpec("off", "off", "关闭灯光", lambda args, device: {"power": False}, _LIGHT_HANDLERS["off"]),
            ActionSpec(
                "set_brightness",
                "set_brightness(brightness)",
                "调节亮度（需配合 brightness 参数）",
                lambda args, device: {
                    "power": True,
                    "brightness": _int_arg(args, "brightness", 50, 0, 100),
                },
                _LIGHT_HANDLERS["set_brightness"],
                preference=PreferenceSpec(
                    "lighting.brightness", lambda a: {"brightness": a["brightness"]}
                ),
            ),
            ActionSpec(
                "set_color",
                "set_color(color)",
                "调节色温（需配合 color 参数）",
                lambda args, device: {"power": True, "color": args.get("color", "暖白")},
                _LIGHT_HANDLERS["set_color"],
                preference=PreferenceSpec("lighting.color", lambda a: {"color": a["color"]}),
            ),
        ),
        default_devices=(
            # 客厅灯**刻意不填 model**。013 给全部设备类型补齐了说明书，
            # 但必须留一台没登记型号的设备，否则"型号未登记 → 知识检索拒答"
            # 这条路径在演示和测试里都不可见了（tests/test_knowledge_rag.py 有两条
            # 用例直接钉着「客厅灯」走 no_model 分支）。留灯而不是留窗帘，
            # 是因为"家里有一盏没牌子的老灯"最贴近真实住宅。
            (LightDevice, {"device_id": "living_room_light", "name": "客厅灯", "location": "客厅", "brightness": 80, "color": "暖白"}),
            (LightDevice, {"device_id": "bedroom_light", "name": "卧室灯", "location": "卧室", "brightness": 60, "color": "暖白", "model": "GlowSoft-L90"}),
            (LightDevice, {"device_id": "kitchen_light", "name": "厨房灯", "location": "厨房", "brightness": 100, "color": "白光", "model": "LumiCore-L200"}),
        ),
        scene_exit="power_off",
    ),
    DeviceCapability(
        device_type=DeviceType.AC,
        tool_name="control_ac",
        device_label="空调",
        tool_summary="控制空调设备。支持打开/关闭、调温、切换模式、调节风速。",
        usage_examples=(
            '  "打开客厅空调，制冷25度"        → action="on", temperature=25, mode="cool"',
            '  "把卧室空调关了"               → action="off"',
            '  "空调温度调到28度"             → action="set_temp", temperature=28',
            '  "客厅空调换到制热模式"          → action="set_mode", mode="heat"',
            '  "风速调高一点"                 → action="set_fan", fan_speed="high"',
        ),
        device_examples='"客厅空调"、"卧室空调"',
        not_found_text="❌ 找不到指定的空调设备。当前可用的空调有: 客厅空调、卧室空调。",
        common_params=(
            ParamSpec("temperature", int, 26, "目标温度 16-30°C，默认 26。仅 action=set_temp 或 on 时有效"),
            ParamSpec("mode", str, "cool", "运行模式: cool(制冷), heat(制热), fan(送风), dry(除湿)。默认 cool"),
            ParamSpec("fan_speed", str, "auto", "风速: auto(自动), low(低), mid(中), high(高)。默认 auto"),
        ),
        actions=(
            ActionSpec(
                "on",
                "on(可带 temperature、mode)",
                "打开空调",
                lambda args, device: {
                    "power": True,
                    "temperature": _int_arg(args, "temperature", 26, 16, 30),
                    "mode": args.get("mode", "cool"),
                },
                _AC_HANDLERS["on"],
            ),
            ActionSpec("off", "off", "关闭空调", lambda args, device: {"power": False}, _AC_HANDLERS["off"]),
            ActionSpec(
                "set_temp",
                "set_temp(temperature)",
                "设置温度（需配合 temperature 参数）",
                lambda args, device: {
                    "power": True,
                    "temperature": _int_arg(args, "temperature", 26, 16, 30),
                },
                _AC_HANDLERS["set_temp"],
                preference=PreferenceSpec("ac.temperature", lambda a: {"temperature": a["temperature"]}),
            ),
            ActionSpec(
                "set_mode",
                "set_mode(mode)",
                "设置模式（需配合 mode 参数）",
                lambda args, device: {"power": True, "mode": args.get("mode", "cool")},
                _AC_HANDLERS["set_mode"],
                preference=PreferenceSpec("ac.mode", lambda a: {"mode": a["mode"]}),
            ),
            ActionSpec(
                "set_fan",
                "set_fan(fan_speed)",
                "设置风速（需配合 fan_speed 参数）",
                lambda args, device: {"fan_speed": args.get("fan_speed", "auto")},
                _AC_HANDLERS["set_fan"],
                preference=PreferenceSpec("ac.fan_speed", lambda a: {"fan_speed": a["fan_speed"]}),
            ),
        ),
        default_devices=(
            # 两台空调刻意给了不同型号。真实家庭很少两台空调同款（不同年份买的），
            # 而这个差异对说明书检索是决定性的：SmartCool-AC2024 的 E4 是室内外机通信异常，
            # FrostLine-AC310 的 E4 是排水泵异常——同一个代码，两套完全不同的处理步骤。
            # 所以用户只说"空调显示 E4"时，系统必须问是哪一台，而不能挑一台猜。
            # 若两台同型号，这条约束在演示和测试里都不可见，型号过滤就成了摆设。
            (ACDevice, {"device_id": "living_room_ac", "name": "客厅空调", "location": "客厅", "temperature": 26, "mode": ACMode.COOL, "fan_speed": FanSpeed.AUTO, "model": "SmartCool-AC2024"}),
            (ACDevice, {"device_id": "bedroom_ac", "name": "卧室空调", "location": "卧室", "temperature": 26, "mode": ACMode.COOL, "fan_speed": FanSpeed.AUTO, "model": "FrostLine-AC310"}),
        ),
        scene_exit="power_off",
    ),
    DeviceCapability(
        device_type=DeviceType.TV,
        tool_name="control_tv",
        device_label="电视",
        tool_summary="控制电视设备。支持打开/关闭、调节音量、静音、切换输入源。",
        usage_examples=(
            '  "打开电视"              → action="on"',
            '  "电视音量调到50"        → action="set_volume", volume=50',
            '  "电视静音"              → action="mute"',
            '  "切换到HDMI 2"          → action="set_channel", channel="HDMI 2"',
        ),
        device_examples='"客厅电视"',
        not_found_text="❌ 找不到指定的电视设备。当前可用的电视有: 客厅电视。",
        common_params=(
            ParamSpec("volume", int, 30, "音量百分比 0-100，默认 30。仅 action=set_volume 时有效"),
            ParamSpec("channel", str, "HDMI 1", "输入源名称。仅 action=set_channel 时有效"),
        ),
        actions=(
            ActionSpec("on", "on", "打开电视", lambda args, device: {"power": True}, _TV_HANDLERS["on"]),
            ActionSpec("off", "off", "关闭电视", lambda args, device: {"power": False}, _TV_HANDLERS["off"]),
            ActionSpec(
                "set_volume",
                "set_volume(volume)",
                "调节音量（需配合 volume 参数）",
                lambda args, device: {"volume": _int_arg(args, "volume", 30, 0, 100)},
                _TV_HANDLERS["set_volume"],
                preference=PreferenceSpec("tv.volume", lambda a: {"volume": a["volume"]}),
            ),
            ActionSpec(
                # mute 是翻转语义，所以期望状态依赖执行前的设备状态。
                "mute",
                "mute",
                "切换静音状态",
                lambda args, device: {"muted": not device.muted},
                _TV_HANDLERS["mute"],
            ),
            ActionSpec(
                "set_channel",
                "set_channel(channel)",
                "切换输入源（需配合 channel 参数）",
                lambda args, device: {"power": True, "channel": args.get("channel", "HDMI 1")},
                _TV_HANDLERS["set_channel"],
                preference=PreferenceSpec("tv.channel", lambda a: {"channel": a["channel"]}),
            ),
        ),
        default_devices=(
            # model 是说明书检索的唯一数据源，必须与 docs/knowledge/catalog.json 的
            # model 字段**逐字相等**。差一个字符（大小写、连字符）不会报错，
            # 只会让型号过滤全部落空，表现为"明明有说明书却一直拒答"。
            (TVDevice, {"device_id": "living_room_tv", "name": "客厅电视", "location": "客厅", "volume": 30, "channel": "HDMI 1", "model": "VisionTV-V1"}),
        ),
        scene_exit="power_off",
    ),
    DeviceCapability(
        device_type=DeviceType.CURTAIN,
        tool_name="control_curtain",
        device_label="窗帘",
        tool_summary="控制窗帘设备。支持完全打开、完全关闭、或调节到指定开合度。",
        usage_examples=(
            '  "打开客厅窗帘"          → action="open"',
            '  "把卧室窗帘关上"        → action="close"',
            '  "窗帘打开一半"          → action="set_position", percentage=50',
        ),
        device_examples='"客厅窗帘"、"卧室窗帘"',
        not_found_text="❌ 找不到指定的窗帘设备。当前可用的窗帘有: 客厅窗帘、卧室窗帘。",
        common_params=(
            ParamSpec("percentage", int, 100, "开合度 0-100（0=全关, 100=全开），默认 100"),
        ),
        actions=(
            ActionSpec("open", "open", "完全打开窗帘", lambda args, device: {"position": 100}, _CURTAIN_HANDLERS["open"]),
            ActionSpec("close", "close", "完全关闭窗帘", lambda args, device: {"position": 0}, _CURTAIN_HANDLERS["close"]),
            ActionSpec(
                "set_position",
                "set_position(percentage)",
                "调节到指定开合度（需配合 percentage 参数）",
                lambda args, device: {"position": _int_arg(args, "percentage", 100, 0, 100)},
                _CURTAIN_HANDLERS["set_position"],
                preference=PreferenceSpec("curtain.position", lambda a: {"percentage": a["percentage"]}),
            ),
        ),
        default_devices=(
            # 两台窗帘刻意用不同型号，理由同两台空调：同型号的话"型号过滤"
            # 这条约束在演示和测试里都不可见，等于摆设。
            (CurtainDevice, {"device_id": "living_room_curtain", "name": "客厅窗帘", "location": "客厅", "position": 0, "model": "SilkRail-C100"}),
            (CurtainDevice, {"device_id": "bedroom_curtain", "name": "卧室窗帘", "location": "卧室", "position": 0, "model": "QuietTrack-C60"}),
        ),
        scene_exit="curtain_close",
    ),
    DeviceCapability(
        device_type=DeviceType.HUMIDIFIER,
        tool_name="control_humidifier",
        device_label="加湿器",
        tool_summary="控制加湿器的开关、目标湿度和雾量档位。",
        usage_examples=(
            '  "打开加湿器"            → action="on"',
            '  "湿度调到70%"           → action="set_humidity", target_humidity=70',
            '  "雾量调高"              → action="set_mist_level", mist_level="high"',
        ),
        device_examples='"客厅加湿器"',
        not_found_text="❌ 找不到指定的加湿器设备。当前可用的加湿器有: 客厅加湿器。",
        common_params=(
            ParamSpec("target_humidity", int, 60, "目标湿度 30-80%，仅 set_humidity 时使用"),
            ParamSpec("mist_level", str, "auto", "auto / low / mid / high，仅 set_mist_level 时使用"),
        ),
        actions=(
            ActionSpec(
                "on", "on", "打开加湿器", lambda args, device: {"power": True},
                _HUMIDIFIER_HANDLERS["on"], precheck=_HUMIDIFIER_HANDLERS["empty_tank"],
            ),
            ActionSpec("off", "off", "关闭加湿器", lambda args, device: {"power": False}, _HUMIDIFIER_HANDLERS["off"]),
            ActionSpec(
                "set_humidity",
                "set_humidity(target_humidity)",
                "设置目标湿度（需配合 target_humidity 参数）",
                lambda args, device: {
                    "power": True,
                    "target_humidity": _int_arg(args, "target_humidity", 60, 30, 80),
                },
                _HUMIDIFIER_HANDLERS["set_humidity"],
                precheck=_HUMIDIFIER_HANDLERS["empty_tank"],
                preference=PreferenceSpec(
                    "humidifier.target_humidity", lambda a: {"target_humidity": a["target_humidity"]}
                ),
            ),
            ActionSpec(
                "set_mist_level",
                "set_mist_level(mist_level)",
                "设置雾量档位（需配合 mist_level 参数）",
                lambda args, device: {"power": True, "mist_level": args.get("mist_level", "auto")},
                _HUMIDIFIER_HANDLERS["set_mist_level"],
                precheck=_HUMIDIFIER_HANDLERS["empty_tank"],
                preference=PreferenceSpec("humidifier.mist_level", lambda a: {"mist_level": a["mist_level"]}),
            ),
        ),
        default_devices=(
            (HumidifierDevice, {"device_id": "living_room_humidifier", "name": "客厅加湿器", "location": "客厅", "target_humidity": 60, "mist_level": FanSpeed.AUTO, "water_level": 100, "model": "MistPure-H50"}),
        ),
        scene_exit="power_off",
    ),
    DeviceCapability(
        device_type=DeviceType.WATER_HEATER,
        tool_name="control_water_heater",
        device_label="电热水器",
        tool_summary="控制电热水器的开关和目标水温。",
        usage_examples=(
            '  "打开热水器"        → device_name="卫生间电热水器", action="on"',
            '  "把热水器调到50度"  → action="set_temp", target_temp=50',
            '  "关掉热水器"        → action="off"',
        ),
        device_examples='"卫生间电热水器"',
        not_found_text="❌ 找不到指定的电热水器。当前可用的电热水器有: 卫生间电热水器。",
        common_params=(
            ParamSpec("target_temp", int, 45, "目标水温 35-75°C，仅 set_temp 或 on 时使用"),
        ),
        actions=(
            ActionSpec("on", "on", "打开电热水器", lambda args, device: {"power": True}, _WATER_HEATER_HANDLERS["on"]),
            ActionSpec("off", "off", "关闭电热水器", lambda args, device: {"power": False}, _WATER_HEATER_HANDLERS["off"]),
            ActionSpec(
                "set_temp",
                "set_temp(target_temp)",
                "设置目标水温（需配合 target_temp 参数）",
                lambda args, device: {
                    "power": True,
                    "target_temp": _int_arg(args, "target_temp", 45, 35, 75),
                },
                _WATER_HEATER_HANDLERS["set_temp"],
                preference=PreferenceSpec("water_heater.target_temp", lambda a: {"target_temp": a["target_temp"]}),
            ),
        ),
        default_devices=(
            (WaterHeaterDevice, {"device_id": "bathroom_water_heater", "name": "卫生间电热水器", "location": "卫生间", "power": False, "target_temp": 45, "model": "AquaWarm-W80"}),
        ),
        scene_exit="power_off",
    ),
    DeviceCapability(
        device_type=DeviceType.LOCK,
        tool_name="control_lock",
        device_label="门锁",
        tool_summary="控制智能门锁的上锁与解锁。",
        usage_examples=(
            '  "把门锁上"      → device_name="玄关门锁", action="lock"',
            '  "解锁门锁"      → action="unlock"（需人工审批）',
        ),
        device_examples='"玄关门锁"',
        not_found_text="❌ 找不到指定的门锁。当前可用的门锁有: 玄关门锁。",
        common_params=(),
        actions=(
            ActionSpec(
                "lock", "lock", "上锁", lambda args, device: {"locked": True},
                _LOCK_HANDLERS["lock"], precheck=_LOCK_HANDLERS["offline"],
            ),
            ActionSpec(
                # 解锁是对外敏感动作，走人工审批；安全判定在 agent/approval.py。
                "unlock", "unlock", "解锁（对外敏感动作，需人工审批）",
                lambda args, device: {"locked": False},
                _LOCK_HANDLERS["unlock"], precheck=_LOCK_HANDLERS["offline"],
            ),
        ),
        default_devices=(
            # 出厂即锁（locked=True）。
            (LockDevice, {"device_id": "entryway_lock", "name": "玄关门锁", "location": "玄关", "locked": True, "battery": 90, "model": "GuardLock-D3"}),
        ),
        scene_exit="lock",
    ),
    DeviceCapability(
        device_type=DeviceType.KETTLE,
        tool_name="control_kettle",
        device_label="烧水壶",
        tool_summary="控制电热水壶的开关、目标水温和「烧开」动作。",
        usage_examples=(
            '  "把水烧开"      → action="boil"',
            '  "烧水到80度"    → action="set_temp", target_temp=80',
            '  "打开烧水壶"    → action="on"',
            '  "关掉烧水壶"    → action="off"',
        ),
        device_examples='"厨房烧水壶"',
        not_found_text="❌ 找不到指定的电热水壶。当前可用的烧水壶有: 厨房烧水壶。",
        common_params=(
            ParamSpec("target_temp", int, 100, "目标水温 40-100°C"),
        ),
        actions=(
            ActionSpec(
                # boil 是一步开机并加热到 100°C 的复合动作：保留它让新增设备不只是
                # 复制 on/off/set_xxx 模板，也给 Planner 演示"带业务语义的动作"。
                "boil", "boil", "一键烧开（开机并加热到 100°C）",
                lambda args, device: {"power": True, "target_temp": 100},
                _KETTLE_HANDLERS["boil"],
            ),
            ActionSpec("on", "on", "打开烧水壶", lambda args, device: {"power": True}, _KETTLE_HANDLERS["on"]),
            ActionSpec("off", "off", "关闭烧水壶", lambda args, device: {"power": False}, _KETTLE_HANDLERS["off"]),
            ActionSpec(
                "set_temp",
                "set_temp(target_temp)",
                "设置目标水温（需配合 target_temp 参数）",
                lambda args, device: {
                    "power": True,
                    "target_temp": _int_arg(args, "target_temp", 100, 40, 100),
                },
                _KETTLE_HANDLERS["set_temp"],
                preference=PreferenceSpec("kettle.target_temp", lambda a: {"target_temp": a["target_temp"]}),
            ),
        ),
        default_devices=(
            (KettleDevice, {"device_id": "kitchen_kettle", "name": "厨房烧水壶", "location": "厨房", "power": False, "target_temp": 100, "model": "QuickBoil-K15"}),
        ),
        scene_exit="power_off",
    ),
)

# ============================================================
# 只读传感器的默认实例（没有 control_xxx 能力，但同属单一数据源，
# 供模拟器注册与 registry.find 的关键词匹配使用）
# ============================================================

SENSOR_DEFAULT_DEVICES: tuple[tuple[type, dict], ...] = (
    # 同类两台传感器刻意用**同一个型号**（真实住宅就是同款买两个）。
    # 副作用是只说"温湿度传感器"会解析成 ambiguous——这不是缺陷：
    # 说明书通用而设备不通用，问的是哪一台仍然要说清楚，因为自证核对读的是
    # 那一台的真实状态（电量、读数）。
    # 初始湿度低于加湿器目标湿度 60%，"开加湿器 → 湿度上升"闭环一开机就能演示。
    (TempHumiditySensor, {"device_id": "living_room_th_sensor", "name": "客厅温湿度传感器", "location": "客厅", "temperature": 27.0, "humidity": 42, "model": "ThermoSense-T20"}),
    (TempHumiditySensor, {"device_id": "bedroom_th_sensor", "name": "卧室温湿度传感器", "location": "卧室", "temperature": 26.0, "humidity": 48, "model": "ThermoSense-T20"}),
    # last_motion_at 留空表示开机时没有任何活动记录，occupied 为 False。
    (PresenceSensor, {"device_id": "living_room_presence", "name": "客厅人体传感器", "location": "客厅", "timeout_minutes": 15, "model": "MotionEye-P10"}),
    (PresenceSensor, {"device_id": "entryway_presence", "name": "玄关人体传感器", "location": "玄关", "timeout_minutes": 5, "model": "MotionEye-P10"}),
)

# registry.find 策略 3 的类型关键词（原 base.py 里手写的 keywords_map 迁移到此）。
# 传感器也在这里声明，因为"温度""有没有人"同样是类型关键词，只是它指向只读设备。
TYPE_KEYWORDS: dict[DeviceType, tuple[str, ...]] = {
    DeviceType.LIGHT: ("灯", "灯光", "照明"),
    DeviceType.AC: ("空调", "冷气", "暖气", "制冷", "制热"),
    DeviceType.TV: ("电视", "电视机"),
    DeviceType.CURTAIN: ("窗帘", "帘", "遮阳"),
    DeviceType.HUMIDIFIER: ("加湿器", "加湿", "雾化器"),
    DeviceType.WATER_HEATER: ("热水器", "电热水器", "洗澡"),
    DeviceType.LOCK: ("门锁", "锁", "大门", "门"),
    DeviceType.KETTLE: ("烧水壶", "热水壶", "水壶", "烧水"),
    DeviceType.TEMP_HUMIDITY_SENSOR: (
        "温湿度传感器", "温湿度计", "温湿度", "温度计", "湿度计", "温度", "湿度",
    ),
    DeviceType.PRESENCE_SENSOR: (
        "人体传感器", "人体存在传感器", "存在传感器", "人体感应", "人体", "有没有人", "有人",
    ),
}

# ============================================================
# 派生视图（全部由上面生成，禁止手写）
# ============================================================

# 可控制工具名（不含只读的 read_sensor / get_device_status）。
CONTROL_TOOL_NAMES: tuple[str, ...] = tuple(cap.tool_name for cap in CAPABILITIES)

CAPABILITIES_BY_TOOL: dict[str, DeviceCapability] = {cap.tool_name: cap for cap in CAPABILITIES}

# 离家/睡眠批量操作按 scene_exit 分组的类型集合（tools/scenes.py 使用）。
SCENE_EXIT_TYPES: dict[str, frozenset[DeviceType]] = {
    behavior: frozenset(cap.device_type for cap in CAPABILITIES if cap.scene_exit == behavior)
    for behavior in ("power_off", "curtain_close", "lock")
}
