"""说明书检查项里「系统可以自己核对」的那一半。

为什么需要这个模块
------------------
说明书给出的排查清单天然分成两类：一类是**系统读一下设备状态就能核对**的
（"确认设置温度低于室温"、"确认运行模式为制冷"），一类是**必须人到现场动手**的
（"检查滤网是否积尘"、"检查室外机是否被遮挡"）。

纯文档检索只能把两类一起念给用户，让用户自己去做四件事。而这个 Agent 手里有
设备的真实状态，所以第一类可以直接核对掉——把"文档问答"变成"可执行诊断"。

这里的设计约束有三条：

1. **说明书正文是权威文本，标注是旁挂的元数据。** 检查项的标注写成 Markdown
   注释（`<!--check:xxx-->`），渲染时不可见，说明书读起来仍然是说明书；解析时
   剥离标注，正文一个字不改。改标注不改原文，也就不存在"为了让代码好解析而
   篡改厂商文本"的问题。
2. **判定读真实设备状态，不问模型。** 跟 verifier 读注册中心真实状态是同一个
   原则：模型会把"设置了 26 度"复述成"温度设置正常"，只有实测值能反驳它。
3. **未知的 check id 是构造期失败，不是运行期静默跳过。** 语料里写错一个 id，
   `KnowledgeBase` 构造就抛错（见 `base.py`），而不是安静地把那一项降级成
   "需人工确认"——后者会让一条本该自动核对的检查项无声消失，
   表现是"诊断突然变笨了"，极难定位。

判定结果用 problem / ok / unknown 三值而不是布尔：`unknown` 是必需的第三态，
因为"房间里没有温湿度传感器所以读不到室温"跟"室温确实高于设定温度"是两件事，
前者必须回退给人确认，后者才是定位到了根因。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel

from ..models import ACMode, BaseDevice, FanSpeed

Verdict = Literal["problem", "ok", "unknown"]


class CheckContext(BaseModel):
    """一次自证核对能看到的全部事实。

    刻意做成一个显式的小对象而不是直接传 registry：检查函数只该看到"这台设备"
    和"这个房间的环境读数"，不该有能力去遍历全屋设备或触发环境推演
    （`registry.tick_environment()` 只允许 `read_sensor` 调用，别处调用会让
    同一轮对话里的读数随调用次数漂移）。缩小检查函数的可见范围，
    就从结构上排除了这类误用。
    """

    device: BaseDevice
    room_temperature: float | None = None
    room_humidity: int | None = None
    # 读数来自哪台传感器，写进给用户看的结论里——用户要能追溯"你说的室温是谁测的"
    sensor_name: str | None = None


class CheckOutcome(BaseModel):
    """单个检查项的核对结果。detail 由代码生成，不经模型改写。"""

    check_id: str
    item_text: str
    verdict: Verdict
    detail: str


class SelfCheck(BaseModel):
    """一条可自证检查项的声明。"""

    check_id: str
    label: str
    evaluate: Callable[[CheckContext], CheckOutcome]

    model_config = {"arbitrary_types_allowed": True}


def _outcome(check_id: str, verdict: Verdict, detail: str) -> CheckOutcome:
    # item_text 由调用方按语料原文填入：核对结果要引用说明书的原始措辞，
    # 而不是这里另写一套说法，否则用户看到的检查项和说明书对不上。
    return CheckOutcome(check_id=check_id, item_text="", verdict=verdict, detail=detail)


def _unsupported(check_id: str, device: BaseDevice, what: str) -> CheckOutcome:
    """设备身上没有这个字段时统一退回 unknown。

    绝不能返回 ok——那等于宣称"核对通过"。语料把某条检查项挂到了不具备该能力的
    设备上（例如给灯挂了空调的模式检查），唯一安全的输出是"无法自动核对"。
    用 `getattr(..., None)` 取字段而不是 `isinstance` 判设备类：检查函数只认能力，
    不认类型，这样新增一种同样带该字段的设备不需要回来改这里。
    """
    return _outcome(check_id, "unknown", f"{device.name}没有{what}信息，无法自动核对。")


def _check_device_is_on(context: CheckContext) -> CheckOutcome:
    device = context.device
    if device.power:
        return _outcome("device_is_on", "ok", f"{device.name}当前处于开机状态。")
    return _outcome(
        "device_is_on", "problem",
        f"{device.name}当前是关机状态——设备没有运行，后续检查项都无从谈起。",
    )


def _check_ac_mode_is_cool(context: CheckContext) -> CheckOutcome:
    device = context.device
    mode = getattr(device, "mode", None)
    if mode is None:
        # 这台设备没有模式字段（例如语料被挂到了非空调设备上）。
        # 不能当成 ok——那等于宣称核对通过；只能交回给人。
        return _outcome("ac_mode_is_cool", "unknown", f"{device.name}没有运行模式信息，无法自动核对。")
    if mode == ACMode.COOL:
        return _outcome("ac_mode_is_cool", "ok", f"{device.name}当前运行模式为制冷。")
    return _outcome(
        "ac_mode_is_cool", "problem",
        f"{device.name}当前运行模式是{mode.value}，不是制冷——送风与除湿模式不会降低室温。",
    )


def _check_ac_target_temp_below_room(context: CheckContext) -> CheckOutcome:
    device = context.device
    target = getattr(device, "temperature", None)
    room = context.room_temperature
    if target is None:
        return _outcome(
            "ac_target_temp_below_room", "unknown",
            f"{device.name}没有目标温度信息，无法自动核对。",
        )
    if room is None:
        # 房间里没有温湿度传感器：这一项退回人工，而不是假定它通过。
        return _outcome(
            "ac_target_temp_below_room", "unknown",
            f"{device.location or '该房间'}没有可读的温度传感器，无法核对设定温度与室温的关系，请手动确认。",
        )
    source = f"（{context.sensor_name}）" if context.sensor_name else ""
    if target < room:
        return _outcome(
            "ac_target_temp_below_room", "ok",
            f"设定温度 {target}°C 低于实测室温 {room}°C{source}，这一项正常。",
        )
    return _outcome(
        "ac_target_temp_below_room", "problem",
        f"设定温度 {target}°C 不低于实测室温 {room}°C{source}，"
        f"温差不足时压缩机不会持续制冷——这很可能就是不制冷的直接原因。",
    )


# 下面三个阈值刻意写成模块常量并各自解释理由：它们是"从哪个值起算异常"的判断，
# 不是实现细节。改动会直接改变诊断结论，必须能被读到、被质疑。
#
# 电量下限取 20%：门锁与无线传感器在低压区的表现是"时好时坏"（指纹偶发失败、
# 上报间隔变长），而不是彻底不工作——正因为它不彻底坏，用户最难自己联想到电池，
# 所以这一项值得系统主动点出来。
_LOW_BATTERY_PERCENT = 20

# 电热水器设定水温下限取 40°C：模型允许 35~75，而 35~40 这一段本身就低于
# 体感温水，"出水温度偏低"的直接原因往往就是设定值太低而非加热故障。
_WATER_HEATER_MIN_USEFUL_TEMP = 40

# 电热水壶的"烧开"下限取 95°C：模型允许 40~100（保温冲奶等场景会设到 40~80），
# 所以"水烧不开"经常是目标温度根本没设到沸点。留 5°C 余量是因为
# 温控探头本身有误差，设到 95 以上就应视为用户要的是沸水。
_KETTLE_BOILING_TEMP = 95


def _check_ac_fan_speed_not_lowest(context: CheckContext) -> CheckOutcome:
    device = context.device
    fan_speed = getattr(device, "fan_speed", None)
    if fan_speed is None:
        return _unsupported("ac_fan_speed_not_lowest", device, "风速")
    if fan_speed == FanSpeed.LOW:
        return _outcome(
            "ac_fan_speed_not_lowest", "problem",
            f"{device.name}当前风速是最低档——风量不足时冷量送不到房间，"
            f"体感上就像制冷能力下降。",
        )
    return _outcome(
        "ac_fan_speed_not_lowest", "ok",
        f"{device.name}当前风速为{fan_speed.value}档，不在最低档。",
    )


def _check_light_brightness_not_zero(context: CheckContext) -> CheckOutcome:
    device = context.device
    brightness = getattr(device, "brightness", None)
    # 用 is None 而不是 not brightness：亮度 0 是合法取值，也正是要报出来的那个异常。
    if brightness is None:
        return _unsupported("light_brightness_not_zero", device, "亮度")
    if brightness <= 0:
        return _outcome(
            "light_brightness_not_zero", "problem",
            f"{device.name}亮度设置为 {brightness}——调光值为 0 时肉眼近似熄灭，"
            f"看起来像灯坏了，其实是调光被调到了底。",
        )
    return _outcome(
        "light_brightness_not_zero", "ok",
        f"{device.name}当前亮度为 {brightness}，不是 0。",
    )


def _check_curtain_not_fully_closed(context: CheckContext) -> CheckOutcome:
    device = context.device
    position = getattr(device, "position", None)
    if position is None:
        return _unsupported("curtain_not_fully_closed", device, "开合位置")
    if position <= 0:
        return _outcome(
            "curtain_not_fully_closed", "problem",
            f"{device.name}当前处于完全关闭位置（0）——没有执行开启动作，"
            f"这与「电机不动」是两件事。",
        )
    return _outcome(
        "curtain_not_fully_closed", "ok",
        f"{device.name}当前开合位置为 {position}，不在完全关闭位置。",
    )


def _check_tv_is_not_muted(context: CheckContext) -> CheckOutcome:
    device = context.device
    muted = getattr(device, "muted", None)
    if muted is None:
        return _unsupported("tv_is_not_muted", device, "静音状态")
    if muted:
        return _outcome(
            "tv_is_not_muted", "problem",
            f"{device.name}当前处于静音状态——画面正常而没有声音，这就是最直接的原因。",
        )
    return _outcome("tv_is_not_muted", "ok", f"{device.name}当前未静音。")


def _check_humidifier_tank_has_water(context: CheckContext) -> CheckOutcome:
    device = context.device
    water_level = getattr(device, "water_level", None)
    if water_level is None:
        return _unsupported("humidifier_tank_has_water", device, "水位")
    if water_level <= 0:
        return _outcome(
            "humidifier_tank_has_water", "problem",
            f"{device.name}水箱水位为 {water_level}，已经空了——缺水保护下换能片不会工作。",
        )
    return _outcome(
        "humidifier_tank_has_water", "ok",
        f"{device.name}水箱水位为 {water_level}，有水。",
    )


def _check_humidifier_target_above_room(context: CheckContext) -> CheckOutcome:
    device = context.device
    target = getattr(device, "target_humidity", None)
    room = context.room_humidity
    if target is None:
        return _unsupported("humidifier_target_above_room", device, "目标湿度")
    if room is None:
        # 同 ac_target_temp_below_room：房间没有湿度读数就退回人工，不假定通过。
        return _outcome(
            "humidifier_target_above_room", "unknown",
            f"{device.location or '该房间'}没有可读的湿度传感器，"
            f"无法核对目标湿度与实测湿度的关系，请手动确认。",
        )
    source = f"（{context.sensor_name}）" if context.sensor_name else ""
    if target > room:
        return _outcome(
            "humidifier_target_above_room", "ok",
            f"目标湿度 {target}% 高于实测湿度 {room}%{source}，这一项正常。",
        )
    return _outcome(
        "humidifier_target_above_room", "problem",
        f"目标湿度 {target}% 不高于实测湿度 {room}%{source}，"
        f"已经达到设定值时设备不会继续加湿——这很可能就是「湿度上不去」的直接原因。",
    )


def _check_water_heater_target_is_high_enough(context: CheckContext) -> CheckOutcome:
    device = context.device
    target = getattr(device, "target_temp", None)
    if target is None:
        return _unsupported("water_heater_target_is_high_enough", device, "设定水温")
    if target < _WATER_HEATER_MIN_USEFUL_TEMP:
        return _outcome(
            "water_heater_target_is_high_enough", "problem",
            f"设定水温只有 {target}°C，低于 {_WATER_HEATER_MIN_USEFUL_TEMP}°C——"
            f"出水偏凉的原因是设定值太低，不是加热故障。",
        )
    return _outcome(
        "water_heater_target_is_high_enough", "ok",
        f"设定水温为 {target}°C，设定值本身没有问题。",
    )


def _check_kettle_target_is_boiling(context: CheckContext) -> CheckOutcome:
    device = context.device
    target = getattr(device, "target_temp", None)
    if target is None:
        return _unsupported("kettle_target_is_boiling", device, "目标水温")
    if target < _KETTLE_BOILING_TEMP:
        return _outcome(
            "kettle_target_is_boiling", "problem",
            f"目标水温设为 {target}°C，低于沸点档（{_KETTLE_BOILING_TEMP}°C 以上）——"
            f"到温即停，所以水不会烧开。",
        )
    return _outcome(
        "kettle_target_is_boiling", "ok",
        f"目标水温为 {target}°C，已设到沸点档。",
    )


def _check_device_battery_not_low(context: CheckContext) -> CheckOutcome:
    device = context.device
    battery = getattr(device, "battery", None)
    # 同样用 is None：电量 0 是合法取值。
    if battery is None:
        return _unsupported("device_battery_not_low", device, "电量")
    if battery < _LOW_BATTERY_PERCENT:
        return _outcome(
            "device_battery_not_low", "problem",
            f"{device.name}电量只剩 {battery}%，低于 {_LOW_BATTERY_PERCENT}%——"
            f"低压区的表现是时好时坏，而不是彻底不工作，很容易被误判成模组损坏。",
        )
    return _outcome(
        "device_battery_not_low", "ok",
        f"{device.name}电量为 {battery}%，电量充足。",
    )


# 可自证检查项的唯一数据源。语料里的 <!--check:xxx--> 只能引用这里已声明的 id，
# 引用不存在的 id 会让 KnowledgeBase 构造失败（见 base.py 的 _parse_checklist 调用点）。
SELF_CHECKS: dict[str, SelfCheck] = {
    check.check_id: check
    for check in (
        SelfCheck(
            check_id="device_is_on",
            label="设备是否处于开机状态",
            evaluate=_check_device_is_on,
        ),
        SelfCheck(
            check_id="ac_mode_is_cool",
            label="运行模式是否为制冷",
            evaluate=_check_ac_mode_is_cool,
        ),
        SelfCheck(
            check_id="ac_target_temp_below_room",
            label="设定温度是否低于室温",
            evaluate=_check_ac_target_temp_below_room,
        ),
        SelfCheck(
            check_id="ac_fan_speed_not_lowest",
            label="风速是否不在最低档",
            evaluate=_check_ac_fan_speed_not_lowest,
        ),
        SelfCheck(
            check_id="light_brightness_not_zero",
            label="亮度是否不为 0",
            evaluate=_check_light_brightness_not_zero,
        ),
        SelfCheck(
            check_id="curtain_not_fully_closed",
            label="是否已离开完全关闭位置",
            evaluate=_check_curtain_not_fully_closed,
        ),
        SelfCheck(
            check_id="tv_is_not_muted",
            label="是否未处于静音",
            evaluate=_check_tv_is_not_muted,
        ),
        SelfCheck(
            check_id="humidifier_tank_has_water",
            label="水箱是否有水",
            evaluate=_check_humidifier_tank_has_water,
        ),
        SelfCheck(
            check_id="humidifier_target_above_room",
            label="目标湿度是否高于实测湿度",
            evaluate=_check_humidifier_target_above_room,
        ),
        SelfCheck(
            check_id="water_heater_target_is_high_enough",
            label="设定水温是否足够高",
            evaluate=_check_water_heater_target_is_high_enough,
        ),
        SelfCheck(
            check_id="kettle_target_is_boiling",
            label="目标水温是否设到沸点档",
            evaluate=_check_kettle_target_is_boiling,
        ),
        SelfCheck(
            check_id="device_battery_not_low",
            label="电量是否充足",
            evaluate=_check_device_battery_not_low,
        ),
    )
}

KNOWN_CHECK_IDS: frozenset[str] = frozenset(SELF_CHECKS)


def run_self_check(check_id: str, item_text: str, context: CheckContext) -> CheckOutcome:
    """执行一条自证检查，并把语料原文回填进结果。

    check_id 未声明时直接 KeyError——语料与代码的引用关系已在 KnowledgeBase
    构造期校验过，能走到这里的 id 必然存在。这里不加兜底是刻意的：
    若真的漏了，应该炸在测试里，而不是变成一条静默消失的检查项。
    """
    outcome = SELF_CHECKS[check_id].evaluate(context)
    return outcome.model_copy(update={"item_text": item_text})
