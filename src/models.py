"""
数据模型模块
============
使用 Pydantic v2 定义所有智能家居设备的数据模型。

设计原则:
  - 严格类型约束: 枚举类型确保值合法（不会出现 "coool" 这种 typo）
  - 字段验证: 亮度 0-100、温度 16-30，自动截断非法值
  - 自描述: 每个字段都有 Field(description=...) 便于 LLM 理解
  - 序列化: 所有模型可转为 JSON，方便 MCP 传输和持久化
"""

from enum import Enum
from typing import Optional, Union
from pydantic import BaseModel, Field, field_validator


# ============================================================
# 枚举定义（确保值的合法性）
# ============================================================

# ============================================================
# DeviceType — 设备类型枚举
# ============================================================
class DeviceType(str, Enum):
    """
    设备类型枚举 — 用"有名字的常量"替代魔法字符串。

    为什么需要这个类？（小白必读）
    ─────────────────────────────
    如果不用枚举，代码里会到处都是裸字符串：
      if device["type"] == "light":       # 手滑写成 "ligth" → bug 潜伏！
      if device["type"] == "LIGHT":       # 大小写不一致 → 匹配失败！
      if device["type"] == "灯光":        # 中英文混用 → 团队协作灾难！

    用枚举之后：
      if device.device_type == DeviceType.LIGHT:   # IDE 自动补全，写错直接报红
      if device.device_type == DeviceType.light:   # ❌ 这行根本跑不起来

    这就是"让编译器帮你干活"的思路 —— 把错误从运行时提前到写代码时。

    双重继承 str + Enum 的作用：
    ────────────────────────────
      str:  让枚举值能直接当字符串用 → f"类型是{DeviceType.LIGHT}" 输出 "类型是light"
      Enum: 提供枚举的核心能力   → 限制取值范围、支持遍历、支持比较
    """

    # ---- 成员定义 ----
    # 每个成员有两个要素:
    #   变量名 (大写)  = 给程序员看的，IDE 自动补全用
    #   值    (字符串) = 给计算机看的，序列化/存储/API 传输用
    LIGHT = "light"       # 灯光设备
    AC = "ac"             # 空调设备 (Air Conditioner)
    TV = "tv"             # 电视设备
    CURTAIN = "curtain"   # 窗帘设备
    HUMIDIFIER = "humidifier"  # 加湿器
    WATER_HEATER = "water_heater"  # 电热水器（洗澡用）
    LOCK = "lock"         # 智能门锁
    KETTLE = "kettle"     # 电热水壶

    # ---- 只读传感器 ----
    # 上面 5 种都是"执行器"：Agent 下命令、设备改状态。
    # 下面 2 种是"传感器"：Agent 只能读，读到的是环境的真实反馈。
    # 这个区别在工具层很重要——传感器没有 control_xxx 工具，
    # 也不会出现在场景模式的批量开关里。
    TEMP_HUMIDITY_SENSOR = "temp_humidity_sensor"  # 温湿度传感器
    PRESENCE_SENSOR = "presence_sensor"            # 人体存在传感器

    # ═══════════════════════════════════════════════════════
    # @property 是什么意思？
    # ═══════════════════════════════════════════════════════
    # @property 是 Python 的"属性装饰器"。
    # 它让一个方法可以像属性一样访问 —— 不需要加括号 ()。
    #
    # 对比:
    #   没有 @property:  DeviceType.LIGHT.label_cn()  → 调用方法，需要 ()
    #   有   @property:  DeviceType.LIGHT.label_cn    → 像读属性一样，不用 ()
    #
    # 为什么这样做？因为"中文标签"是一个"计算出来的值"而非"需要执行的动作"，
    # 用属性访问更自然、更符合直觉。
    # ═══════════════════════════════════════════════════════

    @property
    def label_cn(self) -> str:
        """
        返回当前枚举成员的中文名称。

        用法:
          >>> DeviceType.LIGHT.label_cn
          '灯光'
          >>> DeviceType.AC.label_cn
          '空调'

        为什么用 dict 映射而不是 if/elif？
          - 字典查找是 O(1) 常数时间，20 个类型也不会变慢
          - 新增类型只需加一行字典条目，不用改逻辑
          - 代码结构清晰：所有映射关系一目了然

        self.value 是什么？
          self 就是当前枚举成员（比如 DeviceType.LIGHT）
          self.value 就是该成员的值（比如 "light"）
        """
        # 建立 英文值 → 中文名 的映射表
        labels: dict[str, str] = {
            "light": "灯光",
            "ac": "空调",
            "tv": "电视",
            "curtain": "窗帘",
            "humidifier": "加湿器",
            "water_heater": "电热水器",
            "lock": "门锁",
            "kettle": "电热水壶",
            "temp_humidity_sensor": "温湿度传感器",
            "presence_sensor": "人体存在传感器",
        }
        # dict.get(key, default) 的安全之处：
        #   如果 key 存在 → 返回对应的中文名
        #   如果 key 不存在（未来新增了类型但忘了加映射）→ 返回原始英文值作为兜底
        # 这样永远不会因为 KeyError 而崩溃
        return labels.get(self.value, self.value)


class ACMode(str, Enum):
    """空调运行模式"""
    COOL = "cool"
    HEAT = "heat"
    FAN = "fan"
    DRY = "dry"

    @property
    def label_cn(self) -> str:
        return {"cool": "制冷", "heat": "制热", "fan": "送风", "dry": "除湿"}.get(
            self.value, self.value
        )


class FanSpeed(str, Enum):
    """空调风速"""
    AUTO = "auto"
    LOW = "low"
    MID = "mid"
    HIGH = "high"

    @property
    def label_cn(self) -> str:
        return {"auto": "自动", "low": "低", "mid": "中", "high": "高"}.get(
            self.value, self.value
        )


class CurtainPosition:
    """窗帘位置常量"""
    FULLY_CLOSED = 0
    FULLY_OPEN = 100


# ============================================================
# 设备基类
# ============================================================

class BaseDevice(BaseModel):
    """
    所有智能家居设备的抽象基类。
    每个设备都有: ID (唯一标识)、名称（中文）、类型、开关状态、位置。
    """

    # 这是 Pydantic v2 的“类型注解 + Field 字段配置”写法，完整结构为：
    #
    #   字段名: Python 类型 = Field(默认值或必填标记, 其他约束...)
    #
    # 拆开理解下面的声明：
    #   1. device_id 是字段名，创建 BaseDevice 或其子类时通过它传入设备 ID。
    #   2. “: str”是 Python 类型注解，表示该字段期望得到一个字符串；
    #      Pydantic 会在模型实例化时根据该类型解析并验证传入的数据。
    #   3. Field(...) 用于补充普通类型注解无法表达的字段信息，例如：
    #      是否必填、默认值、数值范围、字符串格式以及接口文档描述等。
    #   4. Field 的第一个参数“...”是 Python 的 Ellipsis 对象；在这里不是省略代码，
    #      而是 Pydantic 的“此字段必填”标记。调用模型时不传 device_id 会验证失败。
    #   5. description 主要用于生成 JSON Schema、API 文档，也能帮助 LLM/开发者
    #      理解字段含义；它本身不会改变 device_id 的实际值。
    #   6. pattern 接收正则表达式，用来限制字符串格式。r 前缀表示“原始字符串”，
    #      可避免 Python 先处理正则中的反斜杠，是编写正则时的常用写法。
    #
    # 正则 r"^[a-z0-9_]+$" 各部分含义：
    #   ^          从字符串开头开始匹配
    #   [a-z0-9_]  每个字符只能是小写英文字母、数字或下划线
    #   +          前面的合法字符至少出现一次，因此不允许空字符串
    #   $          必须一直匹配到字符串结尾
    #
    # 示例：
    #   device_id="living_room_light"  # ✅ 合法
    #   device_id="light_2"            # ✅ 合法
    #   device_id="Living-Room-Light"  # ❌ 含大写字母和连字符，会触发验证错误
    device_id: str = Field(
        ...,  # 必填字段：没有默认值，实例化模型时必须传入 device_id
        description="设备唯一标识，如 living_room_light",
        pattern=r"^[a-z0-9_]+$",  # 只允许小写字母、数字和下划线，且不能为空
    )
    name: str = Field(
        ...,
        description="设备中文名称，如 客厅灯",
    )
    device_type: DeviceType = Field(
        ...,
        description="设备类型",
    )
    power: bool = Field(
        default=False,
        description="电源开关状态: True=开, False=关",
    )
    location: str = Field(
        default="",
        description="设备所在房间，如 客厅、卧室",
    )

    # ---- 通用方法 ----

    def turn_on(self) -> None:
        """打开设备"""
        self.power = True

    def turn_off(self) -> None:
        """关闭设备"""
        self.power = False

    def toggle(self) -> None:
        """切换开关状态"""
        self.power = not self.power

    def to_status_text(self) -> str:
        """生成设备状态描述文本（子类可覆盖）"""
        status = "🟢 开启" if self.power else "🔴 关闭"
        return f"{self.name} ({self.device_id}): {status}"


# ============================================================
# 具体设备模型
# ============================================================

class LightDevice(BaseDevice):
    """
    灯光设备模型。

    属性:
      brightness: 亮度 0-100（0=关灯亮度, 100=最亮）
      color:      色温描述，如"暖白"、"白光"、"暖黄"
    """
    device_type: DeviceType = Field(default=DeviceType.LIGHT, frozen=True)
    brightness: int = Field(default=80, ge=0, le=100, description="亮度百分比 0-100")
    color: str = Field(default="暖白", description="色温/颜色描述")

    # field_validator 是 Pydantic v2 提供的字段验证器装饰器。
    # 参数 "brightness" 指定：下面这个方法专门负责验证 brightness 字段。
    # 当 Pydantic 创建 LightDevice 实例并处理 brightness 时，会自动调用该方法，
    # 因此不需要我们手动执行 clamp_brightness(...)。
    #
    # 装饰器从下往上应用：Python 会先把方法转换为 classmethod，
    # 再由 field_validator 将它注册为 brightness 的字段验证器。
    @field_validator("brightness")
    @classmethod
    def clamp_brightness(cls, v: int) -> int:
        """
        将亮度值限制在 0～100 之间。

        参数说明：
          cls:
            当前模型类，即 LightDevice。因为使用了 @classmethod，所以第一个参数
            是类而不是实例 self。这里虽然没有用到 cls，但验证器仍采用该标准签名。
          v:
            Pydantic 传给验证器的 brightness 值；“: int”表示期望它是整数。

        “-> int”表示本方法最终必须返回一个整数。返回值会作为验证处理后的
        brightness 值，继续交给 Pydantic，并最终保存到模型实例中。

        截断效果：
          v < 0       → 返回 0
          0 <= v <= 100 → 返回 v 本身
          v > 100     → 返回 100

        注意：当前验证器使用 field_validator 的默认 mode="after"，会在字段的
        基础类型和 Field 约束验证之后执行。而 brightness 同时声明了 ge=0、le=100，
        所以超出范围的值通常会先触发 Pydantic ValidationError，无法运行到这里。
        若希望真正把越界输入自动截断，应改用 mode="before"，或移除 ge/le 约束。
        """

        # 内层 min(100, v)：先保证结果不大于 100。
        #   v=120 → 100；v=80 → 80；v=-10 → -10
        upper_bounded = min(100, v)

        # 外层 max(0, ...)：再保证结果不小于 0。
        #   upper_bounded=-10 → 0；80 → 80；100 → 100
        # 两步合起来等价于：return max(0, min(100, v))。
        return max(0, upper_bounded)

    def to_status_text(self) -> str:
        status = "🟢 开启" if self.power else "🔴 关闭"
        return (
            f"{self.name} ({self.device_id}): {status} | "
            f"亮度: {self.brightness}% | 色温: {self.color}"
        )


class ACDevice(BaseDevice):
    """
    空调设备模型。

    属性:
      temperature: 目标温度 16-30°C
      mode:        运行模式（制冷/制热/送风/除湿）
      fan_speed:   风速（自动/低/中/高）
    """
    device_type: DeviceType = Field(default=DeviceType.AC, frozen=True)
    temperature: int = Field(default=26, ge=16, le=30, description="目标温度 16-30°C")
    mode: ACMode = Field(default=ACMode.COOL, description="运行模式")
    fan_speed: FanSpeed = Field(default=FanSpeed.AUTO, description="风速")

    @field_validator("temperature")
    @classmethod
    def clamp_temperature(cls, v: int) -> int:
        """确保温度在 16-30°C 范围内"""
        return max(16, min(30, v))

    def to_status_text(self) -> str:
        status = "🟢 运行中" if self.power else "🔴 关闭"
        return (
            f"{self.name} ({self.device_id}): {status} | "
            f"温度: {self.temperature}°C | "
            f"模式: {self.mode.label_cn} | 风速: {self.fan_speed.label_cn}"
        )


class TVDevice(BaseDevice):
    """
    电视设备模型。

    属性:
      volume:  音量 0-100
      channel: 输入源 / 频道名
      muted:   是否静音
    """
    device_type: DeviceType = Field(default=DeviceType.TV, frozen=True)
    volume: int = Field(default=30, ge=0, le=100, description="音量 0-100")
    channel: str = Field(default="HDMI 1", description="输入源/频道")
    muted: bool = Field(default=False, description="是否静音")

    @field_validator("volume")
    @classmethod
    def clamp_volume(cls, v: int) -> int:
        """确保音量在合法范围"""
        return max(0, min(100, v))

    def to_status_text(self) -> str:
        status = "🟢 开启" if self.power else "🔴 关闭"
        mute_text = "🔇 已静音" if self.muted else ""
        return (
            f"{self.name} ({self.device_id}): {status} | "
            f"音量: {self.volume}% {mute_text} | 输入源: {self.channel}"
        )


class CurtainDevice(BaseDevice):
    """
    窗帘设备模型。

    属性:
      position: 开合度 0-100（0=完全关闭, 100=完全打开）
    """
    device_type: DeviceType = Field(default=DeviceType.CURTAIN, frozen=True)
    position: int = Field(default=0, ge=0, le=100, description="开合度 0-100")

    @field_validator("position")
    @classmethod
    def clamp_position(cls, v: int) -> int:
        """确保位置在合法范围"""
        return max(0, min(100, v))

    def to_status_text(self) -> str:
        if self.position == 0:
            pos_text = "完全关闭"
        elif self.position == 100:
            pos_text = "完全打开"
        else:
            pos_text = f"打开 {self.position}%"
        return f"{self.name} ({self.device_id}): {pos_text}"


class HumidifierDevice(BaseDevice):
    """
    加湿器设备模型。

    属性:
      target_humidity: 目标湿度 30-80%
      mist_level: 雾量档位，自动、低、中、高
      water_level: 水箱余量 0-100%
    """
    device_type: DeviceType = Field(default=DeviceType.HUMIDIFIER, frozen=True)
    target_humidity: int = Field(default=60, ge=30, le=80, description="目标湿度 30-80%")
    mist_level: FanSpeed = Field(default=FanSpeed.AUTO, description="雾量档位")
    water_level: int = Field(default=60, ge=0, le=100, description="水箱余量 0-100%")

    def to_status_text(self) -> str:
        status = "🟢 开启" if self.power else "🔴 关闭"
        return (
            f"{self.name} ({self.device_id}): {status} | "
            f"目标湿度: {self.target_humidity}% | "
            f"雾量: {self.mist_level.label_cn} | 水箱: {self.water_level}%"
        )


class WaterHeaterDevice(BaseDevice):
    """
    电热水器设备模型（洗澡用）。

    属性:
      target_temp: 目标水温 35-75°C（洗澡水温区间，clamp）
    """
    device_type: DeviceType = Field(default=DeviceType.WATER_HEATER, frozen=True)
    target_temp: int = Field(default=45, ge=35, le=75, description="目标水温 35-75°C")

    @field_validator("target_temp")
    @classmethod
    def clamp_target_temp(cls, v: int) -> int:
        """确保水温在 35-75°C 范围内"""
        return max(35, min(75, v))

    def to_status_text(self) -> str:
        status = "🟢 加热中" if self.power else "🔴 关闭"
        return (
            f"{self.name} ({self.device_id}): {status} | "
            f"目标水温: {self.target_temp}°C"
        )


class KettleDevice(BaseDevice):
    """
    电热水壶设备模型。

    属性:
      target_temp: 目标水温 40-100°C（clamp）。boil 动作会拉到 100°C。

    为什么保留一个独特的 boil 动作？
      加湿器、热水器都是 on/off/set_xxx 的老三样，烧水壶的"烧开"是它天然的
      语义动作——一步开机并加热到 100°C。保留它让新增设备不只是复制模板，
      也给 Planner 的 action 词表演示了"带业务语义的复合动作"。
    """
    device_type: DeviceType = Field(default=DeviceType.KETTLE, frozen=True)
    target_temp: int = Field(default=100, ge=40, le=100, description="目标水温 40-100°C")

    @field_validator("target_temp")
    @classmethod
    def clamp_target_temp(cls, v: int) -> int:
        """确保水温在 40-100°C 范围内"""
        return max(40, min(100, v))

    def to_status_text(self) -> str:
        status = "🟢 加热中" if self.power else "🔴 关闭"
        return (
            f"{self.name} ({self.device_id}): {status} | "
            f"目标水温: {self.target_temp}°C"
        )


class LockDevice(BaseDevice):
    """
    智能门锁设备模型。

    属性:
      locked:  是否已上锁（默认 True，出厂即锁）
      battery: 电量 0-100%

    为什么用 locked 而不是 power 表达锁态？
      门锁的"开/关"指的是锁舌的开合，语义上是 locked；power 沿用基类，
      表示门锁本身在线（有电、能联网）。把锁态塞进 power 会和其他执行器的
      "开机/关机"语义打架，验证器读期望状态时也会误判。
    """
    device_type: DeviceType = Field(default=DeviceType.LOCK, frozen=True)
    power: bool = Field(default=True, description="门锁是否在线")
    locked: bool = Field(default=True, description="是否已上锁: True=锁上, False=解锁")
    battery: int = Field(default=100, ge=0, le=100, description="电量 0-100%")

    def to_status_text(self) -> str:
        if not self.power:
            return f"{self.name} ({self.device_id}): ⚠️ 离线"
        state = "🔒 已上锁" if self.locked else "🔓 已解锁"
        return f"{self.name} ({self.device_id}): {state} | 电量 {self.battery}%"


# ============================================================
# 传感器模型（只读设备）
# ============================================================
#
# 传感器和上面的执行器有一个本质区别：
#
#   执行器: Agent 说"开到 26 度" → 设备状态就是 26 度。
#           验证时读回来必然是 26 度，本质上是"自证"。
#   传感器: Agent 说不了话，只能读。读到的值来自环境，
#           不受 Agent 控制，所以它才是真正的"外部反馈"。
#
# 这个区别决定了传感器在架构里的位置：
#   · 没有 control_xxx 工具（改不了），只有 read_sensor（读得到）
#   · 不进 PLANNING_TOOL_NAMES（计划步骤不能"执行"一次读取）
#   · 不进场景模式的批量开关（离家模式不该"关掉"温湿度计）
#   · power 字段在这里表示"传感器在线"，不是"开关"
#
# 有了传感器，Agent 才能从"你说什么我做什么"变成
# "我先看看情况，再决定做什么、做到什么程度"。


class TempHumiditySensor(BaseDevice):
    """
    温湿度传感器模型（只读）。

    属性:
      temperature: 实测温度 °C，允许小数
      humidity:    实测相对湿度 0-100%
      battery:     电量 0-100%
    """
    device_type: DeviceType = Field(
        default=DeviceType.TEMP_HUMIDITY_SENSOR, frozen=True
    )
    power: bool = Field(default=True, description="传感器是否在线")
    temperature: float = Field(
        default=24.0, ge=-40.0, le=80.0, description="实测温度 °C"
    )
    humidity: int = Field(
        default=50, ge=0, le=100, description="实测相对湿度 0-100%"
    )
    battery: int = Field(default=100, ge=0, le=100, description="电量 0-100%")

    def to_status_text(self) -> str:
        if not self.power:
            return f"{self.name} ({self.device_id}): ⚠️ 离线"
        return (
            f"{self.name} ({self.device_id}): "
            f"温度 {self.temperature:.1f}°C | 湿度 {self.humidity}% | "
            f"电量 {self.battery}%"
        )


class PresenceSensor(BaseDevice):
    """
    人体存在传感器模型（只读）。

    属性:
      occupied:        当前是否检测到人
      last_motion_at:  最近一次检测到活动的时间（ISO 8601 字符串，None 表示从未）
      timeout_minutes: 多久没有活动就判定为无人
      battery:         电量 0-100%

    为什么要 timeout_minutes？
      真实的人体传感器只能感知"活动"，不能感知"静止的人"。
      业界做法是：检测到活动 → 置为有人；超过 N 分钟没有新活动 → 回落为无人。
      模拟器按这个规则从 last_motion_at 推算 occupied，而不是随机生成，
      这样测试可以通过设置 last_motion_at 精确控制传感器行为。
    """
    device_type: DeviceType = Field(
        default=DeviceType.PRESENCE_SENSOR, frozen=True
    )
    power: bool = Field(default=True, description="传感器是否在线")
    occupied: bool = Field(default=False, description="当前是否检测到人")
    last_motion_at: Optional[str] = Field(
        default=None, description="最近一次检测到活动的 ISO 8601 时间，None 表示从未"
    )
    timeout_minutes: int = Field(
        default=15, ge=1, le=240, description="多久没有活动就判定为无人（分钟）"
    )
    battery: int = Field(default=100, ge=0, le=100, description="电量 0-100%")

    def to_status_text(self) -> str:
        if not self.power:
            return f"{self.name} ({self.device_id}): ⚠️ 离线"
        state = "🚶 有人" if self.occupied else "🕳️ 无人"
        detail = f" | 最近活动 {self.last_motion_at}" if self.last_motion_at else ""
        return (
            f"{self.name} ({self.device_id}): {state}"
            f"{detail} | 电量 {self.battery}%"
        )


# ============================================================
# 类型别名（方便联合类型使用）
# ============================================================
AnyDevice = Union[
    LightDevice,
    ACDevice,
    TVDevice,
    CurtainDevice,
    HumidifierDevice,
    WaterHeaterDevice,
    KettleDevice,
    LockDevice,
    TempHumiditySensor,
    PresenceSensor,
]

# 只读传感器类型集合。工具层和场景层用它来判断"这台设备不能被控制"，
# 避免每处都硬编码一遍类型列表。
SENSOR_DEVICE_TYPES = frozenset({
    DeviceType.TEMP_HUMIDITY_SENSOR,
    DeviceType.PRESENCE_SENSOR,
})
