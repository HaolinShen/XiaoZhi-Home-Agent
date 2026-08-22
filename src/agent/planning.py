"""Structured planning, deterministic execution, and state verification."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field, model_validator

from ..devices.base import DeviceRegistry
from ..models import DeviceType


class _InvalidArgument(ValueError):
    """计划里的参数无法解析成设备可接受的值。

    单独立一个异常类型，是为了把"参数写错"和真正的程序 bug 分开：前者应该变成
    preparation_error 回喂给 Planner 重写，后者才该往上抛。
    """


def _int_arg(args: dict[str, Any], name: str, default: int, low: int, high: int) -> int:
    """读取整数参数并夹到合法区间，无法解析时报错而不是崩溃。

    这里必须容错：`expected_state_for_step()` 在 executor 的 try 块之外被调用，
    模型只要写出 brightness="很亮" 这种值，裸 `int()` 抛出的 ValueError 就会掀翻
    整张图。转成 preparation_error 后会被判成确定性错误直接 replan —— 既不崩，
    也不会静默套用默认值把"调到很亮"悄悄执行成"调到 50%"再报告成功。
    """
    raw = args.get(name, default)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise _InvalidArgument(
            f"invalid argument: {name} 需要 {low}-{high} 之间的整数，收到 {raw!r}"
        ) from None
    return max(low, min(high, value))


@dataclass(frozen=True)
class ActionSpec:
    """一个 action 的完整声明。

    signature 是喂给 Planner 的合法值文本（括号内为该 action 需附带的参数），
    expected 接收 (arguments, device) 返回执行后设备应达到的状态 —— Verifier 拿它
    跟注册中心的真实状态比对。
    """

    signature: str
    expected: Callable[[dict[str, Any], Any], dict[str, Any]]


@dataclass(frozen=True)
class ToolSpec:
    """一个规划工具操作的设备类型及其全部合法 action。"""

    device_type: DeviceType
    actions: dict[str, ActionSpec]


# 规划分支的单一数据源：工具 → 设备类型 → 合法 action → 期望状态。
#
# 为什么这份声明必须存在：Planner 走 `llm.with_structured_output(ExecutionPlan)`，
# 不像 ReAct 分支那样 `bind_tools`，所以工具 docstring 里写的 "on / off / ..."
# 一个字都到不了模型面前。模型只能凭常识猜 action，于是会写出 Home Assistant
# 风格的 turn_on / turn_off —— 这正是规划第一版反复失败的根因。把合法值显式喂给
# 它，第一版就该是对的。
#
# 为什么合并成一张表：这份信息以前散成三份手写副本（喂 Planner 的词表、
# expected_state_for_step 的近百行 if/elif、工具实现的 if/elif），漏改一处的表现
# 是 Planner 第一版计划稳定失败，且不报错。现在前两者都从这里派生，只剩工具实现
# 那份仍是手写（它带各自的副作用文本与前置检查，无法反射），由
# tests/test_phase_seven.py 的一致性用例双向钉住。
DEVICE_ACTION_SPECS: dict[str, ToolSpec] = {
    "control_light": ToolSpec(DeviceType.LIGHT, {
        "on": ActionSpec("on", lambda args, device: {"power": True}),
        "off": ActionSpec("off", lambda args, device: {"power": False}),
        "set_brightness": ActionSpec(
            "set_brightness(brightness)",
            lambda args, device: {
                "power": True,
                "brightness": _int_arg(args, "brightness", 50, 0, 100),
            },
        ),
        "set_color": ActionSpec(
            "set_color(color)",
            lambda args, device: {"power": True, "color": args.get("color", "暖白")},
        ),
    }),
    "control_ac": ToolSpec(DeviceType.AC, {
        "on": ActionSpec(
            "on(可带 temperature、mode)",
            lambda args, device: {
                "power": True,
                "temperature": _int_arg(args, "temperature", 26, 16, 30),
                "mode": args.get("mode", "cool"),
            },
        ),
        "off": ActionSpec("off", lambda args, device: {"power": False}),
        "set_temp": ActionSpec(
            "set_temp(temperature)",
            lambda args, device: {
                "power": True,
                "temperature": _int_arg(args, "temperature", 26, 16, 30),
            },
        ),
        "set_mode": ActionSpec(
            "set_mode(mode)",
            lambda args, device: {"power": True, "mode": args.get("mode", "cool")},
        ),
        "set_fan": ActionSpec(
            "set_fan(fan_speed)",
            lambda args, device: {"fan_speed": args.get("fan_speed", "auto")},
        ),
    }),
    "control_tv": ToolSpec(DeviceType.TV, {
        "on": ActionSpec("on", lambda args, device: {"power": True}),
        "off": ActionSpec("off", lambda args, device: {"power": False}),
        "set_volume": ActionSpec(
            "set_volume(volume)",
            lambda args, device: {"volume": _int_arg(args, "volume", 30, 0, 100)},
        ),
        # mute 是翻转语义，所以期望状态依赖执行前的设备状态。
        "mute": ActionSpec("mute", lambda args, device: {"muted": not device.muted}),
        "set_channel": ActionSpec(
            "set_channel(channel)",
            lambda args, device: {
                "power": True,
                "channel": args.get("channel", "HDMI 1"),
            },
        ),
    }),
    "control_curtain": ToolSpec(DeviceType.CURTAIN, {
        "open": ActionSpec("open", lambda args, device: {"position": 100}),
        "close": ActionSpec("close", lambda args, device: {"position": 0}),
        "set_position": ActionSpec(
            "set_position(percentage)",
            lambda args, device: {"position": _int_arg(args, "percentage", 100, 0, 100)},
        ),
    }),
    "control_humidifier": ToolSpec(DeviceType.HUMIDIFIER, {
        "on": ActionSpec("on", lambda args, device: {"power": True}),
        "off": ActionSpec("off", lambda args, device: {"power": False}),
        "set_humidity": ActionSpec(
            "set_humidity(target_humidity)",
            lambda args, device: {
                "power": True,
                "target_humidity": _int_arg(args, "target_humidity", 60, 30, 80),
            },
        ),
        "set_mist_level": ActionSpec(
            "set_mist_level(mist_level)",
            lambda args, device: {
                "power": True,
                "mist_level": args.get("mist_level", "auto"),
            },
        ),
    }),
    "control_water_heater": ToolSpec(DeviceType.WATER_HEATER, {
        "on": ActionSpec("on", lambda args, device: {"power": True}),
        "off": ActionSpec("off", lambda args, device: {"power": False}),
        "set_temp": ActionSpec(
            "set_temp(target_temp)",
            lambda args, device: {
                "power": True,
                "target_temp": _int_arg(args, "target_temp", 45, 35, 75),
            },
        ),
    }),
    "control_lock": ToolSpec(DeviceType.LOCK, {
        "lock": ActionSpec("lock", lambda args, device: {"locked": True}),
        "unlock": ActionSpec("unlock", lambda args, device: {"locked": False}),
    }),
    "control_kettle": ToolSpec(DeviceType.KETTLE, {
        "on": ActionSpec("on", lambda args, device: {"power": True}),
        "off": ActionSpec("off", lambda args, device: {"power": False}),
        "set_temp": ActionSpec(
            "set_temp(target_temp)",
            lambda args, device: {
                "power": True,
                "target_temp": _int_arg(args, "target_temp", 100, 40, 100),
            },
        ),
        "boil": ActionSpec(
            "boil",
            lambda args, device: {"power": True, "target_temp": 100},
        ),
    }),
}

# 以下两个常量都是 DEVICE_ACTION_SPECS 的派生视图，不要手写。
PLANNING_TOOL_NAMES = tuple(DEVICE_ACTION_SPECS)
TOOL_ACTIONS: dict[str, str] = {
    tool_name: " / ".join(action.signature for action in spec.actions.values())
    for tool_name, spec in DEVICE_ACTION_SPECS.items()
}


class PlanStep(BaseModel):
    """One executable and independently verifiable device operation."""

    step_id: int = Field(ge=1)
    description: str = Field(min_length=1)
    # 这里必须是静态字面量：structured output 靠它生成 JSON Schema 的 enum，
    # 从而在模型侧就约束住工具名。所以它无法从 DEVICE_ACTION_SPECS 派生，
    # 只能手写 + 由一致性用例钉住两者的键集相等（漏加时测试失败，而非运行时静默）。
    # 传感器故意不在其中：规划分支只做写操作，read_sensor 不该出现在计划里。
    tool_name: Literal[
        "control_light", "control_ac", "control_tv", "control_curtain",
        "control_humidifier", "control_water_heater", "control_lock",
        "control_kettle",
    ]
    arguments: dict[str, Any]


class ExecutionPlan(BaseModel):
    """A bounded sequential plan generated by the planner model."""

    goal: str = Field(min_length=1)
    rationale: str = ""
    steps: list[PlanStep] = Field(min_length=2, max_length=8)

    @model_validator(mode="after")
    def normalize_step_ids(self):
        self.steps = [
            step.model_copy(update={"step_id": index})
            for index, step in enumerate(self.steps, start=1)
        ]
        return self


class VerificationResult(BaseModel):
    """Machine-checkable result produced after one plan step executes."""

    success: bool
    problem_type: Literal[
        "none", "device_not_found", "tool_error", "state_mismatch", "unsupported_action"
    ]
    reason: str
    actual_state: dict[str, Any] = Field(default_factory=dict)
    expected_state: dict[str, Any] = Field(default_factory=dict)


def should_use_planner(text: str) -> bool:
    """Conservatively route explicit custom multi-action requests to Planner."""
    normalized = text.strip()
    if not normalized:
        return False

    # Predefined scene requests remain on the existing ReAct + scene approval path.
    if any(marker in normalized for marker in (
        "回家模式", "离家模式", "睡眠模式", "观影模式", "起床模式",
        "我回来了", "我要出门", "我要睡", "看电影", "起床了",
    )):
        return False

    action_patterns = (
        r"打开", r"开启", r"关闭", r"关掉", r"调到", r"设为",
        r"调成", r"拉开", r"拉上", r"静音", r"切换",
    )
    action_count = sum(len(re.findall(pattern, normalized)) for pattern in action_patterns)
    device_kinds = sum(
        1 for keyword in ("灯", "空调", "电视", "窗帘", "加湿器", "热水器", "门锁", "烧水壶")
        if keyword in normalized
    )
    connectors = any(
        connector in normalized
        for connector in ("并且", "然后", "同时", "再把", "再将", "以及", "顺便")
    )
    return action_count >= 2 and (device_kinds >= 2 or connectors)


def planner_prompt(
    user_goal: str,
    registry: DeviceRegistry,
    memory_context: str,
    failure_feedback: str = "",
) -> str:
    """Build the strict prompt used for initial planning and replanning."""
    feedback = failure_feedback or "无，这是第一次规划。"
    actions = "\n".join(f"  · {tool}: {spec}" for tool, spec in TOOL_ACTIONS.items())
    return f"""你是智能家居任务规划器。请把用户目标拆成 2 到 8 个顺序执行的原子步骤。

用户目标：{user_goal}

可用设备：
{registry.get_device_list_prompt()}

相关长期记忆：
{memory_context}

上一次执行失败信息：
{feedback}

规则：
1. 每一步只能调用一个工具，工具必须是 {', '.join(PLANNING_TOOL_NAMES)} 之一。
2. arguments.action 只能取下列合法值，括号内是该 action 需要附带的参数；
   不要使用 turn_on / turn_off 这类其他平台的命名：
{actions}
3. arguments 必须严格符合对应工具参数；device_name 使用设备中文名称。
4. 不使用 activate_scene，因为本分支用于自定义多步骤目标。
5. 不添加用户没有要求的设备操作。
6. 如果是重新规划，应针对失败原因调整步骤或参数，但仍保持用户原目标。
7. 只输出结构化 ExecutionPlan，不输出额外文本。
"""


def plan_approval_payload(plan: dict[str, Any]) -> dict[str, Any]:
    """Create a serializable interrupt payload for a generated plan."""
    steps = plan.get("steps", [])
    lines = [f"{step['step_id']}. {step['description']}" for step in steps]
    summary = "\n".join(lines)
    return {
        "kind": "plan_approval",
        "question": f"已生成 {len(steps)} 步执行计划，是否开始执行？\n{summary}",
        "risk_level": "medium",
        "summary": summary,
        "plan": plan,
    }


def _unsupported_action(tool_name: str, action: Any) -> str:
    """Build an unsupported-action message that names the valid choices.

    重新规划时这条 feedback 会回喂给 Planner，所以带上合法值列表比只说
    "unsupported action" 有用得多 —— 否则模型只能凭常识反推正确写法，弱一点的
    模型可能改成 close/disable 又错一轮，白白耗掉重新规划额度。
    """
    spec = TOOL_ACTIONS.get(tool_name, "")
    suffix = f"（{tool_name} 仅支持 {spec}）" if spec else ""
    return f"unsupported action: {action}{suffix}"


def expected_state_for_step(
    step: dict[str, Any], registry: DeviceRegistry,
) -> tuple[str | None, dict[str, Any], str | None]:
    """Resolve the target and expected state before executing a plan step."""
    tool_name = step.get("tool_name")
    args = step.get("arguments", {})
    spec = DEVICE_ACTION_SPECS.get(tool_name)
    if spec is None:
        return None, {}, "unsupported tool"
    device = registry.find(str(args.get("device_name", "")), spec.device_type)
    if device is None:
        return None, {}, "device not found or ambiguous"

    action = args.get("action")
    action_spec = spec.actions.get(action) if isinstance(action, str) else None
    if action_spec is None:
        return device.device_id, {}, _unsupported_action(tool_name, action)
    try:
        return device.device_id, action_spec.expected(args, device), None
    except _InvalidArgument as exc:
        return device.device_id, {}, str(exc)


def verify_step(
    registry: DeviceRegistry,
    device_id: str | None,
    expected_state: dict[str, Any],
    tool_result: str,
    preparation_error: str | None = None,
) -> VerificationResult:
    """Verify execution using the actual registry state, not model self-report."""
    if device_id is None:
        return VerificationResult(
            success=False,
            problem_type="device_not_found",
            reason=preparation_error or "target device was not resolved",
            expected_state=expected_state,
        )
    if preparation_error:
        return VerificationResult(
            success=False,
            problem_type="unsupported_action",
            reason=preparation_error,
            expected_state=expected_state,
        )
    if tool_result.startswith("❌"):
        return VerificationResult(
            success=False,
            problem_type="tool_error",
            reason=tool_result,
            expected_state=expected_state,
        )

    device = registry.get(device_id)
    if device is None:
        return VerificationResult(
            success=False,
            problem_type="device_not_found",
            reason=f"device {device_id} disappeared after execution",
            expected_state=expected_state,
        )
    actual = {name: _plain_value(getattr(device, name, None)) for name in expected_state}
    expected = {name: _plain_value(value) for name, value in expected_state.items()}
    mismatches = {
        name: {"expected": expected[name], "actual": actual[name]}
        for name in expected
        if actual[name] != expected[name]
    }
    if mismatches:
        return VerificationResult(
            success=False,
            problem_type="state_mismatch",
            reason=f"device state mismatch: {mismatches}",
            actual_state=actual,
            expected_state=expected,
        )
    return VerificationResult(
        success=True,
        problem_type="none",
        reason="actual device state matches the expected state",
        actual_state=actual,
        expected_state=expected,
    )


def _plain_value(value: Any) -> Any:
    return getattr(value, "value", value)
