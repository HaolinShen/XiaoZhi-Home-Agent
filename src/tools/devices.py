"""
设备工具工厂
============
把 `devices/capabilities.py` 的能力声明转成 LangChain Tool。

设计要点:
  1. **闭包注入（P1）**: 工具不再通过模块级 `_registry` 单例拿依赖，而是
     `build_device_tools(registry, ...)` 返回持有 registry 的闭包工具。
     这消除了"测试忘记复位单例"和"后台执行器无身份调用"两类隐患。
  2. **单一数据源（P0）**: 工具名、JSON Schema、docstring、action 的副作用实现
     全部从能力声明生成；新增设备不再需要在这里手写 if/elif。

身份与偏好观察:
  `enable_preference_tracking` 决定设备工具是否把成功动作记为"重复操作"候选。
  - 图路径（build_graph）: True，身份从 RunnableConfig 取；缺身份直接 raise，
    让"定时动作因缺身份被判失败"这类 bug 立刻暴露，而不是静默吞掉。
  - 后台自动化执行器 / MCP 服务器: False —— 机器触发的动作不该计入
    "重复手动操作"，否则会凭空造出用户从未设过的偏好。这是显式的构造期选择，
    不再是以前那种"逐键检查后安静跳过"的隐式兜底。
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from pydantic import Field, create_model

from ..devices.base import DeviceRegistry
from ..devices.capabilities import CAPABILITIES, DeviceCapability
from ..models import DeviceType
from .memory import make_preference_recorder

PreferenceRecorder = Any  # callable(config, device_id, memory_key, memory_value) -> None


def _build_docstring(cap: DeviceCapability) -> str:
    """从能力声明生成工具 docstring（LLM 读的就是它）。"""
    lines = [cap.tool_summary, "", "使用场景:"]
    lines.extend(cap.usage_examples)
    lines.extend(["", "参数:", f"    device_name: 设备名称，如{cap.device_examples}"])
    lines.append("    action: 操作类型:")
    for action in cap.actions:
        lines.append(f"            - \"{action.name}\": {action.doc}")
    for param in cap.common_params:
        lines.append(f"    {param.name}: {param.description}")
    lines.extend(["", "返回:", "    执行结果的文本描述。"])
    return "\n".join(lines)


def _build_args_schema(cap: DeviceCapability):
    """从能力声明生成工具入参 Schema（模型看到的 JSON Schema 就是它）。"""
    fields: dict[str, Any] = {
        "device_name": (str, Field(description=f"设备名称，如{cap.device_examples}")),
        "action": (
            str,
            Field(description=" / ".join(f"{a.name}: {a.doc}" for a in cap.actions)),
        ),
    }
    for param in cap.common_params:
        fields[param.name] = (
            param.annotation,
            Field(default=param.default, description=param.description),
        )
    return create_model(f"{cap.tool_name}Input", **fields)


def _make_control_tool(
    cap: DeviceCapability, registry: DeviceRegistry, recorder: PreferenceRecorder
) -> StructuredTool:
    def _fn(
        device_name: str,
        action: str,
        config: RunnableConfig = None,  # type: ignore[assignment]  # LangChain 运行时注入 config；默认 None 仅为允许不经 LangChain 直接调用,
        **kwargs: Any,
    ) -> str:
        # config 是 RunnableConfig 类型的具名参数：LangChain 依据签名注入可信身份，
        # 而 JSON Schema 里没有它（模型看不到，也填不了）。
        args = {param.name: kwargs.get(param.name, param.default) for param in cap.common_params}
        device = registry.find(device_name, cap.device_type)
        if device is None:
            return cap.not_found_text.format(device_name=device_name)
        spec = next((a for a in cap.actions if a.name == action), None)
        if spec is None:
            supported = " / ".join(a.name for a in cap.actions)
            return f"❌ 不支持的操作「{action}」。{cap.device_label}支持: {supported}"
        if spec.precheck is not None:
            denied = spec.precheck(device, args)
            if denied:
                return denied
        text, effective = spec.handler(registry, device, args)
        if spec.preference is not None and effective:
            recorder(
                config,
                device.device_id,
                spec.preference.memory_key,
                spec.preference.value_from(effective),
            )
        return text

    _fn.__name__ = cap.tool_name
    _fn.__doc__ = _build_docstring(cap)
    return StructuredTool.from_function(
        _fn,
        name=cap.tool_name,
        description=_fn.__doc__,
        args_schema=_build_args_schema(cap),
        infer_schema=False,
    )


def _make_read_sensor_tool(registry: DeviceRegistry) -> StructuredTool:
    """读取环境传感器（只读）。传感器没有 control_xxx 能力，所以单独定义。"""

    def read_sensor(sensor_type: str, location: str = "") -> str:
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

        lines = [f"📡 **{device_type.label_cn}读数:**"]
        lines.extend(f"  · {dev.to_status_text()}" for dev in sensors.values())
        return "\n".join(lines)

    read_sensor.__doc__ = """读取环境传感器的当前数值。控制设备前先用它了解实际情况。

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
    传感器读数的文本描述。传感器不存在时返回可用房间提示。"""
    return StructuredTool.from_function(
        read_sensor,
        name="read_sensor",
        description=read_sensor.__doc__,
        args_schema=create_model(
            "ReadSensorInput",
            sensor_type=(str, Field(description="temp_humidity(温湿度) 或 presence(人体存在)")),
            location=(str, Field(default="", description="可选房间名，如\"客厅\"、\"玄关\"。留空返回全部")),
        ),
        infer_schema=False,
    )


def _make_get_device_status_tool(registry: DeviceRegistry) -> StructuredTool:
    def get_device_status(query: str = "") -> str:
        _ = query  # 保留参数给未来扩展（按类型筛选）
        # 这是一次显式的"看一眼环境"，所以先推演传感器读数。
        registry.tick_environment()
        return registry.get_status_summary()

    get_device_status.__doc__ = """查询所有智能家居设备的当前状态（含传感器读数）。

无需指定参数即可查看全部设备状态。
也可以指定类型关键词来筛选，如"灯光"、"空调"。

参数:
    query: 可选筛选词（如"灯光"只查看灯光状态）。留空返回全部设备。

返回:
    格式化设备状态报告。"""
    return StructuredTool.from_function(
        get_device_status,
        name="get_device_status",
        description=get_device_status.__doc__,
        args_schema=create_model(
            "GetDeviceStatusInput",
            query=(str, Field(default="", description="可选筛选词，留空返回全部设备")),
        ),
        infer_schema=False,
    )


def build_device_tools(
    registry: DeviceRegistry,
    memory_service=None,
    *,
    enable_preference_tracking: bool = True,
) -> list[StructuredTool]:
    """构建设备工具集：全部 control_xxx + read_sensor + get_device_status。

    registry 与 memory_service 以闭包方式持有，不再经过模块级单例。
    enable_preference_tracking=False 是后台执行器 / MCP 这类"机器触发、无可信身份"
    调用方的显式选择（见模块 docstring）。
    """
    recorder = make_preference_recorder(memory_service, enable_preference_tracking)
    tools: list[StructuredTool] = [
        _make_control_tool(cap, registry, recorder) for cap in CAPABILITIES
    ]
    tools.append(_make_read_sensor_tool(registry))
    tools.append(_make_get_device_status_tool(registry))
    return tools
