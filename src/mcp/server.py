"""
MCP 服务器
==========
使用 Model Context Protocol (MCP) 将智能家居工具暴露给外部 AI 客户端。

MCP 是 Anthropic 推出的开放协议，用于标准化 AI 应用与外部工具之间的通信。
通过 MCP Server，你可以:
  - 在 Claude Desktop 中直接控制智能家居设备
  - 让其他支持 MCP 的 AI 应用使用你的设备工具
  - 实现跨应用的智能家居控制

传输模式:
  - stdio: 标准输入/输出（Claude Desktop 默认使用此模式）
  - sse:   Server-Sent Events（HTTP 长连接，适合 Web 应用）

启动方式:
  # 方式 1: stdio 模式（由 MCP 客户端如 Claude Desktop 启动）
  python -m src.mcp.server

  # 方式 2: SSE 模式（独立启动，等待客户端连接）
  python -m src.mcp.server --transport sse --port 8765

参考:
  - MCP 规范: https://modelcontextprotocol.io/
  - Python SDK: https://github.com/modelcontextprotocol/python-sdk
"""

import sys
import asyncio
from typing import Optional
from loguru import logger

from ..models import DeviceType
from ..devices.base import DeviceRegistry
from ..devices.simulator import SimulatorBackend


# ============================================================
# MCP 服务器构建
# ============================================================

def create_mcp_server(
    registry: Optional[DeviceRegistry] = None,
    server_name: str = "Smart Home Agent",
) -> "FastMCP":
    """
    创建 MCP 服务器实例，注册所有智能家居工具。

    参数:
      registry:     设备注册中心。如果为 None，自动创建模拟后端。
      server_name:  MCP 服务器名称（显示在客户端 UI 中）

    返回:
      配置好的 FastMCP 服务器实例，包含所有智能家居工具

    工作原理:
      1. 将每个 LangChain @tool 函数包装为 MCP tool
      2. MCP 客户端（如 Claude Desktop）可以发现并调用这些工具
      3. 工具调用结果通过 MCP 协议返回给客户端

    使用示例:
      from src.mcp.server import create_mcp_server
      mcp = create_mcp_server(registry)
      mcp.run(transport="stdio")
    """
    from mcp.server import FastMCP  # type: ignore[import-untyped]

    # 如果没有传入注册中心，自动创建
    if registry is None:
        backend = SimulatorBackend()
        registry = DeviceRegistry(backend)

    mcp = FastMCP(server_name)

    # ============================================================
    # 注册 MCP 工具
    # ============================================================
    # 每个 @mcp.tool() 装饰的函数会自动暴露为 MCP 工具
    # 函数签名 + docstring 自动生成 JSON Schema 给客户端

    @mcp.tool()
    async def control_light_mcp(
        device_name: str,
        action: str,
        brightness: int = 50,
        color: str = "暖白",
    ) -> str:
        """控制灯光设备。支持打开/关闭、调节亮度、调节色温。

        :param device_name: 设备名称，如"客厅灯"、"卧室灯"
        :param action: 操作: on(打开), off(关闭), set_brightness(调亮度), set_color(调色温)
        :param brightness: 亮度 0-100
        :param color: 色温描述，如"暖白"、"白光"
        """
        device = registry.find(device_name, DeviceType.LIGHT)
        if device is None:
            return f"❌ 找不到灯光设备「{device_name}」"

        if action == "on":
            registry.update(device.device_id, power=True)
        elif action == "off":
            registry.update(device.device_id, power=False)
        elif action == "set_brightness":
            registry.update(device.device_id, brightness=max(0, min(100, brightness)), power=True)
        elif action == "set_color":
            registry.update(device.device_id, color=color, power=True)
        else:
            return f"❌ 不支持的操作: {action}"
        return f"✅ {device.name}操作成功"

    @mcp.tool()
    async def control_ac_mcp(
        device_name: str,
        action: str,
        temperature: int = 26,
        mode: str = "cool",
        fan_speed: str = "auto",
    ) -> str:
        """控制空调设备。支持打开/关闭、调温、切换模式、调节风速。

        :param device_name: 设备名称，如"客厅空调"
        :param action: 操作: on, off, set_temp, set_mode, set_fan
        :param temperature: 温度 16-30°C
        :param mode: 模式: cool(制冷), heat(制热), fan(送风), dry(除湿)
        :param fan_speed: 风速: auto, low, mid, high
        """
        device = registry.find(device_name, DeviceType.AC)
        if device is None:
            return f"❌ 找不到空调设备「{device_name}」"

        mapping = {
            "on": lambda: registry.update(device.device_id, power=True, temperature=temperature, mode=mode, fan_speed=fan_speed),
            "off": lambda: registry.update(device.device_id, power=False),
            "set_temp": lambda: registry.update(device.device_id, temperature=max(16, min(30, temperature)), power=True),
            "set_mode": lambda: registry.update(device.device_id, mode=mode if mode in ("cool","heat","fan","dry") else "cool", power=True),
            "set_fan": lambda: registry.update(device.device_id, fan_speed=fan_speed if fan_speed in ("auto","low","mid","high") else "auto"),
        }
        fn = mapping.get(action)
        if fn is None:
            return f"❌ 不支持的操作: {action}"
        fn()
        return f"✅ {device.name}操作成功"

    @mcp.tool()
    async def control_tv_mcp(
        device_name: str,
        action: str,
        volume: int = 30,
        channel: str = "HDMI 1",
    ) -> str:
        """控制电视设备。支持打开/关闭、调音量、静音、切换输入源。

        :param device_name: 设备名称
        :param action: 操作: on, off, set_volume, mute, set_channel
        :param volume: 音量 0-100
        :param channel: 输入源名称
        """
        device = registry.find(device_name, DeviceType.TV)
        if device is None:
            return f"❌ 找不到电视设备「{device_name}」"

        if action == "on":
            registry.update(device.device_id, power=True)
        elif action == "off":
            registry.update(device.device_id, power=False)
        elif action == "set_volume":
            registry.update(device.device_id, volume=max(0, min(100, volume)))
        elif action == "mute":
            registry.update(device.device_id, muted=not device.muted)
        elif action == "set_channel":
            registry.update(device.device_id, channel=channel, power=True)
        else:
            return f"❌ 不支持的操作: {action}"
        return f"✅ {device.name}操作成功"

    @mcp.tool()
    async def control_curtain_mcp(
        device_name: str,
        action: str,
        percentage: int = 100,
    ) -> str:
        """控制窗帘设备。支持打开、关闭、调节开合度。

        :param device_name: 设备名称
        :param action: 操作: open, close, set_position
        :param percentage: 开合度 0-100
        """
        device = registry.find(device_name, DeviceType.CURTAIN)
        if device is None:
            return f"❌ 找不到窗帘设备「{device_name}」"

        if action == "open":
            registry.update(device.device_id, position=100)
        elif action == "close":
            registry.update(device.device_id, position=0)
        elif action == "set_position":
            registry.update(device.device_id, position=max(0, min(100, percentage)))
        else:
            return f"❌ 不支持的操作: {action}"
        return f"✅ {device.name}操作成功"

    @mcp.tool()
    async def control_humidifier_mcp(
        device_name: str,
        action: str,
        target_humidity: int = 60,
        mist_level: str = "auto",
    ) -> str:
        """控制加湿器。支持开关、目标湿度和雾量档位。

        :param device_name: 设备名称，如“客厅加湿器”
        :param action: on, off, set_humidity, set_mist_level
        :param target_humidity: 目标湿度 30-80%
        :param mist_level: auto, low, mid, high
        """
        device = registry.find(device_name, DeviceType.HUMIDIFIER)
        if device is None:
            return f"❌ 找不到加湿器设备「{device_name}」"

        if action in {"on", "set_humidity", "set_mist_level"} and device.water_level <= 0:
            return f"❌ {device.name}水箱已空，请加水后再开启。"

        if action == "on":
            registry.update(device.device_id, power=True)
        elif action == "off":
            registry.update(device.device_id, power=False)
        elif action == "set_humidity":
            registry.update(
                device.device_id,
                target_humidity=max(30, min(80, target_humidity)),
                power=True,
            )
        elif action == "set_mist_level":
            if mist_level not in {"auto", "low", "mid", "high"}:
                return f"❌ 无效的雾量档位: {mist_level}"
            registry.update(device.device_id, mist_level=mist_level, power=True)
        else:
            return f"❌ 不支持的操作: {action}"
        return f"✅ {device.name}操作成功"

    @mcp.tool()
    async def control_water_heater_mcp(
        device_name: str,
        action: str,
        target_temp: int = 45,
    ) -> str:
        """控制电热水器。支持开关和目标水温调节。

        :param device_name: 设备名称，如“卫生间电热水器”
        :param action: on, off, set_temp
        :param target_temp: 目标水温 35-75°C
        """
        device = registry.find(device_name, DeviceType.WATER_HEATER)
        if device is None:
            return f"❌ 找不到电热水器设备「{device_name}」"

        if action == "on":
            registry.update(device.device_id, power=True)
        elif action == "off":
            registry.update(device.device_id, power=False)
        elif action == "set_temp":
            registry.update(
                device.device_id,
                target_temp=max(35, min(75, target_temp)),
                power=True,
            )
        else:
            return f"❌ 不支持的操作: {action}"
        return f"✅ {device.name}操作成功"

    @mcp.tool()
    async def control_lock_mcp(device_name: str, action: str) -> str:
        """控制智能门锁上锁与解锁（解锁属于敏感动作）。

        :param device_name: 设备名称，如“玄关门锁”
        :param action: lock(上锁), unlock(解锁)
        """
        device = registry.find(device_name, DeviceType.LOCK)
        if device is None:
            return f"❌ 找不到门锁设备「{device_name}」"
        if not device.power:
            return f"❌ {device.name}离线，无法操作"

        if action == "lock":
            registry.update(device.device_id, locked=True)
        elif action == "unlock":
            registry.update(device.device_id, locked=False)
        else:
            return f"❌ 不支持的操作: {action}"
        return f"✅ {device.name}操作成功"

    @mcp.tool()
    async def control_kettle_mcp(
        device_name: str,
        action: str,
        target_temp: int = 100,
    ) -> str:
        """控制电热水壶。支持开关、目标水温和一键烧开。

        :param device_name: 设备名称，如“厨房烧水壶”
        :param action: on, off, set_temp, boil
        :param target_temp: 目标水温 40-100°C
        """
        device = registry.find(device_name, DeviceType.KETTLE)
        if device is None:
            return f"❌ 找不到电热水壶设备「{device_name}」"

        if action == "boil":
            registry.update(device.device_id, power=True, target_temp=100)
        elif action == "on":
            registry.update(device.device_id, power=True)
        elif action == "off":
            registry.update(device.device_id, power=False)
        elif action == "set_temp":
            registry.update(
                device.device_id,
                target_temp=max(40, min(100, target_temp)),
                power=True,
            )
        else:
            return f"❌ 不支持的操作: {action}"
        return f"✅ {device.name}操作成功"

    @mcp.tool()
    async def read_sensor_mcp(sensor_type: str, location: str = "") -> str:
        """读取环境传感器数值（只读）。控制设备前可先用它了解实际情况。

        :param sensor_type: temp_humidity(温湿度) 或 presence(人体存在)
        :param location: 可选房间名，如“客厅”“玄关”。留空返回全部
        """
        type_map = {
            "temp_humidity": DeviceType.TEMP_HUMIDITY_SENSOR,
            "presence": DeviceType.PRESENCE_SENSOR,
        }
        device_type = type_map.get(sensor_type)
        if device_type is None:
            return f"❌ 不支持的传感器类型: {sensor_type}"

        # 读取前推演一次，让读数反映执行器当前状态。
        registry.tick_environment()

        sensors = registry.get_by_type(device_type)
        wanted = location.strip()
        if wanted:
            sensors = {
                dev_id: dev for dev_id, dev in sensors.items()
                if wanted in dev.location or wanted in dev.name
            }
        if not sensors:
            return f"❌ 找不到{device_type.label_cn}"
        return "\n".join(dev.to_status_text() for dev in sensors.values())

    @mcp.tool()
    async def get_device_status_mcp() -> str:
        """查询所有智能家居设备的当前状态"""
        return registry.get_status_summary()

    @mcp.tool()
    async def activate_scene_mcp(scene_name: str) -> str:
        """激活智能场景模式（回家/离家/睡眠/观影/起床）

        :param scene_name: 场景名称
        """
        # 复用 scenes 模块的逻辑
        from ..tools.scenes import activate_scene as scene_fn
        return scene_fn.invoke({"name": "activate_scene", "arguments": {"scene_name": scene_name}})

    logger.info(f"MCP 服务器已创建 | name={server_name} | 工具数=11")
    return mcp


# ============================================================
# 入口: python -m src.mcp.server
# ============================================================

def main():
    """
    MCP 服务器独立启动入口。

    用法:
      # stdio 模式（Claude Desktop）
      python -m src.mcp.server

      # SSE 模式（独立服务）
      python -m src.mcp.server --transport sse --port 8765
    """
    import argparse

    parser = argparse.ArgumentParser(description="智能家居 MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="传输模式: stdio(默认, 用于 Claude Desktop) 或 sse",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="SSE 模式监听端口 (默认: 8765)",
    )
    args = parser.parse_args()

    # 初始化设备注册中心
    backend = SimulatorBackend()
    registry = DeviceRegistry(backend)

    # 创建并启动 MCP 服务器
    mcp = create_mcp_server(registry)

    if args.transport == "stdio":
        logger.info("MCP 服务器启动 (stdio 模式)")
        mcp.run(transport="stdio")
    else:
        logger.info(f"MCP 服务器启动 (SSE 模式) | port={args.port}")
        mcp.run(transport="sse", port=args.port)


if __name__ == "__main__":
    main()
