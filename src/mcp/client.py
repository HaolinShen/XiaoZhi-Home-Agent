"""
MCP 客户端
==========
连接外部 MCP 服务，将其工具集成到智能家居 Agent 中。

这是"消费侧"的 MCP 集成。通过连接外部 MCP 服务（如天气、日历、新闻等），
Agent 的能力可以从"家居控制"扩展到"生活助理"。

工作流程:
  1. 读取配置中的外部 MCP 服务列表
  2. 为每个服务建立 MCP 连接
  3. 列出服务提供的工具
  4. 将 MCP 工具转换为 LangChain Tool
  5. 合并到 Agent 的工具列表中

支持的传输方式:
  - stdio: 启动子进程，通过标准输入/输出通信
  - sse:   通过 HTTP SSE 连接到远程服务

配置方式 (.env 文件):
  EXTERNAL_MCP_SERVERS=
    {"name":"weather","transport":"stdio","command":"python","args":["weather_server.py"]},
    {"name":"calendar","transport":"sse","url":"http://localhost:8766/sse"}

注意:
  - MCP 客户端是异步的，需要 asyncio 事件循环
  - 如果外部服务不可用，Agent 会优雅降级（仅使用内置工具）
"""

import json
import asyncio
from typing import Optional
from loguru import logger

from langchain_core.tools import StructuredTool


# ============================================================
# 外部 MCP 服务配置模型
# ============================================================

class ExternalMCPService:
    """
    单个外部 MCP 服务的连接配置。

    stdio 模式示例:
      ExternalMCPService(
          name="weather",
          transport="stdio",
          command="python",
          args=["weather_mcp_server.py"],
      )

    sse 模式示例:
      ExternalMCPService(
          name="calendar",
          transport="sse",
          url="http://localhost:8766/sse",
      )
    """
    name: str
    transport: str  # "stdio" 或 "sse"
    command: Optional[str] = None     # stdio 模式: 启动命令
    args: Optional[list[str]] = None  # stdio 模式: 命令行参数
    url: Optional[str] = None         # sse 模式: 服务 URL

    def __init__(self, **kwargs):
        self.name = kwargs.get("name", "unknown")
        self.transport = kwargs.get("transport", "stdio")
        self.command = kwargs.get("command")
        self.args = kwargs.get("args", [])
        self.url = kwargs.get("url")


# ============================================================
# MCP 客户端管理
# ============================================================

class MCPClientManager:
    """
    管理多个外部 MCP 服务的连接和工具发现。

    使用方式:
      manager = MCPClientManager()
      await manager.connect_all(services_config)
      external_tools = await manager.list_all_tools()
      # 将这些工具合并到 Agent 的工具列表中
    """

    def __init__(self):
        self._sessions: dict[str, any] = {}  # name -> ClientSession
        self._tools: list = []

    async def connect_all(self, services: list[ExternalMCPService]) -> int:
        """
        连接到所有配置的外部 MCP 服务。

        参数:
          services: 外部 MCP 服务列表

        返回:
          成功连接的服务数量
        """
        from mcp.client.stdio import stdio_client
        from mcp.client.sse import sse_client
        from mcp import ClientSession

        connected = 0

        for service in services:
            try:
                logger.info(f"连接外部 MCP 服务 | name={service.name} | transport={service.transport}")

                if service.transport == "stdio":
                    # stdio 模式: 启动子进程
                    transport = stdio_client(
                        command=service.command,
                        args=service.args or [],
                    )
                elif service.transport == "sse":
                    # SSE 模式: 连接 HTTP 端点
                    transport = sse_client(service.url)
                else:
                    logger.warning(f"不支持的传输模式: {service.transport}")
                    continue

                # 建立 MCP 会话
                session = await ClientSession(transport)
                await session.initialize()
                self._sessions[service.name] = session
                connected += 1

                # 列出该服务提供的工具
                tools_result = await session.list_tools()
                logger.info(
                    f"MCP 服务已连接 | name={service.name} | "
                    f"工具数={len(tools_result.tools)}"
                )

            except Exception as e:
                logger.warning(
                    f"MCP 服务连接失败 | name={service.name} | error={e}"
                )

        logger.info(f"MCP 客户端: {connected}/{len(services)} 个服务连接成功")
        return connected

    async def list_all_tools(self) -> list:
        """
        获取所有外部 MCP 服务提供的工具列表。

        返回:
          LangChain 兼容的工具列表（可合并到 Agent 的工具中）
        """
        tools = []

        for name, session in self._sessions.items():
            try:
                mcp_tools = await session.list_tools()
                for mcp_tool in mcp_tools.tools:
                    # 将 MCP 工具包装为 LangChain StructuredTool
                    langchain_tool = self._convert_to_langchain_tool(
                        name, session, mcp_tool
                    )
                    tools.append(langchain_tool)
            except Exception as e:
                logger.warning(f"获取工具列表失败 | service={name} | error={e}")

        self._tools = tools
        logger.info(f"共发现 {len(tools)} 个外部 MCP 工具")
        return tools

    def _convert_to_langchain_tool(
        self, service_name: str, session, mcp_tool
    ) -> StructuredTool:
        """
        将 MCP 工具转换为 LangChain 可用的 StructuredTool。

        这是桥梁代码 —— 让 MCP 工具在 LangChain/LangGraph 生态中无缝工作。
        """
        async def _call_tool(**kwargs) -> str:
            """实际调用远程 MCP 工具"""
            result = await session.call_tool(mcp_tool.name, kwargs)
            return str(result.content[0].text) if result.content else str(result)

        return StructuredTool(
            name=f"{service_name}__{mcp_tool.name}",
            description=mcp_tool.description or f"来自 {service_name} 的外部工具",
            coroutine=_call_tool,  # type: ignore[arg-type]
        )

    async def close_all(self):
        """关闭所有 MCP 连接"""
        for name, session in self._sessions.items():
            try:
                await session.close()
                logger.debug(f"MCP 会话已关闭 | name={name}")
            except Exception:
                pass
        self._sessions.clear()


# ============================================================
# 工具函数: 解析配置并连接
# ============================================================

def parse_services_config(config_str: str) -> list[ExternalMCPService]:
    """
    解析 .env 中的外部 MCP 服务配置。

    参数:
      config_str: JSON 格式的服务配置字符串
                 多个服务用逗号分隔

    返回:
      ExternalMCPService 列表

    示例输入:
      '{"name":"weather","transport":"stdio","command":"python","args":["w.py"]}'
    """
    if not config_str or not config_str.strip():
        return []

    services = []
    # 支持多个配置（逗号分隔的 JSON 对象）
    # 先尝试整体解析，再尝试分段解析
    try:
        data = json.loads(config_str)
        if isinstance(data, list):
            for item in data:
                services.append(ExternalMCPService(**item))
        else:
            services.append(ExternalMCPService(**data))
    except json.JSONDecodeError:
        logger.warning(f"外部 MCP 服务配置解析失败: {config_str[:100]}...")

    return services


async def connect_external_tools(config_str: str) -> list:
    """
    一站式函数: 解析配置 → 连接服务 → 返回外部工具列表。

    这是 main.py 调用的便捷入口。

    参数:
      config_str: .env 中的 EXTERNAL_MCP_SERVERS 值

    返回:
      LangChain 兼容的外部工具列表
    """
    services = parse_services_config(config_str)
    if not services:
        logger.info("未配置外部 MCP 服务，仅使用内置工具")
        return []

    manager = MCPClientManager()
    await manager.connect_all(services)
    tools = await manager.list_all_tools()
    return tools
