"""External MCP client adapters for LangChain/LangGraph.

The project CLI is synchronous, while the MCP Python SDK is asynchronous.
This module discovers MCP tools during startup and creates LangChain tools
that support both synchronous and asynchronous invocation. Each invocation
opens a short-lived MCP session, which keeps stdio subprocess lifecycle
management predictable and avoids leaking event-loop resources.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

from langchain_core.tools import StructuredTool
from loguru import logger
from pydantic import BaseModel, Field, create_model


@dataclass(slots=True)
class ExternalMCPService:
    """Connection settings for one external MCP server."""

    name: str = "unknown"
    transport: str = "stdio"
    command: Optional[str] = None
    args: list[str] = field(default_factory=list)
    url: Optional[str] = None
    cwd: Optional[str] = None
    env: Optional[dict[str, str]] = None


def parse_services_config(config_str: str) -> list[ExternalMCPService]:
    """Parse one JSON object or a JSON array from ``EXTERNAL_MCP_SERVERS``."""
    if not config_str or not config_str.strip():
        return []
    try:
        data = json.loads(config_str)
    except json.JSONDecodeError as exc:
        logger.warning(f"外部 MCP 服务配置解析失败: {exc}")
        return []

    items = data if isinstance(data, list) else [data]
    services: list[ExternalMCPService] = []
    for item in items:
        if not isinstance(item, dict):
            logger.warning("外部 MCP 服务配置项必须是 JSON 对象")
            continue
        try:
            service = ExternalMCPService(**item)
        except TypeError as exc:
            logger.warning(f"外部 MCP 服务配置字段无效: {exc}")
            continue
        if service.transport == "stdio" and not service.command:
            logger.warning(f"stdio MCP 服务缺少 command | name={service.name}")
            continue
        if service.transport in {"sse", "streamable_http"} and not service.url:
            logger.warning(f"远程 MCP 服务缺少 url | name={service.name}")
            continue
        services.append(service)
    return services


@asynccontextmanager
async def _open_session(service: ExternalMCPService) -> AsyncIterator[Any]:
    """Open and initialize an MCP session for any supported transport."""
    from mcp import ClientSession

    if service.transport == "stdio":
        from mcp.client.stdio import StdioServerParameters, stdio_client

        command = service.command or "python"
        if command in {"python", "python3", "{python}"}:
            command = sys.executable
        parameters = StdioServerParameters(
            command=command,
            args=service.args,
            env=service.env,
            cwd=service.cwd,
            encoding="utf-8",
            encoding_error_handler="replace",
        )
        async with stdio_client(parameters) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                yield session
        return

    if service.transport == "sse":
        from mcp.client.sse import sse_client

        async with sse_client(service.url or "") as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                yield session
        return

    if service.transport == "streamable_http":
        from mcp.client.streamable_http import streamablehttp_client

        async with streamablehttp_client(service.url or "") as streams:
            read_stream, write_stream = streams[0], streams[1]
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                yield session
        return

    raise ValueError(f"不支持的 MCP 传输模式: {service.transport}")


def _python_type(schema: dict[str, Any]) -> type:
    schema_type = schema.get("type", "string")
    return {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }.get(schema_type, Any)


def _args_model(service_name: str, tool_name: str, schema: dict[str, Any]) -> type[BaseModel]:
    """Convert the common subset of MCP JSON Schema into a Pydantic model."""
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    fields: dict[str, tuple[Any, Any]] = {}
    for name, property_schema in properties.items():
        python_type = _python_type(property_schema)
        description = property_schema.get("description", "")
        if name in required:
            fields[name] = (python_type, Field(..., description=description))
        else:
            default = property_schema.get("default", None)
            fields[name] = (Optional[python_type], Field(default, description=description))

    model_name = re.sub(r"\W+", "_", f"{service_name}_{tool_name}_Input")
    return create_model(model_name, **fields)


def _format_result(result: Any) -> str:
    texts = [getattr(block, "text", "") for block in getattr(result, "content", [])]
    text = "\n".join(item for item in texts if item)
    if getattr(result, "isError", False):
        return f"MCP 工具调用失败：{text or result}"
    return text or str(result)


async def _call_remote_tool(
    service: ExternalMCPService, tool_name: str, arguments: dict[str, Any]
) -> str:
    async with _open_session(service) as session:
        result = await session.call_tool(tool_name, arguments)
        return _format_result(result)


def _build_langchain_tool(service: ExternalMCPService, mcp_tool: Any) -> StructuredTool:
    args_schema = _args_model(service.name, mcp_tool.name, mcp_tool.inputSchema)

    async def call_async(**kwargs: Any) -> str:
        return await _call_remote_tool(service, mcp_tool.name, kwargs)

    def call_sync(**kwargs: Any) -> str:
        return asyncio.run(call_async(**kwargs))

    return StructuredTool.from_function(
        func=call_sync,
        coroutine=call_async,
        name=f"{service.name}__{mcp_tool.name}",
        description=mcp_tool.description or f"来自 {service.name} 的外部 MCP 工具",
        args_schema=args_schema,
    )


async def _discover_service_tools(service: ExternalMCPService) -> list[StructuredTool]:
    async with _open_session(service) as session:
        result = await session.list_tools()
        return [_build_langchain_tool(service, tool) for tool in result.tools]


async def connect_external_tools(config_str: str) -> list[StructuredTool]:
    """Discover configured MCP tools without keeping sessions open."""
    tools: list[StructuredTool] = []
    services = parse_services_config(config_str)
    for service in services:
        try:
            discovered = await _discover_service_tools(service)
            tools.extend(discovered)
            logger.info(f"MCP 服务已连接 | name={service.name} | 工具数={len(discovered)}")
        except Exception as exc:
            logger.warning(f"MCP 服务连接失败 | name={service.name} | error={exc}")
    return tools


def load_external_tools(config_str: str) -> list[StructuredTool]:
    """Synchronous startup helper used by the Typer CLI."""
    if not config_str or not config_str.strip():
        logger.info("未配置外部 MCP 服务，仅使用内置工具")
        return []
    return asyncio.run(connect_external_tools(config_str))
