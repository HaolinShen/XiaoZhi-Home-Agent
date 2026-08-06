"""Weather MCP server backed by the free Open-Meteo APIs.

No weather API key or additional Python dependency is required. The server
first resolves a city name with Open-Meteo Geocoding and then queries current
conditions or a short daily forecast.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from dotenv import load_dotenv
from mcp.server import FastMCP


GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

load_dotenv()

WEATHER_CODES = {
    0: "晴",
    1: "大部晴朗",
    2: "局部多云",
    3: "阴",
    45: "雾",
    48: "雾凇",
    51: "小毛毛雨",
    53: "毛毛雨",
    55: "较强毛毛雨",
    56: "轻微冻毛毛雨",
    57: "较强冻毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    66: "轻微冻雨",
    67: "较强冻雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    77: "米雪",
    80: "小阵雨",
    81: "中等阵雨",
    82: "强阵雨",
    85: "小阵雪",
    86: "强阵雪",
    95: "雷暴",
    96: "雷暴伴小冰雹",
    99: "雷暴伴大冰雹",
}


async def _get_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()


def _requested_location(location: str) -> str:
    return location.strip() or os.getenv("WEATHER_DEFAULT_LOCATION", "").strip()


async def _resolve_location(location: str) -> dict[str, Any] | None:
    data = await _get_json(
        GEOCODING_URL,
        {"name": location, "count": 1, "language": "zh", "format": "json"},
    )
    results = data.get("results") or []
    return results[0] if results else None


def _location_label(place: dict[str, Any]) -> str:
    parts = [place.get("name"), place.get("admin1"), place.get("country")]
    return "，".join(str(part) for part in parts if part)


def _weather_text(code: Any) -> str:
    try:
        return WEATHER_CODES.get(int(code), f"未知天气代码 {code}")
    except (TypeError, ValueError):
        return "天气状况未知"


async def get_current_weather(location: str = "") -> str:
    """Return current weather for a city, or the configured default city."""
    requested = _requested_location(location)
    if not requested:
        return "请提供城市名称，或在 .env 中设置 WEATHER_DEFAULT_LOCATION。"

    try:
        place = await _resolve_location(requested)
        if place is None:
            return f"没有找到地点「{requested}」，请尝试输入更完整的城市名称。"
        data = await _get_json(
            FORECAST_URL,
            {
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": (
                    "temperature_2m,apparent_temperature,relative_humidity_2m,"
                    "precipitation,weather_code,wind_speed_10m"
                ),
                "timezone": "auto",
            },
        )
        current = data.get("current") or {}
        return (
            f"{_location_label(place)}当前天气：{_weather_text(current.get('weather_code'))}；"
            f"温度 {current.get('temperature_2m')}°C，体感 {current.get('apparent_temperature')}°C；"
            f"相对湿度 {current.get('relative_humidity_2m')}%；"
            f"降水 {current.get('precipitation')} mm；"
            f"风速 {current.get('wind_speed_10m')} km/h。"
        )
    except httpx.HTTPError as exc:
        return f"天气服务暂时不可用：{exc}"


async def get_weather_forecast(location: str = "", days: int = 3) -> str:
    """Return a 1-7 day daily forecast for a city."""
    requested = _requested_location(location)
    if not requested:
        return "请提供城市名称，或在 .env 中设置 WEATHER_DEFAULT_LOCATION。"
    days = max(1, min(7, days))

    try:
        place = await _resolve_location(requested)
        if place is None:
            return f"没有找到地点「{requested}」，请尝试输入更完整的城市名称。"
        data = await _get_json(
            FORECAST_URL,
            {
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "daily": (
                    "weather_code,temperature_2m_max,temperature_2m_min,"
                    "precipitation_probability_max"
                ),
                "forecast_days": days,
                "timezone": "auto",
            },
        )
        daily = data.get("daily") or {}
        dates = daily.get("time") or []
        lines = [f"{_location_label(place)}未来 {len(dates)} 天天气："]
        for index, date in enumerate(dates):
            lines.append(
                f"- {date}：{_weather_text(daily['weather_code'][index])}，"
                f"{daily['temperature_2m_min'][index]}–{daily['temperature_2m_max'][index]}°C，"
                f"最高降水概率 {daily['precipitation_probability_max'][index]}%。"
            )
        return "\n".join(lines)
    except (httpx.HTTPError, KeyError, IndexError) as exc:
        return f"天气服务暂时不可用：{exc}"


def create_weather_mcp_server() -> FastMCP:
    mcp = FastMCP("Local Weather", log_level="ERROR")

    @mcp.tool()
    async def current_weather(location: str = "") -> str:
        """查询城市当前天气。location 留空时使用 WEATHER_DEFAULT_LOCATION。"""
        return await get_current_weather(location)

    @mcp.tool()
    async def weather_forecast(location: str = "", days: int = 3) -> str:
        """查询城市未来 1-7 天天气预报。location 留空时使用默认城市。"""
        return await get_weather_forecast(location, days)

    return mcp


def main() -> None:
    create_weather_mcp_server().run(transport="stdio")


if __name__ == "__main__":
    main()
