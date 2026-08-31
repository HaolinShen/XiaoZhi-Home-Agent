"""Weather MCP server backed by the Caiyun Weather API (v2.6).

Caiyun only accepts coordinates, so a city name is first resolved with the
free Open-Meteo geocoding endpoint and then passed to Caiyun as ``lng,lat``
(longitude first). Set ``CAIYUN_WEATHER_TOKEN`` in ``.env``.

The token lives in the request path, so HTTP failures are reported by status
code only — never by echoing the exception, which would carry the URL into
tool results, LLM context and logs.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from dotenv import load_dotenv

from mcp.server import FastMCP

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
CAIYUN_BASE_URL = "https://api.caiyunapp.com/v2.6"

# Free Caiyun tokens cap daily forecasts well below Open-Meteo's 7.
MAX_FORECAST_DAYS = 3

# weather.json rejects very small step counts with HTTP 422; these are verified
# to be accepted and keep the combined payload small.
CURRENT_DAILY_STEPS = 3
CURRENT_HOURLY_STEPS = 24

load_dotenv()

SKYCON_TEXT = {
    "CLEAR_DAY": "晴",
    "CLEAR_NIGHT": "晴",
    "PARTLY_CLOUDY_DAY": "多云",
    "PARTLY_CLOUDY_NIGHT": "多云",
    "CLOUDY": "阴",
    "LIGHT_HAZE": "轻度雾霾",
    "MODERATE_HAZE": "中度雾霾",
    "HEAVY_HAZE": "重度雾霾",
    "LIGHT_RAIN": "小雨",
    "MODERATE_RAIN": "中雨",
    "HEAVY_RAIN": "大雨",
    "STORM_RAIN": "暴雨",
    "FOG": "雾",
    "LIGHT_SNOW": "小雪",
    "MODERATE_SNOW": "中雪",
    "HEAVY_SNOW": "大雪",
    "STORM_SNOW": "暴雪",
    "DUST": "浮尘",
    "SAND": "沙尘",
    "WIND": "大风",
}

TOKEN_MISSING_MESSAGE = "未配置彩云天气 token，请在 .env 中设置 CAIYUN_WEATHER_TOKEN。"


class MissingTokenError(RuntimeError):
    """Raised when ``CAIYUN_WEATHER_TOKEN`` is absent or blank."""


async def _get_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()


def _caiyun_url(place: dict[str, Any], endpoint: str) -> str:
    """Build a Caiyun endpoint URL. Caiyun expects longitude before latitude."""
    token = os.getenv("CAIYUN_WEATHER_TOKEN", "").strip()
    if not token:
        raise MissingTokenError(TOKEN_MISSING_MESSAGE)
    return (
        f"{CAIYUN_BASE_URL}/{token}/"
        f"{float(place['longitude']):.4f},{float(place['latitude']):.4f}/{endpoint}"
    )


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


def _skycon_text(value: Any) -> str:
    if value is None:
        return "天气状况未知"
    return SKYCON_TEXT.get(str(value), f"未知天气现象 {value}")


def _humidity_percent(value: Any) -> str:
    """Caiyun reports humidity as a 0-1 fraction, unlike Open-Meteo's percent."""
    try:
        return f"{float(value) * 100:.0f}%"
    except (TypeError, ValueError):
        return "未知"


def _air_quality_text(realtime: dict[str, Any]) -> str:
    air_quality = realtime.get("air_quality") or {}
    aqi = (air_quality.get("aqi") or {}).get("chn")
    if aqi is None:
        return ""
    description = (air_quality.get("description") or {}).get("chn")
    return f"空气质量 AQI {aqi}" + (f"（{description}）" if description else "")


def _day_label(value: Any) -> str:
    return str(value or "").split("T", 1)[0] or "未知日期"


def _caiyun_failure(data: dict[str, Any]) -> str | None:
    """Caiyun answers HTTP 200 with a non-ok status for quota and token errors."""
    if data.get("status") == "ok":
        return None
    return str(data.get("error") or data.get("status") or "未知错误")


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
            _caiyun_url(place, "weather.json"),
            {
                "lang": "zh_CN", "unit": "metric",
                "dailysteps": CURRENT_DAILY_STEPS,
                "hourlysteps": CURRENT_HOURLY_STEPS,
            },
        )
    except MissingTokenError:
        return TOKEN_MISSING_MESSAGE
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if status == 429:
            return "天气服务调用过于频繁，请稍后再试。"
        return f"天气服务返回异常状态：HTTP {status}。"
    except httpx.HTTPError:
        return "天气服务暂时不可用，请稍后再试。"

    failure = _caiyun_failure(data)
    if failure:
        return f"天气服务返回异常：{failure}"

    result = data.get("result") or {}
    realtime = result.get("realtime") or {}
    wind = realtime.get("wind") or {}
    local_precipitation = (realtime.get("precipitation") or {}).get("local") or {}

    segments = [
        f"{_location_label(place)}当前天气：{_skycon_text(realtime.get('skycon'))}",
        f"温度 {realtime.get('temperature')}°C，体感 {realtime.get('apparent_temperature')}°C",
        f"相对湿度 {_humidity_percent(realtime.get('humidity'))}",
        f"降水强度 {local_precipitation.get('intensity')} mm/h",
        f"风速 {wind.get('speed')} km/h",
    ]
    air_quality = _air_quality_text(realtime)
    if air_quality:
        segments.append(air_quality)

    text = "；".join(segments) + "。"
    keypoint = result.get("forecast_keypoint")
    return f"{text}\n{keypoint}" if keypoint else text


async def get_weather_forecast(location: str = "", days: int = 3) -> str:
    """Return a 1-3 day daily forecast for a city."""
    requested = _requested_location(location)
    if not requested:
        return "请提供城市名称，或在 .env 中设置 WEATHER_DEFAULT_LOCATION。"
    days = max(1, min(MAX_FORECAST_DAYS, days))

    try:
        place = await _resolve_location(requested)
        if place is None:
            return f"没有找到地点「{requested}」，请尝试输入更完整的城市名称。"
        data = await _get_json(
            _caiyun_url(place, "daily.json"),
            {"lang": "zh_CN", "unit": "metric", "dailysteps": days},
        )
    except MissingTokenError:
        return TOKEN_MISSING_MESSAGE
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if status == 429:
            return "天气服务调用过于频繁，请稍后再试。"
        return f"天气服务返回异常状态：HTTP {status}。"
    except httpx.HTTPError:
        return "天气服务暂时不可用，请稍后再试。"

    failure = _caiyun_failure(data)
    if failure:
        return f"天气服务返回异常：{failure}"

    daily = ((data.get("result") or {}).get("daily")) or {}
    temperatures = daily.get("temperature") or []
    skycons = daily.get("skycon") or []
    precipitation = daily.get("precipitation") or []

    lines = [f"{_location_label(place)}未来 {len(temperatures)} 天天气："]
    for index, temperature in enumerate(temperatures):
        skycon = skycons[index].get("value") if index < len(skycons) else None
        probability = (
            precipitation[index].get("probability") if index < len(precipitation) else None
        )
        lines.append(
            f"- {_day_label(temperature.get('date'))}：{_skycon_text(skycon)}，"
            f"{temperature.get('min')}–{temperature.get('max')}°C，"
            f"降水概率 {probability}%。"
        )
    return "\n".join(lines)


def create_weather_mcp_server() -> FastMCP:
    mcp = FastMCP("Caiyun Weather", log_level="ERROR")

    @mcp.tool()
    async def current_weather(location: str = "") -> str:
        """查询城市当前天气。location 留空时使用 WEATHER_DEFAULT_LOCATION。"""
        return await get_current_weather(location)

    @mcp.tool()
    async def weather_forecast(location: str = "", days: int = 3) -> str:
        """查询城市未来 1-3 天天气预报。location 留空时使用默认城市。"""
        return await get_weather_forecast(location, days)

    return mcp


def main() -> None:
    create_weather_mcp_server().run(transport="stdio")


if __name__ == "__main__":
    main()
