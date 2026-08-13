import asyncio
import json
import os
import unittest
from unittest.mock import AsyncMock, patch

from src.mcp import load_external_tools, parse_services_config
from src.mcp.weather_server import get_current_weather, get_weather_forecast

GEOCODED_HANGZHOU = {"results": [{
    "name": "杭州", "admin1": "浙江", "country": "中国",
    "latitude": 30.29, "longitude": 120.16,
}]}
GEOCODED_SHANGHAI = {"results": [{
    "name": "上海", "country": "中国",
    "latitude": 31.22, "longitude": 121.46,
}]}
TOKEN_ENV = {"CAIYUN_WEATHER_TOKEN": "test-token"}


class WeatherMCPTests(unittest.TestCase):
    def test_parse_weather_stdio_service(self):
        services = parse_services_config(json.dumps({
            "name": "weather",
            "transport": "stdio",
            "command": "{python}",
            "args": ["-m", "src.mcp.weather_server"],
        }))
        self.assertEqual(len(services), 1)
        self.assertEqual(services[0].name, "weather")

    def test_current_weather_formats_caiyun_response(self):
        responses = [
            GEOCODED_HANGZHOU,
            {"status": "ok", "result": {
                "realtime": {
                    "temperature": 28.5,
                    "apparent_temperature": 30.1,
                    "humidity": 0.7,
                    "skycon": "PARTLY_CLOUDY_DAY",
                    "wind": {"speed": 8.2, "direction": 177.0},
                    "precipitation": {"local": {"intensity": 0.0}},
                    "air_quality": {
                        "aqi": {"chn": 33, "usa": 45},
                        "description": {"chn": "优"},
                    },
                },
                "forecast_keypoint": "未来两小时不会下雨",
            }},
        ]
        weather_mock = AsyncMock(side_effect=responses)
        with patch.dict(os.environ, TOKEN_ENV), \
                patch("src.mcp.weather_server._get_json", new=weather_mock):
            result = asyncio.run(get_current_weather("杭州"))

        self.assertIn("杭州，浙江，中国", result)
        self.assertIn("多云", result)
        self.assertIn("28.5°C", result)
        self.assertIn("空气质量 AQI 33（优）", result)
        self.assertIn("未来两小时不会下雨", result)
        # Caiyun reports a 0-1 fraction; it must not be printed as "0.7%".
        self.assertIn("相对湿度 70%", result)

    def test_current_weather_requests_longitude_before_latitude(self):
        responses = [GEOCODED_HANGZHOU, {"status": "ok", "result": {"realtime": {}}}]
        weather_mock = AsyncMock(side_effect=responses)
        with patch.dict(os.environ, TOKEN_ENV), \
                patch("src.mcp.weather_server._get_json", new=weather_mock):
            asyncio.run(get_current_weather("杭州"))

        url = weather_mock.await_args_list[1].args[0]
        self.assertIn("/test-token/120.1600,30.2900/weather.json", url)

    def test_forecast_limits_days_and_formats_daily_rows(self):
        responses = [
            GEOCODED_SHANGHAI,
            {"status": "ok", "result": {"daily": {
                "temperature": [{"date": "2026-08-06T00:00+08:00", "min": 26, "max": 33}],
                "skycon": [{"date": "2026-08-06T00:00+08:00", "value": "LIGHT_RAIN"}],
                "precipitation": [{"probability": 60}],
            }}},
        ]
        weather_mock = AsyncMock(side_effect=responses)
        with patch.dict(os.environ, TOKEN_ENV), \
                patch("src.mcp.weather_server._get_json", new=weather_mock):
            result = asyncio.run(get_weather_forecast("上海", 20))

        self.assertIn("2026-08-06", result)
        self.assertIn("小雨", result)
        self.assertIn("26–33°C", result)
        self.assertEqual(weather_mock.await_args_list[1].args[1]["dailysteps"], 3)

    def test_missing_token_is_reported_without_calling_caiyun(self):
        weather_mock = AsyncMock(side_effect=[GEOCODED_HANGZHOU])
        with patch.dict(os.environ, {"CAIYUN_WEATHER_TOKEN": ""}), \
                patch("src.mcp.weather_server._get_json", new=weather_mock):
            result = asyncio.run(get_current_weather("杭州"))

        self.assertIn("CAIYUN_WEATHER_TOKEN", result)
        self.assertEqual(weather_mock.await_count, 1)

    def test_caiyun_error_status_is_surfaced(self):
        responses = [
            GEOCODED_HANGZHOU,
            {"status": "failed", "error": "token is invalid"},
        ]
        with patch.dict(os.environ, TOKEN_ENV), \
                patch("src.mcp.weather_server._get_json", new=AsyncMock(side_effect=responses)):
            result = asyncio.run(get_current_weather("杭州"))

        self.assertIn("token is invalid", result)

    def test_http_failure_does_not_leak_token(self):
        import httpx

        request = httpx.Request("GET", "https://api.caiyunapp.com/v2.6/test-token/1,1/weather.json")
        failure = httpx.HTTPStatusError(
            f"Server error for url {request.url}",
            request=request,
            response=httpx.Response(500, request=request),
        )
        with patch.dict(os.environ, TOKEN_ENV), \
                patch("src.mcp.weather_server._get_json",
                      new=AsyncMock(side_effect=[GEOCODED_HANGZHOU, failure])):
            result = asyncio.run(get_current_weather("杭州"))

        self.assertIn("HTTP 500", result)
        self.assertNotIn("test-token", result)

    def test_stdio_mcp_is_discoverable_and_callable_without_network(self):
        config = json.dumps([{
            "name": "weather",
            "transport": "stdio",
            "command": "{python}",
            "args": ["-m", "src.mcp.weather_server"],
        }])
        tools = load_external_tools(config)
        self.assertEqual(
            [tool.name for tool in tools],
            ["weather__current_weather", "weather__weather_forecast"],
        )
        with patch.dict(os.environ, {"WEATHER_DEFAULT_LOCATION": ""}):
            result = tools[0].invoke({"location": ""})
        self.assertIn("请提供城市名称", result)


if __name__ == "__main__":
    unittest.main()
