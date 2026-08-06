import asyncio
import json
import os
import unittest
from unittest.mock import AsyncMock, patch

from src.mcp import load_external_tools, parse_services_config
from src.mcp.weather_server import get_current_weather, get_weather_forecast


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

    def test_current_weather_formats_open_meteo_response(self):
        responses = [
            {"results": [{
                "name": "杭州", "admin1": "浙江", "country": "中国",
                "latitude": 30.29, "longitude": 120.16,
            }]},
            {"current": {
                "temperature_2m": 28.5,
                "apparent_temperature": 30.1,
                "relative_humidity_2m": 70,
                "precipitation": 0.0,
                "weather_code": 1,
                "wind_speed_10m": 8.2,
            }},
        ]
        with patch("src.mcp.weather_server._get_json", new=AsyncMock(side_effect=responses)):
            result = asyncio.run(get_current_weather("杭州"))
        self.assertIn("杭州，浙江，中国", result)
        self.assertIn("大部晴朗", result)
        self.assertIn("28.5°C", result)

    def test_forecast_limits_days_and_formats_daily_rows(self):
        responses = [
            {"results": [{
                "name": "上海", "country": "中国",
                "latitude": 31.22, "longitude": 121.46,
            }]},
            {"daily": {
                "time": ["2026-08-06"],
                "weather_code": [61],
                "temperature_2m_min": [26],
                "temperature_2m_max": [33],
                "precipitation_probability_max": [60],
            }},
        ]
        weather_mock = AsyncMock(side_effect=responses)
        with patch("src.mcp.weather_server._get_json", new=weather_mock):
            result = asyncio.run(get_weather_forecast("上海", 20))
        self.assertIn("小雨", result)
        self.assertIn("26–33°C", result)
        self.assertEqual(weather_mock.await_args_list[1].args[1]["forecast_days"], 7)

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
