"""
skills/weather.py — 天气查询 Skill

  数据源
  ──────
  open-meteo   Open-Meteo（默认，完全免费，无需 API Key，全球覆盖）
               https://open-meteo.com/
               城市名 → 经纬度：https://geocoding-api.open-meteo.com/v1/search
               天气预报：https://api.open-meteo.com/v1/forecast
  mock         离线 Mock（单元测试 / 演示用）

  包含的 Skill 类
  ────────────────
  WeatherCurrentSkill   当前天气
  WeatherForecastSkill  多日天气预报（1-16 天）
  WeatherAlertSkill     气象预警（open-meteo 不提供，返回空列表）

  工厂函数
  ─────────
  create_weather_skills(provider="open-meteo")   批量创建所有 Skill
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import structlog

from core.models import PermissionLevel, ToolDescriptor

log = structlog.get_logger(__name__)


# ─────────────────────────────────────────────────────────────
# 数据模型
# ─────────────────────────────────────────────────────────────

@dataclass
class WeatherCondition:
    city:        str
    country:     str
    temperature: float
    feels_like:  float
    humidity:    int
    wind_speed:  float
    wind_dir:    str
    description: str
    visibility:  int        # 米
    pressure:    int        # hPa
    uv_index:    float
    sunrise:     str        # HH:MM
    sunset:      str        # HH:MM
    timestamp:   str        # ISO 8601

    def to_dict(self) -> dict:
        return {
            "city":        self.city,
            "country":     self.country,
            "temperature": f"{self.temperature:.1f}°C",
            "feels_like":  f"{self.feels_like:.1f}°C",
            "humidity":    f"{self.humidity}%",
            "wind":        f"{self.wind_speed:.1f} m/s {self.wind_dir}",
            "description": self.description,
            "visibility":  (f"{self.visibility / 1000:.1f} km"
                            if self.visibility >= 1000 else f"{self.visibility} m"),
            "pressure":    f"{self.pressure} hPa",
            "uv_index":    self.uv_index,
            "sunrise":     self.sunrise,
            "sunset":      self.sunset,
            "timestamp":   self.timestamp,
        }


@dataclass
class DailyForecast:
    date:          str
    temp_min:      float
    temp_max:      float
    humidity:      int
    wind_speed:    float
    description:   str
    precipitation: float    # 降水概率 %
    uv_index:      float

    def to_dict(self) -> dict:
        return {
            "date":          self.date,
            "temp_range":    f"{self.temp_min:.0f}~{self.temp_max:.0f}°C",
            "humidity":      f"{self.humidity}%",
            "wind_speed":    f"{self.wind_speed:.1f} m/s",
            "description":   self.description,
            "precipitation": f"{self.precipitation:.0f}%",
            "uv_index":      self.uv_index,
        }


@dataclass
class WeatherAlert:
    city:        str
    country:     str
    event:       str
    severity:    str
    headline:    str
    description: str
    effective:   str
    expires:     str
    source:      str


# ─────────────────────────────────────────────────────────────
# TTL 缓存
# ─────────────────────────────────────────────────────────────

class _TTLCache:
    def __init__(self, ttl_sec: int = 600) -> None:
        self._ttl   = ttl_sec
        self._store: dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if not entry:
            return None
        ts, value = entry
        if time.monotonic() - ts > self._ttl:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        self._store[key] = (time.monotonic(), value)


# ─────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────

def _wind_direction(degrees: float) -> str:
    dirs = ["北", "东北", "东", "东南", "南", "西南", "西", "西北"]
    return dirs[round(degrees / 45) % 8]


# WMO Weather Interpretation Codes → 中文描述
_WMO_CODE: dict[int, str] = {
    0: "晴",
    1: "大部晴朗", 2: "局部多云", 3: "阴",
    45: "雾", 48: "冻雾",
    51: "小毛毛雨", 53: "中毛毛雨", 55: "大毛毛雨",
    56: "小冻毛毛雨", 57: "大冻毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    66: "小冻雨", 67: "大冻雨",
    71: "小雪", 73: "中雪", 75: "大雪", 77: "雪粒",
    80: "小阵雨", 81: "中阵雨", 82: "强阵雨",
    85: "小阵雪", 86: "大阵雪",
    95: "雷暴", 96: "雷暴伴小冰雹", 99: "雷暴伴大冰雹",
}

def _wmo_desc(code: int | None) -> str:
    if code is None:
        return "未知"
    return _WMO_CODE.get(int(code), f"天气代码{code}")


# ─────────────────────────────────────────────────────────────
# Open-Meteo 适配器
# ─────────────────────────────────────────────────────────────

class OpenMeteoAdapter:
    """
    Open-Meteo 适配器。

    完全免费，无需 API Key，全球城市覆盖，小时/日级预报，最多 16 天。
    文档：https://open-meteo.com/en/docs

    性能优化：
      • 持久化 httpx.AsyncClient（HTTP keep-alive，TCP 连接复用）
        首次请求 ~1800ms（含 DNS + 2×TCP 握手），热路径 ~220ms（仅 1 次数据请求）
      • 地理编码结果缓存 1 小时（_geo_cache），重复城市无需再次 HTTP 请求

    ⚠ 国内网络注意事项：
      open-meteo.com 在部分国内网络环境下访问不稳定或被墙。
      如遇超时，解决方案：
        1. 配置 HTTP 代理（.env 中设置 LLM_HTTP_PROXY=http://127.0.0.1:7890）
        2. 申请和风天气免费 API（https://dev.qweather.com/），
           并设置环境变量 QWEATHER_API_KEY，系统将自动切换到国内数据源
    """

    GEO_URL     = "https://geocoding-api.open-meteo.com/v1/search"
    WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
    TIMEOUT_SEC = 20   # 国内网络延迟较高，适当放大

    def __init__(self) -> None:
        import httpx
        # 持久化 client：复用 TCP 连接（keep-alive），避免每次请求重新握手
        # connect_timeout 单独控制握手超时，read_timeout 控制数据传输超时
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=self.TIMEOUT_SEC, read=self.TIMEOUT_SEC,
                                  write=5.0, pool=5.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )
        # 城市名 → (lat, lng, country, timezone) 的进程级缓存（1 小时 TTL）
        self._geo_cache = _TTLCache(ttl_sec=3600)

    async def close(self) -> None:
        """释放底层 TCP 连接池（服务关闭时调用）。"""
        await self._client.aclose()

    # ── 地理编码 ──────────────────────────────────────────────

    async def _geocode(self, city: str) -> tuple[float, float, str, str]:
        """城市名 → (latitude, longitude, country, timezone)，结果缓存 1 小时。"""
        import time as _time
        cached = self._geo_cache.get(city)
        if cached:
            log.debug("weather.geocode_cache_hit", city=city)
            return cached

        t0 = _time.perf_counter()
        r = await self._client.get(
            self.GEO_URL,
            params={"name": city, "count": 1, "language": "zh", "format": "json"},
        )
        r.raise_for_status()
        d = r.json()
        log.debug("weather.geocode_http", city=city,
                  elapsed_ms=round((_time.perf_counter() - t0) * 1000))

        results = d.get("results") or []
        if not results:
            raise ValueError(
                f"Open-Meteo 找不到城市 {city!r}，"
                "请尝试英文城市名（如 Shanghai）或 '城市名,国家代码'（如 '广州,CN'）"
            )

        loc     = results[0]
        lat     = float(loc["latitude"])
        lng     = float(loc["longitude"])
        country = loc.get("country", "")
        tz      = loc.get("timezone", "auto")

        self._geo_cache.set(city, (lat, lng, country, tz))
        return lat, lng, country, tz

    # ── 当前天气 ──────────────────────────────────────────────

    async def current(self, city: str, units: str = "metric") -> WeatherCondition:
        import time as _time
        t_total = _time.perf_counter()

        t0 = _time.perf_counter()
        lat, lng, country, tz = await self._geocode(city)
        log.debug("weather.step_geocode", city=city,
                  elapsed_ms=round((_time.perf_counter() - t0) * 1000))

        t1 = _time.perf_counter()
        r = await self._client.get(
            self.WEATHER_URL,
            params={
                "latitude":  lat,
                "longitude": lng,
                "timezone":  tz,
                "current": ",".join([
                    "temperature_2m",
                    "relative_humidity_2m",
                    "apparent_temperature",
                    "weather_code",
                    "wind_speed_10m",
                    "wind_direction_10m",
                    "surface_pressure",
                    "visibility",
                    "uv_index",
                    "precipitation",
                ]),
                "daily":         "sunrise,sunset",
                "forecast_days": 1,
                "wind_speed_unit": "ms",
            },
        )
        r.raise_for_status()
        d = r.json()
        log.debug("weather.step_fetch_current", city=city,
                  elapsed_ms=round((_time.perf_counter() - t1) * 1000),
                  total_ms=round((_time.perf_counter() - t_total) * 1000))

        cur   = d.get("current", {})
        daily = d.get("daily", {})

        sunrise = (daily.get("sunrise") or [""])[0]
        sunset  = (daily.get("sunset")  or [""])[0]
        sunrise = sunrise[11:16] if len(sunrise) > 10 else sunrise
        sunset  = sunset[11:16]  if len(sunset)  > 10 else sunset

        return WeatherCondition(
            city        = city,
            country     = country,
            temperature = float(cur.get("temperature_2m") or 0),
            feels_like  = float(cur.get("apparent_temperature") or 0),
            humidity    = int(cur.get("relative_humidity_2m") or 0),
            wind_speed  = float(cur.get("wind_speed_10m") or 0),
            wind_dir    = _wind_direction(float(cur.get("wind_direction_10m") or 0)),
            description = _wmo_desc(cur.get("weather_code")),
            visibility  = int(float(cur.get("visibility") or 0)),
            pressure    = int(float(cur.get("surface_pressure") or 0)),
            uv_index    = float(cur.get("uv_index") or 0),
            sunrise     = sunrise,
            sunset      = sunset,
            timestamp   = datetime.now(timezone.utc).isoformat(),
        )

    # ── 多日预报 ──────────────────────────────────────────────

    async def forecast(self, city: str, days: int, units: str = "metric") -> list[DailyForecast]:
        import time as _time
        days = max(1, min(16, days))
        t_total = _time.perf_counter()

        t0 = _time.perf_counter()
        lat, lng, country, tz = await self._geocode(city)
        log.debug("weather.step_geocode", city=city,
                  elapsed_ms=round((_time.perf_counter() - t0) * 1000))

        t1 = _time.perf_counter()
        r = await self._client.get(
            self.WEATHER_URL,
            params={
                "latitude":      lat,
                "longitude":     lng,
                "timezone":      tz,
                "forecast_days": days,
                "daily": ",".join([
                    "weather_code",
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "precipitation_probability_max",
                    "precipitation_sum",
                    "wind_speed_10m_max",
                    "uv_index_max",
                    "relative_humidity_2m_max",
                ]),
                "wind_speed_unit": "ms",
            },
        )
        r.raise_for_status()
        d = r.json()
        log.debug("weather.step_fetch_forecast", city=city, days=days,
                  elapsed_ms=round((_time.perf_counter() - t1) * 1000),
                  total_ms=round((_time.perf_counter() - t_total) * 1000))

        daily = d.get("daily", {})
        dates  = daily.get("time") or []
        codes  = daily.get("weather_code") or []
        t_max  = daily.get("temperature_2m_max") or []
        t_min  = daily.get("temperature_2m_min") or []
        precip = daily.get("precipitation_probability_max") or []
        wind   = daily.get("wind_speed_10m_max") or []
        uv     = daily.get("uv_index_max") or []
        hum    = daily.get("relative_humidity_2m_max") or []

        result = []
        for i, date in enumerate(dates[:days]):
            def _v(lst, idx, default=0):
                try:
                    v = lst[idx]
                    return v if v is not None else default
                except IndexError:
                    return default

            result.append(DailyForecast(
                date          = date,
                temp_max      = float(_v(t_max, i)),
                temp_min      = float(_v(t_min, i)),
                humidity      = int(_v(hum, i)),
                wind_speed    = float(_v(wind, i)),
                description   = _wmo_desc(_v(codes, i, None)),
                precipitation = float(_v(precip, i)),
                uv_index      = float(_v(uv, i)),
            ))
        return result

    # ── 预警（Open-Meteo 不提供，返回空列表）──────────────────

    async def alerts(self, city: str) -> list[WeatherAlert]:
        return []


# ─────────────────────────────────────────────────────────────
# Mock 适配器（离线测试用）
# ─────────────────────────────────────────────────────────────

class MockWeatherAdapter:
    _CONDITIONS = ["晴", "多云", "阴", "小雨", "中雨", "阵雨", "雷阵雨", "小雪", "多云转晴"]
    _ALERTS     = ["暴雨橙色预警", "大风蓝色预警", "高温黄色预警", ""]

    def _seed(self, city: str) -> int:
        return sum(ord(c) for c in city) % 100

    async def current(self, city: str, units: str = "metric") -> WeatherCondition:
        s = self._seed(city)
        return WeatherCondition(
            city=city, country="CN",
            temperature=15.0 + s * 0.2, feels_like=14.0 + s * 0.2,
            humidity=50 + s % 40, wind_speed=2.0 + s * 0.05,
            wind_dir=["北","东北","东","东南","南","西南","西","西北"][s % 8],
            description=self._CONDITIONS[s % len(self._CONDITIONS)],
            visibility=10000 - s * 50, pressure=1013 + s % 20,
            uv_index=round(1.0 + s * 0.08, 1),
            sunrise="06:15", sunset="18:45",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    async def forecast(self, city: str, days: int, units: str = "metric") -> list[DailyForecast]:
        from datetime import date, timedelta
        s = self._seed(city)
        return [
            DailyForecast(
                date          = (date.today() + timedelta(days=i)).isoformat(),
                temp_min      = 10.0 + s * 0.1 + i * 0.5,
                temp_max      = 20.0 + s * 0.1 + i * 0.3,
                humidity      = 55 + (s + i * 7) % 35,
                wind_speed    = 2.5 + (s + i) * 0.1,
                description   = self._CONDITIONS[(s + i) % len(self._CONDITIONS)],
                precipitation = float((s + i * 11) % 80),
                uv_index      = round(2.0 + (s + i) * 0.1, 1),
            )
            for i in range(days)
        ]

    async def alerts(self, city: str) -> list[WeatherAlert]:
        s = self._seed(city)
        event = self._ALERTS[s % len(self._ALERTS)]
        if not event:
            return []
        sev = {"暴雨橙色预警": "severe", "大风蓝色预警": "moderate", "高温黄色预警": "moderate"}
        return [WeatherAlert(
            city=city, country="CN", event=event,
            severity=sev.get(event, "minor"),
            headline=f"{city} 发布{event}",
            description=f"预计未来 24 小时将出现强{event[:2]}，请注意防范。",
            effective=datetime.now(timezone.utc).isoformat(),
            expires=datetime.now(timezone.utc).replace(hour=23, minute=59).isoformat(),
            source="Mock 气象局",
        )]


# ─────────────────────────────────────────────────────────────
# 工厂
# ─────────────────────────────────────────────────────────────

def _build_adapter(provider: str, api_key: str | None = None) -> Any:
    if provider in ("open-meteo", "open_meteo", "openmeteo"):
        return OpenMeteoAdapter()
    if provider == "mock":
        return MockWeatherAdapter()
    raise ValueError(
        f"未知 provider: {provider!r}，可选值：open-meteo / mock\n"
        "Open-Meteo 完全免费且无需 API Key：https://open-meteo.com/"
    )


# ─────────────────────────────────────────────────────────────
# Skill 实现
# ─────────────────────────────────────────────────────────────

class WeatherCurrentSkill:
    """
    当前天气查询 Skill。
    使用 Open-Meteo 免费 API，无需 Key，全球覆盖。
    """

    def __init__(
        self,
        provider:    str        = "open-meteo",
        api_key:     str | None = None,
        units:       str        = "metric",
        cache_ttl:   int        = 600,
        max_retries: int        = 2,
    ) -> None:
        self._adapter   = _build_adapter(provider, api_key)
        self._units     = units
        self._cache     = _TTLCache(cache_ttl) if cache_ttl > 0 else None
        self._max_retry = max_retries
        log.info("weather_current.init", provider=provider)

    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            name="weather_current",
            description=(
                "查询指定城市的实时天气状况。返回温度、湿度、风速风向、"
                "天气描述、能见度、气压、紫外线指数、日出/日落时间。"
                "适合回答「现在XX天气怎么样」「XX适合出行吗」等问题。"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": (
                            "城市名，支持中文（如「上海」）或英文（如「Shanghai」）；"
                            "若中文找不到，可尝试拼音或英文"
                        ),
                    },
                    "units": {
                        "type": "string",
                        "enum": ["metric", "imperial"],
                        "default": "metric",
                        "description": "温度单位：metric=摄氏度，imperial=华氏度",
                    },
                },
                "required": ["city"],
            },
            source="skill",
            permission=PermissionLevel.NETWORK,
            timeout_ms=15_000,
            tags=["weather", "current", "realtime"],
        )

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        city  = arguments["city"].strip()
        units = arguments.get("units", self._units)
        key   = f"current:{city}:{units}"

        if self._cache:
            cached = self._cache.get(key)
            if cached is not None:
                return {**cached, "_cached": True}

        last_exc: Exception | None = None
        for attempt in range(self._max_retry + 1):
            try:
                cond   = await self._adapter.current(city, units)
                result = cond.to_dict()
                if self._cache:
                    self._cache.set(key, result)
                log.info("weather_current.success", city=city, attempt=attempt)
                return result
            except Exception as exc:
                last_exc = exc
                if attempt < self._max_retry:
                    await asyncio.sleep(1.0 * (attempt + 1))
                    log.warning("weather_current.retry",
                                city=city, attempt=attempt, error=str(exc))

        log.error("weather_current.failed", city=city, error=str(last_exc))
        _raise_weather_error(city, last_exc)


class WeatherForecastSkill:
    """
    多日天气预报 Skill（1-16 天，Open-Meteo 免费支持最多 16 天）。
    """

    def __init__(
        self,
        provider:    str        = "open-meteo",
        api_key:     str | None = None,
        units:       str        = "metric",
        cache_ttl:   int        = 1800,
        max_retries: int        = 2,
    ) -> None:
        self._adapter   = _build_adapter(provider, api_key)
        self._units     = units
        self._cache     = _TTLCache(cache_ttl) if cache_ttl > 0 else None
        self._max_retry = max_retries

    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            name="weather_forecast",
            description=(
                "查询城市未来 1-16 天的天气预报。返回每天的温度区间、"
                "降水概率、风速、天气描述和紫外线指数。"
                "适合回答「这周XX天气怎么样」「周末适合去XX吗」等问题。"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名，支持中文或英文",
                    },
                    "days": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 16,
                        "default": 3,
                        "description": "预报天数（1-16 天，默认 3 天）",
                    },
                    "units": {
                        "type": "string",
                        "enum": ["metric", "imperial"],
                        "default": "metric",
                    },
                },
                "required": ["city"],
            },
            source="skill",
            permission=PermissionLevel.NETWORK,
            timeout_ms=15_000,
            tags=["weather", "forecast"],
        )

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        city  = arguments["city"].strip()
        days  = max(1, min(16, int(arguments.get("days", 3))))
        units = arguments.get("units", self._units)
        key   = f"forecast:{city}:{days}:{units}"

        if self._cache:
            cached = self._cache.get(key)
            if cached is not None:
                return {**cached, "_cached": True}

        last_exc: Exception | None = None
        for attempt in range(self._max_retry + 1):
            try:
                forecasts = await self._adapter.forecast(city, days, units)
                result = {
                    "city":     city,
                    "days":     days,
                    "forecast": [f.to_dict() for f in forecasts],
                    "summary":  _summarize_forecast(forecasts),
                }
                if self._cache:
                    self._cache.set(key, result)
                log.info("weather_forecast.success", city=city, days=days, attempt=attempt)
                return result
            except Exception as exc:
                last_exc = exc
                if attempt < self._max_retry:
                    await asyncio.sleep(1.0 * (attempt + 1))
                    log.warning("weather_forecast.retry",
                                city=city, attempt=attempt, error=str(exc))

        _raise_weather_error(city, last_exc)


class WeatherAlertSkill:
    """
    气象预警查询 Skill。
    Open-Meteo 不提供官方预警接口，始终返回空列表（无预警）。
    如需真实预警数据，请配置和风天气（QWeather）并切换 provider。
    """

    def __init__(
        self,
        provider:    str        = "open-meteo",
        api_key:     str | None = None,
        cache_ttl:   int        = 300,
        max_retries: int        = 2,
    ) -> None:
        self._adapter   = _build_adapter(provider, api_key)
        self._cache     = _TTLCache(cache_ttl) if cache_ttl > 0 else None
        self._max_retry = max_retries

    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            name="weather_alert",
            description=(
                "查询城市当前生效的气象预警信息。"
                "返回预警事件名称、严重程度、详细描述和有效时段。"
                "无预警时返回空列表。"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名"},
                },
                "required": ["city"],
            },
            source="skill",
            permission=PermissionLevel.NETWORK,
            timeout_ms=10_000,
            tags=["weather", "alert", "safety"],
        )

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        city = arguments["city"].strip()
        key  = f"alert:{city}"

        if self._cache:
            cached = self._cache.get(key)
            if cached is not None:
                return {**cached, "_cached": True}

        last_exc: Exception | None = None
        for attempt in range(self._max_retry + 1):
            try:
                alerts = await self._adapter.alerts(city)
                result = {
                    "city":      city,
                    "has_alert": bool(alerts),
                    "count":     len(alerts),
                    "alerts": [
                        {
                            "event":       a.event,
                            "severity":    a.severity,
                            "headline":    a.headline,
                            "description": a.description,
                            "effective":   a.effective,
                            "expires":     a.expires,
                            "source":      a.source,
                        }
                        for a in alerts
                    ],
                }
                if self._cache:
                    self._cache.set(key, result)
                log.info("weather_alert.success", city=city, alerts=len(alerts))
                return result
            except Exception as exc:
                last_exc = exc
                if attempt < self._max_retry:
                    await asyncio.sleep(1.0 * (attempt + 1))

        _raise_weather_error(city, last_exc)


# ─────────────────────────────────────────────────────────────
# 辅助：统一错误提示
# ─────────────────────────────────────────────────────────────

def _raise_weather_error(city: str, exc: Exception) -> None:
    """将底层异常转换为带明确提示的 RuntimeError，不做任何降级。"""
    import httpx
    err_str = str(exc)
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout,
                        httpx.ReadTimeout, httpx.TimeoutException)):
        msg = (
            f"无法连接天气服务（open-meteo.com），城市={city!r}：{err_str}\n"
            "可能原因：国内网络访问 open-meteo.com 被限制。\n"
            "解决方案：\n"
            "  1. 配置 HTTP 代理：在 .env 中添加 LLM_HTTP_PROXY=http://127.0.0.1:7890\n"
            "  2. 申请和风天气 API（https://dev.qweather.com/ 免费 1000次/天），\n"
            "     设置环境变量 QWEATHER_API_KEY 后重启服务"
        )
    else:
        msg = f"无法获取 {city!r} 的天气数据：{err_str}"
    raise RuntimeError(msg) from exc


# ─────────────────────────────────────────────────────────────
# 辅助：预报摘要
# ─────────────────────────────────────────────────────────────

def _summarize_forecast(forecasts: list[DailyForecast]) -> str:
    if not forecasts:
        return "暂无预报数据"
    rainy = [f for f in forecasts if f.precipitation >= 50]
    sunny = [f for f in forecasts if f.precipitation < 20 and "晴" in f.description]
    parts = []
    if rainy:
        parts.append(f"{'/'.join(f.date[5:] for f in rainy)} 降水概率较高")
    if sunny:
        parts.append(f"{'/'.join(f.date[5:] for f in sunny)} 晴好")
    temps = [f.temp_max for f in forecasts]
    parts.append(f"最高温 {min(temps):.0f}~{max(temps):.0f}°C")
    return "，".join(parts) if parts else "天气平稳"


# ─────────────────────────────────────────────────────────────
# 便捷工厂
# ─────────────────────────────────────────────────────────────

def create_weather_skills(
    provider:  str        = "open-meteo",
    api_key:   str | None = None,
    units:     str        = "metric",
    cache_ttl: int        = 600,
) -> list:
    """
    一次性创建全部天气 Skill，批量注册到 SkillRegistry。

    示例：
        for skill in create_weather_skills():
            registry.register(skill)
    """
    return [
        WeatherCurrentSkill(provider=provider, api_key=api_key,
                            units=units, cache_ttl=cache_ttl),
        WeatherForecastSkill(provider=provider, api_key=api_key,
                             units=units, cache_ttl=cache_ttl * 3),
        WeatherAlertSkill(provider=provider, api_key=api_key,
                          cache_ttl=cache_ttl // 2),
    ]
