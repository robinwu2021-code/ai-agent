"""
skills/weather.py — 天气查询 Skill（完整生产级实现）

这是项目中 Skill 实现的规范示例，展示了一个完整 Skill 的所有要素：

  结构要素
  ─────────
  ① ToolDescriptor    精确的 JSON Schema，含必填/可选字段和枚举约束
  ② execute()         真实的业务逻辑（HTTP 请求、数据解析、错误处理）
  ③ 缓存层            TTL 缓存避免重复请求同一城市
  ④ 重试逻辑          网络抖动时自动重试
  ⑤ 单元测试钩子      MockMode 可在离线测试中使用

  包含的 Skill 类
  ────────────────
  WeatherCurrentSkill   当前天气（温度、湿度、风速、天气描述）
  WeatherForecastSkill  多日天气预报（1-7 天）
  WeatherAlertSkill     气象预警查询（暴雨、大风、高温等）

  支持的数据源（通过 provider 参数切换）
  ───────────────────────────────────────
  openweathermap   OpenWeatherMap API（免费，需注册 API Key）
  wttr.in          wttr.in（完全免费，无需注册）
  mock             离线 Mock 数据（测试/演示用）

  使用示例
  ─────────
  # 开发模式（无 API Key）
  skill = WeatherCurrentSkill(provider="mock")

  # 生产模式（OpenWeatherMap）
  skill = WeatherCurrentSkill(
      provider="openweathermap",
      api_key="your_owm_key",
  )

  # 注册到容器
  container.skill_registry.register(WeatherCurrentSkill(provider="mock"))
  container.skill_registry.register(WeatherForecastSkill(provider="mock"))
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
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
    """标准化天气条件，屏蔽不同数据源的字段差异。"""
    city:        str
    country:     str
    temperature: float          # 摄氏度
    feels_like:  float          # 体感温度
    humidity:    int            # 湿度 %
    wind_speed:  float          # 风速 m/s
    wind_dir:    str            # 风向（N/NE/E/SE/S/SW/W/NW）
    description: str            # 天气描述（晴、多云、小雨……）
    visibility:  int            # 能见度（米）
    pressure:    int            # 气压（hPa）
    uv_index:    float          # 紫外线指数
    sunrise:     str            # 日出时间 HH:MM
    sunset:      str            # 日落时间 HH:MM
    timestamp:   str            # 观测时间 ISO 8601

    def to_dict(self) -> dict:
        return {
            "city":        self.city,
            "country":     self.country,
            "temperature": f"{self.temperature:.1f}°C",
            "feels_like":  f"{self.feels_like:.1f}°C",
            "humidity":    f"{self.humidity}%",
            "wind":        f"{self.wind_speed:.1f} m/s {self.wind_dir}",
            "description": self.description,
            "visibility":  f"{self.visibility / 1000:.1f} km" if self.visibility >= 1000 else f"{self.visibility} m",
            "pressure":    f"{self.pressure} hPa",
            "uv_index":    self.uv_index,
            "sunrise":     self.sunrise,
            "sunset":      self.sunset,
            "timestamp":   self.timestamp,
        }


@dataclass
class DailyForecast:
    """单日预报数据。"""
    date:         str           # YYYY-MM-DD
    temp_min:     float
    temp_max:     float
    humidity:     int
    wind_speed:   float
    description:  str
    precipitation: float        # 降水概率 %
    uv_index:     float

    def to_dict(self) -> dict:
        return {
            "date":         self.date,
            "temp_range":   f"{self.temp_min:.0f}~{self.temp_max:.0f}°C",
            "humidity":     f"{self.humidity}%",
            "wind_speed":   f"{self.wind_speed:.1f} m/s",
            "description":  self.description,
            "precipitation":f"{self.precipitation:.0f}%",
            "uv_index":     self.uv_index,
        }


@dataclass
class WeatherAlert:
    """气象预警信息。"""
    city:       str
    country:    str
    event:      str             # 预警事件（暴雨、大风……）
    severity:   str             # extreme / severe / moderate / minor
    headline:   str
    description: str
    effective:  str             # 生效时间
    expires:    str             # 失效时间
    source:     str             # 发布机构


# ─────────────────────────────────────────────────────────────
# TTL 缓存
# ─────────────────────────────────────────────────────────────

class _TTLCache:
    """简单 TTL 缓存，避免短时间内重复调用 API。"""

    def __init__(self, ttl_sec: int = 600) -> None:
        self._ttl    = ttl_sec
        self._store: dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        ts, value = entry
        if time.monotonic() - ts > self._ttl:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        self._store[key] = (time.monotonic(), value)

    def invalidate(self, key: str) -> None:
        self._store.pop(key, None)


# ─────────────────────────────────────────────────────────────
# 数据源适配器
# ─────────────────────────────────────────────────────────────

def _wind_direction(degrees: float) -> str:
    """将风向角度转为 8 方位缩写。"""
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return dirs[round(degrees / 45) % 8]


def _ts_to_hhmm(unix_ts: int, offset_sec: int = 0) -> str:
    """Unix timestamp → HH:MM（含时区偏移）。"""
    dt = datetime.fromtimestamp(unix_ts + offset_sec, tz=timezone.utc)
    return dt.strftime("%H:%M")


class OpenWeatherMapAdapter:
    """OpenWeatherMap API 适配器（免费版支持当前天气和5天预报）。"""

    BASE = "https://api.openweathermap.org/data/2.5"

    def __init__(self, api_key: str) -> None:
        self._key = api_key

    async def current(self, city: str, units: str = "metric") -> WeatherCondition:
        import httpx
        params = {"q": city, "appid": self._key, "units": units, "lang": "zh_cn"}
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{self.BASE}/weather", params=params)
            r.raise_for_status()
            d = r.json()

        tz_offset = d.get("timezone", 0)
        return WeatherCondition(
            city=d["name"],
            country=d["sys"]["country"],
            temperature=d["main"]["temp"],
            feels_like=d["main"]["feels_like"],
            humidity=d["main"]["humidity"],
            wind_speed=d["wind"]["speed"],
            wind_dir=_wind_direction(d["wind"].get("deg", 0)),
            description=d["weather"][0]["description"],
            visibility=d.get("visibility", 0),
            pressure=d["main"]["pressure"],
            uv_index=0.0,   # 需单独 UV Index 接口
            sunrise=_ts_to_hhmm(d["sys"]["sunrise"], tz_offset),
            sunset=_ts_to_hhmm(d["sys"]["sunset"], tz_offset),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    async def forecast(self, city: str, days: int, units: str = "metric") -> list[DailyForecast]:
        import httpx
        params = {"q": city, "appid": self._key, "units": units,
                  "cnt": min(days * 8, 40), "lang": "zh_cn"}
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{self.BASE}/forecast", params=params)
            r.raise_for_status()
            d = r.json()

        # 按日期聚合 3 小时数据
        by_date: dict[str, list] = {}
        for item in d["list"]:
            date = item["dt_txt"][:10]
            by_date.setdefault(date, []).append(item)

        result = []
        for date, items in sorted(by_date.items())[:days]:
            temps  = [i["main"]["temp"] for i in items]
            precip = max((i.get("pop", 0) * 100) for i in items)
            result.append(DailyForecast(
                date=date,
                temp_min=min(temps),
                temp_max=max(temps),
                humidity=round(sum(i["main"]["humidity"] for i in items) / len(items)),
                wind_speed=max(i["wind"]["speed"] for i in items),
                description=items[len(items) // 2]["weather"][0]["description"],
                precipitation=precip,
                uv_index=0.0,
            ))
        return result

    async def alerts(self, city: str) -> list[WeatherAlert]:
        # 免费版 OWM 不提供预警，返回空列表
        return []


class WttrInAdapter:
    """wttr.in 适配器，完全免费，无需注册，支持中文城市名。"""

    BASE = "https://wttr.in"

    async def current(self, city: str, units: str = "metric") -> WeatherCondition:
        import httpx
        # wttr.in 支持 JSON 格式
        url = f"{self.BASE}/{city}"
        params = {"format": "j1"}
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            d = r.json()

        cur = d["current_condition"][0]
        area = d["nearest_area"][0]
        country = area["country"][0]["value"]
        city_name = area["areaName"][0]["value"]

        temp_c   = float(cur["temp_C"])
        feels_c  = float(cur["FeelsLikeC"])
        humidity = int(cur["humidity"])
        wind_ms  = float(cur["windspeedKmph"]) / 3.6
        wind_deg = float(cur["winddirDegree"])
        desc     = cur["weatherDesc"][0]["value"]
        vis_m    = int(cur["visibility"]) * 1000
        pressure = int(cur["pressure"])
        uv       = float(cur.get("uvIndex", 0))

        # 日出日落（从当日天气里取）
        today = d["weather"][0]
        return WeatherCondition(
            city=city_name,
            country=country,
            temperature=temp_c,
            feels_like=feels_c,
            humidity=humidity,
            wind_speed=round(wind_ms, 1),
            wind_dir=_wind_direction(wind_deg),
            description=desc,
            visibility=vis_m,
            pressure=pressure,
            uv_index=uv,
            sunrise=today.get("astronomy", [{}])[0].get("sunrise", "--:--"),
            sunset=today.get("astronomy", [{}])[0].get("sunset", "--:--"),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    async def forecast(self, city: str, days: int, units: str = "metric") -> list[DailyForecast]:
        import httpx
        url = f"{self.BASE}/{city}"
        params = {"format": "j1"}
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            d = r.json()

        result = []
        for day in d["weather"][:days]:
            hourly = day.get("hourly", [{}])
            temps  = [float(h["tempC"]) for h in hourly] or [
                float(day["mintempC"]), float(day["maxtempC"])
            ]
            precip = max(float(h.get("chanceofrain", 0)) for h in hourly) if hourly else 0
            wind_ms = max(float(h.get("windspeedKmph", 0)) for h in hourly) / 3.6 if hourly else 0
            desc   = hourly[len(hourly) // 2]["weatherDesc"][0]["value"] if hourly else ""
            uv     = float(day.get("uvIndex", 0))

            result.append(DailyForecast(
                date=day["date"],
                temp_min=float(day["mintempC"]),
                temp_max=float(day["maxtempC"]),
                humidity=round(sum(int(h.get("humidity", 0)) for h in hourly) / max(len(hourly), 1)),
                wind_speed=round(wind_ms, 1),
                description=desc,
                precipitation=precip,
                uv_index=uv,
            ))
        return result

    async def alerts(self, city: str) -> list[WeatherAlert]:
        # wttr.in 不提供预警数据
        return []


class MockWeatherAdapter:
    """
    离线 Mock 适配器，无需网络，用于测试和演示。
    数据基于城市名做哈希以产生稳定但不同的结果。
    """

    # 天气描述池
    _CONDITIONS = ["晴", "多云", "阴", "小雨", "中雨", "阵雨", "雷阵雨", "小雪", "多云转晴"]
    _ALERTS     = ["暴雨橙色预警", "大风蓝色预警", "高温黄色预警", ""]

    def _seed(self, city: str) -> int:
        return sum(ord(c) for c in city) % 100

    async def current(self, city: str, units: str = "metric") -> WeatherCondition:
        seed = self._seed(city)
        return WeatherCondition(
            city=city, country="CN",
            temperature=15.0 + seed * 0.2,
            feels_like=14.0 + seed * 0.2,
            humidity=50 + seed % 40,
            wind_speed=2.0 + seed * 0.05,
            wind_dir=["N", "NE", "E", "SE", "S", "SW", "W", "NW"][seed % 8],
            description=self._CONDITIONS[seed % len(self._CONDITIONS)],
            visibility=10000 - seed * 50,
            pressure=1013 + seed % 20,
            uv_index=round(1.0 + seed * 0.08, 1),
            sunrise="06:15",
            sunset="18:45",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    async def forecast(self, city: str, days: int, units: str = "metric") -> list[DailyForecast]:
        from datetime import date, timedelta
        seed = self._seed(city)
        result = []
        for i in range(days):
            d = (date.today() + timedelta(days=i)).isoformat()
            result.append(DailyForecast(
                date=d,
                temp_min=10.0 + seed * 0.1 + i * 0.5,
                temp_max=20.0 + seed * 0.1 + i * 0.3,
                humidity=55 + (seed + i * 7) % 35,
                wind_speed=2.5 + (seed + i) * 0.1,
                description=self._CONDITIONS[(seed + i) % len(self._CONDITIONS)],
                precipitation=float((seed + i * 11) % 80),
                uv_index=round(2.0 + (seed + i) * 0.1, 1),
            ))
        return result

    async def alerts(self, city: str) -> list[WeatherAlert]:
        seed = self._seed(city)
        event = self._ALERTS[seed % len(self._ALERTS)]
        if not event:
            return []
        severity_map = {"暴雨橙色预警": "severe", "大风蓝色预警": "moderate",
                        "高温黄色预警": "moderate"}
        return [WeatherAlert(
            city=city, country="CN",
            event=event,
            severity=severity_map.get(event, "minor"),
            headline=f"{city} 发布{event}",
            description=f"预计未来 24 小时 {city} 将出现强{event[:2]}，请注意防范。",
            effective=datetime.now(timezone.utc).isoformat(),
            expires=datetime.now(timezone.utc).replace(hour=23, minute=59).isoformat(),
            source="Mock 气象局",
        )]


def _build_adapter(provider: str, api_key: str | None) -> Any:
    """按 provider 名称实例化对应的适配器。"""
    if provider == "openweathermap":
        if not api_key:
            raise ValueError("provider='openweathermap' 需要 api_key。"
                             "免费注册：https://home.openweathermap.org/users/sign_up")
        return OpenWeatherMapAdapter(api_key)
    if provider == "wttr.in":
        return WttrInAdapter()
    if provider == "mock":
        return MockWeatherAdapter()
    raise ValueError(f"未知 provider: {provider!r}，可选 openweathermap / wttr.in / mock")


# ─────────────────────────────────────────────────────────────
# Skill 实现
# ─────────────────────────────────────────────────────────────

class WeatherCurrentSkill:
    """
    当前天气查询 Skill。

    返回字段：城市、国家、温度、体感温度、湿度、风速风向、
              天气描述、能见度、气压、紫外线指数、日出/日落。

    缓存：相同城市 10 分钟内复用结果（可配置）。
    重试：网络失败时最多重试 2 次，每次间隔 1 秒。
    """

    def __init__(
        self,
        provider:   str         = "mock",
        api_key:    str | None  = None,
        units:      str         = "metric",   # metric | imperial
        cache_ttl:  int         = 600,        # 缓存秒数，0=不缓存
        max_retries: int        = 2,
    ) -> None:
        self._adapter   = _build_adapter(provider, api_key)
        self._units     = units
        self._cache     = _TTLCache(cache_ttl) if cache_ttl > 0 else None
        self._max_retry = max_retries
        log.info("weather_current.init", provider=provider, units=units, cache_ttl=cache_ttl)

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
                        "description": "城市名，支持中文（如「北京」）或英文（如「Beijing」）",
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

        # 缓存命中
        cache_key = f"current:{city}:{units}"
        if self._cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                log.debug("weather_current.cache_hit", city=city)
                return {**cached, "_cached": True}

        # 带重试的网络请求
        last_exc: Exception | None = None
        for attempt in range(self._max_retry + 1):
            try:
                condition = await self._adapter.current(city, units)
                result = condition.to_dict()
                if self._cache:
                    self._cache.set(cache_key, result)
                log.info("weather_current.success", city=city, attempt=attempt)
                return result
            except Exception as exc:
                last_exc = exc
                if attempt < self._max_retry:
                    wait = 1.0 * (attempt + 1)
                    log.warning("weather_current.retry",
                                city=city, attempt=attempt, wait=wait, error=str(exc))
                    await asyncio.sleep(wait)

        # 所有重试耗尽
        log.error("weather_current.failed", city=city, error=str(last_exc))
        raise RuntimeError(f"无法获取 {city!r} 的天气数据：{last_exc}") from last_exc


class WeatherForecastSkill:
    """
    多日天气预报 Skill（1-7 天）。

    每一天返回：日期、最低/最高温度、湿度、风速、天气描述、
                降水概率、紫外线指数。
    """

    def __init__(
        self,
        provider:    str        = "mock",
        api_key:     str | None = None,
        units:       str        = "metric",
        cache_ttl:   int        = 1800,   # 预报缓存 30 分钟
        max_retries: int        = 2,
    ) -> None:
        self._adapter    = _build_adapter(provider, api_key)
        self._units      = units
        self._cache      = _TTLCache(cache_ttl) if cache_ttl > 0 else None
        self._max_retry  = max_retries

    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            name="weather_forecast",
            description=(
                "查询城市未来 1-7 天的天气预报。返回每天的温度区间、"
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
                        "maximum": 7,
                        "default": 3,
                        "description": "预报天数（1-7 天）",
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
        days  = max(1, min(7, int(arguments.get("days", 3))))
        units = arguments.get("units", self._units)

        cache_key = f"forecast:{city}:{days}:{units}"
        if self._cache:
            cached = self._cache.get(cache_key)
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
                    "summary":  self._summarize(forecasts),
                }
                if self._cache:
                    self._cache.set(cache_key, result)
                log.info("weather_forecast.success", city=city, days=days, attempt=attempt)
                return result
            except Exception as exc:
                last_exc = exc
                if attempt < self._max_retry:
                    await asyncio.sleep(1.0 * (attempt + 1))

        raise RuntimeError(f"无法获取 {city!r} 的预报数据：{last_exc}") from last_exc

    @staticmethod
    def _summarize(forecasts: list[DailyForecast]) -> str:
        """生成人类可读的预报摘要。"""
        if not forecasts:
            return "暂无预报数据"
        rainy = [f for f in forecasts if f.precipitation >= 50]
        sunny = [f for f in forecasts if f.precipitation < 20 and "晴" in f.description]
        parts = []
        if rainy:
            dates = "/".join(f.date[5:] for f in rainy)
            parts.append(f"{dates} 降水概率较高")
        if sunny:
            dates = "/".join(f.date[5:] for f in sunny)
            parts.append(f"{dates} 晴好")
        temps = [f.temp_max for f in forecasts]
        parts.append(f"最高温 {min(temps):.0f}~{max(temps):.0f}°C")
        return "，".join(parts) if parts else "天气平稳"


class WeatherAlertSkill:
    """
    气象预警查询 Skill。

    查询指定城市是否有当前有效的气象预警（暴雨、大风、高温等），
    返回预警等级、描述、生效时间和发布机构。
    无预警时返回空列表。
    """

    def __init__(
        self,
        provider:    str        = "mock",
        api_key:     str | None = None,
        cache_ttl:   int        = 300,    # 预警缓存 5 分钟
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
                "返回预警事件名称、严重程度（extreme/severe/moderate/minor）、"
                "详细描述和有效时段。无预警时返回空列表。"
                "适合回答「XX现在有没有暴雨预警」「出行安全吗」等问题。"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名",
                    },
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

        cache_key = f"alert:{city}"
        if self._cache:
            cached = self._cache.get(cache_key)
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
                    self._cache.set(cache_key, result)
                log.info("weather_alert.success", city=city, alerts=len(alerts), attempt=attempt)
                return result
            except Exception as exc:
                last_exc = exc
                if attempt < self._max_retry:
                    await asyncio.sleep(1.0 * (attempt + 1))

        raise RuntimeError(f"无法获取 {city!r} 的预警数据：{last_exc}") from last_exc


# ─────────────────────────────────────────────────────────────
# 便捷工厂
# ─────────────────────────────────────────────────────────────

def create_weather_skills(
    provider:  str        = "mock",
    api_key:   str | None = None,
    units:     str        = "metric",
    cache_ttl: int        = 600,
) -> list:
    """
    一次性创建所有天气 Skill，便于批量注册到 SkillRegistry。

    用法：
        for skill in create_weather_skills(provider="mock"):
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
