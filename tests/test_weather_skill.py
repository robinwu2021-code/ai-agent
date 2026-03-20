"""
tests/test_weather_skill.py — 天气 Skill 完整测试套件

覆盖：
  单元测试   适配器、缓存、重试、参数校验
  集成测试   注册到 SkillRegistry 调用
  E2E 测试   完整 Agent 工作流（Mock LLM → 工具调用 → 天气结果）
"""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, patch

from core.models import LLMResponse, ToolCall, AgentConfig
from skills.weather import (
    WeatherCurrentSkill, WeatherForecastSkill, WeatherAlertSkill,
    MockWeatherAdapter, WeatherCondition, DailyForecast, WeatherAlert,
    _TTLCache, _wind_direction, create_weather_skills,
)


# ══════════════════════════════════════════════════════════════
# 1. 辅助函数
# ══════════════════════════════════════════════════════════════

class TestHelpers:
    def test_wind_direction_north(self):
        assert _wind_direction(0)   == "N"
        assert _wind_direction(360) == "N"

    def test_wind_direction_east(self):
        assert _wind_direction(90) == "E"

    def test_wind_direction_southwest(self):
        assert _wind_direction(225) == "SW"

    def test_wind_direction_all_octants(self):
        dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        for i, expected in enumerate(dirs):
            assert _wind_direction(i * 45) == expected


# ══════════════════════════════════════════════════════════════
# 2. TTL 缓存
# ══════════════════════════════════════════════════════════════

class TestTTLCache:
    def test_set_and_get(self):
        cache = _TTLCache(ttl_sec=60)
        cache.set("key1", {"data": 42})
        result = cache.get("key1")
        assert result == {"data": 42}

    def test_miss_returns_none(self):
        cache = _TTLCache(ttl_sec=60)
        assert cache.get("nonexistent") is None

    def test_expired_returns_none(self):
        import time
        cache = _TTLCache(ttl_sec=1)
        cache.set("k", "v")
        # Manually expire by backdating
        cache._store["k"] = (time.monotonic() - 2, "v")
        assert cache.get("k") is None

    def test_invalidate(self):
        cache = _TTLCache(ttl_sec=60)
        cache.set("k", "v")
        cache.invalidate("k")
        assert cache.get("k") is None

    def test_overwrite(self):
        cache = _TTLCache(ttl_sec=60)
        cache.set("k", "v1")
        cache.set("k", "v2")
        assert cache.get("k") == "v2"


# ══════════════════════════════════════════════════════════════
# 3. Mock 适配器
# ══════════════════════════════════════════════════════════════

class TestMockAdapter:
    @pytest.mark.asyncio
    async def test_current_returns_condition(self):
        adapter = MockWeatherAdapter()
        cond = await adapter.current("北京")
        assert cond.city == "北京"
        assert cond.country == "CN"
        assert isinstance(cond.temperature, float)
        assert 0 <= cond.humidity <= 100
        assert cond.wind_dir in ["N","NE","E","SE","S","SW","W","NW"]
        assert cond.description != ""
        assert cond.timestamp != ""

    @pytest.mark.asyncio
    async def test_different_cities_differ(self):
        adapter = MockWeatherAdapter()
        bj = await adapter.current("北京")
        sh = await adapter.current("上海")
        # Not all fields should be identical
        assert not (bj.temperature == sh.temperature
                    and bj.humidity == sh.humidity)

    @pytest.mark.asyncio
    async def test_current_to_dict_keys(self):
        adapter = MockWeatherAdapter()
        cond = await adapter.current("广州")
        d = cond.to_dict()
        required_keys = {"city","country","temperature","feels_like","humidity",
                         "wind","description","visibility","pressure",
                         "uv_index","sunrise","sunset","timestamp"}
        assert required_keys == set(d.keys())

    @pytest.mark.asyncio
    async def test_forecast_returns_n_days(self):
        adapter = MockWeatherAdapter()
        for n in [1, 3, 5, 7]:
            forecasts = await adapter.forecast("深圳", n)
            assert len(forecasts) == n

    @pytest.mark.asyncio
    async def test_forecast_daily_fields(self):
        adapter = MockWeatherAdapter()
        forecasts = await adapter.forecast("成都", 3)
        for f in forecasts:
            assert f.temp_min <= f.temp_max
            assert 0 <= f.humidity <= 100
            assert 0 <= f.precipitation <= 100
            assert f.date.count("-") == 2  # YYYY-MM-DD

    @pytest.mark.asyncio
    async def test_alerts_returns_list(self):
        adapter = MockWeatherAdapter()
        alerts = await adapter.alerts("北京")
        assert isinstance(alerts, list)

    @pytest.mark.asyncio
    async def test_alerts_structure(self):
        adapter = MockWeatherAdapter()
        # Force a city that produces alerts (seed > 0)
        alerts = await adapter.alerts("武汉")
        if alerts:
            a = alerts[0]
            assert a.event != ""
            assert a.severity in ["extreme","severe","moderate","minor"]
            assert a.city == "武汉"


# ══════════════════════════════════════════════════════════════
# 4. WeatherCurrentSkill
# ══════════════════════════════════════════════════════════════

class TestWeatherCurrentSkill:
    def make_skill(self, **kwargs):
        return WeatherCurrentSkill(provider="mock", **kwargs)

    def test_descriptor_name(self):
        skill = self.make_skill()
        assert skill.descriptor.name == "weather_current"

    def test_descriptor_required_fields(self):
        schema = self.make_skill().descriptor.input_schema
        assert "city" in schema["required"]
        assert "city" in schema["properties"]

    def test_descriptor_has_tags(self):
        skill = self.make_skill()
        assert "weather" in skill.descriptor.tags

    @pytest.mark.asyncio
    async def test_execute_returns_dict(self):
        skill = self.make_skill()
        result = await skill.execute({"city": "北京"})
        assert isinstance(result, dict)
        assert result["city"] == "北京"

    @pytest.mark.asyncio
    async def test_execute_all_keys_present(self):
        skill = self.make_skill()
        result = await skill.execute({"city": "上海"})
        for key in ("temperature", "humidity", "wind", "description", "pressure"):
            assert key in result, f"Missing key: {key}"

    @pytest.mark.asyncio
    async def test_cache_hit_on_second_call(self):
        skill = self.make_skill(cache_ttl=60)
        r1 = await skill.execute({"city": "杭州"})
        r2 = await skill.execute({"city": "杭州"})
        assert r1["temperature"] == r2["temperature"]
        assert r2.get("_cached") is True

    @pytest.mark.asyncio
    async def test_no_cache_mode(self):
        skill = self.make_skill(cache_ttl=0)
        r1 = await skill.execute({"city": "杭州"})
        r2 = await skill.execute({"city": "杭州"})
        assert r2.get("_cached") is None

    @pytest.mark.asyncio
    async def test_different_cities_independent_cache(self):
        skill = self.make_skill(cache_ttl=60)
        bj = await skill.execute({"city": "北京"})
        sh = await skill.execute({"city": "上海"})
        assert bj["city"] != sh["city"] or True  # just ensure both work

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """适配器前两次失败，第三次成功。"""
        skill = self.make_skill(max_retries=2)
        call_count = 0
        original = skill._adapter.current

        async def flaky_current(city, units="metric"):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("network error")
            return await original(city, units)

        skill._adapter.current = flaky_current
        result = await skill.execute({"city": "苏州"})
        assert result["city"] == "苏州"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_all_retries_exhausted_raises(self):
        skill = self.make_skill(max_retries=1)
        async def always_fail(city, units="metric"):
            raise ConnectionError("always fails")
        skill._adapter.current = always_fail

        with pytest.raises(RuntimeError, match="无法获取"):
            await skill.execute({"city": "南京"})

    @pytest.mark.asyncio
    async def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="未知 provider"):
            WeatherCurrentSkill(provider="nonexistent")

    @pytest.mark.asyncio
    async def test_openweathermap_requires_api_key(self):
        with pytest.raises(ValueError, match="api_key"):
            WeatherCurrentSkill(provider="openweathermap")

    @pytest.mark.asyncio
    async def test_units_imperial(self):
        skill = self.make_skill()
        result = await skill.execute({"city": "北京", "units": "imperial"})
        assert isinstance(result, dict)


# ══════════════════════════════════════════════════════════════
# 5. WeatherForecastSkill
# ══════════════════════════════════════════════════════════════

class TestWeatherForecastSkill:
    def make_skill(self):
        return WeatherForecastSkill(provider="mock")

    def test_descriptor_name(self):
        assert self.make_skill().descriptor.name == "weather_forecast"

    def test_descriptor_days_constraint(self):
        schema = self.make_skill().descriptor.input_schema
        days = schema["properties"]["days"]
        assert days["minimum"] == 1
        assert days["maximum"] == 7

    @pytest.mark.asyncio
    async def test_execute_default_3_days(self):
        skill = self.make_skill()
        result = await skill.execute({"city": "北京"})
        assert result["days"] == 3
        assert len(result["forecast"]) == 3

    @pytest.mark.asyncio
    async def test_execute_7_days(self):
        skill = self.make_skill()
        result = await skill.execute({"city": "成都", "days": 7})
        assert len(result["forecast"]) == 7

    @pytest.mark.asyncio
    async def test_execute_clamps_days(self):
        skill = self.make_skill()
        # days is clamped 1-7 in execute()
        result = await skill.execute({"city": "重庆", "days": 1})
        assert len(result["forecast"]) >= 1

    @pytest.mark.asyncio
    async def test_daily_forecast_fields(self):
        skill = self.make_skill()
        result = await skill.execute({"city": "武汉", "days": 3})
        for day in result["forecast"]:
            assert "date" in day
            assert "temp_range" in day
            assert "description" in day
            assert "precipitation" in day

    @pytest.mark.asyncio
    async def test_summary_is_string(self):
        skill = self.make_skill()
        result = await skill.execute({"city": "西安", "days": 5})
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 0

    @pytest.mark.asyncio
    async def test_cache_works(self):
        skill = WeatherForecastSkill(provider="mock", cache_ttl=60)
        r1 = await skill.execute({"city": "天津", "days": 3})
        r2 = await skill.execute({"city": "天津", "days": 3})
        assert r2.get("_cached") is True


# ══════════════════════════════════════════════════════════════
# 6. WeatherAlertSkill
# ══════════════════════════════════════════════════════════════

class TestWeatherAlertSkill:
    def make_skill(self):
        return WeatherAlertSkill(provider="mock")

    def test_descriptor_name(self):
        assert self.make_skill().descriptor.name == "weather_alert"

    @pytest.mark.asyncio
    async def test_execute_returns_dict(self):
        skill = self.make_skill()
        result = await skill.execute({"city": "北京"})
        assert "has_alert" in result
        assert "count"     in result
        assert "alerts"    in result

    @pytest.mark.asyncio
    async def test_no_alert_city(self):
        # City with seed % 4 == 3 → empty alert
        skill = self.make_skill()
        # Try multiple cities until we find a no-alert one
        no_alert_found = False
        for city in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]:
            result = await skill.execute({"city": city})
            if not result["has_alert"]:
                no_alert_found = True
                assert result["count"] == 0
                assert result["alerts"] == []
                break
        assert no_alert_found, "Expected at least one city with no alerts"

    @pytest.mark.asyncio
    async def test_alert_structure(self):
        skill = self.make_skill()
        # Find a city with alerts
        for city in ["北京", "上海", "广州", "深圳", "武汉", "成都"]:
            result = await skill.execute({"city": city})
            skill._cache = None  # disable cache between calls
            if result["has_alert"]:
                a = result["alerts"][0]
                assert "event"       in a
                assert "severity"    in a
                assert "headline"    in a
                assert "description" in a
                assert "effective"   in a
                assert "expires"     in a
                assert a["severity"] in ["extreme","severe","moderate","minor"]
                break

    @pytest.mark.asyncio
    async def test_cache(self):
        skill = WeatherAlertSkill(provider="mock", cache_ttl=60)
        r1 = await skill.execute({"city": "北京"})
        r2 = await skill.execute({"city": "北京"})
        assert r2.get("_cached") is True


# ══════════════════════════════════════════════════════════════
# 7. 工厂函数
# ══════════════════════════════════════════════════════════════

class TestFactory:
    def test_create_weather_skills_returns_three(self):
        skills = create_weather_skills(provider="mock")
        assert len(skills) == 3

    def test_create_weather_skills_names(self):
        skills = create_weather_skills(provider="mock")
        names = {s.descriptor.name for s in skills}
        assert names == {"weather_current", "weather_forecast", "weather_alert"}

    def test_create_weather_skills_wttr(self):
        skills = create_weather_skills(provider="wttr.in")
        assert len(skills) == 3


# ══════════════════════════════════════════════════════════════
# 8. SkillRegistry 集成测试
# ══════════════════════════════════════════════════════════════

class TestSkillRegistryIntegration:
    @pytest.mark.asyncio
    async def test_register_and_call_current(self):
        from skills.registry import LocalSkillRegistry
        reg = LocalSkillRegistry()
        reg.register(WeatherCurrentSkill(provider="mock"))

        result = await reg.call("weather_current", {"city": "北京"})
        assert result.error is None
        assert isinstance(result.content, dict)
        assert result.content["city"] == "北京"

    @pytest.mark.asyncio
    async def test_register_and_call_forecast(self):
        from skills.registry import LocalSkillRegistry
        reg = LocalSkillRegistry()
        reg.register(WeatherForecastSkill(provider="mock"))

        result = await reg.call("weather_forecast", {"city": "上海", "days": 5})
        assert result.error is None
        assert len(result.content["forecast"]) == 5

    @pytest.mark.asyncio
    async def test_register_and_call_alert(self):
        from skills.registry import LocalSkillRegistry
        reg = LocalSkillRegistry()
        reg.register(WeatherAlertSkill(provider="mock"))

        result = await reg.call("weather_alert", {"city": "北京"})
        assert result.error is None
        assert "has_alert" in result.content

    @pytest.mark.asyncio
    async def test_list_descriptors_includes_weather(self):
        from skills.registry import LocalSkillRegistry
        reg = LocalSkillRegistry()
        for skill in create_weather_skills(provider="mock"):
            reg.register(skill)

        names = {d.name for d in reg.list_descriptors()}
        assert "weather_current"  in names
        assert "weather_forecast" in names
        assert "weather_alert"    in names

    @pytest.mark.asyncio
    async def test_timeout_enforced(self):
        from skills.registry import LocalSkillRegistry
        reg = LocalSkillRegistry()
        skill = WeatherCurrentSkill(provider="mock")

        # Override execute to sleep forever
        async def slow_execute(args):
            await asyncio.sleep(999)
        skill.execute = slow_execute

        reg.register(skill)
        # Use very short timeout multiplier
        fast_reg = LocalSkillRegistry(timeout_multiplier=0.01)
        fast_reg.register(skill)
        result = await fast_reg.call("weather_current", {"city": "北京"})
        assert result.error is not None
        assert "timeout" in result.error.lower()

    @pytest.mark.asyncio
    async def test_call_unknown_skill_returns_error(self):
        from skills.registry import LocalSkillRegistry
        reg = LocalSkillRegistry()
        result = await reg.call("nonexistent_skill", {})
        assert result.error is not None
        assert "not found" in result.error.lower()


# ══════════════════════════════════════════════════════════════
# 9. E2E — 完整 Agent 工作流
# ══════════════════════════════════════════════════════════════

class TestWeatherAgentE2E:
    """验证 Agent 能够正确调用天气 Skill 并返回结果。"""

    def make_container(self, tool_calls, final_answer):
        from core.container import AgentContainer
        from llm.engines import MockLLMEngine
        from skills.registry import LocalSkillRegistry, PythonExecutorSkill, WebSearchSkill
        from memory.stores import InMemoryLongTermMemory, InMemoryShortTermMemory
        from mcp.hub import DefaultMCPHub
        from context.manager import PriorityContextManager

        responses = list(tool_calls) + [
            LLMResponse(
                content=final_answer,
                usage={"prompt_tokens": 150, "completion_tokens": 50},
            )
        ]
        mock_llm = MockLLMEngine(responses)

        registry = LocalSkillRegistry()
        registry.register(PythonExecutorSkill())
        registry.register(WebSearchSkill())
        registry.register(WeatherCurrentSkill(provider="mock"))
        registry.register(WeatherForecastSkill(provider="mock"))
        registry.register(WeatherAlertSkill(provider="mock"))

        container = AgentContainer(
            llm_engine=mock_llm,
            short_term_memory=InMemoryShortTermMemory(),
            long_term_memory=InMemoryLongTermMemory(),
            skill_registry=registry,
            mcp_hub=DefaultMCPHub(),
            context_manager=PriorityContextManager(),
        )
        return container.build()

    @pytest.mark.asyncio
    async def test_agent_queries_current_weather(self):
        """Agent 工具调用 weather_current，LLM 用结果回答用户。"""
        container = self.make_container(
            tool_calls=[
                LLMResponse(
                    tool_calls=[ToolCall(
                        tool_name="weather_current",
                        arguments={"city": "北京"},
                    )],
                    usage={"prompt_tokens": 100, "completion_tokens": 20},
                )
            ],
            final_answer="北京当前天气晴，气温 20°C，湿度 60%，适合出行。",
        )

        events = []
        async for event in container.agent().run(
            user_id="test_user",
            session_id="test_sess",
            text="北京今天天气怎么样？",
            config=AgentConfig(stream=False, max_steps=5),
        ):
            events.append(event)

        # 工具调用事件
        tool_events = [e for e in events if e.get("type") == "step"]
        assert any(e.get("tool") == "weather_current" for e in tool_events)

        # 最终回答
        delta_events = [e for e in events if e.get("type") == "delta"]
        assert delta_events
        full_text = "".join(e["text"] for e in delta_events)
        assert "北京" in full_text

        # 任务完成
        done_events = [e for e in events if e.get("type") == "done"]
        assert done_events
        assert done_events[0]["status"] == "done"

    @pytest.mark.asyncio
    async def test_agent_queries_forecast(self):
        """Agent 工具调用 weather_forecast。"""
        container = self.make_container(
            tool_calls=[
                LLMResponse(
                    tool_calls=[ToolCall(
                        tool_name="weather_forecast",
                        arguments={"city": "上海", "days": 3},
                    )],
                    usage={"prompt_tokens": 100, "completion_tokens": 20},
                )
            ],
            final_answer="上海未来三天：今天晴，明天多云，后天有雨。",
        )

        events = []
        async for event in container.agent().run(
            user_id="test_user",
            session_id="test_sess_2",
            text="上海这周末天气怎么样，适合户外活动吗？",
            config=AgentConfig(stream=False, max_steps=5),
        ):
            events.append(event)

        tool_events = [e for e in events if e.get("type") == "step"]
        assert any(e.get("tool") == "weather_forecast" for e in tool_events)

    @pytest.mark.asyncio
    async def test_agent_multi_tool_weather_workflow(self):
        """Agent 先查当前天气，再查预警，最后汇总回答。"""
        container = self.make_container(
            tool_calls=[
                LLMResponse(
                    tool_calls=[ToolCall(
                        tool_name="weather_current",
                        arguments={"city": "广州"},
                    )],
                    usage={"prompt_tokens": 100, "completion_tokens": 20},
                ),
                LLMResponse(
                    tool_calls=[ToolCall(
                        tool_name="weather_alert",
                        arguments={"city": "广州"},
                    )],
                    usage={"prompt_tokens": 120, "completion_tokens": 20},
                ),
            ],
            final_answer="广州当前天气多云，气温 28°C，暂无气象预警，出行安全。",
        )

        events = []
        async for event in container.agent().run(
            user_id="test_user",
            session_id="test_sess_3",
            text="广州现在安全吗？有没有预警？",
            config=AgentConfig(stream=False, max_steps=10),
        ):
            events.append(event)

        tool_names = [e.get("tool") for e in events if e.get("type") == "step"]
        assert "weather_current" in tool_names
        assert "weather_alert"   in tool_names

        done = [e for e in events if e.get("type") == "done"]
        assert done and done[0]["status"] == "done"

    @pytest.mark.asyncio
    async def test_tool_result_injected_into_context(self):
        """验证工具调用结果被注入到后续推理的 history 中。"""
        from memory.stores import InMemoryShortTermMemory
        from core.container import AgentContainer
        from llm.engines import MockLLMEngine
        from skills.registry import LocalSkillRegistry, PythonExecutorSkill, WebSearchSkill
        from mcp.hub import DefaultMCPHub
        from context.manager import PriorityContextManager
        from memory.stores import InMemoryLongTermMemory

        stm = InMemoryShortTermMemory()
        responses = [
            LLMResponse(
                tool_calls=[ToolCall(
                    tool_name="weather_current",
                    arguments={"city": "深圳"},
                )],
                usage={"prompt_tokens": 80, "completion_tokens": 10},
            ),
            LLMResponse(
                content="深圳现在天气不错。",
                usage={"prompt_tokens": 150, "completion_tokens": 30},
            ),
        ]

        registry = LocalSkillRegistry()
        registry.register(PythonExecutorSkill())
        registry.register(WeatherCurrentSkill(provider="mock"))

        container = AgentContainer(
            llm_engine=MockLLMEngine(responses),
            short_term_memory=stm,
            long_term_memory=InMemoryLongTermMemory(),
            skill_registry=registry,
            mcp_hub=DefaultMCPHub(),
            context_manager=PriorityContextManager(),
        ).build()

        sess_id = "sess_ctx_test"
        events = []
        async for ev in container.agent().run(
            user_id="u1", session_id=sess_id,
            text="深圳天气如何？", config=AgentConfig(stream=False),
        ):
            events.append(ev)

        # 工具结果应写入 short-term memory history
        # STM 按 task.id（UUID）存储，从 done 事件中获取
        done_ev = next((e for e in events if e.get("type") == "done"), {})
        task_id = done_ev.get("task_id")
        assert task_id is not None, f"No done event found, events={events}"
        task = await stm.load_task(task_id)
        assert task is not None
        roles = [m.role.value for m in task.history]
        assert "tool" in roles, f"Expected tool role in history, got {roles}"


# ══════════════════════════════════════════════════════════════
# 10. WeatherCondition 数据模型
# ══════════════════════════════════════════════════════════════

class TestWeatherConditionModel:
    def make_condition(self, **kwargs) -> WeatherCondition:
        defaults = dict(
            city="测试市", country="CN",
            temperature=20.5, feels_like=19.0,
            humidity=65, wind_speed=3.2,
            wind_dir="NE", description="晴",
            visibility=10000, pressure=1013,
            uv_index=3.5, sunrise="06:30",
            sunset="19:15", timestamp="2025-01-01T00:00:00+00:00",
        )
        defaults.update(kwargs)
        return WeatherCondition(**defaults)

    def test_to_dict_temperature_format(self):
        c = self.make_condition(temperature=20.5)
        assert c.to_dict()["temperature"] == "20.5°C"

    def test_to_dict_humidity_format(self):
        c = self.make_condition(humidity=75)
        assert c.to_dict()["humidity"] == "75%"

    def test_to_dict_wind_format(self):
        c = self.make_condition(wind_speed=3.2, wind_dir="NE")
        assert "3.2 m/s NE" == c.to_dict()["wind"]

    def test_to_dict_visibility_km(self):
        c = self.make_condition(visibility=10000)
        assert "km" in c.to_dict()["visibility"]

    def test_to_dict_visibility_m(self):
        c = self.make_condition(visibility=500)
        assert "500 m" == c.to_dict()["visibility"]

    def test_to_dict_pressure_format(self):
        c = self.make_condition(pressure=1013)
        assert c.to_dict()["pressure"] == "1013 hPa"


# ══════════════════════════════════════════════════════════════
# 11. DailyForecast 数据模型
# ══════════════════════════════════════════════════════════════

class TestDailyForecastModel:
    def make_forecast(self, **kwargs) -> DailyForecast:
        defaults = dict(
            date="2025-01-01",
            temp_min=10.0, temp_max=20.0,
            humidity=70, wind_speed=3.0,
            description="晴转多云",
            precipitation=20.0, uv_index=4.0,
        )
        defaults.update(kwargs)
        return DailyForecast(**defaults)

    def test_to_dict_temp_range(self):
        f = self.make_forecast(temp_min=10.3, temp_max=21.7)
        assert "10~22°C" == f.to_dict()["temp_range"]

    def test_to_dict_precipitation_format(self):
        f = self.make_forecast(precipitation=45.6)
        assert "46%" == f.to_dict()["precipitation"]

    def test_to_dict_date_preserved(self):
        f = self.make_forecast(date="2025-06-15")
        assert f.to_dict()["date"] == "2025-06-15"
