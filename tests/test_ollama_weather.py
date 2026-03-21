"""
tests/test_ollama_weather.py — ollama-qwen3 + WeatherSkill 集成测试

验证：真实 Ollama 模型能够理解天气查询意图，调用 weather_current 工具，
并根据工具返回的数据生成自然语言回答。

所有配置（模型、地址）均从 llm.yaml 读取，测试代码中不硬编码任何参数。

运行方式：
    pytest tests/test_ollama_weather.py --run-live -v
"""
from __future__ import annotations

import pathlib
import pytest

YAML_PATH = pathlib.Path(__file__).parent.parent / "llm.yaml"
ALIAS     = "ollama-qwen3"


# ── fixtures ─────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def run_live(request):
    return request.config.getoption("--run-live", default=False)


@pytest.fixture(scope="session")
def ollama_cfg():
    from utils.llm_config import load_from_yaml
    configs, _ = load_from_yaml(YAML_PATH)
    cfg = next((c for c in configs if c.alias == ALIAS), None)
    assert cfg is not None, f"llm.yaml 中未找到 alias='{ALIAS}'"
    return cfg


@pytest.fixture(scope="session")
def container(ollama_cfg):
    """构建带天气 Skill 的 AgentContainer，引擎来自 llm.yaml。"""
    from core.container import AgentContainer
    from memory.stores import InMemoryShortTermMemory, InMemoryLongTermMemory
    from skills.registry import LocalSkillRegistry
    from skills.weather import WeatherCurrentSkill, WeatherForecastSkill
    from mcp.hub import DefaultMCPHub
    from context.manager import PriorityContextManager

    engine   = ollama_cfg.build_engine()
    registry = LocalSkillRegistry()
    # wttr.in 免费无需 API Key，适合集成测试
    registry.register(WeatherCurrentSkill(provider="wttr.in"))
    registry.register(WeatherForecastSkill(provider="wttr.in"))

    return AgentContainer(
        llm_engine        = engine,
        short_term_memory = InMemoryShortTermMemory(),
        long_term_memory  = InMemoryLongTermMemory(),
        skill_registry    = registry,
        mcp_hub           = DefaultMCPHub(),
        context_manager   = PriorityContextManager(),
    ).build()


# ══════════════════════════════════════════════════════════════════
# 集成测试（需要真实 Ollama 服务）
# ══════════════════════════════════════════════════════════════════

class TestOllamaWeather:

    @pytest.mark.anyio
    async def test_chengdu_current_weather(self, container, run_live):
        """
        主测试：询问成都今天天气，验证 Agent：
          1. 调用了 weather_current 工具
          2. 返回了包含天气信息的自然语言回答
        """
        if not run_live:
            pytest.skip("需加 --run-live 参数")

        from core.models import AgentConfig

        events = []
        async for ev in container.agent().run(
            user_id    = "test_user",
            session_id = "weather_chengdu",
            text       = "今天成都的天气怎么样？",
            config     = AgentConfig(stream=False, max_steps=5),
        ):
            events.append(ev)

        # 1. 工具被调用
        tool_events = [e for e in events if e.get("type") == "step"]
        tool_names  = [e.get("tool") for e in tool_events]
        assert "weather_current" in tool_names, (
            f"期望调用 weather_current，实际工具调用: {tool_names}\n"
            f"所有事件: {events}"
        )

        # 2. 有最终回答
        delta_events = [e for e in events if e.get("type") == "delta"]
        full_text    = "".join(e.get("text", "") for e in delta_events)
        assert len(full_text.strip()) > 0, "LLM 没有返回任何回答"

        # 3. 任务正常结束
        done_events = [e for e in events if e.get("type") == "done"]
        assert done_events, "没有收到 done 事件"
        assert done_events[0]["status"] == "done", (
            f"任务状态异常: {done_events[0]}"
        )

        print(f"\n[成都天气回答]\n{full_text}")

    @pytest.mark.anyio
    async def test_weather_tool_result_contains_city(self, container, run_live):
        """
        验证 weather_current 工具被成功调用且最终回答包含温湿度信息。
        """
        if not run_live:
            pytest.skip("需加 --run-live 参数")

        from core.models import AgentConfig

        events = []
        async for ev in container.agent().run(
            user_id    = "test_user",
            session_id = "weather_city_check",
            text       = "帮我查一下成都现在的温度和湿度",
            config     = AgentConfig(stream=False, max_steps=5),
        ):
            events.append(ev)

        # 1. 工具被成功调用（status=done，无错误）
        tool_done = [
            e for e in events
            if e.get("type") == "step"
            and e.get("tool") == "weather_current"
            and e.get("status") == "done"
            and not e.get("error")
        ]
        assert tool_done, (
            "weather_current 工具未成功调用\n"
            f"所有 step 事件: {[e for e in events if e.get('type') == 'step']}"
        )

        # 2. 最终回答非空
        delta_text = "".join(
            e.get("text", "") for e in events if e.get("type") == "delta"
        )
        assert len(delta_text.strip()) > 0, "LLM 没有返回任何回答"

    @pytest.mark.anyio
    async def test_forecast_query(self, container, run_live):
        """
        询问未来天气预报，验证 Agent 调用 weather_forecast 工具。
        """
        if not run_live:
            pytest.skip("需加 --run-live 参数")

        from core.models import AgentConfig

        events = []
        async for ev in container.agent().run(
            user_id    = "test_user",
            session_id = "weather_forecast_chengdu",
            text       = "成都未来三天天气怎么样？",
            config     = AgentConfig(stream=False, max_steps=5),
        ):
            events.append(ev)

        tool_names = [e.get("tool") for e in events if e.get("type") == "step"]
        assert any(t in ("weather_forecast", "weather_current") for t in tool_names), (
            f"期望调用天气工具，实际: {tool_names}"
        )

        done = [e for e in events if e.get("type") == "done"]
        assert done and done[0]["status"] == "done"
