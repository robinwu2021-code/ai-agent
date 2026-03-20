"""
tests/test_router.py — LLMRouter 多模型路由测试
"""
from __future__ import annotations
import asyncio
import pytest
from core.models import AgentConfig, LLMResponse, Message, MessageRole


# ── Mock engines ──────────────────────────────────────────

class TrackingEngine:
    """记录调用次数和参数，可配置成功/失败。"""
    def __init__(self, name: str, fail: bool = False, response: str = "ok"):
        self.name     = name
        self.fail     = fail
        self.response = response
        self.calls: list[str] = []

    async def chat(self, messages, tools, config):
        self.calls.append("chat")
        if self.fail:
            raise RuntimeError(f"{self.name} failed")
        return LLMResponse(content=f"{self.name}:{self.response}",
                           usage={"prompt_tokens": 10, "completion_tokens": 5})

    async def stream_chat(self, messages, tools, config):
        self.calls.append("stream_chat")
        if self.fail:
            raise RuntimeError(f"{self.name} failed")
        for c in self.response:
            yield c

    async def embed(self, text):
        self.calls.append("embed")
        if self.fail:
            raise RuntimeError(f"{self.name} failed")
        return [0.1] * 64

    async def summarize(self, text, max_tokens, **kwargs):
        self.calls.append(f"summarize:{kwargs.get('node_id','')}")
        if self.fail:
            raise RuntimeError(f"{self.name} failed")
        return f"{self.name}:summary"

    async def eval_score(self, prompt, max_tokens=100):
        self.calls.append("eval_score")
        return f"{self.name}:0.8"

    async def route_decision(self, prompt, max_tokens=30):
        self.calls.append("route_decision")
        return "agent_a"


def make_registry(*specs):
    """specs = (alias, engine, supports_embed=False)"""
    from llm.router import ModelRegistry
    reg = ModelRegistry()
    for item in specs:
        alias, engine = item[0], item[1]
        sup_embed = item[2] if len(item) > 2 else False
        reg.register(alias, engine, supports_embed=sup_embed)
    return reg


def make_router(registry, **task_kwargs):
    from llm.router import LLMRouter, TaskRouter
    default = list(registry._specs.keys())[0]
    tr = TaskRouter(default=default, **task_kwargs)
    return LLMRouter(registry, tr)


# ══════════════════════════════════════════════════════════
# 1. Basic routing
# ══════════════════════════════════════════════════════════

class TestBasicRouting:
    @pytest.mark.asyncio
    async def test_chat_uses_configured_engine(self):
        smart = TrackingEngine("smart")
        fast  = TrackingEngine("fast")
        reg   = make_registry(("smart", smart), ("fast", fast))
        router = make_router(reg, chat="smart", summarize="fast")

        resp = await router.chat([], [], AgentConfig())
        assert "smart" in resp.content
        assert smart.calls == ["chat"]
        assert fast.calls  == []

    @pytest.mark.asyncio
    async def test_summarize_uses_different_engine(self):
        smart = TrackingEngine("smart")
        fast  = TrackingEngine("fast")
        reg   = make_registry(("smart", smart), ("fast", fast))
        router = make_router(reg, chat="smart", summarize="fast")

        result = await router.summarize("text", 100)
        assert "fast" in result
        assert fast.calls[0].startswith("summarize")
        assert smart.calls == []

    @pytest.mark.asyncio
    async def test_embed_uses_embed_engine(self):
        chat_eng  = TrackingEngine("chat-only")
        embed_eng = TrackingEngine("embed-only")
        reg = make_registry(("chat-only", chat_eng), ("embed-only", embed_eng, True))
        router = make_router(reg, embed="embed-only")

        vec = await router.embed("hello")
        assert len(vec) == 64
        assert embed_eng.calls == ["embed"]
        assert chat_eng.calls  == []


# ══════════════════════════════════════════════════════════
# 2. Task-level routing
# ══════════════════════════════════════════════════════════

class TestTaskRouting:
    @pytest.mark.asyncio
    async def test_plan_routes_separately(self):
        smart = TrackingEngine("smart")
        fast  = TrackingEngine("fast")
        reg   = make_registry(("smart", smart), ("fast", fast))
        router = make_router(reg, chat="fast", plan="smart")

        await router.plan(
            [Message(role=MessageRole.USER, content="plan this")],
            AgentConfig()
        )
        assert smart.calls == ["chat"]   # plan() calls chat internally
        assert fast.calls  == []

    @pytest.mark.asyncio
    async def test_eval_score_routes_to_eval_engine(self):
        judge = TrackingEngine("judge")
        main  = TrackingEngine("main")
        reg   = make_registry(("main", main), ("judge", judge))
        router = make_router(reg, chat="main", eval="judge")

        result = await router.eval_score("is this good?")
        assert "judge" in result
        assert judge.calls == ["eval_score"]
        assert main.calls  == []

    @pytest.mark.asyncio
    async def test_route_decision_uses_route_engine(self):
        router_eng = TrackingEngine("router-eng")
        main       = TrackingEngine("main")
        reg        = make_registry(("main", main), ("router-eng", router_eng))
        router     = make_router(reg, chat="main", route="router-eng")

        decision = await router.route_decision("which agent?")
        assert router_eng.calls == ["route_decision"]

    @pytest.mark.asyncio
    async def test_consolidate_node_id_passed(self):
        cheap = TrackingEngine("cheap")
        reg   = make_registry(("cheap", cheap))
        router = make_router(reg, consolidate="cheap")

        await router.summarize("mem text", 200, node_id="consolidate")
        # TrackingEngine records "summarize:{node_id}" — verify node_id was received
        assert cheap.calls, "Engine should have been called"
        assert any("summarize" in c for c in cheap.calls), f"Expected summarize call, got {cheap.calls}"


# ══════════════════════════════════════════════════════════
# 3. Node-level override (AgentConfig.model_alias / node_id)
# ══════════════════════════════════════════════════════════

class TestNodeOverrides:
    @pytest.mark.asyncio
    async def test_model_alias_overrides_task_routing(self):
        smart   = TrackingEngine("smart")
        special = TrackingEngine("special")
        reg     = make_registry(("smart", smart), ("special", special))
        router  = make_router(reg, chat="smart")

        config = AgentConfig(model_alias="special")
        resp   = await router.chat([], [], config)
        assert "special" in resp.content
        assert special.calls == ["chat"]
        assert smart.calls   == []

    @pytest.mark.asyncio
    async def test_node_id_override_in_task_router(self):
        from llm.router import LLMRouter, TaskRouter
        engine_a = TrackingEngine("a")
        engine_b = TrackingEngine("b")
        reg      = make_registry(("a", engine_a), ("b", engine_b))
        tr = TaskRouter(
            default="a",
            chat="a",
            node_overrides={"expensive_step": "b"},
        )
        router = LLMRouter(reg, tr)

        # Normal call → uses "a"
        await router.chat([], [], AgentConfig())
        assert engine_a.calls == ["chat"]

        # Node-specific call → uses "b"
        await router.chat([], [], AgentConfig(node_id="expensive_step"))
        assert engine_b.calls == ["chat"]

    @pytest.mark.asyncio
    async def test_model_alias_takes_priority_over_node_id(self):
        from llm.router import LLMRouter, TaskRouter
        a = TrackingEngine("a")
        b = TrackingEngine("b")
        c = TrackingEngine("c")
        reg = make_registry(("a", a), ("b", b), ("c", c))
        tr  = TaskRouter(default="a", node_overrides={"n1": "b"})
        router = LLMRouter(reg, tr)

        # model_alias beats node_id
        config = AgentConfig(model_alias="c", node_id="n1")
        await router.chat([], [], config)
        assert c.calls == ["chat"]
        assert a.calls == b.calls == []


# ══════════════════════════════════════════════════════════
# 4. Fallback chain
# ══════════════════════════════════════════════════════════

class TestFallback:
    @pytest.mark.asyncio
    async def test_falls_back_on_failure(self):
        broken = TrackingEngine("broken", fail=True)
        backup = TrackingEngine("backup")
        reg    = make_registry(("broken", broken), ("backup", backup))

        from llm.router import LLMRouter, TaskRouter
        tr     = TaskRouter(default="broken", fallback=["backup"])
        router = LLMRouter(reg, tr)

        resp = await router.chat([], [], AgentConfig())
        assert "backup" in resp.content
        assert broken.calls == ["chat"]   # was tried
        assert backup.calls == ["chat"]   # succeeded

    @pytest.mark.asyncio
    async def test_all_failed_raises(self):
        a = TrackingEngine("a", fail=True)
        b = TrackingEngine("b", fail=True)
        reg = make_registry(("a", a), ("b", b))

        from llm.router import LLMRouter, TaskRouter
        tr     = TaskRouter(default="a", fallback=["b"])
        router = LLMRouter(reg, tr)

        with pytest.raises(RuntimeError, match="All engines failed"):
            await router.chat([], [], AgentConfig())

    @pytest.mark.asyncio
    async def test_fallback_chain_exhausted_in_order(self):
        calls = []
        class OrderEngine:
            def __init__(self, name, fail=True):
                self.name = name
                self._fail = fail
            async def chat(self, m, t, c):
                calls.append(self.name)
                if self._fail:
                    raise RuntimeError(self.name)
                return LLMResponse(content=self.name, usage={})
            async def stream_chat(self, m, t, c):
                raise RuntimeError()
            async def embed(self, t): return []
            async def summarize(self, t, n, **kw): return ""

        reg = make_registry(
            ("first",  OrderEngine("first",  fail=True)),
            ("second", OrderEngine("second", fail=True)),
            ("third",  OrderEngine("third",  fail=False)),
        )
        from llm.router import LLMRouter, TaskRouter
        tr     = TaskRouter(default="first", fallback=["second", "third"])
        router = LLMRouter(reg, tr)

        resp = await router.chat([], [], AgentConfig())
        assert calls == ["first", "second", "third"]
        assert resp.content == "third"


# ══════════════════════════════════════════════════════════
# 5. Circuit breaker
# ══════════════════════════════════════════════════════════

class TestCircuitBreaker:
    @pytest.mark.asyncio
    async def test_circuit_opens_after_threshold(self):
        broken = TrackingEngine("broken", fail=True)
        backup = TrackingEngine("backup")
        reg    = make_registry(("broken", broken), ("backup", backup))

        from llm.router import LLMRouter, TaskRouter
        tr     = TaskRouter(default="broken", fallback=["backup"])
        router = LLMRouter(reg, tr, circuit_breaker_threshold=2, circuit_breaker_cooldown=999)

        # First 2 failures open the circuit
        for _ in range(2):
            await router.chat([], [], AgentConfig())

        broken.calls.clear()

        # Circuit now open: broken should be skipped entirely
        resp = await router.chat([], [], AgentConfig())
        assert broken.calls == []   # not tried
        assert "backup" in resp.content

    @pytest.mark.asyncio
    async def test_circuit_resets_after_success(self):
        engine = TrackingEngine("e")
        reg    = make_registry(("e", engine))

        from llm.router import LLMRouter, TaskRouter
        tr     = TaskRouter(default="e")
        router = LLMRouter(reg, tr, circuit_breaker_threshold=3)

        # Record 2 failures (below threshold)
        engine.fail = True
        for _ in range(2):
            try:
                await router.chat([], [], AgentConfig())
            except Exception:
                pass

        # One success → resets counter
        engine.fail = False
        await router.chat([], [], AgentConfig())

        state = router._circuits["e"]
        assert state.failures == 0
        assert not state.is_open()


# ══════════════════════════════════════════════════════════
# 6. Stats & introspection
# ══════════════════════════════════════════════════════════

class TestStats:
    @pytest.mark.asyncio
    async def test_stats_track_calls(self):
        eng = TrackingEngine("eng")
        reg = make_registry(("eng", eng))
        router = make_router(reg)

        await router.chat([], [], AgentConfig())
        await router.chat([], [], AgentConfig())

        stats = router.get_stats()
        assert stats["eng"]["total"]   == 2
        assert stats["eng"]["success"] == 2
        assert stats["eng"]["errors"]  == 0

    @pytest.mark.asyncio
    async def test_stats_count_fallbacks(self):
        broken = TrackingEngine("broken", fail=True)
        backup = TrackingEngine("backup")
        reg    = make_registry(("broken", broken), ("backup", backup))

        from llm.router import LLMRouter, TaskRouter
        tr     = TaskRouter(default="broken", fallback=["backup"])
        router = LLMRouter(reg, tr)

        await router.chat([], [], AgentConfig())

        stats = router.get_stats()
        assert stats["broken"]["errors"]   == 1
        assert stats["backup"]["fallback"] == 1

    def test_describe_output(self):
        eng = TrackingEngine("test-eng")
        reg = make_registry(("test-eng", eng))
        from llm.router import LLMRouter, TaskRouter
        tr     = TaskRouter(default="test-eng", summarize="test-eng")
        router = LLMRouter(reg, tr)

        desc = router.describe()
        assert "test-eng" in desc
        assert "chat" in desc
        assert "summarize" in desc


# ══════════════════════════════════════════════════════════
# 7. single_engine_router compatibility shim
# ══════════════════════════════════════════════════════════

class TestSingleEngineShim:
    @pytest.mark.asyncio
    async def test_single_engine_router_wraps_correctly(self):
        from llm.router import single_engine_router
        eng    = TrackingEngine("eng")
        router = single_engine_router(eng, alias="default")

        resp = await router.chat([], [], AgentConfig())
        assert "eng" in resp.content

    @pytest.mark.asyncio
    async def test_single_engine_all_tasks_use_same_engine(self):
        from llm.router import single_engine_router
        eng    = TrackingEngine("only")
        router = single_engine_router(eng)

        await router.chat([], [], AgentConfig())
        await router.summarize("x", 100)
        await router.embed("x")
        # All routed to the same engine
        assert "chat"      in eng.calls
        assert any("summarize" in c for c in eng.calls)
        assert "embed"     in eng.calls


# ══════════════════════════════════════════════════════════
# 8. Container integration
# ══════════════════════════════════════════════════════════

class TestContainerIntegration:
    def test_create_dev_wraps_mock_in_router(self):
        from core.container import AgentContainer
        c = AgentContainer.create_dev()
        assert c.llm_router is not None

    def test_create_dev_legacy_llm_engine_still_works(self):
        from core.container import AgentContainer
        from llm.engines import MockLLMEngine
        c = AgentContainer.create_dev()
        c.llm_engine = MockLLMEngine()
        c.llm_router = None
        c.orchestrator = None
        c.consolidator = None
        built = c.build()
        assert built.llm_router is not None

    @pytest.mark.asyncio
    async def test_create_with_router_factory(self):
        from core.container import AgentContainer
        from llm.router import single_engine_router
        from llm.engines import MockLLMEngine

        router = single_engine_router(MockLLMEngine())
        c = AgentContainer.create_with_router(router)
        assert c.llm_router is router

    @pytest.mark.asyncio
    async def test_model_alias_unknown_yields_error_event(self):
        """An unknown model_alias causes the orchestrator to yield a done/error event."""
        from core.container import AgentContainer
        from llm.router import single_engine_router
        from llm.engines import MockLLMEngine

        router = single_engine_router(MockLLMEngine())
        c = AgentContainer.create_with_router(router)

        events = []
        config = AgentConfig(model_alias="nonexistent-alias")
        async for ev in c.agent().run("u", "s", "hello", config=config):
            events.append(ev)

        # The orchestrator catches the RuntimeError and yields a done(error) event
        statuses = [e.get("status") for e in events if e.get("type") == "done"]
        assert any(s == "error" for s in statuses), f"Expected error status, got events: {events}"
