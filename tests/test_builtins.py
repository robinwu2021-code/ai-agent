"""
tests/test_builtins.py — 扩展 Skill 和工具测试
"""
import pytest
from skills.builtins import CalculatorSkill, JsonProcessorSkill


class TestCalculatorSkill:
    def make(self):
        return CalculatorSkill()

    @pytest.mark.asyncio
    async def test_basic_arithmetic(self):
        s = self.make()
        r = await s.execute({"expression": "2 + 3 * 4"})
        assert r["result"] == 14

    @pytest.mark.asyncio
    async def test_power(self):
        s = self.make()
        r = await s.execute({"expression": "2 ** 10"})
        assert r["result"] == 1024

    @pytest.mark.asyncio
    async def test_round(self):
        s = self.make()
        r = await s.execute({"expression": "round(3.14159, 2)"})
        assert r["result"] == 3.14

    @pytest.mark.asyncio
    async def test_invalid_expression(self):
        s = self.make()
        with pytest.raises(ValueError):
            await s.execute({"expression": "import os"})

    @pytest.mark.asyncio
    async def test_division(self):
        s = self.make()
        r = await s.execute({"expression": "10 / 4"})
        assert r["result"] == 2.5


class TestJsonProcessorSkill:
    def make(self):
        return JsonProcessorSkill()

    @pytest.mark.asyncio
    async def test_dot_path(self):
        s = self.make()
        data = {"user": {"name": "Alice", "age": 30}}
        r = await s.execute({"data": data, "path": "user.name"})
        assert r == "Alice"

    @pytest.mark.asyncio
    async def test_list_index(self):
        s = self.make()
        data = {"items": [{"id": 1}, {"id": 2}]}
        r = await s.execute({"data": data, "path": "items[1].id"})
        assert r == 2

    @pytest.mark.asyncio
    async def test_multi_key_extract(self):
        s = self.make()
        data = {"name": "Alice", "age": 30, "city": "Beijing"}
        r = await s.execute({"data": data, "keys": ["name", "city"]})
        assert r == {"name": "Alice", "city": "Beijing"}

    @pytest.mark.asyncio
    async def test_json_string_input(self):
        import json
        s = self.make()
        data = json.dumps({"x": 42})
        r = await s.execute({"data": data, "path": "x"})
        assert r == 42

    @pytest.mark.asyncio
    async def test_invalid_path(self):
        s = self.make()
        with pytest.raises(ValueError):
            await s.execute({"data": {"a": 1}, "path": "b.c"})


class TestObservability:
    def test_metrics_counter(self):
        from utils.observability import MetricsCollector
        m = MetricsCollector()
        m.increment("requests", 1, status="ok")
        m.increment("requests", 2, status="ok")
        summary = m.summary()
        assert summary["counters"]["requests{status=ok}"] == 3

    def test_metrics_histogram(self):
        from utils.observability import MetricsCollector
        m = MetricsCollector()
        for v in [10, 20, 30, 40, 50]:
            m.record("latency_ms", v)
        summary = m.summary()
        h = summary["histograms"]["latency_ms"]
        assert h["min"] == 10
        assert h["max"] == 50
        assert h["mean"] == 30

    @pytest.mark.asyncio
    async def test_trace_decorator(self):
        from utils.observability import trace_async, MetricsCollector
        import utils.observability as obs
        # 替换为测试专用收集器
        orig  = obs.metrics
        m     = MetricsCollector()
        obs.metrics = m

        @trace_async("test.fn")
        async def my_fn():
            return 42

        result = await my_fn()
        assert result == 42
        s = m.summary()
        assert s["counters"].get("test.fn.success", 0) == 1
        obs.metrics = orig
