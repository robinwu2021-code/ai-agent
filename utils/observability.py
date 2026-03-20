"""
utils/observability.py — 可观测性工具

包含：
  - 结构化日志初始化（structlog）
  - 简单内存指标收集器（生产替换为 OpenTelemetry）
  - 请求追踪装饰器
"""
from __future__ import annotations

import functools
import time
from collections import defaultdict, deque
from typing import Any, Callable

import structlog


# ─────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────

def configure_logging(level: str = "INFO", json_logs: bool = False) -> None:
    """初始化 structlog，开发用彩色输出，生产用 JSON。"""
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]
    if json_logs:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(__import__("logging"), level.upper(), 20)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


# ─────────────────────────────────────────────
# Metrics Collector
# ─────────────────────────────────────────────

class MetricsCollector:
    """
    轻量内存指标收集器。
    生产环境请替换为 OpenTelemetry + Prometheus/Grafana。
    """

    def __init__(self, window_size: int = 1000) -> None:
        self._counters:   dict[str, int]              = defaultdict(int)
        self._histograms: dict[str, deque[float]]     = defaultdict(lambda: deque(maxlen=window_size))
        self._gauges:     dict[str, float]            = {}

    def increment(self, name: str, value: int = 1, **labels) -> None:
        key = self._key(name, labels)
        self._counters[key] += value

    def record(self, name: str, value: float, **labels) -> None:
        """记录直方图样本（延迟、token 数等）。"""
        key = self._key(name, labels)
        self._histograms[key].append(value)

    def gauge(self, name: str, value: float, **labels) -> None:
        self._gauges[self._key(name, labels)] = value

    def summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {"counters": dict(self._counters), "gauges": dict(self._gauges), "histograms": {}}
        for key, samples in self._histograms.items():
            if not samples:
                continue
            s = sorted(samples)
            n = len(s)
            result["histograms"][key] = {
                "count": n,
                "min":   round(s[0], 2),
                "max":   round(s[-1], 2),
                "mean":  round(sum(s) / n, 2),
                "p50":   round(s[n // 2], 2),
                "p95":   round(s[int(n * 0.95)], 2),
                "p99":   round(s[int(n * 0.99)], 2),
            }
        return result

    @staticmethod
    def _key(name: str, labels: dict) -> str:
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"


# 全局单例
metrics = MetricsCollector()


# ─────────────────────────────────────────────
# Tracing Decorators
# ─────────────────────────────────────────────

def trace_async(name: str | None = None):
    """
    异步函数追踪装饰器，自动记录耗时和错误。

    用法：
        @trace_async("skill.execute")
        async def execute(self, arguments):
            ...
    """
    def decorator(func: Callable):
        metric_name = name or f"{func.__module__}.{func.__qualname__}"

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            log = structlog.get_logger(func.__module__)
            start = time.monotonic()
            try:
                result = await func(*args, **kwargs)
                duration = (time.monotonic() - start) * 1000
                metrics.record(f"{metric_name}.duration_ms", duration)
                metrics.increment(f"{metric_name}.success")
                log.debug(f"{metric_name}.ok", duration_ms=round(duration, 1))
                return result
            except Exception as e:
                duration = (time.monotonic() - start) * 1000
                metrics.record(f"{metric_name}.duration_ms", duration)
                metrics.increment(f"{metric_name}.error")
                log.error(f"{metric_name}.error", error=str(e), duration_ms=round(duration, 1))
                raise
        return wrapper
    return decorator


def trace_generator(name: str | None = None):
    """异步生成器追踪装饰器。"""
    def decorator(func: Callable):
        metric_name = name or f"{func.__module__}.{func.__qualname__}"

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            log   = structlog.get_logger(func.__module__)
            start = time.monotonic()
            count = 0
            try:
                async for item in func(*args, **kwargs):
                    count += 1
                    yield item
                duration = (time.monotonic() - start) * 1000
                metrics.record(f"{metric_name}.duration_ms", duration)
                metrics.increment(f"{metric_name}.success")
                log.debug(f"{metric_name}.ok", duration_ms=round(duration, 1), events=count)
            except Exception as e:
                metrics.increment(f"{metric_name}.error")
                log.error(f"{metric_name}.error", error=str(e))
                raise
        return wrapper
    return decorator
