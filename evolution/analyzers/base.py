"""evolution/analyzers/base.py — 分析器基类"""
from __future__ import annotations

from evolution.store import EvolutionStore


class BaseAnalyzer:
    """
    分析器基类。

    每天由 Scheduler 触发，从 EvolutionStore 读取信号，
    计算质量分/模式/画像等，将结果写回 Store 供 Actor 使用。

    扩展新功能：继承此类，实现 analyze() 方法。
    """

    def __init__(self, store: EvolutionStore, config: dict | None = None) -> None:
        self._store  = store
        self._config = config or {}

    async def analyze(self) -> dict:
        """
        执行分析。返回摘要 dict（供日志/监控使用）。
        子类必须实现。
        """
        raise NotImplementedError
