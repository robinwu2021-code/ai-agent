"""evolution/actors/base.py — 执行器基类"""
from __future__ import annotations

from evolution.store import EvolutionStore


class BaseActor:
    """
    执行器基类。

    每天由 Scheduler 在 Analyzer 完成后触发，
    读取 Store 中的分析结果，对系统做出实际变更
    （注入知识库、更新配置、生成模板等）。

    扩展新功能：继承此类，实现 act() 方法。
    """

    def __init__(
        self,
        store:  EvolutionStore,
        config: dict | None = None,
        **kwargs,  # 接受各 Actor 需要的额外依赖（kb、skill_registry 等）
    ) -> None:
        self._store  = store
        self._config = config or {}

    async def act(self) -> dict:
        """
        执行进化动作。返回摘要 dict。
        子类必须实现。
        """
        raise NotImplementedError
