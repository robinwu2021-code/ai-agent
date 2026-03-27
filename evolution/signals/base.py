"""evolution/signals/base.py — 信号采集器基类"""
from __future__ import annotations

from evolution.bus import EventBus
from evolution.store import EvolutionStore


class BaseSignalCollector:
    """
    信号采集器基类。

    子类通过 subscribe() 向 EventBus 注册监听，
    收到事件后写入 EvolutionStore。

    扩展新功能时：继承此类，实现 register() 方法，
    在 EvolutionModule.setup() 中实例化并调用 register()。
    """

    def __init__(self, bus: EventBus, store: EvolutionStore) -> None:
        self._bus   = bus
        self._store = store

    def register(self) -> None:
        """向 EventBus 订阅事件。子类必须实现。"""
        raise NotImplementedError
