"""
evolution/bus.py — 进化事件总线

设计原则：
  1. 完全非阻塞 —— emit() 永不阻塞调用方，永不抛异常
  2. 缓冲队列   —— 事件入队后异步消费，峰值时自动丢弃（可配置）
  3. 零耦合     —— 功能代码只调用 emit()，无需感知 Handler 存在
  4. 多类型路由 —— 按 event_type 路由到各 Collector
"""
from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, Callable, Awaitable

import structlog

from evolution.models import BaseEvent

log = structlog.get_logger("evolution.bus")


Handler = Callable[[BaseEvent], Awaitable[None]]


class EventBus:
    """进程内异步事件总线（单例，由 EvolutionModule 持有）。"""

    def __init__(self, queue_size: int = 10_000) -> None:
        self._handlers: dict[str, list[Handler]] = defaultdict(list)
        self._queue: asyncio.Queue[BaseEvent] = asyncio.Queue(maxsize=queue_size)
        self._task: asyncio.Task | None = None
        self._running = False
        self._dropped = 0

    # ── 生命周期 ──────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """启动后台消费任务。在事件循环启动后调用。"""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._worker(), name="evolution.bus.worker")
        log.info("evolution.bus.started", queue_size=self._queue.maxsize)

    async def stop(self) -> None:
        """优雅停止：排空队列后取消任务。"""
        self._running = False
        # 等待队列清空（最多 5 秒）
        try:
            await asyncio.wait_for(self._queue.join(), timeout=5.0)
        except asyncio.TimeoutError:
            log.warning("evolution.bus.stop_timeout",
                        remaining=self._queue.qsize())
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        log.info("evolution.bus.stopped", dropped=self._dropped)

    # ── 订阅 / 发布 ──────────────────────────────────────────────────────────

    def subscribe(self, event_type: str, handler: Handler) -> None:
        """注册某类事件的处理器（多个处理器按注册顺序调用）。"""
        self._handlers[event_type].append(handler)
        log.debug("evolution.bus.subscribe",
                  event_type=event_type, handler=handler.__qualname__)

    def subscribe_all(self, handler: Handler) -> None:
        """注册通配处理器，接收所有事件类型。"""
        self.subscribe("*", handler)

    async def emit(self, event: BaseEvent) -> None:
        """
        发布事件。永不阻塞，永不抛异常。
        队列满时记录 warning 并丢弃（避免内存无限增长）。
        """
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            self._dropped += 1
            if self._dropped % 100 == 1:   # 每 100 次打一次 warning
                log.warning("evolution.bus.queue_full",
                            dropped=self._dropped,
                            event_type=getattr(event, "event_type", "?"))
        except Exception as exc:
            log.error("evolution.bus.emit_error", error=str(exc))

    # ── 后台消费 ──────────────────────────────────────────────────────────────

    async def _worker(self) -> None:
        while True:
            try:
                event: BaseEvent = await self._queue.get()
                await self._dispatch(event)
                self._queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("evolution.bus.worker_error", error=str(exc))

    async def _dispatch(self, event: BaseEvent) -> None:
        event_type = getattr(event, "event_type", "")
        handlers = self._handlers.get(event_type, []) + self._handlers.get("*", [])
        for handler in handlers:
            try:
                await handler(event)
            except Exception as exc:
                log.error("evolution.bus.handler_error",
                           event_type=event_type,
                           handler=handler.__qualname__,
                           error=str(exc))

    # ── 调试 ──────────────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "queue_size":  self._queue.qsize(),
            "dropped":     self._dropped,
            "running":     self._running,
            "handler_types": list(self._handlers.keys()),
        }
