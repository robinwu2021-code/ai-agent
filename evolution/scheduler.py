"""
evolution/scheduler.py — APScheduler 定时任务管理

封装 APScheduler，每天定时运行所有分析器和执行器。
"""
from __future__ import annotations

from typing import Callable, Awaitable

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

log = structlog.get_logger("evolution.scheduler")


class EvolutionScheduler:

    def __init__(
        self,
        hour:     int = 2,
        minute:   int = 0,
        timezone: str = "Asia/Shanghai",
    ) -> None:
        self._hour     = hour
        self._minute   = minute
        self._timezone = timezone
        self._scheduler = AsyncIOScheduler(timezone=timezone)

    def add_daily_job(
        self,
        job_id:  str,
        fn:      Callable[[], Awaitable[None]],
    ) -> None:
        """注册每日定时任务。"""
        self._scheduler.add_job(
            fn,
            trigger = CronTrigger(
                hour     = self._hour,
                minute   = self._minute,
                timezone = self._timezone,
            ),
            id              = job_id,
            replace_existing= True,
            misfire_grace_time = 3600,  # 错过触发窗口后最多补跑 1 小时
        )
        log.info("evolution.scheduler.job_added",
                 job_id=job_id, hour=self._hour, minute=self._minute)

    def start(self) -> None:
        if not self._scheduler.running:
            self._scheduler.start()
            log.info("evolution.scheduler.started")

    def stop(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            log.info("evolution.scheduler.stopped")
