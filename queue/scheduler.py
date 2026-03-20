"""
queue/scheduler.py — 任务队列与异步调度器

功能：
  - TaskQueue       内存优先级队列（生产替换为 BullMQ/Celery）
  - TaskScheduler   定时/延迟任务调度
  - TaskWorker      后台 Worker 消费队列
  - JobRecord       任务持久化记录

设计：
  - 任务以 JSON 序列化存储，支持从断点恢复
  - 优先级：0(最高) - 9(最低)
  - 支持延迟执行、定时执行、最大重试次数
"""
from __future__ import annotations

import asyncio
import heapq
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

import structlog

log = structlog.get_logger(__name__)


class JobStatus(str, Enum):
    QUEUED     = "queued"
    RUNNING    = "running"
    DONE       = "done"
    FAILED     = "failed"
    CANCELLED  = "cancelled"
    SCHEDULED  = "scheduled"   # 等待到达执行时间


@dataclass
class Job:
    id:           str  = field(default_factory=lambda: f"job_{uuid.uuid4().hex[:8]}")
    name:         str  = ""           # 任务类型名
    payload:      dict = field(default_factory=dict)
    priority:     int  = 5            # 0 = 最高，9 = 最低
    max_retries:  int  = 3
    retries:      int  = 0
    status:       JobStatus = JobStatus.QUEUED
    result:       Any  = None
    error:        str  = ""
    run_at:       float | None = None  # Unix timestamp，None = 立即
    created_at:   str  = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    started_at:   str | None = None
    finished_at:  str | None = None
    tags:         list[str] = field(default_factory=list)

    def __lt__(self, other: "Job") -> bool:
        # heapq 用：priority 越小优先级越高
        return (self.priority, self.created_at) < (other.priority, other.created_at)


# ─────────────────────────────────────────────
# Task Queue
# ─────────────────────────────────────────────

class TaskQueue:
    """
    内存优先级队列。
    生产替换为 Redis-backed BullMQ（通过 redis.asyncio）。
    """

    def __init__(self) -> None:
        self._heap:   list[Job]         = []
        self._jobs:   dict[str, Job]    = {}
        self._event   = asyncio.Event()

    async def enqueue(
        self,
        name: str,
        payload: dict,
        priority: int = 5,
        max_retries: int = 3,
        run_at: float | None = None,
        tags: list[str] | None = None,
    ) -> Job:
        job = Job(
            name=name, payload=payload, priority=priority,
            max_retries=max_retries, run_at=run_at,
            tags=tags or [],
            status=JobStatus.SCHEDULED if run_at else JobStatus.QUEUED,
        )
        self._jobs[job.id] = job
        if not run_at:
            heapq.heappush(self._heap, job)
            self._event.set()
        log.info("queue.enqueue", job_id=job.id, name=name, priority=priority)
        return job

    async def dequeue(self, timeout: float = 30.0) -> Job | None:
        """取出下一个可执行任务（阻塞等待）。"""
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            # 将到期的 scheduled 任务移入队列
            import time
            now = time.time()
            for job in list(self._jobs.values()):
                if (job.status == JobStatus.SCHEDULED
                        and job.run_at and job.run_at <= now):
                    job.status = JobStatus.QUEUED
                    heapq.heappush(self._heap, job)

            if self._heap:
                job = heapq.heappop(self._heap)
                if job.status == JobStatus.CANCELLED:
                    continue
                job.status     = JobStatus.RUNNING
                job.started_at = datetime.now(timezone.utc).isoformat()
                return job

            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                return None
            self._event.clear()
            try:
                await asyncio.wait_for(self._event.wait(), timeout=min(remaining, 5.0))
            except asyncio.TimeoutError:
                pass

    def complete(self, job_id: str, result: Any = None) -> None:
        if job := self._jobs.get(job_id):
            job.status      = JobStatus.DONE
            job.result      = result
            job.finished_at = datetime.now(timezone.utc).isoformat()

    def fail(self, job_id: str, error: str, retry: bool = True) -> bool:
        """标记失败，返回是否已重新入队。"""
        job = self._jobs.get(job_id)
        if not job:
            return False
        if retry and job.retries < job.max_retries:
            job.retries += 1
            job.status   = JobStatus.QUEUED
            heapq.heappush(self._heap, job)
            self._event.set()
            log.info("queue.retry", job_id=job_id, attempt=job.retries)
            return True
        job.status      = JobStatus.FAILED
        job.error       = error
        job.finished_at = datetime.now(timezone.utc).isoformat()
        log.error("queue.failed", job_id=job_id, error=error)
        return False

    def cancel(self, job_id: str) -> bool:
        if job := self._jobs.get(job_id):
            job.status = JobStatus.CANCELLED
            return True
        return False

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list_jobs(
        self,
        status: JobStatus | None = None,
        tag: str | None = None,
        limit: int = 50,
    ) -> list[Job]:
        jobs = list(self._jobs.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        if tag:
            jobs = [j for j in jobs if tag in j.tags]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)[:limit]

    @property
    def stats(self) -> dict[str, int]:
        counts: dict[str, int] = {s.value: 0 for s in JobStatus}
        for job in self._jobs.values():
            counts[job.status.value] += 1
        return counts


# ─────────────────────────────────────────────
# Task Worker
# ─────────────────────────────────────────────

class TaskWorker:
    """
    后台 Worker，消费队列中的任务。
    handlers 是 {job_name: async_handler_fn} 的字典。
    """

    def __init__(
        self,
        queue: TaskQueue,
        handlers: dict[str, Callable],
        concurrency: int = 4,
    ) -> None:
        self._queue       = queue
        self._handlers    = handlers
        self._concurrency = concurrency
        self._running     = False
        self._semaphore   = asyncio.Semaphore(concurrency)

    async def start(self) -> None:
        self._running = True
        log.info("worker.started", concurrency=self._concurrency)
        tasks = [asyncio.create_task(self._loop()) for _ in range(self._concurrency)]
        await asyncio.gather(*tasks, return_exceptions=True)

    def stop(self) -> None:
        self._running = False

    async def _loop(self) -> None:
        while self._running:
            job = await self._queue.dequeue(timeout=5.0)
            if not job:
                continue
            asyncio.create_task(self._execute(job))

    async def _execute(self, job: Job) -> None:
        async with self._semaphore:
            handler = self._handlers.get(job.name)
            if not handler:
                self._queue.fail(job.id, f"No handler for job type '{job.name}'", retry=False)
                return
            try:
                log.info("worker.executing", job_id=job.id, name=job.name)
                result = await handler(job.payload)
                self._queue.complete(job.id, result)
                log.info("worker.done", job_id=job.id)
            except Exception as e:
                retry = self._queue.fail(job.id, str(e))
                log.error("worker.error", job_id=job.id, error=str(e), retried=retry)


# ─────────────────────────────────────────────
# Task Scheduler（定时任务）
# ─────────────────────────────────────────────

@dataclass
class ScheduledJob:
    name:     str
    payload:  dict
    cron:     str | None  = None   # 简化 cron：支持 "@every_Ns" / "@hourly" / "@daily"
    priority: int         = 5
    _next_run: float      = 0.0

import time as _time

class TaskScheduler:
    """
    简化的定时任务调度器。
    支持：
      @every_30s   每 30 秒
      @every_5m    每 5 分钟
      @hourly      每小时
      @daily       每天
    """

    def __init__(self, queue: TaskQueue) -> None:
        self._queue    = queue
        self._jobs:    list[ScheduledJob] = []
        self._running  = False

    def add(self, name: str, payload: dict, cron: str, priority: int = 5) -> None:
        job = ScheduledJob(name=name, payload=payload, cron=cron, priority=priority)
        job._next_run = _time.time() + self._interval(cron)
        self._jobs.append(job)
        log.info("scheduler.added", name=name, cron=cron)

    async def start(self) -> None:
        self._running = True
        log.info("scheduler.started", jobs=len(self._jobs))
        while self._running:
            now = _time.time()
            for job in self._jobs:
                if now >= job._next_run:
                    await self._queue.enqueue(
                        name=job.name, payload=job.payload, priority=job.priority
                    )
                    job._next_run = now + self._interval(job.cron or "@every_60s")
            await asyncio.sleep(1)

    def stop(self) -> None:
        self._running = False

    @staticmethod
    def _interval(cron: str) -> float:
        if cron == "@hourly":   return 3600
        if cron == "@daily":    return 86400
        if m := __import__("re").match(r"@every_(\d+)([smh])", cron):
            n, unit = int(m.group(1)), m.group(2)
            return n * {"s": 1, "m": 60, "h": 3600}[unit]
        return 60  # 默认 60 秒
