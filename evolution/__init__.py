"""
evolution/__init__.py — 自我进化模块入口

设计原则：
  • 零耦合 —— 功能代码无需修改，进化模块通过「方法包装」注入监听
  • 可插拔 —— enabled=false 时整个模块静默退出，不影响主流程
  • 可扩展 —— extra_signal_collectors / extra_analyzers / extra_actors
             允许未来新功能通过配置注入自定义扩展点

启动方式（server.py 调用一次）：
    from evolution import EvolutionModule
    evo = EvolutionModule.from_yaml()
    await evo.setup(skill_registry=registry, kb=kb_store)

关闭方式：
    await evo.teardown()

事件发布（可选，供功能代码主动埋点）：
    from evolution import get_bus
    await get_bus().emit(FeedbackEvent(...))
"""
from __future__ import annotations

import asyncio
import importlib
import time
import types
from typing import Any

import structlog

from evolution.bus import EventBus
from evolution.config import EvolutionConfig
from evolution.models import BiQueryEvent, FeedbackEvent, RagQueryEvent
from evolution.scheduler import EvolutionScheduler
from evolution.store import EvolutionStore

log = structlog.get_logger("evolution")

# 全局单例（供外部通过 get_bus() / get_store() 访问）
_bus:   EventBus        | None = None
_store: EvolutionStore  | None = None


def get_bus() -> EventBus:
    """获取全局 EventBus（未初始化时返回 NullBus）。"""
    return _bus or _NullBus()


def get_store() -> EvolutionStore | None:
    return _store


class EvolutionModule:
    """自我进化模块，管理整个进化系统的生命周期。"""

    def __init__(self, config: EvolutionConfig | None = None) -> None:
        self._config    = config or EvolutionConfig()
        self._bus       = EventBus()
        self._store     = EvolutionStore(self._config.db_path)
        self._scheduler = EvolutionScheduler(
            hour     = self._config.scheduler.daily_job_hour,
            minute   = self._config.scheduler.daily_job_minute,
            timezone = self._config.scheduler.timezone,
        )
        self._collectors: list = []
        self._analyzers:  list = []
        self._actors:     list = []

    @classmethod
    def from_yaml(cls, path: str = "evolution_config.yaml") -> "EvolutionModule":
        cfg = EvolutionConfig.from_yaml(path)
        return cls(cfg)

    # ── 生命周期 ──────────────────────────────────────────────────────────────

    async def setup(
        self,
        skill_registry = None,   # core.skills.SkillRegistry 或鸭子类型
        kb             = None,   # PersistentKnowledgeBase 或鸭子类型
    ) -> None:
        """
        初始化并启动进化模块。
        在 server 启动完成后调用，传入已初始化的 skill_registry 和 kb。
        """
        if not self._config.enabled:
            log.info("evolution.disabled")
            return

        # 暴露全局单例
        global _bus, _store
        _bus   = self._bus
        _store = self._store

        # 1. 初始化存储
        await self._store.initialize()

        # 2. 启动事件总线
        await self._bus.start()

        # 3. 注册内置信号采集器
        await self._register_collectors()

        # 4. 初始化内置分析器 & 执行器
        self._build_analyzers_and_actors(kb=kb)

        # 5. 注册外部扩展（来自 evolution_config.yaml）
        self._load_extensions(skill_registry=skill_registry, kb=kb)

        # 6. 通过「方法包装」注入功能代码监听（零耦合核心）
        if self._config.signal.bi_enabled and skill_registry is not None:
            self._wrap_bi_skill(skill_registry)

        if self._config.signal.rag_enabled and kb is not None:
            self._wrap_rag_kb(kb)

        # 7. 注册每日定时任务
        self._scheduler.add_daily_job("evolution_daily", self._run_daily_job)
        self._scheduler.start()

        log.info("evolution.setup_complete",
                 collectors=len(self._collectors),
                 analyzers=len(self._analyzers),
                 actors=len(self._actors))

    async def teardown(self) -> None:
        self._scheduler.stop()
        await self._bus.stop()
        await self._store.close()
        log.info("evolution.teardown_complete")

    # ── 手动触发（调试 / 管理接口用）─────────────────────────────────────────

    async def run_now(self) -> dict:
        """立即执行一次全量进化（绕过定时器，用于调试/手动触发）。"""
        return await self._run_daily_job()

    async def emit_feedback(
        self,
        session_id: str,
        query_id:   str,
        source:     str,     # "rag" | "bi"
        score:      float,
        comment:    str = "",
    ) -> None:
        """供 API 层调用的反馈发布接口。"""
        event = FeedbackEvent(
            session_id = session_id,
            query_id   = query_id,
            source     = source,
            score      = score,
            comment    = comment,
        )
        await self._bus.emit(event)

    # ── 内部：信号采集器注册 ──────────────────────────────────────────────────

    async def _register_collectors(self) -> None:
        from evolution.signals.bi  import BiSignalCollector
        from evolution.signals.rag import RagSignalCollector

        if self._config.signal.bi_enabled:
            bi_col = BiSignalCollector(self._bus, self._store)
            bi_col.register()
            self._collectors.append(bi_col)

        if self._config.signal.rag_enabled:
            rag_col = RagSignalCollector(self._bus, self._store)
            rag_col.register()
            self._collectors.append(rag_col)

    # ── 内部：构建分析器 & 执行器 ─────────────────────────────────────────────

    def _build_analyzers_and_actors(self, kb=None) -> None:
        from evolution.analyzers.bi_profiler   import BiProfilerAnalyzer
        from evolution.analyzers.chunk_scorer  import ChunkScorerAnalyzer
        from evolution.analyzers.qa_extractor  import QaExtractorAnalyzer
        from evolution.actors.kb_injector      import KbInjectorActor
        from evolution.actors.param_tuner      import ParamTunerActor
        from evolution.actors.prompt_updater   import PromptUpdaterActor
        from evolution.actors.template_builder import TemplateBuilderActor

        acfg = self._config.analyzer
        actr = self._config.actor

        if acfg.bi_profiler_enabled:
            self._analyzers.append(BiProfilerAnalyzer(self._store))

        if acfg.chunk_scorer_enabled:
            self._analyzers.append(
                ChunkScorerAnalyzer(
                    self._store,
                    config={"low_quality_threshold": acfg.low_quality_threshold},
                )
            )

        if acfg.qa_extractor_enabled:
            self._analyzers.append(
                QaExtractorAnalyzer(
                    self._store,
                    config={"qa_min_feedback_score": acfg.qa_min_feedback_score},
                )
            )

        if actr.template_builder_enabled:
            self._actors.append(TemplateBuilderActor(self._store))

        if actr.kb_injector_enabled and kb is not None:
            self._actors.append(KbInjectorActor(self._store, kb=kb))

        if actr.param_tuner_enabled:
            self._actors.append(ParamTunerActor(self._store))

        if actr.prompt_updater_enabled:
            self._actors.append(PromptUpdaterActor(self._store))

    # ── 内部：加载外部扩展 ────────────────────────────────────────────────────

    def _load_extensions(self, **kwargs) -> None:
        """动态加载 evolution_config.yaml 中配置的扩展类。"""
        for spec in self._config.extra_signal_collectors:
            obj = self._load_class(spec, self._bus, self._store, **kwargs)
            if obj and hasattr(obj, "register"):
                obj.register()
                self._collectors.append(obj)

        for spec in self._config.extra_analyzers:
            obj = self._load_class(spec, self._store, **kwargs)
            if obj:
                self._analyzers.append(obj)

        for spec in self._config.extra_actors:
            obj = self._load_class(spec, self._store, **kwargs)
            if obj:
                self._actors.append(obj)

    @staticmethod
    def _load_class(spec: dict, *args, **kwargs):
        try:
            mod = importlib.import_module(spec["module"])
            cls = getattr(mod, spec["class"])
            cfg = spec.get("config", {})
            return cls(*args, config=cfg, **kwargs)
        except Exception as exc:
            log.error("evolution.extension_load_failed", spec=spec, error=str(exc))
            return None

    # ── 内部：方法包装（零耦合注入）──────────────────────────────────────────

    def _wrap_bi_skill(self, skill_registry) -> None:
        """
        包装 AgentBiSkill.execute()，在调用完成后自动 emit BiQueryEvent。
        不修改 AgentBiSkill 源代码，不需要其感知进化模块存在。
        """
        skill = None
        if hasattr(skill_registry, "get"):
            skill = skill_registry.get("agent_bi")
        elif hasattr(skill_registry, "__getitem__"):
            try:
                skill = skill_registry["agent_bi"]
            except Exception:
                pass

        if skill is None:
            log.warning("evolution.wrap_bi.skill_not_found")
            return

        original_execute = skill.execute
        bus = self._bus

        async def wrapped_execute(arguments: dict[str, Any]) -> dict[str, Any]:
            t0 = time.time()
            result: dict = {}
            try:
                result = await original_execute(arguments)
            except Exception as exc:
                result = {"error": str(exc)}
                raise
            finally:
                latency = int((time.time() - t0) * 1000)
                event = BiQueryEvent(
                    user_text   = arguments.get("_user_text", ""),
                    bra_id      = arguments.get("bra_id") or result.get("bra_id", ""),
                    mode        = result.get("mode", ""),
                    api_payload = result.get("_debug_payload") or {},
                    result_rows = len(result.get("raw_metrics") or []),
                    no_data     = bool(result.get("no_data")),
                    error       = result.get("error", ""),
                    latency_ms  = latency,
                )
                await bus.emit(event)
            return result

        skill.execute = wrapped_execute
        log.info("evolution.wrap_bi.done")

    def _wrap_rag_kb(self, kb) -> None:
        """
        包装知识库的查询方法（search / query / ask），
        在检索完成后自动 emit RagQueryEvent。
        """
        # 尝试常见的方法名
        method_name = None
        for name in ("search", "query", "ask", "retrieve"):
            if callable(getattr(kb, name, None)):
                method_name = name
                break

        if method_name is None:
            log.warning("evolution.wrap_rag.no_query_method",
                        kb_type=type(kb).__name__)
            return

        original = getattr(kb, method_name)
        bus = self._bus

        async def wrapped_query(*args, **kwargs):
            t0 = time.time()
            result = await original(*args, **kwargs)
            latency = int((time.time() - t0) * 1000)

            # 尝试从结果中提取 chunks
            chunks: list[dict] = []
            if hasattr(result, "chunks"):
                for c in (result.chunks or []):
                    chunks.append({
                        "chunk_id": getattr(c, "chunk_id", "") or getattr(c, "id", ""),
                        "score":    getattr(c, "score", 0),
                        "text":     getattr(c, "text", "")[:100],
                    })
            elif isinstance(result, list):
                for c in result:
                    if isinstance(c, dict):
                        chunks.append(c)

            query_text = (
                args[0] if args and isinstance(args[0], str)
                else kwargs.get("query", kwargs.get("text", ""))
            )
            kb_id = getattr(kb, "kb_id", "") or kwargs.get("kb_id", "")

            event = RagQueryEvent(
                kb_id             = kb_id,
                query_text        = str(query_text)[:500],
                retrieved_chunks  = chunks,
                latency_ms        = latency,
                retrieval_strategy = kwargs.get("strategy", ""),
            )
            await bus.emit(event)
            return result

        setattr(kb, method_name, wrapped_query)
        log.info("evolution.wrap_rag.done", method=method_name)

    # ── 内部：每日任务 ────────────────────────────────────────────────────────

    async def _run_daily_job(self) -> dict:
        """按序运行所有分析器 → 执行器。"""
        log.info("evolution.daily_job.start")
        results: dict[str, Any] = {}

        # 分析器
        for analyzer in self._analyzers:
            name = type(analyzer).__name__
            try:
                r = await analyzer.analyze()
                results[name] = r
                log.info("evolution.analyzer.done", name=name, result=r)
            except Exception as exc:
                log.error("evolution.analyzer.failed", name=name, error=str(exc))
                results[name] = {"error": str(exc)}

        # 执行器（在分析器之后）
        for actor in self._actors:
            name = type(actor).__name__
            try:
                r = await actor.act()
                results[name] = r
                log.info("evolution.actor.done", name=name, result=r)
            except Exception as exc:
                log.error("evolution.actor.failed", name=name, error=str(exc))
                results[name] = {"error": str(exc)}

        log.info("evolution.daily_job.done", results=results)
        return results


# ── NullBus（进化模块未启用时的降级实现）─────────────────────────────────────

class _NullBus:
    """进化模块未初始化时使用，emit() 静默丢弃所有事件。"""
    async def emit(self, event) -> None:
        pass
    def subscribe(self, *args, **kwargs) -> None:
        pass
