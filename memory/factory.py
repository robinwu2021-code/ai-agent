"""
memory/factory.py — MemorySystem 工厂

从 llm.yaml memory 节 + LLMRouter 构建完整的三层记忆系统。

设计要点：
  · 所有 engine alias 从现有 LLMRouter 解析，不创建新的 LLM 连接
  · 各层可独立替换：ltm.backend 改为不同值即可
  · 兜底：任何构建失败都回退到 InMemory 实现，系统始终可用
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from utils.llm_config import MemoryConfig, VectorStoreConfig

from memory.protocols import MemorySystem

log = structlog.get_logger(__name__)


class MemoryFactory:
    """从 MemoryConfig + LLMRouter 构建 MemorySystem。"""

    @staticmethod
    def build(
        cfg:    "MemoryConfig",
        router: Any,
        vs_cfg: "VectorStoreConfig | None" = None,
    ) -> MemorySystem:
        """
        同步构建（各后端的 async 初始化均为 lazy，首次使用时才建连接）。

        router:  已初始化的 LLMRouter
        vs_cfg:  全局 VectorStoreConfig（当 ltm.use_global_vector_store=True 时使用）
        """
        stm          = MemoryFactory._build_stm(cfg.stm)
        ltm          = MemoryFactory._build_ltm(cfg.ltm, router, vs_cfg)
        working      = MemoryFactory._build_working(cfg.working, router)
        consolidator = MemoryFactory._build_consolidator(cfg.consolidation, router, ltm)

        log.info(
            "memory.factory.built",
            stm          = type(stm).__name__,
            ltm          = type(ltm).__name__,
            working      = type(working).__name__,
            consolidator = type(consolidator).__name__,
        )
        return MemorySystem(stm=stm, ltm=ltm, working=working, consolidator=consolidator)

    # ── STM ──────────────────────────────────────────────────────

    @staticmethod
    def _build_stm(cfg: Any) -> Any:
        backend = (cfg.backend or "in_memory").lower().strip()

        if backend == "redis":
            try:
                from memory.stores import RedisShortTermMemory
                stm = RedisShortTermMemory(
                    redis_url = cfg.redis_url,
                    ttl       = cfg.ttl_idle,
                    max_ttl   = cfg.ttl_max,
                )
                log.info("memory.stm.redis", url=cfg.redis_url)
                return stm
            except Exception as exc:
                log.warning("memory.stm.redis_failed", error=str(exc),
                            fallback="in_memory")

        from memory.stores import InMemoryShortTermMemory
        log.info("memory.stm.in_memory")
        return InMemoryShortTermMemory()

    # ── LTM ──────────────────────────────────────────────────────

    @staticmethod
    def _build_ltm(cfg: Any, router: Any, vs_cfg: Any) -> Any:
        backend = (cfg.backend or "in_memory").lower().strip()
        effective_vs = vs_cfg if cfg.use_global_vector_store else None

        # ── mem0（推荐）─────────────────────────────────────────
        if backend == "mem0":
            try:
                from memory.ltm.mem0_ltm import Mem0LongTermMemory
                ltm = Mem0LongTermMemory(
                    router           = router,
                    llm_alias        = cfg.llm_engine,
                    embed_alias      = cfg.embed_engine,
                    vector_store_cfg = effective_vs,
                    collection       = cfg.collection,
                    dedup_threshold  = cfg.mem0_dedup_threshold,
                    max_memories     = cfg.mem0_max_memories,
                )
                log.info("memory.ltm.mem0",
                         llm_engine   = cfg.llm_engine   or "router-default",
                         embed_engine = cfg.embed_engine or "router-embed",
                         collection   = cfg.collection)
                return ltm
            except ImportError:
                log.warning("memory.ltm.mem0_not_installed",
                            hint="pip install mem0ai", fallback="in_memory")
            except Exception as exc:
                log.warning("memory.ltm.mem0_failed", error=str(exc),
                            fallback="in_memory")

        # ── Milvus 直连（与 KB 共基础设施）──────────────────────
        if backend == "milvus":
            try:
                from memory.stores import MilvusLongTermMemory
                embed_fn = MemoryFactory._make_embed_fn(router, cfg.embed_engine)
                vs = effective_vs
                uri   = vs.milvus_uri   if vs and vs.backend == "milvus" else ""
                token = vs.milvus_token if vs and vs.backend == "milvus" else ""
                size  = vs.milvus_vector_size if vs and vs.backend == "milvus" else 1536
                if not uri:
                    raise ValueError("milvus.uri 未配置")
                ltm = MilvusLongTermMemory(
                    uri=uri, token=token, embed_fn=embed_fn, vector_size=size
                )
                log.info("memory.ltm.milvus", uri=uri)
                return ltm
            except Exception as exc:
                log.warning("memory.ltm.milvus_failed", error=str(exc),
                            fallback="in_memory")

        # ── Qdrant ───────────────────────────────────────────────
        if backend == "qdrant":
            try:
                from memory.stores import QdrantLongTermMemory
                embed_fn = MemoryFactory._make_embed_fn(router, cfg.embed_engine)
                vs = effective_vs
                url  = vs.qdrant_url  if vs and vs.backend == "qdrant" else ""
                path = vs.qdrant_path if vs and vs.backend == "qdrant" else "./data/qdrant"
                ltm  = QdrantLongTermMemory(
                    url=url or None, path=path or None, embed_fn=embed_fn
                )
                log.info("memory.ltm.qdrant",
                         mode="server" if url else "embedded")
                return ltm
            except Exception as exc:
                log.warning("memory.ltm.qdrant_failed", error=str(exc),
                            fallback="in_memory")

        # ── Zep ──────────────────────────────────────────────────
        if backend == "zep":
            try:
                from memory.ltm.zep_ltm import ZepLongTermMemory
                ltm = ZepLongTermMemory(
                    server_url = cfg.zep_server_url,
                    api_key    = cfg.zep_api_key,
                )
                log.info("memory.ltm.zep", url=cfg.zep_server_url)
                return ltm
            except ImportError:
                log.warning("memory.ltm.zep_not_installed",
                            hint="pip install zep-python", fallback="in_memory")
            except Exception as exc:
                log.warning("memory.ltm.zep_failed", error=str(exc),
                            fallback="in_memory")

        # ── InMemory（兜底）──────────────────────────────────────
        from memory.stores import InMemoryLongTermMemory
        log.info("memory.ltm.in_memory")
        return InMemoryLongTermMemory()

    # ── Working Memory ────────────────────────────────────────────

    @staticmethod
    def _build_working(cfg: Any, router: Any) -> Any:
        """
        构建 WorkingMemory（PriorityContextManager 的增强替代品）。
        summarize_engine 决定历史压缩用哪个引擎。
        """
        # 解析 summarize engine（空→用 router 默认 summarize 路由）
        summarize_fn = MemoryFactory._make_summarize_fn(router, cfg.summarize_engine)

        from context.manager import PriorityContextManager
        mgr = PriorityContextManager(llm_engine=None)
        # 用解析好的 summarize_fn 替换内部引用
        mgr._summarize_fn = summarize_fn
        mgr._ltm_score_min = cfg.ltm_score_min
        mgr.HISTORY_COMPRESS_THRESHOLD = cfg.compress_threshold
        log.info("memory.working.context_manager",
                 summarize_engine = cfg.summarize_engine or "router-default",
                 compress_threshold = cfg.compress_threshold)
        return mgr

    # ── Consolidator ─────────────────────────────────────────────

    @staticmethod
    def _build_consolidator(cfg: Any, router: Any, ltm: Any) -> Any:
        """
        构建 MemoryConsolidator。
        engine alias 决定固化提取用哪个 LLM（可用廉价快速模型）。
        """
        from memory.consolidation import MemoryConsolidator, INCREMENTAL_THRESHOLD

        # 构造一个使用指定 alias 的 router 包装器
        engine_wrapper = MemoryFactory._make_router_wrapper(router, cfg.engine)

        consolidator = MemoryConsolidator(
            llm_engine       = engine_wrapper,
            long_term_memory = ltm,
        )
        # 覆盖固化参数
        import memory.consolidation as _mc
        _mc.INCREMENTAL_THRESHOLD = cfg.incremental_every

        log.info("memory.consolidator.built",
                 engine            = cfg.engine or "router-default",
                 incremental_every = cfg.incremental_every,
                 segment_size      = cfg.segment_size,
                 min_importance    = cfg.min_importance)
        return consolidator

    # ── 内部工具 ─────────────────────────────────────────────────

    @staticmethod
    def _make_embed_fn(router: Any, alias: str) -> Any:
        """
        返回一个 async embed 函数，路由到指定引擎。
        alias 为空时走 router 默认 embed 路由。
        """
        if alias:
            async def _embed(text: str) -> list[float]:
                registry = getattr(router, "_registry", None)
                if registry:
                    engine = registry.get(alias)
                    if engine:
                        return await engine.embed(text)
                return await router.embed(text)
        else:
            async def _embed(text: str) -> list[float]:
                return await router.embed(text)
        return _embed

    @staticmethod
    def _make_summarize_fn(router: Any, alias: str) -> Any:
        """
        返回一个 async summarize 函数，路由到指定引擎。
        alias 为空时走 router 默认 summarize 路由。
        """
        if alias:
            async def _summarize(text: str, max_tokens: int = 200, **kw) -> str:
                registry = getattr(router, "_registry", None)
                if registry:
                    engine = registry.get(alias)
                    if engine:
                        return await engine.summarize(text, max_tokens)
                return await router.summarize(text, max_tokens=max_tokens)
        else:
            async def _summarize(text: str, max_tokens: int = 200, **kw) -> str:
                return await router.summarize(text, max_tokens=max_tokens,
                                              node_id=kw.get("node_id"))
        return _summarize

    @staticmethod
    def _make_router_wrapper(router: Any, alias: str) -> Any:
        """
        返回一个轻量包装器，让 MemoryConsolidator 能通过 engine.summarize() 调用
        指定 alias 的引擎（consolidation.py 调用 self._llm.summarize(prompt, ...)）。
        """
        class _RouterWrapper:
            def __init__(self, r: Any, a: str) -> None:
                self._r = r
                self._a = a

            async def summarize(self, text: str, max_tokens: int = 1200, **kw) -> str:
                node_id = kw.get("node_id", "consolidate")
                if self._a:
                    registry = getattr(self._r, "_registry", None)
                    if registry:
                        engine = registry.get(self._a)
                        if engine:
                            return await engine.summarize(text, max_tokens)
                return await self._r.summarize(text, max_tokens=max_tokens,
                                               node_id=node_id)

            # chat() pass-through（部分代码检测 hasattr(llm, "chat")）
            async def chat(self, *args: Any, **kwargs: Any) -> Any:
                return await self._r.chat(*args, **kwargs)

        return _RouterWrapper(router, alias)
