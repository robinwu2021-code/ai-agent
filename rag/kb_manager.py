"""
rag/kb_manager.py — 知识库统一入口（Facade）

配置优先级（由高到低）：
  1. llm.yaml — LLM 引擎连接参数、向量库连接参数、embedding 引擎
  2. kb_config.yaml — KB 业务逻辑（解析策略 / 分块策略 / 检索参数 / alias 引用）

组件依赖图：
  KBManager
    ├── IngestionPipeline
    │     ├── Parser（docling/unstructured/markitdown）
    │     ├── SmartChunker（structural/semantic/sentence/fixed + contextual）
    │     ├── Embedder（由 llm.yaml router.embed 或 kb_config.llm.embed_engine 决定）
    │     ├── VectorStore（由 llm.yaml vector_store 或 kb_config.vector_store 决定）
    │     ├── KeywordStore（elasticsearch/opensearch/memory）
    │     └── GraphBuilder（可选）
    ├── QueryPipeline
    │     ├── Embedder（共用）
    │     ├── VectorStore（共用）
    │     ├── KeywordStore（共用）
    │     ├── GraphRetriever（可选）
    │     └── Reranker（bge/cohere/llm/none）
    ├── KBFileManager（文件生命周期 + 版本 + 审计）
    └── PermissionManager（目录 + 权限）
"""
from __future__ import annotations

from typing import Callable

import structlog

from rag.config import KBConfig, get_kb_config
from rag.files.manager import KBFileManager
from rag.files.models import OperatorContext
from rag.files.store import KBFileStore
from rag.permissions.manager import PermissionManager
from rag.permissions.models import DirectoryType, PermissionLevel, PermissionRole
from rag.permissions.security_models import SecurityLevel
from rag.pipeline.ingestion import IngestionPipeline
from rag.pipeline.query import QueryPipeline, QueryResult

log = structlog.get_logger(__name__)


class KBManager:
    """
    知识库统一入口。

    用法：
        mgr = await KBManager.create()            # 自动读取 llm.yaml + kb_config.yaml
        file = await mgr.upload_file(
            file_path="report.pdf",
            workspace_id="ws_01",
            kb_id="kb_finance",
            directory_id="dir_01",
            operator=OperatorContext(user_id="alice"),
            security_level="INTERNAL",           # 可选，默认 INTERNAL
        )
        result = await mgr.query("上周销售额是多少？", kb_id="kb_finance",
                                  user_id="alice")   # user_id 用于权限过滤
        print(result.context)
    """

    def __init__(
        self,
        ingestion: IngestionPipeline,
        query_pipeline: QueryPipeline,
        file_manager: KBFileManager,
        permission_manager: PermissionManager,
        security_manager=None,   # rag.permissions.SecurityManager | None
    ) -> None:
        self._ingestion = ingestion
        self._query = query_pipeline
        self._files = file_manager
        self._perms = permission_manager
        self._sec = security_manager   # 多维度安全权限管理器（可选）

    # ── 工厂方法 ──────────────────────────────────────────────────────

    @classmethod
    async def create(
        cls,
        config: KBConfig | None = None,
        llm_fn: Callable | None = None,
    ) -> "KBManager":
        """
        根据配置自动组装所有组件。

        Args:
            config:  KBConfig 实例，默认读取 kb_config.yaml
            llm_fn:  摘要 / 重排序所用 LLM，签名 async (system, user) -> str
                     留空时自动从 llm.yaml 解析 summarize_engine 对应引擎
        """
        cfg = config or get_kb_config()

        # ── 解析器 ─────────────────────────────────────────────────
        from rag.parsers.factory import ParserFactory
        parser = ParserFactory.create(cfg.parser)

        # ── LLM 函数（摘要 / 图谱 / 重排序）─────────────────────────
        if llm_fn is None:
            llm_fn = cls._resolve_llm_fn(cfg)

        # ── 分块器 ─────────────────────────────────────────────────
        from rag.chunkers.factory import ChunkerFactory
        chunker = ChunkerFactory.create(cfg.chunker, llm_fn=llm_fn)

        # ── Embedder（优先 llm.yaml router.embed）─────────────────
        from rag.embedders.factory import EmbedderFactory
        embedder = EmbedderFactory.create_from_kb_config(cfg.llm)

        # ── 向量存储（优先 llm.yaml vector_store）────────────────
        vector_store = await cls._create_vector_store(cfg)

        # ── 关键词存储 ─────────────────────────────────────────────
        from rag.keyword_stores.factory import KeywordStoreFactory
        keyword_store = KeywordStoreFactory.create(cfg.keyword_store)
        await keyword_store.initialize()

        # ── 图谱（可选）──────────────────────────────────────────
        graph_builder = None
        graph_retriever = None
        if cfg.graph.enabled:
            graph_builder, graph_retriever = await cls._create_graph(cfg, embedder, llm_fn)

        # ── 文件管理 ───────────────────────────────────────────────
        fm_cfg = cfg.file_management
        file_store = KBFileStore(db_url=fm_cfg.db_url)
        file_manager = KBFileManager(
            store=file_store,
            storage_base=fm_cfg.storage.base_path,
            keep_versions=fm_cfg.storage.keep_versions,
        )

        # ── 权限管理（目录级）─────────────────────────────────────────
        perm_manager = PermissionManager(db_url=fm_cfg.db_url)

        # ── 多维度安全权限管理器（文档级四层安全）────────────────────
        sec_manager = None
        try:
            sec_cfg = getattr(cfg.permissions, "multi_dim", None)
            if sec_cfg and getattr(sec_cfg, "enabled", False):
                from rag.permissions import init_security_manager
                db_path = getattr(sec_cfg, "db_path", "./data/security.db")
                prefix  = getattr(sec_cfg, "personal_collection_prefix", "kb_personal")
                sec_manager = init_security_manager(db_path=db_path,
                                                    personal_collection_prefix=prefix)
                log.info("kb_manager.security_manager_ready", db_path=db_path)
        except Exception as exc:
            log.warning("kb_manager.security_manager_init_failed", error=str(exc))

        # ── Reranker ───────────────────────────────────────────────
        reranker = cls._create_reranker(cfg, llm_fn) if cfg.reranker.enabled else None

        # ── 流水线组装 ─────────────────────────────────────────────
        s = cfg.file_management.summary
        cs = cfg.chunker.chunk_summary
        ingestion = IngestionPipeline(
            parser=parser,
            chunker=chunker,
            embedder=embedder,
            vector_store=vector_store,
            keyword_store=keyword_store,
            graph_builder=graph_builder,
            file_manager=file_manager,
            llm_fn=llm_fn,
            summary_enabled=s.enabled,
            summary_max_chars=s.max_input_chars,
            generate_diff_summary=s.generate_diff,
            chunk_summary_enabled=cs.enabled,
            chunk_summary_max_chars=cs.max_chars,
            chunk_summary_concurrency=cs.concurrency,
        )

        r = cfg.retrieval
        query_pipeline = QueryPipeline(
            embedder=embedder,
            vector_store=vector_store,
            keyword_store=keyword_store,
            graph_retriever=graph_retriever,
            reranker=reranker,
            vector_top_k=r.vector_top_k,
            keyword_top_k=r.keyword_top_k,
            graph_top_k=r.graph_top_k,
            fusion=r.fusion,
            rrf_k=r.rrf_k,
            weights={
                "vector":  r.weights.vector,
                "keyword": r.weights.keyword,
                "graph":   r.weights.graph,
            },
            enable_graph=r.enable_graph_retrieval,
            final_top_k=cfg.reranker.top_n,
            use_summary_search=cs.enabled,   # 摘要检索与分块摘要开关保持一致
        )

        log.info(
            "kb_manager.created",
            parser=cfg.parser.backend,
            chunker=cfg.chunker.strategy,
            embed_engine=cfg.llm.embed_engine or "(router.embed)",
            vector_store_backend=cfg.vector_store.backend,
            keyword_store=cfg.keyword_store.backend,
            graph=cfg.graph.enabled,
        )

        return cls(
            ingestion=ingestion,
            query_pipeline=query_pipeline,
            file_manager=file_manager,
            permission_manager=perm_manager,
            security_manager=sec_manager,
        )

    # ── 文件操作（委托给 KBFileManager + IngestionPipeline）──────────

    async def upload_file(
        self,
        file_path: str,
        workspace_id: str,
        kb_id: str,
        directory_id: str,
        operator: OperatorContext,
        change_description: str = "",
        security_level: str | SecurityLevel | None = None,
        org_unit_ids: list[str] | None = None,
    ):
        """
        上传文件并触发异步索引流水线。

        Args:
            security_level: 文档安全级别（PUBLIC/INTERNAL/CONFIDENTIAL/PERSONAL）。
                            为 None 时采用 kb_config.yaml 中的 default_level（默认 INTERNAL）。
            org_unit_ids:   INTERNAL 级别时生效；指定可访问的组织单元 ID 列表。
                            空列表表示全员内部可见。
        """
        kb_file = await self._files.upload(
            file_path=file_path,
            workspace_id=workspace_id,
            kb_id=kb_id,
            directory_id=directory_id,
            operator=operator,
            change_description=change_description,
        )

        # ── 多维度安全权限登记 ──────────────────────────────────────
        effective_collection = kb_id   # 默认使用标准 collection
        if self._sec is not None:
            try:
                # 解析安全级别
                if security_level is None:
                    sec_lvl = SecurityLevel.INTERNAL
                elif isinstance(security_level, SecurityLevel):
                    sec_lvl = security_level
                else:
                    sec_lvl = SecurityLevel(str(security_level).upper())

                await self._sec.set_document_permission(
                    doc_id=kb_file.id,
                    kb_id=kb_id,
                    owner_id=operator.user_id,
                    security_level=sec_lvl,
                    org_unit_ids=org_unit_ids or [],
                    operator_id=operator.user_id,
                )

                # PERSONAL 新上传 → 物理路由到独立 collection
                if sec_lvl == SecurityLevel.PERSONAL:
                    effective_collection = (
                        f"{self._sec._prefix}_{operator.user_id}_{kb_id}"
                    )
                    log.info(
                        "kb_manager.personal_collection_routed",
                        doc_id=kb_file.id,
                        collection=effective_collection,
                    )
            except Exception as exc:
                log.warning("kb_manager.set_doc_permission_failed",
                            doc_id=kb_file.id, error=str(exc))

        import asyncio
        asyncio.create_task(self._ingestion.ingest(
            file_path=file_path,
            file_id=kb_file.id,
            version_id=kb_file.current_version_id or "",
            kb_id=effective_collection,   # PERSONAL 时使用独立 collection
            change_type="created",
        ))
        return kb_file

    async def update_file(
        self,
        file_id: str,
        new_file_path: str,
        operator: OperatorContext,
        change_description: str = "",
    ):
        """更新文件内容并触发重索引。"""
        old_file = await self._files._store.get_file(file_id)
        prev_summary = old_file.summary if old_file else ""

        kb_file = await self._files.update(
            file_id=file_id,
            new_file_path=new_file_path,
            operator=operator,
            change_description=change_description,
        )
        await self._ingestion.delete_doc(file_id, kb_file.kb_id)
        import asyncio
        asyncio.create_task(self._ingestion.ingest(
            file_path=new_file_path,
            file_id=kb_file.id,
            version_id=kb_file.current_version_id or "",
            kb_id=kb_file.kb_id,
            change_type="content_updated",
            prev_summary=prev_summary,
        ))
        return kb_file

    async def delete_file(self, file_id: str, operator: OperatorContext) -> None:
        file = await self._files._store.get_file(file_id)
        if file:
            await self._ingestion.delete_doc(file_id, file.kb_id)
        await self._files.delete(file_id, operator)

    async def restore_version(
        self, file_id: str, version_id: str, operator: OperatorContext
    ):
        return await self._files.restore_version(file_id, version_id, operator)

    # ── 查询 ──────────────────────────────────────────────────────────

    async def query(
        self,
        query_text: str,
        kb_id: str,
        top_k: int = 5,
        doc_ids: list[str] | None = None,
        user_id: str | None = None,
        directory_id: str | None = None,
    ) -> QueryResult:
        """
        执行知识库检索查询。

        多维度安全权限过滤（若 SecurityManager 已启用）：
          1. build_retrieval_context()  — 预计算用户可见范围
          2. 向量检索（获取比 top_k 更多的候选）
          3. filter_chunks()           — 后检索二次过滤（纵深防御）
          4. 截断到 top_k 后返回
        """
        # ── 无安全管理器 → 直接检索 ─────────────────────────────────
        if self._sec is None or not user_id:
            return await self._query.query(
                query_text=query_text,
                kb_id=kb_id,
                top_k=top_k,
                doc_ids=doc_ids,
            )

        # ── 构建检索权限上下文 ──────────────────────────────────────
        retrieval_ctx = None
        try:
            retrieval_ctx = await self._sec.build_retrieval_context(
                user_id=user_id, kb_id=kb_id
            )
        except Exception as exc:
            log.warning("kb_manager.build_retrieval_ctx_failed",
                        user_id=user_id, kb_id=kb_id, error=str(exc))

        # ── 向量检索（放宽 top_k 以确保过滤后仍有足够结果）──────────
        fetch_k = max(top_k * 3, top_k + 20) if retrieval_ctx else top_k
        result = await self._query.query(
            query_text=query_text,
            kb_id=kb_id,
            top_k=fetch_k,
            doc_ids=doc_ids,
        )

        # ── 后检索二次权限过滤（纵深防御）──────────────────────────
        if retrieval_ctx is not None and result.chunks:
            try:
                filtered = await self._sec.filter_chunks(
                    chunks=result.chunks,
                    ctx=retrieval_ctx,
                )
                before_count = len(result.chunks)
                after_count  = len(filtered[:top_k])
                # 截断到请求的 top_k
                result = QueryResult(
                    query=result.query,
                    kb_id=kb_id,
                    chunks=filtered[:top_k],
                    context="\n\n".join(c.text for c in filtered[:top_k]),
                )
                log.debug(
                    "kb_manager.query_filtered",
                    user_id=user_id,
                    before=before_count,
                    after=after_count,
                )
            except Exception as exc:
                log.warning("kb_manager.filter_chunks_failed",
                            user_id=user_id, error=str(exc))
                # 过滤失败时安全降级：返回空结果而非未过滤结果
                result = QueryResult(chunks=[], context="", query=result.query,
                                     kb_id=kb_id)

        return result

    # ── 文件查询 ──────────────────────────────────────────────────────

    async def list_files(self, kb_id: str | None = None,
                         directory_id: str | None = None, workspace_id: str | None = None):
        return await self._files.list_files(
            kb_id=kb_id, directory_id=directory_id, workspace_id=workspace_id
        )

    async def get_history(self, file_id: str, limit: int = 20):
        return await self._files.get_history(file_id, limit)

    async def get_audit_log(self, file_id: str | None = None,
                             kb_id: str | None = None, **kwargs):
        return await self._files.get_audit_log(file_id=file_id, kb_id=kb_id, **kwargs)

    # ── 目录与权限 ─────────────────────────────────────────────────────

    async def create_directory(self, workspace_id: str, name: str, created_by: str,
                                parent_id: str | None = None,
                                dir_type: DirectoryType = DirectoryType.PUBLIC):
        return await self._perms.create_directory(
            workspace_id=workspace_id, name=name, created_by=created_by,
            parent_id=parent_id, dir_type=dir_type,
        )

    async def grant_permission(self, workspace_id: str, directory_id: str,
                                user_id: str, role: PermissionRole, granted_by: str,
                                expires_at: float | None = None):
        return await self._perms.grant(
            workspace_id=workspace_id, directory_id=directory_id,
            subject_type="user", subject_id=user_id, role=role,
            granted_by=granted_by, expires_at=expires_at,
        )

    async def check_access(self, user_id: str, directory_id: str,
                            level: PermissionLevel = PermissionLevel.READ) -> bool:
        return await self._perms.check_access(user_id, directory_id, level)

    # ── 内部组件构建 ──────────────────────────────────────────────────

    @staticmethod
    def _resolve_llm_fn(cfg: KBConfig) -> Callable | None:
        """
        从 llm.yaml 解析 summarize_engine（用于文档摘要和图谱构建）。
        返回 async (system, user) -> str 签名的函数，失败时返回 None。
        """
        try:
            from utils.llm_config import load_from_yaml
            engines, router, _, _ = load_from_yaml()
            engine_map = {e.alias: e for e in engines}

            alias = cfg.llm.summarize_engine.strip() or router.summarize or router.default
            llm_cfg = engine_map.get(alias) if alias else None
            if llm_cfg is None and engines:
                llm_cfg = engines[0]   # 回退到第一个引擎

            if llm_cfg is not None:
                engine = llm_cfg.build_engine()

                async def _llm_fn(system: str, user: str) -> str:
                    return await engine.chat(system=system, user=user)

                log.info("kb_manager.llm_fn_resolved",
                         alias=llm_cfg.alias, task="summarize/graph/rerank")
                return _llm_fn
        except Exception as exc:
            log.warning("kb_manager.llm_fn_resolve_failed", error=str(exc))
        return None

    @staticmethod
    async def _create_vector_store(cfg: KBConfig):
        """
        向量库构建：
          1. use_global=True → 使用 llm.yaml vector_store 配置（可用 cfg.vector_store.collection 覆盖 collection 名）
          2. use_global=False 或 llm.yaml 无 vector_store → 使用 kb_config.yaml vector_store 配置
          3. 全部失败 → 内存兜底
        """
        vs_cfg = cfg.vector_store
        collection_override = vs_cfg.collection.strip()

        # ── 优先尝试 llm.yaml vector_store ────────────────────────
        if vs_cfg.use_global:
            try:
                from utils.llm_config import load_from_yaml
                _, _, llm_vs_cfg, _ = load_from_yaml()
                if llm_vs_cfg is not None:
                    # 允许 kb_config 覆盖 collection 名
                    if collection_override:
                        if llm_vs_cfg.backend == "milvus":
                            llm_vs_cfg.milvus_collection = collection_override
                        elif llm_vs_cfg.backend == "qdrant":
                            llm_vs_cfg.qdrant_collection = collection_override
                    store = llm_vs_cfg.build()
                    if store is not None:
                        # Propagate chunk_summary config to vector store
                        if hasattr(store, '_summary_enabled'):
                            store._summary_enabled = cfg.chunker.chunk_summary.enabled
                        await store.initialize()
                        log.info("kb_manager.vector_store_from_llm_yaml",
                                 backend=llm_vs_cfg.backend,
                                 collection=collection_override or "(llm.yaml default)")
                        return store
            except Exception as exc:
                log.warning("kb_manager.llm_yaml_vector_store_failed", error=str(exc))

        # ── 回退到 kb_config.yaml vector_store ────────────────────
        backend = vs_cfg.backend.lower()
        if backend == "qdrant":
            try:
                from rag.qdrant_store import QdrantVectorStore
                c = vs_cfg.qdrant
                store = QdrantVectorStore(
                    path=c.path if c.mode == "embedded" else None,
                    url=c.url if c.mode == "server" else None,
                    collection=collection_override or "kb_chunks",
                    vector_size=cfg.llm.embed_dimensions,
                )
                await store.initialize()
                log.info("kb_manager.vector_store_qdrant", mode=c.mode)
                return store
            except Exception as exc:
                log.warning("kb_manager.qdrant_init_failed", error=str(exc))

        if backend == "milvus":
            try:
                from rag.milvus_store import MilvusVectorStore
                c = vs_cfg.milvus
                uri = c.uri if c.mode == "lite" else f"http://{c.host}:{c.port}"
                store = MilvusVectorStore(
                    uri=uri,
                    collection=collection_override or "kb_chunks",
                    vector_size=cfg.llm.embed_dimensions,
                    summary_enabled=cfg.chunker.chunk_summary.enabled,
                )
                await store.initialize()
                log.info("kb_manager.vector_store_milvus", uri=uri)
                return store
            except Exception as exc:
                log.warning("kb_manager.milvus_init_failed", error=str(exc))

        if backend == "chroma":
            try:
                from rag.chroma_store import ChromaVectorStore  # type: ignore[import]
                c = vs_cfg.chroma
                store = ChromaVectorStore(
                    path=c.path,
                    collection=collection_override or "kb_chunks",
                )
                await store.initialize()
                log.info("kb_manager.vector_store_chroma", path=c.path)
                return store
            except Exception as exc:
                log.warning("kb_manager.chroma_init_failed", error=str(exc))

        # ── 最终兜底：内存向量存储 ────────────────────────────────
        log.warning("kb_manager.using_memory_vector_store")
        return _MemVectorStore()

    @staticmethod
    async def _create_graph(cfg: KBConfig, embedder, llm_fn):
        try:
            from rag.graph.store import KGStore
            from rag.graph.extractor import TripleExtractor
            from rag.graph.resolver import EntityResolver
            from rag.graph.builder import GraphBuilder

            async def _llm_wrapper(prompt: str) -> str:
                if llm_fn is None:
                    return "{}"
                return await llm_fn(
                    system="你是知识图谱专家，请按指定格式输出JSON。",
                    user=prompt,
                )

            store = KGStore()
            await store.initialize()
            extractor = TripleExtractor(llm_fn=_llm_wrapper)
            resolver = EntityResolver(store)
            builder = GraphBuilder(store=store, extractor=extractor, resolver=resolver)

            class _GraphBuilderWrapper:
                async def build_from_chunks(self, chunks, doc_id, kb_id):
                    text = "\n".join(c.text for c in chunks)
                    await builder.build_from_text(text, doc_id=doc_id, source_id=kb_id)

            class _GraphRetriever:
                async def search(self, query, kb_id=None, top_k=10):
                    from rag.graph.retriever import GraphRetriever
                    gr = GraphRetriever(store=store)
                    return await gr.search(query, top_k=top_k)

            return _GraphBuilderWrapper(), _GraphRetriever()
        except Exception as exc:
            log.warning("kb_manager.graph_init_failed", error=str(exc))
            return None, None

    @staticmethod
    def _create_reranker(cfg: KBConfig, llm_fn):
        backend = cfg.reranker.backend.lower()
        if backend == "none" or not backend:
            return None
        if backend == "llm" and llm_fn:
            from rag.knowledge_base import LLMReranker

            class _LLMWrapper:
                async def rerank(self, query, chunks, top_k):
                    from rag.knowledge_base import Chunk
                    mem_chunks = [
                        Chunk(id=c.chunk_id, doc_id=c.doc_id, text=c.text,
                              chunk_index=0, metadata=c.metadata)
                        for c in chunks
                    ]
                    reranker = LLMReranker(llm_fn)
                    reranked = await reranker.rerank(query, mem_chunks, top_k=top_k)
                    id_to_chunk = {c.chunk_id: c for c in chunks}
                    return [id_to_chunk[r.id] for r in reranked if r.id in id_to_chunk]

            return _LLMWrapper()
        if backend == "bge":
            try:
                from rag.rerankers.bge_reranker import BGEReranker
                return BGEReranker(model=cfg.reranker.bge.model, device=cfg.reranker.bge.device)
            except Exception as exc:
                log.warning("kb_manager.bge_reranker_failed", error=str(exc))
        if backend == "cohere":
            try:
                from rag.rerankers.cohere_reranker import CohereReranker
                return CohereReranker(
                    model=cfg.reranker.cohere.model,
                    api_key=cfg.reranker.cohere.api_key,
                )
            except Exception as exc:
                log.warning("kb_manager.cohere_reranker_failed", error=str(exc))
        return None


# ── 内存向量存储兜底（不依赖任何外部服务）────────────────────────────────────

class _MemVectorStore:
    """内存向量存储，无需外部服务，适合单元测试和小数据集。"""

    def __init__(self):
        self._chunks: list = []
        self._embs: list = []
        from rag.knowledge_base import BM25Index
        self._bm25 = BM25Index()

    async def initialize(self):
        pass

    async def upsert_chunks(self, payloads: list[dict]):
        from rag.knowledge_base import Chunk
        for p in payloads:
            c = Chunk(
                id=p["chunk_id"], doc_id=p["doc_id"],
                text=p["text"], chunk_index=p.get("chunk_index", 0),
                metadata=p.get("metadata", {}),
                embedding=p.get("embedding", []),
            )
            self._chunks.append(c)
            self._embs.append(p.get("embedding", []))
        self._bm25.build(self._chunks)

    async def hybrid_search(self, query_vec, query_text, kb_id, top_k, doc_ids=None):
        import numpy as np
        if not self._embs or not query_vec:
            return []
        emb_arr = np.array(self._embs, dtype=float)
        q_arr = np.array(query_vec, dtype=float)
        norms = np.linalg.norm(emb_arr, axis=1) * np.linalg.norm(q_arr)
        norms = np.where(norms == 0, 1e-9, norms)
        sims = emb_arr.dot(q_arr) / norms
        idxs = np.argsort(sims)[::-1][:top_k]
        results = []
        for i in idxs:
            c = self._chunks[i]
            if doc_ids and c.doc_id not in doc_ids:
                continue
            if c.metadata.get("kb_id", kb_id) != kb_id:
                continue
            results.append((float(sims[i]), {
                "chunk_id": c.id, "doc_id": c.doc_id,
                "kb_id": kb_id, "text": c.text,
                "metadata": c.metadata,
            }))
        return results

    async def delete_by_doc_id(self, doc_id: str):
        keep = [(c, e) for c, e in zip(self._chunks, self._embs) if c.doc_id != doc_id]
        self._chunks = [x[0] for x in keep]
        self._embs   = [x[1] for x in keep]

    async def get_stats(self, kb_id):
        return {"chunk_count": len(self._chunks), "total_chars": 0}
