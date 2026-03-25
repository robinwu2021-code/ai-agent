"""
rag/kb_manager.py — 知识库统一入口（Facade）

组装所有可插拔组件，对外提供简洁的操作接口：
  - upload_file / update_file / delete_file / restore_version
  - query
  - list_files / get_file / get_history / get_audit_log
  - create_directory / grant_permission / check_access

组件依赖图：
  KBManager
    ├── IngestionPipeline
    │     ├── Parser（docling/unstructured/markitdown）
    │     ├── SmartChunker（structural/semantic/sentence/fixed + contextual）
    │     ├── Embedder（qwen/openai/bge_local）
    │     ├── VectorStore（qdrant/milvus/chroma）
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
from rag.pipeline.ingestion import IngestionPipeline
from rag.pipeline.query import QueryPipeline, QueryResult

log = structlog.get_logger(__name__)


class KBManager:
    """
    知识库统一入口。

    用法：
        mgr = await KBManager.create()            # 用 kb_config.yaml 自动组装
        file = await mgr.upload_file(
            file_path="report.pdf",
            workspace_id="ws_01",
            kb_id="kb_finance",
            directory_id="dir_01",
            operator=OperatorContext(user_id="alice"),
        )
        result = await mgr.query("上周销售额是多少？", kb_id="kb_finance")
        print(result.context)
    """

    def __init__(
        self,
        ingestion: IngestionPipeline,
        query_pipeline: QueryPipeline,
        file_manager: KBFileManager,
        permission_manager: PermissionManager,
    ) -> None:
        self._ingestion = ingestion
        self._query = query_pipeline
        self._files = file_manager
        self._perms = permission_manager

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
        """
        cfg = config or get_kb_config()

        # ── 解析器 ─────────────────────────────────────────────────
        from rag.parsers.factory import ParserFactory
        parser = ParserFactory.create(cfg.parser)

        # ── 分块器 ─────────────────────────────────────────────────
        from rag.chunkers.factory import ChunkerFactory
        chunker = ChunkerFactory.create(cfg.chunker, llm_fn=llm_fn)

        # ── Embedder ───────────────────────────────────────────────
        from rag.embedders.factory import EmbedderFactory
        embedder = EmbedderFactory.create(cfg.embedder)

        # ── 向量存储（复用现有 QdrantVectorStore / MilvusVectorStore）─
        vector_store = await cls._create_vector_store(cfg)

        # ── 关键词存储 ─────────────────────────────────────────────
        from rag.keyword_stores.factory import KeywordStoreFactory
        keyword_store = KeywordStoreFactory.create(cfg.keyword_store)
        await keyword_store.initialize()

        # ── 图谱（可选，使用现有 GraphBuilder）───────────────────────
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

        # ── 权限管理 ───────────────────────────────────────────────
        perm_manager = PermissionManager(db_url=fm_cfg.db_url)

        # ── Reranker ───────────────────────────────────────────────
        reranker = cls._create_reranker(cfg, llm_fn) if cfg.reranker.enabled else None

        # ── 流水线组装 ─────────────────────────────────────────────
        s = cfg.file_management.summary
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
        )

        log.info("kb_manager.created",
                 parser=cfg.parser.backend,
                 chunker=cfg.chunker.strategy,
                 embedder=cfg.embedder.backend,
                 vector_store=cfg.vector_store.backend,
                 keyword_store=cfg.keyword_store.backend,
                 graph=cfg.graph.enabled)

        return cls(
            ingestion=ingestion,
            query_pipeline=query_pipeline,
            file_manager=file_manager,
            permission_manager=perm_manager,
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
    ):
        """上传文件并触发异步索引流水线。"""
        kb_file = await self._files.upload(
            file_path=file_path,
            workspace_id=workspace_id,
            kb_id=kb_id,
            directory_id=directory_id,
            operator=operator,
            change_description=change_description,
        )
        # 异步触发索引（不阻塞返回）
        import asyncio
        asyncio.create_task(self._ingestion.ingest(
            file_path=file_path,
            file_id=kb_file.id,
            version_id=kb_file.current_version_id or "",
            kb_id=kb_id,
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
        # 获取旧摘要用于 diff
        old_file = await self._files._store.get_file(file_id)
        prev_summary = old_file.summary if old_file else ""

        kb_file = await self._files.update(
            file_id=file_id,
            new_file_path=new_file_path,
            operator=operator,
            change_description=change_description,
        )
        # 先清除旧索引
        await self._ingestion.delete_doc(file_id, kb_file.kb_id)
        # 重新索引
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
        return await self._query.query(
            query_text=query_text,
            kb_id=kb_id,
            top_k=top_k,
            doc_ids=doc_ids,
        )

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
    async def _create_vector_store(cfg: KBConfig):
        backend = cfg.vector_store.backend.lower()
        if backend == "qdrant":
            try:
                from rag.qdrant_store import QdrantVectorStore
                c = cfg.vector_store.qdrant
                store = QdrantVectorStore(
                    path=c.path if c.mode == "embedded" else None,
                    url=c.url if c.mode == "server" else None,
                    collection=c.collection,
                    vector_size=cfg.embedder.qwen.dimensions
                    if cfg.embedder.backend == "qwen"
                    else cfg.embedder.openai.dimensions,
                )
                await store.initialize()
                return store
            except Exception as exc:
                log.warning("kb_manager.qdrant_init_failed", error=str(exc))
        if backend == "milvus":
            try:
                from rag.milvus_store import MilvusVectorStore
                c = cfg.vector_store.milvus
                store = MilvusVectorStore(
                    uri=c.uri if c.mode == "lite" else f"http://{c.host}:{c.port}",
                )
                await store.initialize()
                return store
            except Exception as exc:
                log.warning("kb_manager.milvus_init_failed", error=str(exc))
        # 兜底：内存向量存储（已有实现）
        from rag.knowledge_base import HybridRetriever
        log.warning("kb_manager.using_memory_vector_store")

        class _MemVectorStore:
            """薄包装，使 HybridRetriever 符合 upsert_chunks/hybrid_search 接口。"""
            def __init__(self):
                self._chunks: list = []
                self._embs: list = []
                from rag.knowledge_base import BM25Index
                self._bm25 = BM25Index()

            async def upsert_chunks(self, payloads: list[dict]):
                from rag.knowledge_base import Chunk
                for p in payloads:
                    c = Chunk(id=p["chunk_id"], doc_id=p["doc_id"],
                              text=p["text"], chunk_index=p.get("chunk_index", 0),
                              metadata=p.get("metadata", {}),
                              embedding=p.get("embedding", []))
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
                self._chunks = [c for c in self._chunks if c.doc_id != doc_id]
                self._embs = [e for c, e in zip(self._chunks, self._embs) if c.doc_id != doc_id]

            async def get_stats(self, kb_id):
                return {"chunk_count": len(self._chunks), "total_chars": 0}

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
                    from rag.pipeline.query import RetrievedChunk
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
