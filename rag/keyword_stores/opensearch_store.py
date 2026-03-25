"""
rag/keyword_stores/opensearch_store.py — OpenSearch BM25 关键词检索

AWS/阿里云托管向量检索的选择，API 与 Elasticsearch 兼容。
安装：pip install opensearch-py
"""
from __future__ import annotations

import structlog

from rag.keyword_stores.base import KeywordHit

log = structlog.get_logger(__name__)


class OpenSearchKeywordStore:
    def __init__(
        self,
        url: str = "http://localhost:9200",
        index_prefix: str = "kb_",
        username: str = "",
        password: str = "",
    ) -> None:
        self._url = url
        self._prefix = index_prefix
        self._auth = (username, password) if username else None
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from opensearchpy import AsyncOpenSearch
            kw: dict = {"hosts": [self._url]}
            if self._auth:
                kw["http_auth"] = self._auth
            self._client = AsyncOpenSearch(**kw)
        except ImportError:
            log.warning("opensearch.missing", hint="pip install opensearch-py")
        return self._client

    def _index_name(self, kb_id: str) -> str:
        return f"{self._prefix}{kb_id.lower().replace('/', '_')}"

    async def initialize(self) -> None:
        log.info("opensearch_store.initialized", url=self._url)

    async def index_chunks(self, chunks: list[dict], kb_id: str) -> None:
        client = self._get_client()
        if client is None:
            return
        index = self._index_name(kb_id)
        ops: list = []
        for c in chunks:
            ops.append({"index": {"_index": index, "_id": c["chunk_id"]}})
            ops.append({"chunk_id": c["chunk_id"], "doc_id": c["doc_id"],
                        "kb_id": kb_id, "text": c.get("text", ""),
                        "metadata": c.get("metadata", {})})
        if ops:
            try:
                await client.bulk(body=ops, refresh=True)
            except Exception as exc:
                log.warning("opensearch.index_failed", error=str(exc))

    async def search(self, query: str, kb_id: str,
                     top_k: int = 20, doc_ids: list[str] | None = None) -> list[KeywordHit]:
        client = self._get_client()
        if client is None:
            return []
        index = self._index_name(kb_id)
        try:
            q: dict = {"match": {"text": {"query": query}}}
            if doc_ids:
                q = {"bool": {"must": q, "filter": {"terms": {"doc_id": doc_ids}}}}
            resp = await client.search(index=index, body={"query": q, "size": top_k})
            return [
                KeywordHit(
                    chunk_id=h["_source"]["chunk_id"],
                    doc_id=h["_source"]["doc_id"],
                    kb_id=h["_source"].get("kb_id", kb_id),
                    text=h["_source"].get("text", ""),
                    score=h["_score"],
                    metadata=h["_source"].get("metadata", {}),
                )
                for h in resp["hits"]["hits"]
            ]
        except Exception as exc:
            log.warning("opensearch.search_failed", error=str(exc))
            return []

    async def delete_by_doc_id(self, doc_id: str, kb_id: str) -> None:
        client = self._get_client()
        if client is None:
            return
        try:
            await client.delete_by_query(
                index=self._index_name(kb_id),
                body={"query": {"term": {"doc_id": doc_id}}},
            )
        except Exception as exc:
            log.warning("opensearch.delete_failed", error=str(exc))

    async def delete_by_kb_id(self, kb_id: str) -> None:
        client = self._get_client()
        if client is None:
            return
        try:
            await client.indices.delete(index=self._index_name(kb_id), ignore_unavailable=True)
        except Exception as exc:
            log.warning("opensearch.delete_kb_failed", error=str(exc))
