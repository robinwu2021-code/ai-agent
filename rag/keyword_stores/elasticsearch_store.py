"""
rag/keyword_stores/elasticsearch_store.py — Elasticsearch 8.x BM25 关键词检索

支持中文 IK 分词（需安装 elasticsearch-analysis-ik 插件）。
安装：pip install "elasticsearch[async]>=8.13.0"

切换配置示例（kb_config.yaml）：
  keyword_store:
    backend: elasticsearch
    elasticsearch:
      url: http://localhost:9200
      index_prefix: kb_
      analyzer: ik_max_word
"""
from __future__ import annotations

import asyncio

import structlog

from rag.keyword_stores.base import KeywordHit

log = structlog.get_logger(__name__)

_MAPPING = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
    },
    "mappings": {
        "properties": {
            "chunk_id":   {"type": "keyword"},
            "doc_id":     {"type": "keyword"},
            "kb_id":      {"type": "keyword"},
            "text":       {"type": "text"},           # analyzer 动态注入
            "metadata":   {"type": "object", "enabled": False},
        }
    },
}


class ElasticsearchKeywordStore:
    def __init__(
        self,
        url: str = "http://localhost:9200",
        index_prefix: str = "kb_",
        username: str = "",
        password: str = "",
        analyzer: str = "ik_max_word",
    ) -> None:
        self._url = url
        self._prefix = index_prefix
        self._auth = (username, password) if username else None
        self._analyzer = analyzer
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from elasticsearch import AsyncElasticsearch
            kw: dict = {"hosts": [self._url]}
            if self._auth:
                kw["basic_auth"] = self._auth
            self._client = AsyncElasticsearch(**kw)
        except ImportError:
            log.warning("elasticsearch.missing",
                        hint="pip install 'elasticsearch[async]>=8.13.0'")
        return self._client

    def _index_name(self, kb_id: str) -> str:
        safe = kb_id.lower().replace("/", "_").replace(" ", "_")
        return f"{self._prefix}{safe}"

    async def initialize(self) -> None:
        log.info("elasticsearch_store.initialized", url=self._url)

    async def _ensure_index(self, kb_id: str) -> None:
        client = self._get_client()
        if client is None:
            return
        name = self._index_name(kb_id)
        try:
            exists = await client.indices.exists(index=name)
            if not exists:
                mapping = dict(_MAPPING)
                mapping["mappings"]["properties"]["text"]["analyzer"] = self._analyzer
                await client.indices.create(index=name, body=mapping)
                log.info("elasticsearch.index_created", index=name)
        except Exception as exc:
            log.warning("elasticsearch.ensure_index_failed", index=name, error=str(exc))

    async def index_chunks(self, chunks: list[dict], kb_id: str) -> None:
        client = self._get_client()
        if client is None:
            return
        await self._ensure_index(kb_id)
        index = self._index_name(kb_id)

        ops: list[dict] = []
        for c in chunks:
            ops.append({"index": {"_index": index, "_id": c["chunk_id"]}})
            ops.append({
                "chunk_id": c["chunk_id"],
                "doc_id":   c["doc_id"],
                "kb_id":    kb_id,
                "text":     c.get("text", ""),
                "metadata": c.get("metadata", {}),
            })
        if ops:
            try:
                resp = await client.bulk(operations=ops, refresh=True)
                if resp.get("errors"):
                    log.warning("elasticsearch.bulk_errors", kb_id=kb_id)
                else:
                    log.debug("elasticsearch.indexed", kb_id=kb_id, count=len(chunks))
            except Exception as exc:
                log.warning("elasticsearch.index_failed", error=str(exc))

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
            resp = await client.search(
                index=index,
                query=q,
                size=top_k,
                _source=["chunk_id", "doc_id", "kb_id", "text", "metadata"],
            )
            hits = []
            for h in resp["hits"]["hits"]:
                src = h["_source"]
                hits.append(KeywordHit(
                    chunk_id=src["chunk_id"],
                    doc_id=src["doc_id"],
                    kb_id=src.get("kb_id", kb_id),
                    text=src.get("text", ""),
                    score=h["_score"],
                    metadata=src.get("metadata", {}),
                ))
            return hits
        except Exception as exc:
            log.warning("elasticsearch.search_failed", error=str(exc))
            return []

    async def delete_by_doc_id(self, doc_id: str, kb_id: str) -> None:
        client = self._get_client()
        if client is None:
            return
        try:
            await client.delete_by_query(
                index=self._index_name(kb_id),
                query={"term": {"doc_id": doc_id}},
                refresh=True,
            )
        except Exception as exc:
            log.warning("elasticsearch.delete_failed", doc_id=doc_id, error=str(exc))

    async def delete_by_kb_id(self, kb_id: str) -> None:
        client = self._get_client()
        if client is None:
            return
        try:
            await client.indices.delete(index=self._index_name(kb_id), ignore_unavailable=True)
        except Exception as exc:
            log.warning("elasticsearch.delete_kb_failed", kb_id=kb_id, error=str(exc))
