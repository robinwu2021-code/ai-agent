"""
rag/keyword_stores/elasticsearch_store.py — Elasticsearch 8.x 增强关键词检索

改进点：
  1. 多字段索引：标题字段（3x）、章节标题（2x）、正文（1x）
  2. 短语精确匹配（适合条款编号、技术术语）
  3. 字段级别信息利用（section_level / heading_path）

安装依赖：pip install "elasticsearch[async]>=8.13.0"

切换配置（kb_config.yaml）：
  keyword_store:
    backend: elasticsearch
    elasticsearch:
      url: http://localhost:9200
      index_prefix: kb_
      analyzer: ik_max_word      # 中文 IK 分词器（需安装插件）
"""
from __future__ import annotations

import asyncio
import os

import structlog

from rag.keyword_stores.base import KeywordHit

log = structlog.get_logger(__name__)

# ── 索引 Mapping ──────────────────────────────────────────────────────────────
# 运行时由 _ensure_index 动态注入 analyzer，因此此处使用占位符 {ANALYZER}
_BASE_MAPPING: dict = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
    },
    "mappings": {
        "properties": {
            "chunk_id":      {"type": "keyword"},
            "doc_id":        {"type": "keyword"},
            "kb_id":         {"type": "keyword"},
            # 文档标题 / 来源文件名（权重最高）
            "doc_title":     {"type": "text"},          # analyzer 动态注入
            # 章节标题路径（如 "第2章 > 2.3节 > 安全要求"）
            "headings":      {"type": "text"},          # analyzer 动态注入
            # 正文（主要检索字段）
            "content":       {"type": "text"},          # analyzer 动态注入
            # 正文精确匹配（用更保守的 ik_smart，适合短语/条款号）
            "content_exact": {"type": "text"},          # analyzer 动态注入（ik_smart）
            # 章节深度（0=顶层，1=一级，2=二级…）
            "section_level": {"type": "integer"},
            "metadata":      {"type": "object", "enabled": False},
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
        # 精确匹配字段使用更保守的分词器
        self._exact_analyzer = "ik_smart" if "ik" in analyzer else "standard"
        self._client = None

    # ── 客户端 ───────────────────────────────────────────────────────────────

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
            log.warning(
                "elasticsearch.missing",
                hint="pip install 'elasticsearch[async]>=8.13.0'",
            )
        return self._client

    def _index_name(self, kb_id: str) -> str:
        safe = kb_id.lower().replace("/", "_").replace(" ", "_")
        return f"{self._prefix}{safe}"

    # ── 生命周期 ──────────────────────────────────────────────────────────────

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
                import copy
                mapping = copy.deepcopy(_BASE_MAPPING)
                props = mapping["mappings"]["properties"]
                props["doc_title"]["analyzer"]     = self._analyzer
                props["headings"]["analyzer"]      = self._analyzer
                props["content"]["analyzer"]       = self._analyzer
                props["content_exact"]["analyzer"] = self._exact_analyzer
                await client.indices.create(index=name, body=mapping)
                log.info("elasticsearch.index_created", index=name,
                         analyzer=self._analyzer, exact_analyzer=self._exact_analyzer)
        except Exception as exc:
            log.warning("elasticsearch.ensure_index_failed", index=name, error=str(exc))

    # ── 索引写入 ──────────────────────────────────────────────────────────────

    async def index_chunks(self, chunks: list[dict], kb_id: str) -> None:
        client = self._get_client()
        if client is None:
            return
        await self._ensure_index(kb_id)
        index = self._index_name(kb_id)

        ops: list[dict] = []
        for c in chunks:
            text = c.get("text", "")
            meta = c.get("metadata") or {}

            # ── 从 metadata 提取富字段 ────────────────────────────────────
            # doc_title：来源文件名（去路径、去扩展名）
            source = str(meta.get("source", c.get("doc_id", "")))
            doc_title = os.path.splitext(os.path.basename(source))[0].replace("_", " ")

            # headings：章节标题路径，支持 list 或 string
            hp = meta.get("heading_path", meta.get("headings", ""))
            if isinstance(hp, list):
                headings = " > ".join(str(h) for h in hp if h)
            else:
                headings = str(hp)

            section_level = int(meta.get("section_level", 0) or 0)

            ops.append({"index": {"_index": index, "_id": c["chunk_id"]}})
            ops.append({
                "chunk_id":      c["chunk_id"],
                "doc_id":        c["doc_id"],
                "kb_id":         kb_id,
                "doc_title":     doc_title,
                "headings":      headings,
                "content":       text,
                "content_exact": text,      # 同一文本，不同 analyzer
                "section_level": section_level,
                "metadata":      meta,
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

    # ── 查询 ──────────────────────────────────────────────────────────────────

    def _build_query(self, query: str, doc_ids: list[str] | None = None) -> dict:
        """
        构建复合查询：多字段加权 + 短语精确匹配。

        权重层级：
          doc_title 精确短语 (2.5x) > doc_title 语义 (3x 但宽泛)
          headings (2x) > content 短语 (1.5x) > content 语义 (1x)
        """
        bool_query: dict = {
            "should": [
                # 文档标题匹配（语义宽泛，权重最高）
                {"match": {"doc_title":     {"query": query, "boost": 3.0}}},
                # 文档标题短语精确匹配（适合专有名词）
                {"match_phrase": {"doc_title": {"query": query, "boost": 2.5, "slop": 1}}},
                # 章节标题匹配
                {"match": {"headings":      {"query": query, "boost": 2.0}}},
                # 正文短语精确匹配（条款号、技术术语）
                {"match_phrase": {"content_exact": {"query": query, "boost": 1.5, "slop": 2}}},
                # 正文语义匹配
                {"match": {"content":       {"query": query, "boost": 1.0}}},
            ],
            "minimum_should_match": 1,
        }
        if doc_ids:
            bool_query["filter"] = {"terms": {"doc_id": doc_ids}}
        return {"bool": bool_query}

    async def search(
        self,
        query: str,
        kb_id: str,
        top_k: int = 20,
        doc_ids: list[str] | None = None,
    ) -> list[KeywordHit]:
        client = self._get_client()
        if client is None:
            return []
        index = self._index_name(kb_id)
        try:
            resp = await client.search(
                index=index,
                query=self._build_query(query, doc_ids),
                size=top_k,
                _source=["chunk_id", "doc_id", "kb_id", "content", "metadata",
                         "doc_title", "headings", "section_level"],
            )
            hits = []
            for h in resp["hits"]["hits"]:
                src = h["_source"]
                hits.append(KeywordHit(
                    chunk_id=src["chunk_id"],
                    doc_id=src["doc_id"],
                    kb_id=src.get("kb_id", kb_id),
                    text=src.get("content", ""),
                    score=h["_score"],
                    metadata=src.get("metadata", {}),
                ))
            return hits
        except Exception as exc:
            log.warning("elasticsearch.search_failed", error=str(exc))
            return []

    # ── 删除 ──────────────────────────────────────────────────────────────────

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
            await client.indices.delete(
                index=self._index_name(kb_id), ignore_unavailable=True
            )
        except Exception as exc:
            log.warning("elasticsearch.delete_kb_failed", kb_id=kb_id, error=str(exc))
