"""rag/keyword_stores/factory.py — 关键词检索存储工厂"""
from __future__ import annotations

from rag.config import KeywordStoreConfig
from rag.keyword_stores.base import BaseKeywordStore


class KeywordStoreFactory:
    @staticmethod
    def create(config: KeywordStoreConfig) -> BaseKeywordStore:
        backend = config.backend.lower()

        if backend == "elasticsearch":
            from rag.keyword_stores.elasticsearch_store import ElasticsearchKeywordStore
            c = config.elasticsearch
            return ElasticsearchKeywordStore(
                url=c.url, index_prefix=c.index_prefix,
                username=c.username, password=c.password, analyzer=c.analyzer,
            )
        if backend == "opensearch":
            from rag.keyword_stores.opensearch_store import OpenSearchKeywordStore
            c = config.opensearch
            return OpenSearchKeywordStore(
                url=c.url, index_prefix=c.index_prefix,
                username=c.username, password=c.password,
            )
        if backend == "memory":
            from rag.keyword_stores.memory_store import MemoryKeywordStore
            c = config.memory
            return MemoryKeywordStore(persist=c.persist, path=c.path)

        raise ValueError(
            f"未知关键词存储 backend: {backend!r}，支持: elasticsearch | opensearch | memory"
        )
