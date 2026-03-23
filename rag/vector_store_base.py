"""
rag/vector_store_base.py — 向量数据库统一抽象接口

所有向量数据库后端（Qdrant、Milvus 等）实现此接口，
PersistentKnowledgeBase 通过此接口调用，不依赖任何具体实现。
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class VectorStoreBase(ABC):
    """向量数据库后端抽象基类。"""

    @abstractmethod
    async def initialize(self) -> None:
        """连接数据库，创建 collection/index（幂等）。"""

    @abstractmethod
    async def upsert_chunks(self, chunks: list[dict]) -> None:
        """
        批量写入或更新分块。

        每个 dict 必须包含：
          chunk_id (str), doc_id (str), kb_id (str),
          text (str), embedding (list[float])
        可选：chunk_index (int), meta (dict), created_at (float)
        """

    @abstractmethod
    async def hybrid_search(
        self,
        query_vec:  list[float],
        query_text: str,
        kb_id:      str,
        top_k:      int = 5,
        rrf_k:      int = 60,
        doc_ids:    list[str] | None = None,
    ) -> list[tuple[float, dict]]:
        """
        混合检索（向量 + 关键词），RRF 融合。
        返回 [(score, payload_dict), ...] 按 score 降序。

        doc_ids: 非空时只在指定文档范围内检索（单文件/多文件问答）
        """

    @abstractmethod
    async def vector_search(
        self,
        query_vec: list[float],
        kb_id:     str,
        top_k:     int = 10,
    ) -> list[tuple[float, dict]]:
        """纯向量检索，返回 [(cosine_score, payload_dict), ...]。"""

    @abstractmethod
    async def delete_by_doc_id(self, doc_id: str) -> int:
        """删除文档的所有分块，返回删除数量。"""

    @abstractmethod
    async def delete_by_kb_id(self, kb_id: str) -> int:
        """删除整个知识库的所有分块，返回删除数量。"""

    @abstractmethod
    async def list_chunks(self, doc_id: str) -> list[dict]:
        """列出文档的所有分块（按 chunk_index 排序）。"""

    @abstractmethod
    async def list_chunks_by_kb(
        self, kb_id: str, limit: int = 2000
    ) -> list[dict]:
        """列出知识库的所有分块。"""

    @abstractmethod
    async def get_stats(self, kb_id: str) -> dict:
        """返回 chunk_count / doc_count / total_chars 等统计。"""

    @abstractmethod
    async def collection_info(self) -> dict:
        """返回底层数据库的 collection 元信息（索引配置等）。"""

    async def close(self) -> None:
        """关闭连接（可选实现）。"""
