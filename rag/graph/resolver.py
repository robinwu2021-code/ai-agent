"""rag/graph/resolver.py — 实体消解（去重合并）"""
from __future__ import annotations

import asyncio
import math
import time
import uuid
from typing import Any, Callable

import structlog

from rag.graph.models import Edge, Node, NodeType, Triple
from rag.graph.store import KGStore

logger = structlog.get_logger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Pure Python cosine similarity."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class EntityResolver:
    """
    实体消解器：对 LLM 抽取的三元组进行实体去重与合并。

    对每个实体名称：
    1. 先查精确名称匹配（含别名）
    2. 再查向量相似度（若 embed_fn 提供）
    3. 若超过阈值则复用已有节点；否则创建新节点

    Args:
        store: KGStore 实例。
        embed_fn: 嵌入函数 (text: str) -> list[float]，可为 None（此时跳过向量消解）。
        similarity_threshold: 向量相似度阈值，超过则视为同一实体（默认 0.92）。
    """

    def __init__(
        self,
        store: KGStore,
        embed_fn: Callable[[str], Any] | None = None,
        similarity_threshold: float = 0.92,
    ) -> None:
        self._store = store
        self._embed_fn = embed_fn
        self._threshold = similarity_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def resolve(self, triples: list[Triple], kb_id: str) -> list[Triple]:
        """
        对三元组列表进行实体消解，返回规范化后的三元组列表。

        解析步骤：
        - 收集所有实体名称
        - 对每个实体查找或创建对应节点（更新 store）
        - 用规范名称（已解析节点的 name）替换三元组中的实体名称

        Returns:
            list[Triple]: 实体名称已规范化的三元组列表。
        """
        if not triples:
            return []

        # Collect unique entity names and their types/descs from triples
        entity_info: dict[str, tuple[NodeType, str]] = {}
        for triple in triples:
            if triple.src_name not in entity_info:
                entity_info[triple.src_name] = (triple.src_type, triple.src_desc)
            if triple.dst_name not in entity_info:
                entity_info[triple.dst_name] = (triple.dst_type, triple.dst_desc)

        # Resolve each entity: name -> canonical_name
        name_to_canonical: dict[str, str] = {}
        for entity_name, (entity_type, entity_desc) in entity_info.items():
            canonical = await self._resolve_entity(
                name=entity_name,
                entity_type=entity_type,
                description=entity_desc,
                kb_id=kb_id,
            )
            name_to_canonical[entity_name] = canonical

        # Rebuild triples with canonical names
        resolved: list[Triple] = []
        for triple in triples:
            src_canonical = name_to_canonical.get(triple.src_name, triple.src_name)
            dst_canonical = name_to_canonical.get(triple.dst_name, triple.dst_name)
            resolved.append(
                Triple(
                    src_name=src_canonical,
                    src_type=triple.src_type,
                    src_desc=triple.src_desc,
                    relation=triple.relation,
                    dst_name=dst_canonical,
                    dst_type=triple.dst_type,
                    dst_desc=triple.dst_desc,
                    confidence=triple.confidence,
                    evidence=triple.evidence,
                )
            )

        return resolved

    async def merge_nodes(
        self, keep_id: str, merge_id: str, kb_id: str
    ) -> None:
        """
        将 merge_id 节点合并入 keep_id 节点。

        操作：
        1. 将所有指向 merge_id 的边改指向 keep_id
        2. 将 merge_id 的 aliases 和 doc_ids 合并到 keep_id
        3. 删除 merge_id 节点

        Args:
            keep_id: 保留的节点 ID。
            merge_id: 被合并（删除）的节点 ID。
            kb_id: 知识库 ID。
        """
        if keep_id == merge_id:
            return

        keep_node = await self._store.get_node(keep_id)
        merge_node = await self._store.get_node(merge_id)
        if keep_node is None or merge_node is None:
            logger.warning(
                "merge_nodes_not_found",
                keep_id=keep_id,
                merge_id=merge_id,
            )
            return

        # Collect edges that reference merge_id
        src_edges = await self._store.list_edges(kb_id, src_id=merge_id)
        dst_edges = await self._store.list_edges(kb_id, dst_id=merge_id)

        # Re-upsert those edges with keep_id replacing merge_id
        for edge in src_edges:
            new_edge = Edge(
                id=edge.id,
                kb_id=edge.kb_id,
                src_id=keep_id,
                dst_id=edge.dst_id,
                relation=edge.relation,
                weight=edge.weight,
                doc_id=edge.doc_id,
                chunk_id=edge.chunk_id,
                context=edge.context,
                created_at=edge.created_at,
            )
            await self._store.upsert_edge(new_edge)

        for edge in dst_edges:
            new_edge = Edge(
                id=edge.id,
                kb_id=edge.kb_id,
                src_id=edge.src_id,
                dst_id=keep_id,
                relation=edge.relation,
                weight=edge.weight,
                doc_id=edge.doc_id,
                chunk_id=edge.chunk_id,
                context=edge.context,
                created_at=edge.created_at,
            )
            await self._store.upsert_edge(new_edge)

        # Merge aliases and doc_ids
        merged_aliases = list(
            set(keep_node.aliases)
            | set(merge_node.aliases)
            | {merge_node.name}
        )
        merged_doc_ids = list(set(keep_node.doc_ids) | set(merge_node.doc_ids))

        # Update keep node with merged info
        updated_keep = Node(
            id=keep_node.id,
            kb_id=keep_node.kb_id,
            name=keep_node.name,
            node_type=keep_node.node_type,
            aliases=merged_aliases,
            description=keep_node.description or merge_node.description,
            doc_ids=merged_doc_ids,
            embedding=keep_node.embedding or merge_node.embedding,
            degree=keep_node.degree,
            created_at=keep_node.created_at,
        )
        await self._store.upsert_node(updated_keep)

        # Delete the merged node (this also removes its edges from store,
        # but we've already redirected them)
        await self._store.delete_node(merge_id)

        logger.info(
            "nodes_merged",
            keep_id=keep_id,
            merge_id=merge_id,
            merged_name=merge_node.name,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _resolve_entity(
        self,
        name: str,
        entity_type: NodeType,
        description: str,
        kb_id: str,
    ) -> str:
        """
        Find or create a node for the given entity name.

        Returns:
            str: The canonical name (existing node's name or the input name).
        """
        existing = await self._find_similar_node(name=name, kb_id=kb_id)
        if existing is not None:
            # Add name as alias if it's different from canonical
            if name != existing.name and name not in existing.aliases:
                updated_aliases = list(existing.aliases) + [name]
                updated_node = Node(
                    id=existing.id,
                    kb_id=existing.kb_id,
                    name=existing.name,
                    node_type=existing.node_type,
                    aliases=updated_aliases,
                    description=existing.description or description,
                    doc_ids=existing.doc_ids,
                    embedding=existing.embedding,
                    degree=existing.degree,
                    created_at=existing.created_at,
                )
                await self._store.upsert_node(updated_node)
            return existing.name

        # Create new node
        embedding: list[float] | None = None
        if self._embed_fn is not None:
            try:
                embedding = await self._get_embedding(name)
            except Exception as exc:
                logger.warning("embed_failed", name=name, error=str(exc))

        new_node = Node(
            id=str(uuid.uuid4()),
            kb_id=kb_id,
            name=name,
            node_type=entity_type,
            aliases=[],
            description=description,
            doc_ids=[],
            embedding=embedding,
            degree=0,
            created_at=time.time(),
        )
        await self._store.upsert_node(new_node)
        logger.debug("entity_created", name=name, node_id=new_node.id, kb_id=kb_id)
        return name

    async def _find_similar_node(
        self, name: str, kb_id: str
    ) -> Node | None:
        """
        Find an existing node that matches the given entity name.

        Strategy:
        1. Exact name match (or alias match) via store
        2. Embedding cosine similarity > threshold (if embed_fn available)

        Returns:
            Node | None: The matching node or None if not found.
        """
        # Step 1: Exact/alias match
        node = await self._store.get_node_by_name(name, kb_id)
        if node is not None:
            return node

        # Step 2: Embedding similarity
        if self._embed_fn is None:
            return None

        try:
            embedding = await self._get_embedding(name)
        except Exception as exc:
            logger.warning("embed_failed_in_resolve", name=name, error=str(exc))
            return None

        candidates = await self._store.search_nodes_by_embedding(
            embedding, kb_id, limit=5
        )
        for candidate in candidates:
            if candidate.embedding:
                sim = _cosine_similarity(embedding, candidate.embedding)
                if sim >= self._threshold:
                    logger.debug(
                        "entity_resolved_by_embedding",
                        name=name,
                        canonical=candidate.name,
                        similarity=round(sim, 4),
                    )
                    return candidate

        return None

    async def _get_embedding(self, text: str) -> list[float]:
        """Call embed_fn, supporting both sync and async."""
        if asyncio.iscoroutinefunction(self._embed_fn):
            return await self._embed_fn(text)
        else:
            return await asyncio.to_thread(self._embed_fn, text)
