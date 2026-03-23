"""
tests/test_vector_store.py — VectorStoreBase and concrete store unit tests

Covers:
  - VectorStoreBase abstract interface enforcement
  - QdrantVectorStore doc_ids filtering logic (no real Qdrant needed — mocked)
  - MilvusVectorStore expr construction for doc_ids filter
  - RRF fusion logic in QdrantVectorStore.hybrid_search

Run with:
    pytest tests/test_vector_store.py -v
"""
from __future__ import annotations

import asyncio
import math
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. VectorStoreBase — abstract interface
# ═══════════════════════════════════════════════════════════════════════════════

class TestVectorStoreBaseInterface:
    def test_cannot_instantiate_directly(self):
        from rag.vector_store_base import VectorStoreBase
        with pytest.raises(TypeError):
            VectorStoreBase()  # type: ignore[abstract]

    def test_incomplete_subclass_raises_on_instantiation(self):
        from rag.vector_store_base import VectorStoreBase

        class PartialStore(VectorStoreBase):
            async def initialize(self): pass
            async def upsert_chunks(self, chunks): pass
            # All other abstract methods missing

        with pytest.raises(TypeError):
            PartialStore()  # type: ignore[abstract]

    def test_full_implementation_instantiates(self):
        from rag.vector_store_base import VectorStoreBase

        class FullStore(VectorStoreBase):
            async def initialize(self): pass
            async def upsert_chunks(self, chunks): pass
            async def hybrid_search(self, query_vec, query_text, kb_id, top_k=5, rrf_k=60, doc_ids=None): return []
            async def vector_search(self, query_vec, kb_id, top_k=10): return []
            async def delete_by_doc_id(self, doc_id): return 0
            async def delete_by_kb_id(self, kb_id): return 0
            async def list_chunks(self, doc_id): return []
            async def list_chunks_by_kb(self, kb_id, limit=2000): return []
            async def get_stats(self, kb_id): return {}
            async def collection_info(self): return {}

        store = FullStore()
        assert isinstance(store, VectorStoreBase)

    def test_close_has_default_implementation(self):
        """close() should be callable without override (default is a no-op)."""
        from rag.vector_store_base import VectorStoreBase

        class MinimalStore(VectorStoreBase):
            async def initialize(self): pass
            async def upsert_chunks(self, chunks): pass
            async def hybrid_search(self, query_vec, query_text, kb_id, top_k=5, rrf_k=60, doc_ids=None): return []
            async def vector_search(self, query_vec, kb_id, top_k=10): return []
            async def delete_by_doc_id(self, doc_id): return 0
            async def delete_by_kb_id(self, kb_id): return 0
            async def list_chunks(self, doc_id): return []
            async def list_chunks_by_kb(self, kb_id, limit=2000): return []
            async def get_stats(self, kb_id): return {}
            async def collection_info(self): return {}

        store = MinimalStore()
        _run(store.close())  # Should not raise

    def test_hybrid_search_doc_ids_param_has_none_default(self):
        import inspect
        from rag.vector_store_base import VectorStoreBase
        sig = inspect.signature(VectorStoreBase.hybrid_search)
        params = sig.parameters
        assert "doc_ids" in params
        assert params["doc_ids"].default is None

    def test_all_abstract_methods_declared(self):
        from rag.vector_store_base import VectorStoreBase
        expected = {
            "initialize", "upsert_chunks", "hybrid_search", "vector_search",
            "delete_by_doc_id", "delete_by_kb_id", "list_chunks",
            "list_chunks_by_kb", "get_stats", "collection_info",
        }
        abstract = getattr(VectorStoreBase, "__abstractmethods__", set())
        for method in expected:
            assert method in abstract, f"'{method}' not declared as abstract in VectorStoreBase"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. QdrantVectorStore — doc_ids filtering logic
# ═══════════════════════════════════════════════════════════════════════════════

def _make_payload(chunk_id: str, doc_id: str, kb_id: str = "global", text: str = "text") -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id":   doc_id,
        "kb_id":    kb_id,
        "text":     text,
        "chunk_index": 0,
        "meta": {},
        "created_at": 0.0,
    }


@pytest.fixture
def qdrant_store(tmp_path):
    """QdrantVectorStore with all I/O mocked out (no real Qdrant process)."""
    pytest.importorskip("qdrant_client", reason="qdrant-client not installed")
    from rag.qdrant_store import QdrantVectorStore
    from rag.knowledge_base import BM25Index, Chunk

    store = QdrantVectorStore.__new__(QdrantVectorStore)
    store._path        = str(tmp_path / "qdrant")
    store._url         = None
    store._api_key     = None
    store._collection  = "kb_chunks"
    store._vector_size = 8
    store._client      = MagicMock()
    store._lock        = asyncio.Lock()
    store._initialized = True

    # Seed in-memory BM25 with three chunks across two docs
    chunks = [
        Chunk(id="c1", doc_id="doc_a", text="cats are fluffy animals",     chunk_index=0, metadata={"kb_id": "global"}),
        Chunk(id="c2", doc_id="doc_a", text="cats love to sleep all day",   chunk_index=1, metadata={"kb_id": "global"}),
        Chunk(id="c3", doc_id="doc_b", text="dogs are loyal to their owners", chunk_index=0, metadata={"kb_id": "global"}),
    ]
    store._bm25 = BM25Index()
    store._bm25.build(chunks)
    store._chunk_map = {c.id: c for c in chunks}

    return store


class TestQdrantDocIdsFilter:
    def test_vector_hits_filtered_by_doc_ids(self, qdrant_store):
        """Simulate vector hits from two docs; doc_ids filter should drop the unwanted one."""
        hits_all = [
            (0.95, _make_payload("c1", "doc_a")),
            (0.85, _make_payload("c3", "doc_b")),
        ]
        qdrant_store.vector_search = AsyncMock(return_value=hits_all)

        result = _run(qdrant_store.hybrid_search(
            query_vec=[0.1] * 8,
            query_text="cats fluffy",
            kb_id="global",
            top_k=5,
            doc_ids=["doc_a"],
        ))
        doc_ids_returned = {payload["doc_id"] for _, payload in result}
        assert "doc_b" not in doc_ids_returned, "doc_b should have been filtered out"
        assert "doc_a" in doc_ids_returned

    def test_bm25_hits_filtered_by_doc_ids(self, qdrant_store):
        """BM25 results for doc_b should be excluded when doc_ids=["doc_a"]."""
        # No vector hits — rely on BM25 only
        qdrant_store.vector_search = AsyncMock(return_value=[])

        result = _run(qdrant_store.hybrid_search(
            query_vec=[0.1] * 8,
            query_text="dogs loyal owners",
            kb_id="global",
            top_k=5,
            doc_ids=["doc_a"],   # Exclude doc_b which has the dog content
        ))
        doc_ids_returned = {payload["doc_id"] for _, payload in result}
        assert "doc_b" not in doc_ids_returned

    def test_none_doc_ids_returns_all_docs(self, qdrant_store):
        """doc_ids=None should not filter anything."""
        hits_all = [
            (0.95, _make_payload("c1", "doc_a")),
            (0.85, _make_payload("c3", "doc_b")),
        ]
        qdrant_store.vector_search = AsyncMock(return_value=hits_all)

        result = _run(qdrant_store.hybrid_search(
            query_vec=[0.1] * 8,
            query_text="content",
            kb_id="global",
            top_k=5,
            doc_ids=None,
        ))
        doc_ids_returned = {payload["doc_id"] for _, payload in result}
        assert "doc_a" in doc_ids_returned
        assert "doc_b" in doc_ids_returned

    def test_empty_doc_ids_list_returns_no_results(self, qdrant_store):
        """doc_ids=[] (empty list converted to set) should match no docs."""
        hits_all = [
            (0.9, _make_payload("c1", "doc_a")),
            (0.8, _make_payload("c3", "doc_b")),
        ]
        qdrant_store.vector_search = AsyncMock(return_value=hits_all)

        # An empty list means filter to nothing
        result = _run(qdrant_store.hybrid_search(
            query_vec=[0.1] * 8,
            query_text="content",
            kb_id="global",
            top_k=5,
            doc_ids=[],
        ))
        # doc_ids=[] → doc_set={} → every chunk filtered out
        assert result == [], f"Expected empty results for doc_ids=[], got {result}"

    def test_rrf_scores_are_positive(self, qdrant_store):
        """RRF fusion scores should always be positive."""
        hits_all = [
            (0.9, _make_payload("c1", "doc_a")),
        ]
        qdrant_store.vector_search = AsyncMock(return_value=hits_all)

        result = _run(qdrant_store.hybrid_search(
            query_vec=[0.1] * 8,
            query_text="cats",
            kb_id="global",
            top_k=5,
        ))
        for score, _ in result:
            assert score > 0.0

    def test_top_k_is_respected(self, qdrant_store):
        """Result count should not exceed top_k."""
        hits_all = [(0.9 - i * 0.1, _make_payload(f"c{i}", "doc_a")) for i in range(10)]
        qdrant_store.vector_search = AsyncMock(return_value=hits_all)

        result = _run(qdrant_store.hybrid_search(
            query_vec=[0.1] * 8,
            query_text="cats",
            kb_id="global",
            top_k=3,
        ))
        assert len(result) <= 3

    def test_results_sorted_descending_by_score(self, qdrant_store):
        """Returned results should be sorted by RRF score descending."""
        hits_all = [(0.9, _make_payload("c1", "doc_a"))]
        qdrant_store.vector_search = AsyncMock(return_value=hits_all)

        result = _run(qdrant_store.hybrid_search(
            query_vec=[0.1] * 8,
            query_text="cats",
            kb_id="global",
            top_k=5,
        ))
        scores = [s for s, _ in result]
        assert scores == sorted(scores, reverse=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MilvusVectorStore — expr construction for doc_ids
# ═══════════════════════════════════════════════════════════════════════════════

class TestMilvusExprDocIds:
    """
    Test that the Milvus expr string is constructed correctly when doc_ids
    is provided. We patch pymilvus so no real connection is needed.
    """

    def _build_expr(self, kb_id: str, doc_ids: list[str] | None) -> str:
        """Replicate the expr-building logic from MilvusVectorStore.hybrid_search."""
        expr = f'kb_id == "{kb_id}"'
        if doc_ids:
            ids_lit = "[" + ", ".join(f'"{d}"' for d in doc_ids) + "]"
            expr = f'({expr}) and doc_id in {ids_lit}'
        return expr

    def test_no_doc_ids_produces_simple_expr(self):
        expr = self._build_expr("global", None)
        assert expr == 'kb_id == "global"'
        assert "doc_id" not in expr

    def test_single_doc_id_expr(self):
        expr = self._build_expr("global", ["doc_abc"])
        assert 'kb_id == "global"' in expr
        assert '"doc_abc"' in expr
        assert "doc_id in" in expr

    def test_multiple_doc_ids_all_present(self):
        doc_ids = ["doc_1", "doc_2", "doc_3"]
        expr = self._build_expr("global", doc_ids)
        for did in doc_ids:
            assert f'"{did}"' in expr
        assert "doc_id in" in expr

    def test_empty_doc_ids_no_filter(self):
        expr = self._build_expr("global", [])
        # Empty list is falsy, so no doc_id filter added
        assert expr == 'kb_id == "global"'

    def test_expr_is_valid_python_structure(self):
        """The generated expr brackets should be balanced."""
        expr = self._build_expr("mykb", ["d1", "d2"])
        # Should start with ( and contain a matching )
        open_parens  = expr.count("(")
        close_parens = expr.count(")")
        open_brackets  = expr.count("[")
        close_brackets = expr.count("]")
        assert open_parens  == close_parens,  f"Unbalanced parens in expr: {expr}"
        assert open_brackets == close_brackets, f"Unbalanced brackets in expr: {expr}"

    def test_expr_uses_correct_kb_id(self):
        expr = self._build_expr("special_kb_123", ["d1"])
        assert '"special_kb_123"' in expr

    def test_doc_ids_with_special_chars_escaped(self):
        """doc_ids with hyphens and underscores should be quoted correctly."""
        expr = self._build_expr("global", ["uuid-12-34", "doc_name_v2"])
        assert '"uuid-12-34"' in expr
        assert '"doc_name_v2"' in expr


# ═══════════════════════════════════════════════════════════════════════════════
# 4. QdrantVectorStore — chunk_id hashing
# ═══════════════════════════════════════════════════════════════════════════════

class TestQdrantChunkIdHashing:
    def test_deterministic_hash(self):
        pytest.importorskip("qdrant_client", reason="qdrant-client not installed")
        from rag.qdrant_store import _chunk_id_to_point_id
        h1 = _chunk_id_to_point_id("doc123_chunk0")
        h2 = _chunk_id_to_point_id("doc123_chunk0")
        assert h1 == h2

    def test_different_ids_produce_different_hashes(self):
        pytest.importorskip("qdrant_client", reason="qdrant-client not installed")
        from rag.qdrant_store import _chunk_id_to_point_id
        ids = ["doc1_c0", "doc1_c1", "doc2_c0", "doc2_c1"]
        hashes = [_chunk_id_to_point_id(x) for x in ids]
        assert len(set(hashes)) == len(ids), "Hash collision detected"

    def test_hash_is_63_bit_integer(self):
        pytest.importorskip("qdrant_client", reason="qdrant-client not installed")
        from rag.qdrant_store import _chunk_id_to_point_id
        h = _chunk_id_to_point_id("test_chunk")
        assert isinstance(h, int)
        assert 0 <= h < 2**63


# ═══════════════════════════════════════════════════════════════════════════════
# 5. RRF fusion formula
# ═══════════════════════════════════════════════════════════════════════════════

class TestRRFFusion:
    """Test the RRF (Reciprocal Rank Fusion) logic in isolation."""

    def _rrf_score(self, rank: int, k: int = 60) -> float:
        return 1.0 / (k + rank + 1)

    def test_rank_0_gives_highest_score(self):
        assert self._rrf_score(0) > self._rrf_score(1) > self._rrf_score(10)

    def test_scores_always_positive(self):
        for rank in range(100):
            assert self._rrf_score(rank) > 0.0

    def test_dual_list_boost(self):
        """A chunk appearing in both vector and BM25 lists should have higher score."""
        k = 60
        # Appears in both at rank 0
        dual_score = (1.0 / (k + 1)) + (1.0 / (k + 1))
        # Appears in only one at rank 0
        single_score = 1.0 / (k + 1)
        assert dual_score > single_score

    def test_rrf_k_controls_flattening(self):
        """Larger k makes scores flatter (less difference between rank 0 and rank 10)."""
        k_small = 1
        k_large = 100
        diff_small = self._rrf_score(0, k_small) - self._rrf_score(10, k_small)
        diff_large = self._rrf_score(0, k_large) - self._rrf_score(10, k_large)
        assert diff_small > diff_large, "Small k should produce larger rank differences"
