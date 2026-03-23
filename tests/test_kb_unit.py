"""
tests/test_kb_unit.py — Knowledge Base unit tests (no network / no LLM calls)

Run with:
    pytest tests/test_kb_unit.py -v
"""
from __future__ import annotations

import json
import math
import time
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ─── helpers imported directly ────────────────────────────────────────────────
from rag.store import (
    KBChunk,
    KBDocument,
    SQLiteKBStore,
    _cosine_similarity,
    _deserialize_embedding,
    _serialize_embedding,
)
from rag.persistent_kb import PersistentKnowledgeBase, _chunk_to_kb_chunk, _kb_chunk_to_chunk
from rag.knowledge_base import Chunk


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Embedding serialization round-trip
# ═══════════════════════════════════════════════════════════════════════════════

class TestEmbeddingSerialization:
    def test_roundtrip_json(self):
        emb = [0.1, 0.2, -0.3, 0.999, 1.0, -1.0]
        blob = _serialize_embedding(emb)
        assert isinstance(blob, bytes)
        # Must be valid JSON
        assert json.loads(blob) == pytest.approx(emb)
        # Must deserialize correctly
        assert _deserialize_embedding(blob) == pytest.approx(emb)

    def test_none_roundtrip(self):
        assert _serialize_embedding(None) is None
        assert _deserialize_embedding(None) is None

    def test_large_vector(self):
        emb = [float(i) / 1000 for i in range(1536)]
        blob = _serialize_embedding(emb)
        result = _deserialize_embedding(blob)
        assert len(result) == 1536
        assert result[0] == pytest.approx(0.0)
        assert result[100] == pytest.approx(0.1)

    def test_legacy_pickle_fallback(self):
        """Pickle-encoded embeddings from the old format should still deserialize."""
        import pickle
        emb = [0.5, -0.5, 0.1]
        blob = pickle.dumps(emb)
        result = _deserialize_embedding(blob)
        assert result == pytest.approx(emb)

    def test_invalid_blob_returns_none(self):
        assert _deserialize_embedding(b"not_json_not_pickle_xyzzy") is None


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Cosine similarity
# ═══════════════════════════════════════════════════════════════════════════════

class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert _cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_empty_vectors(self):
        assert _cosine_similarity([], []) == 0.0

    def test_dimension_mismatch_returns_zero(self):
        assert _cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0]) == 0.0

    def test_normalized_vectors(self):
        a = [1 / math.sqrt(2), 1 / math.sqrt(2)]
        b = [1 / math.sqrt(2), 1 / math.sqrt(2)]
        assert _cosine_similarity(a, b) == pytest.approx(1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SQLiteKBStore — CRUD
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def kb_store(tmp_path):
    """A fresh in-memory (tmp file) SQLite store for each test."""
    db_path = str(tmp_path / "test_kb.db")
    store = SQLiteKBStore(db_path=db_path)
    asyncio.get_event_loop().run_until_complete(store.initialize())
    return store


class TestSQLiteKBStore:
    def test_save_and_get_document(self, kb_store):
        doc = KBDocument(
            doc_id="doc1", kb_id="kb_test", filename="test.md",
            source="/tmp/test.md", status="pending", created_at=time.time(),
        )
        saved = asyncio.get_event_loop().run_until_complete(kb_store.save_document(doc))
        fetched = asyncio.get_event_loop().run_until_complete(kb_store.get_document("doc1"))
        assert fetched is not None
        assert fetched.doc_id == "doc1"
        assert fetched.filename == "test.md"

    def test_list_documents_by_kb(self, kb_store):
        loop = asyncio.get_event_loop()
        for i in range(3):
            loop.run_until_complete(kb_store.save_document(KBDocument(
                doc_id=f"d{i}", kb_id="kb_a", filename=f"f{i}.txt",
                source=f"/f{i}.txt", created_at=time.time(),
            )))
        loop.run_until_complete(kb_store.save_document(KBDocument(
            doc_id="d_other", kb_id="kb_b", filename="other.txt",
            source="/other.txt", created_at=time.time(),
        )))
        docs = loop.run_until_complete(kb_store.list_documents("kb_a"))
        assert len(docs) == 3
        assert all(d.kb_id == "kb_a" for d in docs)

    def test_update_document_status(self, kb_store):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(kb_store.save_document(KBDocument(
            doc_id="d_status", kb_id="kb_test", filename="s.txt",
            source="/s.txt", status="pending", created_at=time.time(),
        )))
        loop.run_until_complete(kb_store.update_document_status("d_status", "ready", chunk_count=5))
        doc = loop.run_until_complete(kb_store.get_document("d_status"))
        assert doc.status == "ready"
        assert doc.chunk_count == 5

    def test_save_and_list_chunks(self, kb_store):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(kb_store.save_document(KBDocument(
            doc_id="doc_c", kb_id="kb_test", filename="c.txt",
            source="/c.txt", created_at=time.time(),
        )))
        emb = [0.1, 0.2, 0.3]
        chunks = [
            KBChunk(chunk_id=f"c{i}", doc_id="doc_c", kb_id="kb_test",
                    chunk_index=i, text=f"chunk {i}", embedding=emb,
                    created_at=time.time())
            for i in range(4)
        ]
        loop.run_until_complete(kb_store.save_chunks(chunks))
        result = loop.run_until_complete(kb_store.list_chunks("doc_c"))
        assert len(result) == 4
        assert result[0].embedding == pytest.approx(emb)

    def test_update_chunk_embedding(self, kb_store):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(kb_store.save_document(KBDocument(
            doc_id="doc_emb", kb_id="kb_test", filename="e.txt",
            source="/e.txt", created_at=time.time(),
        )))
        chunk = KBChunk(chunk_id="emb_chunk", doc_id="doc_emb", kb_id="kb_test",
                        chunk_index=0, text="hello", embedding=None, created_at=time.time())
        loop.run_until_complete(kb_store.save_chunk(chunk))
        new_emb = [0.9, 0.8, 0.7]
        loop.run_until_complete(kb_store.update_chunk_embedding("emb_chunk", new_emb))
        chunks = loop.run_until_complete(kb_store.list_chunks("doc_emb"))
        assert chunks[0].embedding == pytest.approx(new_emb)

    def test_delete_document_cascades(self, kb_store):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(kb_store.save_document(KBDocument(
            doc_id="doc_del", kb_id="kb_test", filename="del.txt",
            source="/del.txt", created_at=time.time(),
        )))
        loop.run_until_complete(kb_store.save_chunk(KBChunk(
            chunk_id="del_c1", doc_id="doc_del", kb_id="kb_test",
            chunk_index=0, text="will be deleted", created_at=time.time(),
        )))
        loop.run_until_complete(kb_store.delete_document("doc_del"))
        assert loop.run_until_complete(kb_store.get_document("doc_del")) is None
        assert loop.run_until_complete(kb_store.list_chunks("doc_del")) == []

    def test_get_stats(self, kb_store):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(kb_store.save_document(KBDocument(
            doc_id="doc_stat", kb_id="kb_stat", filename="stat.txt",
            source="/stat.txt", status="ready", char_count=500, created_at=time.time(),
        )))
        loop.run_until_complete(kb_store.save_chunk(KBChunk(
            chunk_id="stat_c", doc_id="doc_stat", kb_id="kb_stat",
            chunk_index=0, text="stat chunk", created_at=time.time(),
        )))
        stats = loop.run_until_complete(kb_store.get_stats("kb_stat"))
        assert stats["documents"] == 1
        assert stats["chunks"] == 1
        assert stats["ready_documents"] == 1
        assert stats["total_chars"] == 500


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PersistentKnowledgeBase — kb_id isolation
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def fake_embed_fn():
    """Returns a deterministic embedding based on text length."""
    async def _embed(text: str) -> list[float]:
        n = len(text)
        # Simple deterministic vector: normalised
        v = [float((n + i) % 17) for i in range(8)]
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]
    return _embed


@pytest.fixture
def pkb(tmp_path, fake_embed_fn):
    db_path = str(tmp_path / "pkb_test.db")
    store = SQLiteKBStore(db_path=db_path)
    asyncio.get_event_loop().run_until_complete(store.initialize())
    kb = PersistentKnowledgeBase(store=store, embed_fn=fake_embed_fn, kb_id="global")
    asyncio.get_event_loop().run_until_complete(kb.initialize())
    return kb


class TestKBIdIsolation:
    def test_query_only_returns_own_kb(self, pkb, fake_embed_fn, tmp_path):
        loop = asyncio.get_event_loop()
        # Add text to two separate kb_ids
        loop.run_until_complete(pkb.add_text("Hello from kb_a", source="src_a", kb_id="kb_a"))
        loop.run_until_complete(pkb.add_text("Hello from kb_b", source="src_b", kb_id="kb_b"))

        results_a = loop.run_until_complete(pkb.query("Hello", kb_id="kb_a", top_k=10))
        results_b = loop.run_until_complete(pkb.query("Hello", kb_id="kb_b", top_k=10))

        assert all(c.kb_id == "kb_a" for c in results_a), "kb_b results leaked into kb_a"
        assert all(c.kb_id == "kb_b" for c in results_b), "kb_a results leaked into kb_b"

    def test_list_documents_isolation(self, pkb):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(pkb.add_text("Doc in alpha", kb_id="alpha"))
        loop.run_until_complete(pkb.add_text("Doc in beta", kb_id="beta"))

        docs_alpha = loop.run_until_complete(pkb.list_documents(kb_id="alpha"))
        docs_beta  = loop.run_until_complete(pkb.list_documents(kb_id="beta"))
        assert len(docs_alpha) == 1
        assert len(docs_beta)  == 1


# ═══════════════════════════════════════════════════════════════════════════════
# 5. reindex_document
# ═══════════════════════════════════════════════════════════════════════════════

class TestReindexDocument:
    def test_reindex_fills_missing_embeddings(self, pkb, tmp_path):
        """Chunks initially saved without embeddings should be embedded on reindex."""
        loop = asyncio.get_event_loop()
        store = pkb._store

        # Save a document with chunks that have no embeddings
        doc = KBDocument(doc_id="reindex_doc", kb_id="global", filename="r.txt",
                         source="/r.txt", status="error", created_at=time.time())
        loop.run_until_complete(store.save_document(doc))
        for i in range(3):
            loop.run_until_complete(store.save_chunk(KBChunk(
                chunk_id=f"rc_{i}", doc_id="reindex_doc", kb_id="global",
                chunk_index=i, text=f"reindex chunk {i}", embedding=None,
                created_at=time.time(),
            )))

        # Reindex should embed all 3 chunks
        result = loop.run_until_complete(pkb.reindex_document("reindex_doc"))
        assert result.status == "ready"
        assert result.chunk_count == 3

        # Verify embeddings are now stored
        stored_chunks = loop.run_until_complete(store.list_chunks("reindex_doc"))
        assert all(c.embedding is not None for c in stored_chunks)

    def test_reindex_missing_doc_raises(self, pkb):
        loop = asyncio.get_event_loop()
        with pytest.raises(ValueError, match="not found"):
            loop.run_until_complete(pkb.reindex_document("nonexistent_doc"))

    def test_reindex_embed_failure_partial(self, tmp_path):
        """If embed_fn fails for some chunks, reindex still completes with partial count."""
        loop = asyncio.get_event_loop()
        call_count = 0

        async def flaky_embed(text: str) -> list[float]:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("embed transient failure")
            return [0.1, 0.2, 0.3]

        db_path = str(tmp_path / "flaky.db")
        store = SQLiteKBStore(db_path=db_path)
        loop.run_until_complete(store.initialize())
        pkb2 = PersistentKnowledgeBase(store=store, embed_fn=flaky_embed, kb_id="global")
        loop.run_until_complete(pkb2.initialize())

        doc = KBDocument(doc_id="flaky_doc", kb_id="global", filename="f.txt",
                         source="/f.txt", status="error", created_at=time.time())
        loop.run_until_complete(store.save_document(doc))
        for i in range(3):
            loop.run_until_complete(store.save_chunk(KBChunk(
                chunk_id=f"fc_{i}", doc_id="flaky_doc", kb_id="global",
                chunk_index=i, text=f"flaky {i}", embedding=None, created_at=time.time(),
            )))

        result = loop.run_until_complete(pkb2.reindex_document("flaky_doc"))
        # 2 out of 3 should succeed (chunk 2 failed)
        assert result.status == "ready"
        assert result.chunk_count == 2


# ═══════════════════════════════════════════════════════════════════════════════
# 6. KBChunk / Chunk conversion helpers
# ═══════════════════════════════════════════════════════════════════════════════

class TestChunkConversion:
    def test_kb_chunk_to_chunk_roundtrip(self):
        kbc = KBChunk(
            chunk_id="cid", doc_id="did", kb_id="kb1",
            chunk_index=2, text="hello world",
            embedding=[0.1, 0.2], meta={"source": "test.txt"},
            created_at=1234567890.0,
        )
        chunk = _kb_chunk_to_chunk(kbc)
        assert chunk.id == "cid"
        assert chunk.doc_id == "did"
        assert chunk.text == "hello world"
        assert chunk.embedding == pytest.approx([0.1, 0.2])
        assert chunk.chunk_index == 2

    def test_chunk_to_kb_chunk(self):
        chunk = Chunk(
            id="cid2", doc_id="did2", text="foo bar",
            chunk_index=0, metadata={"source": "bar.txt"},
            embedding=[0.3, 0.4], score=0.9,
        )
        kbc = _chunk_to_kb_chunk(chunk, kb_id="kb2", doc_id="did2")
        assert kbc.chunk_id == "cid2"
        assert kbc.kb_id == "kb2"
        assert kbc.text == "foo bar"
        assert kbc.embedding == pytest.approx([0.3, 0.4])


# ═══════════════════════════════════════════════════════════════════════════════
# 7. doc_ids scoped filtering (SQLite path)
# ═══════════════════════════════════════════════════════════════════════════════

class TestDocIdsFiltering:
    """Test that PKB.query() respects the doc_ids filter in SQLite (no-vector-store) mode."""

    def test_filter_to_single_doc(self, pkb):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(pkb.add_text("Cats are fluffy animals that purr.", source="alpha.txt", kb_id="scope_kb"))
        loop.run_until_complete(pkb.add_text("Dogs are loyal companions for humans.", source="beta.txt", kb_id="scope_kb"))

        docs = loop.run_until_complete(pkb.list_documents(kb_id="scope_kb"))
        assert len(docs) == 2
        alpha_id = next(d.doc_id for d in docs if d.filename == "alpha.txt")
        beta_id  = next(d.doc_id for d in docs if d.filename == "beta.txt")

        results = loop.run_until_complete(pkb.query("cats fluffy", kb_id="scope_kb", doc_ids=[alpha_id]))
        returned_doc_ids = {c.doc_id for c in results}
        assert beta_id not in returned_doc_ids, "beta doc leaked into single-doc filter results"
        if results:
            assert alpha_id in returned_doc_ids

    def test_filter_to_multiple_docs(self, pkb):
        loop = asyncio.get_event_loop()
        for name in ["d1.txt", "d2.txt", "d3.txt"]:
            loop.run_until_complete(pkb.add_text(f"Information about {name} topic", source=name, kb_id="multi_kb"))

        docs = loop.run_until_complete(pkb.list_documents(kb_id="multi_kb"))
        ids_12  = [d.doc_id for d in docs if d.filename in ("d1.txt", "d2.txt")]
        id_3    = next(d.doc_id for d in docs if d.filename == "d3.txt")

        results = loop.run_until_complete(pkb.query("Information topic", kb_id="multi_kb", doc_ids=ids_12))
        assert id_3 not in {c.doc_id for c in results}, "d3 leaked into 2-doc filtered results"

    def test_none_doc_ids_returns_all(self, pkb):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(pkb.add_text("Text alpha content here.", source="t1.txt", kb_id="all_kb"))
        loop.run_until_complete(pkb.add_text("Text beta content here.", source="t2.txt", kb_id="all_kb"))

        results_none  = loop.run_until_complete(pkb.query("Text content", kb_id="all_kb", doc_ids=None))
        results_empty = loop.run_until_complete(pkb.query("Text content", kb_id="all_kb", doc_ids=[]))
        assert len(results_none) > 0
        assert len(results_empty) > 0
        # Both should return the same set of doc_ids (all docs)
        assert {c.doc_id for c in results_none} == {c.doc_id for c in results_empty}

    def test_nonexistent_doc_id_returns_empty(self, pkb):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(pkb.add_text("Real document content here.", source="real.txt", kb_id="ghost_kb"))
        results = loop.run_until_complete(
            pkb.query("document", kb_id="ghost_kb", doc_ids=["nonexistent_doc_id"])
        )
        assert results == [], f"Expected empty results for nonexistent doc_id, got {results}"

    def test_doc_ids_respects_kb_isolation(self, pkb):
        """doc_ids from a different kb should not produce results in another kb."""
        loop = asyncio.get_event_loop()
        doc_a = loop.run_until_complete(pkb.add_text("Content in kb_x.", source="a.txt", kb_id="kb_x"))
        loop.run_until_complete(pkb.add_text("Content in kb_y.", source="b.txt", kb_id="kb_y"))

        # Pass alpha's doc_id but query kb_y — should get nothing
        results = loop.run_until_complete(
            pkb.query("Content", kb_id="kb_y", doc_ids=[doc_a.doc_id])
        )
        assert results == [], "Cross-kb doc_id should not return results"


# ═══════════════════════════════════════════════════════════════════════════════
# 8. on_progress callback
# ═══════════════════════════════════════════════════════════════════════════════

class TestProgressCallback:
    def test_progress_fires_for_each_chunk(self, pkb):
        loop = asyncio.get_event_loop()
        progress_calls: list[tuple[int, int]] = []

        def on_progress(done: int, total: int) -> None:
            progress_calls.append((done, total))

        # Long enough to produce multiple chunks
        long_text = " ".join([f"Sentence number {i} about interesting topics for testing purposes." for i in range(60)])
        doc = loop.run_until_complete(pkb.add_text(long_text, source="long.txt", on_progress=on_progress))

        assert doc.status == "ready"
        assert len(progress_calls) >= 1, "on_progress was never called"
        # Last call: done == total
        last_done, last_total = progress_calls[-1]
        assert last_done == last_total, f"Final call should have done==total, got {last_done}/{last_total}"
        # done values should be monotonically non-decreasing
        dones = [d for d, _ in progress_calls]
        assert dones == sorted(dones), f"Progress done values not monotonic: {dones}"
        # total should be consistent across all calls
        totals = [t for _, t in progress_calls]
        assert len(set(totals)) == 1, f"total changed across progress calls: {totals}"

    def test_no_progress_callback_still_succeeds(self, pkb):
        loop = asyncio.get_event_loop()
        doc = loop.run_until_complete(pkb.add_text("Short text with no callback.", source="short.txt"))
        assert doc.status == "ready"

    def test_progress_receives_correct_range(self, pkb):
        loop = asyncio.get_event_loop()
        calls: list[tuple[int, int]] = []

        loop.run_until_complete(pkb.add_text(
            "Simple one or two chunk text content here for progress testing.",
            source="simple.txt",
            on_progress=lambda d, t: calls.append((d, t)),
        ))
        # All done values must be 1..total
        for done, total in calls:
            assert 1 <= done <= total, f"Invalid progress: done={done}, total={total}"


# ═══════════════════════════════════════════════════════════════════════════════
# 9. VectorStoreBase abstract interface
# ═══════════════════════════════════════════════════════════════════════════════

class TestVectorStoreBase:
    def test_cannot_instantiate_abstract_class(self):
        from rag.vector_store_base import VectorStoreBase
        with pytest.raises(TypeError, match="abstract"):
            VectorStoreBase()  # type: ignore[abstract]

    def test_partial_implementation_raises(self):
        from rag.vector_store_base import VectorStoreBase

        class PartialStore(VectorStoreBase):
            async def initialize(self): pass
            async def upsert_chunks(self, chunks): pass
            # All other abstract methods missing

        with pytest.raises(TypeError):
            PartialStore()  # type: ignore[abstract]

    def test_full_implementation_is_valid(self):
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
        assert isinstance(store, VectorStoreBase)

    def test_hybrid_search_signature_includes_doc_ids(self):
        """Ensure doc_ids parameter is present in the abstract method signature."""
        import inspect
        from rag.vector_store_base import VectorStoreBase
        sig = inspect.signature(VectorStoreBase.hybrid_search)
        assert "doc_ids" in sig.parameters, "doc_ids param missing from VectorStoreBase.hybrid_search"
        assert sig.parameters["doc_ids"].default is None
