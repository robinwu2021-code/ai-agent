"""
tests/test_kb_api.py — Knowledge Base API integration tests

Run full suite (requires live server):
    pytest tests/test_kb_api.py -v --run-live

Run mock-only tests (no server needed):
    pytest tests/test_kb_api.py -v

The --run-live flag enables tests that actually call the running server
(default: http://localhost:8000).  Mock tests use httpx.AsyncClient with
the FastAPI app directly (no I/O, no LLM).
"""
from __future__ import annotations

import json
import time
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

# Skip entire module if fastapi is not installed (minimal test environments)
fastapi = pytest.importorskip(
    "fastapi",
    reason="fastapi not installed — install project deps (pip install -e .[dev]) to run API tests",
)

# ─── fixtures ─────────────────────────────────────────────────────────────────
# NOTE: --run-live flag and live-marker skip logic live in conftest.py

@pytest.fixture(scope="module")
def live_base():
    return "http://localhost:8000"


# ═══════════════════════════════════════════════════════════════════════════════
# Mock-based tests (no server, no LLM)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def mock_pkb():
    """A fully mocked PersistentKnowledgeBase."""
    from rag.store import KBDocument, KBChunk
    pkb = MagicMock()

    doc = KBDocument(doc_id="mock_doc", kb_id="global", filename="mock.txt",
                     source="inline", status="ready", char_count=100,
                     chunk_count=2, created_at=time.time())
    chunk = KBChunk(chunk_id="mock_chunk", doc_id="mock_doc", kb_id="global",
                    chunk_index=0, text="This is mock content about topic X.",
                    meta={"source": "mock.txt", "filename": "mock.txt", "kb_id": "global"},
                    created_at=time.time())

    pkb.add_text = AsyncMock(return_value=doc)
    pkb.add_file = AsyncMock(return_value=doc)
    pkb.list_documents = AsyncMock(return_value=[doc])
    pkb.query = AsyncMock(return_value=[chunk])
    pkb.delete_document = AsyncMock()
    pkb.reindex_document = AsyncMock(return_value=doc)
    pkb.get_stats = AsyncMock(return_value={
        "documents": 1, "chunks": 2, "ready_documents": 1, "total_chars": 100
    })
    pkb.format_context = MagicMock(return_value="## 相关知识库内容\n\n[1] 来源: mock.txt\nThis is mock content.")
    pkb._store = MagicMock()
    pkb._store.get_document = AsyncMock(return_value=doc)
    pkb._store.list_chunks = AsyncMock(return_value=[chunk])
    pkb._store.list_chunks_by_kb = AsyncMock(return_value=[chunk])
    pkb.list_documents.return_value = [doc]
    # _vector_store=None → kb_build_graph falls through to SQLite path
    pkb._vector_store = None
    return pkb


@pytest.fixture
def mock_kg_store():
    """A fully mocked SQLiteKGStore."""
    from rag.graph.models import Node, Edge, NodeType, SubGraph
    store = MagicMock()

    node = Node(id="n_001", kb_id="global", name="TestEntity",
                node_type=NodeType.ORG, description="A test org",
                aliases=[], doc_ids=["doc_001"], degree=2, created_at=time.time())
    node2 = Node(id="n_002", kb_id="global", name="RelatedEntity",
                 node_type=NodeType.CONCEPT, description="A concept",
                 aliases=[], doc_ids=[], degree=1, created_at=time.time())
    edge = Edge(id="e_001", kb_id="global", src_id="n_001", dst_id="n_002",
                relation="related_to", weight=0.9, context="some evidence",
                created_at=time.time())

    store.get_stats = AsyncMock(return_value={"nodes": 2, "edges": 1, "communities": 0})
    store.list_nodes = AsyncMock(return_value=[node, node2])
    store.get_node = AsyncMock(return_value=node)
    store.get_node_by_name = AsyncMock(return_value=None)   # default: entity not found → will be created
    store.upsert_node = AsyncMock()
    store.upsert_edge = AsyncMock()
    store.delete_node = AsyncMock()
    store.delete_edge = AsyncMock()
    store.delete_by_doc = AsyncMock()
    store.search_nodes_by_text = AsyncMock(return_value=[node])
    store.search_nodes_by_embedding = AsyncMock(return_value=[node])
    store.list_edges = AsyncMock(return_value=[edge])
    store.get_full_graph = AsyncMock(return_value=([node, node2], [edge]))
    store.get_neighbors = AsyncMock(return_value=([node2], [edge]))

    # Expose fixtures for per-test override
    store._node  = node
    store._node2 = node2
    store._edge  = edge
    return store


@pytest.fixture
def mock_graph_builder():
    """A mocked GraphBuilder."""
    gb = MagicMock()
    gb.start_build_job       = MagicMock(return_value="job_kg_001")
    gb.start_build_text_job  = MagicMock(return_value="job_txt_001")
    gb.get_job_status        = MagicMock(return_value={
        "status": "done", "progress": 1.0,
        "result": {"nodes_created": 2, "edges_created": 1, "triples_found": 3},
        "kb_id": "global", "doc_id": "doc_001",
        "created_at": 0.0, "finished_at": 1.0, "error": None,
    })
    gb.list_jobs = MagicMock(return_value=[{
        "status": "done", "kb_id": "global", "doc_id": "doc_001",
        "progress": 1.0, "created_at": 0.0, "finished_at": 1.0, "error": None,
    }])
    gb.rebuild_communities = AsyncMock(return_value={"communities_created": 1, "communities_skipped": 0})
    return gb


@pytest.fixture
def mock_container(mock_pkb):
    """A minimal mock AgentContainer with knowledge_base attached."""
    container = MagicMock()
    container.knowledge_base = mock_pkb

    # Mock LLM router for /kb/ask
    mock_resp = MagicMock()
    mock_resp.content = "这是基于知识库的模拟回答。"
    router = MagicMock()
    router.chat = AsyncMock(return_value=mock_resp)
    router.embed = AsyncMock(return_value=[0.1] * 8)
    container._effective_router = MagicMock(return_value=router)
    container.kg_store = None
    # graph_builder is intentionally left as a MagicMock auto-attribute so
    # TestKBBuildGraph tests that set it to None continue to work correctly.
    return container


@pytest.fixture
def kg_app_client(mock_pkb, mock_kg_store, mock_graph_builder):
    """FastAPI TestClient pre-wired with KG store + GraphBuilder."""
    import sys
    from unittest.mock import MagicMock as _MM
    for _pkg in ("uvicorn",):
        if _pkg not in sys.modules:
            sys.modules[_pkg] = _MM()
    import server as srv

    container = MagicMock()
    container.knowledge_base = mock_pkb

    mock_resp = MagicMock()
    mock_resp.content = "模拟KG回答"
    router = MagicMock()
    router.chat = AsyncMock(return_value=mock_resp)
    router.embed = AsyncMock(return_value=[0.1] * 8)
    container._effective_router = MagicMock(return_value=router)
    container.kg_store      = mock_kg_store
    container.graph_builder = mock_graph_builder

    srv._container = container
    from fastapi.testclient import TestClient
    return TestClient(srv.app)


@pytest.fixture
def app_client(mock_container):
    """FastAPI TestClient with mocked container."""
    import sys
    from unittest.mock import MagicMock as _MM
    # Stub out runtime-only packages that aren't needed in test environments
    for _pkg in ("uvicorn",):
        if _pkg not in sys.modules:
            sys.modules[_pkg] = _MM()
    import server as srv
    srv._container = mock_container
    from fastapi.testclient import TestClient
    return TestClient(srv.app)


class TestKBDocumentsText:
    def test_add_text_success(self, app_client):
        res = app_client.post("/kb/documents/text", json={
            "text": "This is test content for the knowledge base.",
            "filename": "test.txt",
            "kb_id": "global",
        })
        assert res.status_code == 200
        data = res.json()
        assert data["doc_id"] == "mock_doc"
        assert data["status"] == "ready"

    def test_add_text_with_content_alias(self, app_client):
        """Frontend sends 'content' instead of 'text' — should still work."""
        res = app_client.post("/kb/documents/text", json={
            "content": "Some content via alias field.",
            "title": "alias_test.txt",
            "kb_id": "global",
        })
        assert res.status_code == 200

    def test_add_text_empty_body_rejected(self, app_client):
        res = app_client.post("/kb/documents/text", json={
            "text": "   ",
            "kb_id": "global",
        })
        assert res.status_code == 400

    def test_add_text_missing_text_and_content_rejected(self, app_client):
        res = app_client.post("/kb/documents/text", json={"kb_id": "global"})
        assert res.status_code == 400


class TestKBListDocuments:
    def test_list_returns_documents(self, app_client):
        res = app_client.get("/kb/documents?kb_id=global")
        assert res.status_code == 200
        data = res.json()
        assert "documents" in data
        assert isinstance(data["documents"], list)
        assert data["total"] >= 0

    def test_list_default_kb(self, app_client):
        res = app_client.get("/kb/documents")
        assert res.status_code == 200


class TestKBAsk:
    def test_ask_returns_answer(self, app_client):
        res = app_client.post("/kb/ask", json={
            "query": "What is the mock content?",
            "kb_id": "global",
            "top_k": 3,
        })
        assert res.status_code == 200
        data = res.json()
        assert "answer" in data
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0

    def test_ask_returns_citations(self, app_client):
        res = app_client.post("/kb/ask", json={
            "query": "Tell me about topic X",
            "kb_id": "global",
        })
        assert res.status_code == 200
        data = res.json()
        assert "citations" in data
        assert isinstance(data["citations"], list)
        if data["citations"]:
            cit = data["citations"][0]
            assert "index" in cit
            assert "filename" in cit
            assert "text_preview" in cit

    def test_ask_returns_chunks(self, app_client):
        res = app_client.post("/kb/ask", json={"query": "topic", "kb_id": "global"})
        assert res.status_code == 200
        assert "chunks" in res.json()

    def test_ask_with_history(self, app_client):
        res = app_client.post("/kb/ask", json={
            "query": "Follow-up question",
            "kb_id": "global",
            "history": [
                {"role": "user", "content": "Prior question"},
                {"role": "assistant", "content": "Prior answer"},
            ],
        })
        assert res.status_code == 200
        assert "answer" in res.json()


class TestKBAskStream:
    def test_stream_returns_sse(self, app_client):
        """The /kb/ask/stream endpoint must return text/event-stream."""
        res = app_client.post("/kb/ask/stream", json={
            "query": "What is the content?",
            "kb_id": "global",
        })
        assert res.status_code == 200
        assert "text/event-stream" in res.headers.get("content-type", "")

    def test_stream_contains_context_event(self, app_client):
        res = app_client.post("/kb/ask/stream", json={
            "query": "query",
            "kb_id": "global",
        })
        body = res.text
        events = [line[6:] for line in body.splitlines() if line.startswith("data: ") and line[6:] != "[DONE]"]
        parsed = [json.loads(e) for e in events if e]
        types = [e.get("type") for e in parsed]
        assert "context" in types, f"No 'context' event found. Got: {types}"

    def test_stream_contains_delta_events(self, app_client):
        res = app_client.post("/kb/ask/stream", json={"query": "hello", "kb_id": "global"})
        body = res.text
        events = [line[6:] for line in body.splitlines() if line.startswith("data: ") and line[6:] != "[DONE]"]
        parsed = [json.loads(e) for e in events if e]
        types = [e.get("type") for e in parsed]
        assert "delta" in types, f"No 'delta' events found. Got: {types}"

    def test_stream_ends_with_done(self, app_client):
        res = app_client.post("/kb/ask/stream", json={"query": "end", "kb_id": "global"})
        assert "[DONE]" in res.text or '"type": "done"' in res.text


class TestKBReindex:
    def test_reindex_existing_doc(self, app_client):
        res = app_client.post("/kb/documents/mock_doc/reindex")
        assert res.status_code == 200
        data = res.json()
        assert data["doc_id"] == "mock_doc"

    def test_reindex_missing_doc_returns_404(self, app_client, mock_pkb):
        mock_pkb.reindex_document.side_effect = ValueError("Document 'missing' not found")
        res = app_client.post("/kb/documents/missing/reindex")
        assert res.status_code == 404


class TestKBDelete:
    def test_delete_document(self, app_client):
        res = app_client.delete("/kb/documents/mock_doc")
        assert res.status_code == 200
        assert res.json()["ok"] is True


class TestKBStats:
    def test_stats_shape(self, app_client):
        res = app_client.get("/kb/stats?kb_id=global")
        assert res.status_code == 200
        data = res.json()
        assert "documents" in data or "doc_count" in data


# ═══════════════════════════════════════════════════════════════════════════════
# New: doc_ids scoped Q&A
# ═══════════════════════════════════════════════════════════════════════════════

class TestKBAskDocIds:
    def test_ask_with_empty_doc_ids_works(self, app_client, mock_pkb):
        """Empty doc_ids list → same as no filter (query all)."""
        res = app_client.post("/kb/ask", json={
            "query": "What is the content?",
            "kb_id": "global",
            "doc_ids": [],
        })
        assert res.status_code == 200
        # Verify pkb.query was called with doc_ids=None (empty list converted to None)
        call_kwargs = mock_pkb.query.call_args
        assert call_kwargs is not None
        passed_doc_ids = call_kwargs.kwargs.get("doc_ids") or (call_kwargs.args[3] if len(call_kwargs.args) > 3 else None)
        # empty list should be coerced to None
        assert passed_doc_ids is None or passed_doc_ids == []

    def test_ask_with_doc_ids_passes_filter(self, app_client, mock_pkb):
        """Non-empty doc_ids should be forwarded to pkb.query."""
        res = app_client.post("/kb/ask", json={
            "query": "Tell me about the document",
            "kb_id": "global",
            "doc_ids": ["doc_abc", "doc_def"],
        })
        assert res.status_code == 200
        call_kwargs = mock_pkb.query.call_args
        assert call_kwargs is not None
        # Check doc_ids was passed through (keyword arg or positional)
        passed = call_kwargs.kwargs.get("doc_ids")
        assert passed == ["doc_abc", "doc_def"]

    def test_stream_with_doc_ids(self, app_client):
        """SSE stream should accept doc_ids without error."""
        res = app_client.post("/kb/ask/stream", json={
            "query": "scoped query",
            "kb_id": "global",
            "doc_ids": ["mock_doc"],
        })
        assert res.status_code == 200
        assert "text/event-stream" in res.headers.get("content-type", "")
        # Should still contain a context event
        events = [line[6:] for line in res.text.splitlines()
                  if line.startswith("data: ") and line[6:] != "[DONE]"]
        parsed = [json.loads(e) for e in events if e]
        types = [e.get("type") for e in parsed]
        assert "context" in types


# ═══════════════════════════════════════════════════════════════════════════════
# New: /kb/build-graph endpoint
# ═══════════════════════════════════════════════════════════════════════════════

class TestKBBuildGraph:
    def test_build_graph_without_graph_builder_returns_501(self, app_client, mock_container):
        """When graph_builder is None/absent on container, expect 501."""
        mock_container.graph_builder = None
        res = app_client.post("/kb/build-graph/mock_doc?kb_id=global")
        assert res.status_code == 501
        assert "GraphBuilder" in res.json().get("detail", "")

    def test_build_graph_doc_not_found_returns_404(self, app_client, mock_container, mock_pkb):
        """When list_chunks returns empty, expect 404."""
        # Give container a mock graph builder
        mock_gb = MagicMock()
        mock_gb.start_build_job = MagicMock(return_value="job_xyz")
        mock_container.graph_builder = mock_gb
        # Make list_chunks return empty
        mock_pkb._store.list_chunks = AsyncMock(return_value=[])
        res = app_client.post("/kb/build-graph/nonexistent_doc?kb_id=global")
        assert res.status_code == 404

    def test_build_graph_success(self, app_client, mock_container, mock_pkb):
        """Happy path: returns job_id, doc_id, chunks count."""
        from rag.store import KBChunk
        import time as _t
        mock_gb = MagicMock()
        mock_gb.start_build_job = MagicMock(return_value="job_abc123")
        mock_container.graph_builder = mock_gb

        chunks = [
            KBChunk(chunk_id=f"c{i}", doc_id="mock_doc", kb_id="global",
                    chunk_index=i, text=f"chunk text {i}", created_at=_t.time())
            for i in range(3)
        ]
        mock_pkb._store.list_chunks = AsyncMock(return_value=chunks)

        res = app_client.post("/kb/build-graph/mock_doc?kb_id=global")
        assert res.status_code == 200
        data = res.json()
        assert data["job_id"] == "job_abc123"
        assert data["doc_id"] == "mock_doc"
        assert data["chunks"] == 3

    def test_build_graph_calls_start_build_job_with_correct_args(self, app_client, mock_container, mock_pkb):
        """Verify start_build_job receives properly shaped chunk dicts."""
        from rag.store import KBChunk
        import time as _t
        mock_gb = MagicMock()
        mock_gb.start_build_job = MagicMock(return_value="job_verify")
        mock_container.graph_builder = mock_gb

        chunk = KBChunk(chunk_id="ck1", doc_id="doc_verify", kb_id="global",
                        chunk_index=0, text="Hello world text", created_at=_t.time())
        mock_pkb._store.list_chunks = AsyncMock(return_value=[chunk])

        app_client.post("/kb/build-graph/doc_verify?kb_id=global")

        mock_gb.start_build_job.assert_called_once()
        call_kwargs = mock_gb.start_build_job.call_args.kwargs
        assert call_kwargs["doc_id"] == "doc_verify"
        assert call_kwargs["kb_id"] == "global"
        assert len(call_kwargs["chunks"]) == 1
        chunk_dict = call_kwargs["chunks"][0]
        assert chunk_dict["text"] == "Hello world text"
        assert chunk_dict["chunk_id"] == "ck1"


# ═══════════════════════════════════════════════════════════════════════════════
# New: /kb/documents/upload/stream SSE endpoint
# ═══════════════════════════════════════════════════════════════════════════════

class TestKBUploadStream:
    def test_unsupported_file_type_returns_400(self, app_client):
        res = app_client.post(
            "/kb/documents/upload/stream",
            files={"file": ("test.exe", b"binary data", "application/octet-stream")},
            data={"kb_id": "global"},
        )
        assert res.status_code == 400
        assert "不支持" in res.json().get("detail", "")

    def test_upload_stream_returns_sse_content_type(self, app_client):
        res = app_client.post(
            "/kb/documents/upload/stream",
            files={"file": ("test.txt", b"Hello world content for testing.", "text/plain")},
            data={"kb_id": "global"},
        )
        assert res.status_code == 200
        assert "text/event-stream" in res.headers.get("content-type", "")

    def test_upload_stream_contains_parsing_event(self, app_client):
        res = app_client.post(
            "/kb/documents/upload/stream",
            files={"file": ("sample.txt", b"Sample document text for indexing.", "text/plain")},
            data={"kb_id": "global"},
        )
        assert res.status_code == 200
        events = [line[6:] for line in res.text.splitlines()
                  if line.startswith("data: ")]
        parsed = [json.loads(e) for e in events if e]
        types = [e.get("type") for e in parsed]
        assert "progress" in types, f"No 'progress' event found. Got types: {types}"
        # First progress event must have step='parsing'
        first_progress = next(e for e in parsed if e.get("type") == "progress")
        assert first_progress.get("step") == "parsing"

    def test_upload_stream_contains_done_event(self, app_client):
        res = app_client.post(
            "/kb/documents/upload/stream",
            files={"file": ("done_test.txt", b"Content for done event test.", "text/plain")},
            data={"kb_id": "global"},
        )
        events = [line[6:] for line in res.text.splitlines()
                  if line.startswith("data: ")]
        parsed = [json.loads(e) for e in events if e]
        types = [e.get("type") for e in parsed]
        assert "done" in types, f"No 'done' event found. Got types: {types}"
        done_event = next(e for e in parsed if e.get("type") == "done")
        assert "doc_id" in done_event
        assert "chunk_count" in done_event
        assert "status" in done_event

    def test_upload_stream_md_file_accepted(self, app_client):
        res = app_client.post(
            "/kb/documents/upload/stream",
            files={"file": ("readme.md", b"# Title\n\nMarkdown content here.", "text/markdown")},
            data={"kb_id": "global"},
        )
        assert res.status_code == 200

    def test_upload_stream_progress_has_percentage(self, app_client):
        content = b"This is a longer document. " * 50
        res = app_client.post(
            "/kb/documents/upload/stream",
            files={"file": ("long.txt", content, "text/plain")},
            data={"kb_id": "global"},
        )
        events = [line[6:] for line in res.text.splitlines() if line.startswith("data: ")]
        parsed = [json.loads(e) for e in events if e]
        progress_events = [e for e in parsed if e.get("type") == "progress"]
        assert len(progress_events) >= 1
        for pe in progress_events:
            if "progress" in pe:
                assert 0 <= pe["progress"] <= 100, f"Progress out of range: {pe['progress']}"


# ═══════════════════════════════════════════════════════════════════════════════
# Live tests (--run-live flag required)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.live
class TestLiveKBApi:
    """Integration tests against a running server. Pass --run-live to enable."""

    import requests as _requests  # noqa: F401

    def test_health(self, live_base):
        import requests
        res = requests.get(f"{live_base}/health", timeout=5)
        assert res.status_code == 200

    def test_add_text_and_ask(self, live_base):
        import requests
        # 1. Add text
        payload = {
            "text": "Claude AI is developed by Anthropic. It is a helpful assistant.",
            "filename": "live_test.txt",
            "kb_id": "live_test_kb",
        }
        add_res = requests.post(f"{live_base}/kb/documents/text", json=payload, timeout=30)
        assert add_res.status_code == 200
        doc_id = add_res.json()["doc_id"]

        # 2. Ask a question
        ask_res = requests.post(f"{live_base}/kb/ask", json={
            "query": "Who developed Claude AI?",
            "kb_id": "live_test_kb",
            "top_k": 3,
        }, timeout=60)
        assert ask_res.status_code == 200
        data = ask_res.json()
        assert "answer" in data
        assert len(data["answer"]) > 0

        # 3. Cleanup
        requests.delete(f"{live_base}/kb/documents/{doc_id}", timeout=10)

    def test_stream_ask(self, live_base):
        import requests
        res = requests.post(f"{live_base}/kb/ask/stream", json={
            "query": "test stream",
            "kb_id": "live_test_kb_stream",
        }, stream=True, timeout=60)
        assert res.status_code == 200
        events = []
        for line in res.iter_lines():
            if line and line.startswith(b"data: "):
                raw = line[6:]
                if raw != b"[DONE]":
                    events.append(json.loads(raw))
        types = [e.get("type") for e in events]
        assert "context" in types
        assert "done" in types

    def test_stats(self, live_base):
        import requests
        res = requests.get(f"{live_base}/kb/stats", timeout=5)
        assert res.status_code == 200
        data = res.json()
        assert "documents" in data


# ═══════════════════════════════════════════════════════════════════════════════
# Missing KB endpoint coverage
# ═══════════════════════════════════════════════════════════════════════════════

class TestKBGetDocument:
    def test_get_existing_document(self, app_client):
        res = app_client.get("/kb/documents/mock_doc")
        assert res.status_code == 200
        data = res.json()
        assert data["doc_id"] == "mock_doc"
        assert "status" in data
        assert "chunk_count" in data

    def test_get_missing_document_returns_404(self, app_client, mock_pkb):
        mock_pkb._store.get_document = AsyncMock(return_value=None)
        res = app_client.get("/kb/documents/nonexistent_xyz")
        assert res.status_code == 404

    def test_get_document_has_filename(self, app_client):
        res = app_client.get("/kb/documents/mock_doc")
        assert res.status_code == 200
        assert res.json()["filename"] == "mock.txt"


class TestKBChunks:
    def test_list_chunks_for_doc(self, app_client):
        res = app_client.get("/kb/chunks?kb_id=global&doc_id=mock_doc")
        assert res.status_code == 200
        data = res.json()
        assert "chunks" in data
        assert isinstance(data["chunks"], list)
        assert "total" in data

    def test_list_chunks_all_kb(self, app_client):
        res = app_client.get("/kb/chunks?kb_id=global")
        assert res.status_code == 200
        data = res.json()
        assert "chunks" in data
        assert isinstance(data["total"], int)

    def test_list_chunks_no_embedding_in_response(self, app_client):
        """embedding field must be stripped from output (size + privacy)."""
        res = app_client.get("/kb/chunks?kb_id=global&doc_id=mock_doc")
        assert res.status_code == 200
        for chunk in res.json()["chunks"]:
            assert "embedding" not in chunk

    def test_list_chunks_limit_respected(self, app_client):
        res = app_client.get("/kb/chunks?kb_id=global&limit=10")
        assert res.status_code == 200


class TestKBUpload:
    def test_upload_txt_file(self, app_client):
        res = app_client.post(
            "/kb/documents/upload",
            files={"file": ("upload_test.txt", b"Upload test content.", "text/plain")},
            data={"kb_id": "global"},
        )
        assert res.status_code == 200
        data = res.json()
        assert "doc_id" in data

    def test_upload_unsupported_type_returns_400(self, app_client):
        res = app_client.post(
            "/kb/documents/upload",
            files={"file": ("bad.exe", b"\x00\x01\x02", "application/octet-stream")},
            data={"kb_id": "global"},
        )
        assert res.status_code == 400


# ═══════════════════════════════════════════════════════════════════════════════
# TestKBBuildGraph — vector store path (new after fix)
# ═══════════════════════════════════════════════════════════════════════════════

class TestKBBuildGraphVectorStore:
    def test_build_graph_uses_vector_store_when_active(self, app_client, mock_container, mock_pkb):
        """When _vector_store is set, chunks should come from it (not SQLite)."""
        mock_gb = MagicMock()
        mock_gb.start_build_job = MagicMock(return_value="job_vs_001")
        mock_container.graph_builder = mock_gb

        # Attach a mock vector store that returns chunks as dicts
        vs = MagicMock()
        vs.list_chunks = AsyncMock(return_value=[
            {"chunk_id": "vs_c1", "doc_id": "vs_doc", "text": "Vector store chunk 1"},
            {"chunk_id": "vs_c2", "doc_id": "vs_doc", "text": "Vector store chunk 2"},
        ])
        mock_pkb._vector_store = vs
        mock_pkb._store.list_chunks = AsyncMock(return_value=[])  # SQLite is empty

        res = app_client.post("/kb/build-graph/vs_doc?kb_id=global")
        assert res.status_code == 200
        data = res.json()
        assert data["job_id"] == "job_vs_001"
        assert data["chunks"] == 2

        # Restore for other tests
        mock_pkb._vector_store = None

    def test_build_graph_falls_back_to_sqlite_on_vector_store_error(
        self, app_client, mock_container, mock_pkb
    ):
        """If vector store raises, fall through to SQLite."""
        from rag.store import KBChunk
        mock_gb = MagicMock()
        mock_gb.start_build_job = MagicMock(return_value="job_fallback")
        mock_container.graph_builder = mock_gb

        vs = MagicMock()
        vs.list_chunks = AsyncMock(side_effect=RuntimeError("connection timeout"))
        mock_pkb._vector_store = vs

        sqlite_chunks = [KBChunk(chunk_id="sq1", doc_id="doc_fb", kb_id="global",
                                 chunk_index=0, text="SQLite fallback text",
                                 created_at=time.time())]
        mock_pkb._store.list_chunks = AsyncMock(return_value=sqlite_chunks)

        res = app_client.post("/kb/build-graph/doc_fb?kb_id=global")
        assert res.status_code == 200
        assert res.json()["chunks"] == 1

        # Restore
        mock_pkb._vector_store = None
        mock_pkb._store.list_chunks = AsyncMock(return_value=sqlite_chunks)

    def test_build_graph_returns_404_when_both_stores_empty(
        self, app_client, mock_container, mock_pkb
    ):
        """Both vector store and SQLite empty → 404."""
        mock_gb = MagicMock()
        mock_gb.start_build_job = MagicMock(return_value="job_404")
        mock_container.graph_builder = mock_gb

        vs = MagicMock()
        vs.list_chunks = AsyncMock(return_value=[])
        mock_pkb._vector_store = vs
        mock_pkb._store.list_chunks = AsyncMock(return_value=[])

        res = app_client.post("/kb/build-graph/empty_doc?kb_id=global")
        assert res.status_code == 404

        mock_pkb._vector_store = None


# ═══════════════════════════════════════════════════════════════════════════════
# KG: /kg/build/* endpoints
# ═══════════════════════════════════════════════════════════════════════════════

class TestKGBuildText:
    def test_build_from_text_success(self, kg_app_client):
        res = kg_app_client.post("/kg/build/text", json={
            "text": "Anthropic was founded by Dario Amodei. Claude is made by Anthropic.",
            "kb_id": "global",
        })
        assert res.status_code == 200
        data = res.json()
        assert "job_id"  in data
        assert "doc_id"  in data
        assert "kb_id"   in data
        assert data["job_id"] == "job_txt_001"

    def test_build_from_text_auto_generates_doc_id(self, kg_app_client):
        """When doc_id is omitted, server generates one starting with 'doc_'."""
        res = kg_app_client.post("/kg/build/text", json={
            "text": "Some entity text here.",
            "kb_id": "global",
        })
        assert res.status_code == 200
        doc_id = res.json()["doc_id"]
        assert doc_id.startswith("doc_")

    def test_build_from_text_with_explicit_doc_id(self, kg_app_client):
        res = kg_app_client.post("/kg/build/text", json={
            "text": "Entity A is related to Entity B.",
            "kb_id": "global",
            "doc_id": "my_custom_doc",
        })
        assert res.status_code == 200
        assert res.json()["doc_id"] == "my_custom_doc"

    def test_build_from_text_missing_text_returns_422(self, kg_app_client):
        res = kg_app_client.post("/kg/build/text", json={"kb_id": "global"})
        assert res.status_code == 422

    def test_build_text_no_graph_builder_returns_501(self, app_client, mock_container):
        mock_container.graph_builder = None
        res = app_client.post("/kg/build/text", json={"text": "some text", "kb_id": "global"})
        assert res.status_code == 501


class TestKGBuildFile:
    def test_build_from_txt_file(self, kg_app_client):
        res = kg_app_client.post(
            "/kg/build/file",
            files={"file": ("kg_test.txt", b"Alice works at OpenAI. Bob founded DeepMind.", "text/plain")},
            data={"kb_id": "global"},
        )
        assert res.status_code == 200
        data = res.json()
        assert "job_id"    in data
        assert "doc_id"    in data
        assert "kb_id"     in data
        assert "chunks"    in data
        assert "filename"  in data
        assert data["filename"] == "kg_test.txt"

    def test_build_from_md_file(self, kg_app_client):
        content = b"# Companies\n\nMicrosoft acquired GitHub in 2018.\n"
        res = kg_app_client.post(
            "/kg/build/file",
            files={"file": ("notes.md", content, "text/markdown")},
            data={"kb_id": "global"},
        )
        assert res.status_code == 200

    def test_build_file_no_graph_builder_returns_501(self, app_client, mock_container):
        mock_container.graph_builder = None
        res = app_client.post(
            "/kg/build/file",
            files={"file": ("t.txt", b"text", "text/plain")},
            data={"kb_id": "global"},
        )
        assert res.status_code == 501


class TestKGBuildStatus:
    def test_get_job_status_done(self, kg_app_client):
        res = kg_app_client.get("/kg/build/status/job_kg_001")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "done"
        assert data["progress"] == 1.0

    def test_get_job_status_has_result(self, kg_app_client):
        res = kg_app_client.get("/kg/build/status/job_kg_001")
        assert res.status_code == 200
        result = res.json().get("result", {})
        assert "nodes_created"  in result
        assert "edges_created"  in result
        assert "triples_found"  in result

    def test_get_job_status_not_found_returns_404(self, kg_app_client, mock_graph_builder):
        mock_graph_builder.get_job_status = MagicMock(return_value=None)
        res = kg_app_client.get("/kg/build/status/nonexistent_job_xyz")
        assert res.status_code == 404

    def test_get_job_status_no_builder_returns_501(self, app_client, mock_container):
        mock_container.graph_builder = None
        res = app_client.get("/kg/build/status/any_job")
        assert res.status_code == 501


class TestKGBuildJobs:
    def test_list_jobs_returns_list(self, kg_app_client):
        res = kg_app_client.get("/kg/build/jobs")
        assert res.status_code == 200
        data = res.json()
        assert "jobs" in data
        assert isinstance(data["jobs"], list)

    def test_list_jobs_shape(self, kg_app_client):
        res = kg_app_client.get("/kg/build/jobs")
        jobs = res.json()["jobs"]
        assert len(jobs) >= 1
        job = jobs[0]
        assert "status"  in job
        assert "kb_id"   in job

    def test_list_jobs_no_builder_returns_501(self, app_client, mock_container):
        mock_container.graph_builder = None
        res = app_client.get("/kg/build/jobs")
        assert res.status_code == 501


# ═══════════════════════════════════════════════════════════════════════════════
# KG: communities rebuild
# ═══════════════════════════════════════════════════════════════════════════════

class TestKGCommunitiesRebuild:
    def test_rebuild_starts_background_task(self, kg_app_client):
        res = kg_app_client.post("/kg/communities/rebuild?kb_id=global")
        assert res.status_code == 200
        data = res.json()
        assert data["ok"] is True
        assert data["kb_id"] == "global"
        assert "message" in data

    def test_rebuild_custom_kb_id(self, kg_app_client):
        res = kg_app_client.post("/kg/communities/rebuild?kb_id=my_kb")
        assert res.status_code == 200
        assert res.json()["kb_id"] == "my_kb"

    def test_rebuild_no_graph_builder_returns_501(self, app_client, mock_container):
        mock_container.graph_builder = None
        res = app_client.post("/kg/communities/rebuild")
        assert res.status_code == 501


# ═══════════════════════════════════════════════════════════════════════════════
# KG: /kg/query
# ═══════════════════════════════════════════════════════════════════════════════

def _make_mock_gr():
    """Build a mock GraphRetriever with all search methods mocked."""
    from rag.graph.models import Node, Edge, NodeType, SubGraph
    node = Node(id="n_q1", kb_id="global", name="QueryEntity",
                node_type=NodeType.CONCEPT, degree=1, created_at=time.time())
    edge = Edge(id="e_q1", kb_id="global", src_id="n_q1", dst_id="n_q1",
                relation="self_ref", created_at=time.time())
    sub = SubGraph(nodes=[node], edges=[edge], context_text="KG context text",
                   reasoning_chain=[{"src": "A", "relation": "r", "dst": "B"}])

    mock_gr = MagicMock()
    mock_gr.local_search    = AsyncMock(return_value=sub)
    mock_gr.global_search   = AsyncMock(return_value=sub)
    mock_gr.path_search     = AsyncMock(return_value=sub)
    mock_gr.format_for_prompt = MagicMock(return_value="KG context text")
    return mock_gr


class TestKGQuery:
    def test_query_local_mode(self, kg_app_client):
        mock_gr = _make_mock_gr()
        with patch("server._gr", return_value=mock_gr):
            res = kg_app_client.post("/kg/query", json={
                "query": "What is TestEntity?", "kb_id": "global", "mode": "local",
            })
        assert res.status_code == 200
        data = res.json()
        assert "context"     in data
        assert "subgraph"    in data
        assert "nodes_found" in data
        assert "edges_found" in data
        assert data["mode"] == "local"

    def test_query_global_mode(self, kg_app_client):
        mock_gr = _make_mock_gr()
        with patch("server._gr", return_value=mock_gr):
            res = kg_app_client.post("/kg/query", json={
                "query": "Summarize the graph", "kb_id": "global", "mode": "global",
            })
        assert res.status_code == 200
        mock_gr.global_search.assert_called_once()

    def test_query_path_mode(self, kg_app_client):
        mock_gr = _make_mock_gr()
        with patch("server._gr", return_value=mock_gr):
            res = kg_app_client.post("/kg/query", json={
                "query": "How are A and B related?", "kb_id": "global", "mode": "path",
            })
        assert res.status_code == 200
        mock_gr.path_search.assert_called_once()

    def test_query_default_mode_is_local(self, kg_app_client):
        mock_gr = _make_mock_gr()
        with patch("server._gr", return_value=mock_gr):
            res = kg_app_client.post("/kg/query", json={
                "query": "something", "kb_id": "global",
            })
        assert res.status_code == 200
        mock_gr.local_search.assert_called_once()

    def test_query_no_kg_returns_501(self, app_client, mock_container):
        mock_container.kg_store = None
        res = app_client.post("/kg/query", json={"query": "q", "kb_id": "global"})
        assert res.status_code == 501


# ═══════════════════════════════════════════════════════════════════════════════
# KG: /kg/search/entities
# ═══════════════════════════════════════════════════════════════════════════════

class TestKGSearchEntities:
    def test_search_returns_entities_list(self, kg_app_client):
        res = kg_app_client.post(
            "/kg/search/entities",
            params={"query": "TestEntity", "kb_id": "global"},
        )
        assert res.status_code == 200
        data = res.json()
        assert "entities" in data
        assert isinstance(data["entities"], list)

    def test_search_entity_shape(self, kg_app_client):
        res = kg_app_client.post(
            "/kg/search/entities",
            params={"query": "TestEntity", "kb_id": "global"},
        )
        entities = res.json()["entities"]
        if entities:
            e = entities[0]
            assert "id"          in e
            assert "name"        in e
            assert "type"        in e
            assert "description" in e
            assert "degree"      in e

    def test_search_deduplicates_text_and_vector_hits(self, kg_app_client, mock_kg_store):
        """search_nodes_by_text and search_nodes_by_embedding return same node → only one in result."""
        res = kg_app_client.post(
            "/kg/search/entities",
            params={"query": "TestEntity", "kb_id": "global", "limit": "20"},
        )
        ids = [e["id"] for e in res.json()["entities"]]
        assert len(ids) == len(set(ids)), "Duplicate entity IDs in search results"

    def test_search_no_kg_returns_501(self, app_client, mock_container):
        mock_container.kg_store = None
        res = app_client.post("/kg/search/entities", params={"query": "x"})
        assert res.status_code == 501


# ═══════════════════════════════════════════════════════════════════════════════
# KG: /kg/subgraph
# ═══════════════════════════════════════════════════════════════════════════════

class TestKGSubgraph:
    def test_subgraph_returns_nodes_edges(self, kg_app_client):
        res = kg_app_client.post("/kg/subgraph", json={
            "node_id": "n_001", "kb_id": "global", "hops": 1,
        })
        assert res.status_code == 200
        data = res.json()
        assert "nodes" in data
        assert "edges" in data
        assert isinstance(data["nodes"], list)
        assert isinstance(data["edges"], list)

    def test_subgraph_center_node_prepended(self, kg_app_client, mock_kg_store):
        """Center node is included in nodes list when not already present."""
        from rag.graph.models import Node, NodeType
        center = Node(id="n_center", kb_id="global", name="Center",
                      node_type=NodeType.PERSON, degree=5, created_at=time.time())
        neighbor = Node(id="n_neighbor", kb_id="global", name="Neighbor",
                        node_type=NodeType.ORG, degree=1, created_at=time.time())
        mock_kg_store.get_node = AsyncMock(return_value=center)
        mock_kg_store.get_neighbors = AsyncMock(return_value=([neighbor], []))

        res = kg_app_client.post("/kg/subgraph", json={"node_id": "n_center", "kb_id": "global"})
        assert res.status_code == 200
        ids = [n["id"] for n in res.json()["nodes"]]
        assert "n_center" in ids

    def test_subgraph_unknown_node_returns_empty_center(self, kg_app_client, mock_kg_store):
        """If the node doesn't exist, no center is prepended."""
        mock_kg_store.get_node = AsyncMock(return_value=None)
        mock_kg_store.get_neighbors = AsyncMock(return_value=([], []))

        res = kg_app_client.post("/kg/subgraph", json={"node_id": "ghost_node", "kb_id": "global"})
        assert res.status_code == 200
        assert res.json()["nodes"] == []

    def test_subgraph_no_kg_returns_501(self, app_client, mock_container):
        mock_container.kg_store = None
        res = app_client.post("/kg/subgraph", json={"node_id": "x", "kb_id": "global"})
        assert res.status_code == 501


# ═══════════════════════════════════════════════════════════════════════════════
# KG: /kg/path
# ═══════════════════════════════════════════════════════════════════════════════

class TestKGPath:
    def test_path_returns_expected_keys(self, kg_app_client):
        mock_gr = _make_mock_gr()
        with patch("server._gr", return_value=mock_gr):
            res = kg_app_client.post("/kg/path", json={
                "query": "How are A and B related?", "kb_id": "global",
            })
        assert res.status_code == 200
        data = res.json()
        assert "subgraph"        in data
        assert "reasoning_chain" in data
        assert "context"         in data

    def test_path_no_kg_returns_501(self, app_client, mock_container):
        mock_container.kg_store = None
        res = app_client.post("/kg/path", json={"query": "q", "kb_id": "global"})
        assert res.status_code == 501


# ═══════════════════════════════════════════════════════════════════════════════
# KG: /kg/entities (list, detail, patch, delete)
# ═══════════════════════════════════════════════════════════════════════════════

class TestKGEntitiesListing:
    def test_list_entities(self, kg_app_client):
        res = kg_app_client.get("/kg/entities?kb_id=global")
        assert res.status_code == 200
        data = res.json()
        assert "entities" in data
        assert "total"    in data
        assert isinstance(data["entities"], list)

    def test_list_entities_shape(self, kg_app_client):
        entities = kg_app_client.get("/kg/entities?kb_id=global").json()["entities"]
        if entities:
            e = entities[0]
            assert "id"      in e
            assert "name"    in e
            assert "type"    in e
            assert "degree"  in e
            assert "aliases" in e

    def test_list_entities_with_type_filter(self, kg_app_client, mock_kg_store):
        from rag.graph.models import Node, NodeType
        person = Node(id="np1", kb_id="global", name="Alice",
                      node_type=NodeType.PERSON, degree=1, created_at=time.time())
        mock_kg_store.list_nodes = AsyncMock(return_value=[person])
        res = kg_app_client.get("/kg/entities?kb_id=global&node_type=PERSON")
        assert res.status_code == 200

    def test_list_entities_pagination(self, kg_app_client):
        res = kg_app_client.get("/kg/entities?kb_id=global&limit=1&offset=0")
        assert res.status_code == 200

    def test_list_entities_no_kg_returns_501(self, app_client, mock_container):
        mock_container.kg_store = None
        res = app_client.get("/kg/entities?kb_id=global")
        assert res.status_code == 501


class TestKGEntityDetail:
    def test_get_entity_detail(self, kg_app_client):
        res = kg_app_client.get("/kg/entities/n_001")
        assert res.status_code == 200
        data = res.json()
        assert data["id"]   == "n_001"
        assert "name"       in data
        assert "type"       in data
        assert "edges"      in data
        assert isinstance(data["edges"], list)

    def test_get_entity_edges_shape(self, kg_app_client):
        data = kg_app_client.get("/kg/entities/n_001").json()
        for edge in data["edges"]:
            assert "id"       in edge
            assert "src_id"   in edge
            assert "dst_id"   in edge
            assert "relation" in edge

    def test_get_entity_not_found_returns_404(self, kg_app_client, mock_kg_store):
        mock_kg_store.get_node = AsyncMock(return_value=None)
        res = kg_app_client.get("/kg/entities/ghost_id")
        assert res.status_code == 404

    def test_get_entity_no_kg_returns_501(self, app_client, mock_container):
        mock_container.kg_store = None
        res = app_client.get("/kg/entities/any_id")
        assert res.status_code == 501


class TestKGEntityPatch:
    def test_patch_entity_name(self, kg_app_client):
        res = kg_app_client.patch("/kg/entities/n_001", json={"name": "UpdatedName"})
        assert res.status_code == 200
        data = res.json()
        assert data["ok"] is True
        assert "id"   in data
        assert "name" in data

    def test_patch_entity_description(self, kg_app_client):
        res = kg_app_client.patch("/kg/entities/n_001", json={"description": "New description"})
        assert res.status_code == 200
        assert res.json()["ok"] is True

    def test_patch_entity_not_found_returns_404(self, kg_app_client, mock_kg_store):
        mock_kg_store.get_node = AsyncMock(return_value=None)
        res = kg_app_client.patch("/kg/entities/ghost_id", json={"name": "X"})
        assert res.status_code == 404

    def test_patch_entity_no_kg_returns_501(self, app_client, mock_container):
        mock_container.kg_store = None
        res = app_client.patch("/kg/entities/any", json={"name": "X"})
        assert res.status_code == 501


class TestKGEntityDelete:
    def test_delete_entity_returns_ok(self, kg_app_client):
        res = kg_app_client.delete("/kg/entities/n_001")
        assert res.status_code == 200
        assert res.json()["ok"] is True

    def test_delete_entity_calls_store(self, kg_app_client, mock_kg_store):
        kg_app_client.delete("/kg/entities/n_delete_me")
        mock_kg_store.delete_node.assert_called()

    def test_delete_entity_no_kg_returns_501(self, app_client, mock_container):
        mock_container.kg_store = None
        res = app_client.delete("/kg/entities/any")
        assert res.status_code == 501


# ═══════════════════════════════════════════════════════════════════════════════
# KG: /kg/edges
# ═══════════════════════════════════════════════════════════════════════════════

class TestKGEdges:
    def test_add_edge_creates_edge(self, kg_app_client):
        res = kg_app_client.post("/kg/edges", json={
            "kb_id":    "global",
            "src_name": "Alice",
            "relation": "works_at",
            "dst_name": "Anthropic",
            "context":  "Alice works at Anthropic.",
            "weight":   0.95,
        })
        assert res.status_code == 200
        data = res.json()
        assert data["ok"]      is True
        assert "edge_id"       in data
        assert data["src"]     == "Alice"
        assert data["dst"]     == "Anthropic"
        assert data["relation"] == "works_at"

    def test_add_edge_reuses_existing_entity(self, kg_app_client, mock_kg_store):
        """When get_node_by_name finds an existing node it should be reused."""
        from rag.graph.models import Node, NodeType
        existing = Node(id="n_existing", kb_id="global", name="Alice",
                        node_type=NodeType.PERSON, degree=3, created_at=time.time())
        mock_kg_store.get_node_by_name = AsyncMock(return_value=existing)
        res = kg_app_client.post("/kg/edges", json={
            "kb_id": "global", "src_name": "Alice",
            "relation": "knows", "dst_name": "Bob",
        })
        assert res.status_code == 200
        # upsert_node should NOT be called for Alice (she already exists)
        calls = [str(c) for c in mock_kg_store.upsert_node.call_args_list]
        alice_calls = [c for c in calls if "Alice" in c]
        assert len(alice_calls) == 0, "Existing node was re-created"

    def test_add_edge_no_kg_returns_501(self, app_client, mock_container):
        mock_container.kg_store = None
        res = app_client.post("/kg/edges", json={
            "kb_id": "global", "src_name": "A", "relation": "r", "dst_name": "B",
        })
        assert res.status_code == 501

    def test_delete_edge_returns_ok(self, kg_app_client):
        res = kg_app_client.delete("/kg/edges/e_001?kb_id=global")
        assert res.status_code == 200
        assert res.json()["ok"] is True

    def test_delete_edge_calls_store(self, kg_app_client, mock_kg_store):
        kg_app_client.delete("/kg/edges/e_to_delete")
        mock_kg_store.delete_edge.assert_called()

    def test_delete_edge_no_kg_returns_501(self, app_client, mock_container):
        mock_container.kg_store = None
        res = app_client.delete("/kg/edges/any")
        assert res.status_code == 501


# ═══════════════════════════════════════════════════════════════════════════════
# KG: /kg/graph
# ═══════════════════════════════════════════════════════════════════════════════

class TestKGGraph:
    def test_full_graph_returns_nodes_edges_kb_id(self, kg_app_client):
        res = kg_app_client.get("/kg/graph?kb_id=global")
        assert res.status_code == 200
        data = res.json()
        assert "nodes"  in data
        assert "edges"  in data
        assert "kb_id"  in data
        assert data["kb_id"] == "global"

    def test_full_graph_node_shape(self, kg_app_client):
        nodes = kg_app_client.get("/kg/graph?kb_id=global").json()["nodes"]
        if nodes:
            n = nodes[0]
            assert "id"          in n
            assert "name"        in n
            assert "type"        in n
            assert "description" in n
            assert "degree"      in n

    def test_full_graph_edge_shape(self, kg_app_client):
        edges = kg_app_client.get("/kg/graph?kb_id=global").json()["edges"]
        if edges:
            e = edges[0]
            assert "id"       in e
            assert "src_id"   in e
            assert "dst_id"   in e
            assert "relation" in e
            assert "weight"   in e

    def test_full_graph_filters_dangling_edges(self, kg_app_client, mock_kg_store):
        """Edges referencing nodes not in the result set must be filtered out."""
        from rag.graph.models import Node, Edge, NodeType
        n = Node(id="n_only", kb_id="global", name="Solo",
                 node_type=NodeType.OTHER, degree=0, created_at=time.time())
        dangling_edge = Edge(id="e_dangle", kb_id="global",
                             src_id="n_only", dst_id="n_gone",
                             relation="r", created_at=time.time())
        valid_edge = Edge(id="e_valid", kb_id="global",
                          src_id="n_only", dst_id="n_only",
                          relation="self", created_at=time.time())
        mock_kg_store.get_full_graph = AsyncMock(return_value=([n], [dangling_edge, valid_edge]))

        data = kg_app_client.get("/kg/graph?kb_id=global").json()
        edge_ids = [e["id"] for e in data["edges"]]
        assert "e_dangle" not in edge_ids
        assert "e_valid"  in edge_ids

    def test_full_graph_no_kg_returns_501(self, app_client, mock_container):
        mock_container.kg_store = None
        res = app_client.get("/kg/graph?kb_id=global")
        assert res.status_code == 501


# ═══════════════════════════════════════════════════════════════════════════════
# KG: /kg/stats
# ═══════════════════════════════════════════════════════════════════════════════

class TestKGStats:
    def test_stats_returns_kb_id(self, kg_app_client):
        res = kg_app_client.get("/kg/stats?kb_id=global")
        assert res.status_code == 200
        assert res.json()["kb_id"] == "global"

    def test_stats_has_node_edge_counts(self, kg_app_client):
        data = kg_app_client.get("/kg/stats?kb_id=global").json()
        assert "nodes"  in data
        assert "edges"  in data

    def test_stats_no_kg_returns_501(self, app_client, mock_container):
        mock_container.kg_store = None
        res = app_client.get("/kg/stats?kb_id=global")
        assert res.status_code == 501


# ═══════════════════════════════════════════════════════════════════════════════
# KG: /kg/documents/{doc_id} and /kg/reindex/{doc_id}
# ═══════════════════════════════════════════════════════════════════════════════

class TestKGDocumentDelete:
    def test_delete_doc_graph_data(self, kg_app_client):
        res = kg_app_client.delete("/kg/documents/doc_001?kb_id=global")
        assert res.status_code == 200
        data = res.json()
        assert data["ok"]     is True
        assert data["doc_id"] == "doc_001"

    def test_delete_doc_calls_delete_by_doc(self, kg_app_client, mock_kg_store):
        kg_app_client.delete("/kg/documents/doc_to_del?kb_id=test_kb")
        mock_kg_store.delete_by_doc.assert_called_with("doc_to_del", "test_kb")

    def test_delete_doc_no_kg_returns_501(self, app_client, mock_container):
        mock_container.kg_store = None
        res = app_client.delete("/kg/documents/any?kb_id=global")
        assert res.status_code == 501


class TestKGReindex:
    def test_reindex_doc_returns_ok_and_message(self, kg_app_client):
        res = kg_app_client.post("/kg/reindex/doc_001?kb_id=global")
        assert res.status_code == 200
        data = res.json()
        assert data["ok"] is True
        assert "message" in data
        assert len(data["message"]) > 0

    def test_reindex_calls_delete_by_doc(self, kg_app_client, mock_kg_store):
        kg_app_client.post("/kg/reindex/some_doc?kb_id=global")
        mock_kg_store.delete_by_doc.assert_called_with("some_doc", "global")

    def test_reindex_no_kg_returns_501(self, app_client, mock_container):
        mock_container.kg_store = None
        res = app_client.post("/kg/reindex/any?kb_id=global")
        assert res.status_code == 501
