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
    pkb.list_documents.return_value = [doc]
    return pkb


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
    container._effective_router = MagicMock(return_value=router)
    container.kg_store = None
    return container


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
