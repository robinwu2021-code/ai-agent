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
