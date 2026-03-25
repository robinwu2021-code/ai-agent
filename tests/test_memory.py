"""
tests/test_memory.py — Memory system unit tests

Covers:
  - InMemoryShortTermMemory  (STM CRUD, scratchpad, message list)
  - InMemoryLongTermMemory   (write, dedup, search, prune, profile)
  - _time_decay              (importance-weighted decay function)
  - _text_ngram_overlap      (Jaccard n-gram similarity)
  - MemorySystem             (is_workspace_aware facade)
  - MemoryFactory            (in_memory fallback, embed_fn, router_wrapper)
  - ScoredMemory             (dataclass defaults)

Run with:
    pytest tests/test_memory.py -v
"""
from __future__ import annotations

import asyncio
import math
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.models import AgentInput, AgentTask, MemoryEntry, MemoryType, Message, MessageRole


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_task(user_id: str = "u1", session_id: str = "s1") -> AgentTask:
    return AgentTask(
        session_id=session_id,
        user_id=user_id,
        input=AgentInput(text="hello"),
    )


def _make_entry(user_id: str = "u1", text: str = "remember this", importance: float = 0.5) -> MemoryEntry:
    return MemoryEntry(
        user_id=user_id,
        type=MemoryType.SEMANTIC,
        text=text,
        importance=importance,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Time decay utility
# ═══════════════════════════════════════════════════════════════════════════════

class TestTimeDecay:
    def _decay(self, importance: float, days: float) -> float:
        from memory.stores import _time_decay
        return _time_decay(importance, days)

    def test_zero_days_returns_one(self):
        assert self._decay(0.5, 0.0) == pytest.approx(1.0)

    def test_high_importance_decays_slower(self):
        low  = self._decay(0.1, 30)
        high = self._decay(0.9, 30)
        assert high > low, "High importance should decay slower than low importance"

    def test_negative_days_treated_as_zero(self):
        assert self._decay(0.5, -5) == pytest.approx(1.0)

    def test_very_old_memory_near_zero(self):
        score = self._decay(0.1, 365 * 5)  # 5 years, low importance
        assert score < 0.01, f"Very old low-importance memory should be near zero, got {score:.4f}"

    def test_high_importance_very_old_still_above_zero(self):
        score = self._decay(1.0, 365)  # 1 year, max importance
        assert score > 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 2. N-gram overlap utility
# ═══════════════════════════════════════════════════════════════════════════════

class TestNgramOverlap:
    def _overlap(self, a: str, b: str) -> float:
        from memory.stores import _text_ngram_overlap
        return _text_ngram_overlap(a, b)

    def test_identical_strings_return_one(self):
        assert self._overlap("hello world", "hello world") == pytest.approx(1.0)

    def test_completely_different_strings_return_zero(self):
        # No shared bigrams between "abcd" and "wxyz"
        score = self._overlap("abcd", "wxyz")
        assert score == pytest.approx(0.0)

    def test_partial_overlap_between_zero_and_one(self):
        score = self._overlap("hello world", "hello there")
        assert 0.0 < score < 1.0

    def test_short_strings_handled(self):
        # Single char strings produce no bigrams
        score = self._overlap("a", "b")
        assert score == pytest.approx(0.0)

    def test_symmetric(self):
        a, b = "machine learning", "deep learning methods"
        assert self._overlap(a, b) == pytest.approx(self._overlap(b, a))

    def test_empty_strings(self):
        assert self._overlap("", "") == pytest.approx(0.0)
        assert self._overlap("hello", "") == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. InMemoryShortTermMemory
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def stm():
    from memory.stores import InMemoryShortTermMemory
    return InMemoryShortTermMemory()


class TestInMemorySTM:
    def test_save_and_load_task(self, stm):
        task = _make_task()
        _run(stm.save_task(task))
        loaded = _run(stm.load_task(task.id))
        assert loaded is not None
        assert loaded.id == task.id
        assert loaded.user_id == task.user_id

    def test_load_nonexistent_returns_none(self, stm):
        assert _run(stm.load_task("nonexistent")) is None

    def test_delete_task(self, stm):
        task = _make_task()
        _run(stm.save_task(task))
        _run(stm.delete_task(task.id))
        assert _run(stm.load_task(task.id)) is None

    def test_delete_nonexistent_does_not_raise(self, stm):
        _run(stm.delete_task("ghost_task"))  # Should not raise

    def test_append_and_get_messages(self, stm):
        task = _make_task()
        _run(stm.save_task(task))
        msg1 = Message(role=MessageRole.USER, content="Hello")
        msg2 = Message(role=MessageRole.ASSISTANT, content="Hi there")
        _run(stm.append_message(task.id, msg1))
        _run(stm.append_message(task.id, msg2))

        messages = _run(stm.get_messages(task.id))
        assert len(messages) == 2
        assert messages[0].content == "Hello"
        assert messages[1].content == "Hi there"

    def test_get_messages_nonexistent_returns_empty(self, stm):
        assert _run(stm.get_messages("no_task")) == []

    def test_scratchpad_set_and_get(self, stm):
        task = _make_task()
        _run(stm.save_task(task))
        _run(stm.set_scratchpad(task.id, "key1", {"data": 42}))
        val = _run(stm.get_scratchpad(task.id, "key1"))
        assert val == {"data": 42}

    def test_scratchpad_missing_key_returns_none(self, stm):
        task = _make_task()
        _run(stm.save_task(task))
        assert _run(stm.get_scratchpad(task.id, "missing_key")) is None

    def test_scratchpad_overwrite(self, stm):
        task = _make_task()
        _run(stm.save_task(task))
        _run(stm.set_scratchpad(task.id, "k", "v1"))
        _run(stm.set_scratchpad(task.id, "k", "v2"))
        assert _run(stm.get_scratchpad(task.id, "k")) == "v2"

    def test_task_isolation(self, stm):
        t1 = _make_task("user_a", "sess_1")
        t2 = _make_task("user_b", "sess_2")
        _run(stm.save_task(t1))
        _run(stm.save_task(t2))
        loaded_t1 = _run(stm.load_task(t1.id))
        loaded_t2 = _run(stm.load_task(t2.id))
        assert loaded_t1.user_id == "user_a"
        assert loaded_t2.user_id == "user_b"

    def test_loaded_task_is_copy_not_reference(self, stm):
        """Mutating the loaded task should not affect the stored version."""
        task = _make_task()
        _run(stm.save_task(task))
        loaded = _run(stm.load_task(task.id))
        loaded.retries = 999
        fresh = _run(stm.load_task(task.id))
        assert fresh.retries == 0  # Original unchanged


# ═══════════════════════════════════════════════════════════════════════════════
# 4. InMemoryLongTermMemory
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def ltm():
    from memory.stores import InMemoryLongTermMemory
    return InMemoryLongTermMemory()


class TestInMemoryLTM:
    def test_write_and_search(self, ltm):
        entry = _make_entry(text="Python is a programming language")
        _run(ltm.write(entry))
        results = _run(ltm.search("u1", "Python programming"))
        assert len(results) > 0
        assert results[0].text == entry.text

    def test_search_returns_empty_for_no_match(self, ltm):
        entry = _make_entry(text="Python is a programming language")
        _run(ltm.write(entry))
        # Use a query with characters/bigrams that share zero overlap with the entry text
        # ("zzqq xbff wkdd" → bigrams: zz,zq,qq,qx,xb,bf,ff,fw,wk,kd,dd — none in "pythonisaprogramminglanguage")
        results = _run(ltm.search("u1", "zzqq xbff wkdd"))
        assert results == []

    def test_user_isolation(self, ltm):
        _run(ltm.write(_make_entry("user_a", "Secret info for user A only")))
        _run(ltm.write(_make_entry("user_b", "Different info for user B")))
        results_a = _run(ltm.search("user_a", "Secret info user A"))
        results_b = _run(ltm.search("user_b", "Different info user B"))
        assert all(e.user_id == "user_a" for e in results_a)
        assert all(e.user_id == "user_b" for e in results_b)

    def test_dedup_prevents_duplicate_write(self, ltm):
        e1 = _make_entry(text="The sky is blue and beautiful today")
        e2 = _make_entry(text="the sky is blue and beautiful today")  # very similar
        _run(ltm.write(e1))
        _run(ltm.write(e2))
        # Should have been deduped, not stored twice
        results = _run(ltm.search("u1", "sky blue beautiful"))
        assert len(results) == 1

    def test_distinct_entries_both_written(self, ltm):
        _run(ltm.write(_make_entry(text="cats are fuzzy cute animals")))
        _run(ltm.write(_make_entry(text="quantum entanglement physics experiments")))
        entries = ltm._entries
        assert len(entries) == 2

    def test_search_top_k_respected(self, ltm):
        for i in range(10):
            _run(ltm.write(MemoryEntry(
                user_id="u1", type=MemoryType.SEMANTIC,
                text=f"memory entry number {i} about topics",
            )))
        results = _run(ltm.search("u1", "memory entry topics", top_k=3))
        assert len(results) <= 3

    def test_search_boosts_importance_on_access(self, ltm):
        entry = _make_entry(text="important fact to remember always", importance=0.5)
        _run(ltm.write(entry))
        _run(ltm.search("u1", "important fact remember"))
        # Importance should have been boosted
        assert ltm._entries[0].importance > 0.5

    def test_prune_removes_low_score_entries(self, ltm):
        # Write a very old, low-importance entry
        old_entry = MemoryEntry(
            user_id="u1", type=MemoryType.EPISODIC,
            text="very old forgotten memory from long ago",
            importance=0.05,
            created_at=datetime.utcnow() - timedelta(days=365),
        )
        fresh_entry = MemoryEntry(
            user_id="u1", type=MemoryType.SEMANTIC,
            text="important recent memory about current tasks",
            importance=0.9,
        )
        _run(ltm.write(old_entry))
        _run(ltm.write(fresh_entry))
        pruned = _run(ltm.prune("u1", max_items=100, score_threshold=0.1))
        assert pruned >= 1
        remaining_texts = [e.text for e in ltm._entries]
        assert "important recent memory about current tasks" in remaining_texts

    def test_prune_max_items_enforced(self, ltm):
        for i in range(10):
            _run(ltm.write(MemoryEntry(
                user_id="u1", type=MemoryType.SEMANTIC,
                text=f"distinct memory item {i} unique content here",
                importance=0.5,
            )))
        _run(ltm.prune("u1", max_items=5, score_threshold=0.0))
        user_entries = [e for e in ltm._entries if e.user_id == "u1"]
        assert len(user_entries) <= 5

    def test_prune_leaves_other_users_untouched(self, ltm):
        _run(ltm.write(_make_entry("user_a", "content for user a")))
        _run(ltm.write(_make_entry("user_b", "content for user b")))
        _run(ltm.prune("user_a", max_items=0, score_threshold=1.0))
        user_b_entries = [e for e in ltm._entries if e.user_id == "user_b"]
        assert len(user_b_entries) == 1

    def test_get_and_update_profile(self, ltm):
        profile = _run(ltm.get_profile("u1"))
        assert profile == {}
        _run(ltm.update_profile("u1", {"name": "Alice", "lang": "zh"}))
        updated = _run(ltm.get_profile("u1"))
        assert updated["name"] == "Alice"
        assert updated["lang"] == "zh"

    def test_update_profile_merges_data(self, ltm):
        _run(ltm.update_profile("u1", {"a": 1}))
        _run(ltm.update_profile("u1", {"b": 2}))
        profile = _run(ltm.get_profile("u1"))
        assert profile["a"] == 1
        assert profile["b"] == 2

    def test_profile_isolation(self, ltm):
        _run(ltm.update_profile("user_x", {"pref": "dark"}))
        profile_y = _run(ltm.get_profile("user_y"))
        assert profile_y == {}


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MemorySystem facade
# ═══════════════════════════════════════════════════════════════════════════════

class TestMemorySystem:
    def test_is_workspace_aware_true_when_ltm_has_search_mixed(self):
        from memory.protocols import MemorySystem

        mock_ltm = MagicMock()
        mock_ltm.search_mixed = AsyncMock()  # has the optional method
        ms = MemorySystem(stm=None, ltm=mock_ltm, working=None, consolidator=None)
        assert ms.is_workspace_aware() is True

    def test_is_workspace_aware_false_when_ltm_lacks_search_mixed(self):
        from memory.protocols import MemorySystem

        mock_ltm = MagicMock(spec=["write", "search", "prune"])  # no search_mixed
        ms = MemorySystem(stm=None, ltm=mock_ltm, working=None, consolidator=None)
        assert ms.is_workspace_aware() is False

    def test_scored_memory_defaults(self):
        from memory.protocols import ScoredMemory
        sm = ScoredMemory(id="m1", user_id="u1", text="hello")
        assert sm.score == 0.0
        assert sm.importance == 0.5
        assert sm.memory_type == "semantic"
        assert sm.scope == "personal"
        assert sm.metadata == {}


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MemoryFactory
# ═══════════════════════════════════════════════════════════════════════════════

def _make_mock_router():
    router = MagicMock()
    router.embed = AsyncMock(return_value=[0.1] * 8)
    router.summarize = AsyncMock(return_value="summary text")
    router.chat = AsyncMock()
    router._registry = {}
    return router


def _make_memory_config(
    stm_backend="in_memory",
    ltm_backend="in_memory",
):
    """Build a minimal MemoryConfig-like object."""
    stm_cfg = MagicMock()
    stm_cfg.backend = stm_backend

    ltm_cfg = MagicMock()
    ltm_cfg.backend = ltm_backend
    ltm_cfg.use_global_vector_store = False
    ltm_cfg.llm_engine = ""
    ltm_cfg.embed_engine = ""
    ltm_cfg.collection = "memories"
    ltm_cfg.mem0_dedup_threshold = 0.9
    ltm_cfg.mem0_max_memories = 500

    working_cfg = MagicMock()
    working_cfg.summarize_engine = ""
    working_cfg.ltm_score_min = 0.3
    working_cfg.compress_threshold = 20

    consolidation_cfg = MagicMock()
    consolidation_cfg.engine = ""
    consolidation_cfg.incremental_every = 5
    consolidation_cfg.segment_size = 8
    consolidation_cfg.min_importance = 0.3

    cfg = MagicMock()
    cfg.stm = stm_cfg
    cfg.ltm = ltm_cfg
    cfg.working = working_cfg
    cfg.consolidation = consolidation_cfg
    return cfg


class TestMemoryFactory:
    def test_build_returns_memory_system(self):
        from memory.factory import MemoryFactory
        from memory.protocols import MemorySystem
        cfg = _make_memory_config()
        router = _make_mock_router()
        ms = MemoryFactory.build(cfg, router)
        assert isinstance(ms, MemorySystem)

    def test_in_memory_stm_fallback(self):
        from memory.factory import MemoryFactory
        from memory.stores import InMemoryShortTermMemory
        cfg = _make_memory_config(stm_backend="in_memory")
        ms = MemoryFactory.build(cfg, _make_mock_router())
        assert isinstance(ms.stm, InMemoryShortTermMemory)

    def test_in_memory_ltm_fallback(self):
        from memory.factory import MemoryFactory
        from memory.stores import InMemoryLongTermMemory
        cfg = _make_memory_config(ltm_backend="in_memory")
        ms = MemoryFactory.build(cfg, _make_mock_router())
        assert isinstance(ms.ltm, InMemoryLongTermMemory)

    def test_redis_stm_failure_falls_back_to_in_memory(self):
        """If Redis is unavailable, STM should silently fall back to InMemory."""
        from memory.factory import MemoryFactory
        from memory.stores import InMemoryShortTermMemory
        cfg = _make_memory_config(stm_backend="redis")
        cfg.stm.redis_url = "redis://invalid-host-that-does-not-exist:9999"
        cfg.stm.ttl_idle = 1800
        cfg.stm.ttl_max = 86400
        ms = MemoryFactory.build(cfg, _make_mock_router())
        assert isinstance(ms.stm, InMemoryShortTermMemory)

    def test_unknown_ltm_backend_falls_back_to_in_memory(self):
        """Unknown LTM backend should fall through to InMemory."""
        from memory.factory import MemoryFactory
        from memory.stores import InMemoryLongTermMemory
        cfg = _make_memory_config(ltm_backend="unknown_backend_xyz")
        ms = MemoryFactory.build(cfg, _make_mock_router())
        assert isinstance(ms.ltm, InMemoryLongTermMemory)

    def test_make_embed_fn_without_alias(self):
        from memory.factory import MemoryFactory
        router = _make_mock_router()
        embed_fn = MemoryFactory._make_embed_fn(router, "")
        result = _run(embed_fn("test text"))
        router.embed.assert_called_once_with("test text")
        assert result == [0.1] * 8

    def test_make_embed_fn_with_unknown_alias_falls_back_to_router(self):
        from memory.factory import MemoryFactory
        router = _make_mock_router()
        router._registry = {}  # alias not in registry
        embed_fn = MemoryFactory._make_embed_fn(router, "some_engine")
        _run(embed_fn("text"))
        router.embed.assert_called_once()

    def test_router_wrapper_summarize(self):
        from memory.factory import MemoryFactory
        router = _make_mock_router()
        wrapper = MemoryFactory._make_router_wrapper(router, "")
        result = _run(wrapper.summarize("long text to summarize", max_tokens=100))
        assert result == "summary text"

    def test_router_wrapper_has_chat(self):
        from memory.factory import MemoryFactory
        router = _make_mock_router()
        wrapper = MemoryFactory._make_router_wrapper(router, "")
        assert hasattr(wrapper, "chat")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. InMemorySTM protocol compliance
# ═══════════════════════════════════════════════════════════════════════════════

class TestSTMProtocolCompliance:
    def test_implements_short_term_store_protocol(self):
        from memory.protocols import ShortTermStore
        from memory.stores import InMemoryShortTermMemory
        stm = InMemoryShortTermMemory()
        assert isinstance(stm, ShortTermStore)


class TestLTMProtocolCompliance:
    def test_in_memory_ltm_implements_long_term_store_protocol(self):
        from memory.protocols import LongTermStore
        from memory.stores import InMemoryLongTermMemory
        ltm = InMemoryLongTermMemory()
        assert isinstance(ltm, LongTermStore)
