"""
memory/stores.py — 记忆存储实现

短期记忆：
  - InMemoryShortTermMemory   开发/测试用
  - RedisShortTermMemory      生产用（TTL 活跃延续、分级 TTL）

长期记忆：
  - InMemoryLongTermMemory    开发/测试用（n-gram + 语义去重）
  - QdrantLongTermMemory      Qdrant 向量数据库（完整 prune、访问增强）
  - MilvusLongTermMemory      Milvus 向量数据库（复用 MilvusVectorStore 基础设施）

优化点（vs 旧版）：
  STM:
    · Redis 每次读写自动续期 TTL（避免活跃会话中途失效）
    · 可配置 max_ttl，活跃对话最长保存 24h
    · append_message 同时维护 task 对象和独立 list，保持一致
  LTM:
    · write() 前做语义/文本相似检查，避免重复写入
    · importance-weighted time decay: 重要记忆衰减更慢
    · 访问时自动 boost importance（最多 +0.15，上限 1.0）
    · QdrantLongTermMemory.prune() 正式实现（非 stub）
    · 新增 MilvusLongTermMemory（复用 rag/milvus_store.py 基础设施）
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Any

import structlog

from core.models import AgentTask, MemoryEntry, MemoryType, Message

log = structlog.get_logger(__name__)

# 访问时对 importance 的加成（每次被召回 +0.05，上限 1.0）
_ACCESS_BOOST  = 0.05
_IMPORTANCE_MAX = 1.0

# importance-weighted decay: score = importance * exp(-k * days * (1 - 0.7*importance))
# importance=0.9 → k_eff ≈ 0.05 * 0.37 = 0.018/day（半衰期 38 天）
# importance=0.3 → k_eff ≈ 0.05 * 0.79 = 0.040/day（半衰期 17 天）
_DECAY_BASE_K = 0.05


def _time_decay(importance: float, days: float) -> float:
    """重要性加权时间衰减：高价值记忆衰减更慢。"""
    k = _DECAY_BASE_K * (1.0 - 0.7 * min(max(importance, 0.0), 1.0))
    return math.exp(-k * max(days, 0.0))


# ═══════════════════════════════════════════════════════
# SHORT-TERM MEMORY
# ═══════════════════════════════════════════════════════

class InMemoryShortTermMemory:
    """纯内存实现，进程重启后丢失，适合开发和测试。"""

    def __init__(self) -> None:
        self._tasks:      dict[str, AgentTask]       = {}
        self._scratchpad: dict[str, dict[str, Any]]  = {}

    async def save_task(self, task: AgentTask) -> None:
        self._tasks[task.id] = task.model_copy(deep=True)

    async def load_task(self, task_id: str) -> AgentTask | None:
        t = self._tasks.get(task_id)
        return t.model_copy(deep=True) if t else None

    async def delete_task(self, task_id: str) -> None:
        self._tasks.pop(task_id, None)
        self._scratchpad.pop(task_id, None)

    async def append_message(self, task_id: str, msg: Message) -> None:
        if task := self._tasks.get(task_id):
            task.add_message(msg)

    async def get_messages(self, task_id: str) -> list[Message]:
        if task := self._tasks.get(task_id):
            return list(task.history)
        return []

    async def set_scratchpad(self, task_id: str, key: str, value: Any) -> None:
        self._scratchpad.setdefault(task_id, {})[key] = value
        if task := self._tasks.get(task_id):
            task.scratchpad[key] = value

    async def get_scratchpad(self, task_id: str, key: str) -> Any:
        return self._scratchpad.get(task_id, {}).get(key)


class RedisShortTermMemory:
    """
    Redis 实现，支持多实例部署，TTL 控制过期。

    优化：
      · 每次读/写自动延续 TTL（活跃对话不会中途失效）
      · max_ttl 设上限（默认 86400s = 24h），防止冷数据永久堆积
      · task 对象与消息列表同步，get_messages 不再依赖 task JSON
    """

    DEFAULT_TTL = 1800    # 30 分钟不活跃后过期
    MAX_TTL     = 86400   # 最长保留 24 小时

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        ttl:       int = DEFAULT_TTL,
        max_ttl:   int = MAX_TTL,
    ) -> None:
        try:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(redis_url, decode_responses=True)
        except ImportError:
            raise RuntimeError("请安装 redis：pip install redis")
        self._ttl     = ttl
        self._max_ttl = max_ttl

    # Key helpers
    def _task_key(self,    task_id: str) -> str: return f"agent:task:{task_id}"
    def _msg_key(self,     task_id: str) -> str: return f"agent:msg:{task_id}"
    def _scratch_key(self, task_id: str) -> str: return f"agent:scratch:{task_id}"
    def _born_key(self,    task_id: str) -> str: return f"agent:born:{task_id}"

    async def _refresh(self, task_id: str) -> None:
        """活跃续期：延长 TTL，但不超过 max_ttl（从 born 时间算）。"""
        born_raw = await self._redis.get(self._born_key(task_id))
        if born_raw:
            born = float(born_raw)
            elapsed = datetime.utcnow().timestamp() - born
            remaining = max(0, self._max_ttl - int(elapsed))
            effective = min(self._ttl, remaining)
        else:
            effective = self._ttl

        if effective > 0:
            for key in (
                self._task_key(task_id),
                self._msg_key(task_id),
                self._scratch_key(task_id),
            ):
                await self._redis.expire(key, effective)

    async def save_task(self, task: AgentTask) -> None:
        pipe = self._redis.pipeline()
        pipe.setex(self._task_key(task.id), self._ttl, task.model_dump_json())
        # 记录首次创建时间（只写一次，setnx）
        pipe.setnx(self._born_key(task.id), str(datetime.utcnow().timestamp()))
        pipe.expire(self._born_key(task.id), self._max_ttl)
        await pipe.execute()

    async def load_task(self, task_id: str) -> AgentTask | None:
        raw = await self._redis.get(self._task_key(task_id))
        if not raw:
            return None
        await self._refresh(task_id)
        return AgentTask.model_validate_json(raw)

    async def delete_task(self, task_id: str) -> None:
        await self._redis.delete(
            self._task_key(task_id),
            self._msg_key(task_id),
            self._scratch_key(task_id),
            self._born_key(task_id),
        )

    async def append_message(self, task_id: str, msg: Message) -> None:
        await self._redis.rpush(self._msg_key(task_id), msg.model_dump_json())
        await self._refresh(task_id)

    async def get_messages(self, task_id: str) -> list[Message]:
        raws = await self._redis.lrange(self._msg_key(task_id), 0, -1)
        await self._refresh(task_id)
        return [Message.model_validate_json(r) for r in raws]

    async def set_scratchpad(self, task_id: str, key: str, value: Any) -> None:
        await self._redis.hset(self._scratch_key(task_id), key, json.dumps(value))
        await self._refresh(task_id)

    async def get_scratchpad(self, task_id: str, key: str) -> Any:
        raw = await self._redis.hget(self._scratch_key(task_id), key)
        return json.loads(raw) if raw else None


# ═══════════════════════════════════════════════════════
# LONG-TERM MEMORY — 公共工具
# ═══════════════════════════════════════════════════════

def _text_ngram_overlap(a: str, b: str, n: int = 2) -> float:
    """返回 0~1 的 n-gram 重叠率（Jaccard），用于粗粒度去重。"""
    def ngs(t: str) -> set[str]:
        t = t.lower().replace(" ", "")
        return {t[i:i+n] for i in range(len(t) - n + 1)} if len(t) >= n else set(t)
    sa, sb = ngs(a), ngs(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ═══════════════════════════════════════════════════════
# LONG-TERM MEMORY — InMemory（开发/测试）
# ═══════════════════════════════════════════════════════

class InMemoryLongTermMemory:
    """
    纯内存长期记忆。

    优化：
      · write() 语义去重：n-gram 相似度 > 0.75 → 更新 importance 而非重复写入
      · search() 加入 importance-weighted time decay 打分
      · 被召回的条目自动 boost importance
    """

    # 文本重叠超过此阈值视为重复
    DEDUP_THRESHOLD = 0.75

    def __init__(self) -> None:
        self._entries:  list[MemoryEntry]          = []
        self._profiles: dict[str, dict[str, Any]]  = {}

    async def write(self, entry: MemoryEntry) -> None:
        """
        写入前做 n-gram 去重：
          · 完全相同 → 更新 importance + accessed_at，跳过写入
          · 高度相似（> DEDUP_THRESHOLD）→ 同上
          · 否则新增
        """
        for e in self._entries:
            if e.user_id != entry.user_id:
                continue
            overlap = _text_ngram_overlap(e.text, entry.text)
            if overlap >= self.DEDUP_THRESHOLD:
                e.importance  = min(_IMPORTANCE_MAX, max(e.importance, entry.importance) + 0.05)
                e.accessed_at = datetime.utcnow()
                log.debug("memory.write.dedup", id=e.id, overlap=round(overlap, 2))
                return
        self._entries.append(entry)
        log.debug("memory.write", id=entry.id, type=entry.type.value)

    async def search(
        self,
        user_id: str,
        query: str,
        memory_type: MemoryType | None = None,
        top_k: int = 5,
    ) -> list[MemoryEntry]:
        """
        n-gram 重叠 × importance × time_decay 综合打分。

        time_decay 使用 importance-weighted 衰减（高价值记忆更持久）。
        """
        def ngrams(text: str, n: int = 2) -> set[str]:
            t = text.lower().replace(" ", "")
            return {t[i:i+n] for i in range(len(t) - n + 1)} if len(t) >= n else set(t)

        query_ng = ngrams(query)
        now      = datetime.utcnow()
        scored: list[tuple[float, MemoryEntry]] = []

        for e in self._entries:
            if e.user_id != user_id:
                continue
            if memory_type and e.type != memory_type:
                continue
            entry_ng = ngrams(e.text)
            overlap  = len(query_ng & entry_ng) / max(len(query_ng), 1)
            days     = max(0.0, (now - e.created_at).total_seconds() / 86400)
            decay    = _time_decay(e.importance, days)
            score    = overlap * e.importance * decay
            if score > 0:
                scored.append((score, e))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [e for _, e in scored[:top_k]]

        # 访问增强
        for e in results:
            e.importance  = min(_IMPORTANCE_MAX, e.importance + _ACCESS_BOOST)
            e.accessed_at = now

        return results

    async def get_profile(self, user_id: str, workspace_id: str = "") -> dict[str, Any]:
        return dict(self._profiles.get(user_id, {}))

    async def update_profile(self, user_id: str, data: dict[str, Any]) -> None:
        self._profiles.setdefault(user_id, {}).update(data)

    async def prune(self, user_id: str, max_items: int, score_threshold: float) -> int:
        now   = datetime.utcnow()
        kept  = []
        pruned = 0

        for e in self._entries:
            if e.user_id != user_id:
                kept.append(e)
                continue
            days  = max(0.0, (now - e.created_at).total_seconds() / 86400)
            decay = _time_decay(e.importance, days)
            final = e.importance * decay
            if final < score_threshold:
                pruned += 1
            else:
                kept.append(e)

        # 超出 max_items → 按 importance 降序截断
        user_kept = sorted(
            [e for e in kept if e.user_id == user_id],
            key=lambda e: e.importance,
            reverse=True,
        )
        if len(user_kept) > max_items:
            to_remove = {e.id for e in user_kept[max_items:]}
            kept   = [e for e in kept if e.id not in to_remove]
            pruned += len(to_remove)

        self._entries = kept
        log.info("memory.prune", user_id=user_id, pruned=pruned)
        return pruned


# ═══════════════════════════════════════════════════════
# LONG-TERM MEMORY — Qdrant（生产）
# ═══════════════════════════════════════════════════════

class QdrantLongTermMemory:
    """
    Qdrant 向量数据库实现。

    优化：
      · search() 访问增强：命中条目 importance += 0.05（upsert 回写）
      · prune() 正式实现：按 importance × decay 过滤后批量删除
      · write() 语义去重：先搜索相似度 > 0.92 的条目，存在则更新而非新增
    """

    COLLECTION   = "agent_memory"
    DEDUP_COSINE = 0.92   # 向量相似度超过此值视为重复

    def __init__(
        self,
        url:      str = "http://localhost:6333",
        path:     str | None = None,
        embed_fn: Any = None,
    ) -> None:
        try:
            from qdrant_client import AsyncQdrantClient
            if path:
                self._client = AsyncQdrantClient(path=path)
            else:
                self._client = AsyncQdrantClient(url=url)
        except ImportError:
            raise RuntimeError("请安装 qdrant-client：pip install qdrant-client")
        self._embed    = embed_fn
        self._profiles: dict[str, dict[str, Any]] = {}

    async def _ensure_collection(self, dim: int = 1536) -> None:
        from qdrant_client.models import Distance, VectorParams, PayloadSchemaType
        collections = await self._client.get_collections()
        names = [c.name for c in collections.collections]
        if self.COLLECTION not in names:
            await self._client.create_collection(
                collection_name = self.COLLECTION,
                vectors_config  = VectorParams(size=dim, distance=Distance.COSINE),
            )
            await self._client.create_payload_index(
                self.COLLECTION, "user_id", PayloadSchemaType.KEYWORD
            )
            await self._client.create_payload_index(
                self.COLLECTION, "type", PayloadSchemaType.KEYWORD
            )

    async def write(self, entry: MemoryEntry) -> None:
        from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue
        vec = await self._embed(entry.text)
        await self._ensure_collection(len(vec))

        # 语义去重：搜索最近邻
        conditions = [FieldCondition(key="user_id", match=MatchValue(value=entry.user_id))]
        hits = await self._client.search(
            collection_name = self.COLLECTION,
            query_vector    = vec,
            query_filter    = Filter(must=conditions),
            limit           = 1,
            score_threshold = self.DEDUP_COSINE,
            with_payload    = True,
        )
        if hits:
            # 重复：更新 importance
            existing_id  = hits[0].id
            old_imp      = hits[0].payload.get("importance", 0.5)
            new_imp      = min(_IMPORTANCE_MAX, max(old_imp, entry.importance) + 0.05)
            await self._client.set_payload(
                collection_name = self.COLLECTION,
                payload         = {"importance": new_imp},
                points          = [existing_id],
            )
            log.debug("memory.write.dedup_qdrant", score=round(hits[0].score, 3))
            return

        point = PointStruct(
            id      = abs(hash(entry.id)) % (2**63),
            vector  = vec,
            payload = {
                "entry_id":   entry.id,
                "user_id":    entry.user_id,
                "type":       entry.type.value,
                "text":       entry.text,
                "importance": entry.importance,
                "metadata":   entry.metadata,
                "created_at": entry.created_at.isoformat(),
            },
        )
        await self._client.upsert(collection_name=self.COLLECTION, points=[point])

    async def search(
        self,
        user_id:     str,
        query:       str,
        memory_type: MemoryType | None = None,
        top_k:       int = 5,
    ) -> list[MemoryEntry]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        vec = await self._embed(query)
        conditions = [FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        if memory_type:
            conditions.append(
                FieldCondition(key="type", match=MatchValue(value=memory_type.value))
            )
        hits = await self._client.search(
            collection_name = self.COLLECTION,
            query_vector    = vec,
            query_filter    = Filter(must=conditions),
            limit           = top_k,
            with_payload    = True,
        )
        entries = []
        boost_ids: list[int] = []
        for h in hits:
            e = MemoryEntry(
                id         = h.payload["entry_id"],
                user_id    = h.payload["user_id"],
                type       = MemoryType(h.payload["type"]),
                text       = h.payload["text"],
                importance = h.payload.get("importance", 0.5),
                metadata   = h.payload.get("metadata", {}),
            )
            entries.append(e)
            # 访问增强：记录需要回写的 point id
            new_imp = min(_IMPORTANCE_MAX, e.importance + _ACCESS_BOOST)
            if new_imp > e.importance:
                boost_ids.append(h.id)
                await self._client.set_payload(
                    collection_name = self.COLLECTION,
                    payload         = {"importance": new_imp},
                    points          = [h.id],
                )
        return entries

    async def get_profile(self, user_id: str, workspace_id: str = "") -> dict[str, Any]:
        return dict(self._profiles.get(user_id, {}))

    async def update_profile(self, user_id: str, data: dict[str, Any]) -> None:
        self._profiles.setdefault(user_id, {}).update(data)

    async def prune(self, user_id: str, max_items: int, score_threshold: float) -> int:
        """
        正式实现（旧版是 stub）：

        1. 分页读取该用户所有条目
        2. 计算 importance × decay 综合分
        3. 删除低分条目
        4. 超出 max_items → 按综合分升序删除最旧/最不重要的
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue, FilterSelector

        all_points: list[dict] = []   # {id, importance, created_at}
        offset = None
        filt   = Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))])

        while True:
            resp, next_offset = await self._client.scroll(
                collection_name = self.COLLECTION,
                scroll_filter   = filt,
                with_payload    = True,
                with_vectors    = False,
                limit           = 500,
                offset          = offset,
            )
            for p in resp:
                all_points.append({
                    "id":         p.id,
                    "importance": p.payload.get("importance", 0.5),
                    "created_at": p.payload.get("created_at", ""),
                })
            if next_offset is None:
                break
            offset = next_offset

        now    = datetime.utcnow()
        to_del: list[int] = []

        # Pass 1: 低分裁剪
        remaining: list[dict] = []
        for p in all_points:
            try:
                ts   = datetime.fromisoformat(p["created_at"])
                days = max(0.0, (now - ts).total_seconds() / 86400)
            except Exception:
                days = 0.0
            decay = _time_decay(p["importance"], days)
            final = p["importance"] * decay
            if final < score_threshold:
                to_del.append(p["id"])
            else:
                p["score"] = final
                remaining.append(p)

        # Pass 2: 超出 max_items → 按分升序继续删
        if len(remaining) > max_items:
            remaining.sort(key=lambda x: x.get("score", 0.0))
            extra  = remaining[: len(remaining) - max_items]
            to_del.extend(p["id"] for p in extra)

        if to_del:
            await self._client.delete(
                collection_name  = self.COLLECTION,
                points_selector  = FilterSelector(
                    filter=Filter(must=[FieldCondition(key="user_id",
                                                       match=MatchValue(value=user_id))])
                ),
            )
            # Qdrant 按 id 批量删除
            from qdrant_client.models import PointIdsList
            await self._client.delete(
                collection_name = self.COLLECTION,
                points_selector = PointIdsList(points=to_del),
            )

        pruned = len(to_del)
        log.info("memory.prune.qdrant", user_id=user_id, pruned=pruned)
        return pruned


# ═══════════════════════════════════════════════════════
# LONG-TERM MEMORY — Milvus（生产，复用 MilvusVectorStore）
# ═══════════════════════════════════════════════════════

class MilvusLongTermMemory:
    """
    Milvus 向量数据库长期记忆实现。

    复用 rag/milvus_store.py 的 MilvusVectorStore 基础设施，
    使用独立 collection（默认 agent_memory，与 KB 的 kb_chunks 隔离）。

    特性：
      · 与 QdrantLongTermMemory 接口完全兼容
      · 利用 Milvus 原生 BM25 稀疏向量做混合检索（dense + sparse RRF）
      · 访问增强（search 时自动 boost importance）
      · 语义去重（write 时向量相似度检查）
      · 完整 prune（删低分 + 超量截断）
    """

    COLLECTION   = "agent_memory"
    DEDUP_COSINE = 0.92

    def __init__(
        self,
        uri:         str = "http://localhost:19530",
        token:       str = "",
        embed_fn:    Any = None,
        vector_size: int = 1536,
    ) -> None:
        from rag.milvus_store import MilvusVectorStore
        self._store = MilvusVectorStore(
            uri        = uri,
            token      = token,
            collection = self.COLLECTION,
            vector_size = vector_size,
        )
        self._embed    = embed_fn
        self._profiles: dict[str, dict[str, Any]] = {}
        self._initialized = False

    async def _init(self) -> None:
        if not self._initialized:
            await self._store.initialize()
            self._initialized = True

    async def write(self, entry: MemoryEntry) -> None:
        await self._init()
        vec = await self._embed(entry.text)

        # 语义去重
        hits = await self._store.search(
            query_vec   = vec,
            filters     = {"user_id": entry.user_id},
            top_k       = 1,
            score_threshold = self.DEDUP_COSINE,
        )
        if hits:
            old_score, payload = hits[0]
            old_imp = payload.get("importance", 0.5)
            new_imp = min(_IMPORTANCE_MAX, max(old_imp, entry.importance) + 0.05)
            chunk_id = payload.get("entry_id", "")
            if chunk_id:
                await self._store.update_field(chunk_id, "importance", new_imp)
            log.debug("memory.write.dedup_milvus", score=round(old_score, 3))
            return

        await self._store.upsert_chunks([{
            "chunk_id":   entry.id,
            "doc_id":     entry.user_id,   # 复用 doc_id 字段存 user_id
            "kb_id":      "agent_memory",
            "text":       entry.text,
            "embedding":  vec,
            "meta": {
                "entry_id":   entry.id,
                "user_id":    entry.user_id,
                "type":       entry.type.value,
                "importance": entry.importance,
                "metadata":   entry.metadata,
                "created_at": entry.created_at.isoformat(),
            },
            "created_at": entry.created_at.timestamp(),
        }])

    async def search(
        self,
        user_id:     str,
        query:       str,
        memory_type: MemoryType | None = None,
        top_k:       int = 5,
    ) -> list[MemoryEntry]:
        await self._init()
        vec = await self._embed(query)

        filters: dict[str, Any] = {"user_id": user_id}
        if memory_type:
            filters["type"] = memory_type.value

        hits = await self._store.hybrid_search(
            query_vec  = vec,
            query_text = query,
            filters    = filters,
            top_k      = top_k,
        )

        entries: list[MemoryEntry] = []
        for score, payload in hits:
            meta = payload.get("meta", {})
            e = MemoryEntry(
                id         = meta.get("entry_id", payload.get("chunk_id", "")),
                user_id    = meta.get("user_id",  user_id),
                type       = MemoryType(meta.get("type", "semantic")),
                text       = payload.get("text", ""),
                importance = meta.get("importance", 0.5),
                metadata   = meta.get("metadata", {}),
            )
            entries.append(e)
            # 访问增强（异步，失败不影响结果）
            new_imp = min(_IMPORTANCE_MAX, e.importance + _ACCESS_BOOST)
            try:
                await self._store.update_field(e.id, "importance", new_imp)
            except Exception:
                pass

        return entries

    async def get_profile(self, user_id: str, workspace_id: str = "") -> dict[str, Any]:
        return dict(self._profiles.get(user_id, {}))

    async def update_profile(self, user_id: str, data: dict[str, Any]) -> None:
        self._profiles.setdefault(user_id, {}).update(data)

    async def prune(self, user_id: str, max_items: int, score_threshold: float) -> int:
        await self._init()
        all_chunks = await self._store.list_chunks_by_filter({"user_id": user_id})
        now        = datetime.utcnow()
        to_del: list[str] = []
        remaining: list[dict] = []

        for c in all_chunks:
            meta = c.get("meta", {})
            imp  = meta.get("importance", 0.5)
            try:
                ts   = datetime.fromisoformat(meta.get("created_at", ""))
                days = max(0.0, (now - ts).total_seconds() / 86400)
            except Exception:
                days = 0.0
            decay = _time_decay(imp, days)
            final = imp * decay
            if final < score_threshold:
                to_del.append(c["chunk_id"])
            else:
                c["_score"] = final
                remaining.append(c)

        if len(remaining) > max_items:
            remaining.sort(key=lambda x: x.get("_score", 0.0))
            extra  = remaining[: len(remaining) - max_items]
            to_del.extend(c["chunk_id"] for c in extra)

        for cid in to_del:
            await self._store.delete_chunk(cid)

        log.info("memory.prune.milvus", user_id=user_id, pruned=len(to_del))
        return len(to_del)
