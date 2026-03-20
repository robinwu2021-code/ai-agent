"""
memory/stores.py — 记忆存储实现

短期记忆：
  - InMemoryShortTermMemory   开发/测试用
  - RedisShortTermMemory      生产用

长期记忆：
  - InMemoryLongTermMemory    开发/测试用
  - QdrantLongTermMemory      生产用（需安装 qdrant-client）
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Any

import structlog

from core.models import AgentTask, MemoryEntry, MemoryType, Message

log = structlog.get_logger(__name__)


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
    """Redis 实现，支持多实例部署，TTL 控制过期。"""

    TASK_TTL = 1800  # 30 分钟

    def __init__(self, redis_url: str = "redis://localhost:6379", ttl: int = TASK_TTL) -> None:
        try:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(redis_url, decode_responses=True)
        except ImportError:
            raise RuntimeError("请安装 redis：pip install redis")
        self._ttl = ttl

    def _task_key(self, task_id: str) -> str:
        return f"agent:task:{task_id}"

    def _msg_key(self, task_id: str) -> str:
        return f"agent:msg:{task_id}"

    def _scratch_key(self, task_id: str) -> str:
        return f"agent:scratch:{task_id}"

    async def save_task(self, task: AgentTask) -> None:
        data = task.model_dump_json()
        await self._redis.setex(self._task_key(task.id), self._ttl, data)

    async def load_task(self, task_id: str) -> AgentTask | None:
        raw = await self._redis.get(self._task_key(task_id))
        return AgentTask.model_validate_json(raw) if raw else None

    async def delete_task(self, task_id: str) -> None:
        await self._redis.delete(
            self._task_key(task_id),
            self._msg_key(task_id),
            self._scratch_key(task_id),
        )

    async def append_message(self, task_id: str, msg: Message) -> None:
        await self._redis.rpush(self._msg_key(task_id), msg.model_dump_json())
        await self._redis.expire(self._msg_key(task_id), self._ttl)

    async def get_messages(self, task_id: str) -> list[Message]:
        raws = await self._redis.lrange(self._msg_key(task_id), 0, -1)
        return [Message.model_validate_json(r) for r in raws]

    async def set_scratchpad(self, task_id: str, key: str, value: Any) -> None:
        await self._redis.hset(self._scratch_key(task_id), key, json.dumps(value))
        await self._redis.expire(self._scratch_key(task_id), self._ttl)

    async def get_scratchpad(self, task_id: str, key: str) -> Any:
        raw = await self._redis.hget(self._scratch_key(task_id), key)
        return json.loads(raw) if raw else None


# ═══════════════════════════════════════════════════════
# LONG-TERM MEMORY
# ═══════════════════════════════════════════════════════

class InMemoryLongTermMemory:
    """纯内存长期记忆，支持简单的字符串相似度检索，适合开发。"""

    def __init__(self) -> None:
        self._entries:  list[MemoryEntry]          = []
        self._profiles: dict[str, dict[str, Any]]  = {}

    async def write(self, entry: MemoryEntry) -> None:
        # 去重：same user + same text → 更新 importance
        for e in self._entries:
            if e.user_id == entry.user_id and e.text == entry.text:
                e.importance  = max(e.importance, entry.importance)
                e.accessed_at = datetime.utcnow()
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
        """字符级 n-gram 重叠打分，兼容中英文（生产请替换为向量检索）。"""
        def ngrams(text: str, n: int = 2) -> set[str]:
            t = text.lower().replace(" ", "")
            return {t[i:i+n] for i in range(len(t) - n + 1)} if len(t) >= n else set(t)

        query_ng = ngrams(query)
        scored: list[tuple[float, MemoryEntry]] = []
        for e in self._entries:
            if e.user_id != user_id:
                continue
            if memory_type and e.type != memory_type:
                continue
            entry_ng = ngrams(e.text)
            overlap  = len(query_ng & entry_ng)
            score    = overlap / max(len(query_ng), 1) * e.importance
            if score > 0:
                scored.append((score, e))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [e for _, e in scored[:top_k]]
        for e in results:
            e.accessed_at = datetime.utcnow()
        return results

    async def get_profile(self, user_id: str) -> dict[str, Any]:
        return dict(self._profiles.get(user_id, {}))

    async def update_profile(self, user_id: str, data: dict[str, Any]) -> None:
        self._profiles.setdefault(user_id, {}).update(data)

    async def prune(self, user_id: str, max_items: int, score_threshold: float) -> int:
        now   = datetime.utcnow()
        kept  = []
        pruned = 0
        for e in self._entries:
            if e.user_id != user_id:
                kept.append(e); continue
            # 时间衰减：每天衰减 5%
            days  = (now - e.created_at).days
            decay = math.exp(-0.05 * days)
            final_score = e.importance * decay
            if final_score < score_threshold:
                pruned += 1
            else:
                kept.append(e)
        # 超出 max_items 时按分数排序截断
        user_entries = sorted(
            [e for e in kept if e.user_id == user_id],
            key=lambda e: e.importance, reverse=True,
        )
        if len(user_entries) > max_items:
            to_remove = {e.id for e in user_entries[max_items:]}
            kept  = [e for e in kept if e.id not in to_remove]
            pruned += len(to_remove)
        self._entries = kept
        log.info("memory.prune", user_id=user_id, pruned=pruned)
        return pruned


class QdrantLongTermMemory:
    """Qdrant 向量数据库实现，适合生产环境。"""

    COLLECTION = "agent_memory"

    def __init__(
        self,
        url: str = "http://localhost:6333",
        embed_fn: Any = None,  # LLMEngine.embed 函数
    ) -> None:
        try:
            from qdrant_client import AsyncQdrantClient
            from qdrant_client.models import Distance, VectorParams
            self._client     = AsyncQdrantClient(url=url)
            self._Distance   = Distance
            self._VectorParams = VectorParams
        except ImportError:
            raise RuntimeError("请安装 qdrant-client：pip install qdrant-client")
        self._embed    = embed_fn
        self._profiles: dict[str, dict[str, Any]] = {}  # 简单存内存，生产可改 Redis

    async def _ensure_collection(self, dim: int = 1536) -> None:
        from qdrant_client.models import Distance, VectorParams
        collections = await self._client.get_collections()
        names = [c.name for c in collections.collections]
        if self.COLLECTION not in names:
            await self._client.create_collection(
                collection_name=self.COLLECTION,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    async def write(self, entry: MemoryEntry) -> None:
        from qdrant_client.models import PointStruct
        vec = await self._embed(entry.text)
        await self._ensure_collection(len(vec))
        point = PointStruct(
            id=abs(hash(entry.id)) % (2**63),
            vector=vec,
            payload={
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
        user_id: str,
        query: str,
        memory_type: MemoryType | None = None,
        top_k: int = 5,
    ) -> list[MemoryEntry]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        vec = await self._embed(query)
        conditions = [FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        if memory_type:
            conditions.append(FieldCondition(key="type", match=MatchValue(value=memory_type.value)))
        hits = await self._client.search(
            collection_name=self.COLLECTION,
            query_vector=vec,
            query_filter=Filter(must=conditions),
            limit=top_k,
        )
        return [
            MemoryEntry(
                id=h.payload["entry_id"],
                user_id=h.payload["user_id"],
                type=MemoryType(h.payload["type"]),
                text=h.payload["text"],
                importance=h.payload.get("importance", 0.5),
                metadata=h.payload.get("metadata", {}),
            )
            for h in hits
        ]

    async def get_profile(self, user_id: str) -> dict[str, Any]:
        return dict(self._profiles.get(user_id, {}))

    async def update_profile(self, user_id: str, data: dict[str, Any]) -> None:
        self._profiles.setdefault(user_id, {}).update(data)

    async def prune(self, user_id: str, max_items: int, score_threshold: float) -> int:
        # 生产实现：按 importance 过滤后删除低分条目
        log.info("QdrantLongTermMemory.prune: not fully implemented in this stub")
        return 0
