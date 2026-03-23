"""
memory/ltm/mem0_ltm.py — mem0 长期记忆适配器

将 mem0 接入现有 LLMRouter，所有 LLM 调用（提取/去重判断/摘要）和
向量化操作全部路由到 llm.yaml engines 节中已定义的引擎实例，
通过 llm_engine / embed_engine alias 一行切换后端。

切换示例（llm.yaml memory.ltm）：
  本地 Ollama:    llm_engine: ollama-qwen3,  embed_engine: ollama-embed
  本地 vLLM:      llm_engine: vllm-qwen3,    embed_engine: ollama-embed
  云端 Claude:    llm_engine: claude-fast,   embed_engine: (Anthropic 无 embed → Ollama)
  云端 OpenAI:    llm_engine: gpt-4o-mini,   embed_engine: openai-embed

RouterLLMBridge / RouterEmbedBridge 是 sync 桥接层：
  · mem0 内部接口是 sync（generate_response / embed）
  · 我们的 LLMRouter 是 async
  · 桥接通过 run_coroutine_threadsafe 提交到 router 所在的事件循环，
    不创建新 loop，不影响 FastAPI 主循环。
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from typing import Any

import structlog

from memory.protocols import ScoredMemory

log = structlog.get_logger(__name__)

# ─────────────────────────────────────────────────────────────────
# 专用后台事件循环（sync → async 桥接专用）
# ─────────────────────────────────────────────────────────────────

_bridge_loop: asyncio.AbstractEventLoop | None = None
_bridge_lock = threading.Lock()


def _get_bridge_loop() -> asyncio.AbstractEventLoop:
    """
    返回一个持久运行的后台事件循环，供 sync 桥接层调用 async 方法使用。

    设计原因：
      · FastAPI/uvicorn 已有一个主事件循环（asyncio.get_running_loop()）
      · sync 函数（mem0 接口）不能直接 await，也不能 asyncio.run()（会报循环冲突）
      · 在独立后台线程中运行一个专用循环，通过 run_coroutine_threadsafe 跨线程提交任务
      · 后台循环与 router 的 HTTP 连接池共享（因为 router 也被传入 bridge），所以
        所有 HTTP 请求统一在这个循环里发出，连接可以复用
    """
    global _bridge_loop
    with _bridge_lock:
        if _bridge_loop is None or not _bridge_loop.is_running():
            _bridge_loop = asyncio.new_event_loop()
            t = threading.Thread(
                target=_bridge_loop.run_forever,
                name="memory-bridge-loop",
                daemon=True,
            )
            t.start()
            log.debug("memory.bridge_loop.started")
    return _bridge_loop


def _run_async(coro: Any, timeout: float = 120.0) -> Any:
    """把 coroutine 提交到后台桥接循环并等待结果（可从任意线程/同步函数调用）。"""
    loop   = _get_bridge_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=timeout)


# ─────────────────────────────────────────────────────────────────
# RouterLLMBridge — 实现 mem0 BaseLlm 接口，内部走 LLMRouter
# ─────────────────────────────────────────────────────────────────

class RouterLLMBridge:
    """
    mem0 `BaseLlm` 适配器。

    mem0 调用 generate_response(messages: list[dict]) → str
    我们把它桥接到 router._get_engine(alias).chat() 或 router.chat()。

    llm_engine_alias:
      "" 或 None → 使用 router 的 default 引擎（router.chat() with no override）
      "ollama-qwen3" → 强制使用该引擎（AgentConfig.model_alias）
      "vllm-qwen3"   → 切换到 vLLM
      "claude-fast"  → 切换到 Claude Haiku
    """

    config: dict  # mem0 需要这个属性（duck-typing）

    def __init__(self, router: Any, engine_alias: str = "") -> None:
        self._router = router
        self._alias  = engine_alias or ""
        self.config  = {"model": engine_alias or "router-default"}

    def generate_response(
        self,
        messages:        list[dict],
        response_format: dict | None = None,
        tools:           list[dict] | None = None,
        tool_choice:     str = "auto",
    ) -> str:
        return _run_async(self._async_generate(messages))

    async def _async_generate(self, messages: list[dict]) -> str:
        from core.models import AgentConfig, Message, MessageRole

        msgs: list[Message] = []
        for m in messages:
            role    = m.get("role", "user")
            content = m.get("content", "")
            try:
                msgs.append(Message(role=MessageRole(role), content=content))
            except ValueError:
                msgs.append(Message(role=MessageRole.USER, content=content))

        config = AgentConfig(
            max_steps    = 1,
            timeout_ms   = 90_000,
            model_alias  = self._alias or None,
        )
        resp = await self._router.chat(msgs, tools=[], config=config)
        return resp.content or ""


# ─────────────────────────────────────────────────────────────────
# RouterEmbedBridge — 实现 mem0 BaseEmbedder 接口，内部走 LLMRouter
# ─────────────────────────────────────────────────────────────────

class RouterEmbedBridge:
    """
    mem0 `BaseEmbedder` 适配器。

    mem0 调用 embed(text: str) → list[float]
    我们把它桥接到 router.embed()，可指定 embed_engine_alias 决定用哪个 embedding 引擎。

    embed_engine_alias:
      "" 或 None → 使用 router 的 embed 路由（llm.yaml router.embed）
      "ollama-embed" → 强制用 Ollama qwen3-embedding:8b
      "openai-embed" → 切换到 OpenAI text-embedding-3-small
    """

    config: dict

    def __init__(self, router: Any, engine_alias: str = "") -> None:
        self._router = router
        self._alias  = engine_alias or ""
        # 尝试从 router 读取 embedding 维度（用于 mem0 collection 初始化）
        self._dims   = self._probe_dims()
        self.config  = {"model": engine_alias or "router-embed", "embedding_dims": self._dims}

    def _probe_dims(self) -> int:
        """尝试用已知文本测量 embedding 维度，失败则返回 1536。"""
        try:
            vec = _run_async(self._async_embed("test"), timeout=30.0)
            return len(vec)
        except Exception:
            return 1536

    def embed(self, text: str, memory_action: str | None = None) -> list[float]:
        return _run_async(self._async_embed(text))

    async def _async_embed(self, text: str) -> list[float]:
        # 若指定了 embed 引擎 alias，临时切换（利用 model_alias 传递给 router）
        if self._alias:
            # 直接取特定引擎实例
            registry = getattr(self._router, "_registry", None)
            if registry:
                engine = registry.get(self._alias)
                if engine:
                    return await engine.embed(text)
        # 走 router 默认 embed 路由
        return await self._router.embed(text)


# ─────────────────────────────────────────────────────────────────
# Mem0LongTermMemory — 主类
# ─────────────────────────────────────────────────────────────────

class Mem0LongTermMemory:
    """
    基于 mem0 的长期记忆实现。

    特性：
      · LLM 和 Embedding 全部通过 RouterLLMBridge/RouterEmbedBridge 路由
        → 只改 llm.yaml memory.ltm.llm_engine 即可切换 Ollama/vLLM/Claude
      · 向量库复用全局 vector_store 配置（Milvus/Qdrant），独立 collection
      · 自动语义去重（mem0 内置）
      · 自动提取结构化记忆（mem0 内置 LLM pipeline）
      · workspace 多租户：user_id = "scope::ws_id::user_id" 命名规范
    """

    def __init__(
        self,
        router:          Any,
        llm_alias:       str = "",
        embed_alias:     str = "",
        vector_store_cfg: Any = None,   # VectorStoreConfig 或 None（用内置 Qdrant）
        collection:      str = "agent_memory",
        dedup_threshold: float = 0.85,
        max_memories:    int = 10_000,
    ) -> None:
        self._router       = router
        self._llm_alias    = llm_alias
        self._embed_alias  = embed_alias
        self._vs_cfg       = vector_store_cfg
        self._collection   = collection
        self._dedup        = dedup_threshold
        self._max_memories = max_memories
        self._profiles: dict[str, dict] = {}
        self._mem: Any = None   # mem0.AsyncMemory, lazy init

    # ── 初始化（lazy, 首次使用时）─────────────────────────────────

    async def _get_mem(self) -> Any:
        if self._mem is not None:
            return self._mem

        try:
            from mem0 import AsyncMemory
        except ImportError:
            raise RuntimeError(
                "请安装 mem0：pip install mem0ai\n"
                "  或禁用 mem0 后端：llm.yaml memory.ltm.backend 改为 milvus / in_memory"
            )

        llm_bridge   = RouterLLMBridge(self._router, self._llm_alias)
        embed_bridge = RouterEmbedBridge(self._router, self._embed_alias)
        dims         = embed_bridge._dims

        # 向量库配置
        vector_store_conf = self._build_vector_store_conf(dims)

        config = {
            "llm": {
                "provider": "custom",
                "config":   {"custom_llm_class": llm_bridge},
            },
            "embedder": {
                "provider": "custom",
                "config":   {"custom_embedder_class": embed_bridge},
            },
            "vector_store": vector_store_conf,
        }

        self._mem = await AsyncMemory.from_config_async(config)
        log.info(
            "mem0.initialized",
            llm_alias   = self._llm_alias   or "router-default",
            embed_alias = self._embed_alias or "router-embed",
            collection  = self._collection,
            dims        = dims,
        )
        return self._mem

    def _build_vector_store_conf(self, dims: int) -> dict:
        """根据全局 VectorStoreConfig 或默认值构建 mem0 vector_store 配置字典。"""
        vs = self._vs_cfg

        if vs is None:
            # 无配置：mem0 使用内置 in-memory 向量库（仅开发用）
            return {"provider": "memory", "config": {}}

        if vs.backend == "milvus":
            return {
                "provider": "milvus",
                "config": {
                    "uri":              vs.milvus_uri,
                    "token":            vs.milvus_token or "",
                    "collection_name":  self._collection,
                    "embedding_model_dims": dims,
                    "metric_type":      "COSINE",
                    "index_type":       vs.milvus_index_type,
                },
            }

        if vs.backend == "qdrant":
            conf: dict = {
                "collection_name": self._collection,
                "embedding_model_dims": dims,
            }
            if vs.qdrant_url:
                conf["url"]     = vs.qdrant_url
                conf["api_key"] = vs.qdrant_api_key or ""
            else:
                conf["path"] = vs.qdrant_path
            return {"provider": "qdrant", "config": conf}

        # sqlite fallback → mem0 in-memory
        return {"provider": "memory", "config": {}}

    # ── 公共接口 ──────────────────────────────────────────────────

    async def write(self, entry: Any) -> None:
        """写入 MemoryEntry（兼容旧接口）。"""
        mem    = await self._get_mem()
        text   = getattr(entry, "text", str(entry))
        uid    = getattr(entry, "user_id", "global")
        meta   = getattr(entry, "metadata", {}) or {}
        meta["importance"] = getattr(entry, "importance", 0.5)
        meta["type"]       = getattr(entry, "type", "semantic")
        if hasattr(entry, "type") and hasattr(entry.type, "value"):
            meta["type"] = entry.type.value
        await mem.add(text, user_id=uid, metadata=meta)

    async def search(
        self,
        user_id:  str,
        query:    str,
        top_k:    int  = 5,
        filters:  dict = {},
    ) -> list[ScoredMemory]:
        mem  = await self._get_mem()
        raw  = await mem.search(query=query, user_id=user_id, limit=top_k, filters=filters)
        results = raw.get("results", raw) if isinstance(raw, dict) else raw
        out: list[ScoredMemory] = []
        for r in results:
            meta = r.get("metadata") or {}
            out.append(ScoredMemory(
                id          = r.get("id", ""),
                user_id     = user_id,
                text        = r.get("memory", r.get("text", "")),
                score       = float(r.get("score", 0.0)),
                importance  = float(meta.get("importance", 0.5)),
                memory_type = str(meta.get("type", "semantic")),
                scope       = str(meta.get("scope", "personal")),
                metadata    = meta,
            ))
        return out

    async def get_profile(self, user_id: str, workspace_id: str = "") -> dict:
        return dict(self._profiles.get(user_id, {}))

    async def update_profile(self, user_id: str, data: dict) -> None:
        self._profiles.setdefault(user_id, {}).update(data)

    async def prune(self, user_id: str, max_items: int, score_threshold: float) -> int:
        """mem0 自带去重，此处仅做超量截断。"""
        mem = await self._get_mem()
        try:
            all_mems = await mem.get_all(user_id=user_id)
            items = all_mems.get("results", all_mems) if isinstance(all_mems, dict) else all_mems
            if len(items) <= max_items:
                return 0
            # 按 score 升序，删除最弱的
            items.sort(key=lambda x: float((x.get("metadata") or {}).get("importance", 0.5)))
            to_del = items[: len(items) - max_items]
            for item in to_del:
                await mem.delete(memory_id=item["id"])
            log.info("mem0.prune", user_id=user_id, pruned=len(to_del))
            return len(to_del)
        except Exception as exc:
            log.warning("mem0.prune.failed", error=str(exc))
            return 0

    # ── workspace 多租户扩展（兼容 WorkspaceAwareLTM 接口）──────

    async def write_scoped(
        self,
        scope:        Any,
        author_id:    str,
        text:         str,
        workspace_id: str | None = None,
        project_id:   str | None = None,
        memory_type:  str = "semantic",
        importance:   float = 0.5,
        metadata:     dict | None = None,
    ) -> Any:
        """workspace 感知写入：user_id 编码为 scope::ws::proj 格式。"""
        scope_str = scope.value if hasattr(scope, "value") else str(scope)
        uid = self._scoped_uid(scope_str, author_id, workspace_id, project_id)
        mem = await self._get_mem()
        meta = (metadata or {}) | {"importance": importance, "type": memory_type,
                                    "scope": scope_str, "author_id": author_id}
        result = await mem.add(text, user_id=uid, metadata=meta)
        return result

    async def search_mixed(
        self,
        query:        str,
        user_id:      str,
        workspace_id: str | None = None,
        project_id:   str | None = None,
        top_k:        int = 10,
    ) -> list[Any]:
        """
        4 层并发检索，模拟 WorkspaceAwareLTM.search_mixed。
        每层用不同 user_id（作用域编码），加权后合并。
        """
        import asyncio as _aio

        layers = [
            (user_id,                                                          1.00),
            (self._scoped_uid("personal",  user_id, workspace_id, None),      0.95),
            (self._scoped_uid("project",   user_id, workspace_id, project_id), 0.90),
            (self._scoped_uid("workspace", user_id, workspace_id, None),       0.70),
        ]
        # 过滤重复 uid
        seen:    set[str] = set()
        unique   = [(uid, w) for uid, w in layers if uid not in seen and not seen.add(uid)]  # type: ignore

        tasks = [self.search(uid, query, top_k=top_k // 2 + 2) for uid, _ in unique]
        results = await _aio.gather(*tasks, return_exceptions=True)

        # 加权合并，去重
        merged: dict[str, Any] = {}
        for (uid, weight), layer_results in zip(unique, results):
            if isinstance(layer_results, Exception):
                continue
            for r in layer_results:
                if r.id not in merged or merged[r.id].score < r.score * weight:
                    r.score *= weight
                    merged[r.id] = r

        ranked = sorted(merged.values(), key=lambda x: x.score, reverse=True)
        return ranked[:top_k]

    @staticmethod
    def _scoped_uid(
        scope:        str,
        user_id:      str,
        workspace_id: str | None,
        project_id:   str | None,
    ) -> str:
        """将作用域信息编码为 mem0 user_id 字符串。"""
        if scope == "workspace" and workspace_id:
            return f"ws::{workspace_id}"
        if scope == "project" and workspace_id and project_id:
            return f"proj::{workspace_id}::{project_id}"
        if workspace_id:
            return f"personal::{workspace_id}::{user_id}"
        return user_id
