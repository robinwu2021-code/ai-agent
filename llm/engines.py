"""
llm/engines.py — LLM Engine 实现

所有连接信息、模型名均通过构造函数参数传入，不读取任何全局变量或环境变量。
通常由 utils/llm_config.py 的 LLMConfig.build_engine() 实例化。

包含：
  AnthropicEngine   Claude 系列
  OpenAIEngine      OpenAI / Azure / DeepSeek / Groq / Ollama / 任意兼容接口
  MockLLMEngine     测试用
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator

import structlog

from core.models import (
    AgentConfig, LLMResponse, Message, MessageRole,
    ToolCall, ToolDescriptor,
)
from llm.call_logger import get_call_logger

log = structlog.get_logger(__name__)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _messages_to_anthropic(messages: list[Message]) -> tuple[str | None, list[dict]]:
    system = None
    result = []
    for msg in messages:
        if msg.role == MessageRole.SYSTEM:
            system = msg.content
            continue
        if msg.role == MessageRole.TOOL:
            result.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": msg.tool_result.tool_call_id,
                    "content": json.dumps(msg.tool_result.content, ensure_ascii=False),
                }],
            })
            continue
        if msg.tool_calls:
            result.append({
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": tc.id,
                     "name": tc.tool_name, "input": tc.arguments}
                    for tc in msg.tool_calls
                ],
            })
            continue
        result.append({"role": msg.role.value, "content": msg.content or ""})
    return system, result


def _descriptors_to_anthropic(tools: list[ToolDescriptor]) -> list[dict]:
    return [{"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in tools]


# ─────────────────────────────────────────────
# Anthropic Engine
# ─────────────────────────────────────────────

class AnthropicEngine:
    """
    Claude 系列模型。
    所有连接参数通过构造函数传入，不读取任何全局变量或环境变量。
    """

    def __init__(
        self,
        api_key:        str | None = None,
        default_model:  str        = "claude-sonnet-4-20250514",
        summarize_model: str | None = None,
        max_tokens:     int        = 4096,
        timeout_sec:    float      = 120.0,
        max_retries:    int        = 3,
        http_proxy:     str | None = None,
        alias:          str        = "anthropic",   # engine alias for logging
    ):
        try:
            import anthropic
            import httpx

            transport = httpx.AsyncHTTPTransport(proxy=http_proxy) if http_proxy else None
            http_client = httpx.AsyncClient(transport=transport) if transport else None

            self._client = anthropic.AsyncAnthropic(
                api_key=api_key,
                http_client=http_client,
                max_retries=max_retries,
                timeout=timeout_sec,
            )
        except ImportError:
            raise RuntimeError("请安装 anthropic：pip install anthropic")

        self._default_model   = default_model
        self._summarize_model = summarize_model or default_model
        self._max_tokens      = max_tokens
        self._alias           = alias   # stored for call_logger

        log.info("anthropic_engine.init",
                 alias=alias,
                 model=default_model,
                 summarize_model=self._summarize_model,
                 max_tokens=max_tokens,
                 proxy=http_proxy or "(none)")

    def _resolve_model(self, config: AgentConfig) -> str:
        """AgentConfig.model 优先，其次用引擎默认值。"""
        return config.model if config.model else self._default_model

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDescriptor],
        config: AgentConfig,
    ) -> LLMResponse:
        model = self._resolve_model(config)
        clog  = get_call_logger()
        t0    = clog.log_request("anthropic", self._alias, model, "chat",
                                  messages, tools, config)

        system, api_msgs = _messages_to_anthropic(messages)
        api_tools = _descriptors_to_anthropic(tools)
        kwargs: dict[str, Any] = {
            "model":      model,
            "max_tokens": self._max_tokens,
            "messages":   api_msgs,
        }
        if system:    kwargs["system"] = system
        if api_tools: kwargs["tools"]  = api_tools

        try:
            resp = await self._client.messages.create(**kwargs)
        except Exception as exc:
            clog.log_error("anthropic", self._alias, model, "chat", t0, exc)
            raise

        tool_calls, content_text = [], None
        for block in resp.content:
            if block.type == "text":
                content_text = block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, tool_name=block.name,
                                           arguments=block.input or {}))
        result = LLMResponse(
            content=content_text, tool_calls=tool_calls,
            usage={"prompt_tokens": resp.usage.input_tokens,
                   "completion_tokens": resp.usage.output_tokens},
            model=resp.model,
        )
        clog.log_response("anthropic", self._alias, model, "chat", t0, result)
        return result

    async def stream_chat(
        self,
        messages: list[Message],
        tools: list[ToolDescriptor],
        config: AgentConfig,
    ) -> AsyncIterator[str]:
        model = self._resolve_model(config)
        clog  = get_call_logger()
        t0    = clog.log_stream_start("anthropic", self._alias, model, "stream_chat",
                                       messages, tools, config)

        system, api_msgs = _messages_to_anthropic(messages)
        api_tools = _descriptors_to_anthropic(tools)
        kwargs: dict[str, Any] = {
            "model":      model,
            "max_tokens": self._max_tokens,
            "messages":   api_msgs,
        }
        if system:    kwargs["system"] = system
        if api_tools: kwargs["tools"]  = api_tools

        total_chars = 0
        try:
            async with self._client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    total_chars += len(text)
                    yield text
        except Exception as exc:
            clog.log_error("anthropic", self._alias, model, "stream_chat", t0, exc)
            raise
        finally:
            clog.log_stream_end("anthropic", self._alias, model, "stream_chat",
                                 t0, total_chars)

    async def embed(self, text: str) -> list[float]:
        # Anthropic 无 embedding API，回退哈希向量（生产请换 OpenAI embedding）
        log.warning("anthropic_engine.embed_fallback")
        import hashlib
        h = hashlib.md5(text.encode()).digest()
        return [b / 255.0 for b in h] * 96  # 1536-dim

    async def summarize(self, text: str, max_tokens: int) -> str:
        prompt = f"请将以下内容压缩到约 {max_tokens} token 以内，保留核心信息：\n\n{text}"
        resp = await self._client.messages.create(
            model=self._summarize_model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text if resp.content else text[:max_tokens * 3]


# ─────────────────────────────────────────────
# OpenAI-Compatible Engine
# ─────────────────────────────────────────────

class OpenAIEngine:
    """
    OpenAI / Azure / DeepSeek / Groq / Ollama 及任意兼容接口。
    通过 base_url 切换到不同提供商，is_azure=True 时启用 Azure 认证。
    """

    def __init__(
        self,
        api_key:         str | None = None,
        base_url:        str | None = None,
        default_model:   str        = "gpt-4o",
        embedding_model: str        = "text-embedding-3-small",
        summarize_model: str | None = None,
        max_tokens:      int        = 4096,
        timeout_sec:     float      = 120.0,
        max_retries:     int        = 3,
        http_proxy:      str | None = None,
        # Azure 专用
        is_azure:        bool       = False,
        azure_api_version: str      = "2024-02-01",
        alias:           str        = "openai",   # engine alias for logging
    ):
        try:
            import httpx
            proxy_cfg = {"proxy": http_proxy} if http_proxy else {}

            if is_azure:
                from openai import AsyncAzureOpenAI
                self._client = AsyncAzureOpenAI(
                    api_key=api_key,
                    azure_endpoint=base_url or "",
                    api_version=azure_api_version,
                    max_retries=max_retries,
                    timeout=timeout_sec,
                    http_client=httpx.AsyncClient(**proxy_cfg) if http_proxy else None,
                )
            else:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(
                    api_key=api_key or "ollama",  # Ollama 本地不需要真实 key
                    base_url=base_url,
                    max_retries=max_retries,
                    timeout=timeout_sec,
                    http_client=httpx.AsyncClient(**proxy_cfg) if http_proxy else None,
                )
        except ImportError:
            raise RuntimeError("请安装 openai：pip install openai")

        self._default_model   = default_model
        self._embedding_model = embedding_model
        self._summarize_model = summarize_model or default_model
        self._max_tokens      = max_tokens
        self._alias           = alias   # stored for call_logger

        log.info("openai_engine.init",
                 alias=alias,
                 base_url=base_url or "(default)",
                 model=default_model,
                 embedding_model=embedding_model,
                 summarize_model=self._summarize_model,
                 is_azure=is_azure,
                 proxy=http_proxy or "(none)")

    def _resolve_model(self, config: AgentConfig) -> str:
        return config.model if config.model else self._default_model

    def _to_openai_messages(self, messages: list[Message]) -> list[dict]:
        result = []
        for msg in messages:
            if msg.role == MessageRole.TOOL:
                result.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_result.tool_call_id,
                    "content": json.dumps(msg.tool_result.content, ensure_ascii=False),
                })
                continue
            if msg.tool_calls:
                result.append({
                    "role": "assistant", "content": None,
                    "tool_calls": [
                        {"id": tc.id, "type": "function",
                         "function": {"name": tc.tool_name,
                                      "arguments": json.dumps(tc.arguments)}}
                        for tc in msg.tool_calls
                    ],
                })
                continue
            result.append({"role": msg.role.value, "content": msg.content or ""})
        return result

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDescriptor],
        config: AgentConfig,
    ) -> LLMResponse:
        model = self._resolve_model(config)
        clog  = get_call_logger()
        t0    = clog.log_request("openai", self._alias, model, "chat",
                                  messages, tools, config)

        api_msgs  = self._to_openai_messages(messages)
        api_tools = [
            {"type": "function", "function": {
                "name": t.name, "description": t.description,
                "parameters": t.input_schema,
            }} for t in tools
        ] if tools else []

        kwargs: dict[str, Any] = {
            "model":      model,
            "messages":   api_msgs,
            "max_tokens": self._max_tokens,
        }
        if api_tools: kwargs["tools"] = api_tools

        try:
            resp = await self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            clog.log_error("openai", self._alias, model, "chat", t0, exc)
            raise

        choice = resp.choices[0].message
        tool_calls = []
        if choice.tool_calls:
            for tc in choice.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id, tool_name=tc.function.name,
                    arguments=json.loads(tc.function.arguments or "{}"),
                ))

        # openai SDK v2: usage 在部分兼容接口中可能为 None
        usage = resp.usage
        result = LLMResponse(
            content=choice.content, tool_calls=tool_calls,
            usage={
                "prompt_tokens":     getattr(usage, "prompt_tokens", 0) if usage else 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
            },
            model=resp.model,
        )
        clog.log_response("openai", self._alias, model, "chat", t0, result)
        return result

    async def stream_chat(
        self,
        messages: list[Message],
        tools: list[ToolDescriptor],
        config: AgentConfig,
    ) -> AsyncIterator[str]:
        model = self._resolve_model(config)
        clog  = get_call_logger()
        t0    = clog.log_stream_start("openai", self._alias, model, "stream_chat",
                                       messages, tools, config)

        api_msgs = self._to_openai_messages(messages)
        total_chars = 0
        try:
            stream = await self._client.chat.completions.create(
                model=model,
                messages=api_msgs,
                max_tokens=self._max_tokens,
                stream=True,
            )
            async for chunk in stream:
                # openai SDK v2: 最后一个 chunk 的 choices 可能为空列表
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    total_chars += len(delta)
                    yield delta
        except Exception as exc:
            clog.log_error("openai", self._alias, model, "stream_chat", t0, exc)
            raise
        finally:
            clog.log_stream_end("openai", self._alias, model, "stream_chat",
                                 t0, total_chars)

    async def embed(self, text: str) -> list[float]:
        import time

        if not self._embedding_model:
            raise RuntimeError(
                f"[{self._alias}] embed_engine 未配置 embedding_model。"
                "请在 llm.yaml 对应引擎中设置 embedding_model 字段，"
                "或在 kb_config.yaml 中指定 embed_engine。"
            )

        t0 = time.perf_counter()
        log.debug(
            "openai_engine.embed.request",
            model=self._embedding_model,
            text_len=len(text),
            text_preview=text[:120],
        )

        try:
            resp = await self._client.embeddings.create(
                model=self._embedding_model, input=text
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            vec = resp.data[0].embedding
            log.debug(
                "openai_engine.embed.response",
                model=self._embedding_model,
                dims=len(vec),
                elapsed_ms=round(elapsed_ms, 1),
                usage_tokens=getattr(resp.usage, "total_tokens", None),
                vec_preview=[round(v, 6) for v in vec[:6]],
            )
            return vec

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            log.error(
                "openai_engine.embed.failed",
                model=self._embedding_model,
                elapsed_ms=round(elapsed_ms, 1),
                error=str(exc),
                error_type=type(exc).__name__,
            )
            raise

    async def summarize(self, text: str, max_tokens: int) -> str:
        resp = await self._client.chat.completions.create(
            model=self._summarize_model,
            max_tokens=max_tokens,
            messages=[{"role": "user",
                       "content": f"压缩到 {max_tokens} token 以内，保留核心：\n\n{text}"}],
        )
        return resp.choices[0].message.content or text[:max_tokens * 3]


# ─────────────────────────────────────────────
# Mock Engine（测试）
# ─────────────────────────────────────────────

class MockLLMEngine:
    def __init__(self, responses: list[LLMResponse] | None = None):
        self._responses = responses or []
        self._idx = 0

    def _next(self) -> LLMResponse:
        if not self._responses:
            return LLMResponse(content="Mock response",
                               usage={"prompt_tokens": 10, "completion_tokens": 5})
        resp = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        return resp

    async def chat(self, messages, tools, config) -> LLMResponse:
        return self._next()

    async def stream_chat(self, messages, tools, config) -> AsyncIterator[str]:
        resp = self._next()
        for ch in (resp.content or ""):
            yield ch

    async def embed(self, text: str) -> list[float]:
        return [0.0] * 1536

    async def summarize(self, text: str, max_tokens: int) -> str:
        return text[:max_tokens * 3]


# ─────────────────────────────────────────────
# LocalTransformersEngine — 本地 Python 推理（仅 Embedding）
# ─────────────────────────────────────────────

class LocalTransformersEngine:
    """
    通过 transformers 库在本地 Python 进程中直接推理 Embedding 模型。

    仅实现 embed()，chat / stream_chat / summarize 均报错（该引擎不支持推理）。
    由 LLMConfig(sdk="local_transformers").build_engine() 实例化。

    首次调用 embed() 时懒加载模型（触发 transformers AutoModel.from_pretrained），
    后续调用复用已加载的模型权重，无额外 IO 开销。
    """

    def __init__(
        self,
        model:      str,
        device:     str = "cpu",
        batch_size: int = 8,
        max_length: int = 512,
        dimensions: int = 0,
        alias:      str = "local-transformers",
    ) -> None:
        self._alias      = alias
        self._model_path = model
        self._device     = device
        self._batch_size = batch_size
        self._max_length = max_length
        self._dimensions = dimensions
        self._embedder   = None   # 懒加载，首次 embed() 时初始化

    @property
    def dimensions(self) -> int:
        """向量维度（配置值；模型加载后由 Qwen3LocalEmbedder 自动修正）。"""
        if self._embedder is not None:
            return self._embedder.dimensions
        return self._dimensions

    def _get_embedder(self):
        if self._embedder is None:
            from rag.embedders.qwen3_local_embedder import Qwen3LocalEmbedder
            log.info(
                "local_transformers_engine.loading",
                alias      = self._alias,
                model      = self._model_path,
                device     = self._device,
                dimensions = self._dimensions,
            )
            self._embedder = Qwen3LocalEmbedder(
                model      = self._model_path,
                device     = self._device,
                batch_size = self._batch_size,
                max_length = self._max_length,
                dimensions = self._dimensions,
            )
        return self._embedder

    async def embed(self, text: str) -> list[float]:
        return await self._get_embedder().embed(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return await self._get_embedder().embed_batch(texts)

    # ── 以下方法仅给出明确错误，不做任何降级 ──────────────────────

    async def chat(self, messages, tools=None, config=None) -> LLMResponse:
        raise RuntimeError(
            f"[{self._alias}] sdk=local_transformers 仅支持 Embedding，"
            "不能用于 chat 推理。请在 router.chat 中使用其他引擎（如 minimax / ollama-qwen3）。"
        )

    async def stream_chat(self, messages, tools=None, config=None) -> AsyncIterator[str]:
        raise RuntimeError(
            f"[{self._alias}] sdk=local_transformers 仅支持 Embedding，"
            "不能用于 stream_chat 推理。"
        )
        # 让 Python 识别为 AsyncGenerator
        yield  # type: ignore[misc]  # pragma: no cover

    async def summarize(self, text: str, max_tokens: int = 512) -> str:
        raise RuntimeError(
            f"[{self._alias}] sdk=local_transformers 仅支持 Embedding，"
            "不能用于 summarize。"
        )
