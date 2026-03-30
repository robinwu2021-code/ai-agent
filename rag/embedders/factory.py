"""
rag/embedders/factory.py — Embedder 工厂

优先从 llm.yaml 解析 embed engine（通过 LLMConfig）；
本地 BGE 为例外，直接使用 KBLLMConfig.bge_local 配置。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from rag.embedders.base import BaseEmbedder

if TYPE_CHECKING:
    from utils.llm_config import LLMConfig
    from rag.config import KBLLMConfig

log = structlog.get_logger(__name__)


class EmbedderFactory:

    @staticmethod
    def create_from_llm_config(
        llm_cfg: "LLMConfig",
        dimensions: int = 1536,
        batch_size: int = 32,
    ) -> BaseEmbedder:
        """
        从 llm.yaml LLMConfig 实例创建 Embedder。

        所有 openai_compatible 引擎均使用 OpenAIEmbedder（兼容 Ollama / vLLM / Azure
        / DashScope 等所有 OpenAI 兼容接口）。
        Anthropic 引擎无原生 Embedding API，降级到 hash fallback。
        """
        embed_model = llm_cfg.resolved_embedding_model()
        if not embed_model:
            log.warning(
                "embedder_factory.no_embed_model",
                alias=llm_cfg.alias,
                hint="请在 llm.yaml 对应引擎中设置 embedding_model 字段",
            )

        if llm_cfg.sdk == "openai_compatible":
            from rag.embedders.openai_embedder import OpenAIEmbedder
            log.info(
                "embedder_factory.openai_compatible",
                alias=llm_cfg.alias,
                model=embed_model or "(none)",
                base_url=llm_cfg.base_url or "(sdk-default)",
                dimensions=dimensions,
            )
            return OpenAIEmbedder(
                model=embed_model,
                dimensions=dimensions,
                batch_size=batch_size,
                api_key=llm_cfg.api_key or "",
                base_url=llm_cfg.base_url or "",
            )

        if llm_cfg.sdk == "local_transformers":
            # LocalTransformersEngine 内部已持有 Qwen3LocalEmbedder，直接复用
            # 避免二次加载模型权重（build_engine 已创建一个实例）
            from llm.engines import LocalTransformersEngine
            model_path = embed_model or llm_cfg.model or ""
            if not model_path:
                raise RuntimeError(
                    f"[{llm_cfg.alias}] sdk=local_transformers 必须在 llm.yaml 中设置 "
                    "model（本地路径或 HuggingFace ID），如：\n"
                    "  model: D:\\work\\ai\\models\\huggingface\\Qwen\\Qwen3-Embedding\n"
                    "  model: Qwen/Qwen3-Embedding"
                )
            effective_dims = getattr(llm_cfg, "local_dimensions", 0) or dimensions
            log.info(
                "embedder_factory.local_transformers",
                alias=llm_cfg.alias,
                model=model_path,
                device=getattr(llm_cfg, "device", "cpu"),
                dimensions=effective_dims,
            )
            # 返回 LocalTransformersEngine 自身作为 embedder
            # 它实现了 embed(text) / embed_batch(texts) 接口，与 BaseEmbedder 兼容
            engine = LocalTransformersEngine(
                model      = model_path,
                device     = getattr(llm_cfg, "device", "cpu"),
                batch_size = getattr(llm_cfg, "local_batch_size", 8),
                max_length = getattr(llm_cfg, "local_max_length", 512),
                dimensions = effective_dims,
                alias      = llm_cfg.alias,
            )
            engine._dims = effective_dims  # 供 dimensions property 使用
            return engine

        # anthropic 无 embedding API，直接报错
        raise RuntimeError(
            f"[{llm_cfg.alias}] Anthropic SDK 不支持 Embedding API。"
            "请在 kb_config.yaml 中将 embed_engine 设为 openai_compatible 或 "
            "local_transformers 类引擎。"
        )

    @staticmethod
    def create_bge_local(cfg: "KBLLMBGEConfig") -> BaseEmbedder:  # type: ignore[name-defined]
        """创建本地 BGE Embedder（离线推理，不走 llm.yaml）。"""
        from rag.embedders.bge_local_embedder import BGELocalEmbedder
        log.info("embedder_factory.bge_local", model=cfg.model, device=cfg.device)
        return BGELocalEmbedder(model=cfg.model, device=cfg.device, batch_size=cfg.batch_size)

    @staticmethod
    def create_qwen3_local(cfg: "KBLLMQwen3LocalConfig") -> BaseEmbedder:  # type: ignore[name-defined]
        """创建本地 Qwen3-Embedding Embedder（Python 直接推理，不走 llm.yaml / Ollama）。"""
        from rag.embedders.qwen3_local_embedder import Qwen3LocalEmbedder
        log.info(
            "embedder_factory.qwen3_local",
            model=cfg.model,
            device=cfg.device,
            dimensions=cfg.dimensions,
        )
        return Qwen3LocalEmbedder(
            model=cfg.model,
            device=cfg.device,
            batch_size=cfg.batch_size,
            max_length=cfg.max_length,
            dimensions=cfg.dimensions,
        )

    @staticmethod
    def create_from_kb_config(kb_llm: "KBLLMConfig") -> BaseEmbedder:
        """
        从 KBLLMConfig 解析并创建 Embedder（完整流程）。

        解析顺序：
          1. embed_engine == "bge_local"   → 本地 BGE（Python 直接推理）
          2. embed_engine == "qwen3_local" → 本地 Qwen3-Embedding（Python 直接推理）
          3. embed_engine 非空 → 从 llm.yaml engines 中查找对应 alias
          4. embed_engine 为空 → 使用 llm.yaml router.embed 的引擎
          5. llm.yaml 中找不到 supports_embed=True 的引擎 → OpenAIEmbedder 兜底
        """
        engine_key = kb_llm.embed_engine.lower().strip()

        # 特殊情况：本地 BGE
        if engine_key == "bge_local":
            return EmbedderFactory.create_bge_local(kb_llm.bge_local)

        # 特殊情况：本地 Qwen3-Embedding（Python 直接推理）
        if engine_key == "qwen3_local":
            return EmbedderFactory.create_qwen3_local(kb_llm.qwen3_local)

        # 从 llm.yaml 解析引擎
        try:
            from utils.llm_config import load_from_yaml
            engines, router, _, _ = load_from_yaml()
            engine_map = {e.alias: e for e in engines}

            # embed_engine 指定 alias > router.embed > 第一个 supports_embed=True 的引擎
            alias = kb_llm.embed_engine.strip() or router.embed or ""
            engine = engine_map.get(alias) if alias else None
            if engine is None:
                engine = next((e for e in engines if e.supports_embed), None)

            if engine is not None:
                return EmbedderFactory.create_from_llm_config(
                    engine,
                    dimensions=kb_llm.embed_dimensions,
                    batch_size=kb_llm.embed_batch_size,
                )
            raise RuntimeError(
                f"在 llm.yaml 中找不到 embed engine：alias={kb_llm.embed_engine!r}，"
                f"router.embed={router.embed or '(未配置)'}。"
                "请在 kb_config.yaml embed_engine 中指定有效的引擎 alias，"
                "并确保该引擎在 llm.yaml 中配置了 embedding_model 且 supports_embed: true。"
            )
        except Exception as exc:
            raise RuntimeError(
                f"加载 llm.yaml embed engine 失败：{exc}。"
                "请检查 llm.yaml 配置和 embed_engine 设置。"
            ) from exc
