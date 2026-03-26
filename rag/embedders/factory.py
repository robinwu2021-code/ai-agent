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

        # anthropic 无 embedding API → 降级日志后返回 hash embedder
        log.warning(
            "embedder_factory.anthropic_no_embed",
            alias=llm_cfg.alias,
            hint="Anthropic SDK 无原生 Embedding，切换 embed_engine 到 openai_compatible 引擎",
        )
        from rag.embedders.openai_embedder import OpenAIEmbedder
        return OpenAIEmbedder(model="", dimensions=dimensions, batch_size=batch_size)

    @staticmethod
    def create_bge_local(cfg: "KBLLMBGEConfig") -> BaseEmbedder:  # type: ignore[name-defined]
        """创建本地 BGE Embedder（离线推理，不走 llm.yaml）。"""
        from rag.embedders.bge_local_embedder import BGELocalEmbedder
        log.info("embedder_factory.bge_local", model=cfg.model, device=cfg.device)
        return BGELocalEmbedder(model=cfg.model, device=cfg.device, batch_size=cfg.batch_size)

    @staticmethod
    def create_from_kb_config(kb_llm: "KBLLMConfig") -> BaseEmbedder:
        """
        从 KBLLMConfig 解析并创建 Embedder（完整流程）。

        解析顺序：
          1. embed_engine == "bge_local" → 本地 BGE（离线）
          2. embed_engine 非空 → 从 llm.yaml engines 中查找对应 alias
          3. embed_engine 为空 → 使用 llm.yaml router.embed 的引擎
          4. llm.yaml 中找不到 supports_embed=True 的引擎 → OpenAIEmbedder 兜底
        """
        # 特殊情况：本地 BGE
        if kb_llm.embed_engine.lower() == "bge_local":
            return EmbedderFactory.create_bge_local(kb_llm.bge_local)

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
            log.warning("embedder_factory.no_embed_engine_found",
                        embed_engine=kb_llm.embed_engine,
                        router_embed=router.embed or "(none)")
        except Exception as exc:
            log.warning("embedder_factory.llm_yaml_load_failed", error=str(exc))

        # 兜底：OpenAIEmbedder（可通过环境变量 OPENAI_API_KEY / OPENAI_BASE_URL 配置）
        log.warning("embedder_factory.fallback_openai_embedder")
        from rag.embedders.openai_embedder import OpenAIEmbedder
        return OpenAIEmbedder(
            dimensions=kb_llm.embed_dimensions,
            batch_size=kb_llm.embed_batch_size,
        )
