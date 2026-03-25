"""
rag/embedders/bge_local_embedder.py — BGE-M3 本地 Embedding（完全开源离线）

安装：pip install FlagEmbedding
支持中英双语，无需 API Key。
"""
from __future__ import annotations

import asyncio
import structlog

log = structlog.get_logger(__name__)


class BGELocalEmbedder:
    def __init__(self, model: str = "BAAI/bge-m3",
                 device: str = "cpu", batch_size: int = 16) -> None:
        self._model_name = model
        self._device = device
        self._batch_size = batch_size
        self._model = None
        self._dims = 1024

    def _get_model(self):
        if self._model is not None:
            return self._model
        try:
            from FlagEmbedding import FlagModel
            self._model = FlagModel(
                self._model_name,
                use_fp16=(self._device != "cpu"),
            )
            log.info("bge_embedder.loaded", model=self._model_name)
        except ImportError:
            log.warning("bge_embedder.missing", hint="pip install FlagEmbedding")
        except Exception as exc:
            log.warning("bge_embedder.load_failed", error=str(exc))
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dims

    async def embed(self, text: str) -> list[float]:
        r = await self.embed_batch([text])
        return r[0] if r else []

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_model()
        if model is None:
            return [[] for _ in texts]

        all_embs: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i: i + self._batch_size]
            try:
                def _run(b=batch):
                    embs = model.encode(b)
                    return embs.tolist() if hasattr(embs, "tolist") else list(embs)
                result = await asyncio.to_thread(_run)
                if result and not self._dims:
                    self._dims = len(result[0])
                all_embs.extend(result)
            except Exception as exc:
                log.warning("bge_embedder.batch_failed", error=str(exc))
                all_embs.extend([[] for _ in batch])
        return all_embs
