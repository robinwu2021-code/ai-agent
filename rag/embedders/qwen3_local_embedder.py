"""
rag/embedders/qwen3_local_embedder.py — Qwen3-Embedding 本地推理（直接 Python 调用）

通过 transformers 库直接加载 Qwen3-Embedding 模型，无需 Ollama / HTTP 服务。
适合 qwen3-embedding:4b 等轻量模型在应用机器上本地推理。

安装依赖：
    pip install transformers torch

模型下载（二选一）：
    # HuggingFace（需网络）
    model_path = "Qwen/Qwen3-Embedding-4B"

    # 本地路径（离线，先 huggingface-cli download 或手动放置）
    model_path = "/path/to/Qwen3-Embedding-4B"
"""
from __future__ import annotations

import asyncio
import structlog

log = structlog.get_logger(__name__)


class Qwen3LocalEmbedder:
    """
    直接通过 transformers 加载 Qwen3-Embedding 系列模型进行本地推理。

    与 BGELocalEmbedder 类似，模型在第一次调用时懒加载，之后复用。
    推理在线程池中执行，不阻塞 asyncio 事件循环。
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen3-Embedding-4B",
        device: str = "cpu",
        batch_size: int = 16,
        max_length: int = 512,
        dimensions: int = 2560,
    ) -> None:
        self._model_name = model
        self._device = device
        self._batch_size = batch_size
        self._max_length = max_length
        self._dims = dimensions
        self._tokenizer = None
        self._model = None

    def _load(self):
        """懒加载 tokenizer + model（仅首次调用时执行）。"""
        if self._model is not None:
            return True
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            log.info("qwen3_local_embedder.loading", model=self._model_name, device=self._device)
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_name, trust_remote_code=True
            )
            self._model = AutoModel.from_pretrained(
                self._model_name, trust_remote_code=True
            )
            if self._device != "cpu":
                self._model = self._model.to(self._device)
            self._model.eval()

            # 从模型 config 推断实际输出维度
            hidden = getattr(self._model.config, "hidden_size", None)
            if hidden:
                self._dims = hidden
            log.info(
                "qwen3_local_embedder.loaded",
                model=self._model_name,
                dims=self._dims,
            )
            return True
        except ImportError as exc:
            log.warning(
                "qwen3_local_embedder.import_error",
                error=str(exc),
                hint="pip install transformers torch",
            )
        except Exception as exc:
            log.warning("qwen3_local_embedder.load_failed", error=str(exc))
        return False

    @staticmethod
    def _last_token_pool(last_hidden_states, attention_mask):
        """取最后一个有效 token 的隐向量作为句子表示（Qwen3-Embedding 官方方式）。"""
        import torch

        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths,
        ]

    def _encode_batch(self, texts: list[str]) -> list[list[float]]:
        """同步推理一个 batch，在线程池中调用。"""
        import torch
        import torch.nn.functional as F

        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )
        if self._device != "cpu":
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        embeddings = self._last_token_pool(
            outputs.last_hidden_state, inputs["attention_mask"]
        )
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().tolist()

    @property
    def dimensions(self) -> int:
        return self._dims

    async def embed(self, text: str) -> list[float]:
        results = await self.embed_batch([text])
        return results[0] if results else []

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not await asyncio.to_thread(self._load):
            return [[] for _ in texts]

        all_embs: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            try:
                result = await asyncio.to_thread(self._encode_batch, batch)
                all_embs.extend(result)
            except Exception as exc:
                log.warning(
                    "qwen3_local_embedder.batch_failed",
                    batch_start=i,
                    size=len(batch),
                    error=str(exc),
                )
                all_embs.extend([[] for _ in batch])
        return all_embs
