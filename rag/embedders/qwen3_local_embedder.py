"""
rag/embedders/qwen3_local_embedder.py — Qwen3-Embedding 本地推理（直接 Python 调用）

通过 sentence-transformers 或 transformers 库加载 Qwen3-Embedding 模型，
无需 Ollama / HTTP 服务，完全离线可用。

优先策略：
  1. sentence-transformers（Qwen3-Embedding-8B 官方格式，自动处理 pooling/归一化）
  2. transformers AutoModel（回退，适合非 sentence-transformers 格式的模型）

安装依赖：
    pip install sentence-transformers torch
    # 或仅用 transformers 路径：
    pip install transformers torch

模型下载：
    # 下载到指定目录（推荐）
    python -c "
    from huggingface_hub import snapshot_download
    snapshot_download('Qwen/Qwen3-Embedding-8B',
                      local_dir=r'D:/work/ai/models/huggingface/Qwen/Qwen3-Embedding-8B',
                      local_dir_use_symlinks=False)
    "
    # 或直接使用 HuggingFace ID（需要 HF_HOME 指向 D 盘缓存目录）
    model_path = "Qwen/Qwen3-Embedding-8B"
"""
from __future__ import annotations

import asyncio
import structlog

log = structlog.get_logger(__name__)


class Qwen3LocalEmbedder:
    """
    直接通过 sentence-transformers / transformers 加载 Qwen3-Embedding 系列模型。

    加载策略（按优先级）：
      1. sentence-transformers SentenceTransformer（官方推荐，自动处理 pooling 和归一化）
      2. transformers AutoModel + 手动 last-token pooling（兜底，适合非 ST 格式模型）

    模型在第一次调用时懒加载，之后复用。
    推理在线程池中执行，不阻塞 asyncio 事件循环。
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen3-Embedding-8B",
        device: str = "cpu",
        batch_size: int = 8,
        max_length: int = 512,
        dimensions: int = 4096,
    ) -> None:
        self._model_name  = model
        self._device      = device
        self._batch_size  = batch_size
        self._max_length  = max_length
        self._dims        = dimensions
        self._model       = None      # SentenceTransformer 或 AutoModel 实例
        self._tokenizer   = None      # 仅 transformers 路径使用
        self._use_st      = False     # True = sentence-transformers 路径

    # ── 懒加载 ────────────────────────────────────────────────────

    def _load(self) -> bool:
        """首次调用时加载模型，之后直接返回 True。"""
        if self._model is not None:
            return True

        # ── 路径 1：sentence-transformers（Qwen3-Embedding-8B 官方格式）────
        try:
            from sentence_transformers import SentenceTransformer

            log.info("qwen3_local_embedder.loading_st",
                     model=self._model_name, device=self._device)
            self._model = SentenceTransformer(
                self._model_name,
                device=self._device,
                trust_remote_code=True,
            )
            # 从模型获取实际输出维度
            dim = self._model.get_sentence_embedding_dimension()
            if dim:
                self._dims = dim
            self._use_st = True
            log.info("qwen3_local_embedder.loaded",
                     backend="sentence-transformers",
                     model=self._model_name, dims=self._dims)
            return True

        except Exception as st_exc:
            log.debug("qwen3_local_embedder.st_unavailable",
                      error=str(st_exc)[:120],
                      hint="尝试 transformers 路径")

        # ── 路径 2：transformers AutoModel（回退）────────────────
        try:
            from transformers import AutoModel, AutoTokenizer

            log.info("qwen3_local_embedder.loading_transformers",
                     model=self._model_name, device=self._device)
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_name, trust_remote_code=True
            )
            self._model = AutoModel.from_pretrained(
                self._model_name, trust_remote_code=True
            )
            if self._device != "cpu":
                self._model = self._model.to(self._device)
            self._model.eval()
            hidden = getattr(self._model.config, "hidden_size", None)
            if hidden:
                self._dims = hidden
            self._use_st = False
            log.info("qwen3_local_embedder.loaded",
                     backend="transformers",
                     model=self._model_name, dims=self._dims)
            return True

        except ImportError as exc:
            log.warning("qwen3_local_embedder.import_error",
                        error=str(exc),
                        hint="pip install sentence-transformers transformers torch")
        except Exception as exc:
            log.warning("qwen3_local_embedder.load_failed", error=str(exc))

        return False

    # ── sentence-transformers 推理 ────────────────────────────────

    def _encode_batch_st(self, texts: list[str]) -> list[list[float]]:
        """使用 SentenceTransformer.encode() 批量推理（自动归一化）。"""
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.tolist() if hasattr(embeddings, "tolist") else list(embeddings)

    # ── transformers 推理 ─────────────────────────────────────────

    @staticmethod
    def _last_token_pool(last_hidden_states, attention_mask):
        """取最后一个有效 token 的隐向量（Qwen3-Embedding 官方 pooling 方式）。"""
        import torch

        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        seq_lens = attention_mask.sum(dim=1) - 1
        batch    = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch, device=last_hidden_states.device), seq_lens
        ]

    def _encode_batch_transformers(self, texts: list[str]) -> list[list[float]]:
        """使用 AutoModel 手动推理 + last-token pooling + L2 归一化。"""
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

    # ── 公开接口 ──────────────────────────────────────────────────

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

        _encode = self._encode_batch_st if self._use_st else self._encode_batch_transformers

        all_embs: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i: i + self._batch_size]
            try:
                result = await asyncio.to_thread(_encode, batch)
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
