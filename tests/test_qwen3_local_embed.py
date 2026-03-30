"""
tests/test_qwen3_local_embed.py — Qwen3LocalEmbedder + qwen3-embed-local 路由集成测试

涵盖三个层次：
  1. Qwen3LocalEmbedder 单元测试（mock transformers，不需要 GPU / 模型文件）
  2. EmbedderFactory + local_transformers SDK 路由测试（mock 模型加载）
  3. 集成冒烟测试（需要真实模型文件，默认 skip，手动 -m integration 运行）

运行方式：
    # 仅跑 mock 测试（无需模型，CI 友好）
    pytest tests/test_qwen3_local_embed.py -v

    # 包含真实模型集成测试（需要模型已下载到本地）
    pytest tests/test_qwen3_local_embed.py -v -m integration
"""
from __future__ import annotations

import asyncio
import math
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# 辅助
# ─────────────────────────────────────────────────────────────────────────────

def _cosine(a: list[float], b: list[float]) -> float:
    dot  = sum(x * y for x, y in zip(a, b))
    na   = math.sqrt(sum(x * x for x in a))
    nb   = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-9)


def _fake_embeddings(texts: list[str], dims: int = 4096) -> list[list[float]]:
    """生成可重复的假向量（用于 mock）。"""
    import hashlib
    result = []
    for t in texts:
        seed = int(hashlib.md5(t.encode()).hexdigest(), 16) % (2 ** 32)
        import random
        rng = random.Random(seed)
        vec = [rng.gauss(0, 1) for _ in range(dims)]
        norm = math.sqrt(sum(v * v for v in vec)) + 1e-9
        result.append([v / norm for v in vec])
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 1. Qwen3LocalEmbedder 单元测试
# ─────────────────────────────────────────────────────────────────────────────

class TestQwen3LocalEmbedderUnit:
    """纯单元测试，全量 mock transformers，无需模型文件或 GPU。"""

    def _make_embedder(self, dims: int = 4096):
        from rag.embedders.qwen3_local_embedder import Qwen3LocalEmbedder
        return Qwen3LocalEmbedder(
            model="Qwen/Qwen3-Embedding",
            device="cpu",
            batch_size=4,
            max_length=512,
            dimensions=dims,
        )

    def _patch_model(self, embedder, dims: int = 4096):
        """给 embedder 注入 mock tokenizer + model，绕过真实加载。"""
        import torch

        # tokenizer 按实际 batch size 返回正确形状的张量
        def fake_tokenize(texts, **kwargs):
            batch_size = len(texts) if isinstance(texts, list) else 1
            return {
                "input_ids":      torch.ones(batch_size, 10, dtype=torch.long),
                "attention_mask": torch.ones(batch_size, 10, dtype=torch.long),
            }

        mock_tokenizer = MagicMock(side_effect=fake_tokenize)

        # model(**inputs) 返回带 last_hidden_state 的对象
        def fake_forward(**inputs):
            batch = inputs["input_ids"].shape[0]
            out = MagicMock()
            out.last_hidden_state = torch.randn(batch, 10, dims)
            return out

        mock_model = MagicMock(side_effect=fake_forward)
        mock_model.config = MagicMock()
        mock_model.config.hidden_size = dims

        embedder._tokenizer = mock_tokenizer
        embedder._model = mock_model
        embedder._dims = dims

    @pytest.mark.asyncio
    async def test_embed_returns_correct_length(self):
        embedder = self._make_embedder(dims=4096)
        self._patch_model(embedder, dims=4096)

        vec = await embedder.embed("测试文本")
        assert len(vec) == 4096, f"期望维度 4096，实际 {len(vec)}"

    @pytest.mark.asyncio
    async def test_embed_batch_returns_list_of_vecs(self):
        embedder = self._make_embedder(dims=4096)
        self._patch_model(embedder, dims=4096)

        texts = ["文本一", "文本二", "文本三"]
        vecs = await embedder.embed_batch(texts)
        assert len(vecs) == 3
        assert all(len(v) == 4096 for v in vecs), "每个向量维度应为 4096"

    @pytest.mark.asyncio
    async def test_embed_empty_list(self):
        embedder = self._make_embedder()
        result = await embedder.embed_batch([])
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_batch_chunking(self):
        """batch_size=2 时，5 条文本应分 3 批次处理，结果数量不变。"""
        from rag.embedders.qwen3_local_embedder import Qwen3LocalEmbedder
        embedder = Qwen3LocalEmbedder(model="x", device="cpu",
                                      batch_size=2, dimensions=4096)
        self._patch_model(embedder, dims=4096)

        texts = [f"文本{i}" for i in range(5)]
        vecs = await embedder.embed_batch(texts)
        assert len(vecs) == 5

    def test_dimensions_property(self):
        embedder = self._make_embedder(dims=4096)
        assert embedder.dimensions == 4096

    @pytest.mark.asyncio
    async def test_load_failure_returns_empty_vectors(self):
        """transformers 导入失败时，embed_batch 返回空向量列表而不是抛异常。"""
        from rag.embedders.qwen3_local_embedder import Qwen3LocalEmbedder
        embedder = Qwen3LocalEmbedder(model="不存在的路径", device="cpu", dimensions=4096)

        # 模拟 _load 失败
        with patch.object(embedder, "_load", return_value=False):
            vecs = await embedder.embed_batch(["hello", "world"])
        assert vecs == [[], []]

    @pytest.mark.asyncio
    async def test_vectors_are_normalized(self):
        """输出向量经过 L2 归一化，模长应约为 1.0。"""
        import torch
        import torch.nn.functional as F
        from rag.embedders.qwen3_local_embedder import Qwen3LocalEmbedder

        embedder = Qwen3LocalEmbedder(model="x", device="cpu",
                                      batch_size=4, dimensions=4096)
        self._patch_model(embedder, dims=4096)

        vecs = await embedder.embed_batch(["归一化测试"])
        if vecs and vecs[0]:
            norm = math.sqrt(sum(v * v for v in vecs[0]))
            assert abs(norm - 1.0) < 1e-4, f"向量应已 L2 归一化，实际模长={norm:.6f}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. EmbedderFactory + local_transformers SDK 路由测试
# ─────────────────────────────────────────────────────────────────────────────

class TestLocalTransformersFactory:
    """验证 sdk=local_transformers 能正确经由 EmbedderFactory 路由到 Qwen3LocalEmbedder。"""

    def _make_llm_config(self, model: str = "Qwen/Qwen3-Embedding"):
        from utils.llm_config import LLMConfig
        return LLMConfig(
            alias            = "qwen3-embed-local",
            sdk              = "local_transformers",    # type: ignore[arg-type]
            model            = model,
            embedding_model  = model,
            supports_embed   = True,
            supports_tools   = False,
            device           = "cpu",
            local_batch_size = 4,
            local_max_length = 512,
            local_dimensions = 4096,
            cost_tier        = 0,
        )

    def test_create_from_llm_config_returns_qwen3_embedder(self):
        from rag.embedders.factory import EmbedderFactory
        from rag.embedders.qwen3_local_embedder import Qwen3LocalEmbedder

        cfg = self._make_llm_config()
        embedder = EmbedderFactory.create_from_llm_config(cfg, dimensions=4096)
        assert isinstance(embedder, Qwen3LocalEmbedder)

    def test_create_uses_device_field(self):
        from rag.embedders.factory import EmbedderFactory
        from rag.embedders.qwen3_local_embedder import Qwen3LocalEmbedder

        cfg = self._make_llm_config()
        cfg.device = "cpu"
        embedder = EmbedderFactory.create_from_llm_config(cfg, dimensions=4096)
        assert isinstance(embedder, Qwen3LocalEmbedder)
        assert embedder._device == "cpu"

    def test_create_uses_local_dimensions(self):
        from rag.embedders.factory import EmbedderFactory

        cfg = self._make_llm_config()
        cfg.local_dimensions = 4096
        embedder = EmbedderFactory.create_from_llm_config(cfg, dimensions=1536)
        # local_dimensions=4096 优先于传入的 dimensions=1536
        assert embedder.dimensions == 4096

    def test_create_empty_model_raises(self):
        from rag.embedders.factory import EmbedderFactory

        cfg = self._make_llm_config(model="")
        cfg.embedding_model = ""
        with pytest.raises(RuntimeError, match="local_transformers"):
            EmbedderFactory.create_from_llm_config(cfg, dimensions=4096)

    def test_llm_config_from_dict_parses_local_transformers(self):
        from utils.llm_config import LLMConfig

        d = {
            "alias": "qwen3-embed-local",
            "sdk": "local_transformers",
            "model": "D:/models/Qwen3-Embedding",
            "embedding_model": "D:/models/Qwen3-Embedding",
            "device": "cuda",
            "local_batch_size": 8,
            "local_max_length": 512,
            "local_dimensions": 4096,
            "supports_embed": True,
            "supports_tools": False,
            "cost_tier": 0,
        }
        cfg = LLMConfig.from_dict(d)
        assert cfg.sdk == "local_transformers"
        assert cfg.device == "cuda"
        assert cfg.local_batch_size == 8
        assert cfg.local_dimensions == 4096

    def test_build_engine_raises_for_local_transformers(self):
        """local_transformers 仅做 embedding，调 build_engine() 应抛出明确错误。"""
        from utils.llm_config import LLMConfig

        cfg = LLMConfig(
            alias="qwen3-embed-local",
            sdk="local_transformers",   # type: ignore[arg-type]
            model="Qwen/Qwen3-Embedding",
        )
        with pytest.raises(RuntimeError, match="仅支持 Embedding"):
            cfg.build_engine()

    def test_llm_yaml_router_embed_resolves_to_local(self):
        """llm.yaml 中 router.embed=qwen3-embed-local 能解析到 local_transformers 引擎。"""
        from utils.llm_config import load_from_yaml

        engines, router, _, _ = load_from_yaml()
        assert router.embed == "qwen3-embed-local", (
            f"router.embed 应为 qwen3-embed-local，实际为 {router.embed!r}。"
            "请检查 llm.yaml router.embed 字段。"
        )

        engine_map = {e.alias: e for e in engines}
        assert "qwen3-embed-local" in engine_map, (
            "llm.yaml engines 中未找到 alias=qwen3-embed-local"
        )
        engine = engine_map["qwen3-embed-local"]
        assert engine.sdk == "local_transformers"
        assert engine.supports_embed is True

    def test_kb_config_embed_engine_resolves_to_local(self):
        """kb_config.yaml 中 embed_engine 已切换为 qwen3-embed-local。"""
        from rag.config import load_config

        cfg = load_config()
        assert cfg.llm.embed_engine == "qwen3-embed-local", (
            f"kb_config.yaml embed_engine 应为 qwen3-embed-local，"
            f"实际为 {cfg.llm.embed_engine!r}。"
            "请检查 kb_config.yaml knowledge_base.llm.embed_engine 字段。"
        )
        assert cfg.llm.embed_dimensions == 4096


# ─────────────────────────────────────────────────────────────────────────────
# 3. 集成冒烟测试（需要真实模型文件，默认 skip）
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestQwen3LocalEmbedIntegration:
    """
    真实模型推理测试。需要模型已下载到本地。

    运行方式：
        pytest tests/test_qwen3_local_embed.py -v -m integration
    """

    MODEL_PATH = r"D:\work\ai\models\huggingface\Qwen\Qwen3-Embedding"

    @pytest.fixture(scope="class")
    def embedder(self):
        import os
        if not os.path.exists(self.MODEL_PATH):
            pytest.skip(
                f"模型目录不存在：{self.MODEL_PATH}\n"
                "下载命令：huggingface-cli download Qwen/Qwen3-Embedding "
                f"--local-dir {self.MODEL_PATH}"
            )
        from rag.embedders.qwen3_local_embedder import Qwen3LocalEmbedder
        return Qwen3LocalEmbedder(
            model=self.MODEL_PATH,
            device="cpu",
            batch_size=2,
            max_length=512,
            dimensions=4096,
        )

    @pytest.mark.asyncio
    async def test_single_embed_shape(self, embedder):
        vec = await embedder.embed("北京今天天气怎么样")
        assert len(vec) == 4096, f"期望 4096 维，实际 {len(vec)} 维"
        # 应已归一化
        norm = math.sqrt(sum(v * v for v in vec))
        assert abs(norm - 1.0) < 1e-3

    @pytest.mark.asyncio
    async def test_semantic_similarity(self, embedder):
        """语义相近的句子余弦相似度应高于语义无关的句子对。"""
        texts = [
            "今天北京天气晴朗",
            "北京今日阳光明媚",   # 与第一句语义相近
            "量子计算机原理",     # 与第一句语义无关
        ]
        v0, v1, v2 = await embedder.embed_batch(texts)

        sim_near = _cosine(v0, v1)
        sim_far  = _cosine(v0, v2)
        assert sim_near > sim_far, (
            f"语义相近对相似度 {sim_near:.4f} 应大于语义无关对 {sim_far:.4f}"
        )

    @pytest.mark.asyncio
    async def test_batch_consistency(self, embedder):
        """单条 embed 和 batch embed 对同一文本应返回相同向量。"""
        text = "一致性测试文本"
        vec_single = await embedder.embed(text)
        vec_batch  = (await embedder.embed_batch([text]))[0]

        for a, b in zip(vec_single, vec_batch):
            assert abs(a - b) < 1e-5, "单条与批量结果不一致"

    @pytest.mark.asyncio
    async def test_performance(self, embedder):
        """8 条文本批量推理耗时应在 60 秒内（CPU 模式宽限）。"""
        import time
        texts = [f"性能测试文本第{i}条，用于验证批量推理速度。" for i in range(8)]
        t0 = time.perf_counter()
        await embedder.embed_batch(texts)
        elapsed = time.perf_counter() - t0
        assert elapsed < 60, f"8 条文本批量推理耗时 {elapsed:.1f}s，超出预期 60s"
