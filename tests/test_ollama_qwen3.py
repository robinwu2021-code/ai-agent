"""
tests/test_ollama_qwen3.py — ollama-qwen3 配置验证测试

分两层：
  TestOllamaConfig     单元测试：解析 llm.yaml，验证字段正确，不发网络请求
  TestOllamaLive       集成测试：真实调用 Ollama，验证服务可达且模型可用
                        默认跳过，需要 --run-live 参数才执行：
                          pytest tests/test_ollama_qwen3.py --run-live -v
"""
from __future__ import annotations

import pathlib
import pytest
from core.models import AgentConfig, Message, MessageRole

# llm.yaml 路径：项目根目录
YAML_PATH = pathlib.Path(__file__).parent.parent / "llm.yaml"

@pytest.fixture(scope="session")
def run_live(request):
    return request.config.getoption("--run-live", default=False)


# ══════════════════════════════════════════════════════════════════
# 1. 配置解析（单元测试，无网络）
# ══════════════════════════════════════════════════════════════════

class TestOllamaConfig:

    @pytest.fixture(scope="class")
    def ollama_cfg(self):
        from utils.llm_config import load_from_yaml
        configs, router_cfg = load_from_yaml(YAML_PATH)
        match = next((c for c in configs if c.alias == "ollama-qwen3"), None)
        assert match is not None, (
            f"llm.yaml 中未找到 alias='ollama-qwen3'，"
            f"当前 aliases: {[c.alias for c in configs]}"
        )
        return match

    def test_yaml_exists(self):
        assert YAML_PATH.exists(), f"llm.yaml 不存在: {YAML_PATH}"

    def test_alias(self, ollama_cfg):
        assert ollama_cfg.alias == "ollama-qwen3"

    def test_sdk_is_openai_compatible(self, ollama_cfg):
        assert ollama_cfg.sdk == "openai_compatible"

    def test_base_url(self, ollama_cfg):
        assert ollama_cfg.base_url == "http://192.168.0.222:11434/v1"

    def test_model(self, ollama_cfg):
        assert ollama_cfg.model == "qwen3.5:35b-a3b"

    def test_resolved_model(self, ollama_cfg):
        assert ollama_cfg.resolved_model() == "qwen3.5:35b-a3b"

    def test_max_tokens(self, ollama_cfg):
        assert ollama_cfg.max_tokens == 8192

    def test_timeout(self, ollama_cfg):
        assert ollama_cfg.timeout_sec == 300.0

    def test_cost_tier(self, ollama_cfg):
        assert ollama_cfg.cost_tier == 0

    def test_supports_tools(self, ollama_cfg):
        assert ollama_cfg.supports_tools is True

    def test_tags_contain_local(self, ollama_cfg):
        assert "local" in ollama_cfg.tags
        assert "qwen" in ollama_cfg.tags

    def test_no_api_key_required(self, ollama_cfg):
        # Ollama 本地服务不需要 api_key
        assert ollama_cfg.api_key is None

    def test_is_not_azure(self, ollama_cfg):
        assert ollama_cfg.is_azure is False

    def test_build_engine_returns_openai_engine(self, ollama_cfg):
        from llm.engines import OpenAIEngine
        engine = ollama_cfg.build_engine()
        assert isinstance(engine, OpenAIEngine)


# ══════════════════════════════════════════════════════════════════
# 2. 集成测试（需要真实 Ollama 服务，默认跳过）
# ══════════════════════════════════════════════════════════════════

class TestOllamaLive:

    @pytest.fixture(scope="class")
    def engine(self):
        from utils.llm_config import load_from_yaml
        configs, _ = load_from_yaml(YAML_PATH)
        cfg = next(c for c in configs if c.alias == "ollama-qwen3")
        return cfg.build_engine()

    @pytest.mark.anyio
    async def test_service_reachable(self, engine, run_live):
        if not run_live:
            pytest.skip("跳过集成测试，需加 --run-live 参数")

        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                resp = await client.get("http://192.168.0.222:11434/api/tags")
                assert resp.status_code == 200, f"Ollama 服务响应异常: {resp.status_code}"
            except httpx.ConnectError:
                pytest.fail("无法连接 Ollama 服务 http://192.168.0.222:11434，请确认服务已启动")

    @pytest.mark.anyio
    async def test_model_loaded(self, run_live):
        """验证 qwen3.5:35b-a3b 已在 Ollama 中加载。"""
        if not run_live:
            pytest.skip("跳过集成测试，需加 --run-live 参数")

        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get("http://192.168.0.222:11434/api/tags")
            models = [m["name"] for m in resp.json().get("models", [])]
            assert any("qwen3.5" in m for m in models), (
                f"模型 qwen3.5:35b-a3b 未在 Ollama 中找到，"
                f"已加载模型: {models}\n"
                f"请先运行: ollama pull qwen3.5:35b-a3b"
            )

    @pytest.mark.anyio
    async def test_chat_basic(self, engine, run_live):
        """发送一条简单消息，验证返回非空内容。"""
        if not run_live:
            pytest.skip("跳过集成测试，需加 --run-live 参数")

        messages = [Message(role=MessageRole.USER, content="用一个词回答：天空是什么颜色？")]
        config   = AgentConfig()

        resp = await engine.chat(messages, [], config)

        assert resp.content, "响应内容不能为空"
        assert len(resp.content.strip()) > 0
        assert isinstance(resp.usage, dict)
        assert resp.usage.get("prompt_tokens", 0) > 0

    @pytest.mark.anyio
    async def test_chat_returns_model_name(self, engine, run_live):
        """验证响应中包含模型名称信息。"""
        if not run_live:
            pytest.skip("跳过集成测试，需加 --run-live 参数")

        messages = [Message(role=MessageRole.USER, content="Hi")]
        config   = AgentConfig()

        resp = await engine.chat(messages, [], config)
        assert resp.model, "响应中 model 字段不能为空"

    @pytest.mark.anyio
    async def test_chat_respects_max_tokens(self, engine, run_live):
        """验证 max_tokens 限制生效（completion_tokens 不超过配置值）。"""
        if not run_live:
            pytest.skip("跳过集成测试，需加 --run-live 参数")

        messages = [Message(role=MessageRole.USER, content="写一首很长的诗")]
        config   = AgentConfig()

        resp = await engine.chat(messages, [], config)
        completion = resp.usage.get("completion_tokens", 0)
        assert completion <= 8192, f"completion_tokens={completion} 超过 max_tokens=8192"

    @pytest.mark.anyio
    async def test_stream_chat(self, engine, run_live):
        """验证流式输出正常工作。"""
        if not run_live:
            pytest.skip("跳过集成测试，需加 --run-live 参数")

        messages = [Message(role=MessageRole.USER, content="说一个数字")]
        config   = AgentConfig()

        chunks = []
        async for chunk in engine.stream_chat(messages, [], config):
            chunks.append(chunk)

        assert len(chunks) > 0, "流式输出应至少返回一个 chunk"
        full_text = "".join(chunks)
        assert len(full_text.strip()) > 0, "流式输出内容不能为空"
