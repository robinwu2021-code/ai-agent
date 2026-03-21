"""
tests/test_ollama_qwen3.py — ollama-qwen3 配置验证测试

所有连接参数（base_url、model、max_tokens 等）均从 llm.yaml 读取，
测试代码中不硬编码任何配置值。

分两层：
  TestOllamaConfig  单元测试：解析 llm.yaml，验证字段结构与类型，不发网络请求
  TestOllamaLive    集成测试：真实调用 Ollama，验证服务可达且模型可用
                    默认跳过，需要 --run-live 参数才执行：
                      pytest tests/test_ollama_qwen3.py --run-live -v
"""
from __future__ import annotations

import pathlib
import pytest
from core.models import AgentConfig, Message, MessageRole

YAML_PATH = pathlib.Path(__file__).parent.parent / "llm.yaml"
ALIAS     = "ollama-qwen3"


# ── 共享 fixtures ─────────────────────────────────────────────────

@pytest.fixture(scope="session")
def run_live(request):
    return request.config.getoption("--run-live", default=False)


@pytest.fixture(scope="session")
def ollama_cfg():
    """从 llm.yaml 加载 ollama-qwen3 的 LLMConfig，整个 session 只解析一次。"""
    from utils.llm_config import load_from_yaml
    configs, _ = load_from_yaml(YAML_PATH)
    cfg = next((c for c in configs if c.alias == ALIAS), None)
    assert cfg is not None, (
        f"llm.yaml 中未找到 alias='{ALIAS}'，"
        f"当前 aliases: {[c.alias for c in configs]}"
    )
    return cfg


@pytest.fixture(scope="session")
def engine(ollama_cfg):
    """基于 llm.yaml 中的配置构建引擎实例。"""
    return ollama_cfg.build_engine()


# ══════════════════════════════════════════════════════════════════
# 1. 配置解析（单元测试，无网络）
# ══════════════════════════════════════════════════════════════════

class TestOllamaConfig:

    def test_yaml_exists(self):
        print(f"=========={YAML_PATH}=====")
        assert YAML_PATH.exists(), f"llm.yaml 不存在: {YAML_PATH}"

    def test_alias_found(self, ollama_cfg):
        print(f"=========={ollama_cfg.alias}====={ollama_cfg.alias}")
        assert ollama_cfg.alias == ALIAS

    def test_sdk_is_openai_compatible(self, ollama_cfg):
        assert ollama_cfg.sdk == "openai_compatible"

    def test_base_url_is_set(self, ollama_cfg):
        assert ollama_cfg.base_url, "base_url 不能为空"
        assert ollama_cfg.base_url.startswith("http"), "base_url 应以 http 开头"

    def test_model_is_set(self, ollama_cfg):
        assert ollama_cfg.model, "model 不能为空"
        assert ollama_cfg.resolved_model() == ollama_cfg.model

    def test_max_tokens_positive(self, ollama_cfg):
        assert ollama_cfg.max_tokens > 0

    def test_timeout_sufficient_for_large_model(self, ollama_cfg):
        # 大模型推理慢，超时至少应 >= 60s
        assert ollama_cfg.timeout_sec >= 60.0, (
            f"timeout_sec={ollama_cfg.timeout_sec} 对本地大模型过短，建议 >= 60s"
        )

    def test_cost_tier_is_local(self, ollama_cfg):
        # 本地模型 cost_tier 应为 0
        assert ollama_cfg.cost_tier == 0

    def test_supports_tools(self, ollama_cfg):
        assert ollama_cfg.supports_tools is True

    def test_tags_contain_local(self, ollama_cfg):
        assert "local" in ollama_cfg.tags

    def test_no_api_key_required(self, ollama_cfg):
        # Ollama 本地服务不需要 api_key
        assert not ollama_cfg.api_key, "Ollama 本地服务不应配置 api_key"

    def test_is_not_azure(self, ollama_cfg):
        assert ollama_cfg.is_azure is False

    def test_build_engine_returns_openai_engine(self, ollama_cfg):
        from llm.engines import OpenAIEngine
        assert isinstance(ollama_cfg.build_engine(), OpenAIEngine)


# ══════════════════════════════════════════════════════════════════
# 2. 集成测试（需要真实 Ollama 服务，默认跳过）
# ══════════════════════════════════════════════════════════════════

class TestOllamaLive:

    @pytest.mark.anyio
    async def test_service_reachable(self, ollama_cfg, run_live):
        """验证 Ollama 服务可达（地址来自 llm.yaml）。"""
        if not run_live:
            pytest.skip("跳过集成测试，需加 --run-live 参数")

        import httpx
        # base_url 形如 http://host:port/v1，取 host:port 部分拼 /api/tags
        origin = ollama_cfg.base_url.rstrip("/").rsplit("/v1", 1)[0]
        tags_url = f"{origin}/api/tags"

        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                resp = await client.get(tags_url)
                assert resp.status_code == 200, f"Ollama 服务响应异常: {resp.status_code}"
            except httpx.ConnectError:
                pytest.fail(f"无法连接 Ollama 服务 {origin}，请确认服务已启动")

    @pytest.mark.anyio
    async def test_model_loaded(self, ollama_cfg, run_live):
        """验证 llm.yaml 中配置的模型已在 Ollama 中加载。"""
        if not run_live:
            pytest.skip("跳过集成测试，需加 --run-live 参数")

        import httpx
        origin   = ollama_cfg.base_url.rstrip("/").rsplit("/v1", 1)[0]
        tags_url = f"{origin}/api/tags"
        model    = ollama_cfg.resolved_model()

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp   = await client.get(tags_url)
            models = [m["name"] for m in resp.json().get("models", [])]
            # 模型名前缀匹配（如 "qwen3.5:35b-a3b" 匹配 "qwen3.5:35b-a3b:latest"）
            assert any(m.startswith(model.split(":")[0]) for m in models), (
                f"模型 {model!r} 未在 Ollama 中找到\n"
                f"已加载模型: {models}\n"
                f"请先运行: ollama pull {model}"
            )

    @pytest.mark.anyio
    async def test_chat_basic(self, engine, run_live):
        """发送一条简单消息，验证返回非空内容。"""
        if not run_live:
            pytest.skip("跳过集成测试，需加 --run-live 参数")

        messages = [Message(role=MessageRole.USER, content="用一个词回答：天空是什么颜色？")]
        resp = await engine.chat(messages, [], AgentConfig())

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
        resp = await engine.chat(messages, [], AgentConfig())
        assert resp.model, "响应中 model 字段不能为空"

    @pytest.mark.anyio
    async def test_chat_respects_max_tokens(self, ollama_cfg, engine, run_live):
        """验证 completion_tokens 不超过 llm.yaml 中配置的 max_tokens。"""
        if not run_live:
            pytest.skip("跳过集成测试，需加 --run-live 参数")

        messages = [Message(role=MessageRole.USER, content="写一首很长的诗")]
        resp = await engine.chat(messages, [], AgentConfig())
        completion = resp.usage.get("completion_tokens", 0)
        assert completion <= ollama_cfg.max_tokens, (
            f"completion_tokens={completion} 超过 max_tokens={ollama_cfg.max_tokens}"
        )

    @pytest.mark.anyio
    async def test_stream_chat(self, engine, run_live):
        """验证流式输出正常工作。"""
        if not run_live:
            pytest.skip("跳过集成测试，需加 --run-live 参数")

        messages = [Message(role=MessageRole.USER, content="说一个数字")]
        chunks = []
        async for chunk in engine.stream_chat(messages, [], AgentConfig()):
            chunks.append(chunk)

        assert len(chunks) > 0, "流式输出应至少返回一个 chunk"
        assert len("".join(chunks).strip()) > 0, "流式输出内容不能为空"
