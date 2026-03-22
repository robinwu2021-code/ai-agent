"""
tests/test_kb_qa.py — 知识库问答端到端测试

测试完整 RAG 管道，覆盖：
  1. 文档写入 → embed_fn 调用 → SQLite 持久化
  2. 混合检索 (向量 + BM25 RRF)
  3. 检索结果 → LLM 上下文拼装 → chat 生成答案

日志可见性：
  - openai_engine.embed.request / response / error   ← 每次 embedding 调用
  - llm.chat.request / response / error              ← 每次 LLM 调用
  - 同时写入 /tmp/test_llm_calls.jsonl

运行方式：
  # 只跑配置/单元测试（无网络，默认）
  pytest tests/test_kb_qa.py -v

  # 跑全部集成测试（需要 Ollama 真实服务）
  pytest tests/test_kb_qa.py --run-live -v -s

  # 只跑指定测试
  pytest tests/test_kb_qa.py::TestKBQALive::test_full_rag_qa --run-live -v -s
"""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import textwrap

import pytest

# ── 路径修正（与 conftest.py 保持一致）──────────────────────────────
_ROOT = pathlib.Path(__file__).parent.parent
YAML_PATH    = _ROOT / "llm.yaml"
DEMO_DIR     = _ROOT / "docs" / "demo"
LOG_DIR      = _ROOT / "logs"

EMBED_ALIAS  = "ollama-embed"   # llm.yaml 中的 embedding 专属引擎
CHAT_ALIAS   = "ollama-qwen3"   # llm.yaml 中的对话引擎


# ══════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def run_live(request):
    return request.config.getoption("--run-live", default=False)


@pytest.fixture(scope="session")
def llm_configs():
    """从 llm.yaml 加载所有引擎配置（整个 session 只解析一次）。"""
    from utils.llm_config import load_from_yaml
    configs, router_cfg = load_from_yaml(YAML_PATH)
    return configs, router_cfg


@pytest.fixture(scope="session")
def embed_cfg(llm_configs):
    configs, _ = llm_configs
    cfg = next((c for c in configs if c.alias == EMBED_ALIAS), None)
    assert cfg is not None, (
        f"llm.yaml 中未找到 alias='{EMBED_ALIAS}'，"
        f"当前 aliases: {[c.alias for c in configs]}"
    )
    return cfg


@pytest.fixture(scope="session")
def chat_cfg(llm_configs):
    configs, _ = llm_configs
    cfg = next((c for c in configs if c.alias == CHAT_ALIAS), None)
    assert cfg is not None, (
        f"llm.yaml 中未找到 alias='{CHAT_ALIAS}'，"
        f"当前 aliases: {[c.alias for c in configs]}"
    )
    return cfg


@pytest.fixture(scope="session")
def embed_engine(embed_cfg):
    return embed_cfg.build_engine()


@pytest.fixture(scope="session")
def chat_engine(chat_cfg):
    return chat_cfg.build_engine()


@pytest.fixture(scope="session")
def call_logger_setup():
    """
    初始化 LLM 调用全链路日志（DEBUG 级别，同时写 JSONL 文件）。
    整个 session 只初始化一次。
    """
    from llm.call_logger import init_call_logger
    log_file = str(LOG_DIR / "test_llm_calls.jsonl")
    logger = init_call_logger(
        enabled      = True,
        log_level    = "DEBUG",      # 控制台打印全量 request/response
        file_path    = log_file,
        max_bytes    = 5 * 1024 * 1024,
        backup_count = 2,
        msg_preview  = 800,          # 测试时展示更多内容
    )
    print(f"\n📋 LLM 调用日志文件: {log_file}")
    return logger, log_file


@pytest.fixture
def tmp_kb(call_logger_setup, embed_engine):
    """
    每个测试用例创建一个独立的临时 KnowledgeBase（SQLite in tmp dir）。
    embed_fn 使用真实 ollama-embed 引擎（集成测试时），单元测试无需此 fixture。
    """
    from rag.store import SQLiteKBStore
    from rag.persistent_kb import PersistentKnowledgeBase

    with tempfile.TemporaryDirectory(prefix="test_kb_") as tmp_dir:
        db_path = str(pathlib.Path(tmp_dir) / "test_kb.db")
        store   = SQLiteKBStore(db_path=db_path)

        async def _embed(text: str) -> list[float]:
            return await embed_engine.embed(text)

        kb = PersistentKnowledgeBase(
            store    = store,
            embed_fn = _embed,
            kb_id    = "test",
        )
        yield kb, store


# ══════════════════════════════════════════════════════════════════════
# 1. 配置单元测试（无网络）
# ══════════════════════════════════════════════════════════════════════

class TestKBConfig:
    """验证配置文件结构，不发送任何网络请求。"""

    def test_yaml_exists(self):
        assert YAML_PATH.exists(), f"llm.yaml 不存在: {YAML_PATH}"

    def test_embed_alias_found(self, embed_cfg):
        assert embed_cfg.alias == EMBED_ALIAS

    def test_embed_engine_type(self, embed_cfg):
        assert embed_cfg.sdk == "openai_compatible"

    def test_embed_model_set(self, embed_cfg):
        em = embed_cfg.resolved_embedding_model()
        assert em, "embedding_model 不能为空"
        assert "embed" in em.lower() or "bge" in em.lower() or "minilm" in em.lower() \
               or "nomic" in em.lower() or "mxbai" in em.lower() or "qwen" in em.lower(), \
               f"embedding_model 看起来不像 embedding 模型: {em!r}"

    def test_embed_supports_embed_flag(self, embed_cfg):
        assert embed_cfg.supports_embed is True, \
            f"ollama-embed 的 supports_embed 应为 True"

    def test_embed_base_url(self, embed_cfg):
        assert embed_cfg.base_url, "embedding 引擎需要指定 base_url"
        assert embed_cfg.base_url.startswith("http")

    def test_chat_alias_found(self, chat_cfg):
        assert chat_cfg.alias == CHAT_ALIAS

    def test_chat_supports_tools(self, chat_cfg):
        assert chat_cfg.supports_tools is True

    def test_demo_docs_exist(self):
        assert DEMO_DIR.exists(), f"demo 文档目录不存在: {DEMO_DIR}"
        docs = list(DEMO_DIR.glob("*.md"))
        assert len(docs) >= 1, f"demo 目录下应至少有 1 个 .md 文件，当前: {docs}"

    def test_call_logger_init(self, call_logger_setup):
        logger, log_file = call_logger_setup
        assert logger.enabled is True
        print(f"\n✅ LLM 调用日志已启用，输出到: {log_file}")

    @pytest.mark.anyio
    async def test_kb_store_schema(self):
        """验证 SQLiteKBStore 可以建表（不发网络请求）。"""
        from rag.store import SQLiteKBStore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            store = SQLiteKBStore(db_path=str(pathlib.Path(d) / "schema_test.db"))
            try:
                await store.initialize()
                stats = await store.get_stats("test")
                assert stats["documents"] == 0
                assert stats["chunks"] == 0
            finally:
                # 显式关闭 SQLite 连接，避免 Windows 文件锁阻止临时目录清理
                if store._conn is not None:
                    store._conn.close()
                    store._conn = None


# ══════════════════════════════════════════════════════════════════════
# 2. Embedding 集成测试
# ══════════════════════════════════════════════════════════════════════

class TestEmbedLive:
    """验证 ollama-embed 引擎可用且 embedding 质量正常。"""

    @pytest.mark.anyio
    async def test_embed_single_text(self, embed_engine, call_logger_setup, run_live):
        """
        单条文本 embedding，验证：
        - 返回非空向量
        - 维度合理（>= 256）
        - 日志中有 openai_engine.embed.request / response 事件
        """
        if not run_live:
            pytest.skip("跳过集成测试，需加 --run-live 参数")

        text = "星辰科技有限公司成立于2018年，专注于AI营销领域。"
        print(f"\n▶ 调用 embedding: {text!r}")

        vec = await embed_engine.embed(text)

        assert isinstance(vec, list), "embedding 应返回 list[float]"
        assert len(vec) >= 256, f"向量维度过低: {len(vec)}"
        assert all(isinstance(v, float) for v in vec[:10])
        print(f"✅ embedding 成功，维度={len(vec)}，前6维={[round(v,4) for v in vec[:6]]}")

    @pytest.mark.anyio
    async def test_embed_semantic_similarity(self, embed_engine, call_logger_setup, run_live):
        """
        验证语义相似度：语义相近的句子余弦相似度应 > 语义不同的句子。
        同时可在日志中看到 3 次 embed 调用。
        """
        if not run_live:
            pytest.skip("跳过集成测试，需加 --run-live 参数")

        import math

        def cosine(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            na  = math.sqrt(sum(x * x for x in a))
            nb  = math.sqrt(sum(x * x for x in b))
            return dot / (na * nb) if na * nb > 0 else 0.0

        texts = [
            "星辰科技的CEO是张明远",               # 句子A
            "星辰科技的首席执行官叫做张明远",        # 句子B（语义相近A）
            "今天天气晴，适合出去散步",              # 句子C（语义不相关）
        ]
        print(f"\n▶ 计算3条文本的语义相似度（产生3次 embed 调用）…")
        vecs = [await embed_engine.embed(t) for t in texts]

        sim_ab = cosine(vecs[0], vecs[1])
        sim_ac = cosine(vecs[0], vecs[2])

        print(f"  A-B 相似度（语义相近）: {sim_ab:.4f}")
        print(f"  A-C 相似度（语义无关）: {sim_ac:.4f}")

        assert sim_ab > sim_ac, (
            f"相近句子的相似度({sim_ab:.4f})应大于无关句子({sim_ac:.4f})，"
            f"embedding 模型语义感知能力异常"
        )
        print(f"✅ 语义相似度正常，A-B({sim_ab:.4f}) > A-C({sim_ac:.4f})")

    @pytest.mark.anyio
    async def test_embed_logs_written_to_file(self, embed_engine, call_logger_setup, run_live):
        """验证 embedding 调用日志已写入 JSONL 文件。"""
        if not run_live:
            pytest.skip("跳过集成测试，需加 --run-live 参数")

        _, log_file = call_logger_setup
        marker_text = "EMBED_LOG_TEST_MARKER_12345"

        await embed_engine.embed(marker_text)

        log_path = pathlib.Path(log_file)
        assert log_path.exists(), f"日志文件未创建: {log_file}"

        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        req_events = [
            json.loads(l) for l in lines
            if "embed" in l and "request" in l
        ]
        assert len(req_events) > 0, "日志文件中未找到 embed.request 事件"

        resp_events = [
            json.loads(l) for l in lines
            if "embed" in l and "response" in l
        ]
        assert len(resp_events) > 0, "日志文件中未找到 embed.response 事件"

        print(f"\n✅ 日志文件校验通过：{len(req_events)} 条 request，{len(resp_events)} 条 response")
        print(f"   最近一条 response: {json.dumps(resp_events[-1], ensure_ascii=False, indent=2)}")


# ══════════════════════════════════════════════════════════════════════
# 3. 知识库写入集成测试
# ══════════════════════════════════════════════════════════════════════

class TestKBIndexing:
    """测试文档写入 + embedding 索引流程。"""

    @pytest.mark.anyio
    async def test_add_text_and_index(self, tmp_kb, call_logger_setup, run_live):
        """
        写入一段短文本，验证：
        - 文档状态变为 'ready'
        - 分块数 >= 1
        - embedding 调用日志可见
        """
        if not run_live:
            pytest.skip("跳过集成测试，需加 --run-live 参数")

        kb, store = tmp_kb
        await store.initialize()
        await kb.initialize()

        text = textwrap.dedent("""
            星辰科技有限公司（Starlight Technology Co., Ltd）成立于2018年，
            总部位于上海市浦东新区张江高科技园区。
            公司创始人兼CEO为张明远，CTO为李晓红。
            旗舰产品为智星AI营销平台，服务超过3000家企业客户。
            公司于2021年完成B轮融资5000万美元，由红杉资本中国领投。
        """).strip()

        print(f"\n▶ 写入文档（{len(text)} 字）…")
        doc = await kb.add_text(text, source="test_inline", filename="test.txt")

        print(f"  doc_id  = {doc.doc_id}")
        print(f"  status  = {doc.status}")
        print(f"  chunks  = {doc.chunk_count}")

        assert doc.status == "ready", f"文档应为 ready，实际: {doc.status}  error: {doc.error_msg}"
        assert doc.chunk_count >= 1, "至少应有 1 个分块"

        # 验证 store 中有分块
        stats = await store.get_stats("test")
        assert stats["documents"] == 1
        assert stats["chunks"] >= 1
        print(f"✅ 文档索引成功，分块数={doc.chunk_count}")

    @pytest.mark.anyio
    async def test_add_demo_documents(self, tmp_kb, call_logger_setup, run_live):
        """
        写入所有 demo 文档，验证每份文档都索引成功。
        可在日志中看到多次 embed.request / embed.response 调用。
        """
        if not run_live:
            pytest.skip("跳过集成测试，需加 --run-live 参数")

        kb, store = tmp_kb
        await store.initialize()
        await kb.initialize()

        demo_files = list(DEMO_DIR.glob("*.md"))
        assert len(demo_files) >= 1, f"demo 目录下没有 .md 文件: {DEMO_DIR}"

        print(f"\n▶ 写入 {len(demo_files)} 个 demo 文档…")
        for path in demo_files:
            text = path.read_text(encoding="utf-8")
            print(f"  📄 {path.name} ({len(text)} 字)")
            doc = await kb.add_text(text, source=str(path), filename=path.name)
            print(f"     status={doc.status}  chunks={doc.chunk_count}")
            assert doc.status == "ready", \
                f"{path.name} 索引失败: {doc.error_msg}"

        stats = await store.get_stats("test")
        print(f"\n✅ 全部文档写入完成")
        print(f"   文档数 = {stats['documents']}，分块数 = {stats['chunks']}")
        assert stats["documents"] == len(demo_files)
        assert stats["chunks"] >= len(demo_files)


# ══════════════════════════════════════════════════════════════════════
# 4. 完整 RAG 问答集成测试
# ══════════════════════════════════════════════════════════════════════

class TestKBQALive:
    """
    完整 RAG 管道：写入文档 → 检索 → LLM 生成答案。
    每个测试用例都会在控制台打印完整的请求/响应日志。
    """

    # ── helper ──────────────────────────────────────────────────────

    @staticmethod
    async def _build_kb(kb, store, texts: list[tuple[str, str]]) -> None:
        """往 kb 中写入多条文本，(text, filename) 列表。"""
        await store.initialize()
        await kb.initialize()
        for text, fname in texts:
            doc = await kb.add_text(text, source=fname, filename=fname)
            assert doc.status == "ready", f"{fname} 索引失败: {doc.error_msg}"

    @staticmethod
    async def _rag_answer(kb, chat_engine, question: str, top_k: int = 3) -> tuple[str, list]:
        """
        执行一次完整的 RAG 问答：
          1. 向量+BM25 混合检索（embed.request / embed.response 日志）
          2. 拼装上下文
          3. LLM 生成答案 （chat.request / chat.response 日志）
        返回 (answer_text, retrieved_chunks)。
        """
        from core.models import AgentConfig, Message, MessageRole

        # ── Step 1: 检索 ──────────────────────────────────────────
        chunks = await kb.query(question, top_k=top_k)

        # ── Step 2: 拼装 prompt ───────────────────────────────────
        context = kb.format_context(chunks)
        system_prompt = (
            "你是一个专业的问答助手。请根据下方提供的文档内容回答用户问题。\n"
            "如果文档中没有相关信息，请直接说明。\n"
            "回答时请引用文档中的具体信息，保持简洁。\n\n"
            f"## 参考文档\n{context}"
        )
        messages = [
            Message(role=MessageRole.SYSTEM,  content=system_prompt),
            Message(role=MessageRole.USER,    content=question),
        ]

        # ── Step 3: LLM 生成答案 ──────────────────────────────────
        resp = await chat_engine.chat(messages, [], AgentConfig(max_tokens=512))
        return resp.content or "", chunks

    # ── 测试用例 ─────────────────────────────────────────────────

    @pytest.mark.anyio
    async def test_retrieval_returns_results(
        self, tmp_kb, call_logger_setup, embed_engine, run_live
    ):
        """验证向量+BM25 混合检索能找到相关分块。"""
        if not run_live:
            pytest.skip("跳过集成测试，需加 --run-live 参数")

        kb, store = tmp_kb
        await self._build_kb(kb, store, [
            ("星辰科技有限公司成立于2018年，CEO张明远，CTO李晓红，总部上海浦东。", "company.txt"),
            ("智星AI营销平台是公司的旗舰产品，支持微信、抖音、微博等渠道。", "product.txt"),
            ("今天天气晴，最高气温25度，适合户外活动。", "irrelevant.txt"),
        ])

        question = "公司的CEO是谁？"
        print(f"\n▶ 检索问题: {question!r}")
        chunks = await kb.query(question, top_k=3)

        print(f"  检索到 {len(chunks)} 个分块：")
        for i, c in enumerate(chunks):
            print(f"  [{i+1}] 来源={c.meta.get('filename','?')}  分数={c.score:.4f}")
            print(f"       内容预览: {c.text[:80]}…")

        assert len(chunks) > 0, "检索结果不能为空"
        # 包含 CEO 信息的分块应排在前列
        top_text = " ".join(c.text for c in chunks[:2])
        assert "张明远" in top_text or "CEO" in top_text.upper(), \
            f"包含 CEO 信息的分块未进入 Top-2，实际 Top-2: {top_text[:200]}"
        print(f"✅ 检索正常，相关分块排在前列")

    @pytest.mark.anyio
    async def test_full_rag_qa(self, tmp_kb, call_logger_setup, chat_engine, run_live):
        """
        完整 RAG 问答主测试。
        控制台输出包含：
          - embed.request / embed.response（检索阶段）
          - llm.chat.request / llm.chat.response（生成阶段）
        """
        if not run_live:
            pytest.skip("跳过集成测试，需加 --run-live 参数")

        kb, store = tmp_kb

        # 写入公司概览 demo 文档
        company_doc = (DEMO_DIR / "company_overview.md")
        product_doc = (DEMO_DIR / "product_manual.md")
        texts = []
        if company_doc.exists():
            texts.append((company_doc.read_text(encoding="utf-8"), "company_overview.md"))
        if product_doc.exists():
            texts.append((product_doc.read_text(encoding="utf-8"), "product_manual.md"))
        if not texts:
            texts = [(
                "星辰科技有限公司成立于2018年，CEO张明远，CTO李晓红。"
                "旗舰产品智星AI营销平台，已完成B轮融资5000万美元。",
                "inline.txt",
            )]

        print(f"\n▶ 写入 {len(texts)} 份文档并建立索引…")
        await self._build_kb(kb, store, texts)

        qa_pairs = [
            {
                "question": "星辰科技的创始人是谁？他的职位是什么？",
                "expected_keywords": ["张明远", "CEO", "创始人"],
            },
            {
                "question": "智星AI营销平台支持哪些渠道？",
                "expected_keywords": ["微信", "抖音"],
            },
            {
                "question": "公司完成了多少融资？投资方是谁？",
                "expected_keywords": ["5000万", "红杉"],
            },
        ]

        for qa in qa_pairs:
            question = qa["question"]
            print(f"\n{'='*60}")
            print(f"❓ 问题: {question}")
            print(f"{'='*60}")

            answer, chunks = await self._rag_answer(kb, chat_engine, question)

            print(f"\n📋 检索到 {len(chunks)} 个相关分块")
            for i, c in enumerate(chunks):
                print(f"  [{i+1}] {c.meta.get('filename','?')}  score={c.score:.4f}")

            print(f"\n💬 LLM 回答:\n{textwrap.indent(answer, '   ')}")

            assert answer.strip(), f"问题 {question!r} 的回答不能为空"
            for kw in qa["expected_keywords"]:
                assert kw in answer, (
                    f"回答中应包含关键词 {kw!r}\n"
                    f"实际回答: {answer[:300]}"
                )

        print(f"\n✅ 所有 RAG 问答通过")

    @pytest.mark.anyio
    async def test_rag_unknown_question(self, tmp_kb, call_logger_setup, chat_engine, run_live):
        """
        验证当知识库中没有相关信息时，LLM 应回答"不知道"或"文档中未提及"，
        而不是捏造答案。
        """
        if not run_live:
            pytest.skip("跳过集成测试，需加 --run-live 参数")

        kb, store = tmp_kb
        await self._build_kb(kb, store, [
            ("星辰科技有限公司成立于2018年，CEO张明远。", "company.txt"),
        ])

        question = "星辰科技的股价是多少？"
        print(f"\n▶ 测试超出知识范围的问题: {question!r}")
        answer, chunks = await self._rag_answer(kb, chat_engine, question)

        print(f"\n💬 LLM 回答:\n{textwrap.indent(answer, '   ')}")

        # 不应捏造具体股价数字（简单判断：不应出现像 "100元" 这类明确数字答案）
        assert answer.strip(), "回答不能为空"
        # LLM 应声明无法在文档中找到此信息
        no_info_phrases = ["没有", "未提及", "不知道", "无法", "文档中", "未找到", "无相关"]
        assert any(p in answer for p in no_info_phrases), (
            f"对于超出知识范围的问题，LLM 应表明无法回答，而非捏造\n"
            f"实际回答: {answer[:300]}\n"
            f"（若 LLM 坚持给出了合理的'不确定'声明但用词不同，可调整 no_info_phrases）"
        )
        print(f"✅ LLM 正确声明了无法在文档中找到该信息")

    @pytest.mark.anyio
    async def test_log_file_contains_all_event_types(
        self, tmp_kb, call_logger_setup, chat_engine, run_live
    ):
        """
        执行一次完整 RAG 后，读取 JSONL 日志文件，
        验证包含所有预期事件类型。
        """
        if not run_live:
            pytest.skip("跳过集成测试，需加 --run-live 参数")

        _, log_file = call_logger_setup

        kb, store = tmp_kb
        await self._build_kb(kb, store, [
            ("星辰科技公司成立于2018年，产品包括智星AI营销平台。", "log_test.txt"),
        ])
        await self._rag_answer(kb, chat_engine, "公司成立时间是什么时候？")

        log_path = pathlib.Path(log_file)
        assert log_path.exists(), f"日志文件不存在: {log_file}"

        events: dict[str, int] = {}
        for line in log_path.read_text(encoding="utf-8").strip().splitlines():
            try:
                rec = json.loads(line)
                evt = rec.get("event", "")
                events[evt] = events.get(evt, 0) + 1
            except json.JSONDecodeError:
                pass

        print(f"\n📊 JSONL 日志事件统计：")
        for evt, cnt in sorted(events.items()):
            print(f"   {evt:<40s}  ×{cnt}")

        # 验证必须存在的事件类型
        required = [
            "openai_engine.embed.request",
            "openai_engine.embed.response",
            "llm.chat.request",
            "llm.chat.response",
        ]
        for evt in required:
            assert evt in events, (
                f"日志中缺少事件 '{evt}'\n"
                f"实际事件: {list(events.keys())}"
            )

        print(f"\n✅ 所有预期事件类型均已记录")
        print(f"   embed.request  ×{events.get('openai_engine.embed.request', 0)}")
        print(f"   embed.response ×{events.get('openai_engine.embed.response', 0)}")
        print(f"   chat.request   ×{events.get('llm.chat.request', 0)}")
        print(f"   chat.response  ×{events.get('llm.chat.response', 0)}")
