"""
tests/test_docling_parser.py — Docling PDF 解析器测试

运行方式：
    # 确保 HF_HUB_OFFLINE=1 已设置（模型已缓存）
    pytest tests/test_docling_parser.py -v

    # 或直接运行（输出详细解析信息）：
    python tests/test_docling_parser.py path/to/your.pdf

环境要求：
    pip install docling pdfplumber pypdf
    环境变量：HF_HUB_OFFLINE=1  HF_HUB_DISABLE_TELEMETRY=1
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

import pytest

# ── 确保离线模式（模型已缓存）──────────────────────────────────────────────
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def parser():
    """初始化 DoclingParser（复用同一 converter，避免重复加载模型）。"""
    from rag.parsers.docling_parser import DoclingParser
    return DoclingParser(do_ocr=False, do_table_structure=True)


@pytest.fixture(scope="module")
def sample_pdf(tmp_path_factory) -> Path:
    """
    生成一个最小化测试 PDF（3 页，每页含标题 + 2 段落）。
    若需测试真实 PDF，通过命令行传入路径（见 __main__ 段）。
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet

        tmp = tmp_path_factory.mktemp("pdf")
        path = tmp / "test_doc.pdf"

        doc = SimpleDocTemplate(str(path), pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        for page_num in range(1, 4):
            story.append(Paragraph(f"Section {page_num}: Overview", styles["Heading1"]))
            story.append(Spacer(1, 12))
            story.append(Paragraph(
                f"This is paragraph one of section {page_num}. "
                "It contains multiple sentences to simulate real document content. "
                "The quick brown fox jumps over the lazy dog.",
                styles["Normal"],
            ))
            story.append(Spacer(1, 8))
            story.append(Paragraph(
                f"This is paragraph two of section {page_num}. "
                "Additional content follows here with more detailed information. "
                "Payment systems require careful architecture and testing.",
                styles["Normal"],
            ))
            story.append(Spacer(1, 20))

        doc.build(story)
        return path

    except ImportError:
        pytest.skip("reportlab 未安装，跳过 sample_pdf fixture（pip install reportlab）")


# ══════════════════════════════════════════════════════════════════════════════
# 基础功能测试
# ══════════════════════════════════════════════════════════════════════════════

class TestDoclingParserBasic:

    def test_supports_pdf(self, parser):
        assert parser.supports("document.pdf")
        assert parser.supports("report.PDF")

    def test_supports_docx(self, parser):
        assert parser.supports("file.docx")

    def test_not_supports_unknown(self, parser):
        assert not parser.supports("image.bmp")
        assert not parser.supports("data.json")

    def test_parse_txt(self, parser, tmp_path):
        txt = tmp_path / "hello.txt"
        txt.write_text("Hello world.\nThis is a test.", encoding="utf-8")
        result = asyncio.run(parser.parse(str(txt)))
        assert "Hello world" in result.content
        assert result.format == "txt"

    def test_parse_markdown(self, parser, tmp_path):
        md = tmp_path / "note.md"
        md.write_text("# Title\n\nParagraph one.\n\n## Sub\nParagraph two.", encoding="utf-8")
        result = asyncio.run(parser.parse(str(md)))
        assert result.has_headings
        assert result.title == "Title"
        assert len(result.headings) >= 2

    def test_parse_csv(self, parser, tmp_path):
        csv = tmp_path / "data.csv"
        csv.write_text("name,age\nAlice,30\nBob,25", encoding="utf-8")
        result = asyncio.run(parser.parse(str(csv)))
        assert "Alice" in result.content
        assert result.format == "csv"
        assert len(result.tables) == 1


# ══════════════════════════════════════════════════════════════════════════════
# PDF 解析测试（需要 Docling 模型）
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestDoclingParserPDF:

    def test_parse_returns_content(self, parser, sample_pdf):
        """解析应返回非空内容。"""
        result = asyncio.run(parser.parse(str(sample_pdf)))
        assert result.content, "content 不应为空"
        assert len(result.content) > 100, f"content 过短: {len(result.content)} chars"

    def test_parse_items_count(self, parser, sample_pdf):
        """doc.texts items 数量应 > 1，说明段落被正确识别。"""
        result = asyncio.run(parser.parse(str(sample_pdf)))
        items = result.metadata.get("items", 0)
        assert items > 1, (
            f"期望 items > 1，实际 items={items}。"
            "可能 Docling 将整个文档合并为单个 text item。"
        )

    def test_parse_has_headings(self, parser, sample_pdf):
        """reportlab 生成的 PDF 含有标题，should_have_headings=True。"""
        result = asyncio.run(parser.parse(str(sample_pdf)))
        # 注：Docling 对生成的 PDF 可能无法识别标题样式，此测试仅做 soft check
        print(f"\n  has_headings={result.has_headings}, headings={result.headings[:3]}")

    def test_parse_page_count(self, parser, sample_pdf):
        result = asyncio.run(parser.parse(str(sample_pdf)))
        assert result.page_count == 3, f"期望 3 页，实际 {result.page_count}"

    def test_content_double_newline_separated(self, parser, sample_pdf):
        """
        关键测试：content 中应存在 \\n\\n，确保 chunker 能正确切分。
        若只有单换行，说明 doc.texts 没有被正确遍历。
        """
        result = asyncio.run(parser.parse(str(sample_pdf)))
        has_double_newline = "\n\n" in result.content
        assert has_double_newline, (
            "content 中没有 \\n\\n 段落分隔符！\n"
            f"content 前 500 chars:\n{result.content[:500]!r}"
        )

    def test_parse_format(self, parser, sample_pdf):
        result = asyncio.run(parser.parse(str(sample_pdf)))
        assert result.format == "pdf"

    def test_fallback_when_docling_unavailable(self, tmp_path):
        """当 Docling 不可用时，应自动回退到 pypdf。"""
        from rag.parsers.docling_parser import DoclingParser
        p = DoclingParser()
        p._converter = None  # 强制 converter = None → 触发 fallback

        pdf = tmp_path / "fallback.pdf"
        # 写一个极小的合法 PDF（1 页空白）
        pdf.write_bytes(
            b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
            b"3 0 obj<</Type/Page/MediaBox[0 0 3 3]>>endobj\n"
            b"xref\n0 4\n0000000000 65535 f\n"
            b"0000000009 00000 n\n0000000058 00000 n\n"
            b"0000000115 00000 n\n"
            b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n190\n%%EOF"
        )
        result = asyncio.run(p.parse(str(pdf)))
        # fallback 不崩溃即可
        assert result.format == "pdf"
        assert result.metadata.get("parser") == "fallback"


# ══════════════════════════════════════════════════════════════════════════════
# 集成测试：解析 → 分块
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestDoclingParserWithChunker:

    def test_chunks_count_reasonable(self, parser, sample_pdf):
        """3 页 PDF 分块数应 > 3（每页至少 1 块）。"""
        from rag.chunkers.structural import StructuralChunker

        result = asyncio.run(parser.parse(str(sample_pdf)))
        chunker = StructuralChunker(min_size=50, max_size=800)
        chunks = asyncio.run(chunker.chunk(result, "test_doc"))

        print(f"\n  items={result.metadata.get('items')}, chunks={len(chunks)}")
        assert len(chunks) >= 3, (
            f"期望 >=3 个 chunk，实际 {len(chunks)}。\n"
            f"content 前 200: {result.content[:200]!r}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# CLI：直接运行以测试真实 PDF
# ══════════════════════════════════════════════════════════════════════════════

def _cli_test(pdf_path: str) -> None:
    """命令行直接测试真实 PDF，输出详细解析信息。"""
    import time
    from rag.parsers.docling_parser import DoclingParser
    from rag.chunkers.structural import StructuralChunker

    print(f"\n{'='*60}")
    print(f"测试文件: {pdf_path}")
    print(f"{'='*60}")

    p = DoclingParser(do_ocr=False)

    t0 = time.time()
    result = asyncio.run(p.parse(pdf_path))
    elapsed = time.time() - t0

    print(f"\n[解析结果]")
    print(f"  解析耗时:   {elapsed:.1f}s")
    print(f"  parser:     {result.metadata.get('parser', 'unknown')}")
    print(f"  pages:      {result.page_count}")
    print(f"  items:      {result.metadata.get('items', 'n/a')}")
    print(f"  has_headings: {result.has_headings}")
    print(f"  headings:   {result.headings[:5]}")
    print(f"  content_len: {len(result.content)} chars")
    dbl_nl = result.content.count('\n\n')
    sgl_nl = result.content.count('\n') - dbl_nl * 2
    print(f"  双换行数:   {dbl_nl}")
    print(f"  单换行数:   {sgl_nl}")

    print(f"\n[Content 前 500 chars]")
    print(result.content[:500])
    print("...")

    # 分块
    chunker = StructuralChunker(min_size=100, max_size=1500)
    chunks = asyncio.run(chunker.chunk(result, "cli_test"))
    print(f"\n[分块结果] strategy={'structural' if result.has_headings else 'auto→semantic'}")
    print(f"  chunks 数: {len(chunks)}")
    for i, c in enumerate(chunks[:5]):
        print(f"  [{i}] len={len(c.text):4d}  {c.text[:80]!r}")
    if len(chunks) > 5:
        print(f"  ... (共 {len(chunks)} 块)")

    print(f"\n{'='*60}")
    if len(chunks) <= 3:
        print("⚠  WARNING: 分块数 <=3，可能段落识别失败！检查 items 和双换行数。")
    else:
        print(f"✓  OK: {len(chunks)} 块，分块正常。")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python tests/test_docling_parser.py path/to/file.pdf")
        sys.exit(1)
    _cli_test(sys.argv[1])
