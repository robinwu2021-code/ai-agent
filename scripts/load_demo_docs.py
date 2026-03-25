"""
scripts/load_demo_docs.py — 将 docs/demo/ 下的示例文档加载到知识库并构建图谱

用法：
    python scripts/load_demo_docs.py
    python scripts/load_demo_docs.py --server http://localhost:8000 --kb_id global
    python scripts/load_demo_docs.py --only-kb      # 只入知识库，不建图谱
    python scripts/load_demo_docs.py --only-graph   # 只建图谱（文档已在库中）
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

try:
    import requests
except ImportError:
    print("请先安装 requests：pip install requests")
    sys.exit(1)

DEMO_DIR = pathlib.Path(__file__).parent.parent / "docs" / "demo"
DEMO_FILES = [
    ("company_overview.md", "星辰科技公司介绍"),
    ("product_manual.md",   "智星AI营销平台产品手册"),
    ("faq.md",              "常见问题解答"),
]


def load_document(server: str, kb_id: str, filepath: pathlib.Path) -> dict | None:
    """向 /kb/documents/text 提交文档内容。"""
    text = filepath.read_text(encoding="utf-8")
    payload = {
        "text":     text,
        "kb_id":    kb_id,
        "source":   str(filepath),
        "filename": filepath.name,
    }
    r = requests.post(f"{server}/kb/documents/text", json=payload, timeout=60)
    if r.status_code == 200:
        return r.json()
    print(f"  ❌ 上传失败 ({r.status_code}): {r.text[:200]}")
    return None


def wait_for_ready(server: str, doc_id: str, timeout: int = 120) -> bool:
    """轮询文档状态直到 ready 或 error。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = requests.get(f"{server}/kb/documents/{doc_id}", timeout=10)
        if r.status_code != 200:
            return False
        status = r.json().get("status", "")
        if status == "ready":
            return True
        if status == "error":
            print(f"  ❌ 索引失败: {r.json().get('error_msg', '')}")
            return False
        print(f"  ⏳ 状态: {status}，等待…")
        time.sleep(3)
    print("  ⚠ 等待超时")
    return False


def build_graph(server: str, doc_id: str, kb_id: str) -> str | None:
    """触发图谱构建，返回 job_id。"""
    r = requests.post(
        f"{server}/kb/build-graph/{doc_id}",
        params={"kb_id": kb_id},
        timeout=30,
    )
    if r.status_code == 200:
        return r.json().get("job_id")
    print(f"  ❌ 图谱构建启动失败 ({r.status_code}): {r.text[:200]}")
    return None


def wait_for_graph_job(server: str, job_id: str, timeout: int = 300) -> bool:
    """轮询图谱构建任务状态。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = requests.get(f"{server}/kg/build/status/{job_id}", timeout=10)
        if r.status_code != 200:
            return False
        status = r.json().get("status", "")
        result = r.json().get("result") or {}
        if status == "done":
            print(f"  ✅ 图谱构建完成：节点 {result.get('nodes_created',0)} / 边 {result.get('edges_created',0)} / 三元组 {result.get('triples_found',0)}")
            return True
        if status == "error":
            print(f"  ❌ 图谱构建失败: {r.json().get('error', '')}")
            return False
        print(f"  ⏳ 图谱构建中 ({status})…")
        time.sleep(5)
    print("  ⚠ 图谱构建超时")
    return False


def main():
    parser = argparse.ArgumentParser(description="加载示例文档到知识库")
    parser.add_argument("--server",     default="http://localhost:8000", help="Agent 服务地址")
    parser.add_argument("--kb_id",      default="global",                help="知识库 ID")
    parser.add_argument("--only-kb",    action="store_true",             help="只入知识库，不建图谱")
    parser.add_argument("--only-graph", action="store_true",             help="只建图谱（不重新上传文档）")
    args = parser.parse_args()

    server = args.server.rstrip("/")

    # ── 健康检查 ──────────────────────────────────────────────────
    try:
        r = requests.get(f"{server}/health", timeout=5)
        r.raise_for_status()
        print(f"✅ 服务在线: {server}")
    except Exception as exc:
        print(f"❌ 无法连接服务 {server}: {exc}")
        print("请先启动服务：python server.py")
        sys.exit(1)

    doc_ids: list[tuple[str, str]] = []   # [(doc_id, filename)]

    # ── Step 1：上传文档 ──────────────────────────────────────────
    if not args.only_graph:
        print(f"\n📄 加载示例文档到知识库 kb_id={args.kb_id}\n")
        for filename, desc in DEMO_FILES:
            filepath = DEMO_DIR / filename
            if not filepath.exists():
                print(f"  ⚠ 文件不存在: {filepath}")
                continue

            print(f"  上传 [{desc}] → {filename}")
            doc = load_document(server, args.kb_id, filepath)
            if not doc:
                continue

            doc_id = doc.get("doc_id", "")
            status = doc.get("status", "")
            print(f"  📁 doc_id={doc_id}  状态={status}")

            # 等待索引完成
            if status != "ready":
                ok = wait_for_ready(server, doc_id)
                if not ok:
                    print(f"  ⚠ 跳过图谱构建（索引未就绪）")
                    continue

            chunks = doc.get("chunk_count", "?")
            print(f"  ✅ 索引完成，分块数: {chunks}")
            doc_ids.append((doc_id, filename))

    else:
        # 从服务器拉取已有文档
        print(f"\n📋 获取知识库文档列表 kb_id={args.kb_id}")
        r = requests.get(f"{server}/kb/documents", params={"kb_id": args.kb_id}, timeout=10)
        if r.status_code == 200:
            for d in r.json().get("documents", []):
                if d.get("status") == "ready":
                    doc_ids.append((d["doc_id"], d.get("filename", "")))
                    print(f"  📁 {d['filename']}  doc_id={d['doc_id']}")

    # ── Step 2：构建图谱 ──────────────────────────────────────────
    if not args.only_kb and doc_ids:
        print(f"\n🕸  构建知识图谱 kb_id={args.kb_id}\n")
        for doc_id, filename in doc_ids:
            print(f"  构建 [{filename}]…")
            job_id = build_graph(server, doc_id, args.kb_id)
            if not job_id:
                continue
            print(f"  🏃 job_id={job_id}")
            wait_for_graph_job(server, job_id)

    # ── 统计 ──────────────────────────────────────────────────────
    print("\n📊 最终统计\n")
    try:
        r = requests.get(f"{server}/kb/stats", params={"kb_id": args.kb_id}, timeout=10)
        kb_stats = r.json()
        print(f"  知识库：{kb_stats.get('documents',0)} 文档 / {kb_stats.get('chunks',0)} 分块")
    except Exception:
        pass
    try:
        r = requests.get(f"{server}/kg/stats", params={"kb_id": args.kb_id}, timeout=10)
        kg_stats = r.json()
        print(f"  图谱：{kg_stats.get('nodes',0)} 节点 / {kg_stats.get('edges',0)} 边 / {kg_stats.get('communities',0)} 社区")
    except Exception:
        pass

    print("\n🎉 完成！访问以下地址体验功能：")
    print(f"   知识库问答：http://localhost:3000/knowledge")
    print(f"   知识图谱：  http://localhost:3000/graph")


if __name__ == "__main__":
    main()
