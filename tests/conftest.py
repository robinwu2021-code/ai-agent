import sys
import os

# ── 修复 stdlib 遮蔽问题 ───────────────────────────────────────────
# 项目根目录有 queue/ 包，会遮蔽 Python stdlib 的 queue 模块，
# 导致 anyio / httpcore 等依赖 `from queue import Queue` 失败。
# 解决方案：在 anyio 加载前，临时移除项目根路径，提前将 stdlib queue
# 缓存进 sys.modules，之后再还原路径。
_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root in sys.path:
    sys.path.remove(_proj_root)
import queue as _stdlib_queue          # noqa: E402 — 缓存 stdlib queue
sys.path.insert(0, _proj_root)         # 还原项目路径

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="运行需要真实外部服务的集成测试（如 Ollama、vLLM）",
    )


@pytest.fixture
def anyio_backend():
    """强制使用 asyncio 后端（避免 anyio 枚举后端时因 queue 遮蔽导致空参数集）。"""
    return "asyncio"
