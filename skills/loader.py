"""
skills/loader.py — Hub Skill 加载器

用法：
    # 加载全部 hub skill
    skills = SkillLoader.load_all()

    # 加载指定 skill
    skill = SkillLoader.load("datetime_tool")

    # 列出所有可用 skill 名称
    names = SkillLoader.available()

    # 每请求的 Skill 过滤
    reg = FilteredSkillRegistry(base_registry, {"weather_current", "datetime_tool"})

Hub 目录约定
────────────
skills/hub/
  <skill_dir>/
    skill.yaml    # ClawHub 兼容清单（必须）
    __init__.py   # Python 实现（必须），包含 entrypoint 类

skill.yaml 最小格式（ClawHub v1 / Claude Tool 兼容）：
────────────────────────────────────────────────────
name: skill_name          # 全局唯一，LLM 看到的 tool name
version: "1.0.0"
description: "..."
author: "..."
tags: [...]
permission: read          # read | write | execute | admin
timeout_ms: 5000
entrypoint: MySkillClass  # __init__.py 中的类名

input_schema:             # JSON Schema，与 Claude tool format 完全兼容
  type: object
  properties: ...
  required: [...]
"""
from __future__ import annotations

import importlib
import pathlib
from typing import Any

import structlog
import yaml

from core.models import PermissionLevel, ToolDescriptor, ToolResult

log = structlog.get_logger(__name__)

HUB_DIR = pathlib.Path(__file__).parent / "hub"

_PERMISSION_MAP = {
    "read":    PermissionLevel.READ,
    "write":   PermissionLevel.WRITE,
    "network": PermissionLevel.NETWORK,
    "exec":    PermissionLevel.EXEC,
}


# ── 清单解析 ───────────────────────────────────────────────────────

def _load_manifest(skill_dir: pathlib.Path) -> dict[str, Any]:
    """读取并解析 skill.yaml，返回原始字典。"""
    yaml_path = skill_dir / "skill.yaml"
    with open(yaml_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _manifest_to_descriptor(manifest: dict[str, Any]) -> ToolDescriptor:
    """把 skill.yaml 转换为 ToolDescriptor（Claude tool 格式兼容）。"""
    perm_str = manifest.get("permission", "read").lower()
    return ToolDescriptor(
        name          = manifest["name"],
        description   = manifest.get("description", "").strip(),
        input_schema  = manifest.get("input_schema", {"type": "object", "properties": {}}),
        source        = "skill",
        permission    = _PERMISSION_MAP.get(perm_str, PermissionLevel.READ),
        timeout_ms    = int(manifest.get("timeout_ms", 30_000)),
        tags          = manifest.get("tags", []),
    )


# ── 单个 Skill 加载 ────────────────────────────────────────────────

def _load_skill_from_dir(skill_dir: pathlib.Path):
    """从目录加载一个 Skill 实例。"""
    manifest   = _load_manifest(skill_dir)
    cls_name   = manifest.get("entrypoint", "Skill")
    pkg_name   = f"skills.hub.{skill_dir.name}"

    module = importlib.import_module(pkg_name)
    cls    = getattr(module, cls_name)
    return cls()


# ── SkillLoader ────────────────────────────────────────────────────

class SkillLoader:
    """
    发现并加载 skills/hub/ 下的 Skill 包。

    支持热安装：只需在 skills/hub/ 下放置新目录（含 skill.yaml + __init__.py），
    重启服务后自动加载，无需修改其他代码。
    """

    @staticmethod
    def available() -> list[str]:
        """返回所有可用的 hub skill 目录名（即 entrypoint 目录名）。"""
        return [
            d.name
            for d in HUB_DIR.iterdir()
            if d.is_dir() and (d / "skill.yaml").exists() and (d / "__init__.py").exists()
        ]

    @staticmethod
    def manifests() -> list[dict[str, Any]]:
        """返回所有 hub skill 的清单列表（不加载 Python 模块）。"""
        result = []
        for d in HUB_DIR.iterdir():
            yaml_path = d / "skill.yaml"
            if d.is_dir() and yaml_path.exists():
                try:
                    result.append(_load_manifest(d))
                except Exception as e:
                    log.warning("hub.manifest_read_failed", dir=d.name, error=str(e))
        return result

    @staticmethod
    def load(dir_name: str):
        """加载指定目录名的 Skill（如 'datetime', 'currency_exchange'）。"""
        skill_dir = HUB_DIR / dir_name
        if not skill_dir.exists():
            raise FileNotFoundError(f"skill hub 目录不存在: {skill_dir}")
        return _load_skill_from_dir(skill_dir)

    @staticmethod
    def load_all() -> list:
        """加载所有 hub skill，跳过加载失败的。"""
        skills = []
        for dir_name in SkillLoader.available():
            try:
                skill = SkillLoader.load(dir_name)
                log.info("hub.skill_loaded",
                         name=skill.descriptor.name,
                         dir=dir_name,
                         version=_load_manifest(HUB_DIR / dir_name).get("version", "?"))
                skills.append(skill)
            except Exception as e:
                log.warning("hub.skill_load_failed", dir=dir_name, error=str(e))
        return skills

    @staticmethod
    def load_by_names(skill_names: list[str]) -> list:
        """
        按 skill.yaml 中的 name 字段加载一组 Skill。
        name 是 LLM 可见的工具名，不是目录名。
        """
        # 建立 name → dir_name 映射
        name_map: dict[str, str] = {}
        for dir_name in SkillLoader.available():
            try:
                m = _load_manifest(HUB_DIR / dir_name)
                name_map[m["name"]] = dir_name
            except Exception:
                pass

        skills = []
        for name in skill_names:
            dir_name = name_map.get(name)
            if not dir_name:
                log.warning("hub.skill_not_found", name=name)
                continue
            try:
                skills.append(SkillLoader.load(dir_name))
            except Exception as e:
                log.warning("hub.skill_load_failed", name=name, error=str(e))
        return skills


# ── FilteredSkillRegistry ─────────────────────────────────────────

class FilteredSkillRegistry:
    """
    包装一个 SkillRegistry，仅暴露指定名称的 Skill。
    用于按请求限制 Agent 可用的工具集，避免修改全局容器。

    兼容 core.interfaces.SkillRegistry Protocol。
    """

    def __init__(self, base, allowed: set[str] | None):
        """
        base    — 底层 SkillRegistry（LocalSkillRegistry）
        allowed — 允许的 skill name 集合；None 表示不限制
        """
        self._base    = base
        self._allowed = allowed   # None = 全部放行

    def _permit(self, name: str) -> bool:
        return self._allowed is None or name in self._allowed

    def register(self, skill) -> None:
        self._base.register(skill)

    def unregister(self, name: str) -> None:
        self._base.unregister(name)

    def get(self, name: str):
        return self._base.get(name) if self._permit(name) else None

    def list_descriptors(self) -> list[ToolDescriptor]:
        return [d for d in self._base.list_descriptors() if self._permit(d.name)]

    async def call(self, name: str, arguments: dict) -> ToolResult:
        if not self._permit(name):
            return ToolResult(
                tool_call_id="",
                tool_name=name,
                content=None,
                error=f"Skill '{name}' 不在本次请求的允许列表中",
            )
        return await self._base.call(name, arguments)
