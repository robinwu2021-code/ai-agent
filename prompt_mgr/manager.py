"""
prompt_mgr/manager.py — Prompt 版本管理与 A/B 测试

功能：
  - PromptTemplate   带变量的 Prompt 模板
  - PromptRegistry   版本化存储（数据库 / 文件）
  - ABTestRouter     灰度发布与 A/B 实验
  - PromptRenderer   模板渲染（支持变量、条件块）
"""
from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

log = structlog.get_logger(__name__)


@dataclass
class PromptTemplate:
    id:          str
    name:        str
    content:     str          # 支持 {variable} 占位符
    version:     str  = "1.0.0"
    description: str  = ""
    tags:        list[str] = field(default_factory=list)
    is_active:   bool = True
    created_at:  str  = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def render(self, **kwargs) -> str:
        """渲染模板，替换所有 {variable} 占位符。"""
        try:
            return self.content.format(**kwargs)
        except KeyError as e:
            log.warning("prompt.render.missing_var", var=str(e), template=self.name)
            # 对缺失变量保留原始占位符
            result = self.content
            for k, v in kwargs.items():
                result = result.replace(f"{{{k}}}", str(v))
            return result

    @property
    def variables(self) -> list[str]:
        """提取模板中的变量名列表。"""
        return re.findall(r"\{(\w+)\}", self.content)

    @property
    def content_hash(self) -> str:
        return hashlib.md5(self.content.encode()).hexdigest()[:8]


class PromptRegistry:
    """
    Prompt 模板版本化存储。
    内存实现（生产替换为 PostgreSQL + 版本历史表）。
    """

    def __init__(self) -> None:
        self._templates: dict[str, list[PromptTemplate]] = {}
        self._active:    dict[str, str] = {}   # name -> version

        # 内置默认 System Prompt
        self._load_defaults()

    def _load_defaults(self) -> None:
        self.register(PromptTemplate(
            id="sys_default_v1",
            name="system_prompt",
            version="1.0.0",
            description="默认 System Prompt",
            content=(
                "你是一个高效的 AI Agent。\n\n"
                "用户 ID: {user_id}\n"
                "当前任务: {task_summary}\n"
                "用户画像: {profile}\n\n"
                "行为规范：\n"
                "- 优先使用工具完成任务，不要凭空猜测\n"
                "- 每次只调用一个工具，观察结果后再决定下一步\n"
                "- 遇到错误，分析原因后尝试替代方案\n"
                "- 输出格式：简洁、结构化，优先使用 Markdown"
            ),
        ))

    def register(self, template: PromptTemplate) -> None:
        name = template.name
        self._templates.setdefault(name, []).append(template)
        self._active[name] = template.version
        log.info("prompt.registered", name=name, version=template.version)

    def get(self, name: str, version: str | None = None) -> PromptTemplate | None:
        templates = self._templates.get(name, [])
        if not templates:
            return None
        if version:
            return next((t for t in templates if t.version == version), None)
        # 返回 active 版本
        active_ver = self._active.get(name)
        return next((t for t in templates if t.version == active_ver), templates[-1])

    def set_active(self, name: str, version: str) -> bool:
        if not self.get(name, version):
            return False
        self._active[name] = version
        log.info("prompt.activated", name=name, version=version)
        return True

    def list_versions(self, name: str) -> list[dict]:
        return [
            {
                "version":   t.version,
                "active":    t.version == self._active.get(name),
                "hash":      t.content_hash,
                "created_at": t.created_at,
            }
            for t in self._templates.get(name, [])
        ]

    def rollback(self, name: str) -> bool:
        versions = self._templates.get(name, [])
        if len(versions) < 2:
            return False
        prev = versions[-2].version
        return self.set_active(name, prev)


class ABTestRouter:
    """
    A/B 测试路由器。
    为同一 Prompt 配置多个变体，按权重分配流量。
    """

    @dataclass
    class Variant:
        name:    str
        version: str
        weight:  float   # 0.0 - 1.0，所有变体权重之和应为 1.0

    def __init__(self, registry: PromptRegistry) -> None:
        self._registry  = registry
        self._experiments: dict[str, list["ABTestRouter.Variant"]] = {}
        self._assignments: dict[str, str] = {}   # user_id -> variant_name (粘性分配)
        self._metrics: dict[str, dict[str, int]] = {}

    def create_experiment(
        self, prompt_name: str, variants: list["ABTestRouter.Variant"]
    ) -> None:
        total = sum(v.weight for v in variants)
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Variant weights must sum to 1.0, got {total}")
        self._experiments[prompt_name] = variants
        self._metrics[prompt_name] = {v.name: 0 for v in variants}
        log.info("ab_test.created", prompt=prompt_name, variants=[v.name for v in variants])

    def assign(self, prompt_name: str, user_id: str) -> PromptTemplate | None:
        """
        为用户分配变体（粘性：同一用户始终得到同一变体）。
        """
        if prompt_name not in self._experiments:
            return self._registry.get(prompt_name)

        # 粘性分配：基于 user_id 哈希
        cache_key = f"{prompt_name}::{user_id}"
        if cache_key not in self._assignments:
            h       = int(hashlib.md5(cache_key.encode()).hexdigest(), 16)
            rand    = (h % 10000) / 10000.0
            cumulative = 0.0
            chosen  = self._experiments[prompt_name][0].name
            for variant in self._experiments[prompt_name]:
                cumulative += variant.weight
                if rand < cumulative:
                    chosen = variant.name
                    break
            self._assignments[cache_key] = chosen

        variant_name = self._assignments[cache_key]
        variant      = next(
            (v for v in self._experiments[prompt_name] if v.name == variant_name), None
        )
        if not variant:
            return self._registry.get(prompt_name)

        self._metrics[prompt_name][variant_name] = \
            self._metrics[prompt_name].get(variant_name, 0) + 1

        return self._registry.get(prompt_name, variant.version)

    def get_metrics(self, prompt_name: str) -> dict:
        return {
            "experiment":   prompt_name,
            "assignments":  self._metrics.get(prompt_name, {}),
        }


class PromptRenderer:
    """
    统一渲染入口：从 Registry 取模板，支持 A/B 分配，渲染变量。
    """

    def __init__(
        self,
        registry: PromptRegistry,
        ab_router: ABTestRouter | None = None,
    ) -> None:
        self._registry = registry
        self._ab       = ab_router

    def render(
        self, name: str, user_id: str = "", **kwargs
    ) -> str:
        if self._ab and user_id:
            template = self._ab.assign(name, user_id)
        else:
            template = self._registry.get(name)

        if not template:
            log.warning("prompt.not_found", name=name)
            return ""

        return template.render(**kwargs)
