"""
evolution/config.py — 进化模块配置

通过 evolution_config.yaml 驱动，所有参数均可覆盖。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SignalConfig:
    rag_enabled:             bool  = True
    bi_enabled:              bool  = True
    # 隐式信号：用户在此窗口内未追问则视为"接受"
    implicit_accept_window:  int   = 30    # 秒


@dataclass
class SchedulerConfig:
    daily_job_hour:   int  = 2      # 凌晨 2 点执行日任务
    daily_job_minute: int  = 0
    timezone:         str  = "Asia/Shanghai"


@dataclass
class AnalyzerConfig:
    chunk_scorer_enabled:     bool  = True
    low_quality_threshold:    float = 0.3    # 低于此分视为低质量 chunk
    bi_profiler_enabled:      bool  = True
    qa_extractor_enabled:     bool  = True
    qa_min_feedback_score:    float = 4.0    # 反馈分 ≥ 此值才提取 Q&A


@dataclass
class ActorConfig:
    kb_injector_enabled:     bool  = True
    auto_inject:             bool  = True    # False = 进入人工审核队列
    param_tuner_enabled:     bool  = True
    template_builder_enabled: bool = True
    prompt_updater_enabled:  bool  = True
    prompt_auto_apply:       bool  = True    # False = 人工审核后生效
    stale_chunk_days:        int   = 90      # N 天未命中视为过期
    stale_hit_threshold:     int   = 3       # 命中次数 < N 也视为过期


@dataclass
class StorageConfig:
    """
    存储后端配置。

    backend 可选值：
      sqlite   — 全部数据存入 SQLite（默认，向后兼容）
      markdown — 画像/模板/Q&A/配置 → Markdown 文件；信号/Chunk → 轻量 SQLite
      hybrid   — SQLite 处理全部 + Markdown 同步镜像画像类数据（推荐）

    切换说明：
      - sqlite  → hybrid：无需迁移，现有 evolution.db 继续使用，新增 Markdown 输出
      - sqlite  → markdown：信号历史会丢失（仅保留新产生信号），慎用
      - hybrid  → markdown：同上
    """
    backend:         str  = "sqlite"                  # sqlite | markdown | hybrid
    sqlite_path:     str  = "./data/evolution.db"     # SQLite 数据库路径
    markdown_dir:    str  = "./data/evolution"        # Markdown 根目录
    signals_db_path: str  = "./data/evolution_signals.db"  # markdown 模式下的信号库


@dataclass
class EvolutionConfig:
    enabled:   bool             = True
    db_path:   str              = "./data/evolution.db"   # 兼容旧配置，storage.sqlite_path 优先
    storage:   StorageConfig    = field(default_factory=StorageConfig)
    signal:    SignalConfig     = field(default_factory=SignalConfig)
    scheduler: SchedulerConfig  = field(default_factory=SchedulerConfig)
    analyzer:  AnalyzerConfig   = field(default_factory=AnalyzerConfig)
    actor:     ActorConfig      = field(default_factory=ActorConfig)

    # 供未来新功能模块扩展的钩子列表
    # 格式: [{"module": "myapp.plugin", "class": "MySignalCollector", "config": {...}}]
    extra_signal_collectors: list[dict] = field(default_factory=list)
    extra_analyzers:         list[dict] = field(default_factory=list)
    extra_actors:            list[dict] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str = "evolution_config.yaml") -> "EvolutionConfig":
        p = Path(path)
        if not p.exists():
            return cls()
        raw: dict = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        ev = raw.get("evolution", raw)   # 支持顶层或嵌套 "evolution:" key
        return cls._from_dict(ev)

    @classmethod
    def _from_dict(cls, d: dict) -> "EvolutionConfig":
        def _merge(dataclass_type, source: dict):
            """将 dict 安全地合并到 dataclass，忽略未知键。"""
            import dataclasses
            fields = {f.name for f in dataclasses.fields(dataclass_type)}
            kwargs = {k: v for k, v in source.items() if k in fields}
            return dataclass_type(**kwargs)

        cfg = cls(
            enabled = d.get("enabled", True),
            db_path = d.get("db_path", "./data/evolution.db"),
        )
        if "storage" in d:
            cfg.storage = _merge(StorageConfig, d["storage"])
            # db_path 向后兼容：若 storage.sqlite_path 未自定义，使用旧 db_path
            if cfg.storage.sqlite_path == "./data/evolution.db" and cfg.db_path != "./data/evolution.db":
                cfg.storage.sqlite_path = cfg.db_path
        else:
            # 没有 storage 节时，用旧 db_path 作为 sqlite_path
            cfg.storage.sqlite_path = cfg.db_path

        if "signal" in d:
            cfg.signal = _merge(SignalConfig, d["signal"])
        if "scheduler" in d:
            cfg.scheduler = _merge(SchedulerConfig, d["scheduler"])
        if "analyzer" in d:
            cfg.analyzer = _merge(AnalyzerConfig, d["analyzer"])
        if "actor" in d:
            cfg.actor = _merge(ActorConfig, d["actor"])

        cfg.extra_signal_collectors = d.get("extra_signal_collectors", [])
        cfg.extra_analyzers         = d.get("extra_analyzers", [])
        cfg.extra_actors            = d.get("extra_actors", [])
        return cfg
