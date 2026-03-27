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
class EvolutionConfig:
    enabled:   bool             = True
    db_path:   str              = "./data/evolution.db"
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
