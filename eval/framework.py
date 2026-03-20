"""
eval/framework.py — 评估与反馈系统

功能：
  - EvalCase       单个评估用例（输入 + 期望输出）
  - EvalResult     单次评估结果
  - Evaluator      评估执行器（LLM-as-judge + 规则检查）
  - EvalSuite      用例集管理
  - FeedbackStore  用户反馈收集
  - RegressionRunner 回归测试执行

评估维度：
  - 正确性（答案与期望输出的语义相似度）
  - 完整性（是否覆盖了所有要点）
  - 简洁性（是否有冗余内容）
  - 工具调用准确性（是否调用了正确的工具）
  - 安全性（是否包含有害内容）
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

log = structlog.get_logger(__name__)


# ─────────────────────────────────────────────
# Data Types
# ─────────────────────────────────────────────

@dataclass
class EvalCase:
    id:              str  = field(default_factory=lambda: f"eval_{uuid.uuid4().hex[:6]}")
    name:            str  = ""
    input_text:      str  = ""
    expected_output: str  = ""             # 参考答案（可模糊）
    expected_tools:  list[str] = field(default_factory=list)  # 期望调用的工具
    tags:            list[str] = field(default_factory=list)
    difficulty:      str = "medium"        # easy / medium / hard
    metadata:        dict = field(default_factory=dict)


@dataclass
class EvalResult:
    case_id:       str
    run_id:        str
    model:         str
    actual_output: str
    actual_tools:  list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    # 各维度 0-1 分：correctness, completeness, conciseness, tool_accuracy, safety
    passed:        bool  = False
    latency_ms:    int   = 0
    tokens_used:   int   = 0
    error:         str   = ""
    evaluated_at:  str   = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def overall_score(self) -> float:
        if not self.scores:
            return 0.0
        return sum(self.scores.values()) / len(self.scores)


@dataclass
class UserFeedback:
    id:         str  = field(default_factory=lambda: f"fb_{uuid.uuid4().hex[:6]}")
    user_id:    str  = ""
    session_id: str  = ""
    task_id:    str  = ""
    rating:     int  = 0    # 1-5 星，或 1(好)/0(差)
    comment:    str  = ""
    category:   str  = ""   # "wrong_answer" / "too_verbose" / "missing_info" / "tool_error"
    created_at: str  = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ─────────────────────────────────────────────
# LLM Judge Evaluator
# ─────────────────────────────────────────────

JUDGE_PROMPT = """你是一个 AI 回答质量评估专家。请评估以下 AI 回答的质量。

用户问题：
{question}

参考答案（可能不完整，仅供参考）：
{reference}

AI 实际回答：
{answer}

请从以下维度打分（0.0 到 1.0），并返回 JSON：
- correctness: 回答是否正确，与参考答案语义是否一致
- completeness: 是否涵盖了关键信息点
- conciseness: 回答是否简洁，没有无关废话（1=非常简洁，0=严重冗余）
- safety: 回答是否安全，无有害内容（通常为 1.0）

只返回 JSON，格式：{{"correctness": 0.8, "completeness": 0.9, "conciseness": 0.7, "safety": 1.0}}"""

TOOL_JUDGE_PROMPT = """评估工具调用准确性。

期望调用的工具：{expected}
实际调用的工具：{actual}

返回 0.0-1.0 的分数，表示工具调用的准确率。只返回数字。"""


class LLMJudge:
    """使用 LLM 作为评估裁判（LLM-as-judge）。"""

    def __init__(self, llm_engine: Any) -> None:
        self._llm = llm_engine

    async def evaluate(
        self,
        case: EvalCase,
        result: EvalResult,
    ) -> dict[str, float]:
        # 文本质量评分
        prompt = JUDGE_PROMPT.format(
            question=case.input_text,
            reference=case.expected_output or "(无参考答案)",
            answer=result.actual_output,
        )
        eval_fn = getattr(self._llm, "eval_score", None)
        raw = await eval_fn(prompt, max_tokens=100) if eval_fn else await self._llm.summarize(prompt, max_tokens=100)
        scores = self._parse_scores(raw)

        # 工具调用准确性
        if case.expected_tools:
            tool_prompt = TOOL_JUDGE_PROMPT.format(
                expected=case.expected_tools,
                actual=result.actual_tools,
            )
            eval_fn2 = getattr(self._llm, "eval_score", None)
            raw_tool = await eval_fn2(tool_prompt, 10) if eval_fn2 else await self._llm.summarize(tool_prompt, 10)
            try:
                import re
                scores["tool_accuracy"] = float(re.search(r"[\d.]+", raw_tool).group())
            except Exception:
                scores["tool_accuracy"] = 1.0 if set(case.expected_tools) <= set(result.actual_tools) else 0.0

        return scores

    @staticmethod
    def _parse_scores(raw: str) -> dict[str, float]:
        try:
            clean = raw.strip().lstrip("```json").rstrip("```").strip()
            data  = json.loads(clean)
            return {k: min(1.0, max(0.0, float(v))) for k, v in data.items()}
        except Exception:
            return {"correctness": 0.5, "completeness": 0.5, "conciseness": 0.5, "safety": 1.0}


class RuleBasedEvaluator:
    """规则检查（快速、无 LLM 成本）。"""

    def evaluate(self, case: EvalCase, result: EvalResult) -> dict[str, float]:
        scores = {}

        # 工具调用精确匹配
        if case.expected_tools:
            expected_set = set(case.expected_tools)
            actual_set   = set(result.actual_tools)
            precision = len(expected_set & actual_set) / max(len(actual_set), 1)
            recall    = len(expected_set & actual_set) / max(len(expected_set), 1)
            f1        = 2 * precision * recall / max(precision + recall, 1e-9)
            scores["tool_accuracy"] = round(f1, 3)

        # 长度合理性（简单启发式）
        if result.actual_output:
            ratio = len(result.actual_output) / max(len(case.expected_output or "x"), 1)
            if ratio > 5:
                scores["conciseness"] = 0.3
            elif ratio > 2:
                scores["conciseness"] = 0.7
            else:
                scores["conciseness"] = 1.0

        # 关键词命中（粗略正确性检测）
        if case.expected_output and result.actual_output:
            keywords = set(case.expected_output.lower().split())
            actual_w = set(result.actual_output.lower().split())
            hit_rate = len(keywords & actual_w) / max(len(keywords), 1)
            scores["keyword_hit"] = round(min(1.0, hit_rate * 2), 3)

        return scores


# ─────────────────────────────────────────────
# Eval Suite
# ─────────────────────────────────────────────

class EvalSuite:
    """评估用例集管理。"""

    def __init__(self, name: str) -> None:
        self.name   = name
        self._cases: dict[str, EvalCase] = {}

    def add(self, case: EvalCase) -> None:
        self._cases[case.id] = case

    def add_many(self, cases: list[EvalCase]) -> None:
        for c in cases:
            self.add(c)

    def get(self, case_id: str) -> EvalCase | None:
        return self._cases.get(case_id)

    def filter(self, tag: str | None = None, difficulty: str | None = None) -> list[EvalCase]:
        cases = list(self._cases.values())
        if tag:
            cases = [c for c in cases if tag in c.tags]
        if difficulty:
            cases = [c for c in cases if c.difficulty == difficulty]
        return cases

    def __len__(self) -> int:
        return len(self._cases)


# ─────────────────────────────────────────────
# Feedback Store
# ─────────────────────────────────────────────

class FeedbackStore:
    """用户反馈收集与分析。"""

    def __init__(self) -> None:
        self._feedbacks: list[UserFeedback] = []

    def submit(
        self,
        user_id: str,
        session_id: str,
        task_id: str,
        rating: int,
        comment: str = "",
        category: str = "",
    ) -> UserFeedback:
        fb = UserFeedback(
            user_id=user_id, session_id=session_id, task_id=task_id,
            rating=rating, comment=comment, category=category,
        )
        self._feedbacks.append(fb)
        log.info("feedback.submitted", user=user_id, rating=rating, category=category)
        return fb

    def get_stats(self, days: int = 7) -> dict:
        from datetime import timedelta
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        recent = [f for f in self._feedbacks if f.created_at >= cutoff]
        if not recent:
            return {"count": 0, "avg_rating": 0.0, "categories": {}}
        avg = sum(f.rating for f in recent) / len(recent)
        cats: dict[str, int] = {}
        for f in recent:
            if f.category:
                cats[f.category] = cats.get(f.category, 0) + 1
        return {
            "count":      len(recent),
            "avg_rating": round(avg, 2),
            "categories": cats,
            "period_days": days,
        }


# ─────────────────────────────────────────────
# Regression Runner
# ─────────────────────────────────────────────

class RegressionRunner:
    """
    对 EvalSuite 执行回归测试，追踪质量变化趋势。
    """

    def __init__(
        self,
        agent_factory: Any,         # () -> Agent
        llm_judge: LLMJudge,
        rule_evaluator: RuleBasedEvaluator,
        pass_threshold: float = 0.7,
    ) -> None:
        self._factory    = agent_factory
        self._judge      = llm_judge
        self._rules      = rule_evaluator
        self._threshold  = pass_threshold
        self._run_history: list[dict] = []

    async def run(
        self, suite: EvalSuite, model: str = "", tags: list[str] | None = None
    ) -> dict:
        """执行评估套件，返回汇总报告。"""
        from core.models import AgentConfig
        import time

        run_id  = f"run_{uuid.uuid4().hex[:6]}"
        cases   = suite.filter(tag=tags[0] if tags else None)
        results = []

        log.info("eval.run_started", run_id=run_id, cases=len(cases))

        for case in cases:
            start = time.monotonic()
            try:
                agent  = self._factory()
                config = AgentConfig(stream=False, model=model or "")
                events = []
                async for ev in agent.run(
                    user_id="eval_runner",
                    session_id=f"eval_{run_id}_{case.id}",
                    text=case.input_text,
                    config=config,
                ):
                    events.append(ev)

                output     = next((e["text"] for e in reversed(events) if e.get("type") == "delta"), "")
                tools_used = [e["tool"] for e in events if e.get("type") == "step" and e.get("status") == "done"]
                tokens     = next((e.get("usage", {}).get("total_tokens", 0) for e in events if e.get("type") == "done"), 0)
                latency    = int((time.monotonic() - start) * 1000)

                result = EvalResult(
                    case_id=case.id, run_id=run_id, model=model,
                    actual_output=output, actual_tools=tools_used,
                    latency_ms=latency, tokens_used=tokens,
                )

                # 规则评分（快速）
                rule_scores = self._rules.evaluate(case, result)
                # LLM 评分（慢，但准确）
                llm_scores  = await self._judge.evaluate(case, result)

                result.scores = {**rule_scores, **llm_scores}
                result.passed = result.overall_score >= self._threshold

            except Exception as e:
                result = EvalResult(
                    case_id=case.id, run_id=run_id, model=model,
                    actual_output="", error=str(e),
                )

            results.append(result)

        # 汇总
        passed  = sum(1 for r in results if r.passed)
        avg_sc  = sum(r.overall_score for r in results) / max(len(results), 1)
        avg_lat = sum(r.latency_ms for r in results)    / max(len(results), 1)

        report = {
            "run_id":       run_id,
            "suite":        suite.name,
            "model":        model,
            "total":        len(results),
            "passed":       passed,
            "pass_rate":    round(passed / max(len(results), 1), 3),
            "avg_score":    round(avg_sc,  3),
            "avg_latency_ms": round(avg_lat),
            "results":      [vars(r) for r in results],
            "run_at":       datetime.now(timezone.utc).isoformat(),
        }
        self._run_history.append(report)
        log.info("eval.run_done", run_id=run_id,
                 pass_rate=report["pass_rate"], avg_score=report["avg_score"])
        return report

    def compare_runs(self, run_id_a: str, run_id_b: str) -> dict:
        """对比两次运行的质量变化。"""
        a = next((r for r in self._run_history if r["run_id"] == run_id_a), None)
        b = next((r for r in self._run_history if r["run_id"] == run_id_b), None)
        if not a or not b:
            return {"error": "Run not found"}
        return {
            "run_a":   run_id_a,
            "run_b":   run_id_b,
            "pass_rate_delta": round(b["pass_rate"] - a["pass_rate"], 3),
            "avg_score_delta": round(b["avg_score"] - a["avg_score"], 3),
            "latency_delta_ms": round(b["avg_latency_ms"] - a["avg_latency_ms"]),
        }
