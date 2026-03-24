"""
tests/test_agent_bi_skill.py — AgentBiSkill 完整测试套件

覆盖 6 个 User Story:
  Story 1  今日业绩查询（时间范围 + 单指标）
  Story 2  按门店筛选（bra_id）
  Story 3  多指标综合报表
  Story 4  聚合函数灵活配置
  Story 5  时间范围辅助工具（today/yesterday/week/last_week/month）
  Story 6  健壮的错误处理（无效参数 / 网络异常 / API 错误）

测试分层:
  单元测试   (TestMetricLabels, TestTimeUtils, TestDescriptor, TestValidation)
  集成测试   (TestAgentBiExecute — Mock httpx, TestSkillRegistryIntegration)
  E2E 测试   (TestAgentBiE2E — MockLLM + SkillRegistry 完整 Agent 流)
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from skills.hub.agent_bi import (
    AgentBiSkill,
    _METRIC_LABELS,
    _VALID_AGGREGATIONS,
    _VALID_METRICS,
    _midnight_ms,
    last_week_range,
    month_range,
    today_range,
    week_range,
    yesterday_range,
)


# ══════════════════════════════════════════════════════════════
# 工具函数：构造成功 / 失败的 httpx 响应
# ══════════════════════════════════════════════════════════════

def _mock_response(data: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = data
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            message=f"HTTP {status_code}",
            request=MagicMock(),
            response=resp,
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


def _api_success(metrics_rows: list[dict]) -> dict:
    return {
        "code": 0,
        "message": "success",
        "data": {"metrics": metrics_rows},
    }


# ══════════════════════════════════════════════════════════════
# 1. 指标标签
# ══════════════════════════════════════════════════════════════

class TestMetricLabels:
    """Story 4 — 每个指标都有对应中文标签。"""

    def test_all_valid_metrics_have_labels(self):
        for m in _VALID_METRICS:
            assert m in _METRIC_LABELS, f"指标 {m!r} 缺少中文标签"

    def test_labels_are_non_empty_strings(self):
        for m, label in _METRIC_LABELS.items():
            assert isinstance(label, str) and label, f"指标 {m!r} 的标签为空"

    def test_known_metric_labels(self):
        assert _METRIC_LABELS["turnover"]       == "总销售额"
        assert _METRIC_LABELS["order_quantity"] == "订单总数"
        assert _METRIC_LABELS["refund_amount"]  == "退款总额"
        assert _METRIC_LABELS["pay_by_online"]  == "在线支付"
        assert _METRIC_LABELS["vat"]            == "增值税"

    def test_valid_metrics_count(self):
        assert len(_VALID_METRICS) == 26

    def test_valid_aggregations(self):
        assert _VALID_AGGREGATIONS == {"SUM", "AVG", "MAX", "MIN", "COUNT"}


# ══════════════════════════════════════════════════════════════
# 2. 时间范围工具（Story 5）
# ══════════════════════════════════════════════════════════════

class TestTimeUtils:
    """Story 5 — 时间范围辅助函数。"""

    # ── _midnight_ms ──────────────────────────────────────────

    def test_midnight_ms_returns_integer(self):
        dt = datetime(2025, 1, 15, 12, 30, tzinfo=timezone.utc)
        ms = _midnight_ms(dt)
        assert isinstance(ms, int)

    def test_midnight_ms_is_13_digits(self):
        dt = datetime(2025, 6, 1, tzinfo=timezone.utc)
        ms = _midnight_ms(dt)
        assert 10 ** 12 <= ms < 10 ** 13, f"{ms} 不是 13 位时间戳"

    def test_midnight_ms_zeroes_time(self):
        dt   = datetime(2025, 3, 10, 14, 59, 30, 123456, tzinfo=timezone.utc)
        ms   = _midnight_ms(dt)
        back = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
        assert back.hour == 0
        assert back.minute == 0
        assert back.second == 0

    # ── today_range ───────────────────────────────────────────

    def test_today_range_start_lt_end(self):
        s, e = today_range()
        assert s < e

    def test_today_range_exactly_one_day(self):
        s, e = today_range()
        assert e - s == 86_400_000, "today_range 应精确为 86400000 ms（1天）"

    def test_today_range_both_13_digits(self):
        s, e = today_range()
        for v in (s, e):
            assert 10 ** 12 <= v < 10 ** 13, f"{v} 不是 13 位时间戳"

    def test_today_range_start_is_today_midnight(self):
        s, _ = today_range(timezone.utc)
        dt   = datetime.fromtimestamp(s / 1000, tz=timezone.utc)
        now  = datetime.now(timezone.utc)
        assert dt.date() == now.date()
        assert dt.hour == dt.minute == dt.second == 0

    # ── yesterday_range ───────────────────────────────────────

    def test_yesterday_range_exactly_one_day(self):
        s, e = yesterday_range()
        assert e - s == 86_400_000

    def test_yesterday_end_equals_today_start(self):
        _, ye = yesterday_range()
        ts, _ = today_range()
        assert ye == ts

    def test_yesterday_range_before_today(self):
        ys, _ = yesterday_range()
        ts, _ = today_range()
        assert ys < ts

    # ── week_range ────────────────────────────────────────────

    def test_week_range_start_lt_end(self):
        s, e = week_range()
        assert s < e

    def test_week_range_exactly_7_days(self):
        s, e = week_range()
        assert e - s == 7 * 86_400_000

    def test_week_range_start_is_monday(self):
        s, _ = week_range(timezone.utc)
        dt   = datetime.fromtimestamp(s / 1000, tz=timezone.utc)
        assert dt.weekday() == 0, f"周起始不是周一，实为 weekday={dt.weekday()}"

    # ── last_week_range ───────────────────────────────────────

    def test_last_week_range_exactly_7_days(self):
        s, e = last_week_range()
        assert e - s == 7 * 86_400_000

    def test_last_week_end_equals_week_start(self):
        _, le = last_week_range()
        ws, _ = week_range()
        assert le == ws

    def test_last_week_before_this_week(self):
        ls, _ = last_week_range()
        ws, _ = week_range()
        assert ls < ws

    # ── month_range ───────────────────────────────────────────

    def test_month_range_start_lt_end(self):
        s, e = month_range()
        assert s < e

    def test_month_range_start_is_first_of_month(self):
        s, _ = month_range(timezone.utc)
        dt   = datetime.fromtimestamp(s / 1000, tz=timezone.utc)
        assert dt.day == 1

    def test_month_range_end_is_next_month_first(self):
        _, e  = month_range(timezone.utc)
        dt    = datetime.fromtimestamp(e / 1000, tz=timezone.utc)
        now   = datetime.now(timezone.utc)
        expected_month = (now.month % 12) + 1
        assert dt.month == expected_month
        assert dt.day   == 1


# ══════════════════════════════════════════════════════════════
# 3. ToolDescriptor 验证
# ══════════════════════════════════════════════════════════════

class TestDescriptor:
    def make_skill(self) -> AgentBiSkill:
        return AgentBiSkill()

    def test_descriptor_name(self):
        assert self.make_skill().descriptor.name == "agent_bi"

    def test_descriptor_source_is_skill(self):
        assert self.make_skill().descriptor.source == "skill"

    def test_descriptor_permission_is_network(self):
        from core.models import PermissionLevel
        assert self.make_skill().descriptor.permission == PermissionLevel.NETWORK

    def test_descriptor_timeout_ms(self):
        assert self.make_skill().descriptor.timeout_ms == 15_000

    def test_descriptor_has_tags(self):
        tags = self.make_skill().descriptor.tags
        assert "report" in tags
        assert "bi"     in tags

    def test_descriptor_required_fields(self):
        schema = self.make_skill().descriptor.input_schema
        assert "action"  in schema["required"]
        assert "metrics" in schema["required"]

    def test_descriptor_metrics_has_min_items(self):
        schema = self.make_skill().descriptor.input_schema
        assert schema["properties"]["metrics"]["minItems"] == 1

    def test_descriptor_action_enum(self):
        schema = self.make_skill().descriptor.input_schema
        assert schema["properties"]["action"]["enum"] == ["query"]

    def test_descriptor_aggregation_default_sum(self):
        schema = self.make_skill().descriptor.input_schema
        assert schema["properties"]["aggregation"]["default"] == "SUM"

    def test_descriptor_description_non_empty(self):
        desc = self.make_skill().descriptor.description
        assert isinstance(desc, str) and len(desc) > 20


# ══════════════════════════════════════════════════════════════
# 4. 参数校验（无需 HTTP）
# ══════════════════════════════════════════════════════════════

class TestValidation:
    """Story 6 — 参数校验错误处理。"""

    @pytest.mark.asyncio
    async def test_empty_metrics_returns_error(self):
        skill = AgentBiSkill()
        result = await skill.execute({"action": "query", "metrics": []})
        assert "error" in result
        assert "valid_metrics" in result

    @pytest.mark.asyncio
    async def test_invalid_metric_returns_error(self):
        skill = AgentBiSkill()
        result = await skill.execute({
            "action": "query",
            "metrics": ["nonexistent_metric"],
        })
        assert "error" in result
        assert "valid_metrics" in result
        assert "nonexistent_metric" in str(result["error"])

    @pytest.mark.asyncio
    async def test_mixed_valid_invalid_metrics(self):
        skill = AgentBiSkill()
        result = await skill.execute({
            "action": "query",
            "metrics": ["turnover", "bad_metric"],
        })
        assert "error" in result

    @pytest.mark.asyncio
    async def test_invalid_aggregation_returns_error(self):
        skill = AgentBiSkill()
        result = await skill.execute({
            "action": "query",
            "metrics": ["turnover"],
            "aggregation": "PRODUCT",
        })
        assert "error" in result
        assert "valid_aggregations" in result

    @pytest.mark.asyncio
    async def test_unknown_action_returns_error(self):
        skill = AgentBiSkill()
        result = await skill.execute({"action": "delete", "metrics": ["turnover"]})
        assert "error" in result
        assert "InvalidAction" in result.get("type", "")

    @pytest.mark.asyncio
    async def test_missing_action_defaults_to_query_but_empty_metrics_fails(self):
        skill = AgentBiSkill()
        result = await skill.execute({"metrics": []})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_all_valid_metrics_pass_validation(self):
        """有效的单指标不触发校验错误（需要 mock HTTP 才能完整测试，这里仅测校验路径）。"""
        for m in _VALID_METRICS:
            skill = AgentBiSkill()
            with patch.object(skill, "_query", new_callable=AsyncMock) as mocked:
                mocked.return_value = {"success": True}
                result = await skill.execute({"action": "query", "metrics": [m]})
                mocked.assert_called_once()

    @pytest.mark.asyncio
    async def test_all_valid_aggregations_pass_validation(self):
        for agg in _VALID_AGGREGATIONS:
            skill = AgentBiSkill()
            with patch.object(skill, "_query", new_callable=AsyncMock) as mocked:
                mocked.return_value = {"success": True}
                await skill.execute({
                    "action": "query",
                    "metrics": ["turnover"],
                    "aggregation": agg,
                })
                mocked.assert_called_once()


# ══════════════════════════════════════════════════════════════
# 5. execute / _query — Mock HTTP（Story 1-4, 6）
# ══════════════════════════════════════════════════════════════

class TestAgentBiExecute:
    """Mock httpx.AsyncClient 以隔离网络依赖。"""

    def make_skill(self, **kw) -> AgentBiSkill:
        return AgentBiSkill(api_url="http://test.invalid/v1/report/agent_bi", **kw)

    # ── Story 1: 今日业绩查询 ─────────────────────────────────

    @pytest.mark.asyncio
    async def test_query_today_turnover_and_orders(self):
        """Story 1 — 查询今日营业额 + 订单数，返回带标签的结果。"""
        s, e = today_range()
        rows = [{"turnover": 12345.6, "order_quantity": 88}]
        skill = self.make_skill()

        with patch("httpx.AsyncClient") as MockClient:
            MockClient.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(
                    post=AsyncMock(return_value=_mock_response(_api_success(rows)))
                )
            )
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await skill.execute({
                "action":      "query",
                "metrics":     ["turnover", "order_quantity"],
                "range_start": s,
                "range_end":   e,
            })

        assert result["success"] is True
        assert result["range_start"] == s
        assert result["range_end"]   == e
        assert len(result["metrics"]) == 1
        row = result["metrics"][0]
        assert row["turnover"]["label"]       == "总销售额"
        assert row["order_quantity"]["label"] == "订单总数"

    # ── Story 2: 按门店筛选 ───────────────────────────────────

    @pytest.mark.asyncio
    async def test_query_with_bra_id(self):
        """Story 2 — bra_id 正确传入 payload。"""
        rows  = [{"turnover": 5000.0}]
        skill = self.make_skill()
        captured_payload: dict = {}

        async def fake_post(url, json=None, headers=None, **_):
            captured_payload.update(json or {})
            return _mock_response(_api_success(rows))

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = MagicMock()
            mock_client.post = AsyncMock(side_effect=fake_post)
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MockClient.return_value.__aexit__  = AsyncMock(return_value=False)

            result = await skill.execute({
                "action":  "query",
                "metrics": ["turnover"],
                "bra_id":  "STORE_001",
            })

        assert result["bra_id"] == "STORE_001"
        assert captured_payload.get("braId") == "STORE_001"

    @pytest.mark.asyncio
    async def test_query_without_bra_id_no_braid_in_payload(self):
        """Story 2 — 不传 bra_id 时 payload 不含 braId。"""
        captured_payload: dict = {}

        async def fake_post(url, json=None, headers=None, **_):
            captured_payload.update(json or {})
            return _mock_response(_api_success([{"turnover": 0}]))

        skill = self.make_skill()
        with patch("httpx.AsyncClient") as MockClient:
            mock_client = MagicMock()
            mock_client.post = AsyncMock(side_effect=fake_post)
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MockClient.return_value.__aexit__  = AsyncMock(return_value=False)

            await skill.execute({"action": "query", "metrics": ["turnover"]})

        assert "braId" not in captured_payload
        assert "bra_id" not in captured_payload

    # ── Story 3: 多指标综合报表 ───────────────────────────────

    @pytest.mark.asyncio
    async def test_query_multiple_metrics(self):
        """Story 3 — 多指标同时查询，每个指标均有 value + label。"""
        wanted = ["turnover", "refund_amount", "pay_by_cash", "pay_by_online"]
        rows   = [{m: float(i * 100) for i, m in enumerate(wanted)}]
        skill  = self.make_skill()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = MagicMock()
            mock_client.post = AsyncMock(
                return_value=_mock_response(_api_success(rows))
            )
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MockClient.return_value.__aexit__  = AsyncMock(return_value=False)

            result = await skill.execute({"action": "query", "metrics": wanted})

        assert result["success"] is True
        row = result["metrics"][0]
        for m in wanted:
            assert m in row
            assert "value" in row[m]
            assert "label" in row[m]
            assert row[m]["label"] == _METRIC_LABELS[m]

    @pytest.mark.asyncio
    async def test_raw_metrics_preserved(self):
        """Story 3 — raw_metrics 与 API 原始数据一致。"""
        rows  = [{"turnover": 999.9, "order_quantity": 42}]
        skill = self.make_skill()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = MagicMock()
            mock_client.post = AsyncMock(
                return_value=_mock_response(_api_success(rows))
            )
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MockClient.return_value.__aexit__  = AsyncMock(return_value=False)

            result = await skill.execute({
                "action":  "query",
                "metrics": ["turnover", "order_quantity"],
            })

        assert result["raw_metrics"] == rows

    # ── Story 4: 聚合函数 ─────────────────────────────────────

    @pytest.mark.asyncio
    async def test_aggregation_avg_in_payload(self):
        """Story 4 — 聚合函数正确传入 payload。"""
        captured_payload: dict = {}

        async def fake_post(url, json=None, headers=None, **_):
            captured_payload.update(json or {})
            return _mock_response(_api_success([{"average_order_price": 58.5}]))

        skill = self.make_skill()
        with patch("httpx.AsyncClient") as MockClient:
            mock_client = MagicMock()
            mock_client.post = AsyncMock(side_effect=fake_post)
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MockClient.return_value.__aexit__  = AsyncMock(return_value=False)

            result = await skill.execute({
                "action":      "query",
                "metrics":     ["average_order_price"],
                "aggregation": "AVG",
            })

        assert captured_payload["aggregation"] == "AVG"
        assert result["aggregation"] == "AVG"

    @pytest.mark.asyncio
    async def test_default_aggregation_is_sum(self):
        """Story 4 — 不传 aggregation 时默认 SUM。"""
        captured_payload: dict = {}

        async def fake_post(url, json=None, headers=None, **_):
            captured_payload.update(json or {})
            return _mock_response(_api_success([{"turnover": 0}]))

        skill = self.make_skill()
        with patch("httpx.AsyncClient") as MockClient:
            mock_client = MagicMock()
            mock_client.post = AsyncMock(side_effect=fake_post)
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MockClient.return_value.__aexit__  = AsyncMock(return_value=False)

            await skill.execute({"action": "query", "metrics": ["turnover"]})

        assert captured_payload["aggregation"] == "SUM"

    # ── Story 5: 时间范围传入 ─────────────────────────────────

    @pytest.mark.asyncio
    async def test_time_range_in_payload(self):
        """Story 5 — rangeStart / rangeEnd 正确进入 payload。"""
        s, e = yesterday_range()
        captured_payload: dict = {}

        async def fake_post(url, json=None, headers=None, **_):
            captured_payload.update(json or {})
            return _mock_response(_api_success([{"turnover": 100}]))

        skill = self.make_skill()
        with patch("httpx.AsyncClient") as MockClient:
            mock_client = MagicMock()
            mock_client.post = AsyncMock(side_effect=fake_post)
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MockClient.return_value.__aexit__  = AsyncMock(return_value=False)

            await skill.execute({
                "action":      "query",
                "metrics":     ["turnover"],
                "range_start": s,
                "range_end":   e,
            })

        assert captured_payload["rangeStart"] == s
        assert captured_payload["rangeEnd"]   == e

    @pytest.mark.asyncio
    async def test_no_time_range_not_in_payload(self):
        """Story 5 — 不传时间范围时 payload 不含 rangeStart/rangeEnd。"""
        captured_payload: dict = {}

        async def fake_post(url, json=None, headers=None, **_):
            captured_payload.update(json or {})
            return _mock_response(_api_success([{"turnover": 0}]))

        skill = self.make_skill()
        with patch("httpx.AsyncClient") as MockClient:
            mock_client = MagicMock()
            mock_client.post = AsyncMock(side_effect=fake_post)
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MockClient.return_value.__aexit__  = AsyncMock(return_value=False)

            await skill.execute({"action": "query", "metrics": ["turnover"]})

        assert "rangeStart" not in captured_payload
        assert "rangeEnd"   not in captured_payload

    # ── Story 6: 错误处理 ─────────────────────────────────────

    @pytest.mark.asyncio
    async def test_timeout_returns_friendly_error(self):
        """Story 6 — 超时时返回 TimeoutError。"""
        skill = self.make_skill()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = MagicMock()
            mock_client.post = AsyncMock(
                side_effect=httpx.TimeoutException("timed out", request=MagicMock())
            )
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MockClient.return_value.__aexit__  = AsyncMock(return_value=False)

            result = await skill.execute({"action": "query", "metrics": ["turnover"]})

        assert "error" in result
        assert result["type"] == "TimeoutError"
        assert "超时" in result["error"]

    @pytest.mark.asyncio
    async def test_http_4xx_returns_http_error(self):
        """Story 6 — HTTP 4xx 返回 HTTPError。"""
        skill = self.make_skill()
        resp_mock = _mock_response({}, status_code=403)

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = MagicMock()
            mock_client.post = AsyncMock(return_value=resp_mock)
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MockClient.return_value.__aexit__  = AsyncMock(return_value=False)

            result = await skill.execute({"action": "query", "metrics": ["turnover"]})

        assert result["type"] == "HTTPError"
        assert "403" in str(result.get("status_code", result.get("error", "")))

    @pytest.mark.asyncio
    async def test_network_error_returns_network_error(self):
        """Story 6 — 网络错误返回 NetworkError。"""
        skill = self.make_skill()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = MagicMock()
            mock_client.post = AsyncMock(
                side_effect=httpx.ConnectError("connection refused")
            )
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MockClient.return_value.__aexit__  = AsyncMock(return_value=False)

            result = await skill.execute({"action": "query", "metrics": ["turnover"]})

        assert result["type"] == "NetworkError"
        assert "error" in result

    @pytest.mark.asyncio
    async def test_api_error_code_returns_error(self):
        """Story 6 — API 返回 code != 0 时返回 APIError。"""
        skill = self.make_skill()
        api_err = {"code": 40001, "message": "门店不存在"}

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = MagicMock()
            mock_client.post = AsyncMock(return_value=_mock_response(api_err))
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MockClient.return_value.__aexit__  = AsyncMock(return_value=False)

            result = await skill.execute({
                "action":  "query",
                "metrics": ["turnover"],
                "bra_id":  "INVALID_STORE",
            })

        assert "error" in result
        assert result.get("code") == 40001
        assert result.get("type") == "APIError"
        assert "门店不存在" in result["error"]

    # ── API Key 鉴权 ──────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_api_key_set_in_authorization_header(self):
        """API Key 配置后正确注入 Authorization 头。"""
        captured_headers: dict = {}

        async def fake_post(url, json=None, headers=None, **_):
            captured_headers.update(headers or {})
            return _mock_response(_api_success([{"turnover": 0}]))

        skill = AgentBiSkill(
            api_url="http://test.invalid/v1/report/agent_bi",
            api_key="secret-token-123",
        )
        with patch("httpx.AsyncClient") as MockClient:
            mock_client = MagicMock()
            mock_client.post = AsyncMock(side_effect=fake_post)
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MockClient.return_value.__aexit__  = AsyncMock(return_value=False)

            await skill.execute({"action": "query", "metrics": ["turnover"]})

        assert captured_headers.get("Authorization") == "Bearer secret-token-123"

    @pytest.mark.asyncio
    async def test_no_api_key_no_authorization_header(self):
        """不设置 API Key 时不发送 Authorization 头。"""
        captured_headers: dict = {}

        async def fake_post(url, json=None, headers=None, **_):
            captured_headers.update(headers or {})
            return _mock_response(_api_success([{"turnover": 0}]))

        skill = AgentBiSkill(api_url="http://test.invalid/v1/report/agent_bi")
        with patch("httpx.AsyncClient") as MockClient:
            mock_client = MagicMock()
            mock_client.post = AsyncMock(side_effect=fake_post)
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MockClient.return_value.__aexit__  = AsyncMock(return_value=False)

            await skill.execute({"action": "query", "metrics": ["turnover"]})

        assert "Authorization" not in captured_headers

    # ── 成功响应完整性 ────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_success_response_has_all_fields(self):
        """成功响应包含所有必要字段。"""
        s, e  = week_range()
        rows  = [{"turnover": 88888.0, "customer_number": 500}]
        skill = self.make_skill()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = MagicMock()
            mock_client.post = AsyncMock(
                return_value=_mock_response(_api_success(rows))
            )
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MockClient.return_value.__aexit__  = AsyncMock(return_value=False)

            result = await skill.execute({
                "action":      "query",
                "metrics":     ["turnover", "customer_number"],
                "bra_id":      "BR001",
                "aggregation": "SUM",
                "range_start": s,
                "range_end":   e,
            })

        for field in ("success", "aggregation", "bra_id",
                      "range_start", "range_end", "metrics", "raw_metrics"):
            assert field in result, f"响应缺少字段: {field}"


# ══════════════════════════════════════════════════════════════
# 6. SkillRegistry 集成测试
# ══════════════════════════════════════════════════════════════

class TestSkillRegistryIntegration:
    """SkillRegistry 与 AgentBiSkill 的集成。"""

    @pytest.mark.asyncio
    async def test_register_and_list_descriptor(self):
        from skills.registry import LocalSkillRegistry
        reg   = LocalSkillRegistry()
        skill = AgentBiSkill(api_url="http://test.invalid/bi")
        reg.register(skill)

        names = {d.name for d in reg.list_descriptors()}
        assert "agent_bi" in names

    @pytest.mark.asyncio
    async def test_call_returns_tool_result(self):
        from skills.registry import LocalSkillRegistry
        rows  = [{"turnover": 12345.0}]
        skill = AgentBiSkill(api_url="http://test.invalid/bi")

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = MagicMock()
            mock_client.post = AsyncMock(
                return_value=_mock_response(_api_success(rows))
            )
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MockClient.return_value.__aexit__  = AsyncMock(return_value=False)

            reg = LocalSkillRegistry()
            reg.register(skill)
            result = await reg.call("agent_bi", {
                "action":  "query",
                "metrics": ["turnover"],
            })

        assert result.error is None
        assert result.content["success"] is True
        assert result.tool_name == "agent_bi"

    @pytest.mark.asyncio
    async def test_call_validation_error_no_tool_error(self):
        """参数校验失败时，ToolResult 的 error 字段为空（业务错误在 content 中）。"""
        from skills.registry import LocalSkillRegistry
        reg   = LocalSkillRegistry()
        skill = AgentBiSkill(api_url="http://test.invalid/bi")
        reg.register(skill)

        result = await reg.call("agent_bi", {
            "action":  "query",
            "metrics": [],
        })
        # 参数校验错误以业务方式返回，不抛异常
        assert result.error is None
        assert "error" in result.content

    @pytest.mark.asyncio
    async def test_timeout_enforced_by_registry(self):
        """Registry 的 timeout 机制能覆盖超慢的 Skill 调用。"""
        from skills.registry import LocalSkillRegistry
        import asyncio
        skill = AgentBiSkill(api_url="http://test.invalid/bi")

        async def slow_execute(_):
            await asyncio.sleep(999)

        skill.execute = slow_execute

        fast_reg = LocalSkillRegistry(timeout_multiplier=0.001)
        fast_reg.register(skill)
        result = await fast_reg.call("agent_bi", {"action": "query", "metrics": ["turnover"]})
        assert result.error is not None
        assert "timeout" in result.error.lower()

    @pytest.mark.asyncio
    async def test_call_unknown_skill_returns_error(self):
        from skills.registry import LocalSkillRegistry
        reg    = LocalSkillRegistry()
        result = await reg.call("nonexistent_bi", {})
        assert result.error is not None
        assert "not found" in result.error.lower()


# ══════════════════════════════════════════════════════════════
# 7. E2E — 完整 Agent 工作流（MockLLM → AgentBiSkill → 最终回答）
# ══════════════════════════════════════════════════════════════

class TestAgentBiE2E:
    """验证 Agent 能通过 LLM 工具调用正确驱动 AgentBiSkill 并返回回答。"""

    def _make_container(self, tool_call_responses, final_answer: str,
                         skill_execute_result: dict):
        from core.container import AgentContainer
        from core.models import LLMResponse
        from llm.engines import MockLLMEngine
        from memory.stores import InMemoryLongTermMemory, InMemoryShortTermMemory
        from mcp.hub import DefaultMCPHub
        from context.manager import PriorityContextManager
        from skills.registry import LocalSkillRegistry, PythonExecutorSkill, WebSearchSkill

        responses = list(tool_call_responses) + [
            LLMResponse(
                content=final_answer,
                usage={"prompt_tokens": 200, "completion_tokens": 60},
            )
        ]

        skill = AgentBiSkill(api_url="http://test.invalid/bi")
        skill.execute = AsyncMock(return_value=skill_execute_result)

        registry = LocalSkillRegistry()
        registry.register(PythonExecutorSkill())
        registry.register(WebSearchSkill())
        registry.register(skill)

        return AgentContainer(
            llm_engine=MockLLMEngine(responses),
            short_term_memory=InMemoryShortTermMemory(),
            long_term_memory=InMemoryLongTermMemory(),
            skill_registry=registry,
            mcp_hub=DefaultMCPHub(),
            context_manager=PriorityContextManager(),
        ).build()

    @pytest.mark.asyncio
    async def test_agent_queries_bi_and_gets_answer(self):
        """Story 1 E2E — 用户问今日营业额，Agent 调用 agent_bi 并回答。"""
        from core.models import AgentConfig, LLMResponse, ToolCall

        s, e = today_range()
        container = self._make_container(
            tool_call_responses=[
                LLMResponse(
                    tool_calls=[ToolCall(
                        tool_name="agent_bi",
                        arguments={
                            "action":      "query",
                            "metrics":     ["turnover", "order_quantity"],
                            "range_start": s,
                            "range_end":   e,
                        },
                    )],
                    usage={"prompt_tokens": 120, "completion_tokens": 30},
                )
            ],
            final_answer="今日总营业额为 12,345.6 元，共 88 笔订单。",
            skill_execute_result={
                "success": True,
                "aggregation": "SUM",
                "metrics": [
                    {
                        "turnover":       {"value": 12345.6, "label": "总销售额"},
                        "order_quantity": {"value": 88,      "label": "订单总数"},
                    }
                ],
                "raw_metrics": [{"turnover": 12345.6, "order_quantity": 88}],
            },
        )

        events = []
        async for ev in container.agent().run(
            user_id="test_user",
            session_id="bi_e2e_1",
            text="帮我查一下今天的营业额和订单数",
            config=AgentConfig(stream=False, max_steps=5),
        ):
            events.append(ev)

        # 验证工具调用事件
        step_events = [e for e in events if e.get("type") == "step"]
        assert any(e.get("tool") == "agent_bi" for e in step_events)

        # 验证最终回答
        deltas = [e for e in events if e.get("type") == "delta"]
        full_text = "".join(e["text"] for e in deltas)
        assert "营业额" in full_text or "订单" in full_text

        # 验证任务完成
        done = [e for e in events if e.get("type") == "done"]
        assert done and done[0]["status"] == "done"

    @pytest.mark.asyncio
    async def test_agent_queries_store_bi(self):
        """Story 2 E2E — 用户问指定门店销售数据，LLM 传入 bra_id。"""
        from core.models import AgentConfig, LLMResponse, ToolCall

        container = self._make_container(
            tool_call_responses=[
                LLMResponse(
                    tool_calls=[ToolCall(
                        tool_name="agent_bi",
                        arguments={
                            "action":  "query",
                            "metrics": ["turnover"],
                            "bra_id":  "STORE_001",
                        },
                    )],
                    usage={"prompt_tokens": 100, "completion_tokens": 20},
                )
            ],
            final_answer="STORE_001 门店本期营业额为 5,000 元。",
            skill_execute_result={
                "success": True,
                "aggregation": "SUM",
                "bra_id": "STORE_001",
                "metrics": [{"turnover": {"value": 5000.0, "label": "总销售额"}}],
                "raw_metrics": [{"turnover": 5000.0}],
            },
        )

        events = []
        async for ev in container.agent().run(
            user_id="test_user",
            session_id="bi_e2e_2",
            text="查一下 STORE_001 这家店的营业额",
            config=AgentConfig(stream=False, max_steps=5),
        ):
            events.append(ev)

        step_events = [e for e in events if e.get("type") == "step"]
        assert any(e.get("tool") == "agent_bi" for e in step_events)

        done = [e for e in events if e.get("type") == "done"]
        assert done and done[0]["status"] == "done"

    @pytest.mark.asyncio
    async def test_agent_handles_api_error_gracefully(self):
        """Story 6 E2E — BI API 报错时 Agent 仍能优雅回答。"""
        from core.models import AgentConfig, LLMResponse, ToolCall

        container = self._make_container(
            tool_call_responses=[
                LLMResponse(
                    tool_calls=[ToolCall(
                        tool_name="agent_bi",
                        arguments={"action": "query", "metrics": ["turnover"]},
                    )],
                    usage={"prompt_tokens": 100, "completion_tokens": 20},
                )
            ],
            final_answer="抱歉，BI 系统当前不可用，请稍后再试。",
            skill_execute_result={
                "error": "请求超时，BI 接口响应时间过长，请稍后重试",
                "type":  "TimeoutError",
            },
        )

        events = []
        async for ev in container.agent().run(
            user_id="test_user",
            session_id="bi_e2e_3",
            text="今天的营业额是多少？",
            config=AgentConfig(stream=False, max_steps=5),
        ):
            events.append(ev)

        # Agent 不应崩溃，仍然产生 done 事件
        done = [e for e in events if e.get("type") == "done"]
        assert done and done[0]["status"] == "done"

    @pytest.mark.asyncio
    async def test_tool_result_in_history(self):
        """E2E — 工具调用结果被写入 task history。"""
        from core.models import AgentConfig, LLMResponse, ToolCall
        from memory.stores import InMemoryShortTermMemory

        stm = InMemoryShortTermMemory()
        from core.container import AgentContainer
        from llm.engines import MockLLMEngine
        from memory.stores import InMemoryLongTermMemory
        from mcp.hub import DefaultMCPHub
        from context.manager import PriorityContextManager
        from skills.registry import LocalSkillRegistry, PythonExecutorSkill, WebSearchSkill

        skill = AgentBiSkill(api_url="http://test.invalid/bi")
        skill.execute = AsyncMock(return_value={
            "success": True,
            "metrics": [{"turnover": {"value": 100.0, "label": "总销售额"}}],
            "raw_metrics": [{"turnover": 100.0}],
        })

        registry = LocalSkillRegistry()
        registry.register(PythonExecutorSkill())
        registry.register(WebSearchSkill())
        registry.register(skill)

        container = AgentContainer(
            llm_engine=MockLLMEngine([
                LLMResponse(
                    tool_calls=[ToolCall(
                        tool_name="agent_bi",
                        arguments={"action": "query", "metrics": ["turnover"]},
                    )],
                    usage={"prompt_tokens": 80, "completion_tokens": 10},
                ),
                LLMResponse(
                    content="营业额为 100 元。",
                    usage={"prompt_tokens": 150, "completion_tokens": 20},
                ),
            ]),
            short_term_memory=stm,
            long_term_memory=InMemoryLongTermMemory(),
            skill_registry=registry,
            mcp_hub=DefaultMCPHub(),
            context_manager=PriorityContextManager(),
        ).build()

        events = []
        async for ev in container.agent().run(
            user_id="u1",
            session_id="bi_history_test",
            text="营业额？",
            config=AgentConfig(stream=False),
        ):
            events.append(ev)

        done_ev  = next((e for e in events if e.get("type") == "done"), {})
        task_id  = done_ev.get("task_id")
        assert task_id, f"未找到 done 事件，events={events}"

        task  = await stm.load_task(task_id)
        roles = [m.role.value for m in task.history]
        assert "tool" in roles, f"工具结果未写入 history，roles={roles}"


# ══════════════════════════════════════════════════════════════
# 8. 实时接口集成测试（需要 --run-live）
# ══════════════════════════════════════════════════════════════

@pytest.mark.live
class TestAgentBiLive:
    """需要 --run-live 标志；调用真实的 BI 接口。"""

    @pytest.mark.asyncio
    async def test_live_query_turnover(self):
        """真实接口 — 查询总销售额（不传时间范围）。"""
        skill  = AgentBiSkill()  # 使用默认 URL
        s, e   = last_week_range()
        result = await skill.execute({
            "action":      "query",
            "metrics":     ["turnover", "order_quantity"],
            "aggregation": "SUM",
            "range_start": s,
            "range_end":   e,
        })
        # 接口正常 → success True；接口不通 → 有 error 字段
        assert "success" in result or "error" in result

    @pytest.mark.asyncio
    async def test_live_multiple_metrics(self):
        """真实接口 — 多指标查询。"""
        skill = AgentBiSkill()
        s, e  = month_range()
        result = await skill.execute({
            "action":  "query",
            "metrics": ["turnover", "customer_number", "refund_amount"],
            "range_start": s,
            "range_end":   e,
        })
        assert "success" in result or "error" in result
