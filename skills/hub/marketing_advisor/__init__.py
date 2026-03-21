"""
skills/hub/marketing_advisor/__init__.py — 营销顾问 Skill

内置营销策略知识库，通过三阶段引导商家完成需求澄清并生成方案：
  1. analyze_needs     — 缺口分析 + 下一个关键问题
  2. suggest_activities — 从知识库匹配适合的营销活动
  3. create_plan       — 生成带执行步骤和时间轴的完整方案
"""
from __future__ import annotations

from core.models import PermissionLevel, ToolDescriptor

# ─────────────────────────────────────────────────────────────────────────────
# 营销策略知识库
# key: (business_type_keyword, goal_keyword) → list of activity templates
# ─────────────────────────────────────────────────────────────────────────────

_ACTIVITY_PLAYBOOK: list[dict] = [
    # ── 餐饮 ──────────────────────────────────────────────────────────────
    {
        "id": "f_lunch_deal",
        "name": "午市限时套餐",
        "fit": {"business": ["餐饮", "餐厅", "外卖", "快餐", "cafe", "咖啡"], "goal": ["引流", "客流", "午餐"]},
        "description": "推出11:00-14:00专属套餐，价格比单点优惠15-20%，搭配美团/大众点评曝光",
        "channels": ["美团", "大众点评", "店内海报", "微信群"],
        "budget_range": "低（主要为折让成本）",
        "duration": "长期可持续",
        "steps": ["设计2-3个套餐组合", "在外卖平台设置限时优惠", "制作A4海报贴门口"],
        "kpi": "午市客流提升20-40%",
        "startup_days": 1,
        "tags": ["餐饮", "引流", "低预算"],
    },
    {
        "id": "f_checkin",
        "name": "打卡送饮品/小食",
        "fit": {"business": ["餐饮", "餐厅", "咖啡", "甜品", "奶茶"], "goal": ["曝光", "引流", "新客", "品牌"]},
        "description": "顾客在小红书/抖音发打卡帖，凭截图免费赠小食或饮品，吸引年轻客群",
        "channels": ["小红书", "抖音", "微信朋友圈"],
        "budget_range": "低-中（菜品成本）",
        "duration": "1-3个月",
        "steps": ["设计适合出片的出餐场景/摆盘", "制作打卡规则提示牌", "建立审核兑换流程"],
        "kpi": "小红书/抖音提及量月增50条以上",
        "startup_days": 3,
        "tags": ["餐饮", "品牌", "年轻客群"],
    },
    {
        "id": "f_member_card",
        "name": "储值会员卡",
        "fit": {"business": ["餐饮", "餐厅", "咖啡", "甜品"], "goal": ["复购", "留存", "会员"]},
        "description": "充100送20，充200送50，锁定复购。配合会员专属折扣和生日礼品",
        "channels": ["收银台话术", "微信公众号", "店内海报"],
        "budget_range": "低（折让成本）",
        "duration": "长期",
        "steps": ["选择会员系统（美团/微盟/自建小程序）", "设置充值档位", "培训店员话术"],
        "kpi": "会员留存率提升30%，月均消费频次×1.5",
        "startup_days": 7,
        "tags": ["餐饮", "复购", "会员"],
    },

    # ── 零售/实体店 ────────────────────────────────────────────────────────
    {
        "id": "r_flash_sale",
        "name": "限时特卖/清仓活动",
        "fit": {"business": ["零售", "服装", "百货", "超市", "便利店"], "goal": ["促销", "库存", "清仓", "引流"]},
        "description": "每周固定1-2天设置特卖时段，搭配倒计时氛围营造紧迫感",
        "channels": ["店内广播", "微信群", "朋友圈", "门口堆头"],
        "budget_range": "低（折让成本）",
        "duration": "持续进行",
        "steps": ["选出滞销或利润空间大的SKU", "设计特卖价格", "提前1天发微信群预告"],
        "kpi": "特卖日销售额较平日提升50%+",
        "startup_days": 2,
        "tags": ["零售", "促销", "低预算"],
    },
    {
        "id": "r_referral",
        "name": "老带新裂变",
        "fit": {"business": ["零售", "服装", "美容", "服务", "教育", "健身"], "goal": ["新客", "获客", "裂变", "引流"]},
        "description": "老客带新朋友进店，双方各获优惠券/积分/返现。口碑传播成本最低",
        "channels": ["微信好友", "社区群", "线下口耳相传"],
        "budget_range": "低-中（奖励成本）",
        "duration": "1-3个月",
        "steps": ["设计双向奖励方案", "制作推荐码/二维码", "建立到店核销流程"],
        "kpi": "每月新客中转介绍占比达30%+",
        "startup_days": 5,
        "tags": ["新客", "裂变", "低-中预算"],
    },

    # ── 电商/直播 ─────────────────────────────────────────────────────────
    {
        "id": "e_live_sale",
        "name": "直播带货首播活动",
        "fit": {"business": ["电商", "直播", "网店", "淘宝", "拼多多", "抖店"], "goal": ["销售", "转化", "引流", "新品"]},
        "description": "策划首播或大场直播，设置专属直播间价/买赠/秒杀，提前3-5天预热",
        "channels": ["抖音直播", "淘宝直播", "微信视频号"],
        "budget_range": "中（主播费/商品成本）",
        "duration": "单次活动",
        "steps": ["确定直播时间和主播", "设计专属福利品和价格", "预热视频发布", "准备互动话术"],
        "kpi": "首播GMV目标、观看人数、转化率",
        "startup_days": 7,
        "tags": ["电商", "直播", "销售转化"],
    },
    {
        "id": "e_bundle",
        "name": "满减/满赠/组合套餐",
        "fit": {"business": ["电商", "网店", "零售"], "goal": ["客单价", "转化", "促销"]},
        "description": "满100减10/满200减25或买A送B，提升客单价，降低决策门槛",
        "channels": ["店铺首页", "商品详情页", "推送通知"],
        "budget_range": "低（折让成本）",
        "duration": "1-4周",
        "steps": ["计算最优折扣门槛", "配置活动规则", "更新详情页Banner"],
        "kpi": "客单价提升20%，转化率提升15%",
        "startup_days": 1,
        "tags": ["电商", "客单价", "低预算"],
    },

    # ── 节假日通用 ─────────────────────────────────────────────────────────
    {
        "id": "h_holiday",
        "name": "节日主题促销",
        "fit": {"business": ["*"], "goal": ["节日", "活动", "周年", "促销", "礼品"]},
        "description": "结合节日（春节/情人节/618/双11/中秋）策划主题活动，限时折扣+礼品包装",
        "channels": ["全渠道"],
        "budget_range": "中",
        "duration": "节前7-14天启动",
        "steps": ["确定节日主题和视觉", "设计活动规则", "全渠道同步上线", "节后数据复盘"],
        "kpi": "活动期间销售额同比增长30%+",
        "startup_days": 14,
        "tags": ["节假日", "全行业", "中预算"],
    },

    # ── 私域/社群 ─────────────────────────────────────────────────────────
    {
        "id": "p_wechat_group",
        "name": "微信社群运营",
        "fit": {"business": ["*"], "goal": ["复购", "私域", "社群", "粉丝", "留存"]},
        "description": "建立微信客户群，每日/每周发专属福利、新品预告、优惠券，沉淀私域流量",
        "channels": ["微信群", "公众号"],
        "budget_range": "低（人力成本）",
        "duration": "持续",
        "steps": ["建群并设群规", "设计入群欢迎语和福利", "制定每周内容日历", "设置专属群价"],
        "kpi": "群活跃率>30%，月复购贡献≥20%",
        "startup_days": 3,
        "tags": ["私域", "复购", "低预算"],
    },
    {
        "id": "p_xiaohongshu",
        "name": "小红书素人种草",
        "fit": {"business": ["美容", "服装", "餐饮", "零售", "教育", "健身"], "goal": ["品牌", "曝光", "种草", "年轻"]},
        "description": "邀请5-20位素人/达人发布真实体验笔记，低成本提升品牌搜索量",
        "channels": ["小红书"],
        "budget_range": "低-中（达人费用）",
        "duration": "1-2个月",
        "steps": ["筛选匹配品牌调性的素人", "提供产品/服务体验", "引导自然评测发布", "监控数据"],
        "kpi": "品牌关键词搜索量月增100%",
        "startup_days": 14,
        "tags": ["品牌", "种草", "低-中预算"],
    },
]

# 需求字段定义
_REQUIRED_FIELDS = [
    {
        "field": "business_type",
        "label": "经营类型",
        "importance": "critical",
        "question": "您好！请问您是做什么生意的？（比如餐厅、服装店、电商、美容院等）",
    },
    {
        "field": "goal",
        "label": "营销目标",
        "importance": "critical",
        "question": "这次做营销活动，您最想达到什么目标？（比如吸引新顾客、提升复购、冲节日销售额等）",
    },
    {
        "field": "target_customers",
        "label": "目标客群",
        "importance": "high",
        "question": "您的主要顾客是哪类人群？（比如周边上班族、年轻女性、宝妈、学生等）",
    },
    {
        "field": "budget",
        "label": "预算范围",
        "importance": "high",
        "question": "这次活动您预计投入多少预算？（可以给个大概范围，比如5000元以内、1-3万、不限等）",
    },
    {
        "field": "duration",
        "label": "活动周期",
        "importance": "medium",
        "question": "您希望这次活动持续多久？（比如1周、整个月、节假日期间、长期进行等）",
    },
    {
        "field": "current_channels",
        "label": "现有渠道",
        "importance": "medium",
        "question": "您目前有哪些可用的营销渠道？（比如微信群、公众号、抖音账号、美团/大众点评等）",
    },
    {
        "field": "pain_points",
        "label": "当前痛点",
        "importance": "low",
        "question": "您目前营销上最大的困扰是什么？（比如客流少、老客不回头、推广效果差等）",
    },
]


def _score_completeness(profile: dict) -> tuple[float, list[dict]]:
    """计算画像完整度，返回 (分数0-1, 缺失字段列表)"""
    weights = {"critical": 0.3, "high": 0.2, "medium": 0.1, "low": 0.05}
    total_weight = sum(weights[f["importance"]] for f in _REQUIRED_FIELDS)
    missing = []
    earned = 0.0

    for field_def in _REQUIRED_FIELDS:
        val = profile.get(field_def["field"])
        if val and str(val).strip():
            earned += weights[field_def["importance"]]
        else:
            missing.append(field_def)

    return round(earned / total_weight, 2), missing


def _match_activities(profile: dict) -> list[dict]:
    """从策略库中匹配最适合商家的活动"""
    biz   = (profile.get("business_type") or "").lower()
    goal  = (profile.get("goal") or "").lower()
    matches = []

    for act in _ACTIVITY_PLAYBOOK:
        fit = act["fit"]
        biz_match  = any(k in biz for k in fit["business"]) or "*" in fit["business"]
        goal_match = any(k in goal for k in fit["goal"])
        if biz_match and goal_match:
            matches.append(act)
        elif biz_match and not matches:   # 至少保留同行业活动
            matches.append(act)

    # 最多返回 5 个
    return matches[:5]


def _budget_tier(budget_str: str) -> str:
    if not budget_str:
        return "未知"
    s = budget_str.replace(",", "").replace("，", "")
    for word in ["低", "少", "不多", "紧张"]:
        if word in s:
            return "低"
    for word in ["中", "适中", "一般"]:
        if word in s:
            return "中"
    for word in ["高", "多", "充足", "不限"]:
        if word in s:
            return "高"
    # 尝试数字判断
    import re
    nums = re.findall(r"\d+", s)
    if nums:
        mx = max(int(n) for n in nums)
        if mx <= 5000:
            return "低"
        if mx <= 30000:
            return "中"
        return "高"
    return "中"


class MarketingAdvisorSkill:
    descriptor = ToolDescriptor(
        name="marketing_advisor",
        description=(
            "营销顾问工具，帮助商家制定营销活动方案。"
            "先调用 analyze_needs 分析信息完整度，收集到足够信息（completeness>=0.7）后，"
            "调用 suggest_activities 展示具体活动选项，"
            "商家确认偏好后调用 create_plan 生成完整可执行方案。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["analyze_needs", "suggest_activities", "create_plan"],
                    "description": (
                        "analyze_needs=分析信息缺口，"
                        "suggest_activities=推荐营销活动，"
                        "create_plan=生成完整方案"
                    ),
                },
                "profile": {
                    "type": "object",
                    "description": "商家画像字典，已收集的信息",
                },
            },
            "required": ["action", "profile"],
        },
        source="skill",
        permission=PermissionLevel.READ,
        timeout_ms=5_000,
        tags=["marketing", "business", "campaign", "advisor"],
    )

    async def execute(self, arguments: dict) -> dict:
        action  = arguments["action"]
        profile = arguments.get("profile") or {}

        if action == "analyze_needs":
            return self._analyze_needs(profile)
        if action == "suggest_activities":
            return self._suggest_activities(profile)
        if action == "create_plan":
            return self._create_plan(profile)
        return {"error": f"未知 action: {action}"}

    # ── analyze_needs ─────────────────────────────────────────────

    def _analyze_needs(self, profile: dict) -> dict:
        completeness, missing = _score_completeness(profile)
        collected = {
            f["field"]: profile[f["field"]]
            for f in _REQUIRED_FIELDS
            if profile.get(f["field"])
        }

        # 找出最重要的下一个问题
        next_question = None
        for field_def in _REQUIRED_FIELDS:
            if field_def in missing:
                next_question = field_def["question"]
                break

        return {
            "completeness": completeness,
            "ready_to_plan": completeness >= 0.7,
            "collected": collected,
            "missing_fields": [
                {"field": f["field"], "label": f["label"],
                 "importance": f["importance"], "question": f["question"]}
                for f in missing
            ],
            "next_question": next_question,
            "guidance": (
                "信息已足够，可以调用 suggest_activities 给出营销建议。"
                if completeness >= 0.7 else
                f"还需收集 {len(missing)} 项信息（完整度 {int(completeness*100)}%），"
                "请继续向商家提问。"
            ),
        }

    # ── suggest_activities ────────────────────────────────────────

    def _suggest_activities(self, profile: dict) -> dict:
        activities = _match_activities(profile)
        tier       = _budget_tier(profile.get("budget", ""))

        # 根据预算过滤
        budget_filter = {
            "低": ["低（主要为折让成本）", "低（折让成本）", "低（人力成本）", "低-中（达人费用）", "低-中（奖励成本）", "低"],
            "中": ["低", "低-中（菜品成本）", "低-中", "中", "中（折让成本）", "中（主播费/商品成本）"],
            "高": None,  # 不过滤
        }
        allowed_budgets = budget_filter.get(tier)
        if allowed_budgets:
            filtered = [a for a in activities if a["budget_range"] in allowed_budgets]
            if not filtered:
                filtered = activities  # 全部保留，预算不够时给出提示
        else:
            filtered = activities

        return {
            "matched_activities": [
                {
                    "id":           a["id"],
                    "name":         a["name"],
                    "description":  a["description"],
                    "channels":     a["channels"],
                    "budget_range": a["budget_range"],
                    "duration":     a["duration"],
                    "kpi":          a["kpi"],
                    "startup_days": a["startup_days"],
                    "tags":         a["tags"],
                }
                for a in filtered
            ],
            "total": len(filtered),
            "guidance": (
                "以上是根据您的情况筛选出的营销活动方案。"
                "请告诉我您对哪些活动感兴趣，或者想了解某个活动的具体执行细节，"
                "我可以为您生成完整的落地方案。"
            ),
        }

    # ── create_plan ───────────────────────────────────────────────

    def _create_plan(self, profile: dict) -> dict:
        confirmed = profile.get("confirmed_activities") or []
        all_acts   = _match_activities(profile)

        # 选取确认的活动，或默认选前2个
        if confirmed:
            selected = [a for a in all_acts if a["id"] in confirmed or a["name"] in confirmed]
        else:
            selected = all_acts[:2]

        if not selected:
            selected = all_acts[:2]

        # 构建执行计划
        plan_items = []
        day_offset = 0
        for act in selected:
            plan_items.append({
                "activity":    act["name"],
                "description": act["description"],
                "channels":    act["channels"],
                "timeline":    f"第{day_offset + 1}天启动，持续{act['duration']}",
                "steps":       act["steps"],
                "kpi":         act["kpi"],
                "budget_range": act["budget_range"],
            })
            day_offset += act["startup_days"] + 3

        biz_name   = profile.get("business_name") or profile.get("business_type") or "您的商家"
        goal       = profile.get("goal", "提升业绩")
        budget     = profile.get("budget", "待确认")
        duration   = profile.get("duration", "待确认")

        return {
            "merchant":    biz_name,
            "goal":        goal,
            "budget":      budget,
            "duration":    duration,
            "plan_name":   f"「{biz_name}」营销活动方案",
            "activities":  plan_items,
            "total_activities": len(plan_items),
            "summary": (
                f"本方案针对「{biz_name}」制定了 {len(plan_items)} 项营销活动，"
                f"目标：{goal}。请根据实际情况调整执行节奏，"
                "如需修改任意活动的细节，请直接告诉我。"
            ),
            "next_steps": [
                "确认方案内容并安排执行负责人",
                "按时间轴逐步落地各项活动",
                "每周记录数据，对比 KPI 目标",
                "2周后回顾效果，根据数据调整策略",
            ],
        }
