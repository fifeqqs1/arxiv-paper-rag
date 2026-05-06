import re


ARXIV_DIRECT_QUERY_HINTS = (
    "论文",
    "paper",
    "papers",
    "arxiv",
    "文献",
)

MIXED_RESEARCH_PROJECT_HINTS = (
    "repo",
    "repository",
    "仓库",
    "项目",
    "代码",
    "实现",
    "clone",
    "git",
    "diff",
    "本地",
    "我的代码",
    "这个repo",
    "这个项目",
)

MIXED_RESEARCH_ANALYSIS_HINTS = (
    "区别",
    "差异",
    "对比",
    "比较",
    "看一下",
    "看看",
    "分析",
    "检查",
    "review",
    "有没有类似实现",
    "类似实现",
    "对应实现",
    "代码里",
    "仓库里",
    "项目里",
    "repo里",
)

GENERAL_TAVILY_QUERY_HINTS = (
    "天气",
    "气温",
    "降雨",
    "下雨",
    "赛程",
    "比赛",
    "对阵",
    "积分榜",
    "排名",
    "价格",
    "汇率",
    "股价",
    "市值",
    "官网",
    "地址",
    "电话",
    "路线",
    "怎么去",
    "门票",
    "schedule",
    "schedules",
    "match",
    "matches",
    "fixture",
    "fixtures",
    "weather",
    "temperature",
    "price",
    "prices",
    "stock price",
    "exchange rate",
)

NEWS_TAVILY_QUERY_HINTS = (
    "新闻",
    "热点",
    "头条",
    "快讯",
    "动态",
    "发布",
    "宣布",
    "通报",
    "回应",
    "财报",
    "并购",
    "融资",
    "政策",
    "新规",
    "选举",
    "战争",
    "地震",
    "台风",
    "news",
    "headline",
    "headlines",
    "breaking",
    "released",
    "announced",
)

TODAY_TAVILY_QUERY_HINTS = (
    "今天",
    "今日",
    "今晚",
    "今夜",
)

TOMORROW_TAVILY_QUERY_HINTS = (
    "明天",
    "明日",
    "明晚",
    "明夜",
)

YESTERDAY_TAVILY_QUERY_HINTS = (
    "昨天",
    "昨日",
    "昨晚",
    "昨夜",
)

THIS_WEEK_TAVILY_QUERY_HINTS = (
    "本周",
    "这周",
    "本星期",
    "这星期",
)

FAST_PATH_SIMPLE_HINTS = (
    "几点",
    "现在几点",
    "时间",
    "天气",
    "气温",
    "汇率",
    "价格",
    "股价",
    "是什么",
    "什么意思",
    "介绍一下",
    "翻译",
    "官网",
    "地址",
    "电话",
    "怎么去",
    "who is",
    "what is",
    "where is",
    "weather",
    "price",
    "translate",
)

READ_ONLY_ANALYSIS_EXPLANATION_HINTS = (
    "解释",
    "介绍",
    "看看",
    "了解",
    "概览",
    "总览",
    "是干什么的",
    "做什么",
    "用途",
    "作用",
    "how it works",
    "overview",
    "explain",
)

ANALYSIS_ACTION_HINTS = (
    "详细分析",
    "深入分析",
    "分析",
    "详细解读",
    "深入解读",
    "解读",
    "review",
    "code review",
    "审查",
    "对比",
    "比较",
    "梳理",
    "排查",
    "定位",
    "总结",
    "建议",
    "优化",
    "重构",
)

ANALYSIS_SCOPE_HINTS = (
    "项目",
    "代码",
    "仓库",
    "repo",
    "repository",
    "架构",
    "模块",
    "实现",
    "目录",
    "流程",
    "设计",
    "差异",
    "文件",
)

READ_ONLY_ANALYSIS_DISALLOWED_HINTS = (
    "测试",
    "验证",
    "pytest",
    "unittest",
    "ruff",
    "mypy",
    "test",
    "tests",
    "verify",
    "check",
)

SLOW_PATH_ANALYSIS_INTENT_HINTS = (
    "详细分析项目",
    "深入分析项目",
    "详细解读项目",
    "解读这个项目",
    "分析这个项目",
    "项目分析",
    "架构分析",
    "架构对比",
    "设计差异",
    "代码review",
    "review代码",
    "review 代码",
    "代码审查",
    "给出修改建议",
    "修改建议",
    "优化建议",
    "重构建议",
    "改进建议",
    "排查原因",
    "定位问题",
    "梳理流程",
    "分析实现",
    "对比差异",
)

SLOW_PATH_MULTI_STEP_HINTS = (
    "先",
    "然后",
    "接着",
    "再",
    "最后",
    "一步步",
    "逐步",
    "依次",
    "分步骤",
    "step by step",
    "first",
    "then",
    "after that",
    "finally",
)

SLOW_PATH_HIGH_RISK_HINTS = (
    "写入",
    "写文件",
    "创建文件",
    "新建文件",
    "修改文件",
    "编辑文件",
    "修改代码",
    "改代码",
    "修复",
    "修bug",
    "覆盖",
    "删除文件",
    "删除任务",
    "取消任务",
    "修改任务",
    "运行",
    "执行",
    "shell",
    "命令",
    "command",
    "patch",
    "edit",
    "fix",
    "python ",
    "bash ",
    "powershell",
    "skill",
    "技能",
    "rm ",
    "mv ",
)


def contains_query_hint(query: str, lowered: str, hints: tuple[str, ...]) -> bool:
    for hint in hints:
        if hint.isascii():
            if hint in lowered:
                return True
        elif hint in query:
            return True
    return False


def _has_explicit_paper_intent(query: str) -> bool:
    lowered = query.lower()
    return contains_query_hint(query, lowered, ARXIV_DIRECT_QUERY_HINTS)


def _has_project_code_scope(query: str) -> bool:
    lowered = query.lower()
    return contains_query_hint(query, lowered, MIXED_RESEARCH_PROJECT_HINTS)


def _has_project_code_analysis_intent(query: str) -> bool:
    lowered = query.lower()
    return contains_query_hint(query, lowered, MIXED_RESEARCH_ANALYSIS_HINTS)


def _looks_like_mixed_research_task(query: str) -> bool:
    return (
        _has_explicit_paper_intent(query)
        and _has_project_code_scope(query)
        and _has_project_code_analysis_intent(query)
    )


def should_direct_route_to_arxiv_rag(query: str) -> bool:
    return _has_explicit_paper_intent(query) and not _looks_like_mixed_research_task(query)


def infer_tavily_topic(query: str) -> str:
    normalized_query = (query or "").strip()
    if not normalized_query:
        return "general"

    lowered_query = normalized_query.lower()

    if contains_query_hint(normalized_query, lowered_query, GENERAL_TAVILY_QUERY_HINTS):
        return "general"

    if contains_query_hint(normalized_query, lowered_query, NEWS_TAVILY_QUERY_HINTS):
        return "news"

    return "general"


def _looks_like_simple_question(query: str, lowered: str) -> bool:
    simple_hints = FAST_PATH_SIMPLE_HINTS + GENERAL_TAVILY_QUERY_HINTS
    if contains_query_hint(query, lowered, simple_hints):
        return not (
            contains_query_hint(query, lowered, ANALYSIS_ACTION_HINTS)
            or contains_query_hint(query, lowered, ANALYSIS_SCOPE_HINTS)
            or contains_query_hint(query, lowered, SLOW_PATH_MULTI_STEP_HINTS)
            or contains_query_hint(query, lowered, SLOW_PATH_HIGH_RISK_HINTS)
        )

    if len(query) <= 18 and not (
        contains_query_hint(query, lowered, ANALYSIS_ACTION_HINTS)
        or contains_query_hint(query, lowered, ANALYSIS_SCOPE_HINTS)
        or contains_query_hint(query, lowered, SLOW_PATH_MULTI_STEP_HINTS)
        or contains_query_hint(query, lowered, SLOW_PATH_HIGH_RISK_HINTS)
    ):
        return True

    return False


def _looks_like_analysis_review_request(query: str, lowered: str) -> bool:
    if contains_query_hint(query, lowered, SLOW_PATH_ANALYSIS_INTENT_HINTS):
        return True

    has_action = contains_query_hint(query, lowered, ANALYSIS_ACTION_HINTS)
    has_scope = contains_query_hint(query, lowered, ANALYSIS_SCOPE_HINTS)
    return has_action and has_scope


def _looks_like_read_only_analysis_request(query: str, lowered: str) -> bool:
    has_scope = contains_query_hint(query, lowered, ANALYSIS_SCOPE_HINTS)
    if not has_scope:
        return False

    if (
        contains_query_hint(query, lowered, SLOW_PATH_MULTI_STEP_HINTS)
        or contains_query_hint(query, lowered, SLOW_PATH_HIGH_RISK_HINTS)
        or contains_query_hint(query, lowered, READ_ONLY_ANALYSIS_DISALLOWED_HINTS)
    ):
        return False

    return (
        contains_query_hint(query, lowered, ANALYSIS_ACTION_HINTS)
        or contains_query_hint(query, lowered, READ_ONLY_ANALYSIS_EXPLANATION_HINTS)
    )


def _looks_like_clearly_complex_request(query: str, lowered: str) -> bool:
    return contains_query_hint(
        query,
        lowered,
        SLOW_PATH_MULTI_STEP_HINTS,
    )


def build_route_decision(query: str | None, llm_classifier_fn=None) -> dict:
    normalized_query = (query or "").strip()
    if not normalized_query:
        return {
            "route": "fast",
            "goal": "",
            "complexity": "simple",
            "risk_level": "low",
            "planner_required": False,
            "route_locked": False,
            "route_source": "empty_query",
            "route_reason": "empty query",
            "route_confidence": 1.0,
        }

    lowered_query = normalized_query.lower()
    is_multi_step = contains_query_hint(normalized_query, lowered_query, SLOW_PATH_MULTI_STEP_HINTS)
    is_high_risk = contains_query_hint(normalized_query, lowered_query, SLOW_PATH_HIGH_RISK_HINTS)
    is_simple = _looks_like_simple_question(normalized_query, lowered_query)
    is_read_only_analysis = _looks_like_read_only_analysis_request(normalized_query, lowered_query)
    is_mixed_research = _looks_like_mixed_research_task(normalized_query)

    if is_mixed_research:
        return {
            "route": "slow",
            "goal": normalized_query,
            "complexity": "mixed_research",
            "risk_level": "medium",
            "planner_required": True,
            "route_locked": False,
            "route_source": "mixed_research_task",
            "route_reason": "mixed paper plus repo/code comparison request requires structured planning",
            "route_confidence": 1.0,
        }

    if should_direct_route_to_arxiv_rag(normalized_query):
        return {
            "route": "fast",
            "goal": normalized_query,
            "complexity": "simple",
            "risk_level": "low",
            "planner_required": False,
            "route_locked": False,
            "route_source": "arxiv_direct",
            "route_reason": "pure paper query keeps fastest direct route",
            "route_confidence": 1.0,
        }

    if is_high_risk:
        return {
            "route": "slow",
            "goal": normalized_query,
            "complexity": "high_risk" if not is_multi_step else "multi_step_high_risk",
            "risk_level": "high",
            "planner_required": True,
            "route_locked": True,
            "route_source": "rule_high_risk",
            "route_reason": "high-risk task requires slow path and approval guardrails",
            "route_confidence": 1.0,
        }

    if is_multi_step:
        return {
            "route": "slow",
            "goal": normalized_query,
            "complexity": "multi_step",
            "risk_level": "medium",
            "planner_required": True,
            "route_locked": False,
            "route_source": "rule_multi_step",
            "route_reason": "multi-step task requires explicit planning",
            "route_confidence": 1.0,
        }

    if is_simple:
        return {
            "route": "fast",
            "goal": normalized_query,
            "complexity": "simple",
            "risk_level": "low",
            "planner_required": False,
            "route_locked": False,
            "route_source": "rule_simple",
            "route_reason": "simple request can be answered directly",
            "route_confidence": 1.0,
        }

    if is_read_only_analysis:
        return {
            "route": "fast",
            "goal": normalized_query,
            "complexity": "read_only_analysis",
            "risk_level": "low",
            "planner_required": False,
            "route_locked": False,
            "route_source": "rule_read_only_analysis",
            "route_reason": "read-only repository analysis can be answered without slow-path planning",
            "route_confidence": 1.0,
        }

    if _looks_like_clearly_complex_request(normalized_query, lowered_query):
        return {
            "route": "slow",
            "goal": normalized_query,
            "complexity": "complex",
            "risk_level": "medium",
            "planner_required": True,
            "route_locked": False,
            "route_source": "rule_complex",
            "route_reason": "complex request should be planned before execution",
            "route_confidence": 0.8,
        }

    return {
        "route": "slow",
        "goal": normalized_query,
        "complexity": "uncertain",
        "risk_level": "medium",
        "planner_required": True,
        "route_locked": False,
        "route_source": "planner_first_uncertain",
        "route_reason": "non-simple and non-high-risk request goes directly to planner",
        "route_confidence": 0.9,
    }


def route_after_router(state) -> str:
    route = state.get("route", "fast")
    if route == "slow" and not state.get("planner_required", False):
        return "slow"
    return "planner" if state.get("planner_required", False) else "fast"
