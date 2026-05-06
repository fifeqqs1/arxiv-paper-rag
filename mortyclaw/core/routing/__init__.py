from .classifier import (
    ROUTE_CLASSIFIER_CONFIDENCE_THRESHOLD,
    ROUTE_CLASSIFIER_MODEL,
    classify_route_with_llm,
    get_route_classifier_model,
)
from .rules import (
    ANALYSIS_ACTION_HINTS,
    ANALYSIS_SCOPE_HINTS,
    ARXIV_DIRECT_QUERY_HINTS,
    FAST_PATH_SIMPLE_HINTS,
    GENERAL_TAVILY_QUERY_HINTS,
    NEWS_TAVILY_QUERY_HINTS,
    READ_ONLY_ANALYSIS_DISALLOWED_HINTS,
    READ_ONLY_ANALYSIS_EXPLANATION_HINTS,
    SLOW_PATH_ANALYSIS_INTENT_HINTS,
    SLOW_PATH_HIGH_RISK_HINTS,
    SLOW_PATH_MULTI_STEP_HINTS,
    THIS_WEEK_TAVILY_QUERY_HINTS,
    TODAY_TAVILY_QUERY_HINTS,
    TOMORROW_TAVILY_QUERY_HINTS,
    YESTERDAY_TAVILY_QUERY_HINTS,
    build_route_decision,
    contains_query_hint,
    infer_tavily_topic,
    route_after_router,
    should_direct_route_to_arxiv_rag,
)
from .web import (
    _get_local_now as _routing_web_local_now,
    extract_passthrough_payload,
    extract_passthrough_text,
    get_latest_user_query,
    normalize_tavily_tool_calls as _normalize_tavily_tool_calls_impl,
)


def _get_local_now():
    return _routing_web_local_now()


def normalize_tavily_tool_calls(response, latest_user_query, thread_id):
    return _normalize_tavily_tool_calls_impl(
        response,
        latest_user_query,
        thread_id,
        now=_get_local_now(),
    )
