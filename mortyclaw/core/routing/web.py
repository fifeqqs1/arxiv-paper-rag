import json
import re
from datetime import datetime, timedelta

from langchain_core.messages import AIMessage, HumanMessage

from ..logger import audit_logger
from .rules import (
    THIS_WEEK_TAVILY_QUERY_HINTS,
    TODAY_TAVILY_QUERY_HINTS,
    TOMORROW_TAVILY_QUERY_HINTS,
    YESTERDAY_TAVILY_QUERY_HINTS,
    infer_tavily_topic,
)
from ..tools.web import MORTYCLAW_PASSTHROUGH_FLAG


def extract_passthrough_payload(content: str) -> dict | None:
    if not isinstance(content, str) or not content.strip():
        return None

    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict) or payload.get(MORTYCLAW_PASSTHROUGH_FLAG) is not True:
        return None
    return payload


def extract_passthrough_text(tool_message) -> str | None:
    if getattr(tool_message, "type", "") != "tool":
        return None

    content = getattr(tool_message, "content", "")
    payload = extract_passthrough_payload(content)
    if payload is None:
        return None

    display_text = payload.get("display_text") or payload.get("answer")
    if not isinstance(display_text, str) or not display_text.strip():
        return None

    return display_text


def get_latest_user_query(raw_messages) -> str | None:
    if not raw_messages:
        return None

    last_message = raw_messages[-1]
    if not isinstance(last_message, HumanMessage):
        return None

    content = getattr(last_message, "content", "")
    if not isinstance(content, str) or not content.strip():
        return None
    return content


def _get_local_now() -> datetime:
    return datetime.now().astimezone()


def _contains_relative_time_hint(query: str) -> bool:
    return any(
        hint in query
        for hint in (
            *TODAY_TAVILY_QUERY_HINTS,
            *TOMORROW_TAVILY_QUERY_HINTS,
            *YESTERDAY_TAVILY_QUERY_HINTS,
            *THIS_WEEK_TAVILY_QUERY_HINTS,
        )
    )


def _build_relative_time_prefix(query: str, *, now: datetime | None = None) -> str:
    normalized_query = (query or "").strip()
    if not normalized_query:
        return ""

    now_dt = now or _get_local_now()
    today = now_dt.date()

    if any(hint in normalized_query for hint in THIS_WEEK_TAVILY_QUERY_HINTS):
        week_start = today - timedelta(days=today.weekday())
        week_end = week_start + timedelta(days=6)
        return f"{week_start.isoformat()} 至 {week_end.isoformat()}"

    if any(hint in normalized_query for hint in TOMORROW_TAVILY_QUERY_HINTS):
        return (today + timedelta(days=1)).isoformat()

    if any(hint in normalized_query for hint in YESTERDAY_TAVILY_QUERY_HINTS):
        return (today - timedelta(days=1)).isoformat()

    if any(hint in normalized_query for hint in TODAY_TAVILY_QUERY_HINTS):
        return today.isoformat()

    return ""


def _strip_absolute_date_markers(query: str) -> str:
    cleaned = query or ""
    patterns = (
        r"\b\d{4}-\d{1,2}-\d{1,2}\b",
        r"\b\d{4}/\d{1,2}/\d{1,2}\b",
        r"\d{4}年\d{1,2}月\d{1,2}日",
        r"\d{1,2}月\d{1,2}日",
    )
    for pattern in patterns:
        cleaned = re.sub(pattern, " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _strip_relative_time_markers(query: str) -> str:
    cleaned = query or ""
    for hint in (
        *TODAY_TAVILY_QUERY_HINTS,
        *TOMORROW_TAVILY_QUERY_HINTS,
        *YESTERDAY_TAVILY_QUERY_HINTS,
        *THIS_WEEK_TAVILY_QUERY_HINTS,
    ):
        cleaned = cleaned.replace(hint, " ")
    return re.sub(r"\s+", " ", cleaned).strip(" ，。,:;")


def _expand_tavily_query_with_absolute_date(
    tool_query: str | None,
    latest_user_query: str | None,
    *,
    now: datetime | None = None,
) -> str:
    normalized_tool_query = (tool_query or "").strip()
    normalized_latest_query = (latest_user_query or "").strip()
    base_query = normalized_tool_query or normalized_latest_query
    if not base_query:
        return ""

    relative_source = ""
    for candidate in (normalized_latest_query, normalized_tool_query):
        if candidate and _contains_relative_time_hint(candidate):
            relative_source = candidate
            break

    if not relative_source:
        return base_query

    prefix = _build_relative_time_prefix(relative_source, now=now)
    if not prefix:
        return base_query

    cleaned_base_query = _strip_relative_time_markers(_strip_absolute_date_markers(base_query))
    if not cleaned_base_query:
        cleaned_base_query = _strip_relative_time_markers(_strip_absolute_date_markers(normalized_latest_query)) or base_query

    return re.sub(r"\s+", " ", f"{prefix} {cleaned_base_query}").strip()


def normalize_tavily_tool_calls(
    response: AIMessage,
    latest_user_query: str | None,
    thread_id: str,
    *,
    now: datetime | None = None,
) -> AIMessage:
    tool_calls = getattr(response, "tool_calls", None) or []
    if not tool_calls:
        return response

    adjusted_tool_calls = []
    changed = False

    for tool_call in tool_calls:
        if tool_call.get("name") != "tavily_web_search":
            adjusted_tool_calls.append(tool_call)
            continue

        args = dict(tool_call.get("args") or {})
        query_parts = []

        tool_query = args.get("query")
        if isinstance(tool_query, str) and tool_query.strip():
            query_parts.append(tool_query.strip())
        if isinstance(latest_user_query, str) and latest_user_query.strip():
            query_parts.append(latest_user_query.strip())

        inferred_topic = infer_tavily_topic(" ".join(query_parts))
        adjusted_args = dict(args)
        adjusted_query = _expand_tavily_query_with_absolute_date(
            tool_query,
            latest_user_query,
            now=now,
        )
        if adjusted_query:
            adjusted_args["query"] = adjusted_query
        adjusted_args["topic"] = inferred_topic

        if adjusted_args != args:
            changed = True
            audit_logger.log_event(
                thread_id=thread_id,
                event="tool_call_adjusted",
                tool="tavily_web_search",
                original_args=args,
                adjusted_args=adjusted_args,
            )

        adjusted_tool_calls.append({**tool_call, "args": adjusted_args})

    if not changed:
        return response

    return response.model_copy(update={"tool_calls": adjusted_tool_calls})
