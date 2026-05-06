from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, TypedDict


TaskStatus = Literal["scheduled", "completed", "cancelled", "failed"]
SessionStatus = Literal["active", "idle", "closed"]
InboxStatus = Literal["pending", "delivered"]


class TaskRecord(TypedDict):
    task_id: str
    thread_id: str
    description: str
    target_time: str
    repeat: str | None
    repeat_count: int | None
    remaining_runs: int | None
    status: TaskStatus
    created_at: str
    updated_at: str
    last_run_at: str | None


class SessionRecord(TypedDict):
    thread_id: str
    display_name: str
    provider: str
    model: str
    status: SessionStatus
    log_file: str
    created_at: str
    updated_at: str
    last_active_at: str
    metadata_json: str
    title: str
    parent_thread_id: str
    branch_from_message_uid: str
    lineage_root_thread_id: str
    message_count: int
    tool_call_count: int


class InboxEventRecord(TypedDict):
    event_id: str
    thread_id: str
    event_type: str
    payload: str
    status: InboxStatus
    created_at: str
    delivered_at: str | None


class ConversationMessageRecord(TypedDict):
    id: int
    message_uid: str
    thread_id: str
    turn_id: str
    seq: int
    role: str
    content: str
    node_name: str
    route: str
    tool_call_id: str | None
    tool_name: str | None
    tool_calls_json: str
    response_metadata_json: str
    usage_metadata_json: str
    created_at: str
    metadata_json: str


class ConversationToolCallRecord(TypedDict):
    tool_call_id: str
    thread_id: str
    turn_id: str
    assistant_message_uid: str
    tool_name: str
    args_json: str
    result_message_uid: str | None
    result_preview: str
    status: str
    created_at: str
    finished_at: str | None
    metadata_json: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def local_now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_json_loads(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def parse_task_time(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")


def format_task_time(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


def compute_next_run(target_time: str, repeat: str | None) -> str | None:
    if not repeat:
        return None

    target_dt = parse_task_time(target_time)
    repeat_mode = (repeat or "").strip().lower()
    if repeat_mode == "hourly":
        return format_task_time(target_dt + timedelta(hours=1))
    if repeat_mode == "daily":
        return format_task_time(target_dt + timedelta(days=1))
    if repeat_mode == "weekly":
        return format_task_time(target_dt + timedelta(days=7))
    raise ValueError(f"unsupported repeat mode: {repeat}")


def coerce_positive_int(value) -> int | None:
    if value in (None, ""):
        return None
    return max(int(value), 0)


def json_default(value: Any) -> str:
    return str(value)


def safe_json_dumps(value: Any) -> str:
    try:
        return json.dumps(value if value is not None else {}, ensure_ascii=False, default=json_default)
    except TypeError:
        return json.dumps(str(value), ensure_ascii=False)


def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return safe_json_dumps(content)


def normalize_role(message_type: str | None) -> str:
    role = (message_type or "").strip().lower()
    return {
        "human": "user",
        "ai": "assistant",
        "tool": "tool",
        "system": "system",
    }.get(role, role or "unknown")


def short_preview(text: str, limit: int = 600) -> str:
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1] + "…"


def search_terms(text: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()

    def add(term: str) -> None:
        normalized = term.strip().lower()
        if len(normalized) < 2 or normalized in seen:
            return
        seen.add(normalized)
        terms.append(normalized)

    for token in re.findall(r"[A-Za-z0-9_./:-]+", text or ""):
        add(token)
    for chunk in re.findall(r"[\u4e00-\u9fff]+", text or ""):
        if len(chunk) == 2:
            add(chunk)
        elif len(chunk) > 2:
            for index in range(len(chunk) - 1):
                add(chunk[index:index + 2])
    return terms


def build_fts_query(query: str) -> str:
    escaped_terms = [f'"{term.replace(chr(34), "")}"' for term in search_terms(query)]
    return " OR ".join(escaped_terms)


def build_fts_text(*parts: str) -> str:
    text = " ".join(part for part in parts if part)
    return f"{text} {' '.join(search_terms(text))}".strip()
