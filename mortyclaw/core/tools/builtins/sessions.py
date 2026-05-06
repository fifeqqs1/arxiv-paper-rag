from __future__ import annotations

import json
from typing import Any


def get_active_session_thread_id_impl(*, get_active_thread_id_fn) -> str:
    return get_active_thread_id_fn(default="local_geek_master")


def ensure_session_record_impl(
    thread_id: str,
    *,
    get_session_repository_fn,
    build_log_file_path_fn,
) -> None:
    get_session_repository_fn().upsert_session(
        thread_id=thread_id,
        display_name=thread_id,
        status="active",
        log_file=build_log_file_path_fn(thread_id),
    )


def load_session_todo_state_impl(
    thread_id: str,
    *,
    get_session_repository_fn,
) -> dict[str, Any]:
    session = get_session_repository_fn().get_session(thread_id)
    metadata: dict[str, Any] = {}
    if session is not None:
        try:
            metadata = json.loads(session.get("metadata_json", "{}"))
        except json.JSONDecodeError:
            metadata = {}
    todo_state = metadata.get("todo_state") if isinstance(metadata, dict) else {}
    return todo_state if isinstance(todo_state, dict) else {}


def search_sessions_impl(
    *,
    query: str,
    role_filter: str,
    limit: int,
    include_current: bool,
    include_tool_results: bool,
    current_thread_id: str,
    get_conversation_repository_fn,
) -> str:
    try:
        try:
            safe_limit = int(limit)
        except (TypeError, ValueError):
            safe_limit = 3
        safe_limit = max(1, min(safe_limit, 5))
        roles = [role.strip() for role in (role_filter or "").split(",") if role.strip()]
        results = get_conversation_repository_fn().search_sessions(
            query=query or "",
            role_filter=roles or None,
            limit=safe_limit,
            include_current=include_current,
            current_thread_id=current_thread_id,
            include_tool_results=include_tool_results,
        )
        return json.dumps(
            {
                "success": True,
                "query": query or "",
                "mode": "recent" if not (query or "").strip() else "search",
                "current_thread_id": current_thread_id,
                "count": len(results),
                "results": results,
            },
            ensure_ascii=False,
        )
    except Exception as exc:
        return json.dumps(
            {
                "success": False,
                "query": query or "",
                "error": str(exc),
            },
            ensure_ascii=False,
        )
