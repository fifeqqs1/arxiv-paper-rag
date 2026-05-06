from __future__ import annotations

from ..memory_policy import (
    build_long_term_memory_prompt as _build_long_term_memory_prompt_impl,
    build_session_memory_prompt as _build_session_memory_prompt_impl,
    load_long_term_profile_content as _load_long_term_profile_content_impl,
    load_session_memory_records as _load_session_memory_records_impl,
    load_session_project_path as _load_session_project_path_impl,
    schedule_long_term_memory_capture as _schedule_long_term_memory_capture_impl,
    sync_session_memory_from_query as _sync_session_memory_from_query_impl,
    with_working_memory as _with_working_memory_impl,
)
from ..prompts import summarize_discarded_context as _summarize_discarded_context_impl


def sync_session_memory_from_query(
    query: str | None,
    thread_id: str,
    *,
    get_memory_store_fn,
    build_memory_record_fn,
) -> dict:
    return _sync_session_memory_from_query_impl(
        query,
        thread_id,
        get_memory_store_fn=get_memory_store_fn,
        build_memory_record_fn=build_memory_record_fn,
    )


def load_session_memory_records(
    thread_id: str,
    *,
    get_memory_store_fn,
    limit: int,
) -> list[dict]:
    return _load_session_memory_records_impl(
        thread_id,
        get_memory_store_fn=get_memory_store_fn,
        limit=limit,
    )


def build_session_memory_prompt(
    thread_id: str,
    *,
    get_memory_store_fn,
    limit: int,
    prompt_cache,
) -> str:
    return _build_session_memory_prompt_impl(
        thread_id,
        get_memory_store_fn=get_memory_store_fn,
        limit=limit,
        prompt_cache=prompt_cache,
    )


def load_session_project_path(
    thread_id: str,
    *,
    get_memory_store_fn,
) -> str:
    return _load_session_project_path_impl(
        thread_id,
        get_memory_store_fn=get_memory_store_fn,
    )


def schedule_long_term_memory_capture(
    query: str | None,
    *,
    get_async_memory_writer_fn,
    build_memory_record_fn,
    default_long_term_scope: str,
) -> None:
    _schedule_long_term_memory_capture_impl(
        query,
        get_async_memory_writer_fn=get_async_memory_writer_fn,
        build_memory_record_fn=build_memory_record_fn,
        default_long_term_scope=default_long_term_scope,
    )


def load_long_term_profile_content(
    *,
    get_memory_store_fn,
    memory_dir: str,
    default_long_term_scope: str,
    user_profile_memory_type: str,
) -> str:
    return _load_long_term_profile_content_impl(
        get_memory_store_fn=get_memory_store_fn,
        memory_dir=memory_dir,
        default_long_term_scope=default_long_term_scope,
        user_profile_memory_type=user_profile_memory_type,
    )


def build_long_term_memory_prompt(
    query: str | None,
    *,
    get_memory_store_fn,
    memory_dir: str,
    default_long_term_scope: str,
    user_profile_memory_type: str,
    long_term_memory_prompt_limit: int,
    prompt_cache,
) -> str:
    return _build_long_term_memory_prompt_impl(
        query,
        get_memory_store_fn=get_memory_store_fn,
        memory_dir=memory_dir,
        default_long_term_scope=default_long_term_scope,
        user_profile_memory_type=user_profile_memory_type,
        long_term_memory_prompt_limit=long_term_memory_prompt_limit,
        prompt_cache=prompt_cache,
    )


def summarize_discarded_context(
    llm,
    current_summary: str,
    discarded_msgs: list,
    thread_id: str,
    *,
    state: dict | None,
    audit_logger_instance,
    timeout_seconds: float,
) -> str:
    return _summarize_discarded_context_impl(
        llm,
        current_summary,
        discarded_msgs,
        thread_id,
        state=state,
        audit_logger_instance=audit_logger_instance,
        timeout_seconds=timeout_seconds,
    )


def with_working_memory(
    state,
    updates: dict,
    *,
    build_working_memory_snapshot_fn,
) -> dict:
    return _with_working_memory_impl(
        state,
        updates,
        build_working_memory_snapshot_fn=build_working_memory_snapshot_fn,
    )
