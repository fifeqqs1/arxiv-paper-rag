from __future__ import annotations

import threading

from .common import (
    ConversationMessageRecord,
    ConversationToolCallRecord,
    InboxEventRecord,
    InboxStatus,
    SessionRecord,
    SessionStatus,
    TaskRecord,
    TaskStatus,
    build_fts_query,
    build_fts_text,
    coerce_positive_int,
    compute_next_run,
    content_to_text,
    format_task_time,
    json_default,
    local_now_str,
    normalize_role,
    parse_task_time,
    safe_json_dumps,
    safe_json_loads,
    search_terms,
    short_preview,
    utc_now_iso,
)
from .conversations import ConversationRepository
from .sessions import SessionRepository
from .store import RuntimeStore, get_runtime_store
from .tasks import TaskRepository
from .writers import AsyncConversationWriter

_safe_json_loads = safe_json_loads
_safe_json_dumps = safe_json_dumps
_normalize_role = normalize_role
_short_preview = short_preview
_search_terms = search_terms
_build_fts_query = build_fts_query
_build_fts_text = build_fts_text
_content_to_text = content_to_text
_coerce_positive_int = coerce_positive_int
_json_default = json_default


_task_repository: TaskRepository | None = None
_task_repository_lock = threading.Lock()
_session_repository: SessionRepository | None = None
_session_repository_lock = threading.Lock()
_conversation_repository: ConversationRepository | None = None
_conversation_repository_lock = threading.Lock()
_conversation_writer: AsyncConversationWriter | None = None
_conversation_writer_lock = threading.Lock()


def get_task_repository(db_path: str | None = None) -> TaskRepository:
    global _task_repository
    if db_path is not None:
        return TaskRepository(get_runtime_store(db_path=db_path))

    with _task_repository_lock:
        if _task_repository is None:
            _task_repository = TaskRepository(get_runtime_store())
        return _task_repository


def get_session_repository(db_path: str | None = None) -> SessionRepository:
    global _session_repository
    if db_path is not None:
        return SessionRepository(get_runtime_store(db_path=db_path))

    with _session_repository_lock:
        if _session_repository is None:
            _session_repository = SessionRepository(get_runtime_store())
        return _session_repository


def get_conversation_repository(db_path: str | None = None) -> ConversationRepository:
    global _conversation_repository
    if db_path is not None:
        return ConversationRepository(get_runtime_store(db_path=db_path))

    with _conversation_repository_lock:
        if _conversation_repository is None:
            _conversation_repository = ConversationRepository(get_runtime_store())
        return _conversation_repository


def get_conversation_writer(db_path: str | None = None) -> AsyncConversationWriter:
    global _conversation_writer
    if db_path is not None:
        return AsyncConversationWriter(get_conversation_repository(db_path=db_path))

    with _conversation_writer_lock:
        if _conversation_writer is None:
            _conversation_writer = AsyncConversationWriter(get_conversation_repository())
        return _conversation_writer


__all__ = [
    "AsyncConversationWriter",
    "ConversationMessageRecord",
    "ConversationRepository",
    "ConversationToolCallRecord",
    "InboxEventRecord",
    "InboxStatus",
    "RuntimeStore",
    "SessionRecord",
    "SessionRepository",
    "SessionStatus",
    "TaskRecord",
    "TaskRepository",
    "TaskStatus",
    "build_fts_query",
    "build_fts_text",
    "coerce_positive_int",
    "compute_next_run",
    "content_to_text",
    "format_task_time",
    "get_conversation_repository",
    "get_conversation_writer",
    "get_runtime_store",
    "get_session_repository",
    "get_task_repository",
    "json_default",
    "local_now_str",
    "normalize_role",
    "parse_task_time",
    "safe_json_dumps",
    "safe_json_loads",
    "search_terms",
    "short_preview",
    "utc_now_iso",
]
