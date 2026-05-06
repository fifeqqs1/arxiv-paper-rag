from __future__ import annotations

import contextvars
import threading


_active_thread_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "mortyclaw_active_thread_id",
    default="system_default",
)
_thread_local = threading.local()


def set_active_thread_id(thread_id: str | None) -> str:
    normalized = (thread_id or "system_default").strip() or "system_default"
    _active_thread_id_var.set(normalized)
    _thread_local.thread_id = normalized
    return normalized


def get_active_thread_id(default: str = "system_default") -> str:
    thread_id = _active_thread_id_var.get(default)
    if thread_id and thread_id != default:
        return thread_id

    local_thread_id = getattr(_thread_local, "thread_id", "")
    if local_thread_id:
        return local_thread_id
    return default
