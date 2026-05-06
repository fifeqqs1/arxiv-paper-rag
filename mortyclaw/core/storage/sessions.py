from __future__ import annotations

import json
import uuid
from typing import Any

from .common import InboxEventRecord, SessionRecord, SessionStatus, safe_json_loads, utc_now_iso
from .store import RuntimeStore


class SessionRepository:
    def __init__(self, store: RuntimeStore):
        self.store = store

    def upsert_session(
        self,
        *,
        thread_id: str,
        display_name: str | None = None,
        provider: str = "",
        model: str = "",
        status: SessionStatus = "active",
        log_file: str = "",
        metadata: dict | None = None,
        title: str = "",
        parent_thread_id: str = "",
        branch_from_message_uid: str = "",
        lineage_root_thread_id: str = "",
    ) -> SessionRecord:
        normalized_thread_id = (thread_id or "system_default").strip() or "system_default"
        now_utc = utc_now_iso()
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False)
        normalized_parent_thread_id = (parent_thread_id or "").strip()
        normalized_lineage_root_thread_id = (lineage_root_thread_id or "").strip() or normalized_parent_thread_id
        with self.store._lock:
            with self.store._connect() as conn:
                existing = conn.execute(
                    "SELECT * FROM sessions WHERE thread_id = ?",
                    (normalized_thread_id,),
                ).fetchone()
                created_at = existing["created_at"] if existing else now_utc
                conn.execute(
                    """
                    INSERT INTO sessions (
                        thread_id, display_name, provider, model, status, log_file,
                        created_at, updated_at, last_active_at, metadata_json,
                        title, parent_thread_id, branch_from_message_uid, lineage_root_thread_id,
                        message_count, tool_call_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(thread_id) DO UPDATE SET
                        display_name = excluded.display_name,
                        provider = CASE WHEN excluded.provider != '' THEN excluded.provider ELSE sessions.provider END,
                        model = CASE WHEN excluded.model != '' THEN excluded.model ELSE sessions.model END,
                        status = excluded.status,
                        log_file = CASE WHEN excluded.log_file != '' THEN excluded.log_file ELSE sessions.log_file END,
                        updated_at = excluded.updated_at,
                        last_active_at = excluded.last_active_at,
                        metadata_json = excluded.metadata_json,
                        title = CASE WHEN excluded.title != '' THEN excluded.title ELSE sessions.title END,
                        parent_thread_id = CASE
                            WHEN excluded.parent_thread_id != '' THEN excluded.parent_thread_id
                            ELSE sessions.parent_thread_id
                        END,
                        branch_from_message_uid = CASE
                            WHEN excluded.branch_from_message_uid != '' THEN excluded.branch_from_message_uid
                            ELSE sessions.branch_from_message_uid
                        END,
                        lineage_root_thread_id = CASE
                            WHEN excluded.lineage_root_thread_id != '' THEN excluded.lineage_root_thread_id
                            ELSE sessions.lineage_root_thread_id
                        END
                    """,
                    (
                        normalized_thread_id,
                        display_name or normalized_thread_id,
                        provider,
                        model,
                        status,
                        log_file,
                        created_at,
                        now_utc,
                        now_utc,
                        metadata_json,
                        title,
                        normalized_parent_thread_id,
                        branch_from_message_uid,
                        normalized_lineage_root_thread_id,
                        existing["message_count"] if existing else 0,
                        existing["tool_call_count"] if existing else 0,
                    ),
                )
                conn.commit()
                row = conn.execute(
                    "SELECT * FROM sessions WHERE thread_id = ?",
                    (normalized_thread_id,),
                ).fetchone()
        return dict(row)

    def touch_session(self, thread_id: str, *, status: SessionStatus | None = None) -> SessionRecord | None:
        session = self.get_session(thread_id)
        if session is None:
            return None

        next_status = status or session["status"]
        now_utc = utc_now_iso()
        with self.store._lock:
            with self.store._connect() as conn:
                conn.execute(
                    """
                    UPDATE sessions
                    SET status = ?, updated_at = ?, last_active_at = ?
                    WHERE thread_id = ?
                    """,
                    (next_status, now_utc, now_utc, thread_id),
                )
                conn.commit()
                row = conn.execute(
                    "SELECT * FROM sessions WHERE thread_id = ?",
                    (thread_id,),
                ).fetchone()
        return dict(row) if row else None

    def get_session(self, thread_id: str) -> SessionRecord | None:
        with self.store._lock:
            with self.store._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM sessions WHERE thread_id = ?",
                    (thread_id,),
                ).fetchone()
        return dict(row) if row else None

    def get_session_metadata(self, thread_id: str) -> dict[str, Any]:
        session = self.get_session(thread_id)
        if session is None:
            return {}
        return safe_json_loads(session.get("metadata_json"))

    def update_session_metadata(self, thread_id: str, metadata: dict[str, Any]) -> SessionRecord | None:
        session = self.get_session(thread_id)
        if session is None:
            return None
        merged_metadata = metadata if isinstance(metadata, dict) else {}
        return self.upsert_session(
            thread_id=thread_id,
            display_name=session["display_name"],
            provider=session["provider"],
            model=session["model"],
            status=session["status"],
            log_file=session["log_file"],
            metadata=merged_metadata,
            title=session["title"],
            parent_thread_id=session["parent_thread_id"],
            branch_from_message_uid=session["branch_from_message_uid"],
            lineage_root_thread_id=session["lineage_root_thread_id"],
        )

    def get_session_todo_state(self, thread_id: str) -> dict[str, Any]:
        metadata = self.get_session_metadata(thread_id)
        todo_state = metadata.get("todo_state")
        return todo_state if isinstance(todo_state, dict) else {}

    def save_session_todo_state(self, thread_id: str, todo_state: dict[str, Any]) -> SessionRecord:
        session = self.get_session(thread_id)
        if session is None:
            session = self.upsert_session(thread_id=thread_id, display_name=thread_id)
        metadata = safe_json_loads(session.get("metadata_json"))
        metadata["todo_state"] = todo_state if isinstance(todo_state, dict) else {}
        return self.update_session_metadata(thread_id, metadata) or session

    def clear_session_todo_state(self, thread_id: str) -> SessionRecord | None:
        session = self.get_session(thread_id)
        if session is None:
            return None
        metadata = safe_json_loads(session.get("metadata_json"))
        if "todo_state" in metadata:
            metadata.pop("todo_state", None)
        return self.update_session_metadata(thread_id, metadata)

    def list_sessions(
        self,
        *,
        statuses: tuple[str, ...] | None = None,
        limit: int = 100,
    ) -> list[SessionRecord]:
        params: list[object] = []
        where_sql = ""
        if statuses:
            placeholders = ", ".join("?" for _ in statuses)
            where_sql = f"WHERE status IN ({placeholders})"
            params.extend(statuses)
        params.append(limit)

        with self.store._lock:
            with self.store._connect() as conn:
                rows = conn.execute(
                    f"""
                    SELECT *
                    FROM sessions
                    {where_sql}
                    ORDER BY last_active_at DESC, updated_at DESC
                    LIMIT ?
                    """,
                    tuple(params),
                ).fetchall()
        return [dict(row) for row in rows]

    def get_latest_session(self) -> SessionRecord | None:
        sessions = self.list_sessions(limit=1)
        return sessions[0] if sessions else None

    def create_branch_session(
        self,
        *,
        parent_thread_id: str,
        branch_thread_id: str,
        branch_from_message_uid: str = "",
        provider: str = "",
        model: str = "",
        title: str = "",
    ) -> SessionRecord:
        parent = self.get_session(parent_thread_id)
        parent_root = (
            (parent or {}).get("lineage_root_thread_id")
            or (parent or {}).get("parent_thread_id")
            or parent_thread_id
        )
        branch_title = title or f"{parent_thread_id} branch"
        return self.upsert_session(
            thread_id=branch_thread_id,
            display_name=branch_thread_id,
            provider=provider or (parent or {}).get("provider", ""),
            model=model or (parent or {}).get("model", ""),
            status="idle",
            title=branch_title,
            parent_thread_id=parent_thread_id,
            branch_from_message_uid=branch_from_message_uid,
            lineage_root_thread_id=parent_root,
        )

    def enqueue_inbox_event(
        self,
        *,
        thread_id: str,
        event_type: str,
        payload: dict,
    ) -> InboxEventRecord:
        now_utc = utc_now_iso()
        event_id = str(uuid.uuid4())
        serialized_payload = json.dumps(payload, ensure_ascii=False)
        with self.store._lock:
            with self.store._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO session_inbox (
                        event_id, thread_id, event_type, payload, status, created_at, delivered_at
                    ) VALUES (?, ?, ?, ?, 'pending', ?, NULL)
                    """,
                    (
                        event_id,
                        thread_id,
                        event_type,
                        serialized_payload,
                        now_utc,
                    ),
                )
                conn.commit()
                row = conn.execute(
                    "SELECT * FROM session_inbox WHERE event_id = ?",
                    (event_id,),
                ).fetchone()
        return dict(row)

    def list_pending_inbox_events(self, thread_id: str, *, limit: int = 50) -> list[InboxEventRecord]:
        with self.store._lock:
            with self.store._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT *
                    FROM session_inbox
                    WHERE thread_id = ? AND status = 'pending'
                    ORDER BY created_at ASC
                    LIMIT ?
                    """,
                    (thread_id, limit),
                ).fetchall()
        return [dict(row) for row in rows]

    def mark_inbox_event_delivered(self, event_id: str) -> InboxEventRecord | None:
        delivered_at = utc_now_iso()
        with self.store._lock:
            with self.store._connect() as conn:
                conn.execute(
                    """
                    UPDATE session_inbox
                    SET status = 'delivered', delivered_at = ?
                    WHERE event_id = ?
                    """,
                    (delivered_at, event_id),
                )
                conn.commit()
                row = conn.execute(
                    "SELECT * FROM session_inbox WHERE event_id = ?",
                    (event_id,),
                ).fetchone()
        return dict(row) if row else None
