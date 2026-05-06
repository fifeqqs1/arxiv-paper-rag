from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from typing import Any

from .common import (
    ConversationMessageRecord,
    build_fts_query,
    build_fts_text,
    content_to_text,
    normalize_role,
    safe_json_dumps,
    short_preview,
    utc_now_iso,
)
from .store import RuntimeStore


class ConversationRepository:
    def __init__(self, store: RuntimeStore):
        self.store = store

    def append_messages(
        self,
        *,
        thread_id: str,
        turn_id: str,
        messages: list[Any],
        node_name: str = "",
        route: str = "",
    ) -> list[ConversationMessageRecord]:
        normalized_thread_id = (thread_id or "system_default").strip() or "system_default"
        normalized_turn_id = (turn_id or str(uuid.uuid4())).strip() or str(uuid.uuid4())
        if not messages:
            return []

        inserted: list[ConversationMessageRecord] = []
        with self.store._lock:
            with self.store._connect() as conn:
                self._ensure_session_row(conn, normalized_thread_id)
                seq = self._next_message_seq(conn, normalized_thread_id)
                for index, message in enumerate(messages):
                    if not self._is_persistable_message(message):
                        continue
                    record = self._message_to_record(
                        message,
                        thread_id=normalized_thread_id,
                        turn_id=normalized_turn_id,
                        seq=seq,
                        node_name=node_name,
                        route=route,
                        index=index,
                    )
                    seq += 1
                    cursor = conn.execute(
                        """
                        INSERT OR IGNORE INTO conversation_messages (
                            message_uid, thread_id, turn_id, seq, role, content, node_name, route,
                            tool_call_id, tool_name, tool_calls_json, response_metadata_json,
                            usage_metadata_json, created_at, metadata_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            record["message_uid"],
                            record["thread_id"],
                            record["turn_id"],
                            record["seq"],
                            record["role"],
                            record["content"],
                            record["node_name"],
                            record["route"],
                            record["tool_call_id"],
                            record["tool_name"],
                            record["tool_calls_json"],
                            record["response_metadata_json"],
                            record["usage_metadata_json"],
                            record["created_at"],
                            record["metadata_json"],
                        ),
                    )
                    if cursor.rowcount:
                        self._sync_message_fts(conn, record)
                        self._record_tool_calls_from_message(conn, record, message)
                        self._link_tool_result_from_message(conn, record, message)
                        inserted.append(record)
                if inserted:
                    self._refresh_session_stats(conn, normalized_thread_id)
                conn.commit()
        return inserted

    def record_conversation_summary(
        self,
        *,
        thread_id: str,
        summary: str,
        summary_type: str = "context_compression",
        messages: list[Any] | None = None,
        metadata: dict | None = None,
        summary_id: str | None = None,
    ) -> dict:
        normalized_thread_id = (thread_id or "system_default").strip() or "system_default"
        message_uids = [
            str(getattr(message, "id", "") or "")
            for message in (messages or [])
            if getattr(message, "id", None)
        ]
        record = {
            "summary_id": summary_id or str(uuid.uuid4()),
            "thread_id": normalized_thread_id,
            "start_message_uid": message_uids[0] if message_uids else "",
            "end_message_uid": message_uids[-1] if message_uids else "",
            "summary_type": summary_type,
            "summary": summary or "",
            "created_at": utc_now_iso(),
            "metadata_json": safe_json_dumps(metadata or {}),
        }
        with self.store._lock:
            with self.store._connect() as conn:
                self._ensure_session_row(conn, normalized_thread_id)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO conversation_summaries (
                        summary_id, thread_id, start_message_uid, end_message_uid,
                        summary_type, summary, created_at, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record["summary_id"],
                        record["thread_id"],
                        record["start_message_uid"],
                        record["end_message_uid"],
                        record["summary_type"],
                        record["summary"],
                        record["created_at"],
                        record["metadata_json"],
                    ),
                )
                conn.commit()
        return record

    def list_recent_sessions(self, *, limit: int = 10, exclude_thread_id: str | None = None) -> list[dict]:
        limit = max(1, min(int(limit or 10), 50))
        with self.store._lock:
            with self.store._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT *
                    FROM sessions
                    ORDER BY last_active_at DESC, updated_at DESC
                    LIMIT ?
                    """,
                    (limit + 5,),
                ).fetchall()
                result = []
                for row in rows:
                    session = dict(row)
                    if exclude_thread_id and session["thread_id"] == exclude_thread_id:
                        continue
                    preview_row = conn.execute(
                        """
                        SELECT role, content, tool_name, created_at
                        FROM conversation_messages
                        WHERE thread_id = ? AND content != ''
                        ORDER BY seq DESC
                        LIMIT 1
                        """,
                        (session["thread_id"],),
                    ).fetchone()
                    preview = short_preview(preview_row["content"], 220) if preview_row else ""
                    result.append({
                        "thread_id": session["thread_id"],
                        "title": session.get("title") or session.get("display_name") or session["thread_id"],
                        "status": session.get("status", ""),
                        "model": session.get("model", ""),
                        "last_active_at": session.get("last_active_at", ""),
                        "message_count": session.get("message_count", 0),
                        "tool_call_count": session.get("tool_call_count", 0),
                        "preview": preview,
                    })
                    if len(result) >= limit:
                        break
        return result

    def get_session_conversation(self, thread_id: str, *, limit: int | None = None) -> list[ConversationMessageRecord]:
        normalized_thread_id = (thread_id or "").strip()
        if not normalized_thread_id:
            return []
        params: list[Any] = [normalized_thread_id]
        limit_sql = ""
        if limit is not None:
            limit_sql = "LIMIT ?"
            params.append(max(1, int(limit)))

        with self.store._lock:
            with self.store._connect() as conn:
                rows = conn.execute(
                    f"""
                    SELECT *
                    FROM conversation_messages
                    WHERE thread_id = ?
                    ORDER BY seq ASC
                    {limit_sql}
                    """,
                    tuple(params),
                ).fetchall()
        return [dict(row) for row in rows]

    def search_sessions(
        self,
        query: str,
        *,
        role_filter: list[str] | tuple[str, ...] | None = None,
        limit: int = 3,
        include_current: bool = False,
        current_thread_id: str | None = None,
        include_tool_results: bool = True,
    ) -> list[dict]:
        normalized_query = (query or "").strip()
        if not normalized_query:
            return self.list_recent_sessions(limit=limit, exclude_thread_id=None if include_current else current_thread_id)

        fts_query = build_fts_query(normalized_query)
        if not fts_query:
            return []

        limit = max(1, min(int(limit or 3), 5))
        role_values = [normalize_role(role) for role in (role_filter or []) if role]
        raw_limit = max(25, limit * 20)

        with self.store._lock:
            with self.store._connect() as conn:
                current_root = self._resolve_lineage_root(conn, current_thread_id) if current_thread_id else None
                conditions = ["conversation_messages_fts MATCH ?"]
                params: list[Any] = [fts_query]
                if role_values:
                    placeholders = ", ".join("?" for _ in role_values)
                    conditions.append(f"m.role IN ({placeholders})")
                    params.extend(role_values)
                if not include_tool_results:
                    conditions.append("m.role != 'tool'")
                params.append(raw_limit)

                rows = conn.execute(
                    f"""
                    SELECT
                        m.*,
                        s.title,
                        s.display_name,
                        s.provider,
                        s.model,
                        s.created_at AS session_created_at,
                        s.last_active_at,
                        s.parent_thread_id,
                        s.lineage_root_thread_id,
                        snippet(conversation_messages_fts, 4, '[', ']', ' … ', 14) AS snippet,
                        bm25(conversation_messages_fts) AS rank
                    FROM conversation_messages_fts
                    JOIN conversation_messages AS m
                        ON m.message_uid = conversation_messages_fts.message_uid
                    LEFT JOIN sessions AS s
                        ON s.thread_id = m.thread_id
                    WHERE {" AND ".join(conditions)}
                    ORDER BY rank ASC, m.created_at DESC
                    LIMIT ?
                    """,
                    tuple(params),
                ).fetchall()

                grouped: dict[str, dict] = {}
                for row in rows:
                    item = dict(row)
                    thread_id = item["thread_id"]
                    if not include_current:
                        result_root = item.get("lineage_root_thread_id") or thread_id
                        if thread_id == current_thread_id or (current_root and result_root == current_root):
                            continue
                    if thread_id not in grouped:
                        grouped[thread_id] = {
                            "thread_id": thread_id,
                            "title": item.get("title") or item.get("display_name") or thread_id,
                            "when": item.get("last_active_at") or item.get("session_created_at") or item.get("created_at"),
                            "model": "/".join(
                                part for part in (item.get("provider"), item.get("model")) if part
                            ),
                            "hits": [],
                        }
                    if len(grouped[thread_id]["hits"]) < 3:
                        grouped[thread_id]["hits"].append({
                            "message_uid": item["message_uid"],
                            "seq": item["seq"],
                            "role": item["role"],
                            "tool_name": item.get("tool_name") or "",
                            "snippet": item.get("snippet") or short_preview(item.get("content", ""), 260),
                            "content_preview": short_preview(item.get("content", ""), 500),
                            "tool_result_preview": self._tool_result_preview(conn, item),
                            "window": self._message_window(
                                conn,
                                thread_id=thread_id,
                                seq=item["seq"],
                                include_tool_results=include_tool_results,
                            ),
                        })
                    if len(grouped) >= limit and all(len(value["hits"]) >= 1 for value in grouped.values()):
                        break
        return list(grouped.values())[:limit]

    def _is_persistable_message(self, message: Any) -> bool:
        role = normalize_role(getattr(message, "type", ""))
        return role in {"user", "assistant", "tool", "system"}

    def _next_message_seq(self, conn: sqlite3.Connection, thread_id: str) -> int:
        row = conn.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 AS next_seq FROM conversation_messages WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()
        return int(row["next_seq"] if row else 1)

    def _message_to_record(
        self,
        message: Any,
        *,
        thread_id: str,
        turn_id: str,
        seq: int,
        node_name: str,
        route: str,
        index: int,
    ) -> ConversationMessageRecord:
        role = normalize_role(getattr(message, "type", ""))
        content = content_to_text(getattr(message, "content", ""))
        tool_calls = self._extract_tool_calls(message)
        tool_call_id = getattr(message, "tool_call_id", None) or None
        tool_name = getattr(message, "name", None) or None
        additional_kwargs = getattr(message, "additional_kwargs", {}) or {}
        artifact_metadata = additional_kwargs.get("mortyclaw_artifact") or {}
        error_metadata = additional_kwargs.get("mortyclaw_error") or {}
        if not tool_name and tool_calls:
            tool_name = str(tool_calls[0].get("name") or "")
        message_uid = self._message_uid(
            message,
            thread_id=thread_id,
            turn_id=turn_id,
            node_name=node_name,
            index=index,
            role=role,
            content=content,
            tool_call_id=tool_call_id,
            tool_calls=tool_calls,
        )
        return {
            "id": 0,
            "message_uid": message_uid,
            "thread_id": thread_id,
            "turn_id": turn_id,
            "seq": seq,
            "role": role,
            "content": content,
            "node_name": node_name or "",
            "route": route or "",
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "tool_calls_json": safe_json_dumps(tool_calls),
            "response_metadata_json": safe_json_dumps(getattr(message, "response_metadata", {}) or {}),
            "usage_metadata_json": safe_json_dumps(getattr(message, "usage_metadata", {}) or {}),
            "created_at": utc_now_iso(),
            "metadata_json": safe_json_dumps({
                "raw_type": getattr(message, "type", ""),
                "name": getattr(message, "name", None),
                "artifact": artifact_metadata if isinstance(artifact_metadata, dict) else {},
                "error": error_metadata if isinstance(error_metadata, dict) else {},
            }),
        }

    def _message_uid(
        self,
        message: Any,
        *,
        thread_id: str,
        turn_id: str,
        node_name: str,
        index: int,
        role: str,
        content: str,
        tool_call_id: str | None,
        tool_calls: list[dict],
    ) -> str:
        existing_id = getattr(message, "id", None)
        if existing_id:
            return str(existing_id)
        basis = "|".join((
            thread_id,
            turn_id,
            node_name or "",
            str(index),
            role,
            tool_call_id or "",
            content,
            safe_json_dumps(tool_calls),
        ))
        digest = hashlib.sha1(basis.encode("utf-8", "ignore")).hexdigest()
        return f"msg-{digest}"

    def _extract_tool_calls(self, message: Any) -> list[dict]:
        raw_tool_calls = getattr(message, "tool_calls", None) or []
        additional_kwargs = getattr(message, "additional_kwargs", {}) or {}
        if not raw_tool_calls:
            raw_tool_calls = additional_kwargs.get("tool_calls") or []

        normalized = []
        for index, tool_call in enumerate(raw_tool_calls):
            if not isinstance(tool_call, dict):
                continue
            if "function" in tool_call:
                function = tool_call.get("function") or {}
                args = function.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}
                normalized.append({
                    "id": tool_call.get("id") or f"tool-call-{index}",
                    "name": function.get("name") or tool_call.get("name") or "",
                    "args": args,
                })
            else:
                normalized.append({
                    "id": tool_call.get("id") or tool_call.get("tool_call_id") or f"tool-call-{index}",
                    "name": tool_call.get("name") or "",
                    "args": tool_call.get("args") or {},
                })
        return normalized

    def _record_tool_calls_from_message(
        self,
        conn: sqlite3.Connection,
        record: ConversationMessageRecord,
        message: Any,
    ) -> None:
        if record["role"] != "assistant":
            return
        for tool_call in self._extract_tool_calls(message):
            tool_call_id = str(tool_call.get("id") or "")
            if not tool_call_id:
                continue
            conn.execute(
                """
                INSERT INTO conversation_tool_calls (
                    tool_call_id, thread_id, turn_id, assistant_message_uid,
                    tool_name, args_json, result_message_uid, result_preview,
                    status, created_at, finished_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, NULL, '', 'called', ?, NULL, '{}')
                ON CONFLICT(tool_call_id) DO UPDATE SET
                    thread_id = excluded.thread_id,
                    turn_id = excluded.turn_id,
                    assistant_message_uid = excluded.assistant_message_uid,
                    tool_name = excluded.tool_name,
                    args_json = excluded.args_json,
                    status = CASE
                        WHEN conversation_tool_calls.status = 'finished' THEN conversation_tool_calls.status
                        ELSE excluded.status
                    END
                """,
                (
                    tool_call_id,
                    record["thread_id"],
                    record["turn_id"],
                    record["message_uid"],
                    str(tool_call.get("name") or ""),
                    safe_json_dumps(tool_call.get("args") or {}),
                    record["created_at"],
                ),
            )

    def _link_tool_result_from_message(
        self,
        conn: sqlite3.Connection,
        record: ConversationMessageRecord,
        message: Any,
    ) -> None:
        if record["role"] != "tool":
            return
        tool_call_id = getattr(message, "tool_call_id", None) or record.get("tool_call_id")
        if not tool_call_id:
            return
        tool_name = getattr(message, "name", None) or record.get("tool_name") or "unknown_tool"
        preview = short_preview(record["content"], 800)
        finished_at = utc_now_iso()
        message_metadata_json = record.get("metadata_json") or "{}"
        cursor = conn.execute(
            """
            UPDATE conversation_tool_calls
            SET result_message_uid = ?, result_preview = ?, status = 'finished', finished_at = ?,
                tool_name = CASE WHEN tool_name = '' THEN ? ELSE tool_name END,
                metadata_json = ?
            WHERE tool_call_id = ?
            """,
            (record["message_uid"], preview, finished_at, tool_name, message_metadata_json, tool_call_id),
        )
        if cursor.rowcount == 0:
            conn.execute(
                """
                INSERT INTO conversation_tool_calls (
                    tool_call_id, thread_id, turn_id, assistant_message_uid,
                    tool_name, args_json, result_message_uid, result_preview,
                    status, created_at, finished_at, metadata_json
                ) VALUES (?, ?, ?, '', ?, '{}', ?, ?, 'finished', ?, ?, ?)
                """,
                (
                    tool_call_id,
                    record["thread_id"],
                    record["turn_id"],
                    tool_name,
                    record["message_uid"],
                    preview,
                    record["created_at"],
                    finished_at,
                    message_metadata_json,
                ),
            )

    def _sync_message_fts(self, conn: sqlite3.Connection, record: ConversationMessageRecord) -> None:
        conn.execute(
            "DELETE FROM conversation_messages_fts WHERE message_uid = ?",
            (record["message_uid"],),
        )
        search_text = build_fts_text(
            record["role"],
            record.get("tool_name") or "",
            record["content"],
            record["tool_calls_json"],
        )
        conn.execute(
            """
            INSERT INTO conversation_messages_fts (message_uid, thread_id, role, tool_name, search_text)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                record["message_uid"],
                record["thread_id"],
                record["role"],
                record.get("tool_name") or "",
                search_text,
            ),
        )

    def _ensure_session_row(self, conn: sqlite3.Connection, thread_id: str) -> None:
        now_utc = utc_now_iso()
        conn.execute(
            """
            INSERT OR IGNORE INTO sessions (
                thread_id, display_name, provider, model, status, log_file,
                created_at, updated_at, last_active_at, metadata_json,
                title, parent_thread_id, branch_from_message_uid, lineage_root_thread_id,
                message_count, tool_call_count
            ) VALUES (?, ?, '', '', 'idle', '', ?, ?, ?, '{}', '', '', '', ?, 0, 0)
            """,
            (thread_id, thread_id, now_utc, now_utc, now_utc, thread_id),
        )

    def _refresh_session_stats(self, conn: sqlite3.Connection, thread_id: str) -> None:
        now_utc = utc_now_iso()
        message_count = conn.execute(
            "SELECT COUNT(*) AS count FROM conversation_messages WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()["count"]
        tool_call_count = conn.execute(
            "SELECT COUNT(*) AS count FROM conversation_tool_calls WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()["count"]
        title_row = conn.execute(
            """
            SELECT content
            FROM conversation_messages
            WHERE thread_id = ? AND role = 'user' AND content != ''
            ORDER BY seq ASC
            LIMIT 1
            """,
            (thread_id,),
        ).fetchone()
        title = short_preview(title_row["content"], 80) if title_row else thread_id
        conn.execute(
            """
            UPDATE sessions
            SET message_count = ?,
                tool_call_count = ?,
                title = CASE WHEN title = '' THEN ? ELSE title END,
                lineage_root_thread_id = CASE WHEN lineage_root_thread_id = '' THEN ? ELSE lineage_root_thread_id END,
                updated_at = ?,
                last_active_at = ?
            WHERE thread_id = ?
            """,
            (message_count, tool_call_count, title, thread_id, now_utc, now_utc, thread_id),
        )

    def _resolve_lineage_root(self, conn: sqlite3.Connection, thread_id: str | None) -> str | None:
        if not thread_id:
            return None
        row = conn.execute(
            "SELECT parent_thread_id, lineage_root_thread_id FROM sessions WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()
        if not row:
            return thread_id
        return row["lineage_root_thread_id"] or row["parent_thread_id"] or thread_id

    def _message_window(
        self,
        conn: sqlite3.Connection,
        *,
        thread_id: str,
        seq: int,
        include_tool_results: bool,
        before: int = 2,
        after: int = 2,
    ) -> list[dict]:
        conditions = ["thread_id = ?", "seq BETWEEN ? AND ?"]
        params: list[Any] = [thread_id, max(1, seq - before), seq + after]
        if not include_tool_results:
            conditions.append("role != 'tool'")
        rows = conn.execute(
            f"""
            SELECT message_uid, seq, role, content, tool_name, created_at
            FROM conversation_messages
            WHERE {" AND ".join(conditions)}
            ORDER BY seq ASC
            """,
            tuple(params),
        ).fetchall()
        return [
            {
                "message_uid": row["message_uid"],
                "seq": row["seq"],
                "role": row["role"],
                "tool_name": row["tool_name"] or "",
                "content_preview": short_preview(row["content"], 500),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def _tool_result_preview(self, conn: sqlite3.Connection, item: dict) -> str:
        tool_call_id = item.get("tool_call_id")
        if not tool_call_id:
            return ""
        row = conn.execute(
            "SELECT result_preview FROM conversation_tool_calls WHERE tool_call_id = ?",
            (tool_call_id,),
        ).fetchone()
        return row["result_preview"] if row else ""
