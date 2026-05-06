from __future__ import annotations

import os
import sqlite3
import threading

from ..config import RUNTIME_DB_PATH


class RuntimeStore:
    def __init__(self, db_path: str = RUNTIME_DB_PATH):
        self.db_path = db_path
        self._lock = threading.Lock()
        self.ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def ensure_schema(self) -> None:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with self._lock:
            with self._connect() as conn:
                try:
                    conn.execute("PRAGMA journal_mode = WAL")
                except sqlite3.OperationalError:
                    pass
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sessions (
                        thread_id TEXT PRIMARY KEY,
                        display_name TEXT NOT NULL,
                        provider TEXT NOT NULL DEFAULT '',
                        model TEXT NOT NULL DEFAULT '',
                        status TEXT NOT NULL DEFAULT 'idle',
                        log_file TEXT NOT NULL DEFAULT '',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        last_active_at TEXT NOT NULL,
                        metadata_json TEXT NOT NULL DEFAULT '{}'
                    )
                    """
                )
                self._ensure_column(conn, "sessions", "title", "TEXT NOT NULL DEFAULT ''")
                self._ensure_column(conn, "sessions", "parent_thread_id", "TEXT NOT NULL DEFAULT ''")
                self._ensure_column(conn, "sessions", "branch_from_message_uid", "TEXT NOT NULL DEFAULT ''")
                self._ensure_column(conn, "sessions", "lineage_root_thread_id", "TEXT NOT NULL DEFAULT ''")
                self._ensure_column(conn, "sessions", "message_count", "INTEGER NOT NULL DEFAULT 0")
                self._ensure_column(conn, "sessions", "tool_call_count", "INTEGER NOT NULL DEFAULT 0")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS tasks (
                        task_id TEXT PRIMARY KEY,
                        thread_id TEXT NOT NULL,
                        description TEXT NOT NULL,
                        target_time TEXT NOT NULL,
                        repeat TEXT,
                        repeat_count INTEGER,
                        remaining_runs INTEGER,
                        status TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        last_run_at TEXT
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS task_runs (
                        run_id TEXT PRIMARY KEY,
                        task_id TEXT NOT NULL,
                        thread_id TEXT NOT NULL,
                        status TEXT NOT NULL,
                        triggered_at TEXT NOT NULL,
                        finished_at TEXT NOT NULL,
                        result_summary TEXT NOT NULL DEFAULT '',
                        error_message TEXT NOT NULL DEFAULT ''
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS session_inbox (
                        event_id TEXT PRIMARY KEY,
                        thread_id TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        payload TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'pending',
                        created_at TEXT NOT NULL,
                        delivered_at TEXT
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_tasks_thread_status_target
                    ON tasks(thread_id, status, target_time)
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_tasks_status_target
                    ON tasks(status, target_time)
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_sessions_last_active
                    ON sessions(last_active_at DESC)
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_session_inbox_thread_status_created
                    ON session_inbox(thread_id, status, created_at)
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_task_runs_task_triggered
                    ON task_runs(task_id, triggered_at DESC)
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversation_messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        message_uid TEXT UNIQUE NOT NULL,
                        thread_id TEXT NOT NULL,
                        turn_id TEXT NOT NULL,
                        seq INTEGER NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL DEFAULT '',
                        node_name TEXT NOT NULL DEFAULT '',
                        route TEXT NOT NULL DEFAULT '',
                        tool_call_id TEXT,
                        tool_name TEXT,
                        tool_calls_json TEXT NOT NULL DEFAULT '[]',
                        response_metadata_json TEXT NOT NULL DEFAULT '{}',
                        usage_metadata_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL,
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        UNIQUE(thread_id, seq)
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversation_tool_calls (
                        tool_call_id TEXT PRIMARY KEY,
                        thread_id TEXT NOT NULL,
                        turn_id TEXT NOT NULL,
                        assistant_message_uid TEXT NOT NULL,
                        tool_name TEXT NOT NULL,
                        args_json TEXT NOT NULL DEFAULT '{}',
                        result_message_uid TEXT,
                        result_preview TEXT NOT NULL DEFAULT '',
                        status TEXT NOT NULL DEFAULT 'called',
                        created_at TEXT NOT NULL,
                        finished_at TEXT,
                        metadata_json TEXT NOT NULL DEFAULT '{}'
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS conversation_messages_fts
                    USING fts5(
                        message_uid UNINDEXED,
                        thread_id UNINDEXED,
                        role UNINDEXED,
                        tool_name UNINDEXED,
                        search_text,
                        tokenize = 'unicode61'
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversation_summaries (
                        summary_id TEXT PRIMARY KEY,
                        thread_id TEXT NOT NULL,
                        start_message_uid TEXT NOT NULL DEFAULT '',
                        end_message_uid TEXT NOT NULL DEFAULT '',
                        summary_type TEXT NOT NULL,
                        summary TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        metadata_json TEXT NOT NULL DEFAULT '{}'
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_conversation_messages_thread_seq
                    ON conversation_messages(thread_id, seq)
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_conversation_messages_thread_created
                    ON conversation_messages(thread_id, created_at DESC)
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_conversation_tool_calls_thread_created
                    ON conversation_tool_calls(thread_id, created_at DESC)
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_conversation_tool_calls_tool_name
                    ON conversation_tool_calls(tool_name, created_at DESC)
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_conversation_summaries_thread_created
                    ON conversation_summaries(thread_id, created_at DESC)
                    """
                )
                conn.commit()

    def _ensure_column(
        self,
        conn: sqlite3.Connection,
        table_name: str,
        column_name: str,
        column_definition: str,
    ) -> None:
        columns = {
            row["name"]
            for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        if column_name not in columns:
            conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}")


_default_runtime_store: RuntimeStore | None = None
_default_runtime_store_lock = threading.Lock()


def get_runtime_store(db_path: str | None = None) -> RuntimeStore:
    global _default_runtime_store
    if db_path is not None:
        return RuntimeStore(db_path=db_path)

    with _default_runtime_store_lock:
        if _default_runtime_store is None:
            _default_runtime_store = RuntimeStore()
        return _default_runtime_store
