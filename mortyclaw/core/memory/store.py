import os
import queue
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Literal, TypedDict

from ..config import MEMORY_DB_PATH

MemoryLayer = Literal["working", "session", "long_term"]
MemoryStatus = Literal["active", "superseded", "archived", "expired", "deleted"]
DEFAULT_LONG_TERM_SCOPE = "user_default"
USER_PROFILE_MEMORY_TYPE = "user_profile_snapshot"
USER_PROFILE_MEMORY_ID = f"long_term::{DEFAULT_LONG_TERM_SCOPE}::{USER_PROFILE_MEMORY_TYPE}"
_default_store: "MemoryStore | None" = None
_default_store_lock = threading.Lock()
_default_async_writer: "AsyncMemoryWriter | None" = None
_default_async_writer_lock = threading.Lock()


class MemoryRecord(TypedDict):
    memory_id: str
    layer: MemoryLayer
    scope: str
    type: str
    subject: str
    content: str
    source_kind: str
    source_ref: str
    created_at: str
    updated_at: str
    confidence: float
    status: MemoryStatus


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_memory_record(
    *,
    layer: MemoryLayer,
    scope: str,
    type: str,
    content: str,
    source_kind: str,
    subject: str = "",
    source_ref: str = "",
    confidence: float = 1.0,
    status: MemoryStatus = "active",
    memory_id: str | None = None,
    created_at: str | None = None,
    updated_at: str | None = None,
) -> MemoryRecord:
    now = utc_now_iso()
    clamped_confidence = max(0.0, min(float(confidence), 1.0))
    return {
        "memory_id": memory_id or str(uuid.uuid4()),
        "layer": layer,
        "scope": scope,
        "type": type,
        "subject": subject,
        "content": content,
        "source_kind": source_kind,
        "source_ref": source_ref,
        "created_at": created_at or now,
        "updated_at": updated_at or now,
        "confidence": clamped_confidence,
        "status": status,
    }


class MemoryStore:
    def __init__(self, db_path: str = MEMORY_DB_PATH):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._revision = 0
        self._fts_available = False
        self.ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def ensure_schema(self) -> None:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memory_records (
                        memory_id TEXT PRIMARY KEY,
                        layer TEXT NOT NULL,
                        scope TEXT NOT NULL,
                        type TEXT NOT NULL,
                        subject TEXT NOT NULL DEFAULT '',
                        content TEXT NOT NULL,
                        source_kind TEXT NOT NULL,
                        source_ref TEXT NOT NULL DEFAULT '',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        status TEXT NOT NULL
                    )
                    """
                )
                self._ensure_column(conn, "memory_records", "subject", "TEXT NOT NULL DEFAULT ''")
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_memory_records_scope_layer_status
                    ON memory_records(scope, layer, status)
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_memory_records_updated_at
                    ON memory_records(updated_at DESC)
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_memory_records_layer_scope_status_updated
                    ON memory_records(layer, scope, status, updated_at DESC, created_at DESC)
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_memory_records_layer_scope_type_status_updated
                    ON memory_records(layer, scope, type, status, updated_at DESC, created_at DESC)
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_memory_records_layer_scope_type_subject_status_updated
                    ON memory_records(layer, scope, type, subject, status, updated_at DESC, created_at DESC)
                    """
                )
                try:
                    self._ensure_fts_schema(conn)
                    self._fts_available = True
                    self._rebuild_fts_index(conn)
                except sqlite3.OperationalError:
                    self._fts_available = False
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

    def _ensure_fts_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_records_fts
            USING fts5(
                memory_id UNINDEXED,
                search_text,
                tokenize = 'unicode61'
            )
            """
        )

    def _rebuild_fts_index(self, conn: sqlite3.Connection) -> None:
        conn.execute("DELETE FROM memory_records_fts")
        rows = conn.execute("SELECT * FROM memory_records").fetchall()
        for row in rows:
            self._sync_fts_index(conn, dict(row))

    def _sync_fts_index(self, conn: sqlite3.Connection, record: MemoryRecord) -> None:
        if not self._fts_available:
            return
        conn.execute(
            "DELETE FROM memory_records_fts WHERE memory_id = ?",
            (record["memory_id"],),
        )
        conn.execute(
            """
            INSERT INTO memory_records_fts (memory_id, search_text)
            VALUES (?, ?)
            """,
            (record["memory_id"], self._build_fts_text(record)),
        )

    def _build_fts_text(self, record: MemoryRecord) -> str:
        type_aliases = {
            "user_preference": "用户 偏好 喜欢 习惯 风格 回答",
            "user_preference_note": "用户 偏好 喜欢 习惯 风格 回答",
            "project_fact": "项目 路径 仓库 目录 代码库",
            "workflow_preference": "流程 工作流 步骤 测试 执行",
            "safety_preference": "安全 确认 审批 高风险 沙盒 权限",
            "user_profile_snapshot": "用户 画像 偏好 档案",
        }
        subject_aliases = {
            "response_language": "语言 中文 英文 回复 回答",
            "answer_style": "风格 简洁 详细 语气 口吻 回答",
            "addressing": "称呼 名字",
            "project_path": "项目 路径 目录 仓库",
            "project_context": "项目 上下文 背景",
            "testing_workflow": "测试 回归 验证",
            "implementation_order": "顺序 步骤 流程 先 再",
            "workflow_policy": "流程 工作流 规则",
            "approval_policy": "确认 审批 高风险",
            "sandbox_policy": "沙盒 权限 越权",
            "safety_policy": "安全 规则",
        }
        text = " ".join(
            str(record.get(field, "") or "")
            for field in ("type", "subject", "content", "source_ref")
        )
        text = " ".join((
            text,
            type_aliases.get(record.get("type", ""), ""),
            subject_aliases.get(record.get("subject", ""), ""),
        ))
        return f"{text} {' '.join(self._search_terms(text))}"

    def _search_terms(self, text: str) -> list[str]:
        terms: list[str] = []
        seen: set[str] = set()

        def add(term: str) -> None:
            normalized = term.strip().lower()
            if len(normalized) < 2 or normalized in seen:
                return
            seen.add(normalized)
            terms.append(normalized)

        for token in re.findall(r"[A-Za-z0-9_]+", text):
            add(token)
        for chunk in re.findall(r"[\u4e00-\u9fff]+", text):
            if len(chunk) == 2:
                add(chunk)
            elif len(chunk) > 2:
                for index in range(len(chunk) - 1):
                    add(chunk[index:index + 2])
        return terms

    def _build_fts_query(self, query: str) -> str:
        terms = self._search_terms(query)
        escaped_terms = [f'"{term.replace(chr(34), "")}"' for term in terms]
        return " OR ".join(escaped_terms)

    def _supersede_conflicting_memories(self, conn: sqlite3.Connection, record: MemoryRecord) -> int:
        subject = (record.get("subject") or "").strip()
        if record["layer"] != "long_term" or record["status"] != "active" or not subject:
            return 0

        cursor = conn.execute(
            """
            UPDATE memory_records
            SET status = 'superseded', updated_at = ?
            WHERE layer = 'long_term'
              AND scope = ?
              AND type = ?
              AND subject = ?
              AND status = 'active'
              AND memory_id != ?
            """,
            (
                utc_now_iso(),
                record["scope"],
                record["type"],
                subject,
                record["memory_id"],
            ),
        )
        return cursor.rowcount or 0

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    def upsert_memory(self, record: MemoryRecord) -> MemoryRecord:
        normalized = build_memory_record(**record)
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO memory_records (
                        memory_id,
                        layer,
                        scope,
                        type,
                        subject,
                        content,
                        source_kind,
                        source_ref,
                        created_at,
                        updated_at,
                        confidence,
                        status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(memory_id) DO UPDATE SET
                        layer = excluded.layer,
                        scope = excluded.scope,
                        type = excluded.type,
                        subject = excluded.subject,
                        content = excluded.content,
                        source_kind = excluded.source_kind,
                        source_ref = excluded.source_ref,
                        created_at = excluded.created_at,
                        updated_at = excluded.updated_at,
                        confidence = excluded.confidence,
                        status = excluded.status
                    """,
                    (
                        normalized["memory_id"],
                        normalized["layer"],
                        normalized["scope"],
                        normalized["type"],
                        normalized["subject"],
                        normalized["content"],
                        normalized["source_kind"],
                        normalized["source_ref"],
                        normalized["created_at"],
                        normalized["updated_at"],
                        normalized["confidence"],
                        normalized["status"],
                    ),
                )
                superseded_count = self._supersede_conflicting_memories(conn, normalized)
                self._sync_fts_index(conn, normalized)
                conn.commit()
                self._revision += 1 + superseded_count
        return normalized

    def get_memory(self, memory_id: str) -> MemoryRecord | None:
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM memory_records WHERE memory_id = ?",
                    (memory_id,),
                ).fetchone()
        return dict(row) if row else None

    def list_memories(
        self,
        *,
        layer: MemoryLayer | None = None,
        scope: str | None = None,
        memory_type: str | None = None,
        status: MemoryStatus | None = "active",
        limit: int = 20,
    ) -> list[MemoryRecord]:
        conditions = []
        params: list[object] = []
        if layer is not None:
            conditions.append("layer = ?")
            params.append(layer)
        if scope is not None:
            conditions.append("scope = ?")
            params.append(scope)
        if memory_type is not None:
            conditions.append("type = ?")
            params.append(memory_type)
        if status is not None:
            conditions.append("status = ?")
            params.append(status)

        where_sql = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)

        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    f"""
                    SELECT *
                    FROM memory_records
                    {where_sql}
                    ORDER BY updated_at DESC, created_at DESC
                    LIMIT ?
                    """,
                    tuple(params),
                ).fetchall()
        return [dict(row) for row in rows]

    def search_memories(
        self,
        query: str,
        *,
        layer: MemoryLayer | None = None,
        scope: str | None = None,
        memory_types: list[str] | tuple[str, ...] | None = None,
        status: MemoryStatus | None = "active",
        limit: int = 5,
    ) -> list[MemoryRecord]:
        fts_query = self._build_fts_query(query)
        if not fts_query:
            return []
        if not self._fts_available:
            return self._fallback_search_memories(
                query,
                layer=layer,
                scope=scope,
                memory_types=memory_types,
                status=status,
                limit=limit,
            )

        conditions = ["memory_records_fts.search_text MATCH ?"]
        params: list[object] = [fts_query]
        if layer is not None:
            conditions.append("m.layer = ?")
            params.append(layer)
        if scope is not None:
            conditions.append("m.scope = ?")
            params.append(scope)
        if memory_types:
            placeholders = ", ".join("?" for _ in memory_types)
            conditions.append(f"m.type IN ({placeholders})")
            params.extend(memory_types)
        if status is not None:
            conditions.append("m.status = ?")
            params.append(status)
        params.append(limit)

        where_sql = " AND ".join(conditions)
        try:
            with self._lock:
                with self._connect() as conn:
                    rows = conn.execute(
                        f"""
                        SELECT m.*, bm25(memory_records_fts) AS rank
                        FROM memory_records_fts
                        JOIN memory_records AS m ON m.memory_id = memory_records_fts.memory_id
                        WHERE {where_sql}
                        ORDER BY rank ASC, m.confidence DESC, m.updated_at DESC, m.created_at DESC
                        LIMIT ?
                        """,
                        tuple(params),
                    ).fetchall()
        except sqlite3.OperationalError:
            return self._fallback_search_memories(
                query,
                layer=layer,
                scope=scope,
                memory_types=memory_types,
                status=status,
                limit=limit,
            )

        return [dict(row) for row in rows]

    def _fallback_search_memories(
        self,
        query: str,
        *,
        layer: MemoryLayer | None,
        scope: str | None,
        memory_types: list[str] | tuple[str, ...] | None,
        status: MemoryStatus | None,
        limit: int,
    ) -> list[MemoryRecord]:
        terms = self._search_terms(query)
        if not terms:
            return []

        conditions = []
        params: list[object] = []
        if layer is not None:
            conditions.append("layer = ?")
            params.append(layer)
        if scope is not None:
            conditions.append("scope = ?")
            params.append(scope)
        if memory_types:
            placeholders = ", ".join("?" for _ in memory_types)
            conditions.append(f"type IN ({placeholders})")
            params.extend(memory_types)
        if status is not None:
            conditions.append("status = ?")
            params.append(status)

        search_conditions = []
        for term in terms:
            search_conditions.append("(content LIKE ? OR subject LIKE ? OR type LIKE ? OR source_ref LIKE ?)")
            pattern = f"%{term}%"
            params.extend([pattern, pattern, pattern, pattern])
        conditions.append(f"({' OR '.join(search_conditions)})")
        params.append(limit)

        where_sql = f"WHERE {' AND '.join(conditions)}"
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    f"""
                    SELECT *
                    FROM memory_records
                    {where_sql}
                    ORDER BY confidence DESC, updated_at DESC, created_at DESC
                    LIMIT ?
                    """,
                    tuple(params),
                ).fetchall()
        return [dict(row) for row in rows]

    def update_memory_status(
        self,
        memory_id: str,
        *,
        status: MemoryStatus,
    ) -> MemoryRecord | None:
        existing = self.get_memory(memory_id)
        if existing is None:
            return None

        updated = build_memory_record(
            memory_id=existing["memory_id"],
            layer=existing["layer"],
            scope=existing["scope"],
            type=existing["type"],
            subject=existing.get("subject", ""),
            content=existing["content"],
            source_kind=existing["source_kind"],
            source_ref=existing["source_ref"],
            created_at=existing["created_at"],
            updated_at=utc_now_iso(),
            confidence=existing["confidence"],
            status=status,
        )
        return self.upsert_memory(updated)


def get_memory_store(db_path: str | None = None) -> MemoryStore:
    global _default_store
    if db_path is not None:
        return MemoryStore(db_path=db_path)

    with _default_store_lock:
        if _default_store is None:
            _default_store = MemoryStore()
        return _default_store


class AsyncMemoryWriter:
    def __init__(self):
        self.queue: queue.Queue[MemoryRecord | None] = queue.Queue()
        self.worker = threading.Thread(target=self._write_loop, daemon=True)
        self.worker.start()

    def _write_loop(self) -> None:
        while True:
            record = self.queue.get()
            if record is None:
                self.queue.task_done()
                break
            try:
                get_memory_store().upsert_memory(record)
            finally:
                self.queue.task_done()

    def submit(self, record: MemoryRecord) -> None:
        self.queue.put(record)

    def flush(self) -> None:
        self.queue.join()


def get_async_memory_writer() -> AsyncMemoryWriter:
    global _default_async_writer
    with _default_async_writer_lock:
        if _default_async_writer is None:
            _default_async_writer = AsyncMemoryWriter()
        return _default_async_writer
