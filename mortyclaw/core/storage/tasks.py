from __future__ import annotations

import json
import os
import sqlite3
import uuid

from ..config import TASKS_FILE
from .common import TaskRecord, coerce_positive_int, compute_next_run, local_now_str, utc_now_iso
from .store import RuntimeStore


class TaskRepository:
    def __init__(self, store: RuntimeStore):
        self.store = store

    def _task_from_row(self, row: sqlite3.Row | None) -> TaskRecord | None:
        return dict(row) if row is not None else None

    def _fetchone(self, query: str, params: tuple = ()) -> TaskRecord | None:
        with self.store._lock:
            with self.store._connect() as conn:
                row = conn.execute(query, params).fetchone()
        return self._task_from_row(row)

    def get_task(self, task_id: str) -> TaskRecord | None:
        return self._fetchone("SELECT * FROM tasks WHERE task_id = ?", (task_id,))

    def list_tasks(
        self,
        *,
        thread_id: str | None = None,
        statuses: tuple[str, ...] = ("scheduled",),
        limit: int = 200,
    ) -> list[TaskRecord]:
        conditions = []
        params: list[object] = []

        if thread_id is not None:
            conditions.append("thread_id = ?")
            params.append(thread_id)
        if statuses:
            placeholders = ", ".join("?" for _ in statuses)
            conditions.append(f"status IN ({placeholders})")
            params.extend(statuses)

        where_sql = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)

        with self.store._lock:
            with self.store._connect() as conn:
                rows = conn.execute(
                    f"""
                    SELECT *
                    FROM tasks
                    {where_sql}
                    ORDER BY target_time ASC, created_at ASC
                    LIMIT ?
                    """,
                    tuple(params),
                ).fetchall()
        return [dict(row) for row in rows]

    def create_task(
        self,
        *,
        target_time: str,
        description: str,
        repeat: str | None,
        repeat_count: int | None,
        thread_id: str,
        task_id: str | None = None,
    ) -> TaskRecord:
        now_utc = utc_now_iso()
        repeat_count = coerce_positive_int(repeat_count)
        remaining_runs = repeat_count if repeat else None
        record: TaskRecord = {
            "task_id": task_id or str(uuid.uuid4())[:8],
            "thread_id": (thread_id or "system_default").strip() or "system_default",
            "description": description,
            "target_time": target_time,
            "repeat": repeat or None,
            "repeat_count": repeat_count,
            "remaining_runs": remaining_runs,
            "status": "scheduled",
            "created_at": now_utc,
            "updated_at": now_utc,
            "last_run_at": None,
        }
        with self.store._lock:
            with self.store._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO tasks (
                        task_id, thread_id, description, target_time, repeat, repeat_count,
                        remaining_runs, status, created_at, updated_at, last_run_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record["task_id"],
                        record["thread_id"],
                        record["description"],
                        record["target_time"],
                        record["repeat"],
                        record["repeat_count"],
                        record["remaining_runs"],
                        record["status"],
                        record["created_at"],
                        record["updated_at"],
                        record["last_run_at"],
                    ),
                )
                conn.commit()
        self.sync_legacy_file()
        return record

    def update_task(
        self,
        task_id: str,
        *,
        thread_id: str,
        new_time: str | None = None,
        new_description: str | None = None,
    ) -> TaskRecord | None:
        existing = self.get_task(task_id)
        if existing is None or existing["thread_id"] != thread_id or existing["status"] != "scheduled":
            return None

        updated = dict(existing)
        if new_time:
            updated["target_time"] = new_time
        if new_description:
            updated["description"] = new_description
        updated["updated_at"] = utc_now_iso()

        with self.store._lock:
            with self.store._connect() as conn:
                conn.execute(
                    """
                    UPDATE tasks
                    SET description = ?, target_time = ?, updated_at = ?
                    WHERE task_id = ?
                    """,
                    (
                        updated["description"],
                        updated["target_time"],
                        updated["updated_at"],
                        task_id,
                    ),
                )
                conn.commit()
        self.sync_legacy_file()
        return updated

    def cancel_task(self, task_id: str, *, thread_id: str) -> TaskRecord | None:
        existing = self.get_task(task_id)
        if existing is None or existing["thread_id"] != thread_id or existing["status"] != "scheduled":
            return None

        updated_at = utc_now_iso()
        with self.store._lock:
            with self.store._connect() as conn:
                conn.execute(
                    """
                    UPDATE tasks
                    SET status = 'cancelled', updated_at = ?
                    WHERE task_id = ?
                    """,
                    (updated_at, task_id),
                )
                conn.commit()
        self.sync_legacy_file()
        return self.get_task(task_id)

    def list_due_tasks(
        self,
        *,
        now: str | None = None,
        limit: int = 100,
    ) -> list[TaskRecord]:
        due_time = now or local_now_str()
        with self.store._lock:
            with self.store._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT *
                    FROM tasks
                    WHERE status = 'scheduled' AND target_time <= ?
                    ORDER BY target_time ASC, created_at ASC
                    LIMIT ?
                    """,
                    (due_time, limit),
                ).fetchall()
        return [dict(row) for row in rows]

    def record_task_run(
        self,
        *,
        task_id: str,
        thread_id: str,
        status: str,
        triggered_at: str | None = None,
        finished_at: str | None = None,
        result_summary: str = "",
        error_message: str = "",
    ) -> None:
        started = triggered_at or utc_now_iso()
        finished = finished_at or started
        with self.store._lock:
            with self.store._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO task_runs (
                        run_id, task_id, thread_id, status, triggered_at, finished_at,
                        result_summary, error_message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        task_id,
                        thread_id,
                        status,
                        started,
                        finished,
                        result_summary,
                        error_message,
                    ),
                )
                conn.commit()

    def advance_after_dispatch(self, task_id: str, *, dispatched_at: str | None = None) -> TaskRecord | None:
        existing = self.get_task(task_id)
        if existing is None or existing["status"] != "scheduled":
            return existing

        repeat_mode = existing.get("repeat")
        remaining_runs = existing.get("remaining_runs")
        last_run_at = dispatched_at or utc_now_iso()
        next_target = None
        next_status = "completed"
        next_remaining_runs = remaining_runs

        if repeat_mode:
            next_target = compute_next_run(existing["target_time"], repeat_mode)
            if remaining_runs is None:
                next_status = "scheduled"
            elif remaining_runs > 1:
                next_status = "scheduled"
                next_remaining_runs = remaining_runs - 1
            else:
                next_status = "completed"
                next_remaining_runs = 0

        updated_at = utc_now_iso()
        with self.store._lock:
            with self.store._connect() as conn:
                conn.execute(
                    """
                    UPDATE tasks
                    SET target_time = ?, remaining_runs = ?, status = ?, updated_at = ?, last_run_at = ?
                    WHERE task_id = ?
                    """,
                    (
                        next_target or existing["target_time"],
                        next_remaining_runs,
                        next_status,
                        updated_at,
                        last_run_at,
                        task_id,
                    ),
                )
                conn.commit()
        self.sync_legacy_file()
        return self.get_task(task_id)

    def mark_task_failed(self, task_id: str, *, error_message: str = "") -> TaskRecord | None:
        existing = self.get_task(task_id)
        if existing is None:
            return None

        with self.store._lock:
            with self.store._connect() as conn:
                conn.execute(
                    """
                    UPDATE tasks
                    SET status = 'failed', updated_at = ?, last_run_at = ?
                    WHERE task_id = ?
                    """,
                    (utc_now_iso(), utc_now_iso(), task_id),
                )
                conn.commit()
        self.record_task_run(
            task_id=task_id,
            thread_id=existing["thread_id"],
            status="failed",
            error_message=error_message,
        )
        self.sync_legacy_file()
        return self.get_task(task_id)

    def sync_legacy_file(self, file_path: str = TASKS_FILE) -> None:
        active_tasks = self.list_tasks(statuses=("scheduled",), limit=1000)
        legacy_payload = [
            {
                "id": task["task_id"],
                "thread_id": task["thread_id"],
                "target_time": task["target_time"],
                "description": task["description"],
                "repeat": task["repeat"],
                "repeat_count": task["repeat_count"],
            }
            for task in active_tasks
        ]
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(legacy_payload, f, ensure_ascii=False, indent=2)

    def bootstrap_legacy_tasks(self, file_path: str = TASKS_FILE, *, default_thread_id: str = "legacy_default") -> dict:
        with self.store._lock:
            with self.store._connect() as conn:
                row = conn.execute("SELECT COUNT(*) AS task_count FROM tasks").fetchone()
        if row and row["task_count"]:
            return {"imported": 0, "skipped": row["task_count"]}
        return self.import_legacy_tasks(file_path=file_path, default_thread_id=default_thread_id, overwrite=False)

    def import_legacy_tasks(
        self,
        *,
        file_path: str = TASKS_FILE,
        default_thread_id: str = "legacy_default",
        overwrite: bool = False,
    ) -> dict:
        if not os.path.exists(file_path):
            return {"imported": 0, "skipped": 0}

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
        except OSError:
            return {"imported": 0, "skipped": 0}

        if not content:
            return {"imported": 0, "skipped": 0}

        payload = json.loads(content)
        if not isinstance(payload, list):
            return {"imported": 0, "skipped": 0}

        imported = 0
        skipped = 0
        for item in payload:
            if not isinstance(item, dict):
                skipped += 1
                continue

            task_id = str(item.get("id") or str(uuid.uuid4())[:8])
            if not overwrite and self.get_task(task_id) is not None:
                skipped += 1
                continue

            target_time = item.get("target_time")
            description = item.get("description", "")
            if not isinstance(target_time, str) or not isinstance(description, str):
                skipped += 1
                continue

            repeat = item.get("repeat") or None
            repeat_count = coerce_positive_int(item.get("repeat_count"))
            thread_id = (item.get("thread_id") or default_thread_id).strip() or default_thread_id

            now_utc = utc_now_iso()
            with self.store._lock:
                with self.store._connect() as conn:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO tasks (
                            task_id, thread_id, description, target_time, repeat, repeat_count,
                            remaining_runs, status, created_at, updated_at, last_run_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            task_id,
                            thread_id,
                            description,
                            target_time,
                            repeat,
                            repeat_count,
                            repeat_count if repeat else None,
                            "scheduled",
                            now_utc,
                            now_utc,
                            None,
                        ),
                    )
                    conn.commit()
            imported += 1

        self.sync_legacy_file(file_path=file_path)
        return {"imported": imported, "skipped": skipped}
