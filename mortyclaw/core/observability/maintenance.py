from __future__ import annotations

import os
import shutil
import sqlite3
import tarfile
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ..code_index import CODE_INDEX_DB_PATH
from ..config import (
    BACKUPS_DIR,
    DB_PATH,
    LOGS_ARCHIVE_DIR,
    LOGS_DIR,
    MEMORY_DB_PATH,
    RUNTIME_DB_PATH,
)

DEFAULT_LOG_RETENTION_DAYS = 14
DEFAULT_LOG_MAX_BYTES = 5 * 1024 * 1024
DEFAULT_INBOX_RETENTION_DAYS = 7
DEFAULT_TASK_RUN_RETENTION_DAYS = 30
DEFAULT_TASK_RUN_KEEP_RECENT = 20
DEFAULT_STATE_KEEP_LATEST = 30


def _utc_now(now: datetime | str | None = None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    if isinstance(now, datetime):
        return now.astimezone(timezone.utc) if now.tzinfo else now.replace(tzinfo=timezone.utc)
    parsed = _parse_timestamp(now)
    if parsed is None:
        raise ValueError(f"unsupported timestamp: {now}")
    return parsed


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S"):
        try:
            parsed = datetime.strptime(value, fmt)
            return parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _timestamp_slug(now: datetime) -> str:
    return now.astimezone(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _safe_file_size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _directory_size(path: str) -> int:
    total = 0
    if not os.path.isdir(path):
        return 0
    for root, _, files in os.walk(path):
        for name in files:
            total += _safe_file_size(os.path.join(root, name))
    return total


def format_bytes(size_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} {unit}"
        value /= 1024
    return f"{size_bytes} B"


def _sqlite_table_counts(db_path: str, table_names: list[str]) -> dict[str, int]:
    counts = {name: 0 for name in table_names}
    if not os.path.exists(db_path):
        return counts

    conn = sqlite3.connect(db_path)
    try:
        existing = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        for table_name in table_names:
            if table_name in existing:
                counts[table_name] = int(
                    conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                )
    finally:
        conn.close()
    return counts


def collect_doctor_report(
    *,
    state_db_path: str = DB_PATH,
    runtime_db_path: str = RUNTIME_DB_PATH,
    memory_db_path: str = MEMORY_DB_PATH,
    code_index_db_path: str = CODE_INDEX_DB_PATH,
    logs_dir: str = LOGS_DIR,
) -> dict[str, Any]:
    log_files = sorted(
        str(path)
        for path in Path(logs_dir).glob("*.jsonl")
        if path.is_file()
    )
    return {
        "databases": {
            "state": {
                "path": state_db_path,
                "exists": os.path.exists(state_db_path),
                "size_bytes": _safe_file_size(state_db_path),
                "table_counts": _sqlite_table_counts(state_db_path, ["checkpoints", "writes"]),
            },
            "runtime": {
                "path": runtime_db_path,
                "exists": os.path.exists(runtime_db_path),
                "size_bytes": _safe_file_size(runtime_db_path),
                "table_counts": _sqlite_table_counts(
                    runtime_db_path,
                    ["sessions", "tasks", "task_runs", "session_inbox"],
                ),
            },
            "memory": {
                "path": memory_db_path,
                "exists": os.path.exists(memory_db_path),
                "size_bytes": _safe_file_size(memory_db_path),
                "table_counts": _sqlite_table_counts(
                    memory_db_path,
                    ["memory_records", "memory_records_fts"],
                ),
            },
            "code_index": {
                "path": code_index_db_path,
                "exists": os.path.exists(code_index_db_path),
                "size_bytes": _safe_file_size(code_index_db_path),
                "table_counts": _sqlite_table_counts(
                    code_index_db_path,
                    ["files", "symbols", "calls", "imports"],
                ),
            },
        },
        "logs": {
            "path": logs_dir,
            "exists": os.path.isdir(logs_dir),
            "size_bytes": _directory_size(logs_dir),
            "file_count": len(log_files),
            "largest_files": [
                {"path": path, "size_bytes": _safe_file_size(path)}
                for path in sorted(log_files, key=_safe_file_size, reverse=True)[:5]
            ],
        },
    }


def gc_logs(
    *,
    log_dir: str = LOGS_DIR,
    archive_dir: str = LOGS_ARCHIVE_DIR,
    max_age_days: int = DEFAULT_LOG_RETENTION_DAYS,
    max_size_bytes: int = DEFAULT_LOG_MAX_BYTES,
    apply: bool = False,
    now: datetime | str | None = None,
) -> dict[str, Any]:
    now_dt = _utc_now(now)
    candidates: list[dict[str, Any]] = []

    for path in sorted(Path(log_dir).glob("*.jsonl")):
        if not path.is_file():
            continue
        stat = path.stat()
        modified_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        reasons: list[str] = []
        if modified_at <= now_dt - timedelta(days=max_age_days):
            reasons.append(f"older_than_{max_age_days}d")
        if stat.st_size >= max_size_bytes:
            reasons.append(f"larger_than_{max_size_bytes}_bytes")
        if reasons:
            candidates.append(
                {
                    "path": str(path),
                    "size_bytes": stat.st_size,
                    "modified_at": modified_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "reasons": reasons,
                }
            )

    report = {
        "dry_run": not apply,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "archived_count": 0,
        "removed_count": 0,
        "archive_path": None,
    }
    if not apply or not candidates:
        return report

    os.makedirs(archive_dir, exist_ok=True)
    archive_path = os.path.join(archive_dir, f"logs_gc_{_timestamp_slug(now_dt)}.tar.gz")
    with tarfile.open(archive_path, "w:gz") as tar:
        for item in candidates:
            tar.add(item["path"], arcname=os.path.basename(item["path"]))

    removed_count = 0
    for item in candidates:
        os.remove(item["path"])
        removed_count += 1

    report["archive_path"] = archive_path
    report["archived_count"] = len(candidates)
    report["removed_count"] = removed_count
    return report


def gc_runtime(
    *,
    runtime_db_path: str = RUNTIME_DB_PATH,
    inbox_retention_days: int = DEFAULT_INBOX_RETENTION_DAYS,
    task_run_retention_days: int = DEFAULT_TASK_RUN_RETENTION_DAYS,
    keep_recent_task_runs: int = DEFAULT_TASK_RUN_KEEP_RECENT,
    apply: bool = False,
    now: datetime | str | None = None,
) -> dict[str, Any]:
    now_dt = _utc_now(now)
    inbox_cutoff = now_dt - timedelta(days=inbox_retention_days)
    task_run_cutoff = now_dt - timedelta(days=task_run_retention_days)
    report = {
        "dry_run": not apply,
        "inbox": {"candidate_count": 0, "deleted_count": 0, "candidates": []},
        "task_runs": {"candidate_count": 0, "deleted_count": 0, "candidates": []},
    }
    if not os.path.exists(runtime_db_path):
        return report

    conn = sqlite3.connect(runtime_db_path)
    conn.row_factory = sqlite3.Row
    try:
        inbox_candidates: list[dict[str, Any]] = []
        for row in conn.execute(
            """
            SELECT event_id, thread_id, event_type, created_at, delivered_at
            FROM session_inbox
            WHERE status = 'delivered'
            ORDER BY delivered_at ASC, created_at ASC
            """
        ):
            delivered_at = _parse_timestamp(row["delivered_at"])
            if delivered_at and delivered_at <= inbox_cutoff:
                inbox_candidates.append(dict(row))

        task_run_candidates: list[dict[str, Any]] = []
        seen_per_task: defaultdict[str, int] = defaultdict(int)
        for row in conn.execute(
            """
            SELECT run_id, task_id, thread_id, status, triggered_at, finished_at
            FROM task_runs
            ORDER BY task_id ASC, triggered_at DESC, finished_at DESC, run_id DESC
            """
        ):
            task_id = row["task_id"]
            seen_per_task[task_id] += 1
            triggered_at = _parse_timestamp(row["triggered_at"])
            if seen_per_task[task_id] > max(0, keep_recent_task_runs) and triggered_at and triggered_at <= task_run_cutoff:
                task_run_candidates.append(dict(row))

        report["inbox"]["candidate_count"] = len(inbox_candidates)
        report["inbox"]["candidates"] = inbox_candidates
        report["task_runs"]["candidate_count"] = len(task_run_candidates)
        report["task_runs"]["candidates"] = task_run_candidates

        if apply:
            if inbox_candidates:
                conn.executemany(
                    "DELETE FROM session_inbox WHERE event_id = ?",
                    [(item["event_id"],) for item in inbox_candidates],
                )
                report["inbox"]["deleted_count"] = len(inbox_candidates)
            if task_run_candidates:
                conn.executemany(
                    "DELETE FROM task_runs WHERE run_id = ?",
                    [(item["run_id"],) for item in task_run_candidates],
                )
                report["task_runs"]["deleted_count"] = len(task_run_candidates)
            conn.commit()
    finally:
        conn.close()

    return report


def _backup_sqlite_database(db_path: str, backup_dir: str, now_dt: datetime) -> str:
    os.makedirs(backup_dir, exist_ok=True)
    backup_path = os.path.join(
        backup_dir,
        f"{Path(db_path).stem}-{_timestamp_slug(now_dt)}.sqlite3",
    )
    src = sqlite3.connect(db_path)
    dest = sqlite3.connect(backup_path)
    try:
        src.backup(dest)
    finally:
        dest.close()
        src.close()
    return backup_path


def gc_state(
    *,
    state_db_path: str = DB_PATH,
    backup_dir: str = BACKUPS_DIR,
    keep_latest_per_thread: int = DEFAULT_STATE_KEEP_LATEST,
    apply: bool = False,
    now: datetime | str | None = None,
) -> dict[str, Any]:
    keep_latest = max(1, int(keep_latest_per_thread or DEFAULT_STATE_KEEP_LATEST))
    report: dict[str, Any] = {
        "dry_run": not apply,
        "keep_latest_per_thread": keep_latest,
        "checkpoint_candidate_count": 0,
        "write_candidate_count": 0,
        "deleted_checkpoints": 0,
        "deleted_writes": 0,
        "backup_path": None,
        "size_before_bytes": _safe_file_size(state_db_path),
        "size_after_bytes": _safe_file_size(state_db_path),
        "threads": {},
    }
    if not os.path.exists(state_db_path):
        return report

    conn = sqlite3.connect(state_db_path)
    conn.row_factory = sqlite3.Row
    try:
        grouped_rows: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
        for row in conn.execute(
            """
            SELECT thread_id, checkpoint_ns, checkpoint_id
            FROM checkpoints
            ORDER BY thread_id ASC, checkpoint_ns ASC, checkpoint_id DESC
            """
        ):
            grouped_rows[(row["thread_id"], row["checkpoint_ns"])].append(row["checkpoint_id"])

        checkpoint_candidates: list[tuple[str, str, str]] = []
        thread_summaries: dict[str, dict[str, int]] = {}
        for (thread_id, checkpoint_ns), checkpoint_ids in grouped_rows.items():
            kept = checkpoint_ids[:keep_latest]
            prunable = checkpoint_ids[keep_latest:]
            if prunable:
                checkpoint_candidates.extend((thread_id, checkpoint_ns, checkpoint_id) for checkpoint_id in prunable)
            key = f"{thread_id}:{checkpoint_ns or '<root>'}"
            thread_summaries[key] = {
                "checkpoint_total": len(checkpoint_ids),
                "kept": len(kept),
                "checkpoint_prunable": len(prunable),
            }

        candidate_set = set(checkpoint_candidates)
        write_candidates: list[tuple[str, str, str]] = []
        for row in conn.execute(
            "SELECT thread_id, checkpoint_ns, checkpoint_id FROM writes"
        ):
            key = (row["thread_id"], row["checkpoint_ns"], row["checkpoint_id"])
            if key in candidate_set:
                write_candidates.append(key)

        for key, summary in thread_summaries.items():
            summary["write_prunable"] = sum(
                1
                for thread_id, checkpoint_ns, checkpoint_id in write_candidates
                if key == f"{thread_id}:{checkpoint_ns or '<root>'}"
            )

        report["threads"] = thread_summaries
        report["checkpoint_candidate_count"] = len(checkpoint_candidates)
        report["write_candidate_count"] = len(write_candidates)

        if apply and checkpoint_candidates:
            now_dt = _utc_now(now)
            report["backup_path"] = _backup_sqlite_database(state_db_path, backup_dir, now_dt)
            conn.executemany(
                "DELETE FROM writes WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?",
                write_candidates,
            )
            conn.executemany(
                "DELETE FROM checkpoints WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?",
                checkpoint_candidates,
            )
            conn.commit()
            conn.execute("VACUUM")
            conn.commit()
            report["deleted_checkpoints"] = len(checkpoint_candidates)
            report["deleted_writes"] = len(write_candidates)
            report["size_after_bytes"] = _safe_file_size(state_db_path)
    finally:
        conn.close()

    return report


def reset_thread_state(
    *,
    thread_id: str,
    state_db_path: str = DB_PATH,
    vacuum: bool = False,
) -> dict[str, Any]:
    normalized_thread_id = (thread_id or "").strip()
    report: dict[str, Any] = {
        "thread_id": normalized_thread_id,
        "checkpoint_count_before": 0,
        "write_count_before": 0,
        "deleted_checkpoints": 0,
        "deleted_writes": 0,
        "checkpoint_count_after": 0,
        "write_count_after": 0,
        "vacuumed": False,
    }
    if not normalized_thread_id or not os.path.exists(state_db_path):
        return report

    conn = sqlite3.connect(state_db_path)
    try:
        checkpoint_count_before = int(
            conn.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                (normalized_thread_id,),
            ).fetchone()[0]
        )
        write_count_before = int(
            conn.execute(
                "SELECT COUNT(*) FROM writes WHERE thread_id = ?",
                (normalized_thread_id,),
            ).fetchone()[0]
        )

        report["checkpoint_count_before"] = checkpoint_count_before
        report["write_count_before"] = write_count_before

        if checkpoint_count_before or write_count_before:
            conn.execute("DELETE FROM writes WHERE thread_id = ?", (normalized_thread_id,))
            conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (normalized_thread_id,))
            conn.commit()

            report["deleted_checkpoints"] = checkpoint_count_before
            report["deleted_writes"] = write_count_before

            if vacuum:
                conn.execute("VACUUM")
                conn.commit()
                report["vacuumed"] = True

        report["checkpoint_count_after"] = int(
            conn.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                (normalized_thread_id,),
            ).fetchone()[0]
        )
        report["write_count_after"] = int(
            conn.execute(
                "SELECT COUNT(*) FROM writes WHERE thread_id = ?",
                (normalized_thread_id,),
            ).fetchone()[0]
        )
    finally:
        conn.close()

    return report
