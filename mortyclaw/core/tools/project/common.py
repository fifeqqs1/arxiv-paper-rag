from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from ...memory import get_memory_store
from ...runtime_context import get_active_thread_id


EXCLUDED_DIRS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "logs",
    "node_modules",
    "rick",
    "workspace",
}

TEXT_EXTENSIONS = {
    ".c",
    ".cc",
    ".cfg",
    ".cpp",
    ".cs",
    ".css",
    ".go",
    ".h",
    ".hpp",
    ".html",
    ".ini",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".md",
    ".php",
    ".py",
    ".rb",
    ".rs",
    ".sh",
    ".sql",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}

SENSITIVE_FILENAMES = {
    ".env",
    ".npmrc",
    ".pypirc",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
}

SENSITIVE_EXTENSIONS = {".key", ".pem", ".p12", ".pfx"}
PROJECT_ROOT_MARKER_FILES = {
    ".git",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "requirements.txt",
    "package.json",
    "cargo.toml",
    "go.mod",
    "pom.xml",
    "manage.py",
    "readme.md",
    "readme-zh.md",
    "readme-ja.md",
}
PROJECT_ROOT_MARKER_DIRS = {
    "src",
    "tests",
    "docs",
    "agents",
    "web",
    "app",
    "pkg",
}
MAX_READ_BYTES = 2 * 1024 * 1024
MAX_OUTPUT_CHARS = 12000
MAX_RECOVERY_PREVIEW_LINES = 80
JSON_ERROR_KINDS = {
    "OUT_OF_SCOPE_PATH",
    "FILE_NOT_FOUND",
    "FILE_CHANGED_SINCE_READ",
    "OLD_TEXT_NOT_FOUND",
    "OLD_TEXT_AMBIGUOUS",
    "PATCH_PARSE_FAILED",
    "PATCH_CONTEXT_MISMATCH",
    "PATCH_DOES_NOT_APPLY",
    "COMMAND_BLOCKED",
    "COMMAND_FAILED",
}


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n\n...[输出过长，已安全截断]..."


def _tool_payload(*, ok: bool, message: str, error_kind: str = "", **extra) -> str:
    payload = {
        "ok": bool(ok),
        "message": str(message or "").strip(),
        "error_kind": error_kind if error_kind in JSON_ERROR_KINDS else "",
    }
    payload.update({key: value for key, value in extra.items() if value not in ("", None, [], {})})
    return json.dumps(payload, ensure_ascii=False)


def _is_within_dir(target_path: str, base_dir: str) -> bool:
    try:
        target_real = os.path.realpath(target_path)
        base_real = os.path.realpath(base_dir)
        return os.path.commonpath([target_real, base_real]) == base_real
    except ValueError:
        return False


def _is_sensitive_path(path_value: str) -> bool:
    basename = os.path.basename(path_value).lower()
    if basename in SENSITIVE_FILENAMES:
        return True
    _, ext = os.path.splitext(basename)
    return ext.lower() in SENSITIVE_EXTENSIONS


def _session_project_root() -> str:
    thread_id = get_active_thread_id(default="system_default")
    try:
        records = get_memory_store().list_memories(
            layer="session",
            scope=thread_id,
            memory_type="project_path",
            limit=1,
        )
    except Exception:
        return ""
    if not records:
        return ""
    return str(records[0].get("content") or "").strip()


def _scan_project_root_entries(path_value: str) -> tuple[list[os.DirEntry[str]], list[os.DirEntry[str]]]:
    files: list[os.DirEntry[str]] = []
    dirs: list[os.DirEntry[str]] = []
    try:
        with os.scandir(path_value) as entries:
            for entry in entries:
                name = entry.name
                if name in EXCLUDED_DIRS:
                    continue
                if name.startswith(".") and name != ".git":
                    continue
                if entry.is_dir():
                    dirs.append(entry)
                else:
                    files.append(entry)
    except OSError:
        return [], []
    return files, dirs


def _project_root_signal_count(path_value: str) -> int:
    files, dirs = _scan_project_root_entries(path_value)
    score = 0

    file_names = {entry.name.lower() for entry in files}
    dir_names = {entry.name.lower() for entry in dirs}

    score += sum(1 for name in file_names if name in PROJECT_ROOT_MARKER_FILES)
    score += sum(1 for name in dir_names if name in PROJECT_ROOT_MARKER_DIRS)

    if any(name.endswith(".py") for name in file_names):
        score += 1

    return score


def _normalize_nested_project_root(path_value: str) -> str:
    files, dirs = _scan_project_root_entries(path_value)
    if files:
        return path_value
    if len(dirs) != 1:
        return path_value

    child_path = dirs[0].path
    if _project_root_signal_count(path_value) > 0:
        return path_value

    child_signal_count = _project_root_signal_count(child_path)
    if child_signal_count < 2:
        return path_value

    child_files, child_dirs = _scan_project_root_entries(child_path)
    if not child_files and not child_dirs:
        return path_value
    return child_path


def _resolve_project_root(project_root: str = "") -> str:
    candidate = (project_root or "").strip() or _session_project_root()
    if not candidate:
        raise ValueError("请提供 project_root，或先在当前会话中明确项目绝对路径。")

    resolved = os.path.realpath(os.path.expanduser(candidate))
    if os.path.isfile(resolved):
        resolved = os.path.dirname(resolved)
    if not os.path.isdir(resolved):
        raise FileNotFoundError(f"项目路径不存在或不是目录：{candidate}")
    return _normalize_nested_project_root(resolved)


def _resolve_project_path(
    project_root: str,
    filepath: str,
    *,
    allow_missing: bool = False,
    allow_sensitive: bool = False,
) -> str:
    if not filepath or not str(filepath).strip():
        raise ValueError("filepath 不能为空。")

    normalized = os.path.expanduser(str(filepath).strip())
    if os.path.isabs(normalized):
        target = os.path.realpath(normalized)
    else:
        target = os.path.realpath(os.path.join(project_root, normalized))

    if not _is_within_dir(target, project_root):
        raise PermissionError("越权拦截：项目工具只能访问 project_root 内的路径。")
    if not allow_sensitive and _is_sensitive_path(target):
        raise PermissionError("安全拦截：项目工具默认不读取或修改密钥/环境变量类敏感文件。")
    if not allow_missing and not os.path.exists(target):
        raise FileNotFoundError(f"文件不存在：{filepath}")
    return target


def _relative_path(project_root: str, path_value: str) -> str:
    return os.path.relpath(path_value, project_root).replace(os.sep, "/")


def _iter_project_files(
    project_root: str,
    *,
    extensions: set[str] | None = None,
    max_files: int = 3000,
):
    count = 0
    for current_root, dirs, files in os.walk(project_root):
        dirs[:] = [item for item in dirs if item not in EXCLUDED_DIRS and not item.startswith(".cache")]
        for filename in files:
            path = os.path.join(current_root, filename)
            if _is_sensitive_path(path):
                continue
            if extensions is not None and Path(filename).suffix.lower() not in extensions:
                continue
            count += 1
            if count > max_files:
                return
            yield path


def _safe_read_text(path_value: str) -> str:
    if os.path.getsize(path_value) > MAX_READ_BYTES:
        raise ValueError("文件过大，超过 2MB 读取限制。")
    with open(path_value, "r", encoding="utf-8", errors="replace") as handle:
        return handle.read()


def _file_hash_from_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_hash(path_value: str) -> str:
    return _file_hash_from_text(_safe_read_text(path_value))


def _recovery_snapshot(project_root: str, path_value: str) -> dict[str, str]:
    rel_path = _relative_path(project_root, path_value)
    text = _safe_read_text(path_value)
    lines = text.splitlines()
    preview = "\n".join(
        f"{line_number:>5}: {lines[line_number - 1]}"
        for line_number in range(1, min(len(lines), MAX_RECOVERY_PREVIEW_LINES) + 1)
    )
    return {
        "path": rel_path,
        "current_hash": _file_hash_from_text(text),
        "current_excerpt": _truncate(preview, 4000),
    }


def _ensure_expected_version(
    *,
    current_text: str,
    expected_hash: str = "",
) -> tuple[bool, str]:
    if expected_hash:
        current_hash = _file_hash_from_text(current_text)
        if current_hash != str(expected_hash).strip():
            return False, current_hash
        return True, current_hash
    return True, _file_hash_from_text(current_text)


def _looks_like_sensitive_write_target(path_value: str) -> bool:
    return _is_sensitive_path(path_value)


def _write_project_text(path_value: str, content: str) -> int:
    os.makedirs(os.path.dirname(path_value), exist_ok=True)
    with open(path_value, "w", encoding="utf-8", errors="replace") as handle:
        written = handle.write(content)
    return written
