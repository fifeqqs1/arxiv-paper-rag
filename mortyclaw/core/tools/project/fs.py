from __future__ import annotations

import os

from ..base import mortyclaw_tool
from .common import (
    _ensure_expected_version,
    _file_hash,
    _looks_like_sensitive_write_target,
    _recovery_snapshot,
    _relative_path,
    _resolve_project_path,
    _resolve_project_root,
    _safe_read_text,
    _tool_payload,
    _truncate,
    _write_project_text,
)


@mortyclaw_tool
def read_project_file(filepath: str, project_root: str = "", start_line: int = 1, max_lines: int = 240) -> str:
    """
    读取科研/代码项目中的文件片段，返回带行号的内容。

    使用场景：
    - 用户明确给出 project_root 或当前会话已经记住项目路径。
    - 需要检查代码、配置、README、实验脚本、测试文件。

    安全边界：
    - 只能读取 project_root 内的文件。
    - 默认拒绝读取 .env、私钥、证书等敏感文件。
    """
    try:
        root = _resolve_project_root(project_root)
        target = _resolve_project_path(root, filepath)
        start = max(1, int(start_line or 1))
        limit = max(1, min(int(max_lines or 240), 1000))
        text = _safe_read_text(target)
    except Exception as exc:
        return f"read_project_file 失败：{exc}"

    lines = text.splitlines()
    start_index = min(start - 1, len(lines))
    end_index = min(len(lines), start_index + limit)
    body = [
        f"{line_number:>5}: {lines[line_number - 1]}"
        for line_number in range(start_index + 1, end_index + 1)
    ]
    header = (
        f"文件：{_relative_path(root, target)}\n"
        f"行号范围：{start_index + 1}-{end_index} / total={len(lines)}\n"
    )
    return _truncate(header + "\n".join(body))


@mortyclaw_tool
def write_project_file(
    path: str,
    content: str,
    project_root: str = "",
    expected_hash: str = "",
    create_if_missing: bool = False,
) -> str:
    """
    在 project_root 内整文件写入或新建文件。
    适合大改文件、整文件替换，以及 patch 失败后的稳定兜底。
    """
    try:
        root = _resolve_project_root(project_root)
        target = _resolve_project_path(root, path, allow_missing=bool(create_if_missing), allow_sensitive=False)
        if _looks_like_sensitive_write_target(target):
            return _tool_payload(ok=False, error_kind="OUT_OF_SCOPE_PATH", message="安全拦截：不允许写入敏感文件。")
        exists = os.path.exists(target)
        if not exists and not create_if_missing:
            return _tool_payload(ok=False, error_kind="FILE_NOT_FOUND", message=f"文件不存在：{path}")
        current_text = _safe_read_text(target) if exists else ""
        matches, current_hash = _ensure_expected_version(
            current_text=current_text,
            expected_hash=expected_hash,
        )
        if not matches:
            snapshot = _recovery_snapshot(root, target) if exists else {
                "path": _relative_path(root, target),
                "current_hash": current_hash,
                "current_excerpt": "",
            }
            return _tool_payload(
                ok=False,
                error_kind="FILE_CHANGED_SINCE_READ",
                message="目标文件内容已经变化，请基于最新内容重试。",
                **snapshot,
            )
        written = _write_project_text(target, str(content))
        return _tool_payload(
            ok=True,
            message="项目文件已写入。",
            path=_relative_path(root, target),
            bytes_written=written,
            new_hash=_file_hash(target),
            created=bool(not exists),
        )
    except FileNotFoundError as exc:
        return _tool_payload(ok=False, error_kind="FILE_NOT_FOUND", message=str(exc))
    except PermissionError as exc:
        return _tool_payload(ok=False, error_kind="OUT_OF_SCOPE_PATH", message=str(exc))
    except Exception as exc:
        return _tool_payload(ok=False, error_kind="OUT_OF_SCOPE_PATH", message=f"write_project_file 失败：{exc}")


@mortyclaw_tool
def edit_project_file(
    path: str,
    edits: list[dict],
    project_root: str = "",
    expected_hash: str = "",
) -> str:
    """
    在 project_root 内做局部文本编辑。
    每个 edit 支持：
    - {"old_text": "...", "new_text": "..."}
    - {"start_line": 1, "end_line": 3, "new_text": "..."}
    """
    try:
        root = _resolve_project_root(project_root)
        target = _resolve_project_path(root, path, allow_missing=False, allow_sensitive=False)
        current_text = _safe_read_text(target)
        matches, _current_hash = _ensure_expected_version(current_text=current_text, expected_hash=expected_hash)
        if not matches:
            return _tool_payload(
                ok=False,
                error_kind="FILE_CHANGED_SINCE_READ",
                message="目标文件内容已经变化，请先重新读取最新内容。",
                **_recovery_snapshot(root, target),
            )
        if not edits:
            return _tool_payload(ok=False, error_kind="OLD_TEXT_NOT_FOUND", message="edit_project_file 需要至少一个 edit。")

        updated_text = current_text
        for edit in edits:
            if not isinstance(edit, dict):
                return _tool_payload(ok=False, error_kind="OLD_TEXT_NOT_FOUND", message="edit_project_file 的 edits 只能包含对象。")
            old_text = edit.get("old_text")
            new_text = str(edit.get("new_text", ""))
            start_line = edit.get("start_line")
            end_line = edit.get("end_line")
            if old_text not in (None, ""):
                old_text = str(old_text)
                occurrences = updated_text.count(old_text)
                if occurrences == 0:
                    return _tool_payload(
                        ok=False,
                        error_kind="OLD_TEXT_NOT_FOUND",
                        message="未找到要替换的 old_text，请先读取最新文件内容。",
                        **_recovery_snapshot(root, target),
                    )
                if occurrences > 1:
                    return _tool_payload(
                        ok=False,
                        error_kind="OLD_TEXT_AMBIGUOUS",
                        message="old_text 命中多处，无法安全唯一替换。请改用更精确的 old_text 或行范围替换。",
                        **_recovery_snapshot(root, target),
                    )
                updated_text = updated_text.replace(old_text, new_text, 1)
                continue

            if start_line is None or end_line is None:
                return _tool_payload(ok=False, error_kind="OLD_TEXT_NOT_FOUND", message="每个 edit 必须提供 old_text，或同时提供 start_line/end_line。")
            lines = updated_text.splitlines(keepends=True)
            start = int(start_line)
            end = int(end_line)
            if start < 1 or end < start or end > len(lines):
                return _tool_payload(
                    ok=False,
                    error_kind="OLD_TEXT_NOT_FOUND",
                    message="行范围无效，请先读取最新文件确认行号。",
                    **_recovery_snapshot(root, target),
                )
            replacement = new_text
            if replacement and not replacement.endswith("\n") and any(line.endswith("\n") for line in lines[start - 1:end]):
                replacement += "\n"
            lines[start - 1:end] = [replacement]
            updated_text = "".join(lines)

        written = _write_project_text(target, updated_text)
        return _tool_payload(
            ok=True,
            message="项目文件局部编辑已完成。",
            path=_relative_path(root, target),
            bytes_written=written,
            new_hash=_file_hash(target),
            edit_count=len(edits),
        )
    except FileNotFoundError as exc:
        return _tool_payload(ok=False, error_kind="FILE_NOT_FOUND", message=str(exc))
    except PermissionError as exc:
        return _tool_payload(ok=False, error_kind="OUT_OF_SCOPE_PATH", message=str(exc))
    except Exception as exc:
        return _tool_payload(ok=False, error_kind="OLD_TEXT_NOT_FOUND", message=f"edit_project_file 失败：{exc}")
