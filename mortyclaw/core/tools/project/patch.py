from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from ..base import mortyclaw_tool
from .common import (
    _is_sensitive_path,
    _recovery_snapshot,
    _resolve_project_path,
    _resolve_project_root,
    _tool_payload,
    _truncate,
)


def _extract_patch_paths(patch: str) -> list[str]:
    paths: set[str] = set()
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            for part in parts[2:4]:
                if part.startswith(("a/", "b/")):
                    paths.add(part[2:])
        elif line.startswith(("--- ", "+++ ")):
            raw_path = line[4:].strip().split("\t")[0]
            if raw_path == "/dev/null":
                continue
            if raw_path.startswith(("a/", "b/")):
                raw_path = raw_path[2:]
            paths.add(raw_path)
    return sorted(path for path in paths if path and path != "/dev/null")


def _validate_patch_paths(project_root: str, patch: str) -> list[str]:
    paths = _extract_patch_paths(patch)
    if not paths:
        raise ValueError("patch 中未发现可识别的文件路径。请传入 unified diff/git diff 格式。")
    for path in paths:
        if os.path.isabs(path) or ".." in Path(path).parts:
            raise PermissionError(f"patch 路径越权：{path}")
        _resolve_project_path(project_root, path, allow_missing=True)
        if _is_sensitive_path(path):
            raise PermissionError(f"安全拦截：不允许 patch 敏感文件 {path}")
    return paths


def _run_git_apply(project_root: str, patch: str, *, check_only: bool) -> subprocess.CompletedProcess:
    command = ["git", "apply", "--whitespace=nowarn"]
    if check_only:
        command.append("--check")
    command.append("-")
    return subprocess.run(
        command,
        input=patch,
        cwd=project_root,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )


def _git_diff_stat(project_root: str) -> str:
    result = subprocess.run(
        ["git", "-C", project_root, "diff", "--stat"],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=15,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


@mortyclaw_tool
def apply_project_patch(patch: str, project_root: str = "", reason: str = "", dry_run: bool = False) -> str:
    """
    对科研/代码项目应用 unified diff/git diff 补丁。

    使用要求：
    - patch 必须是 unified diff/git diff 格式。
    - 只允许修改 project_root 内文件。
    - 默认拒绝修改 .env、私钥、证书等敏感文件。
    - 适合替代粗粒度整文件覆盖，进行 patch 级代码修改。

    修改后必须继续调用 show_git_diff 和 run_project_tests 验证，再向用户总结：
    改了哪些文件、为什么改、如何验证。
    """
    try:
        root = _resolve_project_root(project_root)
        if not patch or not patch.strip():
            return _tool_payload(ok=False, error_kind="PATCH_PARSE_FAILED", message="apply_project_patch 参数错误：patch 不能为空。")
        if shutil.which("git") is None:
            return _tool_payload(ok=False, error_kind="PATCH_PARSE_FAILED", message="apply_project_patch 不可用：未找到 git 命令。")
        changed_paths = _validate_patch_paths(root, patch)
        check_result = _run_git_apply(root, patch, check_only=True)
        if check_result.returncode != 0:
            stderr_text = _truncate(check_result.stderr or check_result.stdout)
            error_kind = "PATCH_DOES_NOT_APPLY"
            if "corrupt patch" in stderr_text.lower():
                error_kind = "PATCH_PARSE_FAILED"
            elif "patch does not apply" in stderr_text.lower() or "patch failed" in stderr_text.lower():
                error_kind = "PATCH_CONTEXT_MISMATCH"
            snapshot = {}
            if changed_paths:
                try:
                    snapshot = _recovery_snapshot(root, _resolve_project_path(root, changed_paths[0], allow_missing=False))
                except Exception:
                    snapshot = {}
            return _tool_payload(
                ok=False,
                error_kind=error_kind,
                message="patch 校验失败，请基于最新文件内容改用 edit_project_file 或 write_project_file 重试。",
                stderr=stderr_text,
                changed_paths=changed_paths,
                recovery_hint="请先读取最新文件内容，再优先使用 edit_project_file；若是整文件重写，改用 write_project_file。",
                **snapshot,
            )
        if dry_run:
            return _tool_payload(
                ok=True,
                message="patch dry-run 校验通过，尚未修改文件。",
                dry_run=True,
                changed_paths=changed_paths,
            )

        apply_result = _run_git_apply(root, patch, check_only=False)
        if apply_result.returncode != 0:
            stderr_text = _truncate(apply_result.stderr or apply_result.stdout)
            snapshot = {}
            if changed_paths:
                try:
                    snapshot = _recovery_snapshot(root, _resolve_project_path(root, changed_paths[0], allow_missing=False))
                except Exception:
                    snapshot = {}
            return _tool_payload(
                ok=False,
                error_kind="PATCH_DOES_NOT_APPLY",
                message="patch 应用失败，请重新读取最新文件并改用 edit_project_file 或 write_project_file。",
                stderr=stderr_text,
                changed_paths=changed_paths,
                recovery_hint="不要继续盲修 patch；请先读取最新文件内容，再生成新的 edit/write 操作。",
                **snapshot,
            )

        diff_stat = _git_diff_stat(root)
        return _tool_payload(
            ok=True,
            message="patch 已成功应用。",
            changed_paths=changed_paths,
            reason=reason,
            diff_stat=diff_stat,
            next_actions=[
                "调用 show_git_diff 查看实际 diff。",
                "调用 run_project_tests 或 run_project_command 运行验证。",
            ],
        )
    except Exception as exc:
        return _tool_payload(ok=False, error_kind="PATCH_PARSE_FAILED", message=f"apply_project_patch 失败：{exc}")
