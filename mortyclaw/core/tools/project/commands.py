from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from ..base import mortyclaw_tool
from .common import (
    _iter_project_files,
    _relative_path,
    _resolve_project_root,
    _tool_payload,
    _truncate,
)


def _validate_safe_test_command(command: str) -> str:
    normalized = re.sub(r"\s+", " ", (command or "").strip())
    if not normalized:
        return normalized

    lowered = normalized.lower()
    blocked_patterns = (
        r"\brm\s",
        r"\bmv\s",
        r"\bsudo\b",
        r"\bgit\s+reset\b",
        r"\bgit\s+checkout\b",
        r"\bgit\s+clean\b",
        r"\bpython\s+-c\b",
        r"\bnode\s+-e\b",
        r">",
        r";",
        r"\|\s*(sh|bash|zsh|fish)\b",
    )
    for pattern in blocked_patterns:
        if re.search(pattern, lowered):
            raise PermissionError(f"run_project_tests 只允许测试/检查命令，检测到危险片段：{pattern}")

    allowed_prefixes = (
        "python -m unittest",
        "python -m pytest",
        "pytest",
        "uv run pytest",
        "uv run python -m unittest",
        "uv run python -m pytest",
        "ruff",
        "uv run ruff",
        "mypy",
        "uv run mypy",
        "tox",
        "make test",
        "npm test",
        "pnpm test",
        "yarn test",
        "./rick/bin/python -m unittest",
        "./.venv/bin/python -m unittest",
        ".venv/bin/python -m unittest",
    )
    if not lowered.startswith(allowed_prefixes):
        raise PermissionError(
            "run_project_tests 仅允许常见测试/静态检查命令。"
            "可用示例：python -m unittest discover -s tests -q、pytest、uv run pytest、ruff check ."
        )
    return normalized


SAFE_PROJECT_COMMAND_PATTERNS = (
    ("python", "-m", "py_compile"),
    ("python", "-m", "unittest"),
    ("python", "-m", "pytest"),
    ("pytest",),
    ("ruff",),
    ("mypy",),
    ("npm", "test"),
    ("npm", "run", "build"),
    ("pnpm", "test"),
    ("yarn", "test"),
    ("uv", "run"),
    ("tox",),
    ("make", "test"),
    ("./rick/bin/python", "-m", "unittest"),
    ("./.venv/bin/python", "-m", "unittest"),
    (".venv/bin/python", "-m", "unittest"),
)


def _looks_like_safe_script_execution(argv: list[str]) -> bool:
    if len(argv) < 2:
        return False

    executable = os.path.basename(str(argv[0]).strip()).lower()
    if executable not in {"python", "python3", "python3.12"}:
        return False

    script_arg = str(argv[1]).strip()
    if not script_arg or script_arg.startswith("-"):
        return False

    if len(argv) > 2 and any(str(item).strip().startswith("-") for item in argv[2:]):
        return False

    if os.path.isabs(script_arg):
        return False

    script_path = Path(script_arg)
    if ".." in script_path.parts:
        return False

    return script_path.suffix.lower() == ".py"


def _validate_safe_project_command(command: str) -> list[str]:
    normalized = str(command or "").strip()
    if not normalized:
        raise PermissionError("run_project_command 需要显式提供命令。")

    if re.search(r"[;&|><`$()]", normalized):
        raise PermissionError("run_project_command 仅允许白名单命令，禁止 shell 拼接、重定向、管道或命令替换。")

    argv = shlex.split(normalized, posix=True)
    if not argv:
        raise PermissionError("run_project_command 未解析到有效命令。")

    lowered = [item.lower() for item in argv]
    normalized_for_match = list(lowered)
    if normalized_for_match:
        normalized_for_match[0] = os.path.basename(normalized_for_match[0])
    if any(token in {"rm", "mv", "sudo", "bash", "sh", "zsh", "fish"} for token in lowered):
        raise PermissionError("run_project_command 禁止原始 shell 或破坏性系统命令。")

    for pattern in SAFE_PROJECT_COMMAND_PATTERNS:
        lowered_pattern = tuple(item.lower() for item in pattern)
        if len(argv) >= len(pattern) and (
            tuple(lowered[: len(pattern)]) == lowered_pattern
            or tuple(normalized_for_match[: len(pattern)]) == lowered_pattern
        ):
            return argv

    if _looks_like_safe_script_execution(argv):
        return argv

    raise PermissionError(
        "run_project_command 仅允许白名单命令。"
        "可用示例：python -m py_compile main.py、python demo.py、pytest、python -m unittest、npm test、npm run build、uv run pytest。"
    )


def _default_test_command(project_root: str) -> str:
    if os.path.isdir(os.path.join(project_root, "tests")):
        return f"{sys.executable} -m unittest discover -s tests -q"
    py_files = [path for path in _iter_project_files(project_root, extensions={".py"}, max_files=200)]
    if py_files:
        rel_paths = " ".join(_relative_path(project_root, path) for path in py_files[:50])
        return f"{sys.executable} -m py_compile {rel_paths}"
    return ""


@mortyclaw_tool
def show_git_diff(project_root: str, pathspec: str = "", max_chars: int = 12000) -> str:
    """
    查看科研/代码项目当前未提交修改的 git diff。

    使用场景：
    - apply_project_patch 后检查实际改动。
    - 代码检查时查看工作区已有变更。
    """
    try:
        root = _resolve_project_root(project_root)
        if shutil.which("git") is None:
            return "show_git_diff 不可用：未找到 git 命令。"
        command = ["git", "-C", root, "diff", "--"]
        if pathspec:
            if os.path.isabs(pathspec) or ".." in Path(pathspec).parts:
                raise PermissionError("pathspec 不能是绝对路径或包含 ..。")
            command.append(pathspec)
        result = subprocess.run(command, capture_output=True, encoding="utf-8", errors="replace", timeout=30)
        if result.returncode != 0:
            return "show_git_diff 失败：\n" + _truncate(result.stderr or result.stdout)
        diff_text = result.stdout or "当前没有未提交 diff。"
        return _truncate(diff_text, max(1000, min(int(max_chars or 12000), 50000)))
    except Exception as exc:
        return f"show_git_diff 失败：{exc}"


@mortyclaw_tool
def run_project_tests(project_root: str, command: str = "", timeout_seconds: int = 180) -> str:
    """
    在科研/代码项目根目录运行测试或静态检查命令。

    如果 command 为空：
    - 有 tests/ 目录时默认运行当前 Python 解释器的 unittest discover。
    - 否则对 Python 文件运行 py_compile smoke check。

    安全边界：
    - 只允许常见测试/静态检查命令。
    - 拒绝 rm、git reset、python -c、重定向等危险片段。
    """
    try:
        root = _resolve_project_root(project_root)
        generated_default = not (command or "").strip()
        selected_command = _default_test_command(root) if generated_default else _validate_safe_test_command(command)
        if not selected_command:
            return _tool_payload(ok=False, error_kind="COMMAND_BLOCKED", message="run_project_tests 未发现可运行的默认测试命令，请显式传入安全测试命令。")
        timeout = max(5, min(int(timeout_seconds or 180), 1800))
        result = subprocess.run(
            selected_command,
            shell=True,
            cwd=root,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        return _tool_payload(
            ok=result.returncode == 0,
            error_kind="" if result.returncode == 0 else "COMMAND_FAILED",
            message="验证通过。" if result.returncode == 0 else "验证失败，请根据输出继续修复。",
            project_root=root,
            command=selected_command,
            exit_code=result.returncode,
            stdout=_truncate(result.stdout.strip(), 12000),
            stderr=_truncate(result.stderr.strip(), 12000),
            used_default_command=generated_default,
        )
    except subprocess.TimeoutExpired:
        return _tool_payload(ok=False, error_kind="COMMAND_FAILED", message="run_project_tests 执行超时，测试命令可能卡住或耗时过长。")
    except Exception as exc:
        kind = "COMMAND_BLOCKED" if isinstance(exc, PermissionError) else "COMMAND_FAILED"
        return _tool_payload(ok=False, error_kind=kind, message=f"run_project_tests 失败：{exc}")


@mortyclaw_tool
def run_project_command(project_root: str, command: str, timeout_seconds: int = 180) -> str:
    """
    在 project_root 内执行白名单验证/构建命令，不开放原始 shell。
    """
    try:
        root = _resolve_project_root(project_root)
        argv = _validate_safe_project_command(command)
        timeout = max(5, min(int(timeout_seconds or 180), 1800))
        result = subprocess.run(
            argv,
            cwd=root,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        return _tool_payload(
            ok=result.returncode == 0,
            error_kind="" if result.returncode == 0 else "COMMAND_FAILED",
            message="命令执行通过。" if result.returncode == 0 else "命令执行失败。",
            project_root=root,
            command=" ".join(argv),
            exit_code=result.returncode,
            stdout=_truncate(result.stdout.strip(), 12000),
            stderr=_truncate(result.stderr.strip(), 12000),
        )
    except subprocess.TimeoutExpired:
        return _tool_payload(ok=False, error_kind="COMMAND_FAILED", message="run_project_command 执行超时。")
    except Exception as exc:
        kind = "COMMAND_BLOCKED" if isinstance(exc, PermissionError) else "COMMAND_FAILED"
        return _tool_payload(ok=False, error_kind=kind, message=f"run_project_command 失败：{exc}")
