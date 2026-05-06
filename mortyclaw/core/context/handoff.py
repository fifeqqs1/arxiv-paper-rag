from __future__ import annotations

import json
import re
from typing import Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage


HANDOFF_SUMMARY_VERSION = 1
MAX_STEP_ITEMS = 6
MAX_FILE_ITEMS = 8
MAX_COMMAND_ITEMS = 6
MAX_TOOL_RESULT_ITEMS = 6
MAX_NOTE_ITEMS = 6
MAX_RISK_ITEMS = 6
MAX_OPEN_QUESTION_ITEMS = 4
MAX_EVENT_ITEMS = 32
MAX_PATH_ITEMS = 6


class HandoffActiveTask(TypedDict, total=False):
    route: str
    goal: str
    run_status: str
    current_step_index: int
    total_steps: int
    current_step: str
    current_project_path: str
    pending_approval: bool
    approval_reason: str
    last_error: str


class HandoffStep(TypedDict, total=False):
    step: int
    description: str
    status: str
    risk_level: str
    result_summary: str


class HandoffFileArtifact(TypedDict, total=False):
    path: str
    reason: str
    last_observation: str


class HandoffCommandResult(TypedDict, total=False):
    tool_name: str
    command: str
    status: str
    result_summary: str


class HandoffToolResult(TypedDict, total=False):
    tool_name: str
    args_summary: str
    result_summary: str
    related_path: str


class HandoffTodoItem(TypedDict, total=False):
    id: str
    content: str
    status: str


class HandoffSummary(TypedDict, total=False):
    version: int
    goal: str
    active_task: HandoffActiveTask
    completed_steps: list[HandoffStep]
    pending_steps: list[HandoffStep]
    files_touched: list[HandoffFileArtifact]
    commands_run: list[HandoffCommandResult]
    tool_results: list[HandoffToolResult]
    todos: list[HandoffTodoItem]
    context_notes: list[str]
    open_questions: list[str]
    risks: list[str]
    last_user_intent: str


def _truncate_text(value: Any, limit: int = 220) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "..."


def _compact_dict(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if item not in ("", [], {}, None)
    }


def _safe_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _extract_json_object(text: str | None) -> dict[str, Any] | None:
    payload = (text or "").strip()
    if not payload:
        return None
    try:
        data = json.loads(payload)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass

    first = payload.find("{")
    last = payload.rfind("}")
    if first == -1 or last == -1 or last <= first:
        return None
    try:
        data = json.loads(payload[first : last + 1])
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def _dedupe_strings(items: list[str], *, limit: int) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = _truncate_text(item, 260)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
        if len(unique) >= limit:
            break
    return unique


def _dedupe_dicts(items: list[dict[str, Any]], *, limit: int, keys: tuple[str, ...]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        compact = _compact_dict(item)
        if not compact:
            continue
        identity = "|".join(str(compact.get(key, "")) for key in keys)
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(compact)
        if len(unique) >= limit:
            break
    return unique


def _normalize_step(item: dict[str, Any]) -> HandoffStep:
    return _compact_dict({
        "step": int(item.get("step", 0)) if str(item.get("step", "")).isdigit() else item.get("step"),
        "description": _truncate_text(item.get("description", ""), 220),
        "status": _truncate_text(item.get("status", ""), 40),
        "risk_level": _truncate_text(item.get("risk_level", ""), 40),
        "result_summary": _truncate_text(item.get("result_summary", ""), 260),
    })


def _normalize_file(item: dict[str, Any]) -> HandoffFileArtifact:
    return _compact_dict({
        "path": _truncate_text(item.get("path", ""), 220),
        "reason": _truncate_text(item.get("reason", ""), 80),
        "last_observation": _truncate_text(item.get("last_observation", ""), 220),
    })


def _normalize_command(item: dict[str, Any]) -> HandoffCommandResult:
    return _compact_dict({
        "tool_name": _truncate_text(item.get("tool_name", ""), 80),
        "command": _truncate_text(item.get("command", ""), 220),
        "status": _truncate_text(item.get("status", ""), 40),
        "result_summary": _truncate_text(item.get("result_summary", ""), 220),
    })


def _normalize_tool_result(item: dict[str, Any]) -> HandoffToolResult:
    return _compact_dict({
        "tool_name": _truncate_text(item.get("tool_name", ""), 80),
        "args_summary": _truncate_text(item.get("args_summary", ""), 220),
        "result_summary": _truncate_text(item.get("result_summary", ""), 220),
        "related_path": _truncate_text(item.get("related_path", ""), 220),
    })


def _normalize_active_task(item: dict[str, Any]) -> HandoffActiveTask:
    return _compact_dict({
        "route": _truncate_text(item.get("route", ""), 40),
        "goal": _truncate_text(item.get("goal", ""), 260),
        "run_status": _truncate_text(item.get("run_status", ""), 40),
        "current_step_index": int(item.get("current_step_index", 0) or 0),
        "total_steps": int(item.get("total_steps", 0) or 0),
        "current_step": _truncate_text(item.get("current_step", ""), 220),
        "current_project_path": _truncate_text(item.get("current_project_path", ""), 220),
        "pending_approval": bool(item.get("pending_approval", False)),
        "approval_reason": _truncate_text(item.get("approval_reason", ""), 220),
        "last_error": _truncate_text(item.get("last_error", ""), 220),
    })


def _normalize_todo_item(item: dict[str, Any]) -> HandoffTodoItem:
    return _compact_dict({
        "id": _truncate_text(item.get("id", ""), 60),
        "content": _truncate_text(item.get("content", ""), 220),
        "status": _truncate_text(item.get("status", ""), 40),
    })


def normalize_handoff_summary(data: dict[str, Any]) -> HandoffSummary:
    active_task = _normalize_active_task(data.get("active_task") or {})
    goal = _truncate_text(data.get("goal", "") or active_task.get("goal", ""), 260)

    completed_steps = _dedupe_dicts(
        [_normalize_step(item) for item in data.get("completed_steps", []) if isinstance(item, dict)],
        limit=MAX_STEP_ITEMS,
        keys=("step", "description", "result_summary"),
    )
    pending_steps = _dedupe_dicts(
        [_normalize_step(item) for item in data.get("pending_steps", []) if isinstance(item, dict)],
        limit=MAX_STEP_ITEMS,
        keys=("step", "description", "status"),
    )
    files_touched = _dedupe_dicts(
        [_normalize_file(item) for item in data.get("files_touched", []) if isinstance(item, dict)],
        limit=MAX_FILE_ITEMS,
        keys=("path", "reason"),
    )
    commands_run = _dedupe_dicts(
        [_normalize_command(item) for item in data.get("commands_run", []) if isinstance(item, dict)],
        limit=MAX_COMMAND_ITEMS,
        keys=("tool_name", "command", "result_summary"),
    )
    tool_results = _dedupe_dicts(
        [_normalize_tool_result(item) for item in data.get("tool_results", []) if isinstance(item, dict)],
        limit=MAX_TOOL_RESULT_ITEMS,
        keys=("tool_name", "args_summary", "result_summary"),
    )
    todos = _dedupe_dicts(
        [_normalize_todo_item(item) for item in data.get("todos", []) if isinstance(item, dict)],
        limit=MAX_STEP_ITEMS,
        keys=("id", "content", "status"),
    )

    return _compact_dict({
        "version": HANDOFF_SUMMARY_VERSION,
        "goal": goal,
        "active_task": active_task,
        "completed_steps": completed_steps,
        "pending_steps": pending_steps,
        "files_touched": files_touched,
        "commands_run": commands_run,
        "tool_results": tool_results,
        "todos": todos,
        "context_notes": _dedupe_strings([str(item) for item in data.get("context_notes", [])], limit=MAX_NOTE_ITEMS),
        "open_questions": _dedupe_strings([str(item) for item in data.get("open_questions", [])], limit=MAX_OPEN_QUESTION_ITEMS),
        "risks": _dedupe_strings([str(item) for item in data.get("risks", [])], limit=MAX_RISK_ITEMS),
        "last_user_intent": _truncate_text(data.get("last_user_intent", ""), 220),
    })


def parse_handoff_summary(summary_text: str | None) -> HandoffSummary | None:
    data = _extract_json_object(summary_text)
    if not isinstance(data, dict):
        return None
    return normalize_handoff_summary(data)


def _normalize_tool_calls(message: AIMessage) -> list[dict[str, Any]]:
    raw_tool_calls = getattr(message, "tool_calls", None) or getattr(message, "additional_kwargs", {}).get("tool_calls") or []
    normalized: list[dict[str, Any]] = []
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
                "name": function.get("name") or "",
                "args": args if isinstance(args, dict) else {"raw": args},
            })
        else:
            args = tool_call.get("args") or {}
            if not isinstance(args, dict):
                args = {"raw": args}
            normalized.append({
                "id": tool_call.get("id") or tool_call.get("tool_call_id") or f"tool-call-{index}",
                "name": tool_call.get("name") or "",
                "args": args,
            })
    return normalized


def _extract_paths_from_value(value: Any) -> list[str]:
    matches: list[str] = []
    if isinstance(value, str):
        if "/" in value or "\\" in value or re.search(r"\.[A-Za-z0-9]{1,8}$", value):
            matches.append(_truncate_text(value, 220))
        return matches
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in ("path", "file", "root", "dir")):
                matches.extend(_extract_paths_from_value(item))
            elif isinstance(item, (dict, list)):
                matches.extend(_extract_paths_from_value(item))
        return matches
    if isinstance(value, list):
        for item in value:
            matches.extend(_extract_paths_from_value(item))
    return matches


def _extract_command_from_args(args: dict[str, Any]) -> str:
    for key in ("command", "cmd", "shell_command", "test_command"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return _truncate_text(value, 220)
    return ""


def _extract_primary_path(args: dict[str, Any]) -> str:
    candidates = _extract_paths_from_value(args)
    return candidates[0] if candidates else ""


def _message_preview(message: BaseMessage, limit: int = 220) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return _truncate_text(content, limit)
    return _truncate_text(json.dumps(content, ensure_ascii=False), limit)


def _message_text(message: BaseMessage) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def _dedupe_ordered_strings(items: list[str], *, limit: int = MAX_PATH_ITEMS) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = _truncate_text(item, 220).strip().strip("`")
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
        if len(ordered) >= limit:
            break
    return ordered


def _summarize_path_list(paths: list[str], *, limit: int = 3) -> str:
    compact_paths = _dedupe_ordered_strings(paths, limit=limit)
    return ", ".join(compact_paths)


def _extract_paths_from_text(text: str, *, limit: int = MAX_PATH_ITEMS) -> list[str]:
    candidates = re.findall(r"(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+", text or "")
    filtered: list[str] = []
    for candidate in candidates:
        normalized = candidate.strip().strip("`'\"()[]{}:,")
        if not normalized:
            continue
        if normalized.startswith(("http://", "https://")):
            continue
        filtered.append(normalized)
    return _dedupe_ordered_strings(filtered, limit=limit)


def _extract_patch_paths_from_patch_text(patch_text: str) -> list[str]:
    explicit = re.findall(r"^\+\+\+ b/(.+)$", patch_text or "", flags=re.MULTILINE)
    if explicit:
        return _dedupe_ordered_strings(explicit, limit=MAX_PATH_ITEMS)

    diff_pairs = re.findall(r"^diff --git a/(.+?) b/(.+)$", patch_text or "", flags=re.MULTILINE)
    if diff_pairs:
        return _dedupe_ordered_strings([pair[1] for pair in diff_pairs], limit=MAX_PATH_ITEMS)
    return []


def _extract_diff_paths_from_output(text: str) -> list[str]:
    diff_pairs = re.findall(r"^diff --git a/(.+?) b/(.+)$", text or "", flags=re.MULTILINE)
    if diff_pairs:
        return _dedupe_ordered_strings([pair[1] for pair in diff_pairs], limit=MAX_PATH_ITEMS)
    return _extract_paths_from_text(text, limit=MAX_PATH_ITEMS)


def _first_relevant_line(text: str, *, prefer_keywords: tuple[str, ...] = ()) -> str:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return ""

    skip_prefixes = (
        "测试/检查执行完成",
        "项目：",
        "命令：",
        "退出码：",
        "说明：",
        "验证结论：",
        "涉及文件：",
        "修改原因：",
        "当前 git diff --stat：",
        "后续必须执行：",
        "[STDOUT]",
        "[STDERR]",
        "文件：",
        "行号范围：",
    )
    preferred = [line for line in lines if any(keyword.lower() in line.lower() for keyword in prefer_keywords)]
    if preferred:
        return _truncate_text(preferred[0], 220)

    for line in lines:
        if any(line.startswith(prefix) for prefix in skip_prefixes):
            continue
        return _truncate_text(line, 220)
    return _truncate_text(lines[0], 220)


def _extract_read_project_file_call(args: dict[str, Any]) -> dict[str, Any]:
    path = _truncate_text(args.get("filepath", "") or _extract_primary_path(args), 220)
    start_line = int(args.get("start_line", 1) or 1)
    max_lines = int(args.get("max_lines", 0) or 0)
    line_range = f"{start_line}-{start_line + max_lines - 1}" if max_lines > 0 else f"{start_line}+"
    args_summary = f"filepath={path}; lines={line_range}" if path else f"lines={line_range}"
    note = f"准备读取文件 {path} 第 {line_range} 行" if path else "准备读取文件片段"
    return _compact_dict({
        "path": path,
        "paths": [path] if path else [],
        "args_summary": _truncate_text(args_summary, 220),
        "note": _truncate_text(note, 220),
    })


def _extract_search_project_code_call(args: dict[str, Any]) -> dict[str, Any]:
    query = _truncate_text(args.get("query", ""), 120)
    mode = _truncate_text(args.get("mode", "text"), 40)
    file_glob = _truncate_text(args.get("file_glob", ""), 120)
    parts = [f"mode={mode}"]
    if query:
        parts.append(f"query={query}")
    if file_glob:
        parts.append(f"file_glob={file_glob}")
    note = f"准备执行代码搜索：{mode}" + (f" / {query}" if query else "")
    return _compact_dict({
        "args_summary": _truncate_text("; ".join(parts), 220),
        "note": _truncate_text(note, 220),
    })


def _extract_apply_project_patch_call(args: dict[str, Any]) -> dict[str, Any]:
    paths = _extract_patch_paths_from_patch_text(str(args.get("patch", "") or ""))
    reason = _truncate_text(args.get("reason", ""), 120)
    dry_run = bool(args.get("dry_run", False))
    path_summary = _summarize_path_list(paths)
    parts = []
    if path_summary:
        parts.append(f"files={path_summary}")
    if reason:
        parts.append(f"reason={reason}")
    if dry_run:
        parts.append("dry_run=true")
    note = "准备应用补丁"
    if path_summary:
        note += f"：{path_summary}"
    if reason:
        note += f"；原因：{reason}"
    return _compact_dict({
        "path": paths[0] if paths else "",
        "paths": paths,
        "args_summary": _truncate_text("; ".join(parts) or "apply patch", 220),
        "note": _truncate_text(note, 220),
    })


def _extract_edit_project_file_call(args: dict[str, Any]) -> dict[str, Any]:
    path = _truncate_text(args.get("path", "") or "", 220)
    edit_count = len(args.get("edits", []) or [])
    return _compact_dict({
        "path": path,
        "paths": [path] if path else [],
        "args_summary": _truncate_text(f"path={path}; edits={edit_count}" if path else f"edits={edit_count}", 220),
        "note": _truncate_text(f"准备局部编辑 {path}" if path else "准备局部编辑项目文件", 220),
    })


def _extract_write_project_file_call(args: dict[str, Any]) -> dict[str, Any]:
    path = _truncate_text(args.get("path", "") or "", 220)
    return _compact_dict({
        "path": path,
        "paths": [path] if path else [],
        "args_summary": _truncate_text(f"path={path}; write full file" if path else "write project file", 220),
        "note": _truncate_text(f"准备整文件写入 {path}" if path else "准备整文件写入项目文件", 220),
    })


def _extract_show_git_diff_call(args: dict[str, Any]) -> dict[str, Any]:
    pathspec = _truncate_text(args.get("pathspec", "") or "", 220)
    note = f"准备查看 git diff：{pathspec}" if pathspec else "准备查看当前 git diff"
    return _compact_dict({
        "path": pathspec,
        "paths": [pathspec] if pathspec else [],
        "args_summary": _truncate_text(f"pathspec={pathspec}" if pathspec else "git diff", 220),
        "note": _truncate_text(note, 220),
    })


def _extract_run_project_tests_call(args: dict[str, Any]) -> dict[str, Any]:
    command = _truncate_text(args.get("command", "") or "", 220)
    timeout_seconds = args.get("timeout_seconds")
    parts = []
    if command:
        parts.append(f"command={command}")
    else:
        parts.append("command=<default>")
    if timeout_seconds not in ("", None):
        parts.append(f"timeout={timeout_seconds}s")
    note = f"准备运行测试命令：{command}" if command else "准备运行默认项目验证命令"
    return _compact_dict({
        "command": command,
        "args_summary": _truncate_text("; ".join(parts), 220),
        "note": _truncate_text(note, 220),
    })


def _extract_run_project_command_call(args: dict[str, Any]) -> dict[str, Any]:
    command = _truncate_text(args.get("command", "") or "", 220)
    return _compact_dict({
        "command": command,
        "args_summary": _truncate_text(f"command={command}" if command else "project command", 220),
        "note": _truncate_text(f"准备运行项目命令：{command}" if command else "准备运行项目命令", 220),
    })


def _summarize_tool_call(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    normalized_name = (tool_name or "").strip()
    if normalized_name == "read_project_file":
        return _extract_read_project_file_call(args)
    if normalized_name == "search_project_code":
        return _extract_search_project_code_call(args)
    if normalized_name == "apply_project_patch":
        return _extract_apply_project_patch_call(args)
    if normalized_name == "edit_project_file":
        return _extract_edit_project_file_call(args)
    if normalized_name == "write_project_file":
        return _extract_write_project_file_call(args)
    if normalized_name == "show_git_diff":
        return _extract_show_git_diff_call(args)
    if normalized_name == "run_project_tests":
        return _extract_run_project_tests_call(args)
    if normalized_name == "run_project_command":
        return _extract_run_project_command_call(args)
    return _compact_dict({
        "command": _extract_command_from_args(args),
        "path": _extract_primary_path(args),
        "paths": [_extract_primary_path(args)] if _extract_primary_path(args) else [],
        "args_summary": _truncate_text(_safe_json_dumps(args), 220),
    })


def _extract_read_project_file_result(args: dict[str, Any], result_text: str) -> dict[str, Any]:
    path_match = re.search(r"^文件：(.+)$", result_text or "", flags=re.MULTILINE)
    line_match = re.search(r"^行号范围：(.+)$", result_text or "", flags=re.MULTILINE)
    path = _truncate_text(path_match.group(1), 220) if path_match else _truncate_text(args.get("filepath", "") or _extract_primary_path(args), 220)
    line_range = _truncate_text(line_match.group(1), 120) if line_match else ""
    symbol = _first_relevant_line(result_text, prefer_keywords=("def ", "class "))
    summary_parts = []
    if path:
        summary_parts.append(f"读取 {path}")
    else:
        summary_parts.append("已读取文件片段")
    if line_range:
        summary_parts.append(line_range)
    if symbol:
        summary_parts.append(symbol)
    summary = " | ".join(summary_parts)
    note = f"查看了 {path}" if path else "查看了一个文件片段"
    if line_range:
        note += f"（{line_range}）"
    return _compact_dict({
        "path": path,
        "paths": [path] if path else [],
        "result_status": "read",
        "result_summary": _truncate_text(summary, 220),
        "note": _truncate_text(note, 220),
    })


def _extract_search_project_code_result(args: dict[str, Any], result_text: str) -> dict[str, Any]:
    mode = _truncate_text(args.get("mode", "text"), 40)
    query = _truncate_text(args.get("query", ""), 120)
    paths = _extract_paths_from_text(result_text)
    if "未找到" in (result_text or ""):
        summary = f"{mode} 搜索未命中" + (f"：{query}" if query else "")
        note = summary
        status = "miss"
    else:
        path_summary = _summarize_path_list(paths)
        head_line = _first_relevant_line(result_text, prefer_keywords=("[function]", "[class]", "scope=", "训练/推理", "__main__"))
        summary = f"{mode} 搜索命中 {path_summary}" if path_summary else f"{mode} 搜索返回结果"
        if query:
            summary += f"（{query}）"
        if head_line and head_line not in summary:
            summary += f" | {head_line}"
        note = f"代码搜索 {mode}" + (f"：{query}" if query else "")
        status = "hit" if paths or result_text.strip() else "empty"
    return _compact_dict({
        "path": paths[0] if paths else "",
        "paths": paths,
        "result_status": status,
        "result_summary": _truncate_text(summary, 220),
        "note": _truncate_text(note, 220),
    })


def _extract_apply_project_patch_result(args: dict[str, Any], result_text: str) -> dict[str, Any]:
    payload = _extract_json_object(result_text)
    if payload:
        paths = _dedupe_ordered_strings(
            [str(item) for item in (payload.get("changed_paths") or []) if str(item).strip()],
            limit=MAX_PATH_ITEMS,
        )
        status = "applied" if payload.get("ok") else "failed"
        summary = str(payload.get("message", "") or "补丁处理完成")
        return _compact_dict({
            "path": paths[0] if paths else "",
            "paths": paths,
            "result_status": status,
            "result_summary": _truncate_text(summary, 220),
            "note": _truncate_text(summary, 220),
            "risk": _truncate_text(summary, 220) if not payload.get("ok") else "",
        })
    output_paths = re.findall(r"^- (.+)$", result_text or "", flags=re.MULTILINE)
    paths = _dedupe_ordered_strings(output_paths + _extract_patch_paths_from_patch_text(str(args.get("patch", "") or "")), limit=MAX_PATH_ITEMS)
    reason = _truncate_text(args.get("reason", ""), 120)
    lower_text = (result_text or "").lower()
    if "dry-run" in lower_text:
        status = "dry_run"
    elif "已成功应用" in result_text:
        status = "applied"
    elif "失败" in result_text or "error" in lower_text:
        status = "failed"
    else:
        status = "done"
    path_summary = _summarize_path_list(paths)
    summary = "补丁处理完成"
    if status == "applied":
        summary = f"补丁已应用到 {path_summary}" if path_summary else "补丁已成功应用"
    elif status == "dry_run":
        summary = f"补丁 dry-run 校验通过：{path_summary}" if path_summary else "补丁 dry-run 校验通过"
    elif status == "failed":
        detail = _first_relevant_line(result_text, prefer_keywords=("失败", "error", "校验"))
        summary = f"补丁处理失败：{detail}" if detail else "补丁处理失败"
    if reason and status in {"applied", "dry_run"}:
        summary += f"；原因：{reason}"
    note = f"补丁涉及 {path_summary}" if path_summary else "执行了补丁修改"
    if reason:
        note += f"；原因：{reason}"
    return _compact_dict({
        "path": paths[0] if paths else "",
        "paths": paths,
        "result_status": status,
        "result_summary": _truncate_text(summary, 220),
        "note": _truncate_text(note, 220),
        "risk": _truncate_text(summary, 220) if status == "failed" else "",
    })


def _extract_show_git_diff_result(args: dict[str, Any], result_text: str) -> dict[str, Any]:
    pathspec = _truncate_text(args.get("pathspec", "") or "", 220)
    if "当前没有未提交 diff" in (result_text or ""):
        summary = f"git diff 为空" + (f"（{pathspec}）" if pathspec else "")
        return _compact_dict({
            "path": pathspec,
            "paths": [pathspec] if pathspec else [],
            "result_status": "clean",
            "result_summary": _truncate_text(summary, 220),
            "note": _truncate_text(summary, 220),
        })

    paths = _extract_diff_paths_from_output(result_text)
    if pathspec and pathspec not in paths:
        paths = _dedupe_ordered_strings([pathspec] + paths, limit=MAX_PATH_ITEMS)
    path_summary = _summarize_path_list(paths)
    summary = f"git diff 涉及 {path_summary}" if path_summary else "已查看 git diff"
    return _compact_dict({
        "path": paths[0] if paths else pathspec,
        "paths": paths or ([pathspec] if pathspec else []),
        "result_status": "diff",
        "result_summary": _truncate_text(summary, 220),
        "note": _truncate_text(summary, 220),
    })


def _extract_test_output_excerpt(result_text: str) -> str:
    lines = [line.strip() for line in (result_text or "").splitlines() if line.strip()]
    priority_keywords = ("assert", "failed", "error", "exception", "traceback", "失败", "报错")
    for line in lines:
        lowered = line.lower()
        if any(keyword in lowered for keyword in priority_keywords):
            return _truncate_text(line, 220)
    return _first_relevant_line(result_text)


def _extract_run_project_tests_result(args: dict[str, Any], result_text: str) -> dict[str, Any]:
    payload = _extract_json_object(result_text)
    if payload:
        command = _truncate_text(payload.get("command", "") or args.get("command", "") or "", 220)
        status = "passed" if payload.get("ok") else "failed"
        summary = str(payload.get("message", "") or "测试命令已执行")
        return _compact_dict({
            "command": command,
            "result_status": status,
            "result_summary": _truncate_text(summary, 220),
            "note": _truncate_text(summary, 220),
            "risk": _truncate_text(summary, 220) if not payload.get("ok") else "",
        })
    command_match = re.search(r"^命令：(.+)$", result_text or "", flags=re.MULTILINE)
    exit_match = re.search(r"^退出码：(\d+)$", result_text or "", flags=re.MULTILINE)
    command = _truncate_text(command_match.group(1), 220) if command_match else _truncate_text(args.get("command", "") or "", 220)
    exit_code = exit_match.group(1) if exit_match else ""
    lower_text = (result_text or "").lower()
    if "执行超时" in result_text:
        status = "timeout"
    elif "验证结论：通过" in result_text or exit_code == "0":
        status = "passed"
    elif "验证结论：失败" in result_text or (exit_code and exit_code != "0"):
        status = "failed"
    else:
        status = "finished"

    parts = []
    if command:
        parts.append(command)
    if exit_code:
        parts.append(f"exit={exit_code}")
    parts.append({
        "passed": "通过",
        "failed": "失败",
        "timeout": "超时",
        "finished": "完成",
    }.get(status, status))
    excerpt = _extract_test_output_excerpt(result_text)
    if excerpt and excerpt not in parts:
        parts.append(excerpt)
    summary = " | ".join(parts)
    note = f"测试命令结果：{summary}" if summary else "测试命令已执行"
    return _compact_dict({
        "command": command,
        "result_status": status,
        "result_summary": _truncate_text(summary, 220),
        "note": _truncate_text(note, 220),
        "risk": _truncate_text(summary, 220) if status in {"failed", "timeout"} else "",
    })


def _summarize_tool_result(tool_name: str, args: dict[str, Any], result_text: str) -> dict[str, Any]:
    normalized_name = (tool_name or "").strip()
    if normalized_name == "read_project_file":
        return _extract_read_project_file_result(args, result_text)
    if normalized_name == "search_project_code":
        return _extract_search_project_code_result(args, result_text)
    if normalized_name == "apply_project_patch":
        return _extract_apply_project_patch_result(args, result_text)
    if normalized_name in {"edit_project_file", "write_project_file"}:
        return _compact_dict({
            "path": _extract_primary_path(args),
            "paths": [_extract_primary_path(args)] if _extract_primary_path(args) else [],
            "result_status": "done" if "\"ok\": true" in (result_text or "").lower() else "failed",
            "result_summary": _truncate_text(_extract_json_object(result_text).get("message", "") if _extract_json_object(result_text) else result_text, 220),
            "risk": _truncate_text(result_text, 220) if "\"ok\": false" in (result_text or "").lower() else "",
        })
    if normalized_name == "show_git_diff":
        return _extract_show_git_diff_result(args, result_text)
    if normalized_name == "run_project_tests":
        return _extract_run_project_tests_result(args, result_text)
    if normalized_name == "run_project_command":
        return _extract_run_project_tests_result(args, result_text)
    preview = _truncate_text(result_text, 260)
    return _compact_dict({
        "path": _extract_primary_path(args),
        "paths": [_extract_primary_path(args)] if _extract_primary_path(args) else [],
        "command": _extract_command_from_args(args),
        "result_status": "failed" if any(token in preview.lower() for token in ("error", "failed", "exception", "traceback", "失败", "报错")) else "done",
        "result_summary": preview,
        "risk": preview if any(token in preview.lower() for token in ("error", "failed", "exception", "traceback", "失败", "报错")) else "",
    })


def build_discarded_context_payload(discarded_msgs: list[BaseMessage]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    tool_call_registry: dict[str, dict[str, Any]] = {}

    for message in discarded_msgs[-MAX_EVENT_ITEMS:]:
        if isinstance(message, HumanMessage):
            events.append({
                "role": "user",
                "content_preview": _message_preview(message),
            })
            continue

        if isinstance(message, AIMessage):
            tool_calls = _normalize_tool_calls(message)
            if _message_preview(message):
                events.append({
                    "role": "assistant",
                    "content_preview": _message_preview(message),
                })
            for tool_call in tool_calls:
                tool_call_registry[str(tool_call.get("id") or "")] = tool_call
                args = tool_call.get("args") or {}
                tool_name = _truncate_text(tool_call.get("name", ""), 80)
                call_summary = _summarize_tool_call(tool_name, args)
                events.append(_compact_dict({
                    "role": "assistant_tool_call",
                    "tool_call_id": tool_call.get("id"),
                    "tool_name": tool_name,
                    **call_summary,
                }))
            continue

        if isinstance(message, ToolMessage):
            tool_call = tool_call_registry.get(str(getattr(message, "tool_call_id", "") or ""), {})
            args = tool_call.get("args") or {}
            tool_name = _truncate_text(getattr(message, "name", "") or tool_call.get("name", ""), 80)
            call_summary = _summarize_tool_call(tool_name, args)
            result_text = _message_text(message)
            result_summary = _summarize_tool_result(tool_name, args, result_text)
            events.append(_compact_dict({
                "role": "tool",
                "tool_call_id": getattr(message, "tool_call_id", ""),
                "tool_name": tool_name,
                **call_summary,
                **result_summary,
                "result_preview": _message_preview(message, 260),
            }))
            continue

        events.append({
            "role": getattr(message, "type", "unknown"),
            "content_preview": _message_preview(message),
        })

    return events


def build_state_snapshot(state: dict[str, Any] | None) -> dict[str, Any]:
    state = state or {}
    plan = list(state.get("plan", []) or [])
    current_step_index = int(state.get("current_step_index", 0) or 0)
    current_step = plan[current_step_index] if 0 <= current_step_index < len(plan) else {}
    return _compact_dict({
        "goal": _truncate_text(state.get("goal", ""), 260),
        "route": _truncate_text(state.get("route", ""), 40),
        "run_status": _truncate_text(state.get("run_status", ""), 40),
        "current_step_index": current_step_index,
        "current_step": _truncate_text(current_step.get("description", "") if isinstance(current_step, dict) else "", 220),
        "total_steps": len(plan),
        "pending_approval": bool(state.get("pending_approval", False)),
        "approval_reason": _truncate_text(state.get("approval_reason", ""), 220),
        "last_error": _truncate_text(state.get("last_error", ""), 220),
        "current_project_path": _truncate_text(state.get("current_project_path", ""), 220),
        "plan": [
            _compact_dict({
                "step": item.get("step"),
                "description": _truncate_text(item.get("description", ""), 220),
                "status": _truncate_text(item.get("status", ""), 40),
                "risk_level": _truncate_text(item.get("risk_level", ""), 40),
            })
            for item in plan[:MAX_STEP_ITEMS]
            if isinstance(item, dict)
        ],
        "step_results": [
            _compact_dict({
                "step": item.get("step"),
                "description": _truncate_text(item.get("description", ""), 220),
                "result_summary": _truncate_text(item.get("result_summary", ""), 220),
                "status": _truncate_text(item.get("status", "completed"), 40),
            })
            for item in (state.get("step_results", []) or [])[-MAX_STEP_ITEMS:]
            if isinstance(item, dict)
        ],
        "todos": [
            _normalize_todo_item(item)
            for item in (state.get("todos", []) or [])[:MAX_STEP_ITEMS]
            if isinstance(item, dict)
        ],
    })


def _extract_message_signals(discarded_msgs: list[BaseMessage]) -> dict[str, Any]:
    payload = build_discarded_context_payload(discarded_msgs)
    files_touched: list[dict[str, Any]] = []
    commands_run: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []
    context_notes: list[str] = []
    last_user_intent = ""
    risks: list[str] = []
    open_questions: list[str] = []
    completed_tool_calls = {
        str(event.get("tool_call_id", "")).strip()
        for event in payload
        if event.get("role") == "tool" and str(event.get("tool_call_id", "")).strip()
    }

    for event in payload:
        role = event.get("role", "")
        if role == "user":
            preview = str(event.get("content_preview", "")).strip()
            if preview:
                last_user_intent = preview
                context_notes.append(f"用户提到：{preview}")
                if preview.endswith(("?", "？")):
                    open_questions.append(preview)
        elif role == "assistant":
            preview = str(event.get("content_preview", "")).strip()
            if preview:
                context_notes.append(f"助手结论：{preview}")
        elif role == "assistant_tool_call":
            tool_call_id = str(event.get("tool_call_id", "")).strip()
            note = str(event.get("note", "")).strip()
            if note and tool_call_id not in completed_tool_calls:
                context_notes.append(note)
        elif role == "tool":
            paths = [
                str(path).strip()
                for path in (event.get("paths") or [])
                if str(path).strip()
            ]
            if not paths and str(event.get("path", "")).strip():
                paths = [str(event.get("path", "")).strip()]
            observation = str(event.get("result_summary", "") or event.get("result_preview", "")).strip()
            for path in paths:
                files_touched.append({
                    "path": path,
                    "reason": str(event.get("tool_name", "") or "tool"),
                    "last_observation": observation,
                })
            command = str(event.get("command", "")).strip()
            if command:
                commands_run.append({
                    "tool_name": str(event.get("tool_name", "")),
                    "command": command,
                    "status": str(event.get("result_status", "") or "finished"),
                    "result_summary": observation,
                })
            tool_results.append({
                "tool_name": str(event.get("tool_name", "")),
                "args_summary": str(event.get("args_summary", "")),
                "result_summary": observation,
                "related_path": paths[0] if paths else "",
            })
            note = str(event.get("note", "")).strip()
            if note:
                context_notes.append(note)
            risk = str(event.get("risk", "")).strip()
            preview = str(event.get("result_preview", "")).lower()
            if risk:
                risks.append(risk)
            elif any(token in preview for token in ("error", "failed", "exception", "traceback", "失败", "报错")):
                risks.append(str(event.get("result_preview", "")))

    return _compact_dict({
        "files_touched": files_touched,
        "commands_run": commands_run,
        "tool_results": tool_results,
        "context_notes": context_notes,
        "last_user_intent": last_user_intent,
        "risks": risks,
        "open_questions": open_questions,
    })


def _extract_state_signals(state: dict[str, Any] | None) -> dict[str, Any]:
    state = state or {}
    plan = [item for item in (state.get("plan", []) or []) if isinstance(item, dict)]
    current_step_index = int(state.get("current_step_index", 0) or 0)
    current_step = plan[current_step_index] if 0 <= current_step_index < len(plan) else {}
    completed_steps = [
        _compact_dict({
            "step": item.get("step"),
            "description": item.get("description", ""),
            "status": item.get("status", "completed"),
            "risk_level": item.get("risk_level", ""),
            "result_summary": item.get("result_summary", ""),
        })
        for item in (state.get("step_results", []) or [])[-MAX_STEP_ITEMS:]
        if isinstance(item, dict)
    ]
    pending_steps = [
        _compact_dict({
            "step": item.get("step"),
            "description": item.get("description", ""),
            "status": item.get("status", "pending"),
            "risk_level": item.get("risk_level", ""),
        })
        for item in plan[current_step_index: current_step_index + MAX_STEP_ITEMS]
    ]
    files_touched: list[dict[str, Any]] = []
    current_project_path = str(state.get("current_project_path", "") or "").strip()
    if current_project_path:
        files_touched.append({
            "path": current_project_path,
            "reason": "current_project_path",
            "last_observation": "当前会话正在处理的项目根目录",
        })
    risks: list[str] = []
    if state.get("pending_approval") and state.get("approval_reason"):
        risks.append(str(state.get("approval_reason", "")))
    if state.get("last_error"):
        risks.append(str(state.get("last_error", "")))
    return _compact_dict({
        "goal": str(state.get("goal", "") or ""),
        "active_task": {
            "route": state.get("route", ""),
            "goal": state.get("goal", ""),
            "run_status": state.get("run_status", ""),
            "current_step_index": current_step_index,
            "total_steps": len(plan),
            "current_step": current_step.get("description", "") if isinstance(current_step, dict) else "",
            "current_project_path": current_project_path,
            "pending_approval": state.get("pending_approval", False),
            "approval_reason": state.get("approval_reason", ""),
            "last_error": state.get("last_error", ""),
        },
        "completed_steps": completed_steps,
        "pending_steps": pending_steps,
        "todos": [
            _normalize_todo_item(item)
            for item in (state.get("todos", []) or [])[:MAX_STEP_ITEMS]
            if isinstance(item, dict)
        ],
        "files_touched": files_touched,
        "risks": risks,
    })


def _merge_handoff_parts(previous: dict[str, Any], extracted: dict[str, Any], llm_data: dict[str, Any] | None) -> HandoffSummary:
    llm_data = llm_data or {}
    merged = normalize_handoff_summary({
        "goal": llm_data.get("goal") or extracted.get("goal") or previous.get("goal", ""),
        "active_task": {
            **(previous.get("active_task") or {}),
            **(extracted.get("active_task") or {}),
            **(llm_data.get("active_task") or {}),
        },
        "completed_steps": (llm_data.get("completed_steps") or []) + (extracted.get("completed_steps") or []) + (previous.get("completed_steps") or []),
        "pending_steps": (llm_data.get("pending_steps") or []) or (extracted.get("pending_steps") or []) or (previous.get("pending_steps") or []),
        "files_touched": (llm_data.get("files_touched") or []) + (extracted.get("files_touched") or []) + (previous.get("files_touched") or []),
        "commands_run": (llm_data.get("commands_run") or []) + (extracted.get("commands_run") or []) + (previous.get("commands_run") or []),
        "tool_results": (llm_data.get("tool_results") or []) + (extracted.get("tool_results") or []) + (previous.get("tool_results") or []),
        "todos": (llm_data.get("todos") or []) or (extracted.get("todos") or []) or (previous.get("todos") or []),
        "context_notes": (llm_data.get("context_notes") or []) + (extracted.get("context_notes") or []) + (previous.get("context_notes") or []),
        "open_questions": (llm_data.get("open_questions") or []) + (extracted.get("open_questions") or []) + (previous.get("open_questions") or []),
        "risks": (llm_data.get("risks") or []) + (extracted.get("risks") or []) + (previous.get("risks") or []),
        "last_user_intent": llm_data.get("last_user_intent") or extracted.get("last_user_intent") or previous.get("last_user_intent", ""),
    })
    return merged


def build_fallback_handoff_summary(
    current_summary: str,
    discarded_msgs: list[BaseMessage],
    state: dict[str, Any] | None = None,
) -> str:
    previous = parse_handoff_summary(current_summary) or {}
    if not previous and (current_summary or "").strip():
        previous = {"context_notes": [_truncate_text(current_summary, 260)]}
    extracted = {
        **_extract_state_signals(state),
        **_extract_message_signals(discarded_msgs),
    }
    merged = _merge_handoff_parts(previous, extracted, None)
    return _safe_json_dumps(merged)


def build_handoff_summary_prompt(
    current_summary: str,
    discarded_msgs: list[BaseMessage],
    state: dict[str, Any] | None = None,
) -> str:
    current_handoff = parse_handoff_summary(current_summary) or {}
    legacy_note = _truncate_text(current_summary, 260) if not current_handoff and (current_summary or "").strip() else ""
    prompt_payload = {
        "current_handoff": current_handoff,
        "legacy_note": legacy_note,
        "state_snapshot": build_state_snapshot(state),
        "discarded_events": build_discarded_context_payload(discarded_msgs),
    }
    return (
        "你是 MortyClaw 的上下文压缩器。你的任务是把旧上下文压缩成结构化 HandoffSummary。\n\n"
        "必须遵守：\n"
        "1. 只输出一个 JSON 对象，不要 Markdown，不要解释。\n"
        "2. 优先保留当前目标、活动任务、已完成步骤、待完成步骤、关键文件、命令结果、工具结果、风险。\n"
        "3. 旧工具输出必须剪枝，只保留 command/path/result_preview，不要复制长日志全文。\n"
        "4. 不要记录用户长期偏好；那部分由长期记忆模块负责。\n"
        "5. 如果某字段没有内容，用空字符串或空数组。\n\n"
        "目标 JSON 结构：\n"
        "{\n"
        '  "version": 1,\n'
        '  "goal": "",\n'
        '  "active_task": {\n'
        '    "route": "",\n'
        '    "goal": "",\n'
        '    "run_status": "",\n'
        '    "current_step_index": 0,\n'
        '    "total_steps": 0,\n'
        '    "current_step": "",\n'
        '    "current_project_path": "",\n'
        '    "pending_approval": false,\n'
        '    "approval_reason": "",\n'
        '    "last_error": ""\n'
        "  },\n"
        '  "completed_steps": [{"step": 0, "description": "", "status": "", "risk_level": "", "result_summary": ""}],\n'
        '  "pending_steps": [{"step": 0, "description": "", "status": "", "risk_level": ""}],\n'
        '  "todos": [{"id": "", "content": "", "status": ""}],\n'
        '  "files_touched": [{"path": "", "reason": "", "last_observation": ""}],\n'
        '  "commands_run": [{"tool_name": "", "command": "", "status": "", "result_summary": ""}],\n'
        '  "tool_results": [{"tool_name": "", "args_summary": "", "result_summary": "", "related_path": ""}],\n'
        '  "context_notes": [""],\n'
        '  "open_questions": [""],\n'
        '  "risks": [""],\n'
        '  "last_user_intent": ""\n'
        "}\n\n"
        f"输入数据：\n{json.dumps(prompt_payload, ensure_ascii=False, indent=2)}"
    )


def merge_handoff_summary(
    current_summary: str,
    discarded_msgs: list[BaseMessage],
    state: dict[str, Any] | None = None,
    llm_output_text: str | None = None,
) -> str:
    previous = parse_handoff_summary(current_summary) or {}
    if not previous and (current_summary or "").strip():
        previous = {"context_notes": [_truncate_text(current_summary, 260)]}
    extracted = {
        **_extract_state_signals(state),
        **_extract_message_signals(discarded_msgs),
    }
    llm_summary = parse_handoff_summary(llm_output_text) if llm_output_text else None
    merged = _merge_handoff_parts(previous, extracted, llm_summary or {})
    return _safe_json_dumps(merged)


def render_handoff_summary(summary_text: str | None) -> str:
    summary = parse_handoff_summary(summary_text)
    if summary is None:
        return (summary_text or "").strip()

    lines: list[str] = []

    goal = summary.get("goal", "")
    if goal:
        lines.append("目标")
        lines.append(goal)

    active_task = summary.get("active_task") or {}
    if active_task:
        lines.append("")
        lines.append("当前执行状态")
        status_parts = []
        if active_task.get("route"):
            status_parts.append(f"路径：{active_task['route']}")
        if active_task.get("run_status"):
            status_parts.append(f"状态：{active_task['run_status']}")
        total_steps = active_task.get("total_steps", 0) or 0
        current_step_index = active_task.get("current_step_index", 0) or 0
        if total_steps:
            status_parts.append(f"步骤：{current_step_index + 1}/{total_steps}")
        if active_task.get("current_step"):
            status_parts.append(f"当前步骤：{active_task['current_step']}")
        if active_task.get("current_project_path"):
            status_parts.append(f"项目：{active_task['current_project_path']}")
        if active_task.get("pending_approval"):
            status_parts.append("审批：待确认")
        if active_task.get("approval_reason"):
            status_parts.append(f"审批原因：{active_task['approval_reason']}")
        if active_task.get("last_error"):
            status_parts.append(f"最近错误：{active_task['last_error']}")
        lines.extend(status_parts)

    if summary.get("completed_steps"):
        lines.append("")
        lines.append("已完成")
        for step in summary["completed_steps"]:
            label = f"步骤 {step.get('step')}" if step.get("step") is not None else "步骤"
            detail = step.get("description") or step.get("result_summary") or ""
            if step.get("result_summary"):
                detail = f"{detail} | 结果：{step['result_summary']}" if detail else step["result_summary"]
            lines.append(f"- {label}：{detail}")

    if summary.get("pending_steps"):
        lines.append("")
        lines.append("待处理")
        for step in summary["pending_steps"]:
            label = f"步骤 {step.get('step')}" if step.get("step") is not None else "步骤"
            detail = step.get("description") or ""
            status = step.get("status") or "pending"
            if status:
                detail = f"{detail} | 状态：{status}" if detail else f"状态：{status}"
            lines.append(f"- {label}：{detail}")

    if summary.get("todos"):
        lines.append("")
        lines.append("当前 Todo")
        for item in summary["todos"]:
            lines.append(f"- [{item.get('status', 'pending')}] {item.get('content', '')}")

    if summary.get("files_touched"):
        lines.append("")
        lines.append("关键文件")
        for file_item in summary["files_touched"]:
            detail = file_item.get("reason") or file_item.get("last_observation") or ""
            if file_item.get("last_observation") and file_item.get("reason"):
                detail = f"{file_item['reason']} | {file_item['last_observation']}"
            lines.append(f"- {file_item.get('path', '')} {detail}".rstrip())

    if summary.get("commands_run"):
        lines.append("")
        lines.append("命令结果")
        for command in summary["commands_run"]:
            detail = command.get("result_summary") or ""
            if command.get("status"):
                detail = f"{command['status']} | {detail}" if detail else command["status"]
            lines.append(f"- {command.get('tool_name', '')} {command.get('command', '')} {detail}".rstrip())

    if summary.get("tool_results"):
        lines.append("")
        lines.append("工具结果")
        for result in summary["tool_results"]:
            detail = result.get("result_summary") or ""
            if result.get("related_path"):
                detail = f"{result['related_path']} | {detail}" if detail else result["related_path"]
            lines.append(f"- {result.get('tool_name', '')} {detail}".rstrip())

    if summary.get("context_notes"):
        lines.append("")
        lines.append("背景结论")
        for note in summary["context_notes"]:
            lines.append(f"- {note}")

    if summary.get("open_questions"):
        lines.append("")
        lines.append("未决问题")
        for question in summary["open_questions"]:
            lines.append(f"- {question}")

    if summary.get("risks"):
        lines.append("")
        lines.append("风险")
        for risk in summary["risks"]:
            lines.append(f"- {risk}")

    if summary.get("last_user_intent"):
        lines.append("")
        lines.append("最近用户意图")
        lines.append(summary["last_user_intent"])

    return "\n".join(line for line in lines if line is not None).strip()
