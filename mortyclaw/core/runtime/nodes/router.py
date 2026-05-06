from __future__ import annotations

import re

from ...planning import looks_like_file_write_request, step_matches_shell_action, step_matches_test_action


_NUMBERED_REQUIREMENT_PATTERN = re.compile(r"(?<!\d)(\d+)\.\s*")
_LEADING_PROJECT_PATH_PATTERN = re.compile(r"^\s*(?:工作目录是[:：]\s*)?(?:/)?mnt/[^\s]+")
_IMPLEMENTATION_NOUN_HINTS = ("流式输出", "对话历史", "历史对话", "聊天历史", "基础 Agent", "Agent 能力", "日志", "保存", "加载")


def _normalize_requirement_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip(" \n\t-；;，,。:")
    if not cleaned:
        return ""
    primary = re.split(r"\s+-\s+|[；;。]\s*", cleaned, maxsplit=1)[0].strip(" -；;，,。:")
    primary = re.sub(r"^(请在此基础上增加以下功能[:：]?)", "", primary).strip()
    if not primary:
        primary = cleaned
    if not any(token in primary for token in ("实现", "增加", "修改", "支持", "运行", "查看", "检查", "保存", "加载")):
        if any(hint in primary for hint in _IMPLEMENTATION_NOUN_HINTS):
            primary = f"实现{primary}"
    return primary[:64]


def _extract_numbered_requirements(goal: str) -> list[str]:
    normalized_goal = re.sub(r"\s+", " ", str(goal or "")).strip()
    matches = list(_NUMBERED_REQUIREMENT_PATTERN.finditer(normalized_goal))
    if not matches:
        return []

    requirements: list[str] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized_goal)
        requirement = _normalize_requirement_text(normalized_goal[start:end])
        if requirement:
            requirements.append(requirement)
    return requirements


def _looks_like_explicit_project_code_task(
    goal: str,
    *,
    current_project_path: str,
    risk_level: str,
) -> bool:
    task_text = str(goal or "").strip()
    if not task_text or not str(current_project_path or "").strip():
        return False
    return (
        str(risk_level or "").strip().lower() == "high"
        or looks_like_file_write_request(task_text)
        or step_matches_test_action(task_text)
        or step_matches_shell_action(task_text)
    )


def _should_prefer_planner_for_explicit_project_task(
    goal: str,
    *,
    current_project_path: str,
    route_decision: dict | None = None,
) -> bool:
    task_text = str(goal or "").strip()
    project_path = str(current_project_path or "").strip()
    decision = route_decision or {}
    if not task_text or not project_path:
        return False
    if str(decision.get("route", "") or "") != "slow":
        return False
    if str(decision.get("route_source", "") or "") in {"planner_first_uncertain", "mixed_research_task"}:
        return True

    cleaned_goal = _LEADING_PROJECT_PATH_PATTERN.sub("", task_text).strip()
    numbered_requirements = _extract_numbered_requirements(cleaned_goal)
    has_file_write = looks_like_file_write_request(task_text)
    has_runtime_execution = step_matches_shell_action(task_text) or step_matches_test_action(task_text)

    return bool(numbered_requirements) or (has_file_write and has_runtime_execution)


def _build_initial_autonomous_todos(
    goal: str,
    *,
    current_project_path: str = "",
    risk_level: str = "",
) -> list[dict]:
    if not _looks_like_explicit_project_code_task(
        goal,
        current_project_path=current_project_path,
        risk_level=risk_level,
    ):
        return []

    cleaned_goal = _LEADING_PROJECT_PATH_PATTERN.sub("", str(goal or "")).strip()
    requirements = _extract_numbered_requirements(cleaned_goal)
    todos: list[dict] = [
        {
            "id": "todo-1",
            "content": "检查目标文件与当前实现，确认最小修改范围",
            "status": "in_progress",
        }
    ]

    for requirement in requirements[:3]:
        todos.append({
            "id": f"todo-{len(todos) + 1}",
            "content": requirement,
            "status": "pending",
        })

    if len(todos) == 1:
        todos.append({
            "id": "todo-2",
            "content": "在项目文件中实现用户要求的代码修改",
            "status": "pending",
        })

    if not any("验证" in item["content"] or "diff" in item["content"].lower() for item in todos):
        todos.append({
            "id": f"todo-{len(todos) + 1}",
            "content": "查看 diff 并运行验证",
            "status": "pending",
        })

    return todos


def make_router_node(
    *,
    with_working_memory_fn,
    get_latest_user_query_fn,
    schedule_long_term_memory_capture_fn,
    sync_session_memory_from_query_fn,
    load_session_project_path_fn,
    build_route_decision_fn,
    clear_session_todo_state_fn,
    audit_logger_instance,
):
    def router_node(state, config) -> dict:
        thread_id = config.get("configurable", {}).get("thread_id", "system_default")
        latest_user_query = get_latest_user_query_fn(state.get("messages", []))
        schedule_long_term_memory_capture_fn(latest_user_query)
        session_state_updates = sync_session_memory_from_query_fn(latest_user_query, thread_id)
        if not session_state_updates.get("current_project_path"):
            existing_project_path = state.get("current_project_path") or load_session_project_path_fn(thread_id)
            if existing_project_path:
                session_state_updates["current_project_path"] = existing_project_path

        if state.get("pending_approval"):
            audit_logger_instance.log_event(
                thread_id=thread_id,
                event="system_action",
                content="router resumed slow path while waiting for approval response",
            )
            return with_working_memory_fn(state, {
                "route": "slow",
                "goal": state.get("goal", ""),
                "complexity": state.get("complexity", "high_risk"),
                "risk_level": state.get("risk_level", "high"),
                "planner_required": state.get("planner_required", True),
                "route_locked": state.get("route_locked", False),
                "route_source": state.get("route_source", "resume_pending_approval"),
                "route_reason": state.get("route_reason", ""),
                "route_confidence": state.get("route_confidence", 1.0),
                "plan_source": state.get("plan_source", ""),
                "replan_reason": state.get("replan_reason", ""),
                "plan": state.get("plan", []),
                "current_step_index": state.get("current_step_index", 0),
                "step_results": state.get("step_results", []),
                "pending_approval": True,
                "approval_granted": state.get("approval_granted", False),
                "approval_prompted": state.get("approval_prompted", False),
                "approval_reason": state.get("approval_reason", ""),
                "permission_mode": state.get("permission_mode", ""),
                "permission_prompted": state.get("permission_prompted", False),
                "last_error": "",
                "last_error_kind": state.get("last_error_kind", ""),
                "last_recovery_action": state.get("last_recovery_action", ""),
                "retry_count": state.get("retry_count", 0),
                "max_retries": state.get("max_retries", 2) or 2,
                "todos": state.get("todos", []),
                "active_todos": state.get("active_todos", state.get("todos", [])),
                "todo_revision": state.get("todo_revision", 0),
                "todo_needs_announcement": False,
                "last_todo_tool_call_id": state.get("last_todo_tool_call_id", ""),
                "pending_tool_calls": state.get("pending_tool_calls", []),
                "pending_execution_snapshot": state.get("pending_execution_snapshot", {}),
                "slow_execution_mode": state.get("slow_execution_mode", "autonomous"),
                "final_answer": "",
                "run_status": "awaiting_approval_response",
                "execution_guard_status": state.get("execution_guard_status", ""),
                "execution_guard_reason": state.get("execution_guard_reason", ""),
                **session_state_updates,
            })

        if (
            str(state.get("route", "") or "") == "slow"
            and not str(state.get("permission_mode", "") or "").strip().lower()
            and state.get("permission_prompted", False)
            and str(state.get("run_status", "") or "") == "waiting_user"
        ):
            audit_logger_instance.log_event(
                thread_id=thread_id,
                event="system_action",
                content="router resumed slow path while waiting for execution mode selection",
            )
            return with_working_memory_fn(state, {
                "route": "slow",
                "goal": state.get("goal", latest_user_query or ""),
                "complexity": state.get("complexity", "high_risk"),
                "risk_level": state.get("risk_level", "high"),
                "planner_required": state.get("planner_required", False),
                "route_locked": state.get("route_locked", False),
                "route_source": state.get("route_source", "resume_permission_selection"),
                "route_reason": state.get("route_reason", ""),
                "route_confidence": state.get("route_confidence", 1.0),
                "plan_source": state.get("plan_source", ""),
                "replan_reason": state.get("replan_reason", ""),
                "plan": state.get("plan", []),
                "current_step_index": state.get("current_step_index", 0),
                "step_results": state.get("step_results", []),
                "pending_approval": state.get("pending_approval", False),
                "approval_granted": False,
                "approval_prompted": False,
                "approval_reason": state.get("approval_reason", ""),
                "permission_mode": "",
                "permission_prompted": True,
                "last_error": "",
                "last_error_kind": state.get("last_error_kind", ""),
                "last_recovery_action": state.get("last_recovery_action", ""),
                "retry_count": state.get("retry_count", 0),
                "max_retries": state.get("max_retries", 2) or 2,
                "todos": state.get("todos", []),
                "active_todos": state.get("active_todos", state.get("todos", [])),
                "todo_revision": state.get("todo_revision", 0),
                "todo_needs_announcement": False,
                "last_todo_tool_call_id": state.get("last_todo_tool_call_id", ""),
                "pending_tool_calls": state.get("pending_tool_calls", []),
                "pending_execution_snapshot": state.get("pending_execution_snapshot", {}),
                "slow_execution_mode": state.get("slow_execution_mode", "autonomous"),
                "final_answer": "",
                "run_status": "awaiting_permission_mode",
                "execution_guard_status": state.get("execution_guard_status", ""),
                "execution_guard_reason": state.get("execution_guard_reason", ""),
                **session_state_updates,
            })

        route_decision = build_route_decision_fn(latest_user_query)
        prefer_planner_for_explicit_project_task = _should_prefer_planner_for_explicit_project_task(
            latest_user_query or "",
            current_project_path=str(session_state_updates.get("current_project_path", "") or ""),
            route_decision=route_decision,
        )
        autonomous_slow = (
            route_decision.get("route") == "slow"
            and not prefer_planner_for_explicit_project_task
        )
        if autonomous_slow:
            route_decision = {
                **route_decision,
                "planner_required": False,
            }
        if route_decision.get("route") != "slow":
            clear_session_todo_state_fn(thread_id)
        initial_todos = (
            _build_initial_autonomous_todos(
                latest_user_query or "",
                current_project_path=str(session_state_updates.get("current_project_path", "") or ""),
                risk_level=str(route_decision.get("risk_level", "") or ""),
            )
            if autonomous_slow
            else []
        )
        audit_logger_instance.log_event(
            thread_id=thread_id,
            event="system_action",
            content=(
                f"router selected {route_decision['route']} path | "
                f"complexity={route_decision['complexity']} | "
                f"risk={route_decision['risk_level']} | "
                f"source={route_decision.get('route_source', 'unknown')}"
            ),
        )
        return with_working_memory_fn(state, {
            **route_decision,
            "plan_source": "",
            "replan_reason": "",
            "plan": [],
            "current_step_index": 0,
            "step_results": [],
            "pending_approval": False,
            "approval_granted": False,
            "approval_prompted": False,
            "approval_reason": "",
            "permission_mode": "",
            "permission_prompted": False,
            "last_error": "",
            "last_error_kind": "",
            "last_recovery_action": "",
            "retry_count": 0,
            "max_retries": state.get("max_retries", 2) or 2,
            "todos": initial_todos,
            "active_todos": list(initial_todos),
            "todo_revision": 1 if initial_todos else 0,
            "todo_needs_announcement": bool(initial_todos),
            "last_todo_tool_call_id": "",
            "pending_tool_calls": [],
            "pending_execution_snapshot": {},
            "slow_execution_mode": "autonomous" if autonomous_slow else ("structured" if route_decision.get("route") == "slow" else ""),
            "execution_guard_status": "",
            "execution_guard_reason": "",
            "final_answer": "",
            "run_status": "routing",
            **session_state_updates,
        })

    return router_node
