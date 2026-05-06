from __future__ import annotations

import json

from langchain_core.messages import ToolMessage

from ...error_policy import deserialize_classified_error


def _build_step_result(step: dict, *, result_summary: str, outcome: str, verification_status: str, verification_notes: str = "", blocking_issue: str = "") -> dict:
    return {
        "step": step.get("step"),
        "description": step.get("description", ""),
        "result_summary": (result_summary or "")[:200],
        "outcome": outcome,
        "verification_status": verification_status,
        "verification_notes": verification_notes[:200],
        "blocking_issue": blocking_issue[:200],
    }


def _message_additional_kwargs(message) -> dict:
    if message is None:
        return {}
    data = getattr(message, "additional_kwargs", {}) or {}
    return dict(data) if isinstance(data, dict) else {}


def _extract_tool_payload(message) -> dict | None:
    if not isinstance(message, ToolMessage):
        return None
    content = str(getattr(message, "content", "") or "").strip()
    if not content.startswith("{"):
        return None
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _step_expects_runtime_output(step_text: str) -> bool:
    lowered = str(step_text or "").lower()
    return any(
        hint in lowered
        for hint in ("打印", "输出", "结果", "示例", "stdout", "print", "show output", "run result")
    )


def _recover_successful_execution_summary(messages: list, current_step: dict) -> str:
    current_intent = str(current_step.get("intent") or "")
    if current_intent not in {"shell_execute", "test_verify"}:
        return ""

    step_text = str(current_step.get("description") or "")
    for message in reversed(messages or []):
        if not isinstance(message, ToolMessage):
            continue
        tool_name = str(getattr(message, "name", "") or "")
        if tool_name not in {"run_project_command", "run_project_tests"}:
            continue
        payload = _extract_tool_payload(message)
        if not payload or not payload.get("ok", False):
            continue
        if payload.get("exit_code", 0) not in {0, None}:
            continue

        stdout = str(payload.get("stdout", "") or "").strip()
        command = str(payload.get("command", "") or "").strip()
        if current_intent == "shell_execute":
            if not stdout or not _step_expects_runtime_output(step_text):
                continue
            return stdout[:200]

        return stdout[:200] if stdout else f"验证完成：{command or tool_name} 执行通过。"

    return ""


def make_reviewer_node(
    *,
    with_working_memory_fn,
    get_current_plan_step_fn,
    looks_like_step_failure_fn,
    update_plan_step_fn,
    step_requires_approval_fn,
    build_approval_reason_fn,
    classify_error_fn,
    serialize_classified_error_fn,
    plan_to_todos_fn,
    should_enable_todos_fn,
    build_todo_state_from_plan_fn,
    save_session_todo_state_fn,
    clear_session_todo_state_fn,
    audit_logger_instance,
):
    def reviewer_node(state, config) -> dict:
        thread_id = config.get("configurable", {}).get("thread_id", "system_default")
        current_step = get_current_plan_step_fn(state)
        if current_step is None:
            return with_working_memory_fn(state, {"run_status": "done"})

        last_message = state.get("messages", [])[-1] if state.get("messages") else None
        last_content = getattr(last_message, "content", "") if last_message is not None else ""
        additional_kwargs = _message_additional_kwargs(last_message)
        structured_outcome = str(additional_kwargs.get("mortyclaw_step_outcome", "") or "").strip().lower()
        structured_error = additional_kwargs.get("mortyclaw_error")
        current_intent = str(current_step.get("intent") or "")
        run_status_before = str(state.get("run_status", "") or "")

        recovered_success_summary = ""
        structured_classified = deserialize_classified_error(structured_error) if structured_error else None
        if (
            structured_classified is not None
            and structured_classified.kind.value == "unknown"
            and current_intent in {"shell_execute", "test_verify"}
        ):
            recovered_success_summary = _recover_successful_execution_summary(
                state.get("messages", []),
                current_step,
            )
            if recovered_success_summary:
                last_content = recovered_success_summary
                structured_outcome = "success_candidate"
                structured_error = None

        decision_source = "implicit_success"
        should_treat_as_failure = False
        classified = None

        if structured_error or structured_outcome == "failure":
            decision_source = "structured_error"
            should_treat_as_failure = True
        elif looks_like_step_failure_fn(last_content):
            decision_source = "content_failure"
            should_treat_as_failure = True
        elif structured_outcome == "success_candidate":
            decision_source = "success_candidate"
        elif last_message is None:
            decision_source = "tool_output_fallback"
            should_treat_as_failure = True
        elif isinstance(last_message, ToolMessage) and looks_like_step_failure_fn(last_content):
            decision_source = "tool_output_fallback"
            should_treat_as_failure = True

        if should_treat_as_failure:
            structured_classified = deserialize_classified_error(structured_error) if structured_error else None
            classified = structured_classified or classify_error_fn(
                message=last_content,
                state=state,
                tool_name=getattr(last_message, "name", "") if last_message is not None else "",
            )

            if (
                classified.retryable
                or classified.kind.value == "unsafe_tool_scope"
            ) and state.get("retry_count", 0) < state.get("max_retries", 2):
                audit_logger_instance.log_event(
                    thread_id=thread_id,
                    event="system_action",
                    content=(
                        f"reviewer requested retry for step {current_step['step']} | "
                        f"source={decision_source} | kind={classified.kind.value}"
                    ),
                    step=current_step["step"],
                    decision_source=decision_source,
                    error_kind=classified.kind.value,
                    run_status_before=run_status_before,
                    run_status_after="retrying",
                )
                return with_working_memory_fn(state, {
                    "plan": update_plan_step_fn(state.get("plan", []), state.get("current_step_index", 0), status="retrying"),
                    "last_error": last_content[:200],
                    "last_error_kind": classified.kind.value,
                    "last_recovery_action": classified.recovery_action.value,
                    "replan_reason": "",
                    "retry_count": state.get("retry_count", 0) + 1,
                    "run_status": "retrying",
                })

            failure_outcome = "blocked" if classified.kind.value in {"unsafe_tool_scope", "empty_llm_response"} else "failed"
            failed_step_result = _build_step_result(
                current_step,
                result_summary=last_content,
                outcome=failure_outcome,
                verification_status="failed" if current_intent == "test_verify" else "skipped",
                verification_notes=current_step.get("verification_hint", "") if current_intent == "test_verify" else "",
                blocking_issue=classified.user_visible_hint or last_content,
            )
            failed_plan = update_plan_step_fn(state.get("plan", []), state.get("current_step_index", 0), status="failed")
            failed_step_results = list(state.get("step_results", []))
            failed_step_results.append(failed_step_result)

            if classified.recovery_action.value == "abort":
                todos = plan_to_todos_fn(failed_plan, state.get("current_step_index", 0)) if should_enable_todos_fn("slow", failed_plan) else []
                if todos:
                    save_session_todo_state_fn(
                        thread_id,
                        build_todo_state_from_plan_fn(
                            failed_plan,
                            state.get("current_step_index", 0),
                            revision=int(state.get("todo_revision", 0) or 0) + 1,
                            last_event="failed",
                        ),
                    )
                else:
                    clear_session_todo_state_fn(thread_id)
                if last_message is not None and not getattr(last_message, "additional_kwargs", {}).get("mortyclaw_error"):
                    getattr(last_message, "additional_kwargs", {})["mortyclaw_error"] = serialize_classified_error_fn(classified)
                audit_logger_instance.log_event(
                    thread_id=thread_id,
                    event="system_action",
                    content=(
                        f"reviewer failed step {current_step['step']} | "
                        f"source={decision_source} | kind={classified.kind.value}"
                    ),
                    step=current_step["step"],
                    decision_source=decision_source,
                    error_kind=classified.kind.value,
                    run_status_before=run_status_before,
                    run_status_after="failed",
                )
                return with_working_memory_fn(state, {
                    "plan": failed_plan,
                    "step_results": failed_step_results,
                    "todos": todos,
                    "todo_revision": int(state.get("todo_revision", 0) or 0) + (1 if todos else 0),
                    "todo_needs_announcement": False,
                    "last_error": last_content[:200],
                    "last_error_kind": classified.kind.value,
                    "last_recovery_action": classified.recovery_action.value,
                    "replan_reason": classified.user_visible_hint or last_content[:200],
                    "retry_count": 0,
                    "final_answer": classified.user_visible_hint or last_content[:200],
                    "run_status": "failed",
                })

            audit_logger_instance.log_event(
                thread_id=thread_id,
                event="system_action",
                content=(
                    f"reviewer requested replan for step {current_step['step']} | "
                    f"source={decision_source} | kind={classified.kind.value}"
                ),
                step=current_step["step"],
                decision_source=decision_source,
                error_kind=classified.kind.value,
                run_status_before=run_status_before,
                run_status_after="replan_requested",
            )
            return with_working_memory_fn(state, {
                "plan": failed_plan,
                "step_results": failed_step_results,
                "todos": plan_to_todos_fn(failed_plan, state.get("current_step_index", 0)) if should_enable_todos_fn("slow", failed_plan) else [],
                "last_error": last_content[:200],
                "last_error_kind": classified.kind.value,
                "last_recovery_action": classified.recovery_action.value,
                "replan_reason": classified.user_visible_hint or last_content[:200],
                "retry_count": 0,
                "run_status": "replan_requested",
            })

        updated_plan = update_plan_step_fn(state.get("plan", []), state.get("current_step_index", 0), status="completed")
        updated_step_results = list(state.get("step_results", []))
        updated_step_results.append(
            _build_step_result(
                current_step,
                result_summary=last_content,
                outcome="completed",
                verification_status="passed" if current_intent == "test_verify" else "skipped",
                verification_notes=current_step.get("verification_hint", "") if current_intent == "test_verify" else "",
            )
        )

        if state.get("current_step_index", 0) + 1 < len(updated_plan):
            next_step_index = state.get("current_step_index", 0) + 1
            next_step = updated_plan[next_step_index]
            next_step_requires_approval = step_requires_approval_fn(next_step)
            todos = plan_to_todos_fn(updated_plan, next_step_index) if should_enable_todos_fn("slow", updated_plan) else []
            next_todo_revision = int(state.get("todo_revision", 0) or 0) + (1 if todos else 0)
            if todos:
                save_session_todo_state_fn(
                    thread_id,
                    build_todo_state_from_plan_fn(
                        updated_plan,
                        next_step_index,
                        revision=next_todo_revision,
                        last_event="step_advanced",
                    ),
                )
            else:
                clear_session_todo_state_fn(thread_id)
            audit_logger_instance.log_event(
                thread_id=thread_id,
                event="system_action",
                content=(
                    f"reviewer accepted step {current_step['step']} | "
                    f"source={decision_source} | next_step={next_step_index + 1} | "
                    f"approval_required={next_step_requires_approval}"
                ),
                step=current_step["step"],
                decision_source=decision_source,
                error_kind="",
                run_status_before=run_status_before,
                run_status_after="awaiting_step_approval" if next_step_requires_approval else "next_step",
            )
            return with_working_memory_fn(state, {
                "plan": updated_plan,
                "step_results": updated_step_results,
                "current_step_index": next_step_index,
                "pending_approval": next_step_requires_approval,
                "approval_granted": False,
                "approval_prompted": False,
                "approval_reason": build_approval_reason_fn(next_step) if next_step_requires_approval else "",
                "retry_count": 0,
                "last_error": "",
                "last_error_kind": "",
                "last_recovery_action": "",
                "replan_reason": "",
                "todos": todos,
                "todo_revision": next_todo_revision if todos else 0,
                "todo_needs_announcement": bool(todos),
                "run_status": "awaiting_step_approval" if next_step_requires_approval else "next_step",
            })

        audit_logger_instance.log_event(
            thread_id=thread_id,
            event="system_action",
            content=(
                f"reviewer accepted step {current_step['step']} | "
                f"source={decision_source} | workflow_done=true"
            ),
            step=current_step["step"],
            decision_source=decision_source,
            error_kind="",
            run_status_before=run_status_before,
            run_status_after="done",
        )
        final_todos = plan_to_todos_fn(updated_plan, state.get("current_step_index", 0)) if should_enable_todos_fn("slow", updated_plan) else []
        if final_todos:
            save_session_todo_state_fn(
                thread_id,
                build_todo_state_from_plan_fn(
                    updated_plan,
                    state.get("current_step_index", 0),
                    revision=int(state.get("todo_revision", 0) or 0) + 1,
                    last_event="done",
                ),
            )
        else:
            clear_session_todo_state_fn(thread_id)
        return with_working_memory_fn(state, {
            "plan": updated_plan,
            "step_results": updated_step_results,
            "pending_approval": False,
            "approval_granted": False,
            "approval_prompted": False,
            "approval_reason": "",
            "retry_count": 0,
            "last_error": "",
            "last_error_kind": "",
            "last_recovery_action": "",
            "replan_reason": "",
            "todos": final_todos,
            "todo_revision": int(state.get("todo_revision", 0) or 0) + (1 if final_todos else 0),
            "todo_needs_announcement": False,
            "run_status": "done",
        })

    return reviewer_node
