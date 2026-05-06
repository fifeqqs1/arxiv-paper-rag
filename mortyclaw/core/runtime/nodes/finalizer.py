from __future__ import annotations

from langchain_core.messages import AIMessage


def _build_final_answer(state) -> str:
    plan = [dict(step) for step in (state.get("plan", []) or []) if isinstance(step, dict)]
    todo_items = [dict(item) for item in (state.get("todos", []) or []) if isinstance(item, dict)]
    step_results = [dict(item) for item in (state.get("step_results", []) or []) if isinstance(item, dict)]
    run_status = str(state.get("run_status", "") or "")
    failed_steps = [step for step in plan if step.get("status") == "failed"]
    pending_steps = [
        step for step in plan
        if step.get("status") not in {"completed", "failed", "cancelled"}
    ]
    validation_results = [
        item for item in step_results
        if item.get("verification_status") in {"passed", "failed"}
    ]

    intro = "复杂任务已完成。" if run_status == "done" else "复杂任务未完全完成。"
    lines = [intro, "", "Todo 完成情况："]
    if plan:
        for step in plan:
            marker = {
                "completed": "[x]",
                "failed": "[!]",
                "cancelled": "[-]",
                "retrying": "[~]",
            }.get(str(step.get("status") or ""), "[ ]")
            lines.append(f"{marker} {step.get('step')}. {step.get('description', '')}")
    elif todo_items:
        for index, item in enumerate(todo_items, start=1):
            marker = {
                "completed": "[x]",
                "cancelled": "[-]",
                "in_progress": "[>]",
            }.get(str(item.get("status") or ""), "[ ]")
            lines.append(f"{marker} {index}. {item.get('content', '')}")
    else:
        lines.append("- 本次没有生成可执行计划。")

    lines.extend(["", "执行结果："])
    if step_results:
        for item in step_results[-8:]:
            lines.append(
                f"- 步骤 {item.get('step')}: {item.get('description', '')} | "
                f"outcome={item.get('outcome', 'completed')} | {item.get('result_summary', '')}"
            )
    else:
        lines.append("- 暂无可记录的步骤结果。")

    lines.extend(["", "验证结果："])
    if validation_results:
        for item in validation_results[-6:]:
            lines.append(
                f"- 步骤 {item.get('step')}: {item.get('verification_status')} | "
                f"{item.get('verification_notes') or item.get('result_summary', '')}"
            )
    else:
        lines.append("- 本次没有单独的验证步骤，或未产出明确验证结果。")

    lines.extend(["", "遗留问题："])
    leftover_lines: list[str] = []
    if failed_steps:
        leftover_lines.extend(f"- 未完成步骤：{step.get('description', '')}" for step in failed_steps)
    if pending_steps and run_status != "done":
        leftover_lines.extend(f"- 待处理步骤：{step.get('description', '')}" for step in pending_steps[:4])
    if state.get("last_error"):
        leftover_lines.append(f"- 最近错误：{state.get('last_error')}")
    if not leftover_lines:
        leftover_lines.append("- 无明显遗留问题。")
    lines.extend(leftover_lines)
    return "\n".join(lines).strip()


def make_finalizer_node(
    *,
    with_working_memory_fn,
    clear_session_todo_state_fn,
    audit_logger_instance,
):
    def finalizer_node(state, config) -> dict:
        thread_id = config.get("configurable", {}).get("thread_id", "system_default")
        clear_session_todo_state_fn(thread_id)
        autonomous_answer = ""
        should_emit_finalizer_message = True
        if (state.get("slow_execution_mode", "") or "").strip().lower() == "autonomous":
            autonomous_answer = str(state.get("final_answer", "") or "").strip()
            last_message = state.get("messages", [])[-1] if state.get("messages") else None
            last_message_kind = ""
            if last_message is not None:
                additional_kwargs = getattr(last_message, "additional_kwargs", {}) or {}
                if isinstance(additional_kwargs, dict):
                    last_message_kind = str(additional_kwargs.get("mortyclaw_response_kind", "") or "").strip().lower()
            should_emit_finalizer_message = not (
                autonomous_answer
                and last_message is not None
                and str(getattr(last_message, "content", "") or "").strip() == autonomous_answer
                and last_message_kind == "final_answer"
            )
        final_answer = autonomous_answer or _build_final_answer(state)
        audit_logger_instance.log_event(
            thread_id=thread_id,
            event="system_action",
            content=f"finalizer produced slow-path summary with status={state.get('run_status', '')}",
        )
        updates = {"final_answer": final_answer}
        if should_emit_finalizer_message:
            updates["messages"] = [
                AIMessage(content=final_answer, additional_kwargs={"mortyclaw_response_kind": "final_answer"})
            ]
        return with_working_memory_fn(state, updates)

    return finalizer_node
