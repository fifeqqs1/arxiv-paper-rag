from __future__ import annotations


def make_execution_guard_node(
    *,
    with_working_memory_fn,
    validate_pending_execution_snapshot_fn,
    audit_logger_instance,
):
    def execution_guard_node(state, config) -> dict:
        thread_id = config.get("configurable", {}).get("thread_id", "system_default")

        if not (state.get("approval_granted") and state.get("pending_tool_calls")):
            return with_working_memory_fn(state, {
                "execution_guard_status": "skipped",
                "execution_guard_reason": "",
            })

        validation = validate_pending_execution_snapshot_fn(state)
        status = str(validation.get("status", "passed") or "passed")
        reason = str(validation.get("reason", "") or "")

        audit_logger_instance.log_event(
            thread_id=thread_id,
            event="system_action",
            content=f"execution guard {status} | reason={reason or 'none'}",
        )

        if validation.get("ok", False):
            return with_working_memory_fn(state, {
                "execution_guard_status": status,
                "execution_guard_reason": reason,
            })

        return with_working_memory_fn(state, {
            "route": "slow",
            "planner_required": True,
            "route_source": "resume_validation",
            "route_reason": "execution context drift detected",
            "replan_reason": reason or "恢复执行前检测到环境变化，需要重新规划。",
            "approval_granted": False,
            "approval_prompted": False,
            "pending_approval": False,
            "approval_reason": "",
            "permission_mode": "",
            "permission_prompted": False,
            "pending_tool_calls": [],
            "pending_execution_snapshot": {},
            "execution_guard_status": status,
            "execution_guard_reason": reason,
            "last_error": (reason or "")[:200],
            "final_answer": "",
            "run_status": "replan_requested",
        })

    return execution_guard_node
