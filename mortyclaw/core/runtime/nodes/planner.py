from __future__ import annotations


def make_planner_node(
    *,
    with_working_memory_fn,
    get_latest_user_query_fn,
    build_plan_with_llm_fn,
    build_rule_execution_plan_fn,
    build_default_replan_input_fn,
    planner_llm,
    available_tool_names,
    planner_confidence_threshold,
    step_requires_approval_fn,
    build_approval_reason_fn,
    plan_to_todos_fn,
    should_enable_todos_fn,
    build_todo_state_from_plan_fn,
    save_session_todo_state_fn,
    clear_session_todo_state_fn,
    audit_logger_instance,
):
    def planner_node(state, config) -> dict:
        thread_id = config.get("configurable", {}).get("thread_id", "system_default")
        latest_user_query = get_latest_user_query_fn(state.get("messages", []))

        if state.get("pending_approval") and state.get("plan"):
            audit_logger_instance.log_event(
                thread_id=thread_id,
                event="system_action",
                content="planner reused existing plan while waiting for approval response",
            )
            return with_working_memory_fn(state, {
                "route": state.get("route", "slow"),
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
                "todos": state.get("todos", []),
                "active_todos": state.get("active_todos", state.get("todos", [])),
                "todo_revision": state.get("todo_revision", 0),
                "todo_needs_announcement": False,
                "last_todo_tool_call_id": state.get("last_todo_tool_call_id", ""),
                "pending_tool_calls": state.get("pending_tool_calls", []),
                "slow_execution_mode": state.get("slow_execution_mode", "structured"),
                "run_status": "awaiting_approval_response",
            })

        risk_level = state.get("risk_level", "medium")
        plan_source = latest_user_query
        if state.get("run_status") == "replan_requested":
            plan_source = build_default_replan_input_fn(state)

        execution_plan = []
        plan_source_label = "llm_planner"
        planner_payload = None
        route_decision = {
            "route": state.get("route", "slow"),
            "risk_level": risk_level,
            "route_locked": state.get("route_locked", False),
            "route_reason": state.get("route_reason", ""),
        }
        planner_payload = build_plan_with_llm_fn(
            planner_llm,
            plan_source or state.get("goal", ""),
            state,
            route_decision,
            available_tool_names,
        )

        can_downgrade_to_fast = (
            planner_payload
            and state.get("run_status") != "replan_requested"
            and planner_payload.get("route") == "fast"
            and not state.get("route_locked", False)
            and float(planner_payload.get("confidence", 0.0) or 0.0) >= planner_confidence_threshold
        )
        if can_downgrade_to_fast:
            clear_session_todo_state_fn(thread_id)
            audit_logger_instance.log_event(
                thread_id=thread_id,
                event="system_action",
                content="planner downgraded task to fast path after semantic assessment",
            )
            return with_working_memory_fn(state, {
                "route": "fast",
                "planner_required": False,
                "route_locked": False,
                "route_source": "planner_fast",
                "route_reason": planner_payload.get("reason", "") or "planner downgraded to fast",
                "route_confidence": float(planner_payload.get("confidence", 0.0) or 0.0),
                "goal": planner_payload.get("goal") or state.get("goal", latest_user_query or ""),
                "plan_source": "planner_fast",
                "replan_reason": "",
                "plan": [],
                "current_step_index": 0,
                "step_results": [],
                "pending_approval": False,
                "approval_granted": False,
                "approval_prompted": False,
                "approval_reason": "",
                "todos": [],
                "active_todos": [],
                "todo_revision": 0,
                "todo_needs_announcement": False,
                "last_todo_tool_call_id": "",
                "pending_tool_calls": [],
                "pending_execution_snapshot": {},
                "slow_execution_mode": "",
                "run_status": "planned_fast",
            })

        if planner_payload and planner_payload.get("route") == "slow" and planner_payload.get("steps"):
            execution_plan = planner_payload["steps"]
        if not execution_plan:
            execution_plan = build_rule_execution_plan_fn(plan_source or state.get("goal", latest_user_query or ""), risk_level)
            plan_source_label = "rule_fallback"
        current_step = execution_plan[0] if execution_plan else None
        requires_approval = step_requires_approval_fn(current_step) and not state.get("approval_granted", False)
        approval_reason = build_approval_reason_fn(current_step) if requires_approval else ""
        todos = plan_to_todos_fn(execution_plan, 0) if should_enable_todos_fn("slow", execution_plan) else []
        todo_revision = int(state.get("todo_revision", 0) or 0) + (1 if todos else 0)
        if todos:
            save_session_todo_state_fn(
                thread_id,
                build_todo_state_from_plan_fn(
                    execution_plan,
                    0,
                    revision=todo_revision,
                    last_event="planned",
                ),
            )
        else:
            clear_session_todo_state_fn(thread_id)

        audit_logger_instance.log_event(
            thread_id=thread_id,
            event="system_action",
            content=(
                f"planner created {len(execution_plan)} steps; approval_required={requires_approval}; "
                f"source={plan_source_label}"
            ),
        )
        return with_working_memory_fn(state, {
            "route": "slow",
            "planner_required": True,
            "route_locked": state.get("route_locked", False),
            "route_source": plan_source_label,
            "route_reason": (
                (planner_payload or {}).get("reason")
                or state.get("route_reason", "")
                or "planner produced slow-path execution plan"
            ),
            "route_confidence": float((planner_payload or {}).get("confidence", state.get("route_confidence", 1.0)) or 0.0),
            "goal": (planner_payload or {}).get("goal") or state.get("goal", latest_user_query or ""),
            "plan_source": plan_source_label,
            "replan_reason": "",
            "plan": execution_plan,
            "current_step_index": 0,
            "step_results": [] if state.get("run_status") != "replan_requested" else list(state.get("step_results", [])),
            "pending_approval": requires_approval,
            "approval_granted": state.get("approval_granted", False),
            "approval_prompted": False,
            "approval_reason": approval_reason,
            "permission_mode": state.get("permission_mode", ""),
            "permission_prompted": state.get("permission_prompted", False),
            "todos": todos,
            "active_todos": list(todos),
            "todo_revision": todo_revision if todos else 0,
            "todo_needs_announcement": bool(todos),
            "last_todo_tool_call_id": "",
            "pending_tool_calls": [],
            "pending_execution_snapshot": {},
            "slow_execution_mode": "structured",
            "run_status": "planned",
        })

    return planner_node
