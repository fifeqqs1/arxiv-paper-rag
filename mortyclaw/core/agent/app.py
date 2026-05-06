from __future__ import annotations

from typing import Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

from ..context import AgentState


def _coerce_invokable_llm(candidate):
    if candidate is None:
        return None
    if hasattr(candidate, "invoke"):
        return candidate
    binder = getattr(candidate, "bind_tools", None)
    if callable(binder):
        try:
            bound = binder([])
        except Exception:
            return None
        if hasattr(bound, "invoke"):
            return bound
    return None


def route_after_fast_agent(state) -> str:
    if (
        state.get("run_status") == "planner_requested"
        or (
            state.get("route") == "slow"
            and state.get("planner_required", False)
            and state.get("route_source") == "fast_escalation"
        )
    ):
        return "planner"
    return tools_condition(state)


def route_after_execution_guard(state) -> str:
    if state.get("run_status") == "replan_requested":
        return "replan"
    if state.get("run_status") in {"waiting_user", "cancelled"}:
        return "end"
    return "execute"


def create_agent_app(
    provider_name: str = "openai",
    model_name: str = "gpt-4o-mini",
    tools: Optional[list[BaseTool]] = None,
    checkpointer=None,
):
    from . import __dict__ as agent_namespace

    if tools is None:
        dynamic_tools = agent_namespace["load_dynamic_skills"]()
        actual_tools = agent_namespace["BUILTIN_TOOLS"] + dynamic_tools
    else:
        actual_tools = tools

    fast_tool_node = ToolNode(actual_tools)
    slow_tool_node = ToolNode(actual_tools)

    llm = agent_namespace["get_provider"](provider_name=provider_name, model_name=model_name)
    llm_with_tools = llm.bind_tools(actual_tools)
    planner_llm = None
    planner_llm_unavailable = False

    def _load_lightweight_planner_llm():
        nonlocal planner_llm, planner_llm_unavailable
        if planner_llm_unavailable:
            return None
        if planner_llm is not None:
            return planner_llm
        try:
            planner_candidate = agent_namespace["get_provider"](
                provider_name=provider_name,
                model_name=agent_namespace["_get_route_classifier_model"](),
                temperature=0.0,
            )
        except Exception:
            planner_llm_unavailable = True
            return None
        planner_llm = _coerce_invokable_llm(planner_candidate)
        if planner_llm is None:
            planner_llm_unavailable = True
        return planner_llm

    router_node = agent_namespace["make_router_node"](
        with_working_memory_fn=agent_namespace["_with_working_memory"],
        get_latest_user_query_fn=agent_namespace["_get_latest_user_query"],
        schedule_long_term_memory_capture_fn=agent_namespace["_schedule_long_term_memory_capture"],
        sync_session_memory_from_query_fn=agent_namespace["_sync_session_memory_from_query"],
        load_session_project_path_fn=agent_namespace["_load_session_project_path"],
        build_route_decision_fn=agent_namespace["_build_route_decision"],
        clear_session_todo_state_fn=agent_namespace["_clear_session_todo_state"],
        audit_logger_instance=agent_namespace["audit_logger"],
    )
    planner_node = agent_namespace["make_planner_node"](
        with_working_memory_fn=agent_namespace["_with_working_memory"],
        get_latest_user_query_fn=agent_namespace["_get_latest_user_query"],
        build_plan_with_llm_fn=agent_namespace["_build_plan_with_llm"],
        build_rule_execution_plan_fn=agent_namespace["_build_rule_execution_plan"],
        build_default_replan_input_fn=agent_namespace["_build_default_replan_input"],
        planner_llm=_load_lightweight_planner_llm(),
        available_tool_names=[getattr(tool, "name", "") for tool in actual_tools],
        planner_confidence_threshold=agent_namespace["ROUTE_CLASSIFIER_CONFIDENCE_THRESHOLD"],
        step_requires_approval_fn=agent_namespace["_step_requires_approval"],
        build_approval_reason_fn=agent_namespace["_build_approval_reason"],
        plan_to_todos_fn=agent_namespace["plan_to_todos"],
        should_enable_todos_fn=agent_namespace["should_enable_todos"],
        build_todo_state_from_plan_fn=agent_namespace["build_todo_state_from_plan"],
        save_session_todo_state_fn=agent_namespace["_save_session_todo_state"],
        clear_session_todo_state_fn=agent_namespace["_clear_session_todo_state"],
        audit_logger_instance=agent_namespace["audit_logger"],
    )
    approval_gate_node = agent_namespace["make_approval_gate_node"](
        with_working_memory_fn=agent_namespace["_with_working_memory"],
        get_latest_user_query_fn=agent_namespace["_get_latest_user_query"],
        build_approval_message_fn=agent_namespace["_build_approval_message"],
        build_permission_mode_message_fn=agent_namespace["_build_permission_mode_message"],
        parse_permission_mode_response_fn=agent_namespace["_parse_permission_mode_response"],
        build_default_replan_input_fn=agent_namespace["_build_default_replan_input"],
        is_negative_approval_response_fn=agent_namespace["_is_negative_approval_response"],
        is_affirmative_approval_response_fn=agent_namespace["_is_affirmative_approval_response"],
        audit_logger_instance=agent_namespace["audit_logger"],
    )
    execution_guard_node = agent_namespace["make_execution_guard_node"](
        with_working_memory_fn=agent_namespace["_with_working_memory"],
        validate_pending_execution_snapshot_fn=agent_namespace["_validate_pending_execution_snapshot"],
        audit_logger_instance=agent_namespace["audit_logger"],
    )
    reviewer_node = agent_namespace["make_reviewer_node"](
        with_working_memory_fn=agent_namespace["_with_working_memory"],
        get_current_plan_step_fn=agent_namespace["_get_current_plan_step"],
        looks_like_step_failure_fn=agent_namespace["_looks_like_step_failure"],
        update_plan_step_fn=agent_namespace["_update_plan_step"],
        step_requires_approval_fn=agent_namespace["_step_requires_approval"],
        build_approval_reason_fn=agent_namespace["_build_approval_reason"],
        classify_error_fn=agent_namespace["classify_error"],
        serialize_classified_error_fn=agent_namespace["serialize_classified_error"],
        plan_to_todos_fn=agent_namespace["plan_to_todos"],
        should_enable_todos_fn=agent_namespace["should_enable_todos"],
        build_todo_state_from_plan_fn=agent_namespace["build_todo_state_from_plan"],
        save_session_todo_state_fn=agent_namespace["_save_session_todo_state"],
        clear_session_todo_state_fn=agent_namespace["_clear_session_todo_state"],
        audit_logger_instance=agent_namespace["audit_logger"],
    )
    finalizer_node = agent_namespace["make_finalizer_node"](
        with_working_memory_fn=agent_namespace["_with_working_memory"],
        clear_session_todo_state_fn=agent_namespace["_clear_session_todo_state"],
        audit_logger_instance=agent_namespace["audit_logger"],
    )

    def fast_agent_node(state: AgentState, config: RunnableConfig) -> dict:
        return agent_namespace["_run_react_agent_node"](
            state,
            config,
            llm,
            llm_with_tools,
            actual_tools,
            route_mode="fast",
        )

    def slow_agent_node(state: AgentState, config: RunnableConfig) -> dict:
        return agent_namespace["_run_react_agent_node"](
            state,
            config,
            llm,
            llm_with_tools,
            actual_tools,
            route_mode="slow",
        )

    return agent_namespace["compile_agent_workflow"](
        fast_tool_node=fast_tool_node,
        slow_tool_node=slow_tool_node,
        router_node=router_node,
        planner_node=planner_node,
        finalizer_node=finalizer_node,
        approval_gate_node=approval_gate_node,
        execution_guard_node=execution_guard_node,
        fast_agent_node=fast_agent_node,
        slow_agent_node=slow_agent_node,
        reviewer_node=reviewer_node,
        route_after_router_fn=agent_namespace["_route_after_router"],
        route_after_fast_agent_fn=route_after_fast_agent,
        route_after_planner_fn=agent_namespace["_route_after_planner"],
        route_after_approval_gate_fn=agent_namespace["_route_after_approval_gate"],
        route_after_execution_guard_fn=route_after_execution_guard,
        route_after_slow_agent_fn=agent_namespace["_route_after_slow_agent"],
        route_after_reviewer_fn=agent_namespace["_route_after_reviewer"],
        checkpointer=checkpointer,
    )
