from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from ..approval import (
    build_approval_message as _build_approval_message,
    build_approval_reason as _build_approval_reason,
    build_permission_mode_message as _build_permission_mode_message,
    is_affirmative_approval_response as _is_affirmative_approval_response,
    is_negative_approval_response as _is_negative_approval_response,
    parse_permission_mode_response as _parse_permission_mode_response,
    route_after_approval_gate as _route_after_approval_gate,
    step_requires_approval as _step_requires_approval,
)
from ..config import MEMORY_DIR
from ..context import AgentState, build_working_memory_snapshot, trim_context_messages
from ..error_policy import classify_error, looks_like_explicit_failure_text, serialize_classified_error
from ..logger import audit_logger
from ..memory import (
    DEFAULT_LONG_TERM_SCOPE,
    USER_PROFILE_MEMORY_TYPE,
    build_memory_record,
    get_async_memory_writer,
    get_memory_store,
)
from ..memory_policy import LONG_TERM_MEMORY_PROMPT_LIMIT, SESSION_MEMORY_PROMPT_LIMIT, MemoryPromptCache
from ..planning import (
    build_default_replan_input as _build_default_replan_input,
    build_plan_with_llm as _build_plan_with_llm,
    build_rule_execution_plan as _build_rule_execution_plan,
    enforce_slow_step_tool_scope as _enforce_slow_step_tool_scope,
    get_current_plan_step as _get_current_plan_step,
    looks_like_step_failure as _looks_like_step_failure,
    route_after_planner as _route_after_planner,
    route_after_reviewer as _route_after_reviewer,
    route_after_slow_agent as _route_after_slow_agent,
    select_tools_for_current_step as _select_tools_for_current_step,
    update_plan_step as _update_plan_step,
)
from ..prompt_builder import CONTEXT_SUMMARY_TIMEOUT_SECONDS, build_react_prompt_bundle
from ..provider import get_provider
from ..routing import (
    ROUTE_CLASSIFIER_CONFIDENCE_THRESHOLD,
    build_route_decision as _build_route_decision,
    extract_passthrough_payload as _extract_passthrough_payload,
    extract_passthrough_text as _extract_passthrough_text,
    get_latest_user_query as _get_latest_user_query,
    get_route_classifier_model as _get_route_classifier_model,
    infer_tavily_topic as _infer_tavily_topic,
    normalize_tavily_tool_calls as _normalize_tavily_tool_calls,
    route_after_router as _route_after_router,
    should_direct_route_to_arxiv_rag as _should_direct_route_to_arxiv_rag,
)
from ..runtime.execution_guard import (
    build_pending_execution_snapshot as _build_pending_execution_snapshot,
    validate_pending_execution_snapshot as _validate_pending_execution_snapshot,
)
from ..runtime.todos import build_todo_state_from_plan, plan_to_todos, should_enable_todos
from ..runtime_context import set_active_thread_id
from ..runtime_graph import (
    compile_agent_workflow,
    make_approval_gate_node,
    make_execution_guard_node,
    make_finalizer_node,
    make_planner_node,
    make_reviewer_node,
    make_router_node,
)
from ..skill_loader import load_dynamic_skills
from ..storage.runtime import get_conversation_writer, get_session_repository
from ..tools.builtins import BUILTIN_TOOLS
from ..tools.web import arxiv_rag_ask
from .app import create_agent_app
from .memory_bridge import (
    build_long_term_memory_prompt as _build_long_term_memory_prompt_impl,
    build_session_memory_prompt as _build_session_memory_prompt_impl,
    load_long_term_profile_content as _load_long_term_profile_content_impl,
    load_session_memory_records as _load_session_memory_records_impl,
    load_session_project_path as _load_session_project_path_impl,
    schedule_long_term_memory_capture as _schedule_long_term_memory_capture_impl,
    summarize_discarded_context as _summarize_discarded_context_impl,
    sync_session_memory_from_query as _sync_session_memory_from_query_impl,
    with_working_memory as _with_working_memory_impl,
)
from .react_node import ReactNodeDependencies, run_react_agent_node
from .recovery import (
    RESPONSE_KIND_FINAL_ANSWER,
    RESPONSE_KIND_STEP_RESULT,
    STEP_OUTCOME_FAILURE,
    STEP_OUTCOME_SUCCESS_CANDIDATE,
    _annotate_ai_message,
    _complete_autonomous_todos,
    _prepare_recent_tool_messages,
)
from .tool_policy import (
    AUTO_MODE_BLOCKED_TOOL_NAMES,
    FAST_PATH_EXCLUDED_TOOL_NAMES,
    apply_permission_mode_to_tools as _apply_permission_mode_to_tools,
    build_pending_tool_approval_reason as _build_pending_tool_approval_reason,
    destructive_tool_calls as _destructive_tool_calls,
    select_tools_for_fast_route as _select_tools_for_fast_route,
    select_tools_for_autonomous_slow as _select_tools_for_autonomous_slow,
)


_memory_prompt_cache = MemoryPromptCache()


def _sync_session_memory_from_query(query: str | None, thread_id: str) -> dict:
    return _sync_session_memory_from_query_impl(
        query,
        thread_id,
        get_memory_store_fn=get_memory_store,
        build_memory_record_fn=build_memory_record,
    )


def _load_session_memory_records(thread_id: str, *, limit: int = SESSION_MEMORY_PROMPT_LIMIT) -> list[dict]:
    return _load_session_memory_records_impl(
        thread_id,
        get_memory_store_fn=get_memory_store,
        limit=limit,
    )


def _build_session_memory_prompt(thread_id: str, *, limit: int = SESSION_MEMORY_PROMPT_LIMIT) -> str:
    return _build_session_memory_prompt_impl(
        thread_id,
        get_memory_store_fn=get_memory_store,
        limit=limit,
        prompt_cache=_memory_prompt_cache,
    )


def _load_session_project_path(thread_id: str) -> str:
    return _load_session_project_path_impl(
        thread_id,
        get_memory_store_fn=get_memory_store,
    )


def _schedule_long_term_memory_capture(query: str | None) -> None:
    _schedule_long_term_memory_capture_impl(
        query,
        get_async_memory_writer_fn=get_async_memory_writer,
        build_memory_record_fn=build_memory_record,
        default_long_term_scope=DEFAULT_LONG_TERM_SCOPE,
    )


def _load_long_term_profile_content() -> str:
    return _load_long_term_profile_content_impl(
        get_memory_store_fn=get_memory_store,
        memory_dir=MEMORY_DIR,
        default_long_term_scope=DEFAULT_LONG_TERM_SCOPE,
        user_profile_memory_type=USER_PROFILE_MEMORY_TYPE,
    )


def _build_long_term_memory_prompt(query: str | None) -> str:
    return _build_long_term_memory_prompt_impl(
        query,
        get_memory_store_fn=get_memory_store,
        memory_dir=MEMORY_DIR,
        default_long_term_scope=DEFAULT_LONG_TERM_SCOPE,
        user_profile_memory_type=USER_PROFILE_MEMORY_TYPE,
        long_term_memory_prompt_limit=LONG_TERM_MEMORY_PROMPT_LIMIT,
        prompt_cache=_memory_prompt_cache,
    )


def _summarize_discarded_context(
    llm,
    current_summary: str,
    discarded_msgs: list,
    thread_id: str,
    state: dict | None = None,
    timeout_seconds: float = CONTEXT_SUMMARY_TIMEOUT_SECONDS,
) -> str:
    return _summarize_discarded_context_impl(
        llm,
        current_summary,
        discarded_msgs,
        thread_id,
        state=state,
        audit_logger_instance=audit_logger,
        timeout_seconds=timeout_seconds,
    )


def _with_working_memory(state: AgentState, updates: dict) -> dict:
    return _with_working_memory_impl(
        state,
        updates,
        build_working_memory_snapshot_fn=build_working_memory_snapshot,
    )


def _save_session_todo_state(thread_id: str, todo_state: dict) -> None:
    get_session_repository().save_session_todo_state(thread_id, todo_state)


def _clear_session_todo_state(thread_id: str) -> None:
    get_session_repository().clear_session_todo_state(thread_id)


def _run_react_agent_node(
    state: AgentState,
    config: RunnableConfig,
    llm,
    llm_with_tools,
    all_tools: list[BaseTool],
    route_mode: str,
) -> dict:
    deps = ReactNodeDependencies(
        set_active_thread_id_fn=set_active_thread_id,
        prepare_recent_tool_messages_fn=_prepare_recent_tool_messages,
        build_session_memory_prompt_fn=_build_session_memory_prompt,
        should_enable_todos_fn=should_enable_todos,
        build_todo_state_from_plan_fn=build_todo_state_from_plan,
        save_session_todo_state_fn=_save_session_todo_state,
        clear_session_todo_state_fn=_clear_session_todo_state,
        audit_logger_instance=audit_logger,
        extract_passthrough_text_fn=_extract_passthrough_text,
        annotate_ai_message_fn=_annotate_ai_message,
        with_working_memory_fn=_with_working_memory,
        is_affirmative_approval_response_fn=_is_affirmative_approval_response,
        get_latest_user_query_fn=_get_latest_user_query,
        get_current_plan_step_fn=_get_current_plan_step,
        select_tools_for_current_step_fn=_select_tools_for_current_step,
        select_tools_for_fast_route_fn=_select_tools_for_fast_route,
        apply_permission_mode_to_tools_fn=_apply_permission_mode_to_tools,
        select_tools_for_autonomous_slow_fn=_select_tools_for_autonomous_slow,
        should_direct_route_to_arxiv_rag_fn=_should_direct_route_to_arxiv_rag,
        arxiv_rag_tool=arxiv_rag_ask,
        extract_passthrough_payload_fn=_extract_passthrough_payload,
        trim_context_messages_fn=trim_context_messages,
        summarize_discarded_context_fn=_summarize_discarded_context,
        conversation_writer=get_conversation_writer(),
        build_long_term_memory_prompt_fn=_build_long_term_memory_prompt,
        build_react_prompt_bundle_fn=build_react_prompt_bundle,
        classify_error_fn=classify_error,
        serialize_classified_error_fn=serialize_classified_error,
        normalize_tavily_tool_calls_fn=_normalize_tavily_tool_calls,
        enforce_slow_step_tool_scope_fn=_enforce_slow_step_tool_scope,
        destructive_tool_calls_fn=_destructive_tool_calls,
        build_pending_execution_snapshot_fn=_build_pending_execution_snapshot,
        build_pending_tool_approval_reason_fn=_build_pending_tool_approval_reason,
        looks_like_explicit_failure_text_fn=looks_like_explicit_failure_text,
        complete_autonomous_todos_fn=_complete_autonomous_todos,
        fast_path_excluded_tool_names=FAST_PATH_EXCLUDED_TOOL_NAMES,
        auto_mode_blocked_tool_names=AUTO_MODE_BLOCKED_TOOL_NAMES,
        response_kind_final_answer=RESPONSE_KIND_FINAL_ANSWER,
        response_kind_step_result=RESPONSE_KIND_STEP_RESULT,
        step_outcome_failure=STEP_OUTCOME_FAILURE,
        step_outcome_success_candidate=STEP_OUTCOME_SUCCESS_CANDIDATE,
        session_memory_prompt_limit=SESSION_MEMORY_PROMPT_LIMIT,
        context_summary_timeout_seconds=CONTEXT_SUMMARY_TIMEOUT_SECONDS,
    )
    return run_react_agent_node(
        state,
        config,
        llm,
        llm_with_tools,
        all_tools,
        route_mode,
        deps=deps,
    )


__all__ = [
    "AgentState",
    "RESPONSE_KIND_FINAL_ANSWER",
    "RESPONSE_KIND_STEP_RESULT",
    "STEP_OUTCOME_FAILURE",
    "STEP_OUTCOME_SUCCESS_CANDIDATE",
    "_build_long_term_memory_prompt",
    "_build_route_decision",
    "_build_session_memory_prompt",
    "_complete_autonomous_todos",
    "_infer_tavily_topic",
    "_load_long_term_profile_content",
    "_load_session_memory_records",
    "_load_session_project_path",
    "_prepare_recent_tool_messages",
    "_run_react_agent_node",
    "_schedule_long_term_memory_capture",
    "_summarize_discarded_context",
    "_sync_session_memory_from_query",
    "_with_working_memory",
    "create_agent_app",
]
