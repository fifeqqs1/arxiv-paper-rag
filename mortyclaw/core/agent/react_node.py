from __future__ import annotations

import json
from dataclasses import dataclass

from langchain_core.messages import AIMessage, RemoveMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from ..context import AgentState
from .recovery import _extract_classified_error


CONTEXT_TRIM_TRIGGER_TOKENS = 300000
CONTEXT_TRIM_KEEP_TOKENS = 220000
CONTEXT_OVERFLOW_KEEP_TOKENS = 120000
CONTEXT_NON_MESSAGE_RESERVE_TOKENS = 80000


def _current_turn_id(config: RunnableConfig) -> str:
    return config.get("configurable", {}).get("turn_id", "turn-default")


def _resolve_llm_model_name(llm) -> str:
    for attr in ("model_name", "model"):
        value = str(getattr(llm, attr, "") or "").strip()
        if value:
            return value
    return ""


def _should_use_direct_arxiv_shortcut(
    *,
    active_route: str,
    route_source: str,
    effective_user_query: str,
    should_direct_route_to_arxiv_rag_fn,
) -> bool:
    if active_route != "fast":
        return False
    if route_source not in {"arxiv_direct", "pure_paper_task"}:
        return False
    if not effective_user_query:
        return False
    return bool(should_direct_route_to_arxiv_rag_fn(effective_user_query))


def _extract_tool_payload(message) -> dict | None:
    if getattr(message, "type", "") != "tool":
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
    output_hints = (
        "打印",
        "输出",
        "结果",
        "示例",
        "stdout",
        "print",
        "run result",
        "show output",
    )
    return any(hint in lowered for hint in output_hints)


def _build_successful_execution_step_response(
    raw_messages: list,
    current_plan_step: dict | None,
    *,
    deps: ReactNodeDependencies,
):
    if not current_plan_step:
        return None
    current_intent = str(current_plan_step.get("intent") or "")
    if current_intent not in {"shell_execute", "test_verify"}:
        return None

    step_text = str(current_plan_step.get("description") or "")
    trailing_tool_messages = []
    for message in reversed(raw_messages):
        if getattr(message, "type", "") != "tool":
            break
        trailing_tool_messages.append(message)

    for message in trailing_tool_messages:
        tool_name = str(getattr(message, "name", "") or "")
        if tool_name not in {"run_project_command", "run_project_tests"}:
            continue
        payload = _extract_tool_payload(message)
        if not payload or not payload.get("ok", False):
            continue

        stdout = str(payload.get("stdout", "") or "").strip()
        command = str(payload.get("command", "") or "").strip()
        exit_code = payload.get("exit_code", 0)
        if exit_code not in {0, None}:
            continue

        if current_intent == "shell_execute":
            if not stdout or not _step_expects_runtime_output(step_text):
                continue
            success_summary = stdout[:400]
        else:
            success_summary = stdout[:400] if stdout else (
                f"验证完成：{command or tool_name} 执行通过。"
            )

        return deps.annotate_ai_message_fn(
            AIMessage(content=success_summary),
            mortyclaw_step_outcome=deps.step_outcome_success_candidate,
            mortyclaw_response_kind=deps.response_kind_step_result,
        )

    return None


@dataclass(frozen=True)
class ReactNodeDependencies:
    set_active_thread_id_fn: object
    prepare_recent_tool_messages_fn: object
    build_session_memory_prompt_fn: object
    should_enable_todos_fn: object
    build_todo_state_from_plan_fn: object
    save_session_todo_state_fn: object
    clear_session_todo_state_fn: object
    audit_logger_instance: object
    extract_passthrough_text_fn: object
    annotate_ai_message_fn: object
    with_working_memory_fn: object
    is_affirmative_approval_response_fn: object
    get_latest_user_query_fn: object
    get_current_plan_step_fn: object
    select_tools_for_current_step_fn: object
    select_tools_for_fast_route_fn: object
    apply_permission_mode_to_tools_fn: object
    select_tools_for_autonomous_slow_fn: object
    should_direct_route_to_arxiv_rag_fn: object
    arxiv_rag_tool: object
    extract_passthrough_payload_fn: object
    trim_context_messages_fn: object
    summarize_discarded_context_fn: object
    conversation_writer: object
    build_long_term_memory_prompt_fn: object
    build_react_prompt_bundle_fn: object
    classify_error_fn: object
    serialize_classified_error_fn: object
    normalize_tavily_tool_calls_fn: object
    enforce_slow_step_tool_scope_fn: object
    destructive_tool_calls_fn: object
    build_pending_execution_snapshot_fn: object
    build_pending_tool_approval_reason_fn: object
    looks_like_explicit_failure_text_fn: object
    complete_autonomous_todos_fn: object
    fast_path_excluded_tool_names: set[str]
    auto_mode_blocked_tool_names: set[str]
    response_kind_final_answer: str
    response_kind_step_result: str
    step_outcome_failure: str
    step_outcome_success_candidate: str
    session_memory_prompt_limit: int
    context_summary_timeout_seconds: float


def run_react_agent_node(
    state: AgentState,
    config: RunnableConfig,
    llm,
    llm_with_tools,
    all_tools: list[BaseTool],
    route_mode: str,
    *,
    deps: ReactNodeDependencies,
) -> dict:
    thread_id = config.get("configurable", {}).get("thread_id", "system_default")
    turn_id = _current_turn_id(config)
    deps.set_active_thread_id_fn(thread_id)
    raw_messages, preprocessing_updates = deps.prepare_recent_tool_messages_fn(
        state,
        thread_id=thread_id,
        turn_id=turn_id,
    )
    working_state = dict(state)
    for key, value in preprocessing_updates.items():
        if key != "messages":
            working_state[key] = value
    active_route = working_state.get("route", route_mode)
    formatted_session_prompt = deps.build_session_memory_prompt_fn(
        thread_id,
        limit=deps.session_memory_prompt_limit,
    )

    if deps.should_enable_todos_fn(
        active_route,
        working_state.get("plan", []),
        execution_mode=str(working_state.get("slow_execution_mode", "") or ""),
        todos=working_state.get("todos", []),
    ) and working_state.get("todos"):
        todo_state = deps.build_todo_state_from_plan_fn(
            working_state.get("plan", []),
            int(working_state.get("current_step_index", 0) or 0),
            revision=int(working_state.get("todo_revision", 1) or 1),
            last_event="active",
        )
        todo_state["items"] = list(working_state.get("todos", []) or [])
        todo_state["plan_snapshot"] = [dict(step) for step in (working_state.get("plan", []) or []) if isinstance(step, dict)]
        deps.save_session_todo_state_fn(thread_id, todo_state)
    else:
        deps.clear_session_todo_state_fn(thread_id)

    if raw_messages:
        recent_tool_msgs = []
        for msg in reversed(raw_messages):
            if msg.type == "tool":
                recent_tool_msgs.append(msg)
            else:
                break
        for msg in reversed(recent_tool_msgs):
            deps.audit_logger_instance.log_event(
                thread_id=thread_id,
                event="tool_result",
                tool=msg.name,
                result_summary=msg.content[:200],
            )

    passthrough_text = deps.extract_passthrough_text_fn(raw_messages[-1]) if raw_messages else None
    if passthrough_text:
        final_message = deps.annotate_ai_message_fn(
            AIMessage(content=passthrough_text),
            mortyclaw_response_kind=deps.response_kind_final_answer,
        )
        deps.audit_logger_instance.log_event(
            thread_id=thread_id,
            event="ai_message",
            content=passthrough_text,
        )
        return deps.with_working_memory_fn(state, {
            "route": active_route,
            "final_answer": passthrough_text,
            "run_status": "done",
            "messages": [final_message],
        })

    if (
        active_route == "slow"
        and working_state.get("approval_granted")
        and working_state.get("pending_tool_calls")
    ):
        pending_tool_calls = [
            dict(tool_call)
            for tool_call in (working_state.get("pending_tool_calls", []) or [])
            if isinstance(tool_call, dict)
        ]
        if pending_tool_calls:
            deps.audit_logger_instance.log_event(
                thread_id=thread_id,
                event="system_action",
                content=f"slow agent resumed {len(pending_tool_calls)} approved tool call(s)",
            )
            resumed_message = AIMessage(content="", tool_calls=pending_tool_calls)
            return deps.with_working_memory_fn(state, {
                "route": active_route,
                "approval_granted": False,
                "pending_tool_calls": [],
                "pending_execution_snapshot": {},
                "run_status": "running",
                "messages": [resumed_message],
                **preprocessing_updates,
            })

    slow_execution_mode = str(working_state.get("slow_execution_mode", "") or "").strip().lower()
    permission_mode = str(working_state.get("permission_mode", "") or "").strip().lower()
    latest_user_query = deps.get_latest_user_query_fn(raw_messages)
    current_plan_step = (
        deps.get_current_plan_step_fn(working_state)
        if active_route == "slow" and slow_execution_mode != "autonomous"
        else None
    )

    execution_success_response = _build_successful_execution_step_response(
        raw_messages,
        current_plan_step,
        deps=deps,
    )
    if active_route == "slow" and execution_success_response is not None:
        deps.audit_logger_instance.log_event(
            thread_id=thread_id,
            event="ai_message",
            content=str(execution_success_response.content or ""),
        )
        return deps.with_working_memory_fn(state, {
            "route": active_route,
            "final_answer": "",
            "run_status": "review_pending",
            "todo_needs_announcement": False,
            "messages": [execution_success_response],
            **preprocessing_updates,
        })

    effective_user_query = latest_user_query
    active_llm_with_tools = llm_with_tools
    allowed_tool_names: set[str] = {getattr(tool, "name", "") for tool in all_tools}
    if active_route == "fast":
        fast_tools = deps.select_tools_for_fast_route_fn(
            working_state,
            all_tools,
            latest_user_query=latest_user_query,
        )
        allowed_tool_names = {getattr(tool, "name", "") for tool in fast_tools}
        if fast_tools:
            active_llm_with_tools = llm.bind_tools(fast_tools)

    if active_route == "slow" and current_plan_step is not None:
        effective_user_query = current_plan_step["description"]
        allowed_step_tools = deps.select_tools_for_current_step_fn(
            current_plan_step,
            all_tools,
            current_project_path=str(working_state.get("current_project_path", "") or ""),
        )
        allowed_step_tools = deps.apply_permission_mode_to_tools_fn(
            allowed_step_tools,
            permission_mode=permission_mode,
        )
        allowed_tool_names = {getattr(tool, "name", "") for tool in allowed_step_tools}
        if allowed_step_tools:
            active_llm_with_tools = llm.bind_tools(allowed_step_tools)
        else:
            active_llm_with_tools = llm if hasattr(llm, "invoke") else llm_with_tools
    elif active_route == "slow" and slow_execution_mode == "autonomous":
        autonomous_tools = deps.select_tools_for_autonomous_slow_fn(
            working_state,
            all_tools,
            latest_user_query=latest_user_query,
        )
        if str(working_state.get("current_project_path", "") or "").strip():
            autonomous_tools = [
                tool for tool in autonomous_tools
                if getattr(tool, "name", "") != "write_office_file"
            ] or autonomous_tools
        autonomous_tools = deps.apply_permission_mode_to_tools_fn(
            autonomous_tools,
            permission_mode=permission_mode,
        )
        allowed_tool_names = {getattr(tool, "name", "") for tool in autonomous_tools}
        if autonomous_tools:
            active_llm_with_tools = llm.bind_tools(autonomous_tools)
        else:
            active_llm_with_tools = llm if hasattr(llm, "invoke") else llm_with_tools
    elif active_route == "slow" and state.get("goal") and deps.is_affirmative_approval_response_fn(latest_user_query):
        effective_user_query = state["goal"]

    route_source = str(working_state.get("route_source", "") or "")
    if _should_use_direct_arxiv_shortcut(
        active_route=active_route,
        route_source=route_source,
        effective_user_query=effective_user_query,
        should_direct_route_to_arxiv_rag_fn=deps.should_direct_route_to_arxiv_rag_fn,
    ):
        deps.audit_logger_instance.log_event(
            thread_id=thread_id,
            event="tool_call",
            tool="arxiv_rag_ask",
            args={"query": effective_user_query, "session_id": thread_id},
        )
        tool_result = deps.arxiv_rag_tool.invoke({"query": effective_user_query, "session_id": thread_id})
        deps.audit_logger_instance.log_event(
            thread_id=thread_id,
            event="tool_result",
            tool="arxiv_rag_ask",
            result_summary=tool_result[:200],
        )

        passthrough_payload = deps.extract_passthrough_payload_fn(tool_result)
        direct_reply = tool_result
        if passthrough_payload is not None:
            display_text = passthrough_payload.get("display_text") or passthrough_payload.get("answer")
            if isinstance(display_text, str) and display_text.strip():
                direct_reply = display_text

        deps.audit_logger_instance.log_event(
            thread_id=thread_id,
            event="ai_message",
            content=direct_reply,
        )
        final_message = deps.annotate_ai_message_fn(
            AIMessage(content=direct_reply),
            mortyclaw_response_kind=deps.response_kind_final_answer,
        )
        return deps.with_working_memory_fn(state, {
            "route": active_route,
            "final_answer": direct_reply,
            "run_status": "done",
            "messages": [final_message],
        })

    current_summary = state.get("summary", "")
    model_name = _resolve_llm_model_name(llm)
    final_msgs, discarded_msgs = deps.trim_context_messages_fn(
        raw_messages,
        trigger_tokens=CONTEXT_TRIM_TRIGGER_TOKENS,
        keep_tokens=CONTEXT_TRIM_KEEP_TOKENS,
        reserve_tokens=CONTEXT_NON_MESSAGE_RESERVE_TOKENS,
        model_name=model_name,
    )
    state_updates = {"route": active_route}
    for key, value in preprocessing_updates.items():
        if key != "messages":
            state_updates[key] = value
    if preprocessing_updates.get("messages"):
        state_updates["messages"] = list(preprocessing_updates["messages"])

    if discarded_msgs:
        deps.audit_logger_instance.log_event(
            thread_id=thread_id,
            event="system_action",
            content="context trim triggered; summarizing discarded messages",
        )
        active_summary = deps.summarize_discarded_context_fn(
            llm,
            current_summary,
            discarded_msgs,
            thread_id,
            state=working_state,
            timeout_seconds=deps.context_summary_timeout_seconds,
        )

        state_updates["summary"] = active_summary
        state_updates.setdefault("messages", [])
        state_updates["messages"].extend([RemoveMessage(id=m.id) for m in discarded_msgs if m.id])
        deps.conversation_writer.record_summary(
            thread_id=thread_id,
            summary=active_summary,
            summary_type="structured_handoff",
            messages=discarded_msgs,
            metadata={"discarded_message_count": len(discarded_msgs)},
        )
    else:
        active_summary = current_summary

    long_term_prompt = deps.build_long_term_memory_prompt_fn(latest_user_query)
    sys_prompt, llm_messages = deps.build_react_prompt_bundle_fn(
        final_msgs,
        active_route,
        working_state,
        active_summary=active_summary,
        session_prompt=formatted_session_prompt or "",
        long_term_prompt=long_term_prompt,
        current_plan_step=current_plan_step,
        include_approved_goal_context=bool(
            active_route == "slow"
            and working_state.get("goal")
            and deps.is_affirmative_approval_response_fn(latest_user_query)
        ),
    )

    msgs_for_llm = [SystemMessage(content=sys_prompt)] + llm_messages
    for message in msgs_for_llm:
        if isinstance(message.content, str):
            message.content = message.content.encode("utf-8", "ignore").decode("utf-8")

    deps.audit_logger_instance.log_event(
        thread_id=thread_id,
        event="llm_input",
        message_count=len(msgs_for_llm),
    )

    response = None
    for attempt in range(2):
        try:
            response = active_llm_with_tools.invoke(msgs_for_llm)
            break
        except Exception as exc:
            classified = deps.classify_error_fn(exc=exc, state=working_state)
            if classified.kind.value == "context_overflow" and attempt == 0:
                forced_msgs, forced_discarded = deps.trim_context_messages_fn(
                    raw_messages,
                    trigger_tokens=1,
                    keep_tokens=CONTEXT_OVERFLOW_KEEP_TOKENS,
                    reserve_tokens=CONTEXT_NON_MESSAGE_RESERVE_TOKENS,
                    model_name=model_name,
                )
                if forced_discarded:
                    active_summary = deps.summarize_discarded_context_fn(
                        llm,
                        active_summary,
                        forced_discarded,
                        thread_id,
                        state=working_state,
                        timeout_seconds=deps.context_summary_timeout_seconds,
                    )
                    state_updates["summary"] = active_summary
                    state_updates.setdefault("messages", [])
                    state_updates["messages"].extend([RemoveMessage(id=m.id) for m in forced_discarded if m.id])
                    sys_prompt, llm_messages = deps.build_react_prompt_bundle_fn(
                        forced_msgs,
                        active_route,
                        working_state,
                        active_summary=active_summary,
                        session_prompt=formatted_session_prompt or "",
                        long_term_prompt=long_term_prompt,
                        current_plan_step=current_plan_step,
                        include_approved_goal_context=bool(
                            active_route == "slow"
                            and working_state.get("goal")
                            and deps.is_affirmative_approval_response_fn(latest_user_query)
                        ),
                    )
                    msgs_for_llm = [SystemMessage(content=sys_prompt)] + llm_messages
                    continue
            if classified.retryable and attempt + 1 < classified.retry_policy.max_attempts:
                continue
            response = AIMessage(
                content=classified.user_visible_hint or str(exc),
                additional_kwargs={
                    "mortyclaw_error": deps.serialize_classified_error_fn(classified),
                    "mortyclaw_step_outcome": deps.step_outcome_failure,
                },
            )
            break
    if response is None:
        classified = deps.classify_error_fn(message="", state=working_state)
        response = AIMessage(
            content=classified.user_visible_hint,
            additional_kwargs={
                "mortyclaw_error": deps.serialize_classified_error_fn(classified),
                "mortyclaw_step_outcome": deps.step_outcome_failure,
            },
        )

    response = deps.normalize_tavily_tool_calls_fn(response, effective_user_query, thread_id)
    response = deps.enforce_slow_step_tool_scope_fn(response, current_plan_step, allowed_tool_names, thread_id)
    if active_route == "fast":
        fast_escalation_reason = ""
        destructive_calls = deps.destructive_tool_calls_fn(response.tool_calls)
        if destructive_calls:
            blocked_names = ", ".join(
                sorted({str(tool_call.get("name") or "").strip() for tool_call in destructive_calls if tool_call.get("name")})
            ) or "高风险工具"
            fast_escalation_reason = (
                "fast path discovered high-risk tool intent and escalated to planner: "
                f"{blocked_names}"
            )
        else:
            metadata = getattr(response, "additional_kwargs", {}) or {}
            requested_escalation = str(metadata.get("mortyclaw_fast_escalate", "") or "").strip().lower()
            if requested_escalation == "planner":
                fast_escalation_reason = str(metadata.get("mortyclaw_fast_escalate_reason") or "").strip() or (
                    "fast path explicitly requested planner escalation"
                )

        if fast_escalation_reason:
            deps.audit_logger_instance.log_event(
                thread_id=thread_id,
                event="system_action",
                content=fast_escalation_reason,
            )
            state_updates.update({
                "route": "slow",
                "planner_required": True,
                "route_source": "fast_escalation",
                "route_reason": fast_escalation_reason,
                "goal": working_state.get("goal", "") or latest_user_query or "",
                "complexity": "high_risk" if destructive_calls else "complex",
                "risk_level": "high" if destructive_calls else str(working_state.get("risk_level", "medium") or "medium"),
                "pending_tool_calls": [],
                "pending_execution_snapshot": {},
                "pending_approval": False,
                "approval_granted": False,
                "approval_prompted": False,
                "approval_reason": "",
                "final_answer": "",
                "run_status": "planner_requested",
            })
            return deps.with_working_memory_fn(state, state_updates)
    if active_route == "slow" and response.tool_calls:
        destructive_calls = deps.destructive_tool_calls_fn(response.tool_calls)
        if permission_mode == "plan" and destructive_calls:
            blocked_names = ", ".join(
                sorted({str(tool_call.get("name") or "").strip() for tool_call in destructive_calls if tool_call.get("name")})
            ) or "高风险工具"
            response = deps.annotate_ai_message_fn(
                AIMessage(content=(
                    "当前任务处于 `plan` 只读模式，但执行过程中检测到需要修改文件、运行测试或执行命令的操作，"
                    f"已终止本次任务。涉及工具：{blocked_names}。请改用 `ask` 或 `auto` 后重试。"
                )),
                mortyclaw_response_kind=deps.response_kind_final_answer,
            )
            state_updates["pending_tool_calls"] = []
            state_updates["pending_execution_snapshot"] = {}
            state_updates["pending_approval"] = False
            state_updates["approval_granted"] = False
            state_updates["approval_prompted"] = False
            state_updates["approval_reason"] = ""
            state_updates["run_status"] = "failed"
            state_updates["final_answer"] = str(response.content or "")
        elif permission_mode == "auto":
            forbidden_auto_calls = [
                tool_call for tool_call in (response.tool_calls or [])
                if str(tool_call.get("name") or "").strip() in deps.auto_mode_blocked_tool_names
            ]
            if forbidden_auto_calls:
                blocked_names = ", ".join(
                    sorted({str(tool_call.get("name") or "").strip() for tool_call in forbidden_auto_calls if tool_call.get("name")})
                ) or "受限工具"
                response = deps.annotate_ai_message_fn(
                    AIMessage(content=(
                        "当前任务处于 `auto` 模式，但检测到被禁止的原始 shell/batch 操作，"
                        f"已终止本次任务。涉及工具：{blocked_names}。"
                    )),
                    mortyclaw_response_kind=deps.response_kind_final_answer,
                )
                state_updates["pending_tool_calls"] = []
                state_updates["pending_execution_snapshot"] = {}
                state_updates["pending_approval"] = False
                state_updates["approval_granted"] = False
                state_updates["approval_prompted"] = False
                state_updates["approval_reason"] = ""
                state_updates["run_status"] = "failed"
                state_updates["final_answer"] = str(response.content or "")
    approval_staged = False
    if (
        active_route == "slow"
        and slow_execution_mode == "autonomous"
        and response.tool_calls
        and permission_mode not in {"plan", "auto"}
        and not working_state.get("approval_granted", False)
    ):
        destructive_calls = deps.destructive_tool_calls_fn(response.tool_calls)
        if destructive_calls:
            approval_staged = True
            state_updates["pending_approval"] = True
            state_updates["approval_granted"] = False
            state_updates["approval_prompted"] = False
            state_updates["approval_reason"] = deps.build_pending_tool_approval_reason_fn(response.tool_calls)
            state_updates["pending_tool_calls"] = [dict(tool_call) for tool_call in response.tool_calls]
            state_updates["pending_execution_snapshot"] = deps.build_pending_execution_snapshot_fn(
                working_state | state_updates,
                response.tool_calls,
            )
            state_updates["run_status"] = "awaiting_step_approval"
            response = AIMessage(content="")
            deps.audit_logger_instance.log_event(
                thread_id=thread_id,
                event="system_action",
                content=(
                    "slow autonomous agent staged destructive tool call batch for approval: "
                    f"{state_updates['approval_reason']}"
                ),
            )
    if not approval_staged and not response.tool_calls and not str(response.content or "").strip():
        classified = deps.classify_error_fn(message="", state=working_state)
        response = AIMessage(
            content=classified.user_visible_hint,
            additional_kwargs={
                "mortyclaw_error": deps.serialize_classified_error_fn(classified),
                "mortyclaw_step_outcome": deps.step_outcome_failure,
            },
        )

    if response.tool_calls:
        state_updates["run_status"] = "running"
        state_updates["pending_tool_calls"] = []
        state_updates["pending_execution_snapshot"] = {}
        for tool_call in response.tool_calls:
            deps.audit_logger_instance.log_event(
                thread_id=thread_id,
                event="tool_call",
                tool=tool_call["name"],
                args=tool_call["args"],
            )
    elif response.content:
        error_payload = _extract_classified_error(response)
        if active_route == "slow":
            state_updates["final_answer"] = ""
            explicit_failure = bool(error_payload is not None or deps.looks_like_explicit_failure_text_fn(response.content))
            if explicit_failure:
                if error_payload is None:
                    classified = deps.classify_error_fn(message=str(response.content or ""), state=working_state)
                    response = deps.annotate_ai_message_fn(
                        response,
                        mortyclaw_error=deps.serialize_classified_error_fn(classified),
                    )
                    error_payload = classified
                response = deps.annotate_ai_message_fn(
                    response,
                    mortyclaw_step_outcome=deps.step_outcome_failure,
                    mortyclaw_response_kind=deps.response_kind_step_result,
                )
                state_updates["last_error"] = str(response.content or "")[:200]
                state_updates["last_error_kind"] = error_payload.kind.value
                state_updates["last_recovery_action"] = error_payload.recovery_action.value
                if slow_execution_mode == "autonomous":
                    recovery_action = str(error_payload.recovery_action.value or "")
                    retry_count = int(working_state.get("retry_count", 0) or 0)
                    max_retries = int(working_state.get("max_retries", 2) or 2)
                    state_updates["replan_reason"] = ""
                    if recovery_action in {"retry", "compress_and_retry"}:
                        if retry_count < max_retries:
                            state_updates["retry_count"] = retry_count + 1
                            state_updates["run_status"] = "retrying"
                        else:
                            state_updates["retry_count"] = 0
                            state_updates["replan_reason"] = error_payload.user_visible_hint or str(response.content or "")[:200]
                            state_updates["run_status"] = "replan_requested"
                    elif recovery_action == "replan":
                        state_updates["retry_count"] = 0
                        state_updates["replan_reason"] = error_payload.user_visible_hint or str(response.content or "")[:200]
                        state_updates["run_status"] = "replan_requested"
                    else:
                        state_updates["retry_count"] = 0
                        state_updates["final_answer"] = error_payload.user_visible_hint or str(response.content or "")
                        state_updates["run_status"] = "failed"
                else:
                    state_updates["run_status"] = "review_pending"
            else:
                if slow_execution_mode == "autonomous":
                    response = deps.annotate_ai_message_fn(
                        response,
                        mortyclaw_response_kind=deps.response_kind_final_answer,
                    )
                    state_updates.update(deps.complete_autonomous_todos_fn(working_state | state_updates))
                    state_updates["final_answer"] = response.content
                    state_updates["pending_approval"] = False
                    state_updates["approval_granted"] = False
                    state_updates["approval_prompted"] = False
                    state_updates["approval_reason"] = ""
                    state_updates["pending_tool_calls"] = []
                    state_updates["pending_execution_snapshot"] = {}
                    state_updates["retry_count"] = 0
                    state_updates["last_error"] = ""
                    state_updates["last_error_kind"] = ""
                    state_updates["last_recovery_action"] = ""
                    state_updates["replan_reason"] = ""
                    state_updates["run_status"] = "done"
                else:
                    response = deps.annotate_ai_message_fn(
                        response,
                        mortyclaw_step_outcome=deps.step_outcome_success_candidate,
                        mortyclaw_response_kind=deps.response_kind_step_result,
                    )
                    state_updates["run_status"] = "review_pending"
        else:
            response = deps.annotate_ai_message_fn(
                response,
                mortyclaw_response_kind=deps.response_kind_final_answer,
            )
            state_updates["final_answer"] = response.content
            state_updates["run_status"] = "done"
            if error_payload is not None:
                state_updates["last_error"] = str(response.content or "")[:200]
                state_updates["last_error_kind"] = error_payload.kind.value
                state_updates["last_recovery_action"] = error_payload.recovery_action.value
        deps.audit_logger_instance.log_event(
            thread_id=thread_id,
            event="ai_message",
            content=response.content,
        )

    if active_route == "slow":
        state_updates["todo_needs_announcement"] = False

    if "messages" not in state_updates:
        state_updates["messages"] = []
    state_updates["messages"].append(response)

    return deps.with_working_memory_fn(state, state_updates)
