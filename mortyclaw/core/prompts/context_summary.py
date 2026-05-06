import threading

from langchain_core.messages import HumanMessage

from ..context.handoff import (
    build_fallback_handoff_summary,
    build_handoff_summary_prompt,
    merge_handoff_summary,
)


CONTEXT_SUMMARY_TIMEOUT_SECONDS = 8


def build_context_summary_prompt(
    current_summary: str,
    discarded_msgs: list,
    state: dict | None = None,
) -> str:
    return build_handoff_summary_prompt(
        current_summary=current_summary,
        discarded_msgs=discarded_msgs,
        state=state,
    )


def build_fallback_context_summary(
    current_summary: str,
    discarded_msgs: list,
    state: dict | None = None,
) -> str:
    return build_fallback_handoff_summary(
        current_summary=current_summary,
        discarded_msgs=discarded_msgs,
        state=state,
    )


def summarize_discarded_context(
    llm,
    current_summary: str,
    discarded_msgs: list,
    thread_id: str,
    *,
    state: dict | None = None,
    audit_logger_instance,
    timeout_seconds: float = CONTEXT_SUMMARY_TIMEOUT_SECONDS,
) -> str:
    summary_prompt = build_context_summary_prompt(
        current_summary,
        discarded_msgs,
        state=state,
    )

    result_holder: dict[str, str] = {}
    error_holder: dict[str, Exception] = {}

    def worker() -> None:
        try:
            response = llm.invoke([HumanMessage(content=summary_prompt)], config={"callbacks": []})
            result_holder["summary"] = getattr(response, "content", "") or ""
        except Exception as exc:  # pragma: no cover
            error_holder["error"] = exc

    audit_logger_instance.log_event(
        thread_id=thread_id,
        event="system_action",
        content=(
            f"context summarization started for {len(discarded_msgs)} discarded messages "
            f"with timeout={timeout_seconds}s"
        ),
    )

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout_seconds)

    if thread.is_alive():
        audit_logger_instance.log_event(
            thread_id=thread_id,
            event="system_action",
            content="context summarization timed out; fallback summary applied",
        )
        return build_fallback_context_summary(current_summary, discarded_msgs, state=state)

    if "error" in error_holder:
        audit_logger_instance.log_event(
            thread_id=thread_id,
            event="system_action",
            content=f"context summarization failed; fallback summary applied: {error_holder['error']}",
        )
        return build_fallback_context_summary(current_summary, discarded_msgs, state=state)

    summary_text = result_holder.get("summary", "").strip()
    if not summary_text:
        audit_logger_instance.log_event(
            thread_id=thread_id,
            event="system_action",
            content="context summarization returned empty content; fallback summary applied",
        )
        return build_fallback_context_summary(current_summary, discarded_msgs, state=state)

    merged_handoff = merge_handoff_summary(
        current_summary=current_summary,
        discarded_msgs=discarded_msgs,
        state=state,
        llm_output_text=summary_text,
    )

    audit_logger_instance.log_event(
        thread_id=thread_id,
        event="system_action",
        content="context summarization completed successfully",
    )
    return merged_handoff
