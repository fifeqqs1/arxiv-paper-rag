from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool

from ..error_policy import classify_error, serialize_classified_error
from ..logger import audit_logger
from .rules import (
    infer_step_intent,
    looks_like_file_write_request,
    step_matches_shell_action,
    step_matches_skill_usage,
    step_matches_task_creation,
    step_matches_task_deletion,
    step_matches_task_modification,
    step_matches_test_action,
)


SAFE_READONLY_TOOL_NAMES = {
    "list_office_files",
    "read_office_file",
    "tavily_web_search",
    "arxiv_rag_ask",
    "calculator",
    "get_current_time",
    "get_system_model_info",
    "list_scheduled_tasks",
    "read_project_file",
    "search_project_code",
    "show_git_diff",
}

PROJECT_READONLY_TOOL_NAMES = {
    "read_project_file",
    "search_project_code",
    "show_git_diff",
}

OFFICE_READONLY_TOOL_NAMES = {
    "list_office_files",
    "read_office_file",
}

PAPER_RESEARCH_TOOL_NAMES = {
    "arxiv_rag_ask",
}


def _ensure_primary_file_write_tool(
    selected_tools: list[BaseTool],
    all_tools: list[BaseTool],
    *,
    current_project_path: str,
) -> list[BaseTool]:
    selected_names = {getattr(tool, "name", "") for tool in selected_tools}
    if {"write_project_file", "write_office_file"} & selected_names:
        return selected_tools

    preferred_name = "write_project_file" if str(current_project_path or "").strip() else "write_office_file"
    for tool in all_tools:
        if getattr(tool, "name", "") == preferred_name:
            selected_tools.append(tool)
            return selected_tools

    fallback_name = "write_office_file" if preferred_name == "write_project_file" else "write_project_file"
    for tool in all_tools:
        if getattr(tool, "name", "") == fallback_name:
            selected_tools.append(tool)
            return selected_tools

    return selected_tools


def _ensure_execution_tools(
    selected_tools: list[BaseTool],
    all_tools: list[BaseTool],
    *,
    intent: str,
) -> list[BaseTool]:
    selected_names = {getattr(tool, "name", "") for tool in selected_tools}
    if intent == "shell_execute" and {"run_project_command", "run_project_tests"} & selected_names:
        return selected_tools
    if intent == "test_verify" and "run_project_tests" in selected_names:
        return selected_tools

    preferred_names = (
        ("run_project_command", "run_project_tests")
        if intent == "shell_execute"
        else ("run_project_tests",)
    )
    for preferred_name in preferred_names:
        for tool in all_tools:
            if getattr(tool, "name", "") == preferred_name:
                selected_tools.append(tool)
                return selected_tools
    return selected_tools


def _looks_like_project_analysis_step(step_text: str, step_intent: str, current_project_path: str) -> bool:
    if step_intent not in {"analyze", "read", "summarize", "report"}:
        return False
    path_markers = ("/", "\\", ".py", ".ts", ".tsx", ".js", ".jsx", ".md", ".json", ".yaml", ".yml")
    has_path_context = bool(str(current_project_path or "").strip()) or any(marker in step_text for marker in path_markers)
    return has_path_context


def select_tools_for_current_step(
    step: dict | None,
    all_tools: list[BaseTool],
    *,
    current_project_path: str = "",
) -> list[BaseTool]:
    if step is None:
        return all_tools

    step_text = step.get("description", "")
    step_intent = str(step.get("intent") or infer_step_intent(step_text)).strip().lower()
    prefer_project_read_tools = _looks_like_project_analysis_step(step_text, step_intent, current_project_path)
    allowed_tools: list[BaseTool] = []

    for tool in all_tools:
        tool_name = getattr(tool, "name", "")

        if tool_name == "update_todo_list":
            allowed_tools.append(tool)
            continue

        if step_intent == "paper_research":
            if tool_name in PAPER_RESEARCH_TOOL_NAMES:
                allowed_tools.append(tool)
            continue

        if step_intent in {"analyze", "read", "summarize", "report"}:
            if prefer_project_read_tools:
                if tool_name in PROJECT_READONLY_TOOL_NAMES:
                    allowed_tools.append(tool)
                continue
            if tool_name in SAFE_READONLY_TOOL_NAMES:
                allowed_tools.append(tool)
            continue

        if step_intent in {"code_edit", "file_write"} and tool_name in PROJECT_READONLY_TOOL_NAMES:
            allowed_tools.append(tool)
            continue

        if step_intent in {"code_edit", "file_write"} and tool_name in {"write_office_file", "edit_project_file", "write_project_file", "apply_project_patch"}:
            allowed_tools.append(tool)
            continue

        if step_intent == "shell_execute" and tool_name in {"execute_office_shell", "run_project_tests", "run_project_command"}:
            allowed_tools.append(tool)
            continue

        if step_intent == "test_verify" and tool_name in PROJECT_READONLY_TOOL_NAMES:
            allowed_tools.append(tool)
            continue

        if step_intent == "test_verify" and tool_name in {"run_project_tests", "run_project_command"}:
            allowed_tools.append(tool)
            continue

        if tool_name == "edit_project_file" and looks_like_file_write_request(step_text):
            allowed_tools.append(tool)
            continue

        if tool_name == "write_project_file" and looks_like_file_write_request(step_text):
            allowed_tools.append(tool)
            continue

        if tool_name == "write_office_file" and looks_like_file_write_request(step_text):
            allowed_tools.append(tool)
            continue

        if tool_name == "apply_project_patch" and looks_like_file_write_request(step_text):
            allowed_tools.append(tool)
            continue

        if tool_name == "execute_office_shell" and step_matches_shell_action(step_text):
            allowed_tools.append(tool)
            continue

        if tool_name in {"run_project_tests", "run_project_command"} and (step_matches_shell_action(step_text) or step_matches_test_action(step_text)):
            allowed_tools.append(tool)
            continue

        if tool_name == "schedule_task" and step_matches_task_creation(step_text):
            allowed_tools.append(tool)
            continue

        if tool_name == "modify_scheduled_task" and step_matches_task_modification(step_text):
            allowed_tools.append(tool)
            continue

        if tool_name == "delete_scheduled_task" and step_matches_task_deletion(step_text):
            allowed_tools.append(tool)
            continue

        if step_matches_skill_usage(step_text) and tool_name not in SAFE_READONLY_TOOL_NAMES:
            allowed_tools.append(tool)

    if step_intent == "file_write":
        allowed_tools = _ensure_primary_file_write_tool(
            allowed_tools,
            all_tools,
            current_project_path=current_project_path,
        )
    elif step_intent in {"shell_execute", "test_verify"}:
        allowed_tools = _ensure_execution_tools(
            allowed_tools,
            all_tools,
            intent=step_intent,
        )

    return allowed_tools


def enforce_slow_step_tool_scope(
    response: AIMessage,
    current_step: dict | None,
    allowed_tool_names: set[str],
    thread_id: str,
) -> AIMessage:
    if current_step is None or not response.tool_calls:
        return response

    disallowed_calls = [
        tool_call for tool_call in response.tool_calls
        if tool_call.get("name") not in allowed_tool_names
    ]
    if not disallowed_calls:
        return response

    blocked_names = ", ".join(tool_call.get("name", "<unknown>") for tool_call in disallowed_calls)
    allowed_names = ", ".join(sorted(allowed_tool_names)) if allowed_tool_names else "无需工具"
    audit_logger.log_event(
        thread_id=thread_id,
        event="system_action",
        content=(
            f"blocked out-of-scope tool call(s) for slow step {current_step.get('step', '?')}: "
            f"{blocked_names} | allowed={allowed_names}"
        ),
    )
    return AIMessage(
        content=(
            "执行失败：系统拦截了越界工具调用。\n"
            f"当前步骤：{current_step.get('description', '未知步骤')}\n"
            f"允许工具：{allowed_names}\n"
            f"禁止工具：{blocked_names}\n"
            "请只完成当前步骤，不要提前执行后续高风险操作。"
        ),
        additional_kwargs={
            "mortyclaw_error": serialize_classified_error(
                classify_error(
                    message=(
                        "越界工具调用。"
                        f" 当前步骤：{current_step.get('description', '')}"
                        f" 禁止工具：{blocked_names}"
                    ),
                )
            )
        },
    )
