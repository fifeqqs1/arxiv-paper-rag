from langchain_core.tools import BaseTool

from ..context import AgentState
from ..planning import (
    looks_like_file_write_request,
    step_matches_shell_action,
    step_matches_test_action,
)


FAST_PATH_EXCLUDED_TOOL_NAMES = {"update_todo_list"}
FAST_PATH_PROJECT_ANALYSIS_TOOL_NAMES = {
    "read_project_file",
    "search_project_code",
    "show_git_diff",
    "tavily_web_search",
    "calculator",
    "get_current_time",
    "get_system_model_info",
}
SLOW_DESTRUCTIVE_TOOL_NAMES = {
    "edit_project_file",
    "write_project_file",
    "apply_project_patch",
    "write_office_file",
    "run_project_tests",
    "run_project_command",
    "execute_office_shell",
}
PLAN_MODE_BLOCKED_TOOL_NAMES = set(SLOW_DESTRUCTIVE_TOOL_NAMES)
AUTO_MODE_BLOCKED_TOOL_NAMES = {"execute_office_shell"}
AUTONOMOUS_PROJECT_READ_TOOL_NAMES = {
    "read_project_file",
    "search_project_code",
    "show_git_diff",
    "update_todo_list",
}
AUTONOMOUS_PROJECT_WRITE_TOOL_NAMES = {
    "read_project_file",
    "search_project_code",
    "show_git_diff",
    "edit_project_file",
    "write_project_file",
    "apply_project_patch",
    "run_project_tests",
    "run_project_command",
    "update_todo_list",
}


def _looks_like_fast_project_analysis(
    state: AgentState,
    *,
    latest_user_query: str,
) -> bool:
    complexity = str(state.get("complexity", "") or "").strip().lower()
    if complexity != "read_only_analysis":
        return False

    current_project_path = str(state.get("current_project_path", "") or "").strip()
    if current_project_path:
        return True

    query = str(latest_user_query or "").strip().lower()
    if not query:
        return False

    if any(marker in query for marker in ("/", "\\", ".py", ".ts", ".tsx", ".js", ".jsx", ".md", ".json", ".yaml", ".yml")):
        return True

    return False


def select_tools_for_fast_route(
    state: AgentState,
    all_tools: list[BaseTool],
    *,
    latest_user_query: str,
) -> list[BaseTool]:
    baseline_tools = [
        tool for tool in all_tools
        if getattr(tool, "name", "") not in FAST_PATH_EXCLUDED_TOOL_NAMES
    ]

    if not _looks_like_fast_project_analysis(state, latest_user_query=latest_user_query):
        return baseline_tools

    selected_tools = [
        tool for tool in baseline_tools
        if getattr(tool, "name", "") in FAST_PATH_PROJECT_ANALYSIS_TOOL_NAMES
    ]
    return selected_tools or baseline_tools


def select_tools_for_autonomous_slow(
    state: AgentState,
    all_tools: list[BaseTool],
    *,
    latest_user_query: str,
) -> list[BaseTool]:
    current_project_path = str(state.get("current_project_path", "") or "").strip()
    if not current_project_path:
        return all_tools

    task_text = str(state.get("goal", "") or latest_user_query or "").strip()
    if not task_text:
        return all_tools

    is_project_write_task = (
        str(state.get("risk_level", "") or "").strip().lower() == "high"
        or looks_like_file_write_request(task_text)
        or step_matches_test_action(task_text)
        or step_matches_shell_action(task_text)
    )
    allowed_tool_names = (
        AUTONOMOUS_PROJECT_WRITE_TOOL_NAMES
        if is_project_write_task
        else AUTONOMOUS_PROJECT_READ_TOOL_NAMES
    )
    selected_tools = [
        tool for tool in all_tools
        if getattr(tool, "name", "") in allowed_tool_names
    ]
    return selected_tools or all_tools


def apply_permission_mode_to_tools(
    tools: list[BaseTool],
    *,
    permission_mode: str,
) -> list[BaseTool]:
    normalized_mode = str(permission_mode or "").strip().lower()
    if normalized_mode == "plan":
        blocked = PLAN_MODE_BLOCKED_TOOL_NAMES
    elif normalized_mode == "auto":
        blocked = AUTO_MODE_BLOCKED_TOOL_NAMES
    else:
        return tools
    filtered = [
        tool for tool in tools
        if getattr(tool, "name", "") not in blocked
    ]
    return filtered or tools


def destructive_tool_calls(tool_calls: list[dict] | None) -> list[dict]:
    return [
        dict(tool_call)
        for tool_call in (tool_calls or [])
        if isinstance(tool_call, dict)
        and str(tool_call.get("name") or "").strip() in SLOW_DESTRUCTIVE_TOOL_NAMES
    ]


def build_pending_tool_approval_reason(tool_calls: list[dict] | None) -> str:
    destructive_calls = destructive_tool_calls(tool_calls)
    if not destructive_calls:
        return ""
    tool_names: list[str] = []
    for tool_call in destructive_calls:
        tool_name = str(tool_call.get("name") or "").strip()
        if tool_name and tool_name not in tool_names:
            tool_names.append(tool_name)
    if not tool_names:
        return ""
    return f"本轮待执行的高风险工具调用：{', '.join(tool_names)}"
