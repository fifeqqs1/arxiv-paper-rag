import threading
from typing import Any

from ..base import mortyclaw_tool
from ..office import execute_office_shell, list_office_files, read_office_file, write_office_file
from ..project_tools import (
    apply_project_patch,
    edit_project_file,
    read_project_file,
    run_project_command,
    run_project_tests,
    search_project_code,
    show_git_diff,
    write_project_file,
)
from ..summarize import summarize_content
from ..web import arxiv_rag_ask, tavily_web_search
from ...config import MEMORY_DIR, TASKS_FILE
from ...logger import build_log_file_path
from ...memory import (
    DEFAULT_LONG_TERM_SCOPE,
    USER_PROFILE_MEMORY_ID,
    USER_PROFILE_MEMORY_TYPE,
    build_memory_record,
    get_memory_store,
)
from ...runtime.todos import build_todo_state, merge_tool_written_todos, normalize_todos
from ...runtime_context import get_active_thread_id
from ...storage.runtime import get_conversation_repository, get_session_repository, get_task_repository
from .profile import save_user_profile_impl
from .sessions import (
    ensure_session_record_impl,
    get_active_session_thread_id_impl,
    load_session_todo_state_impl,
    search_sessions_impl,
)
from .tasks import (
    delete_scheduled_task_impl,
    ensure_task_store_bootstrapped_impl,
    list_scheduled_tasks_impl,
    modify_scheduled_task_impl,
    schedule_task_impl,
)
from .system import calculator_impl, get_current_time_impl, get_system_model_info_impl
from .todo import TodoInputItem, UpdateTodoListArgs, update_todo_list_impl


tasks_lock = threading.Lock()
PROFILE_PATH = f"{MEMORY_DIR}/user_profile.md"


def ensure_task_store_bootstrapped() -> None:
    ensure_task_store_bootstrapped_impl(
        get_task_repository_fn=get_task_repository,
        file_path=TASKS_FILE,
    )


def get_active_session_thread_id() -> str:
    return get_active_session_thread_id_impl(
        get_active_thread_id_fn=get_active_thread_id,
    )


def ensure_session_record(thread_id: str) -> None:
    ensure_session_record_impl(
        thread_id,
        get_session_repository_fn=get_session_repository,
        build_log_file_path_fn=build_log_file_path,
    )


def load_session_todo_state(thread_id: str) -> dict[str, Any]:
    return load_session_todo_state_impl(
        thread_id,
        get_session_repository_fn=get_session_repository,
    )


@mortyclaw_tool
def get_system_model_info() -> str:
    """
    获取当前 MortyClaw 正在运行的底层大模型（LLM）型号和提供商信息。
    当用户询问“你是基于什么模型”、“你的底层大模型是什么”、“你是GPT还是GLM”、“现在用的什么模型”等身份问题时，调用此工具。
    """
    return get_system_model_info_impl()


@mortyclaw_tool
def get_current_time() -> str:
    """
    获取当前的系统时间和日期。
    当用户询问“现在几点”、“今天星期几”、“今天几号”等与当前时间相关的问题时，调用此工具。
    """
    return get_current_time_impl()


@mortyclaw_tool
def calculator(expression: str) -> str:
    """
    一个简单的数学计算器。
    用于计算基础的数学表达式，例如: '3 * 5' 或 '100 / 4'。
    注意：参数 expression 必须是一个合法的 Python 数学表达式字符串。
    """
    return calculator_impl(expression)


@mortyclaw_tool
def save_user_profile(new_content: str) -> str:
    """
    更新用户的全局显性记忆档案。
    当你发现用户的偏好发生改变，或者有新的重要事实需要记录时：
    1.请先调用 read_user_profile 获取当前的完整档案。
    2.在你的上下文中，将新信息融入档案，并删去冲突或过时的旧信息。
    3.将修改后的一整篇完整 Markdown 文本作为 new_content 参数传入此工具。
    注意：此操作将完全覆盖旧文件！请确保传入的是完整的最新档案。
    """
    return save_user_profile_impl(
        new_content,
        get_memory_store_fn=get_memory_store,
        build_memory_record_fn=build_memory_record,
        memory_dir=MEMORY_DIR,
        profile_path=PROFILE_PATH,
        default_long_term_scope=DEFAULT_LONG_TERM_SCOPE,
        user_profile_memory_id=USER_PROFILE_MEMORY_ID,
        user_profile_memory_type=USER_PROFILE_MEMORY_TYPE,
    )


@mortyclaw_tool
def schedule_task(target_time: str, description: str, repeat: str = None, repeat_count: int = None) -> str:
    """
    为一个未来的任务设定闹钟或提醒。
    参数 target_time 必须是严格的格式："YYYY-MM-DD HH:MM:SS"（请先调用 get_current_time 获取当前时间，并在其基础上推算）。
    参数 description 是需要执行的动作或要说的话。
    """
    return schedule_task_impl(
        target_time=target_time,
        description=description,
        repeat=repeat,
        repeat_count=repeat_count,
        ensure_task_store_bootstrapped_fn=ensure_task_store_bootstrapped,
        get_active_session_thread_id_fn=get_active_session_thread_id,
        ensure_session_record_fn=ensure_session_record,
        get_task_repository_fn=get_task_repository,
        tasks_file=TASKS_FILE,
    )


@mortyclaw_tool
def list_scheduled_tasks() -> str:
    """
    查看当前所有待处理的定时任务列表。
    当用户询问“我都有哪些任务”、“查一下闹钟”、“刚才定了什么”时调用此工具。
    """
    return list_scheduled_tasks_impl(
        ensure_task_store_bootstrapped_fn=ensure_task_store_bootstrapped,
        get_active_session_thread_id_fn=get_active_session_thread_id,
        get_task_repository_fn=get_task_repository,
    )


@mortyclaw_tool
def delete_scheduled_task(task_id: str) -> str:
    """
    根据任务 ID 取消或删除一个定时任务。
    """
    return delete_scheduled_task_impl(
        task_id=task_id,
        ensure_task_store_bootstrapped_fn=ensure_task_store_bootstrapped,
        get_active_session_thread_id_fn=get_active_session_thread_id,
        get_task_repository_fn=get_task_repository,
        tasks_file=TASKS_FILE,
    )


@mortyclaw_tool
def modify_scheduled_task(task_id: str, new_time: str = None, new_description: str = None) -> str:
    """
    修改现有定时任务的时间或内容。
    """
    return modify_scheduled_task_impl(
        task_id=task_id,
        new_time=new_time,
        new_description=new_description,
        ensure_task_store_bootstrapped_fn=ensure_task_store_bootstrapped,
        get_active_session_thread_id_fn=get_active_session_thread_id,
        get_task_repository_fn=get_task_repository,
        tasks_file=TASKS_FILE,
    )


@mortyclaw_tool
def search_sessions(
    query: str = "",
    role_filter: str = "",
    limit: int = 3,
    include_current: bool = False,
    include_tool_results: bool = True,
) -> str:
    """
    搜索 MortyClaw 的历史会话、旧对话和以前执行过的工具结果。
    """
    return search_sessions_impl(
        query=query,
        role_filter=role_filter,
        limit=limit,
        include_current=include_current,
        include_tool_results=include_tool_results,
        current_thread_id=get_active_session_thread_id(),
        get_conversation_repository_fn=get_conversation_repository,
    )


@mortyclaw_tool(args_schema=UpdateTodoListArgs)
def update_todo_list(items: list[dict[str, Any]], reason: str = "") -> str:
    """
    更新当前复杂任务的 Todo checklist。
    """
    thread_id = get_active_session_thread_id()
    session_repo = get_session_repository()
    todo_state = load_session_todo_state(thread_id)
    return update_todo_list_impl(
        items=items,
        reason=reason,
        thread_id=thread_id,
        session_repo=session_repo,
        todo_state=todo_state,
        build_todo_state_fn=build_todo_state,
        merge_tool_written_todos_fn=merge_tool_written_todos,
        normalize_todos_fn=normalize_todos,
    )


from .registry import BUILTIN_TOOLS  # noqa: E402


__all__ = [
    "BUILTIN_TOOLS",
    "MEMORY_DIR",
    "PROFILE_PATH",
    "TASKS_FILE",
    "TodoInputItem",
    "UpdateTodoListArgs",
    "apply_project_patch",
    "arxiv_rag_ask",
    "calculator",
    "delete_scheduled_task",
    "edit_project_file",
    "ensure_session_record",
    "ensure_task_store_bootstrapped",
    "execute_office_shell",
    "get_active_session_thread_id",
    "get_active_thread_id",
    "get_conversation_repository",
    "get_current_time",
    "get_memory_store",
    "get_session_repository",
    "get_system_model_info",
    "get_task_repository",
    "list_office_files",
    "list_scheduled_tasks",
    "load_session_todo_state",
    "modify_scheduled_task",
    "read_office_file",
    "read_project_file",
    "run_project_command",
    "run_project_tests",
    "save_user_profile",
    "schedule_task",
    "search_project_code",
    "search_sessions",
    "show_git_diff",
    "summarize_content",
    "tavily_web_search",
    "tasks_lock",
    "update_todo_list",
    "write_office_file",
    "write_project_file",
]
