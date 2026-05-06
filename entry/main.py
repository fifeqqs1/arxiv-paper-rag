import os
import sys
import time
import asyncio
import random
import json
import re
import uuid
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from prompt_toolkit import PromptSession, print_formatted_text
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.styles import Style
from prompt_toolkit.application import get_app, get_app_or_none, run_in_terminal

from mortyclaw.core.agent import create_agent_app
from mortyclaw.core.config import DB_PATH
from mortyclaw.core.bus import task_queue
from mortyclaw.core.logger import build_log_file_path
from mortyclaw.core.maintenance import reset_thread_state
from mortyclaw.core.runtime_context import set_active_thread_id
from mortyclaw.core.storage.runtime import get_conversation_writer, get_session_repository, get_task_repository
from mortyclaw.core.runtime.tool_results import prepare_tool_messages_for_budget

UI_CYAN = "\033[38;5;51m"
UI_PURPLE = "\033[38;5;141m"
UI_SILVER = "\033[38;5;250m"
UI_DIM = "\033[2m"
UI_BOLD = "\033[1m"
UI_RESET = "\033[0m"
UI_WHITE = "\033[37m"
ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*m")
PANEL_WIDTH = 96


def _ansi_strikethrough(text: str) -> str:
    return f"\033[9m{text}\033[0m"


def _effective_plan_render_state(node_data: dict | None) -> dict:
    if not isinstance(node_data, dict):
        return {}
    working_memory = node_data.get("working_memory")
    merged: dict = dict(working_memory) if isinstance(working_memory, dict) else {}
    merged.update(node_data)
    return merged


def _normalize_plan_items(node_data: dict | None) -> list[dict]:
    node_data = _effective_plan_render_state(node_data)
    todos = node_data.get("todos") or node_data.get("active_todos")
    if isinstance(todos, list) and len(todos) >= 2:
        items = []
        for index, item in enumerate(todos, start=1):
            if not isinstance(item, dict):
                continue
            items.append({
                "step": index,
                "content": str(item.get("content", "")).strip(),
                "status": str(item.get("status", "pending")).strip().lower() or "pending",
            })
        if items:
            return items

    plan = node_data.get("plan")
    if not isinstance(plan, list) or len(plan) < 2:
        return []

    current_step_index = int(node_data.get("current_step_index", 0) or 0)
    items = []
    for index, step in enumerate(plan, start=1):
        if not isinstance(step, dict):
            continue
        step_status = str(step.get("status", "pending")).strip().lower() or "pending"
        if step_status == "completed" or index - 1 < current_step_index:
            status = "completed"
        elif index - 1 == current_step_index:
            status = "in_progress"
        elif step_status == "cancelled":
            status = "cancelled"
        else:
            status = "pending"
        items.append({
            "step": int(step.get("step") or index),
            "content": str(step.get("description", "")).strip(),
            "status": status,
        })
    return items


def _plan_render_signature(node_data: dict | None) -> str:
    render_state = _effective_plan_render_state(node_data)
    items = _normalize_plan_items(render_state)
    if not items:
        return ""
    pending_approval = bool(render_state.get("pending_approval", False))
    run_status = str(render_state.get("run_status", "") or "")
    serialized = json.dumps(
        {
            "items": items,
            "pending_approval": pending_approval,
            "run_status": run_status,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return serialized


def _render_current_plan_block(node_data: dict | None, *, frame: str = "⠋") -> str:
    lines, _spinner_line_index = _build_current_plan_lines(node_data, frame=frame)
    return "\n".join(lines)


def _build_current_plan_lines(node_data: dict | None, *, frame: str = "⠋") -> tuple[list[str], int | None]:
    render_state = _effective_plan_render_state(node_data)
    items = _normalize_plan_items(render_state)
    if not items:
        return [], None

    lines = [f"  {UI_CYAN}● Current Plan{UI_RESET}"]
    spinner_line_index = None
    for item in items:
        content = item["content"] or "(未命名步骤)"
        step_label = f"{item['step']}. {content}"
        status = item.get("status", "pending")
        if status == "completed":
            lines.append(f"    {UI_PURPLE}[x]{UI_RESET} {UI_DIM}{_ansi_strikethrough(step_label)}{UI_RESET}")
        elif status == "in_progress":
            lines.append(f"    {UI_CYAN}{frame}{UI_RESET} {UI_WHITE}{step_label}{UI_RESET}")
            spinner_line_index = len(lines) - 1
        elif status == "cancelled":
            lines.append(f"    {UI_PURPLE}[~]{UI_RESET} {UI_DIM}{step_label}{UI_RESET}")
        else:
            lines.append(f"    {UI_SILVER}[ ]{UI_RESET} {UI_SILVER}{step_label}{UI_RESET}")

    if render_state.get("pending_approval"):
        lines.append(f"    {UI_PURPLE}↳ waiting approval{UI_RESET}")
    return lines, spinner_line_index


def _message_additional_kwargs(message) -> dict:
    data = getattr(message, "additional_kwargs", {}) or {}
    return dict(data) if isinstance(data, dict) else {}


def _message_response_kind(message) -> str:
    return str(_message_additional_kwargs(message).get("mortyclaw_response_kind", "") or "").strip().lower()


def _message_step_outcome(message) -> str:
    return str(_message_additional_kwargs(message).get("mortyclaw_step_outcome", "") or "").strip().lower()


def _format_node_message_output(text: str) -> str:
    lines = str(text or "").strip().split('\n')
    if not lines:
        return ""
    formatted_out = f"  \033[38;5;141m❯\033[0m \033[38;5;250m{lines[0]}"
    for line in lines[1:]:
        formatted_out += f"\n    {line}"
    formatted_out += "\033[0m"
    return formatted_out


def _render_stream_node_message(node_name: str, node_data: dict | None) -> str:
    if not isinstance(node_data, dict):
        return ""
    node_messages = node_data.get("messages")
    if not isinstance(node_messages, list) or not node_messages:
        return ""

    last_msg = node_messages[-1]
    if getattr(last_msg, "tool_calls", None):
        return ""

    content = str(getattr(last_msg, "content", "") or "").strip()
    if not content:
        return ""

    if node_name == "slow_agent":
        if _message_response_kind(last_msg) == "step_result":
            step_number = int(node_data.get("current_step_index", 0) or 0) + 1
            status_line = (
                f"步骤 {step_number} 遇到异常，正在恢复..."
                if _message_step_outcome(last_msg) == "failure"
                else f"步骤 {step_number} 结果已生成，正在审查..."
            )
            return f"  {UI_PURPLE}✦{UI_RESET} {UI_SILVER}{status_line}{UI_RESET}"
        if _message_response_kind(last_msg) != "final_answer":
            return ""

    if node_name in {"fast_agent", "approval_gate", "finalizer"} or _message_response_kind(last_msg) == "final_answer":
        return _format_node_message_output(content)

    return ""


def _should_render_message_below_live_plan(node_name: str, node_data: dict | None) -> bool:
    if not isinstance(node_data, dict):
        return False
    node_messages = node_data.get("messages")
    if not isinstance(node_messages, list) or not node_messages:
        return False
    if node_name == "approval_gate":
        return True
    last_msg = node_messages[-1]
    if node_name in {"slow_agent", "finalizer"} and _message_response_kind(last_msg) == "final_answer":
        return bool(_plan_render_signature(node_data))
    return False


def _should_pre_render_live_plan(
    *,
    rendered_line_count: int,
    has_live_content: bool,
    keep_live_block_above: bool,
) -> bool:
    return bool(keep_live_block_above and has_live_content and rendered_line_count <= 0)


def _is_slow_step_result_message(node_name: str, node_data: dict | None) -> bool:
    if node_name != "slow_agent" or not isinstance(node_data, dict):
        return False
    node_messages = node_data.get("messages")
    if not isinstance(node_messages, list) or not node_messages:
        return False
    last_msg = node_messages[-1]
    if getattr(last_msg, "tool_calls", None):
        return False
    return _message_response_kind(last_msg) == "step_result"


def _should_suspend_prompt_for_live_output(app_instance=None) -> bool:
    instance = app_instance if app_instance is not None else get_app_or_none()
    return bool(instance is not None and getattr(instance, "is_running", False))


async def _run_live_output_operation(func, *, app_instance=None, run_in_terminal_fn=run_in_terminal):
    if _should_suspend_prompt_for_live_output(app_instance):
        return await run_in_terminal_fn(func, render_cli_done=False)
    return func()


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def _build_prompt_session(*, bottom_toolbar):
    custom_style = Style.from_dict({
        "bottom-toolbar": "bg:default fg:default noreverse",
    })
    return PromptSession(
        bottom_toolbar=bottom_toolbar,
        style=custom_style,
        erase_when_done=True,
        reserve_space_for_menu=0,
    )


def type_line(text: str, delay: float = 0.008):
    for ch in text:
        print(ch, end='', flush=True)
        time.sleep(delay)
    print()

def print_banner():
    clear_screen()

    CYAN = '\033[38;5;51m'
    PURPLE = '\033[38;5;141m'
    SILVER = '\033[38;5;250m'
    DIM = '\033[2m'
    BOLD = '\033[1m'
    RESET = '\033[0m'
    WHITE = '\033[37m'

    logo = f"""{CYAN}{BOLD}
███╗   ███╗ ██████╗ ██████╗ ████████╗██╗   ██╗
████╗ ████║██╔═══██╗██╔══██╗╚══██╔══╝╚██╗ ██╔╝
██╔████╔██║██║   ██║██████╔╝   ██║    ╚████╔╝
██║╚██╔╝██║██║   ██║██╔══██╗   ██║     ╚██╔╝
██║ ╚═╝ ██║╚██████╔╝██║  ██║   ██║      ██║
╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝      ╚═╝

 ██████╗██╗      █████╗ ██╗    ██╗
██╔════╝██║     ██╔══██╗██║    ██║
██║     ██║     ███████║██║ █╗ ██║
██║     ██║     ██╔══██║██║███╗██║
╚██████╗███████╗██║  ██║╚███╔███╔╝
 ╚═════╝╚══════╝╚═╝  ╚═╝ ╚══╝╚══╝
{RESET}"""

    sub_title = f"{WHITE}{BOLD} 😈 Welcome to the {PURPLE}{BOLD}MortyClaw{RESET}{WHITE}{BOLD} !  {RESET}"

    quotes = [
        "It works on my machine.",
        "It compiles! Ship it.",
        "Git commit, push, pray.",
        "There's no place like 127.0.0.1.",
        "sudo make me a sandwich.",
        "Works fine in dev.",
        "May the source be with you.",
        "Ctrl+C, Ctrl+V, Deploy.",
        "Hello, World."
    ]
    quote = random.choice(quotes)
    meta = f" {SILVER}✦{RESET} {CYAN}{quote}{RESET}"

    tip = (
        f"{PURPLE} ✦ {RESET}"
        f"{SILVER}{PURPLE}{BOLD}MortyClaw{RESET} 已完成启动。输入命令开始，输入 {PURPLE}/exit{RESET}{SILVER} 退出。{RESET}\n"
    )

    print(logo)
    print(sub_title)
    print() 
    time.sleep(0.12)
    print(meta)
    print() 
    type_line(tip, delay=0.004)
    _print_command_panel(
        "Command Dock",
        [
            f"{UI_CYAN}/sessions{UI_RESET}  {UI_SILVER}查看最近会话、当前会话标记、模型与状态{UI_RESET}",
            f"{UI_CYAN}/tasks{UI_RESET}     {UI_SILVER}查看当前会话的待执行任务队列{UI_RESET}",
            f"{UI_CYAN}/new{UI_RESET}       {UI_SILVER}开启全新对话并切换到新的 thread_id{UI_RESET}",
            f"{UI_CYAN}/clear{UI_RESET}     {UI_SILVER}清空当前屏幕显示，不重置对话{UI_RESET}",
            f"{UI_PURPLE}/reset{UI_RESET}     {UI_SILVER}重置当前会话上下文，保留 thread_id{UI_RESET}",
            f"{UI_PURPLE}/exit{UI_RESET}      {UI_SILVER}保存状态并退出 MortyClaw{UI_RESET}",
        ],
        footer="这些是本地 UI 快捷命令，不会发送给 LLM，也不会污染对话上下文。",
    )


def cprint(text="", end="\n"):
    print_formatted_text(ANSI(str(text)), end=end)


def _shorten(value: str | None, limit: int = 32) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _visible_len(value: str) -> int:
    return len(ANSI_PATTERN.sub("", value))


def _panel_line(content: str = "") -> str:
    return f"  {UI_PURPLE}│{UI_RESET} {content}"


def _print_command_panel(title: str, rows: list[str], *, footer: str = "") -> None:
    border = f"{UI_PURPLE}╭{'─' * (PANEL_WIDTH + 2)}╮{UI_RESET}"
    divider = f"{UI_PURPLE}├{'─' * (PANEL_WIDTH + 2)}┤{UI_RESET}"
    footer_line = f"{UI_PURPLE}╰{'─' * (PANEL_WIDTH + 2)}╯{UI_RESET}"
    cprint(f"  {border}")
    cprint(_panel_line(f"{UI_CYAN}{UI_BOLD}{title}{UI_RESET}"))
    cprint(f"  {divider}")
    for row in rows:
        cprint(_panel_line(row))
    if footer:
        cprint(f"  {divider}")
        cprint(_panel_line(f"{UI_DIM}{UI_SILVER}{footer}{UI_RESET}"))
    cprint(f"  {footer_line}")
    cprint()


def render_sessions_panel(session_repository, current_thread_id: str, *, limit: int = 12) -> None:
    sessions = session_repository.list_sessions(limit=limit)
    if not sessions:
        _print_command_panel(
            "Sessions",
            [f"{UI_SILVER}暂无会话记录。启动一次 {UI_CYAN}mortyclaw run --new-session{UI_SILVER} 后会出现在这里。{UI_RESET}"],
        )
        return

    rows = []
    for session in sessions:
        is_current = session["thread_id"] == current_thread_id
        marker = f"{UI_CYAN}●{UI_RESET}" if is_current else f"{UI_SILVER}○{UI_RESET}"
        label_text = "current" if is_current else "session"
        label = f"{UI_PURPLE}{label_text:<7}{UI_RESET}" if is_current else f"{UI_DIM}{UI_SILVER}{label_text:<7}{UI_RESET}"
        model = "/".join(part for part in (session.get("provider"), session.get("model")) if part) or "unknown"
        rows.append(
            f"{marker} {UI_WHITE}{_shorten(session['thread_id'], 22):<22}{UI_RESET} "
            f"{label} "
            f"{UI_SILVER}status={session['status']:<7} model={_shorten(model, 14):<14} "
            f"last={session['last_active_at']}{UI_RESET}"
        )

    _print_command_panel(
        "Sessions",
        rows,
        footer="提示：使用 mortyclaw run --thread-id <id> 可回到指定会话，monitor --thread-id <id> 可单独监控。",
    )


def render_tasks_panel(current_thread_id: str, *, limit: int = 20) -> None:
    task_repository = get_task_repository()
    task_repository.bootstrap_legacy_tasks(default_thread_id="local_geek_master")
    tasks = task_repository.list_tasks(
        thread_id=current_thread_id,
        statuses=("scheduled",),
        limit=limit,
    )

    if not tasks:
        _print_command_panel(
            "Tasks",
            [f"{UI_SILVER}当前会话没有待执行任务。你可以直接说：明天上午 9 点提醒我开会。{UI_RESET}"],
            footer=f"当前会话：{current_thread_id}",
        )
        return

    rows = []
    for task in tasks:
        repeat = task.get("repeat") or "once"
        remaining = task.get("remaining_runs")
        remaining_text = "∞" if remaining is None and task.get("repeat") else (str(remaining) if remaining is not None else "-")
        rows.append(
            f"{UI_CYAN}#{task['task_id']:<8}{UI_RESET} "
            f"{UI_WHITE}{task['target_time']}{UI_RESET} "
            f"{UI_PURPLE}{repeat:<6}{UI_RESET} "
            f"{UI_SILVER}left={remaining_text:<3}{UI_RESET} "
            f"{_shorten(task['description'], 34)}"
        )

    _print_command_panel(
        "Tasks",
        rows,
        footer=f"当前仅显示会话 {current_thread_id} 的 scheduled 任务；到期任务由 mortyclaw heartbeat 投递。",
    )


def _generate_next_runtime_thread_id(session_repository) -> str:
    used_numbers = set()
    for session in session_repository.list_sessions(limit=10000):
        thread_id = str(session.get("thread_id", "")).strip()
        if not thread_id.startswith("session-"):
            continue
        suffix = thread_id.removeprefix("session-")
        if suffix.isdigit():
            used_numbers.add(int(suffix))

    next_number = 1
    while next_number in used_numbers:
        next_number += 1
    return f"session-{next_number}"


def _is_transient_test_thread_id(thread_id: str | None) -> bool:
    normalized = (thread_id or "").strip().lower()
    return normalized.startswith("test_")


def _resolve_initial_thread_id(session_repository, requested_thread_id: str | None = None) -> str:
    explicit_thread_id = str(requested_thread_id or "").strip()
    if explicit_thread_id:
        return explicit_thread_id

    env_thread_id = os.getenv("MORTYCLAW_THREAD_ID", "").strip()
    if env_thread_id:
        return env_thread_id

    for session in session_repository.list_sessions(limit=100):
        latest_thread_id = str((session or {}).get("thread_id", "")).strip()
        if latest_thread_id and not _is_transient_test_thread_id(latest_thread_id):
            return latest_thread_id

    return _generate_next_runtime_thread_id(session_repository)


async def async_main(thread_id: str | None = None):
    print_banner()
    
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    load_dotenv(env_path)
    
    current_provider = os.getenv("DEFAULT_PROVIDER", "aliyun")
    current_model = os.getenv("DEFAULT_MODEL", "glm-5")
    session_repository = get_session_repository()
    current_thread_id = _resolve_initial_thread_id(session_repository, thread_id)
    set_active_thread_id(current_thread_id)
    session_repository.upsert_session(
        thread_id=current_thread_id,
        display_name=current_thread_id,
        provider=current_provider,
        model=current_model,
        status="active",
        log_file=build_log_file_path(current_thread_id),
    )
    conversation_writer = get_conversation_writer()

    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as memory:
        app = create_agent_app(provider_name=current_provider, model_name=current_model, checkpointer=memory)
        queued_inbox_event_ids: set[str] = set()
        shutdown_event = asyncio.Event()
        terminal_output = None

        class SpinnerState:
            action_words = [
                "Thinking...",
                "Reviewing context...",
                "Reasoning about next action...",
                "Preparing next tool call...",
                "Processing current findings...",
                "Waiting for model response...",
                "Analyzing task state...",
                "Updating execution context...",
                "Continuing slow task..."
            ]
            current_words = [] 
            is_spinning = False
            start_time = 0
            frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
            is_tool_calling = False 
            tool_msg = ""           

        spinner = SpinnerState()
        agent_like_nodes = {"agent", "fast_agent", "slow_agent"}
        stage_labels = {
            "router": "正在判断任务路径",
            "planner": "正在生成执行计划",
            "approval_gate": "正在等待风险确认",
            "reviewer": "正在审查当前步骤结果",
            "fast_tools": "正在执行快路径工具",
            "slow_tools": "正在执行慢路径工具",
            "finalizer": "正在生成最终总结",
        }

        def _get_terminal_output():
            if terminal_output is not None:
                return terminal_output
            app_instance = get_app_or_none()
            if app_instance is not None:
                return getattr(app_instance, "output", None)
            return None

        def _terminal_write_raw(text: str) -> None:
            output = _get_terminal_output()
            if output is not None:
                output.write_raw(text)
                output.flush()
                return
            sys.__stdout__.write(text)
            sys.__stdout__.flush()

        def _terminal_clear_lines(line_count: int) -> None:
            if line_count <= 0:
                return
            output = _get_terminal_output()
            if output is not None:
                output.cursor_up(line_count)
                output.erase_down()
                output.flush()
                return
            sys.__stdout__.write(f"\033[{line_count}F\033[J")
            sys.__stdout__.flush()

        def _terminal_replace_line(total_lines: int, line_index: int, text: str) -> None:
            if total_lines <= 0 or line_index < 0 or line_index >= total_lines:
                return

            lines_up = total_lines - line_index
            output = _get_terminal_output()
            if output is not None:
                output.cursor_up(lines_up)
                output.write_raw("\r")
                output.erase_end_of_line()
                output.write_raw(text)
                output.write_raw("\r")
                output.cursor_down(lines_up)
                output.write_raw("\r")
                output.flush()
                return

            sys.__stdout__.write(f"\033[{lines_up}A\r\033[K{text}\r\033[{lines_up}B\r")
            sys.__stdout__.flush()

        class LivePlanDisplay:
            def __init__(self):
                self.node_data: dict | None = None
                self.rendered_line_count = 0
                self.rendered_lines: list[str] = []
                self.spinner_line_index: int | None = None
                self.last_frame = ""

            def has_content(self) -> bool:
                return bool(_plan_render_signature(self.node_data))

            def set_node_data(self, node_data: dict | None) -> None:
                self.node_data = dict(node_data) if isinstance(node_data, dict) else None

            async def clear(self) -> None:
                if self.rendered_line_count <= 0:
                    return

                line_count = self.rendered_line_count

                def op() -> None:
                    _terminal_clear_lines(line_count)

                await _run_live_output_operation(op, app_instance=get_app_or_none())
                self.rendered_line_count = 0
                self.rendered_lines = []
                self.spinner_line_index = None
                self.last_frame = ""

            async def render(self) -> None:
                lines, spinner_line_index = _build_current_plan_lines(self.node_data, frame=self.current_frame())
                if not lines:
                    await self.clear()
                    return

                block = "\n".join(lines)
                previous_line_count = self.rendered_line_count

                def op() -> None:
                    if previous_line_count > 0:
                        _terminal_clear_lines(previous_line_count)
                    _terminal_write_raw(block)
                    if not block.endswith("\n"):
                        _terminal_write_raw("\n")

                await _run_live_output_operation(op, app_instance=get_app_or_none())
                self.rendered_line_count = len(lines)
                self.rendered_lines = lines
                self.spinner_line_index = spinner_line_index
                self.last_frame = self.current_frame()

            async def refresh(self) -> None:
                if not self.has_content() or self.rendered_line_count <= 0:
                    return
                if self.spinner_line_index is None:
                    return

                frame = self.current_frame()
                if frame == self.last_frame:
                    return

                lines, spinner_line_index = _build_current_plan_lines(self.node_data, frame=frame)
                if not lines:
                    await self.clear()
                    return
                if spinner_line_index is None or spinner_line_index != self.spinner_line_index:
                    await self.render()
                    return
                if len(lines) != self.rendered_line_count:
                    await self.render()
                    return

                next_line = lines[spinner_line_index]
                current_line = self.rendered_lines[spinner_line_index]
                if next_line != current_line:
                    total_lines = self.rendered_line_count

                    def op() -> None:
                        _terminal_replace_line(total_lines, spinner_line_index, next_line)

                    await _run_live_output_operation(op, app_instance=get_app_or_none())
                    self.rendered_lines = lines
                self.last_frame = frame

            async def freeze(self) -> None:
                if self.has_content() and self.rendered_line_count == 0:
                    await self.render()
                self.rendered_line_count = 0
                self.rendered_lines = []
                self.spinner_line_index = None
                self.last_frame = ""

            def current_frame(self) -> str:
                if not spinner.is_spinning or not spinner.frames:
                    return "⠋"
                elapsed = time.time() - spinner.start_time
                return spinner.frames[int(elapsed * 12) % len(spinner.frames)]

        live_plan_display = LivePlanDisplay()

        async def cprint_live(
            text: str = "",
            end: str = "\n",
            *,
            restore_live_block: bool = True,
            keep_live_block_above: bool = False,
        ) -> None:
            had_live_block = live_plan_display.rendered_line_count > 0
            if had_live_block:
                await live_plan_display.clear()

            if _should_pre_render_live_plan(
                rendered_line_count=live_plan_display.rendered_line_count,
                has_live_content=live_plan_display.has_content(),
                keep_live_block_above=keep_live_block_above,
            ):
                await live_plan_display.render()
            elif had_live_block and keep_live_block_above:
                await live_plan_display.render()

            def op() -> None:
                cprint(text, end=end)

            await _run_live_output_operation(op, app_instance=get_app_or_none())
            if had_live_block and restore_live_block and not keep_live_block_above:
                await live_plan_display.render()


        def get_bottom_toolbar():
            if not spinner.is_spinning:
                return ANSI("") 
            
            elapsed = time.time() - spinner.start_time
            if spinner.is_tool_calling:
                display_msg = spinner.tool_msg
            else:
                idx_word = int(elapsed) % len(spinner.current_words)
                display_msg = f"👾 {spinner.current_words[idx_word]}"

            idx_frame = int(elapsed * 12) % len(spinner.frames)
            frame = spinner.frames[idx_frame]
            

            return ANSI(f"  \033[38;5;51m{frame}\033[0m \033[38;5;250m{display_msg}\033[0m \033[38;5;141m[{elapsed:.1f}s]\033[0m")

        prompt_message = ANSI("  \033[38;5;51m❯\033[0m ")
        placeholder_text = ANSI("\033[3m\033[38;5;242minput...  /new  /sessions  /tasks  /clear  /reset\033[0m")

        async def agent_worker():
            while True:
                queue_item = await task_queue.get()
                last_plan_signature = ""
                inbox_event_id = None
                if isinstance(queue_item, dict):
                    user_input = str(queue_item.get("content", "")).strip()
                    inbox_event_id = queue_item.get("event_id")
                else:
                    user_input = str(queue_item).strip()

                if user_input.lower() in ["/exit", "/quit"]:
                    if inbox_event_id:
                        session_repository.mark_inbox_event_delivered(inbox_event_id)
                        queued_inbox_event_ids.discard(inbox_event_id)
                    task_queue.task_done()
                    break

                live_plan_display.node_data = None
                live_plan_display.rendered_line_count = 0
                spinner.current_words = spinner.action_words.copy()
                random.shuffle(spinner.current_words)
                
                spinner.start_time = time.time()
                spinner.is_spinning = True
                spinner.is_tool_calling = False
                
                turn_id = str(uuid.uuid4())
                user_message = HumanMessage(content=user_input, id=f"{turn_id}:user")
                conversation_writer.append_messages(
                    thread_id=current_thread_id,
                    turn_id=turn_id,
                    messages=[user_message],
                    node_name="user_input",
                    route="input",
                )
                inputs = {"messages": [user_message]}
                processed_successfully = False
                try:
                    set_active_thread_id(current_thread_id)
                    session_repository.touch_session(current_thread_id, status="active")
                    turn_config = {"configurable": {"thread_id": current_thread_id, "turn_id": turn_id}}
                    async for event in app.astream(inputs, config=turn_config, stream_mode="updates"):
                        for node_name, node_data in event.items():
                            if isinstance(node_data, dict):
                                plan_signature = _plan_render_signature(node_data)
                                if plan_signature and plan_signature != last_plan_signature:
                                    live_plan_display.set_node_data(node_data)
                                    await live_plan_display.render()
                                    last_plan_signature = plan_signature
                            node_messages = node_data.get("messages") if isinstance(node_data, dict) else None
                            if isinstance(node_messages, list) and node_messages:
                                node_messages = prepare_tool_messages_for_budget(
                                    node_messages,
                                    thread_id=current_thread_id,
                                    turn_id=turn_id,
                                )
                                conversation_writer.append_messages(
                                    thread_id=current_thread_id,
                                    turn_id=turn_id,
                                    messages=node_messages,
                                    node_name=node_name,
                                    route=str(node_data.get("route", "")),
                                )

                            if node_name in stage_labels:
                                spinner.is_tool_calling = True
                                spinner.tool_msg = f"{stage_labels[node_name]}..."

                            if node_name in agent_like_nodes:
                                last_msg = node_data["messages"][-1]
                                
                                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                                    for tc in last_msg.tool_calls:
                                        spinner.is_tool_calling = True
                                        spinner.tool_msg = f"唤醒内置工具 : {tc['name']}..."
                                        if not live_plan_display.has_content():
                                            await cprint_live(f"  ●\033[38;5;51m Tool Call: \033[0m{tc['name']}")
                                            await cprint_live('')
                            rendered_message = _render_stream_node_message(node_name, node_data)
                            if rendered_message:
                                if _is_slow_step_result_message(node_name, node_data):
                                    spinner.is_tool_calling = True
                                    spinner.tool_msg = f"{stage_labels['reviewer']}..."
                                    if not live_plan_display.has_content():
                                        await cprint_live(rendered_message)
                                else:
                                    spinner.is_spinning = False
                                    await cprint_live(
                                        rendered_message,
                                        restore_live_block=False,
                                        keep_live_block_above=_should_render_message_below_live_plan(node_name, node_data),
                                    )
                            else:
                                spinner.is_tool_calling = False 
                    processed_successfully = True
                except Exception as e:
                    spinner.is_spinning = False
                    await cprint_live(f"  \033[31m[ ⚠️ 引擎异常 : {e} ]\033[0m")

                if inbox_event_id and processed_successfully:
                    session_repository.mark_inbox_event_delivered(inbox_event_id)
                if inbox_event_id:
                    queued_inbox_event_ids.discard(inbox_event_id)

                spinner.is_spinning = False
                await live_plan_display.freeze()
                await cprint_live() # 空出舒适的行距
                task_queue.task_done()

        async def inbox_poller():
            while not shutdown_event.is_set():
                try:
                    pending_events = session_repository.list_pending_inbox_events(current_thread_id, limit=20)
                    for inbox_event in pending_events:
                        if shutdown_event.is_set():
                            break
                        event_id = inbox_event["event_id"]
                        if event_id in queued_inbox_event_ids:
                            continue

                        try:
                            payload = json.loads(inbox_event["payload"])
                        except json.JSONDecodeError:
                            session_repository.mark_inbox_event_delivered(event_id)
                            continue

                        content = str(payload.get("content", "")).strip()
                        if not content:
                            session_repository.mark_inbox_event_delivered(event_id)
                            continue

                        queued_inbox_event_ids.add(event_id)
                        await task_queue.put({
                            "kind": "session_inbox",
                            "event_id": event_id,
                            "content": content,
                        })
                except asyncio.CancelledError:
                    raise
                except Exception:
                    pass
                await asyncio.sleep(1.0)

        async def user_input_loop():
            nonlocal app, terminal_output, current_thread_id
            session = _build_prompt_session(bottom_toolbar=get_bottom_toolbar)
            terminal_output = session.app.output
            
            async def redraw_timer():
                while True:
                    if spinner.is_spinning:
                        await live_plan_display.refresh()
                        try:
                            get_app().invalidate()
                        except Exception:
                            pass
                    await asyncio.sleep(0.08)
                    
            redraw_task = asyncio.create_task(redraw_timer())
            
            while True:
                try:
                    user_input = await session.prompt_async(prompt_message, placeholder=placeholder_text)

                    user_input = user_input.strip()
                    if not user_input:
                        continue
                    

                    padded_bubble = f"  ❯ {user_input}    "
                    cprint(f"\033[48;2;38;38;38m\033[38;5;255m{padded_bubble}\033[0m\n")

                    lowered_input = user_input.lower()
                    if lowered_input == "/sessions":
                        render_sessions_panel(session_repository, current_thread_id)
                        continue
                    if lowered_input == "/tasks":
                        render_tasks_panel(current_thread_id)
                        continue
                    if lowered_input == "/new":
                        if spinner.is_spinning or not task_queue.empty():
                            cprint(f"  {UI_PURPLE}✦{UI_RESET} {UI_SILVER}当前仍有任务在执行，请等待完成后再新建会话。{UI_RESET}")
                            cprint()
                            continue

                        previous_thread_id = current_thread_id
                        spinner.is_spinning = False
                        spinner.is_tool_calling = False
                        spinner.tool_msg = ""
                        live_plan_display.node_data = None
                        await live_plan_display.clear()
                        clear_screen()
                        current_thread_id = _generate_next_runtime_thread_id(session_repository)
                        set_active_thread_id(current_thread_id)
                        session_repository.touch_session(previous_thread_id, status="idle")
                        session_repository.upsert_session(
                            thread_id=current_thread_id,
                            display_name=current_thread_id,
                            provider=current_provider,
                            model=current_model,
                            status="active",
                            log_file=build_log_file_path(current_thread_id),
                        )
                        cprint(
                            f"  {UI_PURPLE}✦{UI_RESET} {UI_SILVER}已开启新会话："
                            f"{UI_WHITE}{current_thread_id}{UI_RESET}{UI_SILVER} "
                            f"(from {previous_thread_id}){UI_RESET}"
                        )
                        cprint(f"  {UI_DIM}{UI_SILVER}后续输入都会写入这个新的 thread_id。{UI_RESET}")
                        cprint()
                        session = _build_prompt_session(bottom_toolbar=get_bottom_toolbar)
                        terminal_output = session.app.output
                        continue
                    if lowered_input == "/clear":
                        clear_screen()
                        continue
                    if lowered_input == "/reset":
                        if spinner.is_spinning or not task_queue.empty():
                            cprint(f"  {UI_PURPLE}✦{UI_RESET} {UI_SILVER}当前仍有任务在执行，请等待完成后再重置会话。{UI_RESET}")
                            cprint()
                            continue

                        try:
                            report = reset_thread_state(thread_id=current_thread_id)
                            app = create_agent_app(
                                provider_name=current_provider,
                                model_name=current_model,
                                checkpointer=memory,
                            )
                            cprint(
                                f"  {UI_PURPLE}✦{UI_RESET} {UI_SILVER}当前会话上下文已重置："
                                f"thread_id={current_thread_id} | "
                                f"deleted_checkpoints={report['deleted_checkpoints']} | "
                                f"deleted_writes={report['deleted_writes']}{UI_RESET}"
                            )
                            cprint(f"  {UI_DIM}{UI_SILVER}已保留当前会话 ID，接下来会从干净上下文继续对话。{UI_RESET}")
                            cprint()
                        except Exception as exc:
                            cprint(f"  \033[31m[ ⚠️ 重置失败 : {exc} ]\033[0m")
                            cprint()
                        continue
                    
                    await task_queue.put(user_input)
                    if user_input.lower() in ["/exit", "/quit"]:
                        shutdown_event.set()
                        cprint("  \033[38;5;141m✦ 记忆已固化，MortyClaw 进入休眠。\033[0m")
                        break
                        
                except (KeyboardInterrupt, EOFError):
                    shutdown_event.set()
                    cprint("\n  \033[38;5;141m✦ 强制中断，MortyClaw 进入休眠。\033[0m")
                    await task_queue.put("/exit")
                    break

            redraw_task.cancel() 

        with patch_stdout():
            worker = asyncio.create_task(agent_worker())
            inbox_worker = asyncio.create_task(inbox_poller())
            try:
                await user_input_loop()
                await task_queue.join()
            finally:
                shutdown_event.set()
                worker.cancel()
                inbox_worker.cancel()
                conversation_writer.flush()
                session_repository.touch_session(current_thread_id, status="idle")

def main(thread_id: str | None = None):
    asyncio.run(async_main(thread_id=thread_id))

if __name__ == "__main__":
    main()
