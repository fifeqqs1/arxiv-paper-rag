import json
import os
import time
from datetime import datetime

from rich import box
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme

from mortyclaw.core.logger import build_log_file_path
from mortyclaw.core.storage.runtime import get_session_repository


morty_theme = Theme({
    "info": "dim cyan",
    "warning": "color(141)",
    "error": "bold red",
    "llm_input": "dim white",
    "tool_call": "bold yellow",
    "tool_result": "bold green",
    "ai_message": "bold bright_magenta",
    "timestamp": "dim white",
})

console = Console(theme=morty_theme)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def list_known_sessions(limit: int = 30) -> list[dict]:
    return get_session_repository().list_sessions(limit=limit)


def resolve_monitor_thread_id(thread_id: str | None = None, *, latest: bool = False) -> str | None:
    if thread_id:
        return thread_id

    latest_session = get_session_repository().get_latest_session()
    if latest or latest_session is not None:
        return latest_session["thread_id"] if latest_session else None
    return None


def print_session_list() -> None:
    sessions = list_known_sessions()
    if not sessions:
        console.print("[warning]当前没有可监控的会话记录。[/warning]")
        return

    lines = [
        f"- {session['thread_id']} | status={session['status']} | model={session['model'] or 'unknown'} | last_active={session['last_active_at']}"
        for session in sessions
    ]
    console.print("[info]可用会话：[/info]\n" + "\n".join(lines))


def print_header(thread_id: str):
    monster = (
        "  ▄█▄▄█▄  \n"
        " ▀██████▀ \n"
        " ██▄██▄██ \n"
        "  ▀    ▀  "
    )

    content = Text(justify="center")
    content.append("\n  Live Stream  \n\n", style="bold white italic")
    content.append(monster + "\n\n", style="color(141)")
    content.append("   What is MortyClaw doing?    \n", style="dim white italic")
    content.append(f"\nSession: {thread_id}\n", style="bold cyan")

    panel = Panel(
        Align.center(content),
        title="[bold color(141)] MortyClaw [/bold color(141)]",
        title_align="left",
        border_style="color(141)",
        box=box.ROUNDED,
        width=48,
        padding=0,
    )

    console.print(Align.center(panel))
    console.print()


def tail_f(filepath: str, *, thread_id: str):
    if not os.path.exists(filepath):
        console.print(f"[warning]⏳ 等待会话 {thread_id} 的日志文件生成...[/warning]")
        while not os.path.exists(filepath):
            time.sleep(0.5)

    with open(filepath, "r", encoding="utf-8") as f:
        f.seek(0, 2)
        print_header(thread_id)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.1)
                continue
            yield line


def render_event(line: str):
    try:
        data = json.loads(line.strip())
        event = data.get("event")
        ts_str = data.get("ts", "")
        try:
            if ts_str.endswith("Z"):
                ts_str = ts_str[:-1] + "+00:00"
            dt_local = datetime.fromisoformat(ts_str).astimezone()
            ts = dt_local.strftime("%H:%M:%S")
        except Exception:
            ts = ts_str.split("T")[-1][:8]

        prefix = f"[timestamp][ {ts} ][/timestamp] "

        if event == "llm_input":
            count = data.get("message_count", 0)
            console.print(f"{prefix}[llm_input]🧠 神经元唤醒：发送了 {count} 条上下文记忆...[/llm_input]")

        elif event == "tool_call":
            tool_name = data.get("tool", "unknown")
            args_str = json.dumps(data.get("args", {}), ensure_ascii=False, indent=2)
            content = f"[bold white] ● 使用工具: [/bold white][bold color(141)]{tool_name}[/bold color(141)]\n传入参数:\n{args_str}"
            console.print(Panel(content, title=f"✦ 意图决断 [ {ts} ]", title_align="left", border_style="color(141)", width=72))

        elif event == "tool_result":
            tool_name = data.get("tool", "unknown")
            result = data.get("result_summary", "")
            display_result = result[:300] + "\n...[截断]..." if len(result) > 300 else result
            content = f"[bold white] ● 执行结果: [/bold white][bold cyan]{tool_name}[/bold cyan]\n{display_result}"
            console.print(Panel(content, title=f"✦ 环境回传 [ {ts} ]", title_align="left", border_style="cyan", width=72))

        elif event == "ai_message":
            content = data.get("content", "")
            display_content = content[:400] + "\n...[截断]..." if len(content) > 400 else content
            console.print(Panel(display_content or "(空回复)", title=f"✦ AI 回答 [ {ts} ]", title_align="left", border_style="bright_magenta", width=72))

        elif event == "tool_call_adjusted":
            tool_name = data.get("tool", "unknown")
            original_args = json.dumps(data.get("original_args", {}), ensure_ascii=False)
            adjusted_args = json.dumps(data.get("adjusted_args", {}), ensure_ascii=False)
            console.print(f"{prefix}[warning]✦ 工具参数已调整：{tool_name} | {original_args} -> {adjusted_args}[/warning]")

        elif event == "system_action":
            action = data.get("content", "")
            console.print(f"{prefix}[warning]✦ 底层状态机：{action}[/warning]")
    except Exception:
        pass


def main(thread_id: str | None = None, latest: bool = False, list_sessions: bool = False):
    if list_sessions:
        print_session_list()
        return

    resolved_thread_id = resolve_monitor_thread_id(thread_id, latest=latest)
    if not resolved_thread_id:
        console.print("[warning]当前没有可监控的会话。可先运行 `mortyclaw sessions` 查看。[/warning]")
        return

    log_file = build_log_file_path(resolved_thread_id)
    try:
        console.clear()
        for line in tail_f(log_file, thread_id=resolved_thread_id):
            render_event(line)
    except KeyboardInterrupt:
        console.print("\n[warning]✦ 监控网络已断开。[/warning]")


if __name__ == "__main__":
    main()
