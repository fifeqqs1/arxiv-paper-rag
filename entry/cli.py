import os
import typer
import questionary
import logging
import asyncio
import re
import json
from rich.console import Console
from rich.panel import Panel
from rich.status import Status
from dotenv import set_key, load_dotenv, unset_key
import sys

from mortyclaw.core.provider import get_provider
from mortyclaw.core.config import TASKS_FILE
from mortyclaw.core.maintenance import (
    collect_doctor_report,
    format_bytes,
    gc_logs,
    gc_runtime,
    gc_state,
)
from mortyclaw.core.storage.runtime import get_conversation_repository, get_session_repository, get_task_repository
from langchain_core.messages import HumanMessage

ENTRY_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ENTRY_DIR) 

os.chdir(PROJECT_ROOT)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

app = typer.Typer(help="MortyClaw - 极客专属的赛博智能终端")
gc_app = typer.Typer(help="运行态垃圾回收与归档工具")
console = Console()

morty_style = questionary.Style([
    ('qmark', 'fg:#8d52ff bold'),       
    ('question', 'fg:#00ffff bold'),    
    ('answer', 'fg:#8d52ff bold'),      
    ('pointer', 'fg:#00ffff bold'),     
    ('highlighted', 'fg:#00ffff bold'), 
    ('selected', 'fg:#00ffff'),
    ('instruction', 'fg:#808080 dim'),  
])

ENV_PATH = os.path.join(PROJECT_ROOT, ".env")
SHORT_SESSION_ID_PATTERN = re.compile(r"^session-(\d{1,5})$")
app.add_typer(gc_app, name="gc")


def _is_transient_test_thread_id(thread_id: str | None) -> bool:
    normalized = (thread_id or "").strip().lower()
    return normalized.startswith("test_")


def _generate_thread_id(session_repository=None) -> str:
    session_repo = session_repository or get_session_repository()
    used_numbers = set()
    for session in session_repo.list_sessions(limit=10000):
        match = SHORT_SESSION_ID_PATTERN.match(session.get("thread_id", ""))
        if match:
            used_numbers.add(int(match.group(1)))

    next_number = 1
    while next_number in used_numbers:
        next_number += 1
    return f"session-{next_number}"


def _resolve_default_thread_id(session_repository=None) -> str:
    session_repo = session_repository or get_session_repository()
    for session in session_repo.list_sessions(limit=100):
        latest_thread_id = (session or {}).get("thread_id", "").strip()
        if latest_thread_id and not _is_transient_test_thread_id(latest_thread_id):
            return latest_thread_id
    return _generate_thread_id(session_repo)


def _print_doctor_report(report: dict) -> None:
    db_lines = []
    for label, info in report["databases"].items():
        table_summary = ", ".join(
            f"{table}={count}" for table, count in info["table_counts"].items()
        )
        db_lines.append(
            f"- {label}: {format_bytes(info['size_bytes'])} | {table_summary} | {info['path']}"
        )

    log_lines = [
        f"- logs: {format_bytes(report['logs']['size_bytes'])} | files={report['logs']['file_count']} | {report['logs']['path']}"
    ]
    for item in report["logs"]["largest_files"]:
        log_lines.append(
            f"  - {os.path.basename(item['path'])}: {format_bytes(item['size_bytes'])}"
        )

    console.print(
        Panel(
            "[bold #00ffff]Databases[/bold #00ffff]\n"
            + "\n".join(db_lines)
            + "\n\n[bold #00ffff]Logs[/bold #00ffff]\n"
            + "\n".join(log_lines),
            title="[bold white]MortyClaw Doctor[/bold white]",
            border_style="#8d52ff",
        )
    )


def _print_gc_report(name: str, report: dict) -> None:
    mode = "dry-run" if report.get("dry_run", True) else "apply"
    lines = [f"mode={mode}"]

    if name == "logs":
        lines.append(f"candidate_count={report['candidate_count']}")
        if report.get("archive_path"):
            lines.append(f"archive={report['archive_path']}")
        lines.append(f"removed_count={report['removed_count']}")
    elif name == "runtime":
        lines.append(
            "session_inbox: "
            f"candidate={report['inbox']['candidate_count']} "
            f"deleted={report['inbox']['deleted_count']}"
        )
        lines.append(
            "task_runs: "
            f"candidate={report['task_runs']['candidate_count']} "
            f"deleted={report['task_runs']['deleted_count']}"
        )
    elif name == "state":
        lines.append(
            f"keep_latest_per_thread={report['keep_latest_per_thread']} "
            f"checkpoint_candidate_count={report['checkpoint_candidate_count']} "
            f"write_candidate_count={report['write_candidate_count']}"
        )
        if report.get("backup_path"):
            lines.append(f"backup={report['backup_path']}")
        lines.append(
            f"size_before={format_bytes(report['size_before_bytes'])} "
            f"size_after={format_bytes(report['size_after_bytes'])}"
        )

    console.print(
        Panel(
            "\n".join(lines),
            title=f"[bold white]MortyClaw GC · {name}[/bold white]",
            border_style="#00ffff" if not report.get("dry_run", True) else "#8d52ff",
        )
    )

    if name == "logs" and report.get("candidates"):
        for item in report["candidates"][:10]:
            console.print(
                f"[dim]- {item['path']} | reasons={','.join(item['reasons'])} | size={format_bytes(item['size_bytes'])}[/dim]"
            )
    if name == "runtime":
        for item in report["inbox"]["candidates"][:10]:
            console.print(
                f"[dim]- inbox {item['event_id']} | thread={item['thread_id']} | delivered_at={item['delivered_at']}[/dim]"
            )
        for item in report["task_runs"]["candidates"][:10]:
            console.print(
                f"[dim]- task_run {item['run_id']} | task={item['task_id']} | triggered_at={item['triggered_at']}[/dim]"
            )
    if name == "state":
        for thread_key, summary in sorted(report["threads"].items()):
            if summary["checkpoint_prunable"] > 0:
                console.print(
                    f"[dim]- {thread_key} | checkpoints={summary['checkpoint_total']} | "
                    f"prunable={summary['checkpoint_prunable']} | writes={summary['write_prunable']}[/dim]"
                )

@app.command("config")
def config_wizard():
    console.clear()
    console.print(Panel(
        "😈 Welcome to [bold #8d52ff]MortyClaw[/bold #8d52ff]...\n\n☁️[dim] 请完成模型配置，我们将把密钥安全固化在本地。[/dim]",
        title="[bold white]✦  MortyClaw Config[/bold white]",
        border_style="#8d52ff"
    ))
    provider_raw = questionary.select(
        "选择你的模型提供商 (Provider):",
        choices=["openai", "anthropic", "aliyun (openai compatible)","tencent (openai compatible)", "z.ai (openai compatible)", "other (openai compatible)", "ollama"],
        style=morty_style,
        instruction="(按上下键选择，回车确认)"
    ).ask()

    if not provider_raw:
        console.print("[dim #8d52ff]✦   录入中断，MortyClaw 配置已取消。[/dim #8d52ff]")
        return

    provider = provider_raw.split(" ")[0].strip()
    is_openai_compatible = "openai" in provider_raw.lower()

    model_name = questionary.text(
        "输入指定的模型型号 (如 gpt-4o-mini, qwen-max, glm-4 等):",
        style=morty_style
    ).ask()

    if model_name is None:
        console.print("[dim #8d52ff]✦   录入中断，MortyClaw 配置已取消。[/dim #8d52ff]")
        return

    api_key = ""
    env_key = ""
    if provider != "ollama":
        if is_openai_compatible:
            env_key = "OPENAI_API_KEY"
        elif provider == "anthropic":
            env_key = "ANTHROPIC_API_KEY"

        api_key = questionary.password(
            f"输入你的 {env_key} (对应 {provider_raw}):",
            style=morty_style
        ).ask()

        if api_key is None:
            console.print("[dim #8d52ff]✦   录入中断，MortyClaw 配置已取消。[/dim #8d52ff]")
            return

    base_url = ""
    if provider in ["openai", "anthropic"]:
        base_url = questionary.text(
            f"输入 {provider} 代理 Base URL (直连请直接回车跳过):",
            style=morty_style
        ).ask()
    elif provider == "ollama":
        base_url = questionary.text(
            "输入 Ollama Base URL (默认 http://localhost:11434，直接回车跳过):",
            style=morty_style
        ).ask()
    else:
        base_url = questionary.text(
            "输入兼容 Base URL (不填直接回车将使用官方默认地址):",
            style=morty_style
        ).ask()

    if base_url is None:
        console.print("[dim #8d52ff]✦   录入中断，MortyClaw 配置已取消。[/dim #8d52ff]")
        return

    console.print("\n[dim]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/dim]")

    with Status(f"[bold #8d52ff]正在连接 {provider.upper()} 引擎并发送探测包...[/bold #8d52ff]", spinner="dots", spinner_style="#00ffff"):
        try:
            if env_key and api_key:
                os.environ[env_key] = api_key
            if base_url:
                if is_openai_compatible:
                    os.environ["OPENAI_API_BASE"] = base_url
                else:
                    os.environ[f"{provider.upper()}_BASE_URL"] = base_url

            llm = get_provider(provider_name=provider, model_name=model_name)
            response = llm.invoke([HumanMessage(content="回复我'收到'。")])

            console.print(" [bold #00ffff][ 配置成功!][/bold #00ffff]")
            
        except Exception as e:

            console.print(f" [bold #8d52ff][ 配置失败!][/bold #8d52ff]  无法连接到模型，请检查 Key、Base URL、模型型号 或 网络！\n[dim]错误信息: {str(e)}[/dim]")
            return


    if not os.path.exists(ENV_PATH):
        open(ENV_PATH, 'w').close()

    logging.getLogger("dotenv.main").setLevel(logging.ERROR)

    unset_key(ENV_PATH, "OPENAI_API_BASE")
    unset_key(ENV_PATH, "ANTHROPIC_BASE_URL")
    unset_key(ENV_PATH, "OLLAMA_BASE_URL")

    if env_key and api_key:
        set_key(ENV_PATH, env_key, api_key)
        
    if base_url:
        if is_openai_compatible:
            set_key(ENV_PATH, "OPENAI_API_BASE", base_url)
        else:
            set_key(ENV_PATH, f"{provider.upper()}_BASE_URL", base_url)
    
    set_key(ENV_PATH, "DEFAULT_PROVIDER", provider)
    set_key(ENV_PATH, "DEFAULT_MODEL", model_name)

    console.print(Panel(
        f"配置已保存至 [#8d52ff]{ENV_PATH}[/#8d52ff]\n"
        f"当前默认提供商: [#8d52ff]{provider}[/#8d52ff] | 模型: [#8d52ff]{model_name}[/#8d52ff]\n\n"
        f"👉 输入 [bold #00ffff]mortyclaw run[/bold #00ffff] 即可启动系统！",
        border_style="#00ffff"
    ))

def _show_boot_error():
    console.print(Panel(
        "[bold #00ffff]MortyClaw未完成配置![/bold #00ffff]\n\n"
        "[#8d52ff]检测到 API Key、模型或Baseurl。请重新执行以下命令完成配置：[/#8d52ff]\n"
        "[bold #00ffff]mortyclaw config[/bold #00ffff]",
        title="[bold #8d52ff]⚠️ Boot Sequence Failed[/bold #8d52ff]",
        border_style="#8d52ff"
    ))


@app.command("run")
def run_agent(
    thread_id: str | None = typer.Option(None, "--thread-id", help="指定要运行的会话 thread_id"),
    new_session: bool = typer.Option(False, "--new", "--new-session", help="创建一个新的短编号会话 ID，例如 session-1"),
    branch_from: str | None = typer.Option(None, "--branch-from", help="从指定历史会话创建一个语义分支会话"),
):
    load_dotenv(ENV_PATH)
    provider = os.getenv("DEFAULT_PROVIDER")
    model = os.getenv("DEFAULT_MODEL")
    if not provider or not model:
        _show_boot_error()
        raise typer.Exit()
    if provider != "ollama":
        if provider in ["openai", "aliyun", "z.ai", "tencent", "other"]: 
            if not os.getenv("OPENAI_API_KEY"):
                _show_boot_error()
                raise typer.Exit()
                
        elif provider == "anthropic":
            if not os.getenv("ANTHROPIC_API_KEY"):
                _show_boot_error()
                raise typer.Exit()

    session_repository = get_session_repository()
    resolved_thread_id = thread_id
    if new_session and not resolved_thread_id:
        resolved_thread_id = _generate_thread_id(session_repository)
    if branch_from and not resolved_thread_id:
        resolved_thread_id = _generate_thread_id(session_repository)
    resolved_thread_id = (resolved_thread_id or _resolve_default_thread_id(session_repository)).strip()
    if not resolved_thread_id:
        resolved_thread_id = _generate_thread_id(session_repository)
    if branch_from:
        session_repository.create_branch_session(
            parent_thread_id=branch_from,
            branch_thread_id=resolved_thread_id,
            provider=provider,
            model=model,
        )
        console.print(
            f"[dim #8d52ff]已创建会话分支：{resolved_thread_id} <- {branch_from}。"
            "当前分支保留 lineage 元数据，运行状态仍由新的 thread_id 独立恢复。[/dim #8d52ff]"
        )

    import entry.main as mortyclaw_main
    mortyclaw_main.main(thread_id=resolved_thread_id)

@app.command("monitor")
def run_monitor(
    thread_id: str | None = typer.Option(None, "--thread-id", help="指定要监控的会话 thread_id"),
    latest: bool = typer.Option(False, "--latest", help="自动选择最近活跃的会话"),
    list_sessions: bool = typer.Option(False, "--list-sessions", help="列出当前已知会话"),
):
    try:
        import entry.monitor as mortyclaw_monitor
        mortyclaw_monitor.main(thread_id=thread_id, latest=latest, list_sessions=list_sessions)
    except ImportError as e:
        console.print(f"[bold red]启动失败：找不到监视器模块！[/bold red]\n[dim]请确保 monitor.py 和 cli.py 在同一目录下。\n报错信息: {e}[/dim]")


@app.command("heartbeat")
def run_heartbeat(
    interval: int = typer.Option(10, "--interval", min=1, help="轮询检查间隔，单位秒"),
    once: bool = typer.Option(False, "--once", help="只执行一次到期任务扫描"),
):
    from mortyclaw.core.heartbeat import pacemaker_loop, process_due_tasks_once

    if once:
        triggered = process_due_tasks_once()
        console.print(f"[bold #00ffff]本次心跳共投递 {len(triggered)} 个到期任务。[/bold #00ffff]")
        return

    console.print(f"[bold #00ffff]Heartbeat 已启动[/bold #00ffff] [dim](interval={interval}s，Ctrl+C 停止)[/dim]")
    try:
        asyncio.run(pacemaker_loop(check_interval=interval))
    except KeyboardInterrupt:
        console.print("[dim #8d52ff]Heartbeat 已停止。[/dim #8d52ff]")


@app.command("sessions")
def list_sessions(limit: int = typer.Option(20, "--limit", min=1, help="最多展示多少个会话")):
    sessions = get_session_repository().list_sessions(limit=limit)
    if not sessions:
        console.print("[dim]当前还没有会话记录。[/dim]")
        return

    lines = []
    for item in sessions:
        lines.append(
            f"- {item['thread_id']} | status={item['status']} | model={item['model'] or 'unknown'} | last_active={item['last_active_at']}"
        )
    console.print("[bold #00ffff]已记录会话：[/bold #00ffff]\n" + "\n".join(lines))


@app.command("session-search")
def session_search(
    query: str = typer.Argument("", help="要搜索的历史关键词；留空则列出最近会话"),
    role_filter: str = typer.Option("", "--role-filter", help="限制角色，例如 user,assistant,tool"),
    limit: int = typer.Option(3, "--limit", min=1, max=5, help="最多返回多少个会话"),
    include_current: bool = typer.Option(False, "--include-current", help="是否包含当前/指定会话"),
    current_thread_id: str | None = typer.Option(None, "--current-thread-id", help="用于排除当前 lineage 的 thread_id"),
    include_tool_results: bool = typer.Option(True, "--tool-results/--no-tool-results", help="是否搜索和展示工具结果"),
):
    roles = [role.strip() for role in role_filter.split(",") if role.strip()]
    results = get_conversation_repository().search_sessions(
        query=query,
        role_filter=roles or None,
        limit=limit,
        include_current=include_current,
        current_thread_id=current_thread_id,
        include_tool_results=include_tool_results,
    )
    console.print(json.dumps({
        "success": True,
        "query": query,
        "mode": "recent" if not query.strip() else "search",
        "count": len(results),
        "results": results,
    }, ensure_ascii=False, indent=2))


@app.command("session-show")
def session_show(
    thread_id: str = typer.Argument(..., help="要展示的会话 thread_id"),
    limit: int = typer.Option(80, "--limit", min=1, help="最多展示多少条消息"),
):
    messages = get_conversation_repository().get_session_conversation(thread_id, limit=limit)
    if not messages:
        console.print(f"[dim]没有找到会话 {thread_id} 的 conversation message 记录。[/dim]")
        return

    lines = []
    for message in messages:
        tool = f" tool={message['tool_name']}" if message.get("tool_name") else ""
        preview = re.sub(r"\s+", " ", message.get("content", "")).strip()
        if len(preview) > 220:
            preview = preview[:219] + "…"
        lines.append(
            f"#{message['seq']:04d} {message['created_at']} {message['role']}{tool}: {preview}"
        )
    console.print("[bold #00ffff]Conversation Messages[/bold #00ffff]\n" + "\n".join(lines))


@app.command("doctor")
def run_doctor():
    _print_doctor_report(collect_doctor_report())


@gc_app.command("logs")
def run_gc_logs(
    apply: bool = typer.Option(False, "--apply", help="真正执行归档；默认仅 dry-run 预览"),
):
    _print_gc_report("logs", gc_logs(apply=apply))


@gc_app.command("runtime")
def run_gc_runtime(
    apply: bool = typer.Option(False, "--apply", help="真正执行清理；默认仅 dry-run 预览"),
):
    _print_gc_report("runtime", gc_runtime(apply=apply))


@gc_app.command("state")
def run_gc_state(
    apply: bool = typer.Option(False, "--apply", help="真正执行 checkpoint 裁剪；默认仅 dry-run 预览"),
):
    _print_gc_report("state", gc_state(apply=apply))


@app.command("migrate-tasks")
def migrate_tasks(
    source_path: str = typer.Option(TASKS_FILE, "--source", help="旧 tasks.json 的路径"),
    default_thread_id: str = typer.Option("local_geek_master", "--default-thread-id", help="旧任务默认归属的会话 ID"),
    force: bool = typer.Option(False, "--force", help="已存在同 ID 任务时允许覆盖"),
):
    result = get_task_repository().import_legacy_tasks(
        file_path=source_path,
        default_thread_id=default_thread_id,
        overwrite=force,
    )
    console.print(
        f"[bold #00ffff]任务迁移完成[/bold #00ffff] [dim](imported={result['imported']}, skipped={result['skipped']})[/dim]"
    )

def main():
    app()

if __name__ == "__main__":
    main()
