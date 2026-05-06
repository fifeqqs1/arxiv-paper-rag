from ..planning import get_current_plan_step
from .policy import build_approval_reason


def _describe_pending_tool_calls(state) -> str:
    pending_tool_calls = [
        tool_call for tool_call in (state.get("pending_tool_calls", []) or [])
        if isinstance(tool_call, dict) and tool_call.get("name")
    ]
    if not pending_tool_calls:
        return ""
    tool_names = []
    for tool_call in pending_tool_calls:
        tool_name = str(tool_call.get("name") or "").strip()
        if tool_name and tool_name not in tool_names:
            tool_names.append(tool_name)
    if not tool_names:
        return ""
    return f"本轮待执行的高风险工具：{', '.join(tool_names)}"


def _pending_tool_names(state) -> list[str]:
    pending_tool_calls = [
        tool_call for tool_call in (state.get("pending_tool_calls", []) or [])
        if isinstance(tool_call, dict) and tool_call.get("name")
    ]
    tool_names: list[str] = []
    for tool_call in pending_tool_calls:
        tool_name = str(tool_call.get("name") or "").strip()
        if tool_name and tool_name not in tool_names:
            tool_names.append(tool_name)
    return tool_names


def _current_todo_context(state) -> tuple[int, int, str] | None:
    todos = state.get("todos") or state.get("active_todos") or []
    if not isinstance(todos, list) or not todos:
        return None

    normalized_items: list[dict] = []
    for index, item in enumerate(todos, start=1):
        if not isinstance(item, dict):
            continue
        normalized_items.append({
            "step": index,
            "content": str(item.get("content", "")).strip(),
            "status": str(item.get("status", "pending")).strip().lower() or "pending",
        })
    if not normalized_items:
        return None

    current_item = next((item for item in normalized_items if item["status"] == "in_progress"), None)
    if current_item is None:
        current_item = next((item for item in normalized_items if item["status"] == "pending"), None)
    if current_item is None:
        current_item = normalized_items[-1]
    return current_item["step"], len(normalized_items), current_item["content"]


def _requires_read_only_mode(task_text: str) -> bool:
    lowered = (task_text or "").lower()
    write_markers = (
        "修改", "修复", "新增", "增加", "实现", "写入", "保存", "创建", "删除",
        "patch", "edit", "fix", "write", "save", "create",
    )
    exec_markers = (
        "运行", "执行", "测试", "验证", "pytest", "unittest", "py_compile",
        "shell", "命令", "bash", "python ",
    )
    return any(marker in task_text or marker in lowered for marker in write_markers + exec_markers)


def build_permission_mode_message(state) -> str:
    task_text = str(state.get("goal", "") or "")
    plan_note = ""
    if _requires_read_only_mode(task_text):
        plan_note = "\n- `plan`：只读分析模式；如果任务后续需要修改文件、运行测试或执行命令，将直接终止。"
    else:
        plan_note = "\n- `plan`：只读分析模式；只允许读取和分析，不执行写入或命令。"
    return (
        "该任务已进入 slow path。开始前请选择执行权限模式：\n"
        "- `ask`：保持当前默认行为；遇到高风险写入/测试操作时继续逐次询问你是否执行。"
        f"{plan_note}\n"
        "- `auto`：自动执行允许的高风险操作，不再逐次询问；但仍禁止 `execute_office_shell` 等原始 bash/shell 操作。\n\n"
        "请直接回复：`ask`、`plan` 或 `auto`。"
    )


def build_approval_message(state) -> str:
    current_step = get_current_plan_step(state)
    plan = state.get("plan", []) or []
    pending_tool_names = _pending_tool_names(state)
    pending_tool_summary = _describe_pending_tool_calls(state)
    todo_context = _current_todo_context(state)
    if (
        not str(state.get("permission_mode", "") or "").strip().lower()
        and not pending_tool_names
        and current_step is None
        and not plan
        and todo_context is None
    ):
        return build_permission_mode_message(state)
    if current_step is None:
        if pending_tool_names and todo_context is not None:
            step_number, total_steps, description = todo_context
            tool_title = "、".join(pending_tool_names)
            return (
                f"待审批高风险工具调用：{tool_title}\n"
                f"当前步骤 {step_number}/{total_steps}：{description}\n\n"
                f"原因：{state.get('approval_reason') or '需要在执行前确认风险。'}\n"
                "这是高风险操作，是否现在继续？\n"
                f"{pending_tool_summary}\n"
                "请回复“确认执行”继续，或回复“取消”/“稍后再说”终止。"
            )
        plan_lines = "\n".join(
            f"{step['step']}. {step['description']} [risk={step['risk_level']}]"
            for step in plan
        )
        summary_suffix = f"\n{pending_tool_summary}" if pending_tool_summary else ""
        return (
            "该任务已进入 slow path，且包含高风险操作。\n"
            f"原因：{state.get('approval_reason') or '需要在执行前确认风险。'}\n"
            f"计划：\n{plan_lines if plan_lines else '暂无计划'}{summary_suffix}\n\n"
            "请回复“确认执行”继续，或回复“取消”/“稍后再说”终止。"
        )

    total_steps = len(plan)
    if pending_tool_names:
        tool_title = "、".join(pending_tool_names)
        step_context = (
            f"\n当前步骤 {current_step['step']}/{total_steps}：{current_step['description']}"
            if current_step and total_steps
            else ""
        )
        return (
            f"待审批高风险工具调用：{tool_title}{step_context}\n\n"
            f"原因：{state.get('approval_reason') or build_approval_reason(current_step)}\n"
            "这是高风险操作，是否现在继续？\n"
            f"{pending_tool_summary}\n"
            "请回复“确认执行”继续，或回复“取消”/“稍后再说”终止。"
        )

    summary_suffix = f"\n{pending_tool_summary}" if pending_tool_summary else ""
    return (
        f"待执行步骤 {current_step['step']}/{total_steps}：{current_step['description']}\n\n"
        f"原因：{state.get('approval_reason') or build_approval_reason(current_step)}\n"
        f"这是高风险操作，是否现在继续？{summary_suffix}\n请回复“确认执行”继续，或回复“取消”/“稍后再说”终止。"
    )
