from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, TypedDict


TodoStatus = str
ACTIVE_TODO_STATUSES = {"pending", "in_progress"}
VALID_TODO_STATUSES = {"pending", "in_progress", "completed", "cancelled"}


class TodoItem(TypedDict, total=False):
    id: str
    content: str
    status: TodoStatus
    source_step: int | None


class TodoState(TypedDict, total=False):
    items: list[TodoItem]
    revision: int
    updated_at: str
    last_event: str


def todo_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def should_enable_todos(
    route: str | None,
    plan: list[dict] | None,
    *,
    execution_mode: str = "",
    todos: list[dict[str, Any]] | None = None,
) -> bool:
    if (route or "") != "slow":
        return False
    if (execution_mode or "").strip().lower() == "autonomous":
        return bool(normalize_todos(todos))
    return len(plan or []) >= 2


def _normalize_content(value: Any) -> str:
    text = str(value or "").strip()
    return text or "(未命名任务)"


def _normalize_status(value: Any) -> str:
    status = str(value or "pending").strip().lower()
    return status if status in VALID_TODO_STATUSES else "pending"


def _normalize_item(item: dict[str, Any], *, fallback_id: str) -> TodoItem:
    source_step = item.get("source_step")
    if isinstance(source_step, str) and source_step.isdigit():
        source_step = int(source_step)
    elif not isinstance(source_step, int):
        source_step = None

    normalized_id = str(item.get("id") or "").strip() or fallback_id
    return {
        "id": normalized_id,
        "content": _normalize_content(item.get("content")),
        "status": _normalize_status(item.get("status")),
        "source_step": source_step,
    }


def normalize_todos(items: list[dict[str, Any]] | None) -> list[TodoItem]:
    items = items or []
    normalized: list[TodoItem] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        fallback_id = f"todo-{index}"
        normalized_item = _normalize_item(item, fallback_id=fallback_id)
        if normalized_item["id"] in seen_ids:
            normalized_item["id"] = f"{normalized_item['id']}-{index}"
        seen_ids.add(normalized_item["id"])
        normalized.append(normalized_item)

    in_progress_indices = [idx for idx, item in enumerate(normalized) if item.get("status") == "in_progress"]
    if len(in_progress_indices) > 1:
        first = in_progress_indices[0]
        for idx in in_progress_indices[1:]:
            normalized[idx]["status"] = "pending"
        normalized[first]["status"] = "in_progress"
    elif not in_progress_indices:
        first_pending = next((idx for idx, item in enumerate(normalized) if item.get("status") == "pending"), None)
        if first_pending is not None:
            normalized[first_pending]["status"] = "in_progress"

    return normalized


def plan_to_todos(plan: list[dict] | None, current_step_index: int) -> list[TodoItem]:
    plan = [dict(step) for step in (plan or []) if isinstance(step, dict)]
    todos: list[TodoItem] = []
    for index, step in enumerate(plan):
        step_number = int(step.get("step") or (index + 1))
        step_status = str(step.get("status") or "pending").strip().lower()
        if step_status == "completed" or index < current_step_index:
            todo_status = "completed"
        elif index == current_step_index:
            todo_status = "in_progress"
        elif step_status == "cancelled":
            todo_status = "cancelled"
        else:
            todo_status = "pending"
        todos.append({
            "id": f"step-{step_number}",
            "content": _normalize_content(step.get("description")),
            "status": todo_status,
            "source_step": step_number,
        })
    return normalize_todos(todos)


def build_todo_state(
    items: list[dict[str, Any]] | None,
    *,
    revision: int = 1,
    last_event: str = "",
) -> TodoState:
    return {
        "items": normalize_todos(items),
        "revision": max(1, int(revision or 1)),
        "updated_at": todo_timestamp(),
        "last_event": (last_event or "").strip(),
    }


def build_todo_state_from_plan(
    plan: list[dict] | None,
    current_step_index: int,
    *,
    revision: int = 1,
    last_event: str = "",
) -> TodoState:
    return build_todo_state(
        plan_to_todos(plan, current_step_index),
        revision=revision,
        last_event=last_event,
    )


def _preserve_completed_items(existing_todos: list[dict[str, Any]]) -> list[TodoItem]:
    preserved: list[TodoItem] = []
    for index, item in enumerate(existing_todos, start=1):
        normalized = _normalize_item(item, fallback_id=f"completed-{index}")
        if normalized.get("status") == "completed":
            preserved.append(normalized)
    return preserved


def merge_tool_written_todos(
    existing_plan: list[dict] | None,
    existing_todos: list[dict[str, Any]] | None,
    requested_todos: list[dict[str, Any]] | None,
) -> list[TodoItem]:
    existing_plan = [dict(step) for step in (existing_plan or []) if isinstance(step, dict)]
    existing_todos = normalize_todos(existing_todos)
    requested_todos = requested_todos or []

    completed_items = _preserve_completed_items(existing_todos)
    completed_by_id = {item["id"]: item for item in completed_items}
    existing_by_id = {item["id"]: item for item in existing_todos}
    merged_items: list[TodoItem] = []
    mentioned_ids: set[str] = set()

    for index, raw_item in enumerate(requested_todos, start=1):
        if not isinstance(raw_item, dict):
            continue
        candidate = _normalize_item(raw_item, fallback_id=f"todo-{index}")
        existing = existing_by_id.get(candidate["id"])
        if candidate["id"] in completed_by_id:
            merged_items.append(completed_by_id[candidate["id"]])
            mentioned_ids.add(candidate["id"])
            continue
        if existing and existing.get("source_step") is not None and candidate.get("source_step") is None:
            candidate["source_step"] = existing.get("source_step")
        merged_items.append(candidate)
        mentioned_ids.add(candidate["id"])

    items = [
        item for item in completed_items
        if item["id"] not in mentioned_ids
    ]
    items.extend(merged_items)
    if not items and existing_plan:
        items = plan_to_todos(existing_plan, 0)
    return normalize_todos(items)


def todos_to_plan(
    existing_plan: list[dict] | None,
    normalized_todos: list[dict[str, Any]] | None,
    risk_fallback,
) -> tuple[list[dict[str, Any]], int]:
    existing_plan = [dict(step) for step in (existing_plan or []) if isinstance(step, dict)]
    normalized_todos = normalize_todos(normalized_todos)
    existing_by_source = {
        int(step.get("step")): step
        for step in existing_plan
        if str(step.get("step", "")).isdigit()
    }

    rebuilt_plan: list[dict[str, Any]] = []
    current_step_index = 0
    current_found = False
    for index, todo in enumerate(normalized_todos, start=1):
        source_step = todo.get("source_step")
        source_plan = existing_by_source.get(int(source_step)) if isinstance(source_step, int) else None
        status = str(todo.get("status") or "pending")
        if status not in VALID_TODO_STATUSES:
            status = "pending"
        plan_status = "pending"
        if status == "completed":
            plan_status = "completed"
        elif status == "in_progress":
            plan_status = "pending"
            current_step_index = index - 1
            current_found = True
        elif status == "cancelled":
            plan_status = "cancelled"

        description = todo.get("content") or (source_plan or {}).get("description") or ""
        risk_level = (source_plan or {}).get("risk_level") or risk_fallback(description)
        rebuilt_plan.append({
            "step": index,
            "description": description,
            "status": plan_status,
            "risk_level": risk_level,
            "intent": (source_plan or {}).get("intent") or "analyze",
            "success_criteria": (source_plan or {}).get("success_criteria") or "",
            "verification_hint": (source_plan or {}).get("verification_hint") or "",
            "needs_tools": bool((source_plan or {}).get("needs_tools", True)),
        })

    if not current_found:
        first_pending = next((idx for idx, step in enumerate(rebuilt_plan) if step.get("status") not in {"completed", "cancelled"}), None)
        if first_pending is not None:
            current_step_index = first_pending
            normalized_todos[first_pending]["status"] = "in_progress"
        elif rebuilt_plan:
            current_step_index = max(0, len(rebuilt_plan) - 1)

    return rebuilt_plan, current_step_index


def render_todo_for_prompt(items: list[dict[str, Any]] | None) -> str:
    normalized = normalize_todos(items)
    if not normalized:
        return ""

    lines: list[str] = []
    in_progress = [item for item in normalized if item.get("status") == "in_progress"][:1]
    pending = [item for item in normalized if item.get("status") == "pending"][:4]
    completed = [item for item in normalized if item.get("status") == "completed"][:2]

    if in_progress:
        lines.append("进行中")
        for item in in_progress:
            lines.append(f"- [in_progress] {item['content']}")
    if pending:
        lines.append("待处理")
        for item in pending:
            lines.append(f"- [pending] {item['content']}")
    if completed:
        lines.append("已完成")
        for item in completed:
            lines.append(f"- [completed] {item['content']}")
    return "\n".join(lines).strip()


def render_todo_for_chat(items: list[dict[str, Any]] | None, *, title: str = "当前 Todo") -> str:
    normalized = normalize_todos(items)
    if not normalized:
        return ""

    markers = {
        "pending": "[pending]",
        "in_progress": "[in_progress]",
        "completed": "[completed]",
        "cancelled": "[cancelled]",
    }
    lines = [title]
    for item in normalized[:6]:
        lines.append(f"{markers.get(item.get('status', ''), '[pending]')} {item.get('content', '')}")
    return "\n".join(lines).strip()


def copy_todo_items(items: list[dict[str, Any]] | None) -> list[TodoItem]:
    return deepcopy(normalize_todos(items))
