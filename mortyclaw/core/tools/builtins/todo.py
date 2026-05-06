from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field


class TodoInputItem(BaseModel):
    id: str | None = Field(default=None, description="Todo 项 id；留空时系统会生成。")
    content: str = Field(description="Todo 项描述。")
    status: str = Field(description="状态：pending、in_progress、completed、cancelled。")


class UpdateTodoListArgs(BaseModel):
    items: list[TodoInputItem] = Field(description="要写入的 todo 列表。")
    reason: str = Field(default="", description="这次更新 todo 的原因，可选。")


def update_todo_list_impl(
    *,
    items: list[dict[str, Any]],
    reason: str,
    thread_id: str,
    session_repo,
    todo_state: dict[str, Any],
    build_todo_state_fn,
    merge_tool_written_todos_fn,
    normalize_todos_fn,
) -> str:
    requested_items = [
        item.model_dump() if hasattr(item, "model_dump") else dict(item)
        for item in items
    ]
    if not todo_state.get("items"):
        next_state = build_todo_state_fn(
            requested_items,
            revision=1,
            last_event="todo_tool_init",
        )
        next_state["reason"] = reason or ""
        next_state["plan_snapshot"] = []
        session_repo.save_session_todo_state(thread_id, next_state)
        normalized_items = normalize_todos_fn(next_state.get("items"))
        return json.dumps(
            {
                "success": True,
                "thread_id": thread_id,
                "todo_state": next_state,
                "summary": {
                    "total": len(next_state.get("items", [])),
                    "in_progress": sum(1 for item in normalized_items if item.get("status") == "in_progress"),
                    "completed": sum(1 for item in normalized_items if item.get("status") == "completed"),
                },
            },
            ensure_ascii=False,
        )

    existing_items = todo_state.get("items") or []
    plan_snapshot = todo_state.get("plan_snapshot") or []
    merged_items = merge_tool_written_todos_fn(plan_snapshot, existing_items, requested_items)
    next_revision = int(todo_state.get("revision", 0) or 0) + 1
    next_state = build_todo_state_fn(
        merged_items,
        revision=next_revision,
        last_event="todo_tool",
    )
    next_state["reason"] = reason or ""
    next_state["plan_snapshot"] = plan_snapshot
    session_repo.save_session_todo_state(thread_id, next_state)
    normalized_items = normalize_todos_fn(next_state.get("items"))
    return json.dumps(
        {
            "success": True,
            "thread_id": thread_id,
            "todo_state": next_state,
            "summary": {
                "total": len(next_state.get("items", [])),
                "in_progress": sum(1 for item in normalized_items if item.get("status") == "in_progress"),
                "completed": sum(1 for item in normalized_items if item.get("status") == "completed"),
            },
        },
        ensure_ascii=False,
    )
