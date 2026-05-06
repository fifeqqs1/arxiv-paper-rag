from typing import Annotated, NotRequired, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class WorkingMemoryState(TypedDict, total=False):
    goal: str
    plan: list[dict]
    current_step_index: int
    planner_required: bool
    route_locked: bool
    route_source: str
    route_reason: str
    route_confidence: float
    plan_source: str
    replan_reason: str
    pending_approval: bool
    approval_reason: str
    recent_tool_results: list[dict]
    last_error: str
    last_error_kind: str
    last_recovery_action: str
    current_project_path: str
    current_mode: str
    permission_mode: str
    run_status: str
    todos: list[dict]
    active_todos: list[dict]
    slow_execution_mode: str
    pending_tool_calls: list[dict]
    pending_execution_snapshot: dict
    execution_guard_status: str
    execution_guard_reason: str


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    summary: str
    working_memory: NotRequired[WorkingMemoryState]
    route: str
    goal: str
    complexity: str
    risk_level: str
    planner_required: bool
    route_locked: bool
    route_source: str
    route_reason: str
    route_confidence: float
    plan_source: str
    replan_reason: str
    plan: list[dict]
    current_step_index: int
    step_results: list[dict]
    pending_approval: bool
    approval_granted: bool
    approval_prompted: bool
    approval_reason: str
    permission_mode: NotRequired[str]
    permission_prompted: NotRequired[bool]
    last_error: str
    last_error_kind: str
    last_recovery_action: str
    retry_count: int
    max_retries: int
    todos: NotRequired[list[dict]]
    active_todos: NotRequired[list[dict]]
    todo_revision: NotRequired[int]
    todo_needs_announcement: NotRequired[bool]
    last_todo_tool_call_id: NotRequired[str]
    pending_tool_calls: NotRequired[list[dict]]
    pending_execution_snapshot: NotRequired[dict]
    slow_execution_mode: NotRequired[str]
    execution_guard_status: NotRequired[str]
    execution_guard_reason: NotRequired[str]
    final_answer: str
    run_status: str
    current_project_path: NotRequired[str]


def build_working_memory_snapshot(
    state: AgentState,
    *,
    recent_tool_results_limit: int = 3,
) -> WorkingMemoryState:
    step_results = list(state.get("step_results", []) or [])
    return {
        "goal": state.get("goal", ""),
        "plan": [dict(step) for step in (state.get("plan", []) or [])],
        "current_step_index": state.get("current_step_index", 0),
        "planner_required": state.get("planner_required", False),
        "route_locked": state.get("route_locked", False),
        "route_source": state.get("route_source", ""),
        "route_reason": state.get("route_reason", ""),
        "route_confidence": state.get("route_confidence", 0.0),
        "plan_source": state.get("plan_source", ""),
        "replan_reason": state.get("replan_reason", ""),
        "pending_approval": state.get("pending_approval", False),
        "approval_reason": state.get("approval_reason", ""),
        "recent_tool_results": step_results[-recent_tool_results_limit:],
        "last_error": state.get("last_error", ""),
        "last_error_kind": state.get("last_error_kind", ""),
        "last_recovery_action": state.get("last_recovery_action", ""),
        "current_project_path": state.get("current_project_path", ""),
        "current_mode": state.get("route", ""),
        "permission_mode": state.get("permission_mode", ""),
        "run_status": state.get("run_status", ""),
        "todos": [dict(item) for item in (state.get("todos", []) or []) if isinstance(item, dict)],
        "active_todos": [dict(item) for item in (state.get("active_todos", state.get("todos", [])) or []) if isinstance(item, dict)],
        "slow_execution_mode": state.get("slow_execution_mode", ""),
        "pending_tool_calls": [dict(item) for item in (state.get("pending_tool_calls", []) or []) if isinstance(item, dict)],
        "pending_execution_snapshot": dict(state.get("pending_execution_snapshot", {}) or {}),
        "execution_guard_status": state.get("execution_guard_status", ""),
        "execution_guard_reason": state.get("execution_guard_reason", ""),
    }
