from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import tools_condition

from ..context import AgentState


def compile_agent_workflow(
    *,
    fast_tool_node,
    slow_tool_node,
    router_node,
    planner_node,
    finalizer_node,
    approval_gate_node,
    execution_guard_node,
    fast_agent_node,
    slow_agent_node,
    reviewer_node,
    route_after_router_fn,
    route_after_fast_agent_fn,
    route_after_planner_fn,
    route_after_approval_gate_fn,
    route_after_execution_guard_fn,
    route_after_slow_agent_fn,
    route_after_reviewer_fn,
    checkpointer,
):
    workflow = StateGraph(AgentState)

    workflow.add_node("router", router_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("finalizer", finalizer_node)
    workflow.add_node("approval_gate", approval_gate_node)
    workflow.add_node("execution_guard", execution_guard_node)
    workflow.add_node("fast_agent", fast_agent_node)
    workflow.add_node("slow_agent", slow_agent_node)
    workflow.add_node("reviewer", reviewer_node)
    workflow.add_node("fast_tools", fast_tool_node)
    workflow.add_node("slow_tools", slow_tool_node)

    workflow.add_edge(START, "router")
    workflow.add_conditional_edges(
        "router",
        route_after_router_fn,
        {
            "fast": "fast_agent",
            "planner": "planner",
            "slow": "approval_gate",
        },
    )
    workflow.add_conditional_edges(
        "planner",
        route_after_planner_fn,
        {
            "fast": "fast_agent",
            "slow": "approval_gate",
        },
    )
    workflow.add_conditional_edges(
        "approval_gate",
        route_after_approval_gate_fn,
        {
            "execute": "execution_guard",
            "end": END,
        },
    )
    workflow.add_conditional_edges(
        "execution_guard",
        route_after_execution_guard_fn,
        {
            "execute": "slow_agent",
            "replan": "planner",
            "end": END,
        },
    )

    workflow.add_conditional_edges(
        "fast_agent",
        route_after_fast_agent_fn,
        {
            "planner": "planner",
            "tools": "fast_tools",
            "__end__": END,
        },
    )
    workflow.add_conditional_edges(
        "slow_agent",
        route_after_slow_agent_fn,
        {
            "tools": "slow_tools",
            "approval": "approval_gate",
            "replan": "planner",
            "retry": "slow_agent",
            "finalize": "finalizer",
            "reviewer": "reviewer",
            "end": END,
        },
    )
    workflow.add_conditional_edges(
        "reviewer",
        route_after_reviewer_fn,
        {
            "execute": "slow_agent",
            "approval": "approval_gate",
            "replan": "planner",
            "finalize": "finalizer",
            "end": END,
        },
    )
    workflow.add_edge("finalizer", END)

    workflow.add_edge("fast_tools", "fast_agent")
    workflow.add_edge("slow_tools", "slow_agent")

    return workflow.compile(checkpointer=checkpointer)
