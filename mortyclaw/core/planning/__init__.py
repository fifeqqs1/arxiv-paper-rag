import json

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import tools_condition

from ..error_policy import looks_like_explicit_failure_text
from .rules import (
    VALID_STEP_INTENTS,
    build_execution_plan,
    build_rule_execution_plan,
    classify_step_risk,
    infer_step_intent,
    looks_like_file_write_request,
    normalize_plan_steps,
    step_matches_shell_action,
    step_matches_skill_usage,
    step_matches_task_creation,
    step_matches_task_deletion,
    step_matches_task_modification,
    step_matches_test_action,
)
from .tool_scope import enforce_slow_step_tool_scope, select_tools_for_current_step


def _extract_json_object(text: str) -> dict | None:
    if not isinstance(text, str):
        return None

    stripped = text.strip()
    candidates = [stripped]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(stripped[start:end + 1])

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def build_llm_planner_prompt(
    query: str,
    state: dict | None,
    route_decision: dict,
    current_project_path: str,
    available_tool_names: list[str],
) -> str:
    state = state or {}
    return f"""
你是 MortyClaw 的任务规划器。请根据用户请求决定：
1. 这个任务是否仍然需要 slow-path 执行
2. 如果需要，请输出线性的结构化 Todo/执行步骤

规则：
- 只输出一个 JSON 对象，不要输出 Markdown，不要补充解释
- 高风险任务不能逃避规划；如果 route_locked=true，不允许把任务降级为 fast
- step 必须是线性的、可执行的、互相有先后顺序
- 一次只允许一个当前执行步骤，步骤之间不要并行
- 如果任务需要代码修改、文件写入或命令执行，要明确标记相应 intent
- 不要生成空步骤，也不要把“和用户确认”写成步骤；审批由系统处理
- 如果当前任务是 mixed_research_task，必须先生成 paper_research 步骤提炼论文方法，再生成 repo/代码检查步骤，最后生成差异综合或结论步骤

允许的 intent:
- analyze
- read
- paper_research
- code_edit
- file_write
- shell_execute
- test_verify
- summarize
- report

输出 JSON schema：
{{
  "route": "fast" 或 "slow",
  "goal": "任务目标",
  "reason": "一句简短原因",
  "confidence": 0 到 1 的小数,
  "steps": [
    {{
      "description": "步骤描述",
      "intent": "analyze|read|paper_research|code_edit|file_write|shell_execute|test_verify|summarize|report",
      "risk_level": "low|medium|high",
      "success_criteria": "完成标准",
      "verification_hint": "验证建议",
      "needs_tools": true
    }}
  ]
}}

当前路由信息：
- route={route_decision.get("route", "")}
- risk_level={route_decision.get("risk_level", "")}
- route_locked={bool(route_decision.get("route_locked", False))}
- route_source={route_decision.get("route_source", "")}
- route_reason={route_decision.get("route_reason", "")}

当前项目路径：
{current_project_path or "(未设置)"}

可用工具：
{", ".join(sorted(available_tool_names)) if available_tool_names else "(无)"}

如果这是 replan，请参考现有上下文：
- 原始目标：{state.get("goal", "")}
- 上次规划来源：{state.get("plan_source", "")}
- 当前 replan 原因：{state.get("replan_reason", "")}
- 已完成步骤摘要：{json.dumps(state.get("step_results", [])[-4:], ensure_ascii=False)}

用户请求：
{query}
""".strip()


def parse_llm_planner_payload(text: str) -> dict | None:
    return _extract_json_object(text)


def normalize_llm_plan_payload(
    payload: dict,
    *,
    route_locked: bool,
    fallback_risk_level: str,
) -> dict | None:
    if not isinstance(payload, dict):
        return None

    route = str(payload.get("route") or "slow").strip().lower()
    if route not in {"fast", "slow"}:
        route = "slow"
    if route_locked:
        route = "slow"

    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    goal = str(payload.get("goal") or "").strip()
    reason = str(payload.get("reason") or "").strip()
    steps = normalize_plan_steps(
        payload.get("steps") or [],
        fallback_risk_level=fallback_risk_level,
    )

    return {
        "route": route,
        "goal": goal,
        "reason": reason,
        "confidence": confidence,
        "steps": steps,
    }


def build_plan_with_llm(
    llm,
    query: str,
    state: dict | None,
    route_decision: dict,
    available_tool_names: list[str],
) -> dict | None:
    if llm is None or not query:
        return None

    planner_prompt = build_llm_planner_prompt(
        query=query,
        state=state,
        route_decision=route_decision,
        current_project_path=str((state or {}).get("current_project_path") or ""),
        available_tool_names=available_tool_names,
    )
    try:
        response = llm.invoke([HumanMessage(content=planner_prompt)], config={"callbacks": []})
    except TypeError:
        try:
            response = llm.invoke([HumanMessage(content=planner_prompt)])
        except Exception:
            return None
    except Exception:
        return None

    content = getattr(response, "content", "")
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    payload = parse_llm_planner_payload(content if isinstance(content, str) else str(content))
    if payload is None:
        return None
    return normalize_llm_plan_payload(
        payload,
        route_locked=bool(route_decision.get("route_locked", False)),
        fallback_risk_level=str(route_decision.get("risk_level", "medium") or "medium"),
    )


def build_default_replan_input(state: dict | None) -> str:
    state = state or {}
    goal = str(state.get("goal", "") or "").strip()
    replan_reason = str(state.get("replan_reason", "") or state.get("last_error", "")).strip()
    completed_steps = [
        step for step in (state.get("step_results", []) or [])
        if isinstance(step, dict) and step.get("outcome") == "completed"
    ]
    current_step = get_current_plan_step(state)
    completed_text = "；".join(
        f"{item.get('step')}. {item.get('description')}"
        for item in completed_steps[-5:]
    )
    failing_text = ""
    if current_step is not None:
        failing_text = f"当前失败/阻塞步骤：{current_step.get('description', '')}"
    parts = [
        f"原始目标：{goal}",
        f"已完成步骤：{completed_text or '暂无'}",
        failing_text,
        f"重规划原因：{replan_reason or '当前步骤未达成目标'}",
    ]
    return "\n".join(part for part in parts if part).strip()


def get_current_plan_step(state) -> dict | None:
    plan = state.get("plan", []) or []
    current_step_index = state.get("current_step_index", 0)
    if 0 <= current_step_index < len(plan):
        return plan[current_step_index]
    return None


def update_plan_step(plan: list[dict], step_index: int, *, status: str) -> list[dict]:
    updated_plan = [dict(step) for step in (plan or [])]
    if 0 <= step_index < len(updated_plan):
        updated_plan[step_index]["status"] = status
    return updated_plan


def looks_like_step_failure(content: str | None) -> bool:
    return looks_like_explicit_failure_text(content)


def route_after_slow_agent(state) -> str:
    tool_route = tools_condition(state)
    if tool_route == "tools":
        return "tools"
    if state.get("pending_approval") or state.get("run_status") == "awaiting_step_approval":
        return "approval"
    if state.get("run_status") == "replan_requested":
        return "replan"
    if (state.get("slow_execution_mode", "") or "").strip().lower() == "autonomous":
        if state.get("run_status") == "retrying":
            return "retry"
        if state.get("run_status") in {"done", "failed"}:
            return "finalize"
        if state.get("run_status") == "fallback_review":
            return "reviewer"
        return "end"
    return "reviewer"


def route_after_planner(state) -> str:
    return "fast" if state.get("route") == "fast" else "slow"


def route_after_reviewer(state) -> str:
    run_status = state.get("run_status")
    if state.get("pending_approval") or run_status == "awaiting_step_approval":
        return "approval"
    if run_status in {"next_step", "retrying"}:
        return "execute"
    if run_status == "replan_requested":
        return "replan"
    if run_status in {"done", "failed"}:
        return "finalize"
    return "end"
