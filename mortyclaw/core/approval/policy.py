import re


APPROVAL_YES_RESPONSES = {
    "确认",
    "确认执行",
    "继续",
    "继续执行",
    "同意",
    "批准",
    "开始执行",
    "可以",
    "yes",
    "ok",
    "y",
}

APPROVAL_NO_RESPONSES = {
    "取消",
    "停止",
    "不用了",
    "算了",
    "拒绝",
    "no",
    "n",
}

PERMISSION_MODE_ALIASES = {
    "ask": "ask",
    "询问": "ask",
    "确认后执行": "ask",
    "plan": "plan",
    "只读": "plan",
    "readonly": "plan",
    "read-only": "plan",
    "auto": "auto",
    "自动": "auto",
    "自动执行": "auto",
}


def normalize_reply_text(query: str | None) -> str:
    return re.sub(r"[\s，。！？!?,；;：:]+", "", (query or "").strip().lower())


def is_affirmative_approval_response(query: str | None) -> bool:
    return normalize_reply_text(query) in APPROVAL_YES_RESPONSES


def is_negative_approval_response(query: str | None) -> bool:
    return normalize_reply_text(query) in APPROVAL_NO_RESPONSES


def parse_permission_mode_response(query: str | None) -> str:
    return PERMISSION_MODE_ALIASES.get(normalize_reply_text(query), "")


def step_requires_approval(step: dict | None) -> bool:
    return bool(step) and step.get("risk_level") == "high"


def build_approval_reason(step: dict | None) -> str:
    if not step:
        return "计划包含写入/删除/命令执行等高风险操作。"
    return f"步骤 {step['step']} 包含高风险操作：{step['description']}"


def route_after_approval_gate(state) -> str:
    if state.get("run_status") in {"waiting_user", "cancelled"}:
        return "end"
    return "execute"
