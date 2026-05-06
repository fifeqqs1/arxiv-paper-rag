import json
import os

from langchain_core.messages import HumanMessage


ROUTE_CLASSIFIER_MODEL = "qwen3.5-flash"
ROUTE_CLASSIFIER_CONFIDENCE_THRESHOLD = 0.7


def get_route_classifier_model() -> str:
    return (
        os.getenv("ROUTE_CLASSIFIER_MODEL", "").strip()
        or os.getenv("DEFAULT_MODEL", "").strip()
        or ROUTE_CLASSIFIER_MODEL
    )


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


def classify_route_with_llm(query: str, llm) -> dict | None:
    if not query or llm is None:
        return None

    classifier_prompt = f"""
你是一个任务复杂度路由器。请判断下面的请求应该走 fast 还是 slow。

规则：
- slow：多步骤项目/代码/架构分析、review、对比、排查、给出修改建议，或任何需要显式计划的复杂任务
- fast：简单事实问答、简短解释、单次直接回答即可完成的问题

只输出一个 JSON 对象，不要输出 Markdown，不要补充解释：
{{
  "route": "fast" 或 "slow",
  "reason": "一句简短原因",
  "confidence": 0 到 1 之间的小数
}}

用户请求：
{query}
""".strip()

    try:
        response = llm.invoke([HumanMessage(content=classifier_prompt)], config={"callbacks": []})
    except TypeError:
        try:
            response = llm.invoke([HumanMessage(content=classifier_prompt)])
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

    payload = _extract_json_object(content if isinstance(content, str) else str(content))
    if payload is None:
        return None

    route = str(payload.get("route", "")).strip().lower()
    if route not in {"fast", "slow"}:
        return None

    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    confidence = max(0.0, min(1.0, confidence))
    reason = str(payload.get("reason", "")).strip()
    return {
        "route": route,
        "reason": reason,
        "confidence": confidence,
    }
