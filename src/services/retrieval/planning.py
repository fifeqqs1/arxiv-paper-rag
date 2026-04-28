import re
from dataclasses import dataclass

SECTION_INTENT_HINTS = {
    "experiment": (
        "实验",
        "效果",
        "指标",
        "结果",
        "评估",
        "验证",
        "基线",
        "对比了哪些",
        "哪些方法进行对比",
        "baseline",
        "baselines",
        "experiment",
        "experiments",
        "evaluation",
        "results",
        "ablation",
    ),
    "method": (
        "方法",
        "模型",
        "算法",
        "架构",
        "框架",
        "流程",
        "怎么做",
        "method",
        "methods",
        "approach",
        "model",
        "algorithm",
        "architecture",
    ),
    "limitation": ("局限", "不足", "缺点", "limitation", "limitations", "failure"),
    "conclusion": ("结论", "总结", "conclusion", "conclusions"),
    "related_work": ("相关工作", "已有工作", "related work", "prior work"),
}

SECTION_QUERY_EXPANSIONS = {
    "experiment": "experiment evaluation results baselines ablation benchmark metrics",
    "method": "method approach model architecture algorithm framework",
    "limitation": "limitation discussion failure future work",
    "conclusion": "conclusion results summary",
    "related_work": "related work prior work literature",
}


@dataclass(frozen=True)
class RetrievalPlan:
    original_query: str
    subqueries: list[str]
    section_types: list[str]


def build_retrieval_plan(query: str, max_subqueries: int = 4, section_aware: bool = True) -> RetrievalPlan:
    """Build a small retrieval plan without involving an LLM."""
    normalized = " ".join(query.split())
    section_types = _detect_section_types(normalized) if section_aware else []

    subqueries = [normalized] if normalized else []
    subqueries.extend(_split_complex_query(normalized))

    for section_type in section_types:
        expansion = SECTION_QUERY_EXPANSIONS.get(section_type)
        if expansion:
            subqueries.append(f"{normalized} {expansion}")

    deduped: list[str] = []
    seen: set[str] = set()
    for subquery in subqueries:
        compact = " ".join(subquery.split())
        key = compact.lower()
        if not compact or key in seen:
            continue
        seen.add(key)
        deduped.append(compact)
        if len(deduped) >= max(1, max_subqueries):
            break

    return RetrievalPlan(original_query=normalized, subqueries=deduped, section_types=section_types)


def _detect_section_types(query: str) -> list[str]:
    lowered = query.lower()
    detected: list[str] = []
    for section_type, hints in SECTION_INTENT_HINTS.items():
        if any(hint in query or hint in lowered for hint in hints):
            detected.append(section_type)
    return detected


def _split_complex_query(query: str) -> list[str]:
    if len(query) < 24:
        return []

    parts = re.split(r"[，,；;？?。]\s*|\s+(?:and|or|plus)\s+", query, flags=re.IGNORECASE)
    expanded_parts: list[str] = []
    for part in parts:
        expanded_parts.extend(re.split(r"(?:以及|还有|并且|同时|和哪些|有什么|用了什么)", part))

    return [part.strip(" ：:") for part in expanded_parts if len(part.strip()) >= 6]
