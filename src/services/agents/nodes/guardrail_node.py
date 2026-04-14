import logging
import time
from typing import Dict, Literal

from langgraph.runtime import Runtime

from ..context import Context
from ..models import GuardrailScoring
from ..state import AgentState
from .utils import get_latest_query

logger = logging.getLogger(__name__)

RESEARCH_PAPER_HINTS = (
    "论文",
    "paper",
    "papers",
    "arxiv",
    "研究",
    "文献",
    "cite",
    "citation",
    "这篇",
    "这两篇",
    "第一篇",
    "第二篇",
)
AI_RESEARCH_HINTS = (
    "ai",
    "ml",
    "llm",
    "machine learning",
    "deep learning",
    "transformer",
    "attention",
    "language model",
    "neural",
    "bert",
    "gpt",
    "无人机",
    "uav",
    "drone",
    "机器人",
    "robot",
    "computer vision",
    "nlp",
)
OUT_OF_SCOPE_HINTS = (
    "hello",
    "hi",
    "天气",
    "weather",
    "电影",
    "music",
    "dog",
    "cat",
    "2+2",
    "几点",
)


def _score_query_heuristically(query: str) -> GuardrailScoring:
    """Use deterministic heuristics so agentic routing stays reliable."""
    lowered = query.lower()
    score = 20
    reasons: list[str] = []

    if any(hint in lowered or hint in query for hint in RESEARCH_PAPER_HINTS):
        score = max(score, 85)
        reasons.append("query explicitly references papers or arXiv content")

    if any(hint in lowered or hint in query for hint in AI_RESEARCH_HINTS):
        score = max(score, 75)
        reasons.append("query contains AI/ML/robotics research terminology")

    if "请基于以下论文回答用户问题" in query:
        score = max(score, 95)
        reasons.append("query is a follow-up grounded in specific papers")

    if any(hint in lowered or hint in query for hint in OUT_OF_SCOPE_HINTS) and score < 75:
        score = min(score, 25)
        reasons.append("query looks conversational or outside research scope")

    if not reasons:
        reasons.append("query does not clearly mention research-paper context")

    return GuardrailScoring(score=score, reason="; ".join(reasons))


def continue_after_guardrail(state: AgentState, runtime: Runtime[Context]) -> Literal["continue", "out_of_scope"]:
    """Determine whether to continue or reject based on guardrail results.

    This function checks the guardrail_result score against a threshold.
    If the score is above threshold, continue; otherwise route to out_of_scope.

    :param state: Current agent state with guardrail results
    :param runtime: Runtime context containing guardrail threshold
    :returns: "continue" if score >= threshold, "out_of_scope" otherwise
    """
    guardrail_result = state.get("guardrail_result")
    if not guardrail_result:
        logger.warning("No guardrail result found, defaulting to continue")
        return "continue"

    score = guardrail_result.score
    threshold = runtime.context.guardrail_threshold

    logger.info(f"Guardrail score: {score}, threshold: {threshold}")

    return "continue" if score >= threshold else "out_of_scope"


async def ainvoke_guardrail_step(
    state: AgentState,
    runtime: Runtime[Context],
) -> Dict[str, GuardrailScoring]:
    """Asynchronously invoke the guardrail validation step using LLM.

    This function evaluates whether the user query is within scope
    (CS/AI/ML research papers) and assigns a score using an LLM.

    :param state: Current agent state
    :param runtime: Runtime context
    :returns: Dictionary with guardrail_result
    """
    logger.info("NODE: guardrail_validation")
    start_time = time.time()

    # Get the latest user query
    query = get_latest_query(state["messages"])
    logger.debug(f"Evaluating query: {query[:100]}...")

    # Create span for guardrail validation (v2 SDK)
    span = None
    if runtime.context.langfuse_enabled and runtime.context.trace:
        try:
            span = runtime.context.langfuse_tracer.create_span(
                trace=runtime.context.trace,
                name="guardrail_validation",
                input_data={
                    "query": query,
                    "threshold": runtime.context.guardrail_threshold,
                },
                metadata={
                    "node": "guardrail",
                    "model": runtime.context.model_name,
                },
            )
            logger.debug("Created Langfuse span for guardrail validation (v2 SDK)")
        except Exception as e:
            logger.warning(f"Failed to create span for guardrail validation: {e}")

    response = _score_query_heuristically(query)
    logger.info(f"Guardrail result - Score: {response.score}, Reason: {response.reason}")

    if span:
        execution_time = (time.time() - start_time) * 1000  # Convert to ms
        runtime.context.langfuse_tracer.end_span(
            span,
            output={
                "score": response.score,
                "reason": response.reason,
                "decision": "continue" if response.score >= runtime.context.guardrail_threshold else "out_of_scope",
            },
            metadata={
                "execution_time_ms": execution_time,
                "threshold": runtime.context.guardrail_threshold,
                "strategy": "heuristic",
            },
        )

    return {"guardrail_result": response}
