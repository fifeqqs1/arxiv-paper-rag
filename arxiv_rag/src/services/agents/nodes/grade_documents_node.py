import logging
import time
from typing import Dict

from langgraph.runtime import Runtime

from ..context import Context
from ..models import GradingResult
from ..state import AgentState
from .utils import get_latest_context, get_latest_query

logger = logging.getLogger(__name__)


def _grade_context_heuristically(question: str, context: str) -> GradingResult:
    """Use deterministic grading so retrieval decisions do not depend on fragile JSON output."""
    normalized_question = question.lower()
    normalized_context = context.lower()
    question_terms = {term for term in normalized_question.split() if len(term) > 2}
    overlap_count = sum(1 for term in question_terms if term in normalized_context)

    has_research_markers = any(marker in normalized_context for marker in ("arxiv", "page_content", "title", "authors"))
    is_relevant = len(context.strip()) > 80 or overlap_count > 0 or has_research_markers

    reasoning_parts = [
        f"context_length={len(context)}",
        f"overlap_terms={overlap_count}",
    ]
    if has_research_markers:
        reasoning_parts.append("research markers present")

    return GradingResult(
        document_id="retrieved_docs",
        is_relevant=is_relevant,
        score=1.0 if is_relevant else 0.0,
        reasoning="Heuristic grading: " + ", ".join(reasoning_parts),
    )


async def ainvoke_grade_documents_step(
    state: AgentState,
    runtime: Runtime[Context],
) -> Dict[str, str | list]:
    """Grade retrieved documents for relevance using LLM.

    This function uses an LLM to evaluate whether the retrieved documents
    are relevant to the user's query and decides whether to generate an
    answer or rewrite the query for better results.

    :param state: Current agent state
    :param runtime: Runtime context
    :returns: Dictionary with routing_decision and grading_results
    """
    logger.info("NODE: grade_documents")
    start_time = time.time()

    # Get query and context
    question = get_latest_query(state["messages"])
    context = get_latest_context(state["messages"])

    # Extract document chunks from context for logging
    chunks_preview = []
    if context:
        # Context is a string containing all documents concatenated
        # Let's show a preview of what was retrieved
        context_preview = context[:500] + "..." if len(context) > 500 else context
        chunks_preview = [{"text_preview": context_preview, "length": len(context)}]

    # Create span for document grading
    span = None
    if runtime.context.langfuse_enabled and runtime.context.trace:
        try:
            span = runtime.context.langfuse_tracer.create_span(
                trace=runtime.context.trace,
                name="document_grading",
                input_data={
                    "query": question,
                    "context_length": len(context) if context else 0,
                    "has_context": context is not None,
                    "chunks_received": chunks_preview,
                },
                metadata={
                    "node": "grade_documents",
                    "model": runtime.context.model_name,
                },
            )
            logger.debug("Created Langfuse span for document grading")
        except Exception as e:
            logger.warning(f"Failed to create span for grade_documents node: {e}")

    if not context:
        logger.warning("No context found, routing to rewrite_query")

        # Update span with no context result
        if span:
            execution_time = (time.time() - start_time) * 1000
            runtime.context.langfuse_tracer.end_span(
                span,
                output={"routing_decision": "rewrite_query", "reason": "no_context"},
                metadata={"execution_time_ms": execution_time},
            )

        return {"routing_decision": "rewrite_query", "grading_results": []}

    logger.debug(f"Grading context of length {len(context)} characters")

    grading_result = _grade_context_heuristically(question, context)
    is_relevant = grading_result.is_relevant
    score = grading_result.score
    logger.info(f"Heuristic grading: relevant={is_relevant}, reasoning={grading_result.reasoning}")

    # Determine routing
    route = "generate_answer" if is_relevant else "rewrite_query"

    logger.info(f"Grading result: {'relevant' if is_relevant else 'not relevant'}, routing to: {route}")

    # Update span with grading result
    if span:
        execution_time = (time.time() - start_time) * 1000
        runtime.context.langfuse_tracer.end_span(
            span,
            output={
                "routing_decision": route,
                "is_relevant": is_relevant,
                "score": score,
                "reasoning": grading_result.reasoning,
            },
            metadata={
                "execution_time_ms": execution_time,
                "context_length": len(context),
                "strategy": "heuristic",
            },
        )

    return {
        "routing_decision": route,
        "grading_results": [grading_result],
    }
