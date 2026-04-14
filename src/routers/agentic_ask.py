import logging

from fastapi import APIRouter, HTTPException
from src.dependencies import CacheDep, EmbeddingsDep, LangfuseDep, OllamaDep, OpenSearchDep
from src.routers.ask import ask_question
from src.schemas.api.ask import AgenticAskResponse, AskRequest, FeedbackRequest, FeedbackResponse
from src.services.agents.factory import make_agentic_rag_service

router = APIRouter(prefix="/api/v1", tags=["agentic-rag"])
logger = logging.getLogger(__name__)


def _should_fallback_to_standard_rag(answer: str) -> bool:
    lowered = answer.lower()
    fallback_markers = (
        "encountered an error",
        "please try again or rephrase",
        "failed to generate rag answer",
        "generation failed",
        "暂时无法生成",
        "i apologize, but i can only help with questions about academic research papers",
        "outside my domain of expertise",
    )
    return any(marker in lowered for marker in fallback_markers)


@router.post("/ask-agentic", response_model=AgenticAskResponse)
async def ask_agentic(
    request: AskRequest,
    opensearch_client: OpenSearchDep,
    ollama_client: OllamaDep,
    embeddings_service: EmbeddingsDep,
    langfuse_tracer: LangfuseDep,
    cache_client: CacheDep,
) -> AgenticAskResponse:
    """
    Agentic RAG endpoint with intelligent retrieval and query refinement.

    Features:
    - Decides if retrieval is needed
    - Grades document relevance
    - Rewrites queries if needed
    - Provides reasoning transparency

    The agent will automatically:
    1. Determine if the question requires research paper retrieval
    2. If needed, search for relevant papers
    3. Grade retrieved documents for relevance
    4. Rewrite the query if documents aren't relevant
    5. Generate an answer with citations

    Args:
        request: Question and parameters
        agentic_rag: Injected agentic RAG service

    Returns:
        Answer with sources and reasoning steps

    Raises:
        HTTPException: If processing fails
    """
    try:
        agentic_rag = make_agentic_rag_service(
            opensearch_client=opensearch_client,
            ollama_client=ollama_client,
            embeddings_client=embeddings_service,
            langfuse_tracer=langfuse_tracer,
            top_k=request.top_k,
            use_hybrid=request.use_hybrid,
        )

        result = await agentic_rag.ask(
            query=request.query,
            model=request.model,
            provider=request.provider,
        )

        if _should_fallback_to_standard_rag(result.get("answer", "")):
            logger.warning("Agentic RAG returned an internal error answer, falling back to standard /ask pipeline")
            fallback_response = await ask_question(
                request=request,
                opensearch_client=opensearch_client,
                embeddings_service=embeddings_service,
                ollama_client=ollama_client,
                langfuse_tracer=langfuse_tracer,
                cache_client=cache_client,
            )
            return AgenticAskResponse(
                query=fallback_response.query,
                answer=fallback_response.answer,
                sources=fallback_response.sources,
                chunks_used=fallback_response.chunks_used,
                search_mode=fallback_response.search_mode,
                reasoning_steps=["Agentic pipeline degraded gracefully to standard RAG"],
                retrieval_attempts=result.get("retrieval_attempts", 0),
                trace_id=result.get("trace_id"),
            )

        return AgenticAskResponse(
            query=result["query"],
            answer=result["answer"],
            sources=result.get("sources", []),
            chunks_used=request.top_k,
            search_mode="hybrid" if request.use_hybrid else "bm25",
            reasoning_steps=result.get("reasoning_steps", []),
            retrieval_attempts=result.get("retrieval_attempts", 0),
            trace_id=result.get("trace_id"),
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Agentic endpoint failed, falling back to standard /ask pipeline: {e}")
        try:
            fallback_response = await ask_question(
                request=request,
                opensearch_client=opensearch_client,
                embeddings_service=embeddings_service,
                ollama_client=ollama_client,
                langfuse_tracer=langfuse_tracer,
                cache_client=cache_client,
            )
            return AgenticAskResponse(
                query=fallback_response.query,
                answer=fallback_response.answer,
                sources=fallback_response.sources,
                chunks_used=fallback_response.chunks_used,
                search_mode=fallback_response.search_mode,
                reasoning_steps=["Agentic pipeline failed, fell back to standard RAG"],
                retrieval_attempts=0,
                trace_id=None,
            )
        except Exception as fallback_exc:
            raise HTTPException(status_code=500, detail=f"Error processing question: {str(fallback_exc)}")


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    langfuse_tracer: LangfuseDep,
) -> FeedbackResponse:
    """
    Submit user feedback for an agentic RAG response.

    This endpoint allows users to rate the quality of answers and provide
    optional comments. Feedback is tracked in Langfuse for continuous improvement.

    Args:
        request: Feedback data including trace_id, score, and optional comment
        langfuse_tracer: Injected Langfuse tracer service

    Returns:
        FeedbackResponse indicating success or failure

    Raises:
        HTTPException: If feedback submission fails
    """
    try:
        if not langfuse_tracer:
            raise HTTPException(
                status_code=503,
                detail="Langfuse tracing is disabled. Cannot submit feedback."
            )

        success = langfuse_tracer.submit_feedback(
            trace_id=request.trace_id,
            score=request.score,
            comment=request.comment,
        )

        if success:
            # Flush to ensure feedback is sent immediately
            langfuse_tracer.flush()

            return FeedbackResponse(
                success=True,
                message="Feedback recorded successfully"
            )
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to submit feedback to Langfuse"
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error submitting feedback: {str(e)}"
        )
