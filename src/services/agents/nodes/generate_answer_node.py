import logging
import time
from typing import Dict, List

from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from ..context import Context
from ..models import SourceItem
from ..state import AgentState
from .utils import get_latest_query

logger = logging.getLogger(__name__)


async def ainvoke_generate_answer_step(
    state: AgentState,
    runtime: Runtime[Context],
) -> Dict[str, List[AIMessage]]:
    """Generate final answer using retrieved documents.

    This node generates a comprehensive answer to the
    user's question based on the retrieved context using an LLM.

    :param state: Current agent state
    :param runtime: Runtime context
    :returns: Dictionary with messages containing the generated answer
    """
    logger.info("NODE: generate_answer")
    start_time = time.time()

    # Get question and query to retrieve against
    question = get_latest_query(state["messages"])
    retrieval_query = state.get("rewritten_query") or state.get("original_query") or question
    logger.debug(f"Generating answer for query: {question[:100]}...")

    # Create span for answer generation
    span = None
    if runtime.context.langfuse_enabled and runtime.context.trace:
        try:
            span = runtime.context.langfuse_tracer.create_span(
                trace=runtime.context.trace,
                name="answer_generation",
                input_data={
                    "query": question,
                    "retrieval_query": retrieval_query,
                },
                metadata={
                    "node": "generate_answer",
                    "model": runtime.context.model_name,
                    "temperature": runtime.context.temperature,
                },
            )
            logger.debug("Created Langfuse span for answer generation")
        except Exception as e:
            logger.warning(f"Failed to create span for generate_answer node: {e}")

    try:
        query_embedding = None
        try:
            query_embedding = await runtime.context.embeddings_client.embed_query(retrieval_query)
        except Exception as exc:
            logger.warning(f"Failed to generate embedding in generate_answer node, falling back to BM25: {exc}")

        search_results = runtime.context.opensearch_client.search_unified(
            query=retrieval_query,
            query_embedding=query_embedding,
            size=runtime.context.top_k,
            use_hybrid=query_embedding is not None,
        )

        hits = search_results.get("hits", [])
        chunks = []
        relevant_sources: List[SourceItem] = []
        seen_urls = set()
        chunks_preview = []

        for hit in hits:
            arxiv_id = hit.get("arxiv_id", "")
            chunk_text = hit.get("chunk_text", hit.get("abstract", ""))
            chunks.append({"arxiv_id": arxiv_id, "chunk_text": chunk_text})

            if chunk_text and len(chunks_preview) < 3:
                preview = chunk_text[:500] + "..." if len(chunk_text) > 500 else chunk_text
                chunks_preview.append({"text_preview": preview, "length": len(chunk_text)})

            if not arxiv_id:
                continue

            clean_id = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
            url = f"https://arxiv.org/pdf/{clean_id}.pdf"
            if url in seen_urls:
                continue

            authors = hit.get("authors", "")
            relevant_sources.append(
                SourceItem(
                    arxiv_id=arxiv_id,
                    title=hit.get("title", ""),
                    authors=[author.strip() for author in authors.split(",") if author.strip()],
                    url=url,
                    relevance_score=float(hit.get("score", 0.0) or 0.0),
                )
            )
            seen_urls.add(url)

        sources_count = len(relevant_sources)

        if not chunks:
            answer = "我暂时没有在已检索到的论文内容中找到足够信息来回答这个问题。"
        else:
            logger.info("Invoking stable RAG answer generation")
            rag_response = await runtime.context.ollama_client.generate_rag_answer(
                query=question,
                chunks=chunks,
                model=runtime.context.model_name,
                provider=runtime.context.llm_provider,
            )
            answer = rag_response.get("answer", "").strip() or "暂时没有生成有效回答。"

        logger.info(f"Generated answer of length: {len(answer)} characters")

        # Update span with successful result
        if span:
            execution_time = (time.time() - start_time) * 1000
            runtime.context.langfuse_tracer.end_span(
                span,
                output={
                    "answer_length": len(answer),
                    "sources_used": sources_count,
                },
                metadata={
                    "execution_time_ms": execution_time,
                    "chunks_retrieved": len(chunks),
                    "chunks_used": chunks_preview,
                },
            )

    except Exception as e:
        logger.error(f"LLM answer generation failed: {e}, falling back to error message")

        # Fallback to error message if LLM fails
        answer = f"I apologize, but I encountered an error while generating the answer: {str(e)}\n\nPlease try again or rephrase your question."

        # Update span with error
        if span:
            execution_time = (time.time() - start_time) * 1000
            runtime.context.langfuse_tracer.update_span(
                span,
                output={"error": str(e), "fallback": True},
                metadata={"execution_time_ms": execution_time},
                level="ERROR",
            )
            runtime.context.langfuse_tracer.end_span(span)

    return {
        "messages": [AIMessage(content=answer)],
        "relevant_sources": relevant_sources if 'relevant_sources' in locals() else [],
    }
