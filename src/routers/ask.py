import json
import logging
import time
from typing import Dict, List

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from src.dependencies import CacheDep, EmbeddingsDep, LangfuseDep, OllamaDep, OpenSearchDep
from src.schemas.api.ask import AskRequest, AskResponse
from src.services.langfuse.tracer import RAGTracer

logger = logging.getLogger(__name__)

DIRECT_ARXIV_CHUNKS_PER_PAPER = 6

# Two separate routers - one for regular ask, one for streaming
ask_router = APIRouter(tags=["ask"])
stream_router = APIRouter(tags=["stream"])


async def _prepare_chunks_and_sources(
    request: AskRequest,
    opensearch_client,
    embeddings_service,
    rag_tracer: RAGTracer,
    trace=None,
) -> tuple[List[Dict], List[str], List[str]]:
    """Retrieve and prepare chunks for RAG with clean tracing."""

    # Handle embeddings for hybrid search
    query_embedding = None
    if request.use_hybrid:
        with rag_tracer.trace_embedding(trace, request.query) as embedding_span:
            try:
                query_embedding = await embeddings_service.embed_query(request.query)
                logger.info("Generated query embedding for hybrid search")
            except Exception as e:
                logger.warning(f"Failed to generate embeddings, falling back to BM25: {e}")
                if embedding_span:
                    rag_tracer.tracer.update_span(embedding_span, output={"success": False, "error": str(e)})

    # Search with tracing
    with rag_tracer.trace_search(trace, request.query, request.top_k) as search_span:
        search_results = opensearch_client.search_unified(
            query=request.query,
            query_embedding=query_embedding,
            size=request.top_k,
            from_=0,
            categories=request.categories,
            use_hybrid=request.use_hybrid and query_embedding is not None,
            min_score=0.0,
        )

        # Extract essential data for LLM
        chunks = []
        arxiv_ids = []
        sources_set = set()

        for hit in search_results.get("hits", []):
            arxiv_id = hit.get("arxiv_id", "")

            # Minimal chunk data for LLM
            chunks.append(
                {
                    "arxiv_id": arxiv_id,
                    "chunk_text": hit.get("chunk_text", hit.get("abstract", "")),
                }
            )

            if arxiv_id:
                arxiv_ids.append(arxiv_id)
                arxiv_id_clean = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
                sources_set.add(f"https://arxiv.org/pdf/{arxiv_id_clean}.pdf")

        # End search span with essential metadata
        rag_tracer.end_search(search_span, chunks, arxiv_ids, search_results.get("total", 0))

    return chunks, list(sources_set), arxiv_ids


async def _prepare_chunks_from_arxiv_ids(
    request: AskRequest,
    opensearch_client,
    rag_tracer: RAGTracer,
    trace=None,
) -> tuple[List[Dict], List[str], List[str]]:
    """Load chunks directly from specific papers, bypassing global search."""

    chunks = []
    requested_arxiv_ids = request.arxiv_ids or []
    chunk_limit = request.direct_chunks_per_paper or DIRECT_ARXIV_CHUNKS_PER_PAPER
    sources_set = set()
    loaded_arxiv_ids = []

    with rag_tracer.trace_search(trace, request.query, request.top_k) as search_span:
        for requested_arxiv_id in requested_arxiv_ids:
            paper_chunks = opensearch_client.get_chunks_by_paper(requested_arxiv_id)
            if not paper_chunks:
                logger.warning(f"No indexed chunks found for requested paper {requested_arxiv_id}")
                continue

            loaded_arxiv_ids.append(requested_arxiv_id)
            clean_id = requested_arxiv_id.split("v")[0] if "v" in requested_arxiv_id else requested_arxiv_id
            sources_set.add(f"https://arxiv.org/pdf/{clean_id}.pdf")

            for hit in paper_chunks[:chunk_limit]:
                chunks.append(
                    {
                        "arxiv_id": hit.get("arxiv_id", requested_arxiv_id),
                        "chunk_text": hit.get("chunk_text", hit.get("abstract", "")),
                    }
                )

        rag_tracer.end_search(search_span, chunks, loaded_arxiv_ids, len(chunks))

    return chunks, list(sources_set), loaded_arxiv_ids


@ask_router.post("/ask", response_model=AskResponse)
async def ask_question(
    request: AskRequest,
    opensearch_client: OpenSearchDep,
    embeddings_service: EmbeddingsDep,
    ollama_client: OllamaDep,
    langfuse_tracer: LangfuseDep,
    cache_client: CacheDep,
) -> AskResponse:
    """Clean RAG endpoint with essential tracing and exact match caching."""

    rag_tracer = RAGTracer(langfuse_tracer)
    start_time = time.time()

    with rag_tracer.trace_request("api_user", request.query) as trace:
        try:
            # Check exact cache first
            cached_response = None
            if cache_client:
                try:
                    cached_response = await cache_client.find_cached_response(request)
                    if cached_response:
                        logger.info("Returning cached response for exact query match")
                        return cached_response
                except Exception as e:
                    logger.warning(f"Cache check failed, proceeding with normal flow: {e}")

            # Generate query embedding for hybrid search if needed
            query_embedding = None

            # Retrieve chunks
            if request.arxiv_ids:
                chunks, sources, _ = await _prepare_chunks_from_arxiv_ids(
                    request,
                    opensearch_client,
                    rag_tracer,
                    trace,
                )
            else:
                chunks, sources, _ = await _prepare_chunks_and_sources(
                    request, opensearch_client, embeddings_service, rag_tracer, trace
                )

            if not chunks:
                response = AskResponse(
                    query=request.query,
                    answer="我暂时没有在已检索到的论文内容中找到足够信息来回答这个问题。",
                    sources=[],
                    chunks_used=0,
                    search_mode="direct" if request.arxiv_ids else ("bm25" if not request.use_hybrid else "hybrid"),
                )
                rag_tracer.end_request(trace, response.answer, time.time() - start_time)
                return response

            # Build prompt
            with rag_tracer.trace_prompt_construction(trace, chunks) as prompt_span:
                from src.services.ollama.prompts import RAGPromptBuilder

                prompt_builder = RAGPromptBuilder()

                try:
                    prompt_data = prompt_builder.create_structured_prompt(request.query, chunks)
                    final_prompt = prompt_data["prompt"]
                except Exception:
                    final_prompt = prompt_builder.create_rag_prompt(request.query, chunks)

                rag_tracer.end_prompt(prompt_span, final_prompt)

            # Generate answer
            with rag_tracer.trace_generation(trace, request.model, final_prompt) as gen_span:
                rag_response = await ollama_client.generate_rag_answer(
                    query=request.query,
                    chunks=chunks,
                    model=request.model,
                    provider=request.provider,
                )
                answer = rag_response.get("answer", "Unable to generate answer")
                rag_tracer.end_generation(gen_span, answer, request.model)

            # Prepare response
            response = AskResponse(
                query=request.query,
                answer=answer,
                sources=sources,
                chunks_used=len(chunks),
                search_mode="direct" if request.arxiv_ids else ("bm25" if not request.use_hybrid else "hybrid"),
            )

            rag_tracer.end_request(trace, answer, time.time() - start_time)

            # Store response in exact match cache
            if cache_client:
                try:
                    await cache_client.store_response(request, response)
                except Exception as e:
                    logger.warning(f"Failed to store response in cache: {e}")

            return response

        except Exception as e:
            logger.error(f"Error processing request: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@stream_router.post("/stream")
async def ask_question_stream(
    request: AskRequest,
    opensearch_client: OpenSearchDep,
    embeddings_service: EmbeddingsDep,
    ollama_client: OllamaDep,
    langfuse_tracer: LangfuseDep,
    cache_client: CacheDep,
) -> StreamingResponse:
    """Clean streaming RAG endpoint."""

    async def generate_stream():
        rag_tracer = RAGTracer(langfuse_tracer)
        start_time = time.time()

        with rag_tracer.trace_request("api_user", request.query) as trace:
            try:
                # Check exact cache first
                if cache_client:
                    try:
                        cached_response = await cache_client.find_cached_response(request)
                        if cached_response:
                            logger.info("Returning cached response for exact streaming query match")

                            # Send metadata first (same format as non-cached)
                            metadata_response = {
                                "sources": cached_response.sources,
                                "chunks_used": cached_response.chunks_used,
                                "search_mode": cached_response.search_mode,
                            }
                            yield f"data: {json.dumps(metadata_response)}\n\n"

                            # Stream the cached response in chunks
                            for chunk in cached_response.answer.split():
                                yield f"data: {json.dumps({'chunk': chunk + ' '})}\n\n"

                            # Send completion signal with just the final answer
                            yield f"data: {json.dumps({'answer': cached_response.answer, 'done': True})}\n\n"
                            return
                    except Exception as e:
                        logger.warning(f"Cache check failed, proceeding with normal flow: {e}")

                # Retrieve chunks
                if request.arxiv_ids:
                    chunks, sources, _ = await _prepare_chunks_from_arxiv_ids(
                        request,
                        opensearch_client,
                        rag_tracer,
                        trace,
                    )
                else:
                    chunks, sources, _ = await _prepare_chunks_and_sources(
                        request, opensearch_client, embeddings_service, rag_tracer, trace
                    )

                if not chunks:
                    yield f"data: {json.dumps({'answer': '我暂时没有在已检索到的论文内容中找到足够信息来回答这个问题。', 'sources': [], 'done': True})}\n\n"
                    return

                # Send metadata first
                search_mode = "direct" if request.arxiv_ids else ("bm25" if not request.use_hybrid else "hybrid")
                metadata_response = {"sources": sources, "chunks_used": len(chunks), "search_mode": search_mode}
                yield f"data: {json.dumps(metadata_response)}\n\n"

                # Build prompt
                with rag_tracer.trace_prompt_construction(trace, chunks) as prompt_span:
                    from src.services.ollama.prompts import RAGPromptBuilder

                    prompt_builder = RAGPromptBuilder()
                    final_prompt = prompt_builder.create_rag_prompt(request.query, chunks)
                    rag_tracer.end_prompt(prompt_span, final_prompt)

                # Stream generation
                with rag_tracer.trace_generation(trace, request.model, final_prompt) as gen_span:
                    full_response = ""
                    async for chunk in ollama_client.generate_rag_answer_stream(
                        query=request.query,
                        chunks=chunks,
                        model=request.model,
                        provider=request.provider,
                    ):
                        if chunk.get("response"):
                            text_chunk = chunk["response"]
                            full_response += text_chunk
                            yield f"data: {json.dumps({'chunk': text_chunk})}\n\n"

                        if chunk.get("done", False):
                            rag_tracer.end_generation(gen_span, full_response, request.model)
                            yield f"data: {json.dumps({'answer': full_response, 'done': True})}\n\n"
                            break

                rag_tracer.end_request(trace, full_response, time.time() - start_time)

                # Store response in exact match cache
                if cache_client and full_response:
                    try:
                        search_mode = "bm25" if not request.use_hybrid else "hybrid"
                        response_to_cache = AskResponse(
                            query=request.query,
                            answer=full_response,
                            sources=sources,
                            chunks_used=len(chunks),
                            search_mode=search_mode,
                        )
                        await cache_client.store_response(request, response_to_cache)
                    except Exception as e:
                        logger.warning(f"Failed to store streaming response in cache: {e}")

            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate_stream(), media_type="text/plain", headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )
