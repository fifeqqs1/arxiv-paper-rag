import logging

from fastapi import APIRouter, HTTPException, Request
from src.schemas.api.feishu import FeishuLocalReplyRequest, FeishuLocalReplyResponse
from src.services.feishu.bot import FeishuLocalRuntime
from src.services.feishu.factory import make_local_feishu_bot

logger = logging.getLogger(__name__)

router = APIRouter(tags=["feishu-local"])


def _get_local_feishu_bot(request: Request):
    existing_bot = getattr(request.app.state, "local_feishu_bot", None)
    if existing_bot is not None:
        return existing_bot

    cache_client = getattr(request.app.state, "cache_client", None)
    redis_client = getattr(cache_client, "redis", None) if cache_client is not None else None
    local_runtime = FeishuLocalRuntime(
        opensearch_client=request.app.state.opensearch_client,
        embeddings_service=request.app.state.embeddings_service,
        ollama_client=request.app.state.ollama_client,
        langfuse_tracer=getattr(request.app.state, "langfuse_tracer", None),
        cache_client=cache_client,
    )
    bot = make_local_feishu_bot(
        settings=request.app.state.settings,
        redis_client=redis_client,
        database=getattr(request.app.state, "database", None),
        local_runtime=local_runtime,
    )
    request.app.state.local_feishu_bot = bot
    return bot


@router.post("/feishu/reply", response_model=FeishuLocalReplyResponse, response_model_exclude_none=True)
async def feishu_local_reply(
    payload: FeishuLocalReplyRequest,
    request: Request,
) -> FeishuLocalReplyResponse:
    """Expose the Feishu conversation logic as a local API for non-Feishu clients."""
    bot = _get_local_feishu_bot(request)

    try:
        if payload.eval_debug:
            debug_reply = await bot.build_local_reply_debug_async(session_id=payload.session_id, query=payload.query)
            return FeishuLocalReplyResponse(
                session_id=payload.session_id,
                query=payload.query,
                answer=debug_reply.answer,
                contexts=debug_reply.contexts,
                sources=debug_reply.sources,
                intent=debug_reply.intent,
                rewritten_query=debug_reply.rewritten_query,
                route=debug_reply.route,
            )
        answer = await bot.build_local_reply_async(session_id=payload.session_id, query=payload.query)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Failed to build local Feishu reply: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to build Feishu reply") from exc

    return FeishuLocalReplyResponse(
        session_id=payload.session_id,
        query=payload.query,
        answer=answer,
    )
