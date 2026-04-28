import logging
from typing import Any

from .planning import RetrievalPlan, build_retrieval_plan
from .reranker import Reranker

logger = logging.getLogger(__name__)

DEFAULT_RETRIEVAL_CANDIDATE_MULTIPLIER = 4
DEFAULT_RETRIEVAL_MAX_CANDIDATES = 50
DEFAULT_RETRIEVAL_MAX_SUBQUERIES = 4


def _setting_value(settings: Any, name: str, default: Any) -> Any:
    value = getattr(settings, name, default) if settings is not None else default
    # unittest.mock.Mock creates child mocks for any attribute; treat them as missing.
    if value.__class__.__module__.startswith("unittest.mock"):
        return default
    return value


def _int_setting(settings: Any, name: str, default: int) -> int:
    try:
        return int(_setting_value(settings, name, default))
    except (TypeError, ValueError):
        return default


def _bool_setting(settings: Any, name: str, default: bool) -> bool:
    value = _setting_value(settings, name, default)
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def candidate_size(top_k: int, settings: Any = None) -> int:
    multiplier = _int_setting(settings, "retrieval_candidate_multiplier", DEFAULT_RETRIEVAL_CANDIDATE_MULTIPLIER)
    max_candidates = _int_setting(settings, "retrieval_max_candidates", DEFAULT_RETRIEVAL_MAX_CANDIDATES)
    return min(max(top_k * max(1, multiplier), top_k), max(top_k, max_candidates))


async def retrieve_relevant_hits(
    *,
    query: str,
    opensearch_client: Any,
    embeddings_client: Any = None,
    top_k: int = 3,
    use_hybrid: bool = True,
    categories: list[str] | None = None,
    from_: int = 0,
    min_score: float = 0.0,
) -> tuple[list[dict[str, Any]], RetrievalPlan]:
    """Retrieve candidates with query decomposition and local reranking."""
    settings = getattr(opensearch_client, "settings", None)
    plan = build_retrieval_plan(
        query,
        max_subqueries=_int_setting(settings, "retrieval_max_subqueries", DEFAULT_RETRIEVAL_MAX_SUBQUERIES),
        section_aware=_bool_setting(settings, "retrieval_section_aware", True),
    )
    size = candidate_size(top_k, settings)
    all_hits: list[dict[str, Any]] = []

    for subquery in plan.subqueries:
        query_embedding = None
        if use_hybrid and embeddings_client is not None:
            try:
                query_embedding = await embeddings_client.embed_query(subquery)
            except Exception as exc:
                logger.warning("Failed to embed retrieval subquery, falling back to BM25: %s", exc)

        search_results = opensearch_client.search_unified(
            query=subquery,
            query_embedding=query_embedding,
            size=size,
            from_=from_,
            categories=categories,
            use_hybrid=use_hybrid and query_embedding is not None,
            min_score=min_score,
            section_types=plan.section_types,
        )
        hits = search_results.get("hits", [])
        for hit in hits:
            hit["retrieval_subquery"] = subquery
        all_hits.extend(hits)

    if _bool_setting(settings, "retrieval_rerank_enabled", True):
        return (
            Reranker().rerank(
                query=query,
                hits=all_hits,
                top_k=top_k,
                section_types=plan.section_types,
                subqueries=plan.subqueries,
            ),
            plan,
        )

    return all_hits[:top_k], plan
