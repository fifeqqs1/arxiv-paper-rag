import json
import logging

from langchain_core.tools import tool
from src.services.embeddings.jina_client import JinaEmbeddingsClient
from src.services.opensearch.client import OpenSearchClient
from src.services.retrieval import retrieve_relevant_hits

logger = logging.getLogger(__name__)


def create_retriever_tool(
    opensearch_client: OpenSearchClient,
    embeddings_client: JinaEmbeddingsClient,
    top_k: int = 3,
    use_hybrid: bool = True,
):
    """Create a retriever tool that wraps OpenSearch service.

    :param opensearch_client: Existing OpenSearch service
    :param embeddings_client: Existing Jina embeddings service
    :param top_k: Number of chunks to retrieve
    :param use_hybrid: Use hybrid search (BM25 + vector)
    :returns: LangChain tool for retrieving papers
    """

    @tool
    async def retrieve_papers(query: str) -> str:
        """Search and return relevant arXiv research papers.

        Use this tool when the user asks about:
        - Machine learning concepts or techniques
        - Deep learning architectures
        - Natural language processing
        - Computer vision methods
        - AI research topics
        - Specific algorithms or models

        :param query: The search query describing what papers to find
        :returns: List of relevant paper excerpts with metadata
        """
        logger.info(f"Retrieving papers for query: {query[:100]}...")
        logger.debug(f"Search mode: {'hybrid' if use_hybrid else 'bm25'}, top_k: {top_k}")

        logger.debug("Searching OpenSearch with retrieval planning")
        hits, plan = await retrieve_relevant_hits(
            query=query,
            opensearch_client=opensearch_client,
            embeddings_client=embeddings_client,
            top_k=top_k,
            use_hybrid=use_hybrid,
        )
        logger.debug("Retrieval plan: subqueries=%s, section_types=%s", plan.subqueries, plan.section_types)

        logger.info(f"Found {len(hits)} documents from OpenSearch")
        payload_hits = []
        for hit in hits:
            payload_hits.append(
                {
                    "arxiv_id": hit.get("arxiv_id", ""),
                    "title": hit.get("title", ""),
                    "authors": hit.get("authors", ""),
                    "score": hit.get("score", 0.0),
                    "chunk_text": hit.get("chunk_text", ""),
                    "section_title": hit.get("section_title", hit.get("section_name", "")),
                    "section_path": hit.get("section_path", []),
                    "section_type": hit.get("section_type", ""),
                    "search_mode": "hybrid" if use_hybrid else "bm25",
                    "top_k": top_k,
                }
            )

        logger.info(f"✓ Retrieved {len(payload_hits)} papers successfully")
        return json.dumps(
            {
                "hits": payload_hits,
                "plan": {
                    "subqueries": plan.subqueries,
                    "section_types": plan.section_types,
                },
            },
            ensure_ascii=False,
        )

    return retrieve_papers
