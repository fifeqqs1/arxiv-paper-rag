import logging

from src.config import get_settings
from src.repositories.paper import PaperRepository
from src.services.opensearch.factory import make_opensearch_client_fresh

from .common import get_cached_services

logger = logging.getLogger(__name__)


def prune_local_papers(**context):
    """Delete old papers and their chunks when the local store exceeds the retention limit."""
    settings = get_settings()
    if not settings.paper_retention_enabled:
        logger.info("Paper retention is disabled")
        return {"status": "skipped", "reason": "retention_disabled"}

    max_papers = max(settings.paper_retention_max_papers, 0)
    _arxiv_client, _pdf_parser, database, _metadata_fetcher, _cached_opensearch_client = get_cached_services()
    opensearch_client = make_opensearch_client_fresh(settings)

    with database.get_session() as session:
        paper_repo = PaperRepository(session)
        total_papers = paper_repo.get_count()
        if total_papers <= max_papers:
            logger.info("Paper retention check skipped: total=%s, max=%s", total_papers, max_papers)
            return {
                "status": "ok",
                "total_papers": total_papers,
                "max_papers": max_papers,
                "papers_deleted": 0,
                "chunks_deleted": 0,
            }

        papers_to_delete = paper_repo.get_excess_papers(keep_latest=max_papers, limit=total_papers - max_papers)
        chunks_deleted = 0
        for paper in papers_to_delete:
            chunks_deleted += opensearch_client.delete_paper_chunks(paper.arxiv_id)

        papers_deleted = paper_repo.delete_by_ids([paper.id for paper in papers_to_delete])

    result = {
        "status": "ok",
        "total_papers": total_papers,
        "max_papers": max_papers,
        "papers_deleted": papers_deleted,
        "chunks_deleted": chunks_deleted,
    }
    logger.info(
        "Paper retention complete: deleted %s papers and %s chunks (max_papers=%s)",
        papers_deleted,
        chunks_deleted,
        max_papers,
    )

    ti = context.get("ti")
    if ti:
        ti.xcom_push(key="retention_results", value=result)

    return result
