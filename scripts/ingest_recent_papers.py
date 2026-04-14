#!/usr/bin/env python3
"""One-off arXiv ingestion helper for local testing.

Fetches papers from arXiv for a given date or date range, processes PDFs,
stores the results in PostgreSQL, and indexes chunks into OpenSearch.
"""

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import desc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


logger = logging.getLogger("ingest_recent_papers")

VALID_BOOL_ENV_VALUES = {"1", "0", "true", "false", "yes", "no", "on", "off", ""}


def parse_args() -> argparse.Namespace:
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")

    parser = argparse.ArgumentParser(
        description="Fetch a few arXiv papers, store them in Postgres, and index them into OpenSearch."
    )
    parser.add_argument(
        "--target-date",
        default=yesterday,
        help="Single arXiv submission date to fetch, in YYYYMMDD format. Default: yesterday.",
    )
    parser.add_argument("--from-date", help="Start date in YYYYMMDD format. Overrides --target-date when set.")
    parser.add_argument("--to-date", help="End date in YYYYMMDD format. Overrides --target-date when set.")
    parser.add_argument(
        "--search-query",
        help="Custom arXiv query for latest matching papers, e.g. 'all:UAV OR all:drone'. Skips date filtering when used.",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        help="Maximum number of papers to fetch. Defaults to ARXIV__MAX_RESULTS from .env.",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Skip PDF parsing. Faster, but indexed content will usually be empty.",
    )
    return parser.parse_args()


def validate_date(label: str, value: str) -> str:
    try:
        datetime.strptime(value, "%Y%m%d")
    except ValueError as exc:
        raise SystemExit(f"{label} must be in YYYYMMDD format, got: {value}") from exc
    return value


def serialize_paper(paper: Any) -> dict[str, Any]:
    return {
        "id": str(paper.id),
        "arxiv_id": paper.arxiv_id,
        "title": paper.title,
        "authors": paper.authors,
        "abstract": paper.abstract,
        "categories": paper.categories,
        "published_date": paper.published_date,
        "raw_text": paper.raw_text,
        "sections": paper.sections,
    }


def apply_local_host_overrides() -> None:
    if Path("/.dockerenv").exists():
        return

    db_url = os.environ.get("POSTGRES_DATABASE_URL", "")
    if "@postgres:" in db_url:
        os.environ["POSTGRES_DATABASE_URL"] = db_url.replace("@postgres:", "@localhost:")
    else:
        os.environ.setdefault("POSTGRES_DATABASE_URL", "postgresql+psycopg2://rag_user:rag_password@localhost:5432/rag_db")

    opensearch_host = os.environ.get("OPENSEARCH__HOST", "")
    if opensearch_host == "http://opensearch:9200":
        os.environ["OPENSEARCH__HOST"] = "http://localhost:9200"
    else:
        os.environ.setdefault("OPENSEARCH__HOST", "http://localhost:9200")

    ollama_host = os.environ.get("OLLAMA_HOST", "")
    if ollama_host == "http://ollama:11434":
        os.environ["OLLAMA_HOST"] = "http://localhost:11435"
    else:
        os.environ.setdefault("OLLAMA_HOST", "http://localhost:11435")


async def run() -> int:
    debug_env = os.environ.get("DEBUG", "")
    if debug_env.lower() not in VALID_BOOL_ENV_VALUES:
        os.environ["DEBUG"] = "true"
    apply_local_host_overrides()

    from src.config import get_settings
    from src.db.factory import make_database
    from src.models.paper import Paper
    from src.services.arxiv.factory import make_arxiv_client
    from src.services.indexing.factory import make_hybrid_indexing_service
    from src.services.metadata_fetcher import make_metadata_fetcher
    from src.services.opensearch.factory import make_opensearch_client_fresh
    from src.services.pdf_parser.factory import make_pdf_parser_service

    args = parse_args()
    settings = get_settings()

    search_query = args.search_query.strip() if args.search_query else ""
    from_date = args.from_date
    to_date = args.to_date
    if search_query:
        from_date = None
        to_date = None
    elif from_date or to_date:
        if not (from_date and to_date):
            raise SystemExit("--from-date and --to-date must be provided together.")
        from_date = validate_date("from-date", from_date)
        to_date = validate_date("to-date", to_date)
    else:
        from_date = validate_date("target-date", args.target_date)
        to_date = from_date

    max_results = args.max_results or settings.arxiv.max_results
    process_pdfs = not args.no_pdf

    logger.info(
        "Starting one-off ingestion: search_query=%s, from_date=%s, to_date=%s, category=%s, max_results=%s, process_pdfs=%s",
        search_query or "<default>",
        from_date or "<none>",
        to_date or "<none>",
        settings.arxiv.search_category,
        max_results,
        process_pdfs,
    )

    database = make_database()
    arxiv_client = make_arxiv_client()
    pdf_parser = make_pdf_parser_service()
    metadata_fetcher = make_metadata_fetcher(arxiv_client, pdf_parser, settings=settings)
    indexing_service = make_hybrid_indexing_service(settings=settings)
    opensearch_client = make_opensearch_client_fresh(settings)

    try:
        with database.get_session() as session:
            fetch_results = await metadata_fetcher.fetch_and_process_papers(
                max_results=max_results,
                from_date=from_date,
                to_date=to_date,
                search_query=search_query or None,
                process_pdfs=process_pdfs,
                store_to_db=True,
                db_session=session,
            )

            papers_stored = fetch_results.get("papers_stored", 0)
            papers = session.query(Paper).order_by(desc(Paper.created_at)).limit(papers_stored).all()

        if fetch_results.get("papers_fetched", 0) == 0:
            logger.warning("No papers were fetched. Try a different date or category.")
            return 0

        if papers_stored == 0:
            logger.warning("Papers were fetched but none were stored. Check the ingestion logs above.")
            return 1

        papers_data = [serialize_paper(paper) for paper in papers]
        index_stats = await indexing_service.index_papers_batch(papers=papers_data, replace_existing=True)
        index_count = opensearch_client.client.count(index=opensearch_client.index_name)["count"]

        print("")
        print("Ingestion complete")
        if search_query:
            print(f"- Search query: {search_query}")
        else:
            print(f"- Date range: {from_date} -> {to_date}")
        print(f"- Category: {settings.arxiv.search_category}")
        print(f"- Papers fetched: {fetch_results['papers_fetched']}")
        print(f"- Papers stored: {papers_stored}")
        print(f"- PDFs parsed: {fetch_results['pdfs_parsed']}")
        print(f"- Papers indexed: {index_stats['papers_processed']}")
        print(f"- Chunks indexed: {index_stats['total_chunks_indexed']}")
        print(f"- OpenSearch total chunks: {index_count}")
        return 0
    finally:
        shutdown = getattr(database, "shutdown", None)
        if callable(shutdown):
            shutdown()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(main())
