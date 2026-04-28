import logging
import re
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from src.config import Settings
from src.db.factory import make_database
from src.models.paper import Paper
from src.repositories.paper import PaperRepository
from src.services.arxiv.factory import make_arxiv_client
from src.services.indexing.factory import make_hybrid_indexing_service
from src.services.metadata_fetcher import make_metadata_fetcher
from src.services.pdf_parser.factory import make_pdf_parser_service

logger = logging.getLogger(__name__)

GENERIC_PAPER_PATTERNS = (
    r"给我找",
    r"帮我找",
    r"帮我推荐",
    r"给我推荐",
    r"推荐",
    r"找",
    r"论文",
    r"papers?",
    r"research",
    r"相关",
    r"方向",
    r"主题",
    r"几篇",
    r"一篇",
    r"两篇",
    r"二篇",
    r"三篇",
    r"四篇",
    r"五篇",
    r"\d+\s*篇",
)

TOKEN_EXPANSIONS = (
    (
        ("无人机", "无人驾驶飞行器", "uav", "drone", "drones", "unmanned aerial vehicle", "quadrotor"),
        ("unmanned aerial vehicle", "UAV", "drone", "quadrotor"),
    ),
    (
        ("vln", "vision language navigation", "vision-language navigation", "视觉语言导航", "视觉-语言导航"),
        ("vision language navigation", "vision-language navigation", "VLN"),
    ),
    (
        ("导航", "navigation", "path planning", "waypoint"),
        ("navigation", "path planning"),
    ),
    (
        ("机器人", "robot", "robotics"),
        ("robotics", "robot"),
    ),
    (
        ("大模型", "大型语言模型", "语言模型", "llm", "large language model"),
        ("large language model", "LLM"),
    ),
    (
        ("幻觉", "hallucination", "hallucinations"),
        ("hallucination", "faithfulness", "factuality"),
    ),
    (
        ("检索增强", "rag", "retrieval augmented generation", "retrieval-augmented generation"),
        ("retrieval augmented generation", "RAG"),
    ),
    (
        ("多模态", "multimodal", "multi-modal"),
        ("multimodal", "multi-modal"),
    ),
    (
        ("视觉语言", "视觉-语言", "vision language", "vision-language"),
        ("vision language", "vision-language"),
    ),
    (
        ("具身智能", "embodied ai", "embodied intelligence"),
        ("embodied AI", "embodied intelligence"),
    ),
    (
        ("强化学习", "reinforcement learning", "rl"),
        ("reinforcement learning", "RL"),
    ),
    (
        ("扩散模型", "diffusion model", "diffusion"),
        ("diffusion model", "diffusion"),
    ),
    (
        ("图神经网络", "gnn", "graph neural network"),
        ("graph neural network", "GNN"),
    ),
)

ENGLISH_STOPWORDS = {
    "find",
    "show",
    "give",
    "me",
    "paper",
    "papers",
    "related",
    "about",
    "for",
    "the",
    "a",
    "an",
    "please",
}

ROBOTICS_HINTS = {
    "uav",
    "drone",
    "quadrotor",
    "unmanned aerial vehicle",
    "vln",
    "vision language navigation",
    "vision-language navigation",
    "navigation",
    "path planning",
    "robot",
    "robotics",
}

LANGUAGE_MODEL_HINTS = {
    "large language model",
    "llm",
    "hallucination",
    "faithfulness",
    "factuality",
    "retrieval augmented generation",
    "rag",
}

VISION_HINTS = {
    "multimodal",
    "multi-modal",
    "vision language",
    "vision-language",
}


@dataclass(frozen=True)
class PaperIngestionResult:
    search_text: str
    arxiv_query: str
    papers_fetched: int
    papers_stored: int
    papers_indexed: int
    chunks_indexed: int
    fetched_arxiv_ids: list[str]


class FeishuPaperIngestionService:
    """Fetch, parse, and index extra arXiv papers for Feishu follow-up searches."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.database = make_database()
        self.arxiv_client = make_arxiv_client()
        self.pdf_parser = make_pdf_parser_service()
        self.metadata_fetcher = make_metadata_fetcher(
            arxiv_client=self.arxiv_client,
            pdf_parser=self.pdf_parser,
            settings=settings,
        )
        self.indexing_service = make_hybrid_indexing_service(settings=settings)

    @classmethod
    def build_local_search_text(cls, query: str) -> str:
        terms = cls._expand_query_terms(query)
        return " ".join(terms).strip() or query.strip()

    @classmethod
    def build_arxiv_query(cls, query: str) -> str:
        terms = cls._expand_query_terms(query)
        if not terms:
            return "cat:cs.RO"

        clauses = []
        for term in terms:
            normalized = term.strip()
            if not normalized:
                continue
            if re.search(r"\s|-", normalized):
                clauses.append(f'all:"{normalized}"')
            else:
                clauses.append(f"all:{normalized}")

        base_query = f"({' OR '.join(dict.fromkeys(clauses))})" if clauses else "cat:cs.RO"

        normalized_terms = {term.lower() for term in terms}
        if normalized_terms & ROBOTICS_HINTS:
            base_query = f"{base_query} AND (cat:cs.RO OR cat:cs.CV OR cat:cs.AI OR cat:cs.LG)"
        elif normalized_terms & LANGUAGE_MODEL_HINTS:
            base_query = f"{base_query} AND (cat:cs.CL OR cat:cs.AI OR cat:cs.LG)"
        elif normalized_terms & VISION_HINTS:
            base_query = f"{base_query} AND (cat:cs.CV OR cat:cs.AI OR cat:cs.LG)"

        return base_query

    @classmethod
    def _expand_query_terms(cls, query: str) -> list[str]:
        normalized = re.sub(r"\s+", " ", query.strip().lower())
        cleaned = normalized
        for pattern in GENERIC_PAPER_PATTERNS:
            cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"[^\w\s\-]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        terms: list[str] = []
        for triggers, expansions in TOKEN_EXPANSIONS:
            if any(trigger in normalized for trigger in triggers):
                terms.extend(expansions)

        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{1,}", cleaned):
            lowered = token.lower()
            if lowered in ENGLISH_STOPWORDS:
                continue
            if token.upper() in {"UAV", "VLN"}:
                terms.append(token.upper())
            else:
                terms.append(lowered)

        if not terms and cleaned:
            terms.append(cleaned)

        deduped: list[str] = []
        seen: set[str] = set()
        for term in terms:
            key = term.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(term)

        return deduped

    async def ingest_missing_papers(
        self,
        query: str,
        requested_count: int,
        existing_ids: Optional[Sequence[str]] = None,
    ) -> PaperIngestionResult:
        existing_id_set = {paper_id for paper_id in (existing_ids or []) if paper_id}
        search_text = self.build_local_search_text(query)
        arxiv_query = self.build_arxiv_query(query)
        fetch_limit = max(self.settings.feishu.auto_ingest_max_results, requested_count + 2)

        logger.info(
            "Auto-ingesting papers for Feishu query '%s' using arXiv query '%s' (limit=%s)",
            query,
            arxiv_query,
            fetch_limit,
        )

        with self.database.get_session() as session:
            fetch_results = await self.metadata_fetcher.fetch_and_process_papers(
                max_results=fetch_limit,
                search_query=arxiv_query,
                process_pdfs=self.settings.feishu.auto_ingest_process_pdfs,
                store_to_db=True,
                db_session=session,
            )

            fetched_ids = [
                arxiv_id
                for arxiv_id in fetch_results.get("fetched_arxiv_ids", [])
                if arxiv_id and arxiv_id not in existing_id_set
            ]
            papers_to_index = self._load_papers_for_indexing(session, fetched_ids)

            index_stats = {
                "papers_processed": 0,
                "total_chunks_indexed": 0,
            }
            if papers_to_index:
                index_stats = await self.indexing_service.index_papers_batch(
                    papers=papers_to_index,
                    replace_existing=True,
                )

        return PaperIngestionResult(
            search_text=search_text,
            arxiv_query=arxiv_query,
            papers_fetched=fetch_results.get("papers_fetched", 0),
            papers_stored=fetch_results.get("papers_stored", 0),
            papers_indexed=index_stats.get("papers_processed", 0),
            chunks_indexed=index_stats.get("total_chunks_indexed", 0),
            fetched_arxiv_ids=fetched_ids,
        )

    def _load_papers_for_indexing(self, session, arxiv_ids: Sequence[str]) -> list[dict[str, Any]]:
        repo = PaperRepository(session)
        papers_data: list[dict[str, Any]] = []
        for arxiv_id in arxiv_ids:
            paper = repo.get_by_arxiv_id(arxiv_id)
            if not paper:
                continue
            papers_data.append(self._serialize_paper(paper))
        return papers_data

    @staticmethod
    def _serialize_paper(paper: Paper) -> dict[str, Any]:
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
