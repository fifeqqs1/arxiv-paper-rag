from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from src.routers.ask import _prepare_chunks_and_sources
from src.schemas.api.ask import AskRequest


class _DummyTracer:
    tracer = SimpleNamespace(update_span=lambda *args, **kwargs: None)

    def trace_embedding(self, *args, **kwargs):
        return nullcontext(None)

    def trace_search(self, *args, **kwargs):
        return nullcontext(None)

    def end_search(self, *args, **kwargs):
        return None


class _FakeOpenSearch:
    def __init__(self):
        self.settings = SimpleNamespace(
            retrieval_candidate_multiplier=4,
            retrieval_max_candidates=10,
            retrieval_max_subqueries=3,
            retrieval_rerank_enabled=True,
            retrieval_section_aware=True,
        )
        self.calls = []

    def search_unified(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "total": 2,
            "hits": [
                {
                    "chunk_id": "method",
                    "arxiv_id": "2601.00001v1",
                    "chunk_text": "The method uses a controller.",
                    "section_title": "Method",
                    "section_type": "method",
                    "score": 4.0,
                },
                {
                    "chunk_id": "experiment",
                    "arxiv_id": "2601.00001v1",
                    "chunk_text": "Experiments compare against baselines and report results.",
                    "section_title": "Experiments",
                    "section_type": "experiment",
                    "score": 1.0,
                },
            ],
        }


@pytest.mark.asyncio
async def test_prepare_chunks_decomposes_and_reranks_section_aware_query():
    opensearch = _FakeOpenSearch()
    request = AskRequest(query="实验部分用了什么方法，有什么效果？", top_k=1, use_hybrid=False)

    chunks, sources, arxiv_ids = await _prepare_chunks_and_sources(
        request=request,
        opensearch_client=opensearch,
        embeddings_service=None,
        rag_tracer=_DummyTracer(),
    )

    assert chunks[0]["chunk_id"] == "experiment"
    assert chunks[0]["section_type"] == "experiment"
    assert sources == ["https://arxiv.org/pdf/2601.00001.pdf"]
    assert arxiv_ids == ["2601.00001v1"]
    assert opensearch.calls[0]["section_types"] == ["experiment", "method"]
