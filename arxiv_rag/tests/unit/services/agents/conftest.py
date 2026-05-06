from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from src.services.agents.context import Context


@pytest.fixture(autouse=True)
def valid_debug_env(monkeypatch):
    monkeypatch.setenv("DEBUG", "true")


@pytest.fixture
def mock_opensearch_client():
    client = Mock()
    client.settings = SimpleNamespace(
        retrieval_candidate_multiplier=1,
        retrieval_max_candidates=10,
        retrieval_max_subqueries=4,
        retrieval_rerank_enabled=True,
        retrieval_section_aware=True,
    )
    client.search_unified = Mock(
        return_value={
            "total": 2,
            "hits": [
                {
                    "chunk_id": "transformer-1",
                    "chunk_text": "Transformers are neural network architectures based on self-attention mechanisms.",
                    "arxiv_id": "1706.03762",
                    "title": "Attention Is All You Need",
                    "authors": "Ashish Vaswani, Noam Shazeer",
                    "score": 0.95,
                    "section_title": "Model Architecture",
                    "section_type": "method",
                },
                {
                    "chunk_id": "bert-1",
                    "chunk_text": "BERT uses bidirectional transformer encoders for language understanding.",
                    "arxiv_id": "1810.04805",
                    "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                    "authors": "Jacob Devlin, Ming-Wei Chang",
                    "score": 0.89,
                    "section_title": "Experiments",
                    "section_type": "experiment",
                },
            ],
        }
    )
    client.get_chunks_by_paper = Mock(return_value=[])
    return client


@pytest.fixture
def mock_jina_embeddings_client():
    client = Mock()
    client.embed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
    return client


@pytest.fixture
def mock_ollama_client():
    client = Mock()
    client.generate = AsyncMock(
        return_value={"response": "What are the key concepts in transformer neural network architectures?"}
    )
    client.generate_rag_answer = AsyncMock(
        return_value={"answer": "Based on the papers, transformers are neural network architectures."}
    )
    client.create_llm = Mock()
    return client


@pytest.fixture
def test_context(mock_opensearch_client, mock_ollama_client, mock_jina_embeddings_client):
    return Context(
        ollama_client=mock_ollama_client,
        opensearch_client=mock_opensearch_client,
        embeddings_client=mock_jina_embeddings_client,
        langfuse_tracer=None,
        langfuse_enabled=False,
        top_k=2,
        use_hybrid=True,
        max_retrieval_attempts=2,
        guardrail_threshold=60,
    )


@pytest.fixture
def sample_human_message():
    return HumanMessage(content="What is machine learning?")


@pytest.fixture
def sample_ai_message():
    return AIMessage(content="Machine learning is a field of AI.")


@pytest.fixture
def sample_tool_message():
    return ToolMessage(
        content="Transformers are neural network architectures based on self-attention mechanisms.",
        name="retrieve_papers",
        tool_call_id="retrieve_1",
    )
