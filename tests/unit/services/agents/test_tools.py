import json
from unittest.mock import Mock

import pytest
from src.services.agents.tools import create_retriever_tool


@pytest.mark.asyncio
async def test_create_retriever_tool_basic(mock_opensearch_client, mock_jina_embeddings_client):
    """Test basic retriever tool creation and invocation."""
    tool = create_retriever_tool(
        opensearch_client=mock_opensearch_client,
        embeddings_client=mock_jina_embeddings_client,
        top_k=2,
        use_hybrid=True,
    )

    # Verify tool properties
    assert tool.name == "retrieve_papers"
    assert "Search and return relevant arXiv research papers" in tool.description

    # Invoke tool
    result = await tool.ainvoke({"query": "machine learning"})
    payload = json.loads(result)
    hits = payload["hits"]

    # Verify result
    assert isinstance(payload, dict)
    assert len(hits) == 2

    # Verify first document
    first_doc = hits[0]
    assert first_doc["chunk_text"] == "Transformers are neural network architectures based on self-attention mechanisms."
    assert first_doc["arxiv_id"] == "1706.03762"
    assert first_doc["title"] == "Attention Is All You Need"
    assert first_doc["score"] == 0.95
    assert payload["plan"]["subqueries"] == ["machine learning"]

    # Verify embeddings were generated
    mock_jina_embeddings_client.embed_query.assert_called_once_with("machine learning")

    # Verify search was called correctly
    mock_opensearch_client.search_unified.assert_called_once()
    call_args = mock_opensearch_client.search_unified.call_args
    assert call_args.kwargs["query"] == "machine learning"
    assert call_args.kwargs["size"] == 2  # search_unified uses 'size', not 'top_k'
    assert call_args.kwargs["use_hybrid"] is True


@pytest.mark.asyncio
async def test_retriever_tool_empty_results(mock_opensearch_client, mock_jina_embeddings_client):
    """Test retriever tool with no results."""
    mock_opensearch_client.search_unified = Mock(return_value={"hits": []})

    tool = create_retriever_tool(
        opensearch_client=mock_opensearch_client,
        embeddings_client=mock_jina_embeddings_client,
    )

    result = await tool.ainvoke({"query": "nonexistent topic"})
    payload = json.loads(result)

    assert payload["hits"] == []
    assert payload["plan"]["subqueries"] == ["nonexistent topic"]


@pytest.mark.asyncio
async def test_retriever_tool_custom_top_k(mock_opensearch_client, mock_jina_embeddings_client):
    """Test retriever tool with custom top_k parameter."""
    tool = create_retriever_tool(
        opensearch_client=mock_opensearch_client,
        embeddings_client=mock_jina_embeddings_client,
        top_k=5,
        use_hybrid=False,
    )

    await tool.ainvoke({"query": "test query"})

    call_args = mock_opensearch_client.search_unified.call_args
    # search_unified uses 'size' parameter, not 'top_k'
    assert call_args.kwargs["size"] == 5
    assert call_args.kwargs["use_hybrid"] is False


@pytest.mark.asyncio
async def test_retriever_tool_metadata_fields(mock_opensearch_client, mock_jina_embeddings_client):
    """Test that all expected metadata fields are present."""
    mock_opensearch_client.search_unified = Mock(return_value={
        "hits": [
            {
                "chunk_text": "Test content",
                "arxiv_id": "2301.00001",
                "title": "Test Paper",
                "authors": "Author One, Author Two",
                "score": 0.95,
                "section_name": "Introduction",
            }
        ]
    })

    tool = create_retriever_tool(
        opensearch_client=mock_opensearch_client,
        embeddings_client=mock_jina_embeddings_client,
    )

    result = await tool.ainvoke({"query": "test"})
    payload = json.loads(result)

    doc = payload["hits"][0]
    assert "arxiv_id" in doc
    assert "title" in doc
    assert "authors" in doc
    assert "score" in doc
    assert "chunk_text" in doc
    assert "section_title" in doc


@pytest.mark.asyncio
async def test_retriever_tool_decomposes_and_boosts_sections(mock_opensearch_client, mock_jina_embeddings_client):
    """Test complex questions use section-aware retrieval planning."""
    mock_opensearch_client.search_unified = Mock(
        return_value={
            "hits": [
                {
                    "chunk_id": "method",
                    "chunk_text": "The method uses a controller.",
                    "arxiv_id": "2601.00001",
                    "title": "Policy Controller",
                    "authors": "Author One",
                    "score": 4.0,
                    "section_title": "Method",
                    "section_type": "method",
                },
                {
                    "chunk_id": "experiment",
                    "chunk_text": "Experiments compare against baselines and report success rates.",
                    "arxiv_id": "2601.00001",
                    "title": "Policy Controller",
                    "authors": "Author One",
                    "score": 1.0,
                    "section_title": "Experiments",
                    "section_type": "experiment",
                },
            ]
        }
    )

    tool = create_retriever_tool(
        opensearch_client=mock_opensearch_client,
        embeddings_client=mock_jina_embeddings_client,
        top_k=1,
        use_hybrid=True,
    )

    result = await tool.ainvoke({"query": "实验部分用了什么方法，有什么效果？"})
    payload = json.loads(result)

    assert mock_opensearch_client.search_unified.call_count > 1
    assert mock_opensearch_client.search_unified.call_args_list[0].kwargs["section_types"] == ["experiment", "method"]
    assert payload["hits"][0]["section_type"] == "experiment"
    assert payload["plan"]["section_types"] == ["experiment", "method"]
