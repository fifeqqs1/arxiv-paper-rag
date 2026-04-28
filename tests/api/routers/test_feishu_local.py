from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from src.main import app


@pytest.fixture
async def client():
    app.state.settings = SimpleNamespace()
    app.state.database = None
    app.state.cache_client = None
    app.state.opensearch_client = object()
    app.state.embeddings_service = object()
    app.state.ollama_client = object()
    app.state.langfuse_tracer = None
    if hasattr(app.state, "local_feishu_bot"):
        delattr(app.state, "local_feishu_bot")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client


async def test_feishu_local_reply_endpoint(client):
    fake_bot = MagicMock()
    fake_bot.build_local_reply_async = AsyncMock(return_value="这是 Feishu 会话逻辑的最终回答")

    with patch("src.routers.feishu_local.make_local_feishu_bot", return_value=fake_bot) as mock_factory:
        response = await client.post(
            "/api/v1/feishu/reply",
            json={
                "session_id": "local_geek_master",
                "query": "给我推荐一篇UAV论文",
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "local_geek_master"
    assert data["query"] == "给我推荐一篇UAV论文"
    assert data["answer"] == "这是 Feishu 会话逻辑的最终回答"
    assert "contexts" not in data
    assert "intent" not in data
    fake_bot.build_local_reply_async.assert_called_once_with(
        session_id="local_geek_master",
        query="给我推荐一篇UAV论文",
    )
    _, kwargs = mock_factory.call_args
    assert kwargs["local_runtime"].opensearch_client is app.state.opensearch_client
    assert kwargs["local_runtime"].embeddings_service is app.state.embeddings_service
    assert kwargs["local_runtime"].ollama_client is app.state.ollama_client
    assert kwargs["local_runtime"].cache_client is app.state.cache_client


async def test_feishu_local_reply_eval_debug_endpoint(client):
    fake_bot = MagicMock()
    fake_bot.build_local_reply_debug_async = AsyncMock(
        return_value=SimpleNamespace(
            answer="debug answer",
            contexts=["retrieved context"],
            sources=["https://arxiv.org/abs/2603.16279"],
            intent="general_rag",
            rewritten_query="rewritten",
            route="/api/v1/ask",
        )
    )

    with patch("src.routers.feishu_local.make_local_feishu_bot", return_value=fake_bot):
        response = await client.post(
            "/api/v1/feishu/reply",
            json={
                "session_id": "eval-thread",
                "query": "什么是RAG？",
                "eval_debug": True,
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "debug answer"
    assert data["contexts"] == ["retrieved context"]
    assert data["sources"] == ["https://arxiv.org/abs/2603.16279"]
    assert data["intent"] == "general_rag"
    assert data["rewritten_query"] == "rewritten"
    assert data["route"] == "/api/v1/ask"
    fake_bot.build_local_reply_debug_async.assert_awaited_once_with(session_id="eval-thread", query="什么是RAG？")


async def test_feishu_local_reply_validation_errors(client):
    response = await client.post("/api/v1/feishu/reply", json={"query": "test"})
    assert response.status_code == 422

    response = await client.post("/api/v1/feishu/reply", json={"session_id": "thread-1", "query": ""})
    assert response.status_code == 422
