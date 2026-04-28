import pytest
from src.config import Settings
from src.exceptions import OllamaConnectionError
from src.services.ollama.client import OllamaClient


class _FakeResponse:
    status_code = 200

    def json(self):
        return {
            "response": "ok",
            "prompt_eval_count": 2,
            "eval_count": 1,
            "total_duration": 1_000_000,
        }


class _FakeAsyncClient:
    captured_init_kwargs = None
    captured_payload = None

    def __init__(self, *args, **kwargs):
        self.__class__.captured_init_kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def post(self, url, json):
        self.__class__.captured_payload = json
        return _FakeResponse()


@pytest.mark.asyncio
async def test_ollama_generate_disables_thinking_by_default(monkeypatch):
    monkeypatch.setattr("src.services.ollama.client.httpx.AsyncClient", _FakeAsyncClient)
    settings = Settings(debug=True, ollama_host="http://127.0.0.1:11435", ollama_model="qwen2.5:7b")
    client = OllamaClient(settings)

    response = await client.generate(model="qwen2.5:7b", prompt="hello", provider="ollama")

    assert response["response"] == "ok"
    assert _FakeAsyncClient.captured_init_kwargs["trust_env"] is False
    assert _FakeAsyncClient.captured_payload["model"] == "qwen2.5:7b"
    assert _FakeAsyncClient.captured_payload["think"] is False
    assert _FakeAsyncClient.captured_payload["options"]["num_gpu"] == -1


def test_langchain_model_defaults_to_gpu_and_no_reasoning():
    settings = Settings(debug=True, ollama_host="http://127.0.0.1:11435", ollama_model="qwen2.5:7b")
    client = OllamaClient(settings)

    model = client.get_langchain_model("qwen2.5:7b")

    assert model.num_gpu == -1
    assert model.reasoning is False


@pytest.mark.asyncio
async def test_explicit_qwen_provider_does_not_fall_back_to_ollama_without_api_key():
    settings = Settings(
        debug=True,
        ollama_host="http://127.0.0.1:11435",
        ollama_model="qwen2.5:7b",
        qwen_api_key="",
    )
    client = OllamaClient(settings)

    with pytest.raises(OllamaConnectionError, match="QWEN_API_KEY is not configured for qwen_api provider"):
        await client.generate(model="qwen3.5-plus", prompt="hello", provider="qwen_api")


@pytest.mark.asyncio
async def test_default_provider_does_not_fall_back_to_ollama_without_api_key():
    settings = Settings(
        debug=True,
        ollama_host="http://127.0.0.1:11435",
        ollama_model="qwen2.5:7b",
        qwen_api_key="",
    )
    client = OllamaClient(settings)

    with pytest.raises(OllamaConnectionError, match="QWEN_API_KEY is not configured for qwen_api provider"):
        await client.generate(model="qwen3.5-plus", prompt="hello")
