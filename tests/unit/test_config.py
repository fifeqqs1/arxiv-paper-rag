from unittest.mock import patch

from src.config import Settings


def _settings_without_env() -> Settings:
    with patch.dict("os.environ", {}, clear=True):
        return Settings(_env_file=None)


def test_settings_initialization():
    """Test settings can be initialized."""
    settings = _settings_without_env()

    assert settings.app_version == "0.1.0"
    assert settings.debug is True
    assert settings.environment == "development"
    assert settings.service_name == "rag-api"


def test_settings_postgres_defaults():
    """Test PostgreSQL default configuration."""
    settings = _settings_without_env()

    assert "postgresql://" in settings.postgres_database_url
    assert settings.postgres_echo_sql is False
    assert settings.postgres_pool_size == 20
    assert settings.postgres_max_overflow == 0


def test_settings_opensearch_defaults():
    """Test OpenSearch default configuration."""
    settings = _settings_without_env()

    assert settings.opensearch.host in [
        "http://localhost:9200",
        "http://opensearch:9200",
    ]
    assert settings.opensearch.index_name == "arxiv-papers"


def test_settings_ollama_defaults():
    """Test Ollama default configuration."""
    settings = _settings_without_env()

    assert settings.ollama_host in [
        "http://localhost:11434",
        "http://ollama:11434",
        "http://127.0.0.1:11434",
        "http://127.0.0.1:11435",
    ]
    assert settings.ollama_model == "qwen2.5:7b"
    assert settings.ollama_num_gpu == -1


def test_settings_default_llm_provider_is_qwen_api():
    """Test the project defaults to the hosted Qwen API."""
    settings = _settings_without_env()

    assert settings.resolve_llm_provider() == "qwen_api"
    assert settings.resolve_llm_model() == "qwen3.5-plus"
