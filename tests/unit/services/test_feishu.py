from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from src.config import FeishuSettings
from src.services.feishu.bot import FeishuBot, FeishuMessageContext, FeishuPaperReference
from src.services.feishu.factory import make_feishu_bot


def _mock_lark_builder():
    builder = MagicMock()
    builder.app_id.return_value = builder
    builder.app_secret.return_value = builder
    builder.log_level.return_value = builder
    builder.build.return_value = MagicMock()
    return builder


def _build_test_bot(**kwargs) -> FeishuBot:
    settings = SimpleNamespace(
        resolve_llm_provider=lambda provider=None: "qwen_api" if provider == "qwen_api" else "ollama",
        resolve_llm_model=lambda provider=None, model="": model or ("qwen3.5-plus" if provider == "qwen_api" else "qwen2.5:7b"),
        feishu=FeishuSettings(
            enabled=True,
            app_id="cli_test",
            app_secret="secret_test",
            api_base_url="http://localhost:8001",
            llm_provider="qwen_api",
            model="qwen3.5-plus",
        )
    )
    with patch("src.services.feishu.bot.lark.Client.builder", return_value=_mock_lark_builder()):
        return FeishuBot(settings=settings, **kwargs)


class _FakeDatabase:
    @contextmanager
    def get_session(self):
        yield object()


class _FakeConversationRepository:
    def __init__(self):
        self.conversation_key = ""
        self.conversation = SimpleNamespace(
            id=uuid4(),
            recent_papers=[],
            active_papers=[],
            last_intent="",
            last_query="",
        )
        self.messages = []
        self.summary = ""
        self.source_message_count = 0
        self.cleared = False
        self.cleared_all = False

    def get_or_create_session(self, **kwargs):
        self.conversation_key = kwargs["conversation_key"]
        return self.conversation

    def get_session_by_key(self, conversation_key):
        if conversation_key != self.conversation_key:
            return None
        return self.conversation

    def update_session_state(self, conversation, **kwargs):
        conversation.recent_papers = kwargs["recent_papers"]
        conversation.active_papers = kwargs["active_papers"]
        conversation.last_intent = kwargs["last_intent"]
        conversation.last_query = kwargs["last_query"]
        return conversation

    def append_message(self, **kwargs):
        self.messages.append(
            SimpleNamespace(
                role=kwargs["role"],
                content=kwargs["content"],
                intent=kwargs.get("intent", ""),
                message_id=kwargs.get("message_id", ""),
                created_at="2026-04-14T00:00:00+00:00",
            )
        )
        return self.messages[-1]

    def get_recent_messages(self, session_id, limit=12):
        return self.messages[-limit:]

    def count_messages(self, session_id):
        return len(self.messages)

    def get_summary(self, session_id):
        return self.summary

    def upsert_summary(self, **kwargs):
        self.summary = kwargs["summary"]
        self.source_message_count = kwargs["source_message_count"]

    def clear_session_memory(self, conversation_key):
        self.cleared = True
        self.messages.clear()
        self.summary = ""
        self.conversation.recent_papers = []
        self.conversation.active_papers = []
        self.conversation.last_intent = ""
        self.conversation.last_query = ""

    def clear_all_memory(self):
        self.cleared_all = True
        self.clear_session_memory(self.conversation_key)


class _FakeRedis:
    def __init__(self):
        self.store = {}
        self.expirations = {}

    def set(self, key, value, ex=None, nx=False):
        if nx and key in self.store:
            return False
        self.store[key] = value
        self.expirations[key] = ex
        return True

    def get(self, key):
        return self.store.get(key)

    def delete(self, *keys):
        for key in keys:
            self.store.pop(key, None)
            self.expirations.pop(key, None)

    def scan_iter(self, match=None):
        if match == "feishu:context:*":
            for key in list(self.store):
                if str(key).startswith("feishu:context:"):
                    yield key
            return
        yield from list(self.store)


class TestFeishuSettings:
    """Test Feishu settings."""

    def test_default_settings(self):
        with patch.dict("os.environ", {}, clear=True):
            settings = FeishuSettings(_env_file=None)

        assert settings.enabled is False
        assert settings.app_id == ""
        assert settings.app_secret == ""
        assert settings.api_base_url == "http://localhost:8001"
        assert settings.llm_provider == "qwen_api"
        assert settings.use_hybrid is True
        assert settings.context_ttl_seconds == 3600
        assert settings.auto_ingest_enabled is True

    def test_custom_settings(self):
        settings = FeishuSettings(
            enabled=True,
            app_id="cli_xxx",
            app_secret="secret_xxx",
            api_base_url="http://api:8000",
            llm_provider="qwen_api",
            model="qwen3.5-plus",
        )

        assert settings.enabled is True
        assert settings.app_id == "cli_xxx"
        assert settings.api_base_url == "http://api:8000"
        assert settings.llm_provider == "qwen_api"
        assert settings.model == "qwen3.5-plus"


class TestFeishuFactory:
    """Test Feishu bot factory."""

    @patch("src.services.feishu.factory.get_settings")
    def test_factory_disabled(self, mock_settings):
        mock_settings.return_value.feishu.enabled = False

        bot = make_feishu_bot()

        assert bot is None

    @patch("src.services.feishu.factory.get_settings")
    def test_factory_without_credentials(self, mock_settings):
        mock_settings.return_value.feishu.enabled = True
        mock_settings.return_value.feishu.app_id = ""
        mock_settings.return_value.feishu.app_secret = ""

        bot = make_feishu_bot()

        assert bot is None

    @patch("src.services.feishu.factory.make_database")
    @patch("src.services.feishu.factory.make_redis_client")
    @patch("src.services.feishu.factory.get_settings")
    @patch("src.services.feishu.factory.FeishuBot")
    def test_factory_success(self, mock_bot_cls, mock_settings, mock_make_redis, mock_make_database):
        mock_settings.return_value.feishu.enabled = True
        mock_settings.return_value.feishu.app_id = "cli_test"
        mock_settings.return_value.feishu.app_secret = "secret_test"
        mock_make_redis.return_value = MagicMock()
        mock_make_database.return_value = MagicMock()

        bot = make_feishu_bot()

        assert bot is mock_bot_cls.return_value
        mock_bot_cls.assert_called_once_with(
            settings=mock_settings.return_value,
            redis_client=mock_make_redis.return_value,
            database=mock_make_database.return_value,
        )


class TestFeishuBotHelpers:
    """Test Feishu event normalization helpers."""

    def _make_context(self, query: str = "What is RAG?") -> FeishuMessageContext:
        return FeishuMessageContext(
            message_id="om_ctx_1",
            chat_id="oc_chat_1",
            chat_type="p2p",
            sender_open_id="ou_user_1",
            receive_id="ou_user_1",
            receive_id_type="open_id",
            query=query,
        )

    def _make_papers(self) -> list[FeishuPaperReference]:
        return [
            FeishuPaperReference(
                arxiv_id="2604.09544v1",
                title="Large Language Models Generate Harmful Content Using a Distinct, Unified Mechanism",
                abstract="Paper one abstract",
            ),
            FeishuPaperReference(
                arxiv_id="2604.09532v1",
                title="Seeing is Believing: Robust Vision-Guided Cross-Modal Prompt Learning under Label Noise",
                abstract="Paper two abstract",
            ),
        ]

    def test_extract_p2p_text_message(self):
        bot = _build_test_bot()

        context = bot._extract_message_context(
            {
                "event": {
                    "sender": {"sender_id": {"open_id": "ou_user_1"}},
                    "message": {
                        "message_id": "om_test_1",
                        "chat_id": "oc_chat_1",
                        "chat_type": "p2p",
                        "message_type": "text",
                        "content": "{\"text\":\"What is RAG?\"}",
                    },
                }
            }
        )

        assert context is not None
        assert context.receive_id_type == "open_id"
        assert context.receive_id == "ou_user_1"
        assert context.query == "What is RAG?"

    def test_extract_group_mention_message(self):
        bot = _build_test_bot()

        context = bot._extract_message_context(
            {
                "event": {
                    "sender": {"sender_id": {"open_id": "ou_user_2"}},
                    "message": {
                        "message_id": "om_test_2",
                        "chat_id": "oc_group_1",
                        "chat_type": "group",
                        "message_type": "text",
                        "content": "{\"text\":\"@_user_1 summarize transformers\"}",
                        "mentions": [{"name": "RAG Bot"}],
                    },
                }
            }
        )

        assert context is not None
        assert context.receive_id_type == "chat_id"
        assert context.receive_id == "oc_group_1"
        assert context.query == "summarize transformers"

    def test_ignore_group_message_without_mentions(self):
        bot = _build_test_bot()

        context = bot._extract_message_context(
            {
                "event": {
                    "sender": {"sender_id": {"open_id": "ou_user_3"}},
                    "message": {
                        "message_id": "om_test_3",
                        "chat_id": "oc_group_2",
                        "chat_type": "group",
                        "message_type": "text",
                        "content": "{\"text\":\"hello all\"}",
                        "mentions": [],
                    },
                }
            }
        )

        assert context is None

    def test_ignore_non_text_message(self):
        bot = _build_test_bot()

        context = bot._extract_message_context(
            {
                "event": {
                    "sender": {"sender_id": {"open_id": "ou_user_4"}},
                    "message": {
                        "message_id": "om_test_4",
                        "chat_id": "oc_group_3",
                        "chat_type": "p2p",
                        "message_type": "image",
                        "content": "{}",
                    },
                }
            }
        )

        assert context is None

    def test_normalize_sources(self):
        bot = _build_test_bot()

        sources = bot._normalize_sources(
            [
                "https://arxiv.org/pdf/1706.03762.pdf",
                "https://example.com/custom-source",
            ]
        )

        assert sources == [
            "https://arxiv.org/abs/1706.03762",
            "https://example.com/custom-source",
        ]

    def test_store_and_load_recent_papers_from_memory(self):
        bot = _build_test_bot()
        context = self._make_context()
        papers = self._make_papers()

        bot._store_recent_papers(context, papers)
        loaded = bot._load_recent_papers(context)

        assert loaded == papers

    def test_resolve_referenced_papers_second_paper(self):
        bot = _build_test_bot()
        context = self._make_context()
        papers = self._make_papers()
        bot._store_recent_papers(context, papers)

        resolved = bot._resolve_referenced_papers(context, "详细解释第二篇论文")

        assert resolved == [papers[1]]

    def test_build_reply_uses_stored_context_for_follow_up(self):
        bot = _build_test_bot()
        context = self._make_context()
        papers = self._make_papers()
        bot._store_recent_papers(context, papers)

        with patch.object(bot, "_ask_rag_api", side_effect=lambda **kwargs: kwargs["query"]):
            reply = bot._build_reply(self._make_context("这两篇论文主要讲什么？"))

        assert papers[0].title in reply
        assert papers[1].title in reply
        assert "用户原始问题：这两篇论文主要讲什么？" in reply

    def test_build_reply_without_context_returns_memory_hint(self):
        bot = _build_test_bot()

        reply = bot._build_reply(self._make_context("这两篇论文主要讲什么？"))

        assert "还没有记住" in reply

    def test_build_reply_search_query_stores_recommended_papers(self):
        bot = _build_test_bot()
        context = self._make_context("给我找两篇AI方向的论文")
        papers = self._make_papers()

        with patch.object(bot, "_find_recommended_papers", return_value=papers):
            reply = bot._build_reply(context)

        loaded = bot._load_recent_papers(context)

        assert "当前已索引的论文库里找到了 2 篇相关论文" in reply
        assert loaded == papers

    def test_build_reply_persists_messages_summary_and_state(self):
        repo = _FakeConversationRepository()
        bot = _build_test_bot(
            database=_FakeDatabase(),
            conversation_repository_factory=lambda session: repo,
        )
        context = self._make_context("What is RAG?")

        with patch.object(bot, "_ask_rag_api", return_value="RAG answer"):
            reply = bot._build_reply(context)

        assert reply == "RAG answer"
        assert [message.role for message in repo.messages] == ["user", "assistant"]
        assert repo.conversation.last_query == "What is RAG?"
        assert repo.source_message_count == 2
        assert "What is RAG?" in repo.summary
        assert "RAG answer" in repo.summary

        bot._memory_context.clear()
        loaded = bot._load_conversation_state(context)

        assert loaded.summary == repo.summary
        assert loaded.persisted_message_count == 2
        assert [message.role for message in loaded.recent_messages] == ["user", "assistant"]

    def test_redis_hot_context_stores_recent_messages(self):
        redis_client = _FakeRedis()
        repo = _FakeConversationRepository()
        bot = _build_test_bot(
            redis_client=redis_client,
            database=_FakeDatabase(),
            conversation_repository_factory=lambda session: repo,
        )
        context = self._make_context("What is RAG?")

        with patch.object(bot, "_ask_rag_api", return_value="RAG answer"):
            bot._build_reply(context)

        key = bot._context_key(context)
        assert key in redis_client.store
        assert redis_client.expirations[key] == bot.feishu_settings.context_ttl_seconds

        repo.summary = "database should not be read while Redis is warm"
        loaded = bot._load_conversation_state(context)

        assert loaded.summary != repo.summary
        assert loaded.recent_messages[-1].content == "RAG answer"

    def test_extract_requested_paper_count_chinese(self):
        bot = _build_test_bot()

        assert bot._extract_requested_paper_count("给我找两篇AI方向的论文") == 2
        assert bot._extract_requested_paper_count("推荐3篇机器学习论文") == 3

    def test_clear_recent_papers_from_memory(self):
        bot = _build_test_bot()
        context = self._make_context()
        papers = self._make_papers()
        bot._store_recent_papers(context, papers)

        bot._clear_recent_papers(context)

        assert bot._load_recent_papers(context) == []

    def test_clear_all_conversation_memory_clears_cache_and_database(self):
        redis_client = _FakeRedis()
        repo = _FakeConversationRepository()
        bot = _build_test_bot(
            redis_client=redis_client,
            database=_FakeDatabase(),
            conversation_repository_factory=lambda session: repo,
        )
        context = self._make_context()
        papers = self._make_papers()
        bot._store_recent_papers(context, papers)
        redis_client.set("feishu:processed:message-1", "1")
        bot._memory_context["feishu:context:p2p:local-only"] = (9999999999, "cached")

        bot._clear_all_conversation_memory()

        assert repo.cleared_all is True
        assert "feishu:processed:message-1" in redis_client.store
        assert not any(key.startswith("feishu:context:") for key in redis_client.store)
        assert bot._memory_context == {}
        assert bot._load_recent_papers(context) == []

    def test_build_reply_reset_command_clears_context(self):
        bot = _build_test_bot()
        context = self._make_context()
        papers = self._make_papers()
        bot._store_recent_papers(context, papers)

        reply = bot._build_reply(self._make_context("新对话"))

        assert "清空当前会话记忆" in reply
        assert bot._load_recent_papers(context) == []

    def test_build_reply_new_message_alias_clears_context(self):
        bot = _build_test_bot()
        context = self._make_context()
        papers = self._make_papers()
        bot._store_recent_papers(context, papers)

        reply = bot._build_reply(self._make_context("新消息"))

        assert "清空当前会话记忆" in reply
        assert bot._load_recent_papers(context) == []

    @patch("src.services.feishu.bot.requests.post")
    def test_ask_rag_api_uses_qwen_provider_for_feishu(self, mock_post):
        bot = _build_test_bot()
        mock_response = MagicMock()
        mock_response.json.return_value = {"answer": "好的", "sources": []}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        reply = bot._ask_rag_api("测试一下")

        assert reply == "好的"
        _, kwargs = mock_post.call_args
        assert kwargs["json"]["provider"] == "qwen_api"
        assert kwargs["json"]["model"] == "qwen3.5-plus"

    @patch("src.services.feishu.bot.requests.post")
    def test_ask_rag_api_compacts_repeated_single_paper_citations(self, mock_post):
        bot = _build_test_bot()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "answer": (
                "1. 背景：无人机拦截很难 [arXiv:2603.16279v1]。\n"
                "2. 方法：采用竞争性 MARL [arXiv:2603.16279v1]。\n"
                "3. 结果：捕获率更高 [arXiv:2603.16279v1]。"
            ),
            "sources": ["https://arxiv.org/pdf/2603.16279.pdf"],
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        reply = bot._ask_rag_api("详细分析这篇论文")

        assert reply.count("[arXiv:2603.16279v1]") == 1
        assert "1. 背景：无人机拦截很难。" in reply
        assert "参考来源：\n1. https://arxiv.org/abs/2603.16279" in reply

    @patch("src.services.feishu.bot.requests.post")
    def test_ask_rag_api_reference_query_uses_direct_papers_and_standard_endpoint(self, mock_post):
        bot = _build_test_bot()
        mock_response = MagicMock()
        mock_response.json.return_value = {"answer": "好的", "sources": []}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        reply = bot._ask_rag_api(
            query="请基于以下论文回答用户问题：Paper A (arXiv:1); Paper B (arXiv:2)。用户问题：这两篇论文讲什么？",
            arxiv_ids=["1", "2"],
            force_standard=True,
        )

        assert reply == "好的"
        args, kwargs = mock_post.call_args
        assert kwargs["json"]["use_hybrid"] is False
        assert kwargs["json"]["arxiv_ids"] == ["1", "2"]
        assert args[0].endswith("/api/v1/ask")

    def test_build_reply_reference_query_uses_direct_paper_ids(self):
        bot = _build_test_bot()
        context = self._make_context()
        papers = self._make_papers()
        bot._store_recent_papers(context, papers)

        with patch.object(bot, "_ask_rag_api", return_value="好的") as mock_ask:
            reply = bot._build_reply(self._make_context("这两篇论文都是讲什么的 详细分析 越详细越好"))

        assert reply == "好的"
        assert mock_ask.call_args.kwargs["arxiv_ids"] == [papers[0].arxiv_id, papers[1].arxiv_id]
        assert mock_ask.call_args.kwargs["force_standard"] is True

    def test_build_reply_implicit_follow_up_uses_active_papers(self):
        bot = _build_test_bot()
        context = self._make_context()
        papers = self._make_papers()
        bot._store_recent_papers(context, papers)

        with patch.object(bot, "_ask_rag_api", return_value="好的") as mock_ask:
            reply = bot._build_reply(self._make_context("太短了，再详细分析一下"))

        assert reply == "好的"
        assert mock_ask.call_args.kwargs["arxiv_ids"] == [papers[0].arxiv_id, papers[1].arxiv_id]
        assert mock_ask.call_args.kwargs["force_standard"] is True
        assert mock_ask.call_args.kwargs["direct_chunks_per_paper"] == 22
        assert "研究背景与问题" in mock_ask.call_args.kwargs["query"]

    def test_build_reply_compare_follow_up_uses_compare_prompt(self):
        bot = _build_test_bot()
        context = self._make_context()
        papers = self._make_papers()
        bot._store_recent_papers(context, papers)

        with patch.object(bot, "_ask_rag_api", return_value="好的") as mock_ask:
            reply = bot._build_reply(self._make_context("详细对比一下它们的区别"))

        assert reply == "好的"
        assert mock_ask.call_args.kwargs["arxiv_ids"] == [papers[0].arxiv_id, papers[1].arxiv_id]
        assert mock_ask.call_args.kwargs["direct_chunks_per_paper"] == 18
        assert "实验环境" in mock_ask.call_args.kwargs["query"]

    def test_find_recommended_papers_auto_ingests_when_local_results_are_insufficient(self):
        bot = _build_test_bot()
        context = self._make_context("你给我找三篇无人机VLN的论文")
        local_papers = [self._make_papers()[0]]
        refreshed_papers = self._make_papers() + [
            FeishuPaperReference(
                arxiv_id="2604.00003v1",
                title="Embodied UAV Vision-Language Navigation",
                abstract="Paper three abstract",
            )
        ]
        bot._paper_ingestion_service = SimpleNamespace(
            ingest_missing_papers=AsyncMock(
                return_value=SimpleNamespace(
                    search_text="unmanned aerial vehicle UAV drone vision language navigation VLN",
                    papers_fetched=3,
                    papers_stored=3,
                    papers_indexed=3,
                    chunks_indexed=12,
                )
            )
        )

        with (
            patch.object(bot, "_search_papers_via_api", side_effect=[local_papers, refreshed_papers]),
            patch.object(bot, "_send_progress_message") as mock_progress,
        ):
            papers = bot._find_recommended_papers(context, context.query, 3)

        assert [paper.arxiv_id for paper in papers] == [paper.arxiv_id for paper in refreshed_papers[:3]]
        mock_progress.assert_called_once()
        bot._paper_ingestion_service.ingest_missing_papers.assert_awaited_once()

    @patch("src.services.feishu.bot.requests.post")
    def test_fetch_latest_indexed_papers_uses_api_instead_of_direct_opensearch(self, mock_post):
        bot = _build_test_bot()
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "hits": [
                {
                    "arxiv_id": "2604.00001v1",
                    "title": "Paper A",
                    "abstract": "A",
                    "published_date": "2026-04-10T00:00:00",
                },
                {
                    "arxiv_id": "2604.00002v1",
                    "title": "Paper B",
                    "abstract": "B",
                    "published_date": "2026-04-09T00:00:00",
                },
            ]
        }
        mock_post.return_value = mock_response

        papers = bot._fetch_latest_indexed_papers(2)

        assert [paper.arxiv_id for paper in papers] == ["2604.00001v1", "2604.00002v1"]
        _, kwargs = mock_post.call_args
        assert kwargs["json"]["latest_papers"] is True
