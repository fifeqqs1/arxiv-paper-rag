import asyncio
import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Optional
from uuid import uuid4

import lark_oapi as lark
import redis
import requests
from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody, CreateMessageResponse
from src.config import Settings
from src.repositories.feishu_conversation import FeishuConversationRepository, StoredConversationMessage
from src.routers.agentic_ask import ask_agentic
from src.routers.ask import ask_question
from src.schemas.api.ask import AskRequest
from src.services.citations import compact_repeated_single_paper_citations
from src.services.feishu.paper_ingestion import FeishuPaperIngestionService

logger = logging.getLogger(__name__)

MENTION_PLACEHOLDER_RE = re.compile(r"@_user_\d+\s*")
PAPER_SEARCH_HINTS = (
    "找",
    "推荐",
    "给我几篇",
    "给我两篇",
    "给我三篇",
    "相关论文",
    "论文推荐",
    "papers",
    "paper",
    "find",
    "recommend",
    "show me",
    "related papers",
)
REFERENCE_HINTS = (
    "这篇",
    "这两篇",
    "这几篇",
    "上一篇",
    "上一条",
    "刚才",
    "前面",
    "上面",
    "第一篇",
    "第二篇",
    "第三篇",
    "those papers",
    "these papers",
    "the first paper",
    "the second paper",
)
FOLLOWUP_HINTS = (
    "太短了",
    "再详细",
    "详细一点",
    "更详细",
    "越详细越好",
    "展开讲",
    "展开说",
    "继续",
    "继续讲",
    "再讲讲",
    "深入分析",
    "深入讲解",
    "细讲",
    "详细分析",
    "详细讲解",
)
DETAIL_HINTS = (
    "详细",
    "展开",
    "深入",
    "讲解",
    "分析",
    "解释",
    "总结",
    "概述",
    "方法",
    "实验",
    "结果",
    "贡献",
    "局限",
    "创新",
    "应用场景",
)
METHOD_QUESTION_HINTS = (
    "用了什么方法",
    "用什么方法",
    "什么方法",
    "方法是什么",
    "核心方法",
    "方法细节",
    "技术路线",
    "算法设计",
    "模型架构",
    "框架设计",
    "怎么做",
    "如何实现",
    "method",
    "methods",
    "approach",
    "algorithm",
    "architecture",
    "framework",
)
COMPLEX_REASONING_HINTS = (
    "为什么",
    "原因",
    "推导",
    "proof",
    "证明",
    "逐步",
    "一步一步",
    "拆解",
    "分析一下原因",
    "trade-off",
    "权衡",
    "优缺点",
    "适不适合",
    "是否适合",
    "怎么选",
    "路线",
)
MEDIUM_COMPLEXITY_HINTS = (
    "总结",
    "概述",
    "讲讲",
    "解释",
    "方法",
    "实验",
    "结果",
    "贡献",
    "局限",
)
COMPARE_HINTS = (
    "对比",
    "比较",
    "区别",
    "差异",
    "异同",
    "不同",
    "优缺点",
    "哪个更",
    "谁更",
    "compare",
)
BASELINE_QUESTION_HINTS = (
    "和哪些方法进行对比",
    "与哪些方法进行对比",
    "跟哪些方法进行对比",
    "和哪些方法对比",
    "与哪些方法对比",
    "跟哪些方法对比",
    "和哪些方法比较了",
    "与哪些方法比较了",
    "跟哪些方法比较了",
    "和什么方法比较了",
    "与什么方法比较了",
    "跟什么方法比较了",
    "对比了哪些方法",
    "比较了哪些方法",
    "哪些基线",
    "哪些baseline",
    "baseline",
    "baselines",
    "compared with",
    "compared against",
)
EXPERIMENT_QUESTION_HINTS = (
    "实验部分",
    "实验方法",
    "实验设置",
    "实验结果",
    "实验效果",
    "验证方法",
    "消融实验",
    "对比实验",
    "experiment",
    "experiments",
    "experimental",
)
CONTEXTUAL_REFERENCE_HINTS = (
    "它",
    "它们",
    "该论文",
    "这些论文",
    "这些工作",
    "上述论文",
    "上述工作",
    "这个工作",
    "这个方法",
)
ORDINAL_HINTS = {
    "第一篇": 0,
    "第1篇": 0,
    "first paper": 0,
    "第二篇": 1,
    "第2篇": 1,
    "second paper": 1,
    "第三篇": 2,
    "第3篇": 2,
    "third paper": 2,
}
CHINESE_COUNT_HINTS = {
    "一": 1,
    "一篇": 1,
    "1篇": 1,
    "两": 2,
    "两篇": 2,
    "二": 2,
    "二篇": 2,
    "2篇": 2,
    "三": 3,
    "三篇": 3,
    "3篇": 3,
    "四": 4,
    "四篇": 4,
    "4篇": 4,
    "五": 5,
    "五篇": 5,
    "5篇": 5,
}
DEFAULT_PAPER_RECOMMENDATION_COUNT = 3
MAX_PAPER_RECOMMENDATION_COUNT = 5
SEARCH_POOL_MULTIPLIER = 8
LATEST_PAPER_POOL_MIN = 20
RECENT_MESSAGE_LIMIT = 12
SUMMARY_MAX_CHARS = 1600
CONTEXTUAL_QUERY_MAX_CHARS = 950
DEFAULT_CONVERSATION_MAX_MESSAGES = 100
DEFAULT_WORKER_MAX_WORKERS = 2
DEFAULT_WORKER_QUEUE_SIZE = 4
DEFAULT_HTTP_MAX_RETRIES = 1
DEFAULT_HTTP_RETRY_BACKOFF_SECONDS = 0.5
DEFAULT_MESSAGE_CHUNK_CHARS = 3500
RETRY_HTTP_STATUS_CODES = {429, 500, 502, 503, 504}
RESET_MEMORY_COMMANDS = ("新对话", "新消息")
INTENT_RESET = "reset"
INTENT_PAPER_SEARCH = "paper_search"
INTENT_PAPER_SUMMARY = "paper_summary"
INTENT_PAPER_METHOD = "paper_method"
INTENT_PAPER_EXPERIMENT = "paper_experiment"
INTENT_PAPER_DEEP_DIVE = "paper_deep_dive"
INTENT_PAPER_COMPARE = "paper_compare"
INTENT_GENERAL_RAG = "general_rag"
QUERY_COMPLEXITY_SIMPLE = "simple"
QUERY_COMPLEXITY_MEDIUM = "medium"
QUERY_COMPLEXITY_COMPLEX = "complex"
PAPER_RELATED_INTENTS = {
    INTENT_PAPER_SEARCH,
    INTENT_PAPER_SUMMARY,
    INTENT_PAPER_METHOD,
    INTENT_PAPER_EXPERIMENT,
    INTENT_PAPER_DEEP_DIVE,
    INTENT_PAPER_COMPARE,
}


@dataclass(frozen=True)
class FeishuMessageContext:
    """Normalized Feishu message payload used by the bot worker."""

    message_id: str
    chat_id: str
    chat_type: str
    sender_open_id: str
    receive_id: str
    receive_id_type: str
    query: str


@dataclass(frozen=True)
class FeishuPaperReference:
    """Indexed paper metadata stored for follow-up questions."""

    arxiv_id: str
    title: str
    abstract: str = ""
    published_date: str = ""


@dataclass(frozen=True)
class FeishuConversationTurn:
    """A compact message stored in hot context for fast follow-up handling."""

    role: str
    content: str
    intent: str = ""
    message_id: str = ""
    created_at: str = ""


@dataclass(frozen=True)
class FeishuConversationState:
    """Lightweight per-chat memory for paper-centric follow-up handling."""

    recent_papers: list[FeishuPaperReference]
    active_papers: list[FeishuPaperReference]
    last_intent: str = ""
    last_query: str = ""
    summary: str = ""
    recent_messages: list[FeishuConversationTurn] = field(default_factory=list)
    persisted_message_count: int = 0


@dataclass(frozen=True)
class FeishuRouteDecision:
    endpoint_path: str
    complexity: str


@dataclass(frozen=True)
class FeishuIntentDecision:
    """Routing result for a single Feishu user message."""

    intent: str
    papers: list[FeishuPaperReference]


@dataclass(frozen=True)
class FeishuRAGDebugResult:
    """RAG response plus evaluation metadata captured from the Feishu route."""

    reply: str
    answer: str
    contexts: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    rewritten_query: str = ""
    route: str = ""


@dataclass(frozen=True)
class FeishuLocalDebugReply:
    """Local Feishu reply payload used by offline evaluators."""

    answer: str
    contexts: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    intent: str = ""
    rewritten_query: str = ""
    route: str = ""


@dataclass(frozen=True)
class FeishuLocalRuntime:
    """In-process services used by the local Feishu endpoint."""

    opensearch_client: Any
    embeddings_service: Any
    ollama_client: Any
    langfuse_tracer: Any = None
    cache_client: Any = None


class FeishuBot:
    """Minimal Feishu bot powered by the existing RAG API."""

    def __init__(
        self,
        settings: Settings,
        redis_client: Optional[redis.Redis] = None,
        database: Optional[Any] = None,
        conversation_repository_factory: Callable[[Any], FeishuConversationRepository] = FeishuConversationRepository,
        client: Optional[Any] = None,
        build_client: bool = True,
        local_runtime: Optional[FeishuLocalRuntime] = None,
    ):
        self.settings = settings
        self.feishu_settings = settings.feishu
        self.redis = redis_client
        self.database = database
        self._local_runtime = local_runtime
        self._conversation_repository_factory = conversation_repository_factory
        if client is not None:
            self.client = client
        elif build_client and self.feishu_settings.app_id and self.feishu_settings.app_secret:
            self.client = (
                lark.Client.builder()
                .app_id(self.feishu_settings.app_id)
                .app_secret(self.feishu_settings.app_secret)
                .log_level(lark.LogLevel.INFO)
                .build()
            )
        else:
            self.client = None
        self._send_lock = threading.Lock()
        self._context_lock = threading.Lock()
        self._memory_context: dict[str, tuple[float, str]] = {}
        self._paper_ingestion_service: Optional[FeishuPaperIngestionService] = None
        self._cleanup_stop_event = threading.Event()
        self._cleanup_thread: Optional[threading.Thread] = None
        self._conversation_max_messages = max(
            0,
            int(getattr(self.feishu_settings, "conversation_max_messages", DEFAULT_CONVERSATION_MAX_MESSAGES)),
        )
        self._worker_max_workers = max(
            1,
            int(getattr(self.feishu_settings, "worker_max_workers", DEFAULT_WORKER_MAX_WORKERS)),
        )
        self._worker_queue_size = max(
            0,
            int(getattr(self.feishu_settings, "worker_queue_size", DEFAULT_WORKER_QUEUE_SIZE)),
        )
        self._worker_executor = ThreadPoolExecutor(
            max_workers=self._worker_max_workers,
            thread_name_prefix="feishu-message",
        )
        self._worker_slots = threading.BoundedSemaphore(self._worker_max_workers + self._worker_queue_size)
        self._dedupe_lock = threading.Lock()
        self._local_dedupe: dict[str, tuple[float, str]] = {}
        self._conversation_locks_guard = threading.Lock()
        self._conversation_locks: dict[str, threading.Lock] = {}
        self._async_conversation_locks_guard = threading.Lock()
        self._async_conversation_locks: dict[str, asyncio.Lock] = {}
        self._http_session = requests.Session()
        self._http_max_retries = max(
            0,
            int(getattr(self.feishu_settings, "http_max_retries", DEFAULT_HTTP_MAX_RETRIES)),
        )
        self._http_retry_backoff_seconds = max(
            0.0,
            float(getattr(self.feishu_settings, "http_retry_backoff_seconds", DEFAULT_HTTP_RETRY_BACKOFF_SECONDS)),
        )
        self._message_chunk_chars = max(
            500,
            int(getattr(self.feishu_settings, "message_chunk_chars", DEFAULT_MESSAGE_CHUNK_CHARS)),
        )

    def start(self) -> None:
        """Start the Feishu long-connection client."""
        logger.info("Starting Feishu bot...")
        self._start_daily_cleanup_worker()

        event_handler = (
            lark.EventDispatcherHandler.builder("", "")
            .register_p2_im_message_receive_v1(self._handle_message_event)
            .build()
        )

        ws_client = lark.ws.Client(
            self.feishu_settings.app_id,
            self.feishu_settings.app_secret,
            event_handler=event_handler,
            log_level=lark.LogLevel.INFO,
        )

        ws_client.start()

    def _start_daily_cleanup_worker(self) -> None:
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            return

        self._cleanup_stop_event.clear()
        self._cleanup_thread = threading.Thread(
            target=self._daily_cleanup_loop,
            name="feishu-conversation-cleanup",
            daemon=True,
        )
        self._cleanup_thread.start()

    def _daily_cleanup_loop(self) -> None:
        while not self._cleanup_stop_event.is_set():
            wait_seconds = self._seconds_until_next_cleanup()
            if self._cleanup_stop_event.wait(wait_seconds):
                return
            self._clear_all_conversation_memory()

    def _seconds_until_next_cleanup(self, now: Optional[datetime] = None) -> float:
        now = now or datetime.now().astimezone()
        cleanup_hour = max(0, min(23, int(getattr(self.feishu_settings, "memory_cleanup_hour", 0))))
        cleanup_minute = max(0, min(59, int(getattr(self.feishu_settings, "memory_cleanup_minute", 0))))
        next_cleanup = now.replace(hour=cleanup_hour, minute=cleanup_minute, second=0, microsecond=0)
        if next_cleanup <= now:
            next_cleanup += timedelta(days=1)
        return max((next_cleanup - now).total_seconds(), 1.0)

    def _clear_all_conversation_memory(self) -> None:
        """Clear Feishu conversation memory from cache and persistent storage."""
        if self.redis:
            try:
                keys = list(self.redis.scan_iter(match="feishu:context:*"))
                if keys:
                    self.redis.delete(*keys)
            except Exception as exc:
                logger.warning(f"Failed to clear Feishu conversation memory from Redis: {exc}")

        with self._context_lock:
            self._memory_context.clear()

        if self.database:
            try:
                with self.database.get_session() as session:
                    repo = self._conversation_repository_factory(session)
                    repo.clear_all_memory()
            except Exception as exc:
                logger.warning(f"Failed to clear Feishu conversation memory from PostgreSQL: {exc}")

    def _handle_message_event(self, data: Any) -> None:
        """Handle Feishu events quickly and dispatch background work."""
        try:
            context = self._extract_message_context(data)
            if not context:
                return

            if not self._mark_message_processing(context.message_id):
                logger.info(f"Skipping duplicate Feishu message: {context.message_id}")
                return

            if not self._worker_slots.acquire(blocking=False):
                logger.warning(
                    "Feishu worker queue is full; rejecting message %s",
                    context.message_id,
                )
                if self._send_busy_message(context):
                    self._mark_message_done(context.message_id)
                else:
                    self._mark_message_failed(context.message_id)
                return

            try:
                future = self._worker_executor.submit(self._process_message, context)
            except Exception:
                self._worker_slots.release()
                self._mark_message_failed(context.message_id)
                raise
            future.add_done_callback(lambda _: self._worker_slots.release())
        except Exception as exc:
            logger.error(f"Failed to dispatch Feishu event: {exc}", exc_info=True)

    def _process_message(self, context: FeishuMessageContext) -> None:
        """Call the existing ask API and send the answer back to Feishu."""
        send_succeeded = False
        try:
            with self._get_conversation_lock(context):
                answer_text = self._build_reply(context)
        except Exception as exc:
            logger.error(f"Failed to get RAG answer for Feishu message {context.message_id}: {exc}", exc_info=True)
            answer_text = "暂时无法获取回答，请稍后重试。"

        try:
            self._send_reply_text(
                receive_id=context.receive_id,
                receive_id_type=context.receive_id_type,
                text=answer_text,
            )
            send_succeeded = True
        except Exception as exc:
            logger.error(f"Failed to send Feishu reply for {context.message_id}: {exc}", exc_info=True)
        finally:
            if send_succeeded:
                self._mark_message_done(context.message_id)
            else:
                self._mark_message_failed(context.message_id)

    def _get_conversation_lock(self, context: FeishuMessageContext) -> threading.Lock:
        key = self._context_key(context)
        with self._conversation_locks_guard:
            lock = self._conversation_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._conversation_locks[key] = lock
            return lock

    def _get_async_conversation_lock(self, context: FeishuMessageContext) -> asyncio.Lock:
        key = self._context_key(context)
        with self._async_conversation_locks_guard:
            lock = self._async_conversation_locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._async_conversation_locks[key] = lock
            return lock

    @staticmethod
    def _build_local_context(*, session_id: str, query: str) -> FeishuMessageContext:
        normalized_session_id = str(session_id or "").strip()
        if not normalized_session_id:
            raise ValueError("session_id is required")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query is required")

        return FeishuMessageContext(
            message_id=f"local_{uuid4().hex}",
            chat_id=normalized_session_id,
            chat_type="p2p",
            sender_open_id=normalized_session_id,
            receive_id=normalized_session_id,
            receive_id_type="local",
            query=query,
        )

    def build_local_reply(self, *, session_id: str, query: str) -> str:
        """Run the Feishu conversation pipeline locally without sending Feishu messages."""
        context = self._build_local_context(session_id=session_id, query=query)

        with self._get_conversation_lock(context):
            return self._build_reply(context)

    async def build_local_reply_async(self, *, session_id: str, query: str) -> str:
        """Run the local Feishu pipeline using in-process services."""
        context = self._build_local_context(session_id=session_id, query=query)

        async with self._get_async_conversation_lock(context):
            return await self._build_reply_async(context)

    async def build_local_reply_debug_async(self, *, session_id: str, query: str) -> FeishuLocalDebugReply:
        """Run the local Feishu pipeline and return metadata for offline evaluation."""
        context = self._build_local_context(session_id=session_id, query=query)

        async with self._get_async_conversation_lock(context):
            return await self._build_reply_debug_async(context)

    def _send_busy_message(self, context: FeishuMessageContext) -> bool:
        try:
            self._send_text_message(
                receive_id=context.receive_id,
                receive_id_type=context.receive_id_type,
                text="我现在还有几条问题在处理中，请稍后再发一次。",
            )
            return True
        except Exception as exc:
            logger.warning(f"Failed to send Feishu busy message for {context.message_id}: {exc}")
            return False

    def _build_reply(self, context: FeishuMessageContext) -> str:
        """Route paper-finding and follow-up questions with paper-aware intent handling."""
        query = context.query.strip()
        state = self._load_conversation_state(context)
        decision = self._classify_intent(query, state)

        if decision.intent == INTENT_RESET:
            self._clear_recent_papers(context)
            return "好的，已经清空当前会话记忆。我们可以重新开始。"

        if decision.intent == INTENT_PAPER_SEARCH:
            requested_count = self._extract_requested_paper_count(query)
            papers = self._find_recommended_papers(context, query, requested_count)

            if not papers:
                reply = "我暂时没有在当前已索引的论文库里找到合适的论文。你可以换一个更具体的方向，或者先让我补充导入更多论文。"
                self._remember_exchange(
                    context,
                    state=FeishuConversationState(
                        recent_papers=state.recent_papers,
                        active_papers=state.active_papers,
                        last_intent=decision.intent,
                        last_query=query,
                        summary=state.summary,
                        recent_messages=state.recent_messages,
                        persisted_message_count=state.persisted_message_count,
                    ),
                    intent=decision.intent,
                    user_query=query,
                    assistant_reply=reply,
                )
                return reply

            reply = self._format_paper_recommendations(query, papers)
            self._remember_exchange(
                context,
                state=FeishuConversationState(
                    recent_papers=papers,
                    active_papers=papers,
                    last_intent=decision.intent,
                    last_query=query,
                    summary=state.summary,
                    recent_messages=state.recent_messages,
                    persisted_message_count=state.persisted_message_count,
                ),
                intent=decision.intent,
                user_query=query,
                assistant_reply=reply,
            )
            return reply

        if decision.intent in {
            INTENT_PAPER_SUMMARY,
            INTENT_PAPER_METHOD,
            INTENT_PAPER_EXPERIMENT,
            INTENT_PAPER_DEEP_DIVE,
            INTENT_PAPER_COMPARE,
        }:
            if not decision.papers:
                reply = "我现在还没有记住你上一轮提到的论文。你可以先让我推荐几篇，或者直接告诉我论文标题 / arXiv ID。"
                self._remember_exchange(
                    context,
                    state=FeishuConversationState(
                        recent_papers=state.recent_papers,
                        active_papers=state.active_papers,
                        last_intent=decision.intent,
                        last_query=query,
                        summary=state.summary,
                        recent_messages=state.recent_messages,
                        persisted_message_count=state.persisted_message_count,
                    ),
                    intent=decision.intent,
                    user_query=query,
                    assistant_reply=reply,
                )
                return reply

            reply = self._ask_paper_focused_rag(query, decision.papers, decision.intent)
            self._remember_exchange(
                context,
                state=FeishuConversationState(
                    recent_papers=state.recent_papers or decision.papers,
                    active_papers=decision.papers,
                    last_intent=decision.intent,
                    last_query=query,
                    summary=state.summary,
                    recent_messages=state.recent_messages,
                    persisted_message_count=state.persisted_message_count,
                ),
                intent=decision.intent,
                user_query=query,
                assistant_reply=reply,
            )
            return reply

        rag_query = self._build_general_contextual_query(query, state)
        route = self._select_general_rag_route(query, state)
        reply = self._ask_rag_api(query=rag_query, endpoint_path=route.endpoint_path)
        self._remember_exchange(
            context,
            state=FeishuConversationState(
                recent_papers=state.recent_papers,
                active_papers=state.active_papers,
                last_intent=INTENT_GENERAL_RAG,
                last_query=query,
                summary=state.summary,
                recent_messages=state.recent_messages,
                persisted_message_count=state.persisted_message_count,
            ),
            intent=INTENT_GENERAL_RAG,
            user_query=query,
            assistant_reply=reply,
        )
        return reply

    async def _build_reply_async(self, context: FeishuMessageContext) -> str:
        """Async variant used by the local Feishu endpoint to avoid self-HTTP calls."""
        query = context.query.strip()
        state = self._load_conversation_state(context)
        decision = self._classify_intent(query, state)

        if decision.intent == INTENT_RESET:
            self._clear_recent_papers(context)
            return "好的，已经清空当前会话记忆。我们可以重新开始。"

        if decision.intent == INTENT_PAPER_SEARCH:
            requested_count = self._extract_requested_paper_count(query)
            papers = await self._find_recommended_papers_async(context, query, requested_count)

            if not papers:
                reply = "我暂时没有在当前已索引的论文库里找到合适的论文。你可以换一个更具体的方向，或者先让我补充导入更多论文。"
                self._remember_exchange(
                    context,
                    state=FeishuConversationState(
                        recent_papers=state.recent_papers,
                        active_papers=state.active_papers,
                        last_intent=decision.intent,
                        last_query=query,
                        summary=state.summary,
                        recent_messages=state.recent_messages,
                        persisted_message_count=state.persisted_message_count,
                    ),
                    intent=decision.intent,
                    user_query=query,
                    assistant_reply=reply,
                )
                return reply

            reply = self._format_paper_recommendations(query, papers)
            self._remember_exchange(
                context,
                state=FeishuConversationState(
                    recent_papers=papers,
                    active_papers=papers,
                    last_intent=decision.intent,
                    last_query=query,
                    summary=state.summary,
                    recent_messages=state.recent_messages,
                    persisted_message_count=state.persisted_message_count,
                ),
                intent=decision.intent,
                user_query=query,
                assistant_reply=reply,
            )
            return reply

        if decision.intent in {
            INTENT_PAPER_SUMMARY,
            INTENT_PAPER_METHOD,
            INTENT_PAPER_EXPERIMENT,
            INTENT_PAPER_DEEP_DIVE,
            INTENT_PAPER_COMPARE,
        }:
            if not decision.papers:
                reply = "我现在还没有记住你上一轮提到的论文。你可以先让我推荐几篇，或者直接告诉我论文标题 / arXiv ID。"
                self._remember_exchange(
                    context,
                    state=FeishuConversationState(
                        recent_papers=state.recent_papers,
                        active_papers=state.active_papers,
                        last_intent=decision.intent,
                        last_query=query,
                        summary=state.summary,
                        recent_messages=state.recent_messages,
                        persisted_message_count=state.persisted_message_count,
                    ),
                    intent=decision.intent,
                    user_query=query,
                    assistant_reply=reply,
                )
                return reply

            rewritten_query = self._build_paper_focused_query(query, decision.papers, decision.intent)
            reply = await self._ask_rag_api_async(
                query=rewritten_query,
                arxiv_ids=[paper.arxiv_id for paper in decision.papers],
                force_standard=True,
                direct_chunks_per_paper=self._direct_chunks_per_paper(decision.intent, len(decision.papers)),
            )
            self._remember_exchange(
                context,
                state=FeishuConversationState(
                    recent_papers=state.recent_papers or decision.papers,
                    active_papers=decision.papers,
                    last_intent=decision.intent,
                    last_query=query,
                    summary=state.summary,
                    recent_messages=state.recent_messages,
                    persisted_message_count=state.persisted_message_count,
                ),
                intent=decision.intent,
                user_query=query,
                assistant_reply=reply,
            )
            return reply

        rag_query = self._build_general_contextual_query(query, state)
        route = self._select_general_rag_route(query, state)
        reply = await self._ask_rag_api_async(query=rag_query, endpoint_path=route.endpoint_path)
        self._remember_exchange(
            context,
            state=FeishuConversationState(
                recent_papers=state.recent_papers,
                active_papers=state.active_papers,
                last_intent=INTENT_GENERAL_RAG,
                last_query=query,
                summary=state.summary,
                recent_messages=state.recent_messages,
                persisted_message_count=state.persisted_message_count,
            ),
            intent=INTENT_GENERAL_RAG,
            user_query=query,
            assistant_reply=reply,
        )
        return reply

    async def _build_reply_debug_async(self, context: FeishuMessageContext) -> FeishuLocalDebugReply:
        """Async local reply variant that exposes RAG metadata for evaluation."""
        query = context.query.strip()
        state = self._load_conversation_state(context)
        decision = self._classify_intent(query, state)

        if decision.intent == INTENT_RESET:
            self._clear_recent_papers(context)
            return FeishuLocalDebugReply(
                answer="好的，已经清空当前会话记忆。我们可以重新开始。",
                intent=decision.intent,
            )

        if decision.intent == INTENT_PAPER_SEARCH:
            requested_count = self._extract_requested_paper_count(query)
            papers = await self._find_recommended_papers_async(context, query, requested_count)

            if not papers:
                reply = "我暂时没有在当前已索引的论文库里找到合适的论文。你可以换一个更具体的方向，或者先让我补充导入更多论文。"
                self._remember_exchange(
                    context,
                    state=FeishuConversationState(
                        recent_papers=state.recent_papers,
                        active_papers=state.active_papers,
                        last_intent=decision.intent,
                        last_query=query,
                        summary=state.summary,
                        recent_messages=state.recent_messages,
                        persisted_message_count=state.persisted_message_count,
                    ),
                    intent=decision.intent,
                    user_query=query,
                    assistant_reply=reply,
                )
                return FeishuLocalDebugReply(answer=reply, intent=decision.intent, route="paper_search")

            reply = self._format_paper_recommendations(query, papers)
            self._remember_exchange(
                context,
                state=FeishuConversationState(
                    recent_papers=papers,
                    active_papers=papers,
                    last_intent=decision.intent,
                    last_query=query,
                    summary=state.summary,
                    recent_messages=state.recent_messages,
                    persisted_message_count=state.persisted_message_count,
                ),
                intent=decision.intent,
                user_query=query,
                assistant_reply=reply,
            )
            sources = [f"https://arxiv.org/abs/{paper.arxiv_id.split('v')[0]}" for paper in papers]
            contexts = [paper.abstract for paper in papers if paper.abstract]
            return FeishuLocalDebugReply(
                answer=reply,
                contexts=contexts,
                sources=sources,
                intent=decision.intent,
                rewritten_query=FeishuPaperIngestionService.build_local_search_text(query),
                route="paper_search",
            )

        if decision.intent in {
            INTENT_PAPER_SUMMARY,
            INTENT_PAPER_METHOD,
            INTENT_PAPER_EXPERIMENT,
            INTENT_PAPER_DEEP_DIVE,
            INTENT_PAPER_COMPARE,
        }:
            if not decision.papers:
                reply = "我现在还没有记住你上一轮提到的论文。你可以先让我推荐几篇，或者直接告诉我论文标题 / arXiv ID。"
                self._remember_exchange(
                    context,
                    state=FeishuConversationState(
                        recent_papers=state.recent_papers,
                        active_papers=state.active_papers,
                        last_intent=decision.intent,
                        last_query=query,
                        summary=state.summary,
                        recent_messages=state.recent_messages,
                        persisted_message_count=state.persisted_message_count,
                    ),
                    intent=decision.intent,
                    user_query=query,
                    assistant_reply=reply,
                )
                return FeishuLocalDebugReply(answer=reply, intent=decision.intent, route="/api/v1/ask")

            rewritten_query = self._build_paper_focused_query(query, decision.papers, decision.intent)
            rag_result = await self._ask_rag_api_debug_async(
                query=rewritten_query,
                arxiv_ids=[paper.arxiv_id for paper in decision.papers],
                force_standard=True,
                direct_chunks_per_paper=self._direct_chunks_per_paper(decision.intent, len(decision.papers)),
            )
            self._remember_exchange(
                context,
                state=FeishuConversationState(
                    recent_papers=state.recent_papers or decision.papers,
                    active_papers=decision.papers,
                    last_intent=decision.intent,
                    last_query=query,
                    summary=state.summary,
                    recent_messages=state.recent_messages,
                    persisted_message_count=state.persisted_message_count,
                ),
                intent=decision.intent,
                user_query=query,
                assistant_reply=rag_result.reply,
            )
            return FeishuLocalDebugReply(
                answer=rag_result.answer,
                contexts=rag_result.contexts,
                sources=rag_result.sources,
                intent=decision.intent,
                rewritten_query=rewritten_query,
                route=rag_result.route,
            )

        rag_query = self._build_general_contextual_query(query, state)
        route = self._select_general_rag_route(query, state)
        rag_result = await self._ask_rag_api_debug_async(query=rag_query, endpoint_path=route.endpoint_path)
        self._remember_exchange(
            context,
            state=FeishuConversationState(
                recent_papers=state.recent_papers,
                active_papers=state.active_papers,
                last_intent=INTENT_GENERAL_RAG,
                last_query=query,
                summary=state.summary,
                recent_messages=state.recent_messages,
                persisted_message_count=state.persisted_message_count,
            ),
            intent=INTENT_GENERAL_RAG,
            user_query=query,
            assistant_reply=rag_result.reply,
        )
        return FeishuLocalDebugReply(
            answer=rag_result.answer,
            contexts=rag_result.contexts,
            sources=rag_result.sources,
            intent=INTENT_GENERAL_RAG,
            rewritten_query=rag_query,
            route=rag_result.route,
        )

    def _ask_rag_api(
        self,
        query: str,
        arxiv_ids: Optional[list[str]] = None,
        force_standard: bool = False,
        direct_chunks_per_paper: Optional[int] = None,
        endpoint_path: Optional[str] = None,
    ) -> str:
        """Call the configured ask endpoint so the bot reuses the current RAG stack."""
        if self._local_runtime is not None:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(
                    self._ask_rag_api_async(
                        query=query,
                        arxiv_ids=arxiv_ids,
                        force_standard=force_standard,
                        direct_chunks_per_paper=direct_chunks_per_paper,
                        endpoint_path=endpoint_path,
                    )
                )
            raise RuntimeError("Use build_local_reply_async for local Feishu requests inside async contexts.")

        provider = self.settings.resolve_llm_provider(self.feishu_settings.llm_provider)
        model = self.settings.resolve_llm_model(provider, self.feishu_settings.model)
        payload = {
            "query": query,
            "top_k": self.feishu_settings.top_k,
            "use_hybrid": False if arxiv_ids else self.feishu_settings.use_hybrid,
            "provider": provider,
            "model": model,
        }
        if arxiv_ids:
            payload["arxiv_ids"] = arxiv_ids
        if direct_chunks_per_paper:
            payload["direct_chunks_per_paper"] = direct_chunks_per_paper

        endpoint_path = (
            "/api/v1/ask"
            if force_standard or arxiv_ids
            else (endpoint_path or self.feishu_settings.ask_endpoint_path.strip() or "/api/v1/ask")
        )
        data = self._post_api_json(
            endpoint_path=endpoint_path,
            payload=payload,
            timeout=self.feishu_settings.request_timeout_seconds,
        )

        return self._format_rag_reply(
            answer=data.get("answer", "").strip() or "暂时没有生成有效回答。",
            sources=data.get("sources", []),
        )

    async def _ask_rag_api_async(
        self,
        query: str,
        arxiv_ids: Optional[list[str]] = None,
        force_standard: bool = False,
        direct_chunks_per_paper: Optional[int] = None,
        endpoint_path: Optional[str] = None,
    ) -> str:
        """Call RAG in-process when running behind the local Feishu endpoint."""
        if self._local_runtime is None:
            return await asyncio.to_thread(
                self._ask_rag_api,
                query,
                arxiv_ids,
                force_standard,
                direct_chunks_per_paper,
                endpoint_path,
            )

        provider = self.settings.resolve_llm_provider(self.feishu_settings.llm_provider)
        model = self.settings.resolve_llm_model(provider, self.feishu_settings.model)
        resolved_endpoint_path = (
            "/api/v1/ask"
            if force_standard or arxiv_ids
            else (endpoint_path or self.feishu_settings.ask_endpoint_path.strip() or "/api/v1/ask")
        )
        request = AskRequest(
            query=query,
            top_k=self.feishu_settings.top_k,
            use_hybrid=False if arxiv_ids else self.feishu_settings.use_hybrid,
            provider=provider,
            model=model,
            arxiv_ids=arxiv_ids,
            direct_chunks_per_paper=direct_chunks_per_paper,
        )

        if resolved_endpoint_path == "/api/v1/ask-agentic" and not (force_standard or arxiv_ids):
            response = await ask_agentic(
                request=request,
                opensearch_client=self._local_runtime.opensearch_client,
                ollama_client=self._local_runtime.ollama_client,
                embeddings_service=self._local_runtime.embeddings_service,
                langfuse_tracer=self._local_runtime.langfuse_tracer,
                cache_client=self._local_runtime.cache_client,
            )
        else:
            response = await ask_question(
                request=request,
                opensearch_client=self._local_runtime.opensearch_client,
                embeddings_service=self._local_runtime.embeddings_service,
                ollama_client=self._local_runtime.ollama_client,
                langfuse_tracer=self._local_runtime.langfuse_tracer,
                cache_client=self._local_runtime.cache_client,
            )

        return self._format_rag_reply(
            answer=getattr(response, "answer", "").strip() or "暂时没有生成有效回答。",
            sources=getattr(response, "sources", []),
        )

    async def _ask_rag_api_debug_async(
        self,
        query: str,
        arxiv_ids: Optional[list[str]] = None,
        force_standard: bool = False,
        direct_chunks_per_paper: Optional[int] = None,
        endpoint_path: Optional[str] = None,
    ) -> FeishuRAGDebugResult:
        """Call RAG and keep the retrieved contexts for Ragas evaluation."""
        if self._local_runtime is None:
            reply = await asyncio.to_thread(
                self._ask_rag_api,
                query,
                arxiv_ids,
                force_standard,
                direct_chunks_per_paper,
                endpoint_path,
            )
            return FeishuRAGDebugResult(reply=reply, answer=reply, rewritten_query=query, route=endpoint_path or "/api/v1/ask")

        provider = self.settings.resolve_llm_provider(self.feishu_settings.llm_provider)
        model = self.settings.resolve_llm_model(provider, self.feishu_settings.model)
        resolved_endpoint_path = (
            "/api/v1/ask"
            if force_standard or arxiv_ids
            else (endpoint_path or self.feishu_settings.ask_endpoint_path.strip() or "/api/v1/ask")
        )
        request = AskRequest(
            query=query,
            top_k=self.feishu_settings.top_k,
            use_hybrid=False if arxiv_ids else self.feishu_settings.use_hybrid,
            provider=provider,
            model=model,
            arxiv_ids=arxiv_ids,
            direct_chunks_per_paper=direct_chunks_per_paper,
            include_contexts=True,
        )

        if resolved_endpoint_path == "/api/v1/ask-agentic" and not (force_standard or arxiv_ids):
            response = await ask_agentic(
                request=request,
                opensearch_client=self._local_runtime.opensearch_client,
                ollama_client=self._local_runtime.ollama_client,
                embeddings_service=self._local_runtime.embeddings_service,
                langfuse_tracer=self._local_runtime.langfuse_tracer,
                cache_client=self._local_runtime.cache_client,
            )
        else:
            response = await ask_question(
                request=request,
                opensearch_client=self._local_runtime.opensearch_client,
                embeddings_service=self._local_runtime.embeddings_service,
                ollama_client=self._local_runtime.ollama_client,
                langfuse_tracer=self._local_runtime.langfuse_tracer,
                cache_client=self._local_runtime.cache_client,
            )

        answer = getattr(response, "answer", "").strip() or "暂时没有生成有效回答。"
        sources = self._normalize_sources(getattr(response, "sources", []))
        return FeishuRAGDebugResult(
            reply=self._format_rag_reply(answer=answer, sources=getattr(response, "sources", [])),
            answer=answer,
            contexts=[context for context in (getattr(response, "contexts", None) or []) if context],
            sources=sources,
            rewritten_query=query,
            route=resolved_endpoint_path,
        )

    def _format_rag_reply(self, *, answer: str, sources: list[str]) -> str:
        answer = compact_repeated_single_paper_citations(answer)
        normalized_sources = self._normalize_sources(sources)

        if not normalized_sources:
            return answer

        source_lines = "\n".join(f"{idx}. {url}" for idx, url in enumerate(normalized_sources, 1))
        return f"{answer}\n\n参考来源：\n{source_lines}"

    def _post_api_json(self, endpoint_path: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
        endpoint = endpoint_path if endpoint_path.startswith("/") else f"/{endpoint_path}"
        url = f"{self.feishu_settings.api_base_url.rstrip('/')}{endpoint}"
        return self._post_json(url=url, payload=payload, timeout=timeout)

    def _post_json(self, url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
        attempts = self._http_max_retries + 1
        last_error: Optional[Exception] = None

        for attempt in range(attempts):
            started = time.monotonic()
            try:
                response = self._http_session.post(url, json=payload, timeout=timeout)
                elapsed = time.monotonic() - started
                status_code = getattr(response, "status_code", None)

                if (
                    isinstance(status_code, int)
                    and status_code in RETRY_HTTP_STATUS_CODES
                    and attempt < attempts - 1
                ):
                    logger.warning(
                        "Feishu API POST %s returned retryable status=%s latency=%.2fs attempt=%s/%s",
                        url,
                        status_code,
                        elapsed,
                        attempt + 1,
                        attempts,
                    )
                    self._sleep_before_http_retry(attempt)
                    continue

                response.raise_for_status()
                logger.info(
                    "Feishu API POST %s completed status=%s latency=%.2fs",
                    url,
                    status_code if isinstance(status_code, int) else "unknown",
                    elapsed,
                )
                data = response.json()
                return data if isinstance(data, dict) else {}
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_error = exc
                if attempt >= attempts - 1:
                    break
                logger.warning(
                    "Feishu API POST %s failed with %s; retrying attempt=%s/%s",
                    url,
                    exc.__class__.__name__,
                    attempt + 1,
                    attempts,
                )
                self._sleep_before_http_retry(attempt)
            except requests.HTTPError as exc:
                last_error = exc
                response = getattr(exc, "response", None)
                status_code = getattr(response, "status_code", None)
                if (
                    isinstance(status_code, int)
                    and status_code in RETRY_HTTP_STATUS_CODES
                    and attempt < attempts - 1
                ):
                    logger.warning(
                        "Feishu API POST %s raised retryable HTTP status=%s attempt=%s/%s",
                        url,
                        status_code,
                        attempt + 1,
                        attempts,
                    )
                    self._sleep_before_http_retry(attempt)
                    continue
                raise

        if last_error:
            raise last_error
        raise RuntimeError(f"Feishu API POST {url} failed without a response")

    def _sleep_before_http_retry(self, attempt: int) -> None:
        if self._http_retry_backoff_seconds <= 0:
            return
        time.sleep(self._http_retry_backoff_seconds * (2**attempt))

    def _build_general_contextual_query(self, query: str, state: FeishuConversationState) -> str:
        """Rewrite short/general follow-ups with persisted conversation memory."""
        if not self._looks_like_general_followup(query, state):
            return query

        history = self._format_recent_messages_for_prompt(state.recent_messages[-8:])
        parts = [
            "请结合下面的历史对话理解用户的连续追问，再回答当前问题。",
            "如果当前问题已经是独立新问题，可以忽略不相关历史。",
        ]
        if state.summary:
            parts.append(f"历史摘要：{state.summary}")
        if history:
            parts.append(f"最近对话：\n{history}")
        parts.append(f"当前问题：{query}")
        contextual_query = "\n".join(parts)
        return self._truncate(contextual_query, CONTEXTUAL_QUERY_MAX_CHARS)

    def _looks_like_general_followup(self, query: str, state: FeishuConversationState) -> bool:
        if not (state.summary or state.recent_messages or state.last_query):
            return False

        lowered = query.lower()
        hints = FOLLOWUP_HINTS + DETAIL_HINTS + COMPARE_HINTS + CONTEXTUAL_REFERENCE_HINTS + (
            "刚才",
            "前面",
            "上面",
            "上一条",
            "继续",
            "that",
            "it",
            "them",
            "more",
        )
        if any(hint in query or hint in lowered for hint in hints):
            return True

        return len(query.strip()) <= 24 and bool(state.last_query)

    def _remember_exchange(
        self,
        context: FeishuMessageContext,
        *,
        state: FeishuConversationState,
        intent: str,
        user_query: str,
        assistant_reply: str,
    ) -> None:
        user_turn = FeishuConversationTurn(
            role="user",
            content=user_query,
            intent=intent,
            message_id=context.message_id,
        )
        assistant_turn = FeishuConversationTurn(
            role="assistant",
            content=assistant_reply,
            intent=intent,
        )
        recent_messages = (state.recent_messages + [user_turn, assistant_turn])[-RECENT_MESSAGE_LIMIT:]
        estimated_count = state.persisted_message_count + 2 if state.persisted_message_count else len(recent_messages)
        summary = self._build_conversation_summary(state.summary, recent_messages)
        updated_state = FeishuConversationState(
            recent_papers=state.recent_papers,
            active_papers=state.active_papers,
            last_intent=state.last_intent,
            last_query=state.last_query,
            summary=summary,
            recent_messages=recent_messages,
            persisted_message_count=estimated_count,
        )

        if self.database:
            persisted_state = self._persist_exchange_to_database(
                context,
                state=updated_state,
                intent=intent,
                user_query=user_query,
                assistant_reply=assistant_reply,
            )
            self._write_cached_conversation_state(context, persisted_state)
            return

        self._write_cached_conversation_state(context, updated_state)

    def _persist_exchange_to_database(
        self,
        context: FeishuMessageContext,
        *,
        state: FeishuConversationState,
        intent: str,
        user_query: str,
        assistant_reply: str,
    ) -> FeishuConversationState:
        try:
            with self.database.get_session() as session:
                repo = self._conversation_repository_factory(session)
                message_count, stored_recent_messages = repo.append_exchange_and_update_state(
                    conversation_key=self._context_key(context),
                    chat_id=context.chat_id,
                    chat_type=context.chat_type,
                    sender_open_id=context.sender_open_id,
                    receive_id=context.receive_id,
                    receive_id_type=context.receive_id_type,
                    user_query=user_query,
                    assistant_reply=assistant_reply,
                    intent=intent,
                    message_id=context.message_id,
                    recent_papers=self._serialize_papers(state.recent_papers),
                    active_papers=self._serialize_papers(state.active_papers),
                    last_intent=state.last_intent,
                    last_query=state.last_query,
                    summary=state.summary,
                    keep_latest=self._conversation_max_messages,
                    recent_limit=RECENT_MESSAGE_LIMIT,
                )
                persisted_recent_messages = self._turns_from_stored_messages(stored_recent_messages)
                return FeishuConversationState(
                    recent_papers=state.recent_papers,
                    active_papers=state.active_papers,
                    last_intent=state.last_intent,
                    last_query=state.last_query,
                    summary=state.summary,
                    recent_messages=persisted_recent_messages or state.recent_messages,
                    persisted_message_count=message_count,
                )
        except Exception as exc:
            logger.warning(f"Failed to persist Feishu conversation memory to PostgreSQL: {exc}")
            return state

    def _classify_intent(self, query: str, state: FeishuConversationState) -> FeishuIntentDecision:
        if self._is_reset_query(query):
            return FeishuIntentDecision(intent=INTENT_RESET, papers=[])

        if self._is_paper_search_query(query):
            return FeishuIntentDecision(intent=INTENT_PAPER_SEARCH, papers=[])

        referenced_papers = self._resolve_followup_papers(query, state)
        if referenced_papers:
            if self._is_compare_query(query):
                return FeishuIntentDecision(intent=INTENT_PAPER_COMPARE, papers=referenced_papers)
            if self._is_experiment_question(query):
                return FeishuIntentDecision(intent=INTENT_PAPER_EXPERIMENT, papers=referenced_papers)
            if self._is_method_question(query):
                return FeishuIntentDecision(intent=INTENT_PAPER_METHOD, papers=referenced_papers)
            if self._is_deep_dive_query(query):
                return FeishuIntentDecision(intent=INTENT_PAPER_DEEP_DIVE, papers=referenced_papers)
            return FeishuIntentDecision(intent=INTENT_PAPER_SUMMARY, papers=referenced_papers)

        if self._is_reference_query(query):
            if self._is_compare_query(query):
                return FeishuIntentDecision(intent=INTENT_PAPER_COMPARE, papers=[])
            if self._is_experiment_question(query):
                return FeishuIntentDecision(intent=INTENT_PAPER_EXPERIMENT, papers=[])
            if self._is_method_question(query):
                return FeishuIntentDecision(intent=INTENT_PAPER_METHOD, papers=[])
            if self._is_deep_dive_query(query):
                return FeishuIntentDecision(intent=INTENT_PAPER_DEEP_DIVE, papers=[])
            return FeishuIntentDecision(intent=INTENT_PAPER_SUMMARY, papers=[])

        return FeishuIntentDecision(intent=INTENT_GENERAL_RAG, papers=[])

    def _select_general_rag_route(self, query: str, state: FeishuConversationState) -> FeishuRouteDecision:
        complexity = self._classify_general_query_complexity(query, state)
        endpoint_path = "/api/v1/ask-agentic" if complexity == QUERY_COMPLEXITY_COMPLEX else "/api/v1/ask"
        return FeishuRouteDecision(endpoint_path=endpoint_path, complexity=complexity)

    def _classify_general_query_complexity(self, query: str, state: FeishuConversationState) -> str:
        normalized = " ".join(query.split())
        lowered = normalized.lower()

        if self._is_compare_query(normalized):
            return QUERY_COMPLEXITY_COMPLEX

        if any(hint in normalized or hint in lowered for hint in COMPLEX_REASONING_HINTS):
            return QUERY_COMPLEXITY_COMPLEX

        split_parts = [
            part.strip()
            for part in re.split(r"[，,；;？?。]\s*|\s+(?:and|or|plus)\s+", normalized, flags=re.IGNORECASE)
            if part.strip()
        ]
        if len(split_parts) >= 3:
            return QUERY_COMPLEXITY_COMPLEX

        connective_count = sum(
            normalized.count(token) for token in ("以及", "并且", "同时", "分别", "先", "再", "然后", "另外")
        )
        if connective_count >= 2:
            return QUERY_COMPLEXITY_COMPLEX

        if len(normalized) >= 80:
            return QUERY_COMPLEXITY_COMPLEX

        if any(hint in normalized or hint in lowered for hint in MEDIUM_COMPLEXITY_HINTS):
            return QUERY_COMPLEXITY_MEDIUM

        if state.last_intent in PAPER_RELATED_INTENTS and len(normalized) <= 24:
            return QUERY_COMPLEXITY_MEDIUM

        return QUERY_COMPLEXITY_SIMPLE

    def _resolve_followup_papers(self, query: str, state: FeishuConversationState) -> list[FeishuPaperReference]:
        explicit_papers = self._resolve_referenced_papers_from_list(state.recent_papers or state.active_papers, query)
        if explicit_papers:
            return explicit_papers

        if not self._looks_like_contextual_followup(query, state):
            return []

        if state.active_papers:
            return state.active_papers
        return state.recent_papers[: min(3, len(state.recent_papers))]

    def _looks_like_contextual_followup(self, query: str, state: FeishuConversationState) -> bool:
        if not (state.active_papers or state.recent_papers):
            return False

        lowered = query.lower()
        if any(hint in query or hint in lowered for hint in FOLLOWUP_HINTS):
            return True
        if any(hint in query or hint in lowered for hint in DETAIL_HINTS):
            return True
        if any(hint in query or hint in lowered for hint in COMPARE_HINTS):
            return True
        if any(hint in query or hint in lowered for hint in CONTEXTUAL_REFERENCE_HINTS):
            return True
        return len(query.strip()) <= 18 and state.last_intent in PAPER_RELATED_INTENTS

    @staticmethod
    def _is_compare_query(query: str) -> bool:
        lowered = query.lower()
        if any(hint in query or hint in lowered for hint in BASELINE_QUESTION_HINTS):
            return False
        return any(hint in query or hint in lowered for hint in COMPARE_HINTS)

    @staticmethod
    def _is_deep_dive_query(query: str) -> bool:
        if FeishuBot._is_experiment_question(query) or FeishuBot._is_method_question(query):
            return False
        lowered = query.lower()
        return any(hint in query or hint in lowered for hint in FOLLOWUP_HINTS + DETAIL_HINTS)

    @staticmethod
    def _is_method_question(query: str) -> bool:
        lowered = query.lower()
        if any(hint in query or hint in lowered for hint in COMPARE_HINTS):
            return False
        if any(hint in query or hint in lowered for hint in EXPERIMENT_QUESTION_HINTS + BASELINE_QUESTION_HINTS):
            return False
        return any(hint in query or hint in lowered for hint in METHOD_QUESTION_HINTS)

    @staticmethod
    def _is_experiment_question(query: str) -> bool:
        lowered = query.lower()
        return any(hint in query or hint in lowered for hint in EXPERIMENT_QUESTION_HINTS + BASELINE_QUESTION_HINTS)

    def _ask_paper_focused_rag(
        self,
        query: str,
        papers: list[FeishuPaperReference],
        intent: str,
    ) -> str:
        rewritten_query = self._build_paper_focused_query(query, papers, intent)
        return self._ask_rag_api(
            query=rewritten_query,
            arxiv_ids=[paper.arxiv_id for paper in papers],
            force_standard=True,
            direct_chunks_per_paper=self._direct_chunks_per_paper(intent, len(papers)),
        )

    def _build_paper_focused_query(
        self,
        query: str,
        papers: list[FeishuPaperReference],
        intent: str,
    ) -> str:
        paper_lines = "\n".join(
            f"{index}. {paper.title} (arXiv:{paper.arxiv_id})" for index, paper in enumerate(papers, 1)
        )

        if intent == INTENT_PAPER_COMPARE:
            task = (
                "请严格只基于下面这些论文的内容作答，不要引入其它论文。"
                "先分别概括每篇论文的研究问题、核心方法、实验设置和主要结果，"
                "再从研究目标、输入模态/传感器、技术路线、实验环境、关键指标、适用场景、优势与局限做详细对比。"
                "回答不要压缩成短摘要；请用小标题和对比表述展开，至少包含“逐篇分析”和“综合对比”两大部分。"
                "本次属于深度分析模式，系统会提供较多论文片段；请充分利用这些片段，而不是只做摘要。"
            )
        elif intent == INTENT_PAPER_EXPERIMENT:
            if len(papers) == 1:
                task = (
                    "请只聚焦这篇论文的实验部分回答用户问题，不要展开无关背景。"
                    "必须分别说明：1) 实验或验证用了什么方法/设置；2) 主要效果、指标和结果；"
                    "3) 和哪些基线、传统方法或消融变体进行了对比；4) 这些对比说明了什么。"
                    "如果论文片段没有覆盖某项信息，请明确说“论文片段未覆盖该信息”。"
                )
            else:
                task = (
                    "请按论文分别聚焦实验部分回答用户问题，不要展开无关背景。"
                    "每篇论文都必须说明：1) 实验或验证用了什么方法/设置；2) 主要效果、指标和结果；"
                    "3) 和哪些基线、传统方法或消融变体进行了对比；4) 这些对比说明了什么。"
                    "最后只做简短横向总结，不要把问题改写成泛泛的论文间综合对比。"
                    "如果论文片段没有覆盖某项信息，请明确说“论文片段未覆盖该信息”。"
                )
        elif intent == INTENT_PAPER_METHOD:
            if len(papers) == 1:
                task = (
                    "请只回答这篇论文的方法，不要扩展成整篇论文综述。"
                    "优先说明：1) 方法属于什么总体框架或范式；2) 关键模块或系统组成；"
                    "3) 输入、状态/观测、动作、输出分别是什么；4) 训练或优化算法、损失/奖励如何设计；"
                    "5) 方法相比传统做法的关键差异。"
                    "不要默认展开背景、意义、贡献、局限、适用场景，除非用户明确追问。"
                    "如果论文片段没有覆盖某项信息，请明确说“论文片段未覆盖该信息”。"
                )
            else:
                task = (
                    "请按论文分别回答它们用了什么方法，不要扩展成整篇论文综述。"
                    "每篇论文优先说明：1) 总体方法框架；2) 关键模块；3) 输入与输出；"
                    "4) 训练或优化算法、损失/奖励设计；5) 与传统做法的关键差异。"
                    "最后只做简短方法对比，不要把问题改写成背景、贡献、局限的大而全总结。"
                    "如果论文片段没有覆盖某项信息，请明确说“论文片段未覆盖该信息”。"
                )
        elif intent == INTENT_PAPER_DEEP_DIVE:
            if len(papers) == 1:
                task = (
                    "请对这篇论文做深入讲解，不要写成短摘要。"
                    "至少覆盖：1) 研究背景与问题；2) 为什么这个问题重要；3) 核心思路；4) 方法流程；"
                    "5) 关键技术细节；6) 实验设置；7) 主要结果和指标；8) 贡献；9) 局限；10) 适用场景。"
                    "每个部分至少写 2-4 句，尽量解释清楚因果逻辑。"
                    "本次属于深度分析模式，系统会提供较多论文片段；请充分利用这些片段，而不是只做摘要。"
                )
            else:
                task = (
                    "请分别详细分析每篇论文，不要写成短摘要。"
                    "对每篇论文都至少覆盖：1) 研究背景与问题；2) 为什么这个问题重要；3) 核心思路；"
                    "4) 方法流程；5) 关键技术细节；6) 实验设置；7) 主要结果和指标；8) 贡献；9) 局限；"
                    "10) 适用场景。最后再给出综合联系与差异。"
                    "每篇论文至少写 5-8 个要点，每个要点 2-4 句；如果用户要求“详细”，总回答不要少于约 1500 个汉字。"
                    "本次属于深度分析模式，系统会提供较多论文片段；请充分利用这些片段，而不是只做摘要。"
                )
        else:
            if len(papers) == 1:
                task = "请围绕这篇论文回答用户问题，清楚说明研究问题、方法、实验和结论；不要只给一句话摘要。"
            else:
                task = "请分别说明这些论文各自在解决什么问题、用了什么方法、实验发现了什么，并给出对比；不要只给一句话摘要。"

        return (
            f"{task}\n"
            "如果检索上下文不足以支持某个细节，请明确说明“论文片段未覆盖该信息”，不要编造。\n"
            "请默认使用简体中文，回答要结构化，使用小标题、编号列表和清晰段落。"
            "优先直接回答用户原始问题，不要把单点问题擅自扩展成整篇论文综述。"
            "不要用“背景/方法/结果”各一句的压缩格式糊弄用户。\n"
            f"目标论文：\n{paper_lines}\n"
            f"用户原始问题：{query}"
        )

    @staticmethod
    def _direct_chunks_per_paper(intent: str, paper_count: int) -> int:
        if intent == INTENT_PAPER_COMPARE:
            return 10
        if intent == INTENT_PAPER_EXPERIMENT:
            return 10 if paper_count == 1 else 8
        if intent == INTENT_PAPER_METHOD:
            return 12 if paper_count == 1 else 10
        if intent == INTENT_PAPER_DEEP_DIVE:
            return 16 if paper_count == 1 else 12
        return 8

    @property
    def paper_ingestion_service(self) -> FeishuPaperIngestionService:
        if self._paper_ingestion_service is None:
            self._paper_ingestion_service = FeishuPaperIngestionService(self.settings)
        return self._paper_ingestion_service

    def _find_recommended_papers(
        self, context: FeishuMessageContext, query: str, requested_count: int
    ) -> list[FeishuPaperReference]:
        """Return real indexed papers instead of letting the LLM invent paper titles from references."""
        search_query = FeishuPaperIngestionService.build_local_search_text(query)
        papers = self._search_papers_via_api(search_query, requested_count)
        if len(papers) >= requested_count:
            return papers[:requested_count]

        existing_ids = {paper.arxiv_id for paper in papers}
        if self.feishu_settings.auto_ingest_enabled:
            self._send_progress_message(
                context,
                f"当前索引里只找到 {len(papers)} 篇相关论文，我正在从 arXiv 补充并建立索引，请稍等片刻。",
            )
            try:
                ingestion_result = asyncio.run(
                    self.paper_ingestion_service.ingest_missing_papers(
                        query=query,
                        requested_count=requested_count,
                        existing_ids=existing_ids,
                    )
                )
                logger.info(
                    "Feishu auto-ingest finished: fetched=%s, stored=%s, indexed=%s, chunks=%s",
                    ingestion_result.papers_fetched,
                    ingestion_result.papers_stored,
                    ingestion_result.papers_indexed,
                    ingestion_result.chunks_indexed,
                )
                papers = self._search_papers_via_api(ingestion_result.search_text, requested_count)
                existing_ids = {paper.arxiv_id for paper in papers}
                if len(papers) >= requested_count:
                    return papers[:requested_count]
            except Exception as exc:
                logger.error(f"Feishu auto-ingest failed for query '{query}': {exc}", exc_info=True)

        for paper in self._fetch_latest_indexed_papers(requested_count):
            if paper.arxiv_id in existing_ids:
                continue
            papers.append(paper)
            existing_ids.add(paper.arxiv_id)
            if len(papers) >= requested_count:
                break

        return papers[:requested_count]

    async def _find_recommended_papers_async(
        self, context: FeishuMessageContext, query: str, requested_count: int
    ) -> list[FeishuPaperReference]:
        """Async variant that avoids asyncio.run inside the local API route."""
        search_query = FeishuPaperIngestionService.build_local_search_text(query)
        papers = self._search_papers_via_api(search_query, requested_count)
        if len(papers) >= requested_count:
            return papers[:requested_count]

        existing_ids = {paper.arxiv_id for paper in papers}
        if self.feishu_settings.auto_ingest_enabled:
            self._send_progress_message(
                context,
                f"当前索引里只找到 {len(papers)} 篇相关论文，我正在从 arXiv 补充并建立索引，请稍等片刻。",
            )
            try:
                ingestion_result = await self.paper_ingestion_service.ingest_missing_papers(
                    query=query,
                    requested_count=requested_count,
                    existing_ids=existing_ids,
                )
                logger.info(
                    "Feishu auto-ingest finished: fetched=%s, stored=%s, indexed=%s, chunks=%s",
                    ingestion_result.papers_fetched,
                    ingestion_result.papers_stored,
                    ingestion_result.papers_indexed,
                    ingestion_result.chunks_indexed,
                )
                papers = self._search_papers_via_api(ingestion_result.search_text, requested_count)
                existing_ids = {paper.arxiv_id for paper in papers}
                if len(papers) >= requested_count:
                    return papers[:requested_count]
            except Exception as exc:
                logger.error(f"Feishu auto-ingest failed for query '{query}': {exc}", exc_info=True)

        for paper in self._fetch_latest_indexed_papers(requested_count):
            if paper.arxiv_id in existing_ids:
                continue
            papers.append(paper)
            existing_ids.add(paper.arxiv_id)
            if len(papers) >= requested_count:
                break

        return papers[:requested_count]

    def _send_progress_message(self, context: FeishuMessageContext, text: str) -> None:
        if context.receive_id_type == "local":
            return
        try:
            self._send_text_message(
                receive_id=context.receive_id,
                receive_id_type=context.receive_id_type,
                text=text,
            )
        except Exception as exc:
            logger.warning(f"Failed to send Feishu progress message for {context.message_id}: {exc}")

    def _search_papers_via_api(self, query: str, requested_count: int) -> list[FeishuPaperReference]:
        """Use BM25 search for paper recommendations because it is faster and easier to deduplicate."""
        if self._local_runtime is not None:
            opensearch_client = self._local_runtime.opensearch_client
            if not opensearch_client.health_check():
                return []
            results = opensearch_client.search_unified(
                query=query,
                query_embedding=None,
                size=max(requested_count * SEARCH_POOL_MULTIPLIER, 10),
                from_=0,
                categories=None,
                latest=False,
                use_hybrid=False,
                min_score=0.0,
            )
            return self._deduplicate_papers(results.get("hits", []), requested_count)

        payload = {
            "query": query,
            "size": max(requested_count * SEARCH_POOL_MULTIPLIER, 10),
            "use_hybrid": False,
            "latest_papers": False,
        }
        data = self._post_api_json(
            endpoint_path="/api/v1/hybrid-search/",
            payload=payload,
            timeout=min(self.feishu_settings.request_timeout_seconds, 60),
        )

        return self._deduplicate_papers(data.get("hits", []), requested_count)

    def _fetch_latest_indexed_papers(self, requested_count: int) -> list[FeishuPaperReference]:
        if self._local_runtime is not None:
            opensearch_client = self._local_runtime.opensearch_client
            if not opensearch_client.health_check():
                return []
            results = opensearch_client.search_unified(
                query="",
                query_embedding=None,
                size=max(requested_count * SEARCH_POOL_MULTIPLIER, LATEST_PAPER_POOL_MIN),
                from_=0,
                categories=None,
                latest=True,
                use_hybrid=False,
                min_score=0.0,
            )
            return self._deduplicate_papers(results.get("hits", []), requested_count)

        payload = {
            "size": max(requested_count * SEARCH_POOL_MULTIPLIER, LATEST_PAPER_POOL_MIN),
            "use_hybrid": False,
            "latest_papers": True,
        }

        data = self._post_api_json(
            endpoint_path="/api/v1/hybrid-search/",
            payload=payload,
            timeout=min(self.feishu_settings.request_timeout_seconds, 60),
        )

        hits = data.get("hits", [])
        return self._deduplicate_papers(hits, requested_count)

    def _deduplicate_papers(self, hits: list[dict[str, Any]], requested_count: int) -> list[FeishuPaperReference]:
        """Collapse chunk-level hits into distinct papers."""
        papers: list[FeishuPaperReference] = []
        seen_ids: set[str] = set()

        for hit in hits:
            arxiv_id = str(hit.get("arxiv_id", "")).strip()
            title = str(hit.get("title", "")).strip()
            if not arxiv_id or not title or arxiv_id in seen_ids:
                continue

            papers.append(
                FeishuPaperReference(
                    arxiv_id=arxiv_id,
                    title=title,
                    abstract=str(hit.get("abstract", "") or "").strip(),
                    published_date=str(hit.get("published_date", "") or "").strip(),
                )
            )
            seen_ids.add(arxiv_id)

            if len(papers) >= requested_count:
                break

        return papers

    def _format_paper_recommendations(self, query: str, papers: list[FeishuPaperReference]) -> str:
        """Return a paper list that is guaranteed to come from the current index."""
        lines = [f"我在当前已索引的论文库里找到了 {len(papers)} 篇相关论文：", ""]
        for index, paper in enumerate(papers, 1):
            summary = paper.abstract[:220].strip()
            if summary and len(paper.abstract) > 220:
                summary += "..."
            lines.append(f"{index}. {paper.title}")
            lines.append(f"   arXiv: {paper.arxiv_id}")
            if summary:
                lines.append(f"   摘要：{summary}")
            lines.append(f"   链接：https://arxiv.org/abs/{paper.arxiv_id.split('v')[0]}")
            lines.append("")

        lines.append("如果你愿意，我可以继续详细解释其中某一篇，或者对比这几篇论文的区别。")
        return "\n".join(lines).strip()

    def _is_paper_search_query(self, query: str) -> bool:
        lowered = query.lower()
        if ("论文" not in query) and ("paper" not in lowered) and ("papers" not in lowered):
            return False
        return any(hint in lowered or hint in query for hint in PAPER_SEARCH_HINTS)

    def _is_reference_query(self, query: str) -> bool:
        lowered = query.lower()
        return any(hint in lowered or hint in query for hint in REFERENCE_HINTS)

    def _extract_requested_paper_count(self, query: str) -> int:
        match = re.search(r"(\d+)\s*篇", query)
        if match:
            return max(1, min(int(match.group(1)), MAX_PAPER_RECOMMENDATION_COUNT))

        chinese_match = re.search(r"(一|两|二|三|四|五)\s*篇", query)
        if chinese_match:
            return CHINESE_COUNT_HINTS.get(chinese_match.group(0), CHINESE_COUNT_HINTS[chinese_match.group(1)])

        lowered = query.lower()
        match = re.search(r"(\d+)\s*(paper|papers)", lowered)
        if match:
            return max(1, min(int(match.group(1)), MAX_PAPER_RECOMMENDATION_COUNT))

        return DEFAULT_PAPER_RECOMMENDATION_COUNT

    def _resolve_referenced_papers(self, context: FeishuMessageContext, query: str) -> list[FeishuPaperReference]:
        return self._resolve_referenced_papers_from_list(self._load_recent_papers(context), query)

    def _resolve_referenced_papers_from_list(
        self,
        papers: list[FeishuPaperReference],
        query: str,
    ) -> list[FeishuPaperReference]:
        if not papers:
            return []
        lowered = query.lower()
        for hint, index in ORDINAL_HINTS.items():
            if hint in query or hint in lowered:
                return [papers[index]] if index < len(papers) else []

        if "这篇" in query and "这两篇" not in query and "这几篇" not in query:
            return papers[:1]

        if "这两篇" in query or "前两篇" in query or "those two papers" in lowered:
            return papers[:2]

        if "这几篇" in query or "these papers" in lowered or "those papers" in lowered:
            return papers

        return []

    def _rewrite_query_with_context(self, query: str, papers: list[FeishuPaperReference]) -> str:
        paper_descriptions = "; ".join(f"{paper.title} (arXiv:{paper.arxiv_id})" for paper in papers)
        rewritten = f"请基于以下论文回答用户问题：{paper_descriptions}。用户问题：{query}"
        return rewritten[:500]

    def _is_reset_query(self, query: str) -> bool:
        normalized = query.strip()
        return normalized in RESET_MEMORY_COMMANDS

    def _context_key(self, context: FeishuMessageContext) -> str:
        if context.chat_type == "group":
            return f"feishu:context:group:{context.chat_id}:{context.sender_open_id}"
        return f"feishu:context:p2p:{context.sender_open_id or context.chat_id}"

    def _store_recent_papers(self, context: FeishuMessageContext, papers: list[FeishuPaperReference]) -> None:
        previous_state = self._load_conversation_state(context)
        self._store_conversation_state(
            context,
            FeishuConversationState(
                recent_papers=papers,
                active_papers=papers,
                last_intent=INTENT_PAPER_SEARCH,
                last_query=context.query,
                summary=previous_state.summary,
                recent_messages=previous_state.recent_messages,
                persisted_message_count=previous_state.persisted_message_count,
            ),
        )

    def _store_conversation_state(self, context: FeishuMessageContext, state: FeishuConversationState) -> None:
        persisted_state = state
        if self.database:
            persisted_state = self._persist_conversation_state_to_database(context, state)

        self._write_cached_conversation_state(context, persisted_state)

    def _persist_conversation_state_to_database(
        self,
        context: FeishuMessageContext,
        state: FeishuConversationState,
    ) -> FeishuConversationState:
        try:
            with self.database.get_session() as session:
                repo = self._conversation_repository_factory(session)
                conversation = repo.get_or_create_session(
                    conversation_key=self._context_key(context),
                    chat_id=context.chat_id,
                    chat_type=context.chat_type,
                    sender_open_id=context.sender_open_id,
                    receive_id=context.receive_id,
                    receive_id_type=context.receive_id_type,
                )
                repo.update_session_state(
                    conversation,
                    recent_papers=self._serialize_papers(state.recent_papers),
                    active_papers=self._serialize_papers(state.active_papers),
                    last_intent=state.last_intent,
                    last_query=state.last_query,
                )
                message_count = repo.count_messages(conversation.id)
                if state.summary or message_count:
                    repo.upsert_summary(
                        session_id=conversation.id,
                        summary=state.summary,
                        source_message_count=message_count,
                    )
                return FeishuConversationState(
                    recent_papers=state.recent_papers,
                    active_papers=state.active_papers,
                    last_intent=state.last_intent,
                    last_query=state.last_query,
                    summary=state.summary,
                    recent_messages=state.recent_messages,
                    persisted_message_count=message_count or state.persisted_message_count,
                )
        except Exception as exc:
            logger.warning(f"Failed to persist Feishu conversation state to PostgreSQL: {exc}")
            return state

    def _write_cached_conversation_state(self, context: FeishuMessageContext, state: FeishuConversationState) -> None:
        payload = self._conversation_state_to_json(state)
        key = self._context_key(context)
        expires_at = time.time() + self.feishu_settings.context_ttl_seconds

        if self.redis:
            self.redis.set(key, payload, ex=self.feishu_settings.context_ttl_seconds)
            return

        with self._context_lock:
            self._memory_context[key] = (expires_at, payload)

    def _clear_recent_papers(self, context: FeishuMessageContext) -> None:
        key = self._context_key(context)

        if self.redis:
            self.redis.delete(key)

        with self._context_lock:
            self._memory_context.pop(key, None)

        if self.database:
            try:
                with self.database.get_session() as session:
                    repo = self._conversation_repository_factory(session)
                    repo.clear_session_memory(key)
            except Exception as exc:
                logger.warning(f"Failed to clear Feishu conversation memory from PostgreSQL: {exc}")

    def _load_recent_papers(self, context: FeishuMessageContext) -> list[FeishuPaperReference]:
        return self._load_conversation_state(context).recent_papers

    def _load_conversation_state(self, context: FeishuMessageContext) -> FeishuConversationState:
        key = self._context_key(context)
        payload: Optional[str] = None

        if self.redis:
            payload = self.redis.get(key)
        with self._context_lock:
            cached = self._memory_context.get(key)
            if not payload and cached:
                expires_at, raw_payload = cached
                if expires_at >= time.time():
                    payload = raw_payload
                else:
                    self._memory_context.pop(key, None)

        if payload:
            parsed_state = self._conversation_state_from_json(payload)
            if parsed_state:
                return parsed_state

        persisted_state = self._load_conversation_state_from_database(context)
        if persisted_state:
            self._write_cached_conversation_state(context, persisted_state)
            return persisted_state

        return FeishuConversationState(recent_papers=[], active_papers=[])

    def _load_conversation_state_from_database(
        self,
        context: FeishuMessageContext,
    ) -> Optional[FeishuConversationState]:
        if not self.database:
            return None

        try:
            with self.database.get_session() as session:
                repo = self._conversation_repository_factory(session)
                conversation = repo.get_session_by_key(self._context_key(context))
                if not conversation:
                    return None

                recent_messages = self._turns_from_stored_messages(
                    repo.get_recent_messages(conversation.id, RECENT_MESSAGE_LIMIT)
                )
                return FeishuConversationState(
                    recent_papers=self._deserialize_papers(conversation.recent_papers or []),
                    active_papers=self._deserialize_papers(conversation.active_papers or []),
                    last_intent=conversation.last_intent or "",
                    last_query=conversation.last_query or "",
                    summary=repo.get_summary(conversation.id),
                    recent_messages=recent_messages,
                    persisted_message_count=repo.count_messages(conversation.id),
                )
        except Exception as exc:
            logger.warning(f"Failed to load Feishu conversation memory from PostgreSQL: {exc}")
            return None

    def _conversation_state_to_json(self, state: FeishuConversationState) -> str:
        return json.dumps(
            {
                "recent_papers": self._serialize_papers(state.recent_papers),
                "active_papers": self._serialize_papers(state.active_papers),
                "last_intent": state.last_intent,
                "last_query": state.last_query,
                "summary": state.summary,
                "recent_messages": self._serialize_messages(state.recent_messages),
                "persisted_message_count": state.persisted_message_count,
            },
            ensure_ascii=False,
        )

    def _conversation_state_from_json(self, payload: str) -> Optional[FeishuConversationState]:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            logger.warning("Failed to decode stored Feishu conversation context")
            return None

        if isinstance(data, list):
            papers = self._deserialize_papers(data)
            return FeishuConversationState(recent_papers=papers, active_papers=papers)

        recent_papers = self._deserialize_papers(data.get("recent_papers", []))
        active_papers = self._deserialize_papers(data.get("active_papers", []))
        if not active_papers:
            active_papers = recent_papers

        return FeishuConversationState(
            recent_papers=recent_papers,
            active_papers=active_papers,
            last_intent=str(data.get("last_intent", "") or ""),
            last_query=str(data.get("last_query", "") or ""),
            summary=str(data.get("summary", "") or ""),
            recent_messages=self._deserialize_messages(data.get("recent_messages", [])),
            persisted_message_count=int(data.get("persisted_message_count") or 0),
        )

    @staticmethod
    def _serialize_papers(papers: list[FeishuPaperReference]) -> list[dict[str, str]]:
        return [paper.__dict__ for paper in papers]

    @staticmethod
    def _deserialize_papers(items: list[dict[str, Any]]) -> list[FeishuPaperReference]:
        papers: list[FeishuPaperReference] = []
        for item in items:
            arxiv_id = str(item.get("arxiv_id", "")).strip()
            title = str(item.get("title", "")).strip()
            if not arxiv_id or not title:
                continue
            papers.append(
                FeishuPaperReference(
                    arxiv_id=arxiv_id,
                    title=title,
                    abstract=str(item.get("abstract", "") or "").strip(),
                    published_date=str(item.get("published_date", "") or "").strip(),
                )
            )
        return papers

    @staticmethod
    def _serialize_messages(messages: list[FeishuConversationTurn]) -> list[dict[str, str]]:
        return [message.__dict__ for message in messages]

    @staticmethod
    def _deserialize_messages(items: list[dict[str, Any]]) -> list[FeishuConversationTurn]:
        messages: list[FeishuConversationTurn] = []
        for item in items:
            content = str(item.get("content", "") or "").strip()
            role = str(item.get("role", "") or "").strip()
            if not role or not content:
                continue
            messages.append(
                FeishuConversationTurn(
                    role=role,
                    content=content,
                    intent=str(item.get("intent", "") or ""),
                    message_id=str(item.get("message_id", "") or ""),
                    created_at=str(item.get("created_at", "") or ""),
                )
            )
        return messages

    @staticmethod
    def _turns_from_stored_messages(messages: list[StoredConversationMessage]) -> list[FeishuConversationTurn]:
        return [
            FeishuConversationTurn(
                role=message.role,
                content=message.content,
                intent=message.intent,
                message_id=message.message_id,
                created_at=message.created_at,
            )
            for message in messages
        ]

    def _format_recent_messages_for_prompt(self, messages: list[FeishuConversationTurn]) -> str:
        lines: list[str] = []
        for message in messages:
            label = "用户" if message.role == "user" else "助手"
            lines.append(f"{label}: {self._truncate(message.content, 180)}")
        return "\n".join(lines)

    def _build_conversation_summary(
        self,
        previous_summary: str,
        recent_messages: list[FeishuConversationTurn],
    ) -> str:
        lines: list[str] = []
        if previous_summary:
            lines.append(previous_summary.strip())
        for message in recent_messages[-6:]:
            label = "用户" if message.role == "user" else "助手"
            lines.append(f"{label}: {self._truncate(message.content, 220)}")

        summary = "\n".join(line for line in lines if line).strip()
        return self._truncate_from_left(summary, SUMMARY_MAX_CHARS)

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."

    @staticmethod
    def _truncate_from_left(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return "..." + text[-(max_chars - 3) :].lstrip()

    def _send_reply_text(self, receive_id: str, receive_id_type: str, text: str) -> None:
        chunks = self._split_text_message(text)
        for chunk in chunks:
            self._send_text_message(
                receive_id=receive_id,
                receive_id_type=receive_id_type,
                text=chunk,
            )

    def _split_text_message(self, text: str) -> list[str]:
        if len(text) <= self._message_chunk_chars:
            return [text]

        chunks: list[str] = []
        current = ""
        for line in text.splitlines(keepends=True):
            if len(line) > self._message_chunk_chars:
                if current:
                    chunks.append(current.rstrip())
                    current = ""
                chunks.extend(
                    line[index : index + self._message_chunk_chars].rstrip()
                    for index in range(0, len(line), self._message_chunk_chars)
                )
                continue

            if current and len(current) + len(line) > self._message_chunk_chars:
                chunks.append(current.rstrip())
                current = line
            else:
                current += line

        if current:
            chunks.append(current.rstrip())

        if len(chunks) <= 1:
            return chunks

        return [f"({index}/{len(chunks)})\n{chunk}" for index, chunk in enumerate(chunks, 1)]

    def _send_text_message(self, receive_id: str, receive_id_type: str, text: str) -> None:
        """Send a plain text reply back to Feishu."""
        if self.client is None:
            raise RuntimeError("Feishu client is not configured")

        request: CreateMessageRequest = (
            CreateMessageRequest.builder()
            .receive_id_type(receive_id_type)
            .request_body(
                CreateMessageRequestBody.builder()
                .receive_id(receive_id)
                .msg_type("text")
                .content(json.dumps({"text": text}, ensure_ascii=False))
                .uuid(uuid4().hex)
                .build()
            )
            .build()
        )

        with self._send_lock:
            response: CreateMessageResponse = self.client.im.v1.message.create(request)

        if response.success():
            logger.info(f"Sent Feishu reply to {receive_id_type}={receive_id}")
            return

        raise RuntimeError(
            f"client.im.v1.message.create failed, code={response.code}, msg={response.msg}, log_id={response.get_log_id()}"
        )

    def _mark_message_processing(self, message_id: str) -> bool:
        """Mark a message as in-flight so duplicate pushed events are skipped."""
        if not message_id:
            return True

        key = f"feishu:processed:{message_id}"
        if self.redis:
            try:
                return bool(
                    self.redis.set(
                        key,
                        "processing",
                        ex=self.feishu_settings.dedupe_ttl_seconds,
                        nx=True,
                    )
                )
            except Exception as exc:
                logger.warning(f"Failed to mark Feishu message in Redis, using local dedupe: {exc}")

        return self._mark_local_message_processing(key)

    def _mark_message_done(self, message_id: str) -> None:
        if not message_id:
            return

        key = f"feishu:processed:{message_id}"
        if self.redis:
            try:
                self.redis.set(key, "done", ex=self.feishu_settings.dedupe_ttl_seconds)
                return
            except Exception as exc:
                logger.warning(f"Failed to mark Feishu message done in Redis, using local dedupe: {exc}")

        self._set_local_dedupe_state(key, "done")

    def _mark_message_failed(self, message_id: str) -> None:
        if not message_id:
            return

        key = f"feishu:processed:{message_id}"
        if self.redis:
            try:
                self.redis.delete(key)
                return
            except Exception as exc:
                logger.warning(f"Failed to clear failed Feishu message from Redis, using local dedupe: {exc}")

        with self._dedupe_lock:
            self._local_dedupe.pop(key, None)

    def _mark_local_message_processing(self, key: str) -> bool:
        now = time.time()
        expires_at = now + self.feishu_settings.dedupe_ttl_seconds
        with self._dedupe_lock:
            self._prune_local_dedupe(now)
            existing = self._local_dedupe.get(key)
            if existing and existing[0] >= now:
                return False
            self._local_dedupe[key] = (expires_at, "processing")
            return True

    def _set_local_dedupe_state(self, key: str, state: str) -> None:
        expires_at = time.time() + self.feishu_settings.dedupe_ttl_seconds
        with self._dedupe_lock:
            self._local_dedupe[key] = (expires_at, state)

    def _prune_local_dedupe(self, now: Optional[float] = None) -> None:
        now = now or time.time()
        expired = [key for key, (expires_at, _) in self._local_dedupe.items() if expires_at < now]
        for key in expired:
            self._local_dedupe.pop(key, None)

    def _extract_message_context(self, event_data: Any) -> Optional[FeishuMessageContext]:
        """Normalize SDK event data into the minimum payload the worker needs."""
        event = self._to_event_dict(event_data)

        message = event.get("event", {}).get("message", {})
        sender = event.get("event", {}).get("sender", {})
        sender_id = sender.get("sender_id", {})

        if message.get("message_type") != "text":
            logger.info("Ignoring non-text Feishu message")
            return None

        chat_type = message.get("chat_type", "")
        mentions = message.get("mentions") or []
        if chat_type == "group":
            if not mentions:
                logger.info("Ignoring Feishu group message without mentions")
                return None
            if not self._mentions_configured_bot(mentions):
                logger.info("Ignoring Feishu group message that did not mention this bot")
                return None

        query = self._extract_query_text(message.get("content"))
        if not query:
            logger.info("Ignoring empty Feishu text message")
            return None

        receive_id_type = "chat_id" if chat_type == "group" else "open_id"
        receive_id = message.get("chat_id") if chat_type == "group" else sender_id.get("open_id", "")
        if not receive_id:
            logger.warning("Failed to resolve Feishu reply target")
            return None

        return FeishuMessageContext(
            message_id=message.get("message_id", ""),
            chat_id=message.get("chat_id", ""),
            chat_type=chat_type,
            sender_open_id=sender_id.get("open_id", ""),
            receive_id=receive_id,
            receive_id_type=receive_id_type,
            query=query,
        )

    def _mentions_configured_bot(self, mentions: list[Any]) -> bool:
        configured_ids = {
            value
            for value in (
                str(getattr(self.feishu_settings, "bot_open_id", "") or "").strip(),
            )
            if value
        }
        configured_names = {
            value.lower()
            for value in (
                str(getattr(self.feishu_settings, "bot_name", "") or "").strip(),
            )
            if value
        }

        if not configured_ids and not configured_names:
            return True

        for mention in mentions:
            if not isinstance(mention, dict):
                continue
            mention_name = str(mention.get("name", "") or "").strip().lower()
            if mention_name and mention_name in configured_names:
                return True
            if configured_ids & self._flatten_mention_values(mention):
                return True

        return False

    def _flatten_mention_values(self, value: Any) -> set[str]:
        if isinstance(value, dict):
            values: set[str] = set()
            for item in value.values():
                values.update(self._flatten_mention_values(item))
            return values
        if isinstance(value, list):
            values = set()
            for item in value:
                values.update(self._flatten_mention_values(item))
            return values
        if value is None:
            return set()
        text = str(value).strip()
        return {text} if text else set()

    def _to_event_dict(self, event_data: Any) -> dict[str, Any]:
        """Convert SDK event models to plain dictionaries."""
        if isinstance(event_data, dict):
            return event_data

        raw_payload = lark.JSON.marshal(event_data)
        return json.loads(raw_payload)

    def _extract_query_text(self, raw_content: Any) -> str:
        """Parse Feishu text payload and remove mention placeholders."""
        if isinstance(raw_content, str):
            try:
                parsed_content = json.loads(raw_content)
            except json.JSONDecodeError:
                logger.warning("Failed to decode Feishu message content as JSON")
                return ""
        elif isinstance(raw_content, dict):
            parsed_content = raw_content
        else:
            return ""

        text = parsed_content.get("text", "")
        text = MENTION_PLACEHOLDER_RE.sub("", text)
        return " ".join(text.split())

    def _normalize_sources(self, sources: list[str]) -> list[str]:
        """Convert PDF source links to more readable arXiv abstract links."""
        normalized: list[str] = []
        for source in sources[: self.feishu_settings.source_limit]:
            if source.endswith(".pdf"):
                normalized.append(source.replace("/pdf/", "/abs/").replace(".pdf", ""))
            else:
                normalized.append(source)
        return normalized
