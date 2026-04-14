import asyncio
import json
import logging
import re
import threading
import time
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
RESET_MEMORY_COMMANDS = ("新对话", "新消息")
INTENT_RESET = "reset"
INTENT_PAPER_SEARCH = "paper_search"
INTENT_PAPER_SUMMARY = "paper_summary"
INTENT_PAPER_DEEP_DIVE = "paper_deep_dive"
INTENT_PAPER_COMPARE = "paper_compare"
INTENT_GENERAL_RAG = "general_rag"
PAPER_RELATED_INTENTS = {
    INTENT_PAPER_SEARCH,
    INTENT_PAPER_SUMMARY,
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
class FeishuIntentDecision:
    """Routing result for a single Feishu user message."""

    intent: str
    papers: list[FeishuPaperReference]


class FeishuBot:
    """Minimal Feishu bot powered by the existing RAG API."""

    def __init__(
        self,
        settings: Settings,
        redis_client: Optional[redis.Redis] = None,
        database: Optional[Any] = None,
        conversation_repository_factory: Callable[[Any], FeishuConversationRepository] = FeishuConversationRepository,
    ):
        self.settings = settings
        self.feishu_settings = settings.feishu
        self.redis = redis_client
        self.database = database
        self._conversation_repository_factory = conversation_repository_factory
        self.client = (
            lark.Client.builder()
            .app_id(self.feishu_settings.app_id)
            .app_secret(self.feishu_settings.app_secret)
            .log_level(lark.LogLevel.INFO)
            .build()
        )
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

            worker = threading.Thread(target=self._process_message, args=(context,), daemon=True)
            worker.start()
        except Exception as exc:
            logger.error(f"Failed to dispatch Feishu event: {exc}", exc_info=True)

    def _process_message(self, context: FeishuMessageContext) -> None:
        """Call the existing ask API and send the answer back to Feishu."""
        try:
            answer_text = self._build_reply(context)
        except Exception as exc:
            logger.error(f"Failed to get RAG answer for Feishu message {context.message_id}: {exc}", exc_info=True)
            answer_text = "暂时无法获取回答，请稍后重试。"

        try:
            self._send_text_message(
                receive_id=context.receive_id,
                receive_id_type=context.receive_id_type,
                text=answer_text,
            )
        except Exception as exc:
            logger.error(f"Failed to send Feishu reply for {context.message_id}: {exc}", exc_info=True)

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

        if decision.intent in {INTENT_PAPER_SUMMARY, INTENT_PAPER_DEEP_DIVE, INTENT_PAPER_COMPARE}:
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
        reply = self._ask_rag_api(query=rag_query)
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

    def _ask_rag_api(
        self,
        query: str,
        arxiv_ids: Optional[list[str]] = None,
        force_standard: bool = False,
        direct_chunks_per_paper: Optional[int] = None,
    ) -> str:
        """Call the configured ask endpoint so the bot reuses the current RAG stack."""
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

        endpoint_path = "/api/v1/ask" if force_standard else (self.feishu_settings.ask_endpoint_path.strip() or "/api/v1/ask-agentic")
        response = requests.post(
            f"{self.feishu_settings.api_base_url.rstrip('/')}{endpoint_path}",
            json=payload,
            timeout=self.feishu_settings.request_timeout_seconds,
        )
        response.raise_for_status()

        data = response.json()
        answer = data.get("answer", "").strip() or "暂时没有生成有效回答。"
        answer = compact_repeated_single_paper_citations(answer)
        sources = self._normalize_sources(data.get("sources", []))

        if not sources:
            return answer

        source_lines = "\n".join(f"{idx}. {url}" for idx, url in enumerate(sources, 1))
        return f"{answer}\n\n参考来源：\n{source_lines}"

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
                conversation = repo.get_or_create_session(
                    conversation_key=self._context_key(context),
                    chat_id=context.chat_id,
                    chat_type=context.chat_type,
                    sender_open_id=context.sender_open_id,
                    receive_id=context.receive_id,
                    receive_id_type=context.receive_id_type,
                )
                repo.append_message(
                    session_id=conversation.id,
                    role="user",
                    content=user_query,
                    intent=intent,
                    message_id=context.message_id,
                    metadata={"chat_type": context.chat_type},
                )
                repo.append_message(
                    session_id=conversation.id,
                    role="assistant",
                    content=assistant_reply,
                    intent=intent,
                    metadata={"receive_id_type": context.receive_id_type},
                )
                message_count = repo.count_messages(conversation.id)
                persisted_recent_messages = self._turns_from_stored_messages(
                    repo.get_recent_messages(conversation.id, RECENT_MESSAGE_LIMIT)
                )
                repo.update_session_state(
                    conversation,
                    recent_papers=self._serialize_papers(state.recent_papers),
                    active_papers=self._serialize_papers(state.active_papers),
                    last_intent=state.last_intent,
                    last_query=state.last_query,
                )
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
            if self._is_deep_dive_query(query):
                return FeishuIntentDecision(intent=INTENT_PAPER_DEEP_DIVE, papers=referenced_papers)
            return FeishuIntentDecision(intent=INTENT_PAPER_SUMMARY, papers=referenced_papers)

        if self._is_reference_query(query):
            if self._is_compare_query(query):
                return FeishuIntentDecision(intent=INTENT_PAPER_COMPARE, papers=[])
            if self._is_deep_dive_query(query):
                return FeishuIntentDecision(intent=INTENT_PAPER_DEEP_DIVE, papers=[])
            return FeishuIntentDecision(intent=INTENT_PAPER_SUMMARY, papers=[])

        return FeishuIntentDecision(intent=INTENT_GENERAL_RAG, papers=[])

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
        return any(hint in query or hint in lowered for hint in COMPARE_HINTS)

    @staticmethod
    def _is_deep_dive_query(query: str) -> bool:
        lowered = query.lower()
        return any(hint in query or hint in lowered for hint in FOLLOWUP_HINTS + DETAIL_HINTS)

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
            "不要用“背景/方法/结果”各一句的压缩格式糊弄用户。\n"
            f"目标论文：\n{paper_lines}\n"
            f"用户原始问题：{query}"
        )

    @staticmethod
    def _direct_chunks_per_paper(intent: str, paper_count: int) -> int:
        if intent == INTENT_PAPER_COMPARE:
            return 18
        if intent == INTENT_PAPER_DEEP_DIVE:
            return 30 if paper_count == 1 else 22
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

    def _send_progress_message(self, context: FeishuMessageContext, text: str) -> None:
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
        payload = {
            "query": query,
            "size": max(requested_count * SEARCH_POOL_MULTIPLIER, 10),
            "use_hybrid": False,
            "latest_papers": False,
        }
        response = requests.post(
            f"{self.feishu_settings.api_base_url.rstrip('/')}/api/v1/hybrid-search/",
            json=payload,
            timeout=min(self.feishu_settings.request_timeout_seconds, 60),
        )
        response.raise_for_status()

        data = response.json()
        return self._deduplicate_papers(data.get("hits", []), requested_count)

    def _fetch_latest_indexed_papers(self, requested_count: int) -> list[FeishuPaperReference]:
        payload = {
            "size": max(requested_count * SEARCH_POOL_MULTIPLIER, LATEST_PAPER_POOL_MIN),
            "use_hybrid": False,
            "latest_papers": True,
        }

        response = requests.post(
            f"{self.feishu_settings.api_base_url.rstrip('/')}/api/v1/hybrid-search/",
            json=payload,
            timeout=min(self.feishu_settings.request_timeout_seconds, 60),
        )
        response.raise_for_status()

        hits = response.json().get("hits", [])
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

        for hint, count in CHINESE_COUNT_HINTS.items():
            if hint in query:
                return count

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

        return papers[: min(2, len(papers))]

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

    def _send_text_message(self, receive_id: str, receive_id_type: str, text: str) -> None:
        """Send a plain text reply back to Feishu."""
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
        """Use Redis to avoid processing duplicate pushed events."""
        if not message_id or not self.redis:
            return True

        key = f"feishu:processed:{message_id}"
        return bool(
            self.redis.set(
                key,
                "1",
                ex=self.feishu_settings.dedupe_ttl_seconds,
                nx=True,
            )
        )

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
        if chat_type == "group" and not mentions:
            logger.info("Ignoring Feishu group message without mentions")
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
