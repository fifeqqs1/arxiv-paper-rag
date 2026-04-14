from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session
from src.models.conversation import (
    FeishuConversationMessage,
    FeishuConversationSession,
    FeishuConversationSummary,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class StoredConversationMessage:
    role: str
    content: str
    intent: str = ""
    message_id: str = ""
    created_at: str = ""


class FeishuConversationRepository:
    """Persistent storage for Feishu multi-turn conversation memory."""

    def __init__(self, session: Session):
        self.session = session

    def get_or_create_session(
        self,
        *,
        conversation_key: str,
        chat_id: str,
        chat_type: str,
        sender_open_id: str,
        receive_id: str,
        receive_id_type: str,
    ) -> FeishuConversationSession:
        stmt = select(FeishuConversationSession).where(
            FeishuConversationSession.conversation_key == conversation_key
        )
        conversation = self.session.scalar(stmt)
        now = utc_now()

        if conversation:
            conversation.chat_id = chat_id
            conversation.chat_type = chat_type
            conversation.sender_open_id = sender_open_id
            conversation.receive_id = receive_id
            conversation.receive_id_type = receive_id_type
            conversation.last_seen_at = now
            conversation.updated_at = now
            self.session.add(conversation)
            self.session.commit()
            self.session.refresh(conversation)
            return conversation

        conversation = FeishuConversationSession(
            conversation_key=conversation_key,
            chat_id=chat_id,
            chat_type=chat_type,
            sender_open_id=sender_open_id,
            receive_id=receive_id,
            receive_id_type=receive_id_type,
            recent_papers=[],
            active_papers=[],
            created_at=now,
            updated_at=now,
            last_seen_at=now,
        )
        self.session.add(conversation)
        self.session.commit()
        self.session.refresh(conversation)
        return conversation

    def get_session_by_key(self, conversation_key: str) -> Optional[FeishuConversationSession]:
        stmt = select(FeishuConversationSession).where(
            FeishuConversationSession.conversation_key == conversation_key
        )
        return self.session.scalar(stmt)

    def update_session_state(
        self,
        conversation: FeishuConversationSession,
        *,
        recent_papers: list[dict[str, Any]],
        active_papers: list[dict[str, Any]],
        last_intent: str,
        last_query: str,
    ) -> FeishuConversationSession:
        conversation.recent_papers = recent_papers
        conversation.active_papers = active_papers
        conversation.last_intent = last_intent
        conversation.last_query = last_query
        conversation.last_seen_at = utc_now()
        conversation.updated_at = utc_now()
        self.session.add(conversation)
        self.session.commit()
        self.session.refresh(conversation)
        return conversation

    def append_message(
        self,
        *,
        session_id: UUID,
        role: str,
        content: str,
        intent: str = "",
        message_id: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> FeishuConversationMessage:
        message = FeishuConversationMessage(
            session_id=session_id,
            role=role,
            content=content,
            intent=intent,
            message_id=message_id or None,
            message_metadata=metadata or {},
        )
        self.session.add(message)
        self.session.commit()
        self.session.refresh(message)
        return message

    def get_recent_messages(self, session_id: UUID, limit: int = 12) -> list[StoredConversationMessage]:
        stmt = (
            select(FeishuConversationMessage)
            .where(FeishuConversationMessage.session_id == session_id)
            .order_by(FeishuConversationMessage.created_at.desc())
            .limit(limit)
        )
        messages = list(self.session.scalars(stmt))
        messages.reverse()
        return [
            StoredConversationMessage(
                role=message.role,
                content=message.content,
                intent=message.intent or "",
                message_id=message.message_id or "",
                created_at=message.created_at.isoformat() if message.created_at else "",
            )
            for message in messages
        ]

    def count_messages(self, session_id: UUID) -> int:
        stmt = select(func.count(FeishuConversationMessage.id)).where(
            FeishuConversationMessage.session_id == session_id
        )
        return self.session.scalar(stmt) or 0

    def prune_messages(self, session_id: UUID, keep_latest: int) -> int:
        """Keep only the latest N raw messages for a conversation."""
        if keep_latest <= 0:
            stmt = delete(FeishuConversationMessage).where(
                FeishuConversationMessage.session_id == session_id
            )
            result = self.session.execute(stmt)
            self.session.commit()
            return result.rowcount or 0

        stmt = (
            select(FeishuConversationMessage.id)
            .where(FeishuConversationMessage.session_id == session_id)
            .order_by(FeishuConversationMessage.created_at.desc())
            .offset(keep_latest)
        )
        message_ids = list(self.session.scalars(stmt))
        if not message_ids:
            return 0

        result = self.session.execute(
            delete(FeishuConversationMessage).where(
                FeishuConversationMessage.id.in_(message_ids)
            )
        )
        self.session.commit()
        return result.rowcount or 0

    def get_summary(self, session_id: UUID) -> str:
        stmt = select(FeishuConversationSummary).where(
            FeishuConversationSummary.session_id == session_id
        )
        summary = self.session.scalar(stmt)
        return summary.summary if summary else ""

    def upsert_summary(self, *, session_id: UUID, summary: str, source_message_count: int) -> None:
        stmt = select(FeishuConversationSummary).where(
            FeishuConversationSummary.session_id == session_id
        )
        row = self.session.scalar(stmt)
        now = utc_now()

        if row:
            row.summary = summary
            row.source_message_count = source_message_count
            row.updated_at = now
        else:
            row = FeishuConversationSummary(
                session_id=session_id,
                summary=summary,
                source_message_count=source_message_count,
                created_at=now,
                updated_at=now,
            )

        self.session.add(row)
        self.session.commit()

    def clear_session_memory(self, conversation_key: str) -> None:
        conversation = self.get_session_by_key(conversation_key)
        if not conversation:
            return

        self.session.execute(
            delete(FeishuConversationMessage).where(
                FeishuConversationMessage.session_id == conversation.id
            )
        )
        self.session.execute(
            delete(FeishuConversationSummary).where(
                FeishuConversationSummary.session_id == conversation.id
            )
        )
        conversation.recent_papers = []
        conversation.active_papers = []
        conversation.last_intent = ""
        conversation.last_query = ""
        conversation.updated_at = utc_now()
        conversation.last_seen_at = utc_now()
        self.session.add(conversation)
        self.session.commit()

    def clear_all_memory(self) -> None:
        """Clear all Feishu conversation messages, summaries, and active state."""
        self.session.execute(delete(FeishuConversationMessage))
        self.session.execute(delete(FeishuConversationSummary))

        now = utc_now()
        conversations = list(self.session.scalars(select(FeishuConversationSession)))
        for conversation in conversations:
            conversation.recent_papers = []
            conversation.active_papers = []
            conversation.last_intent = ""
            conversation.last_query = ""
            conversation.updated_at = now
            conversation.last_seen_at = now
            self.session.add(conversation)

        self.session.commit()
