import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from src.db.interfaces.postgresql import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class FeishuConversationSession(Base):
    __tablename__ = "feishu_conversation_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_key = Column(String(255), unique=True, nullable=False, index=True)
    chat_id = Column(String(255), nullable=False, index=True)
    chat_type = Column(String(32), nullable=False)
    sender_open_id = Column(String(255), nullable=False, index=True)
    receive_id = Column(String(255), nullable=False)
    receive_id_type = Column(String(32), nullable=False)

    recent_papers = Column(JSONB, nullable=False, default=list)
    active_papers = Column(JSONB, nullable=False, default=list)
    last_intent = Column(String(64), nullable=False, default="")
    last_query = Column(Text, nullable=False, default="")

    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)
    last_seen_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)


class FeishuConversationMessage(Base):
    __tablename__ = "feishu_conversation_messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("feishu_conversation_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role = Column(String(32), nullable=False)
    content = Column(Text, nullable=False)
    intent = Column(String(64), nullable=False, default="")
    message_id = Column(String(255), nullable=True, index=True)
    message_metadata = Column("metadata", JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False, index=True)

    __table_args__ = (
        Index("ix_feishu_conversation_messages_session_created", "session_id", "created_at"),
    )


class FeishuConversationSummary(Base):
    __tablename__ = "feishu_conversation_summaries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("feishu_conversation_sessions.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
        index=True,
    )
    summary = Column(Text, nullable=False, default="")
    source_message_count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)
