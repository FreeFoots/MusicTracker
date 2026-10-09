import uuid
from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from .schemas import now


def uid() -> str:
    return uuid.uuid4().hex


class Base(DeclarativeBase):
    pass


class Query(Base):
    __tablename__ = "queries"
    __table_args__ = (UniqueConstraint("idempotency_key"),)
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uid)
    input: Mapped[str] = mapped_column(Text)
    request: Mapped[dict] = mapped_column(JSON)
    request_hash: Mapped[str] = mapped_column(String(64))
    idempotency_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    parent_id: Mapped[str | None] = mapped_column(ForeignKey("queries.id"), nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="queued", index=True)
    persistence_status: Mapped[str] = mapped_column(String(20), default="pending")
    plan: Mapped[dict] = mapped_column(JSON, default=dict)
    checkpoint: Mapped[dict] = mapped_column(JSON, default=dict)
    report: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    lease_owner: Mapped[str | None] = mapped_column(String(32), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    conversation_id: Mapped[str | None] = mapped_column(ForeignKey("conversations.id"), nullable=True, index=True)


class Conversation(Base):
    __tablename__ = "conversations"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uid)
    title: Mapped[str] = mapped_column(String(100), default="音乐研究会话")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class AgentStep(Base):
    __tablename__ = "agent_steps"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query_id: Mapped[str] = mapped_column(ForeignKey("queries.id"), index=True)
    kind: Mapped[str] = mapped_column(String(30))
    payload: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Event(Base):
    __tablename__ = "events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query_id: Mapped[str] = mapped_column(ForeignKey("queries.id"), index=True)
    type: Mapped[str] = mapped_column(String(50))
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class ToolCall(Base):
    __tablename__ = "tool_calls"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uid)
    query_id: Mapped[str] = mapped_column(ForeignKey("queries.id"), index=True)
    tool: Mapped[str] = mapped_column(String(80))
    arguments: Mapped[dict] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(30))
    error_code: Mapped[str | None] = mapped_column(String(80), nullable=True)
    duration_ms: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Entity(Base):
    __tablename__ = "entities"
    __table_args__ = (UniqueConstraint("platform", "platform_id"),)
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uid)
    platform: Mapped[str] = mapped_column(String(20))
    platform_id: Mapped[str] = mapped_column(String(40))
    title: Mapped[str] = mapped_column(Text)
    artists: Mapped[list] = mapped_column(JSON)
    metadata_json: Mapped[dict] = mapped_column(JSON)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class QueryResult(Base):
    __tablename__ = "query_results"
    __table_args__ = (UniqueConstraint("query_id", "entity_id"),)
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uid)
    query_id: Mapped[str] = mapped_column(ForeignKey("queries.id"), index=True)
    entity_id: Mapped[str] = mapped_column(ForeignKey("entities.id"), index=True)
    snapshot: Mapped[dict] = mapped_column(JSON)
    origin: Mapped[str] = mapped_column(String(20), default="live")
    reused_from_query_id: Mapped[str | None] = mapped_column(ForeignKey("queries.id"), nullable=True)


class Observation(Base):
    __tablename__ = "observations"
    __table_args__ = (UniqueConstraint("query_id", "entity_id", "metric"),)
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uid)
    query_id: Mapped[str] = mapped_column(ForeignKey("queries.id"), index=True)
    entity_id: Mapped[str] = mapped_column(ForeignKey("entities.id"), index=True)
    attempt_id: Mapped[str | None] = mapped_column(ForeignKey("tool_calls.id"), nullable=True)
    metric: Mapped[str] = mapped_column(String(50))
    value: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(20))
    data: Mapped[dict] = mapped_column(JSON)
    collected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class ReportVersion(Base):
    __tablename__ = "report_versions"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uid)
    query_id: Mapped[str] = mapped_column(ForeignKey("queries.id"), index=True)
    report: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
