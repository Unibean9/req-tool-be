import enum
import uuid
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint, text
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from app.models.base import AuditMixin, Base


class AgentSessionStatus(enum.StrEnum):
    ACTIVE = "active"
    WAITING_FOR_HUMAN = "waiting_for_human"
    COMPLETED = "completed"
    FAILED = "failed"
    TURN_FAILED = "turn_failed"
    EXPIRED = "expired"


class AgentSessionInterruptType(enum.StrEnum):
    ASK_HUMAN = "ask_human"
    PROPOSE_ARTIFACTS = "propose_artifacts"
    STREAM_RESPONSE = "stream_response"


class AgentMessageRole(enum.StrEnum):
    AGENT = "agent"
    USER = "user"


class AgentToolCallStatus(enum.StrEnum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"
    SUPERSEDED = "superseded"


class AgentTurnTriggerType(enum.StrEnum):
    USER_MESSAGE = "user_message"
    APPROVAL = "approval"
    CANCEL = "cancel"
    RETRY = "retry"


class TurnExecutionStatus(enum.StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING = "waiting"
    TERMINAL = "terminal"


def enum_column(enum_class: type[enum.Enum], **kwargs):
    return mapped_column(
        SAEnum(
            enum_class,
            values_callable=lambda values: [item.value for item in values],
            validate_strings=True,
        ),
        **kwargs,
    )


def jsonb_column(*args, **kwargs):
    return mapped_column(*args, JSON().with_variant(postgresql.JSONB, "postgresql"), **kwargs)


class AgentSession(AuditMixin, Base):
    __tablename__ = "agent_sessions"
    __table_args__ = (
        Index("ix_agent_sessions_project_id", "project_id"),
        Index("ix_agent_sessions_artifact_type", "artifact_type"),
        Index("ix_agent_sessions_workflow_area", "workflow_area"),
        Index("ix_agent_sessions_status", "status"),
        Index("ix_agent_sessions_provider_config_id", "provider_config_id"),
        Index("ix_agent_sessions_created_by_id", "created_by_id"),
        Index(
            "uq_agent_sessions_active_project_artifact_type_user",
            "project_id",
            "artifact_type",
            "created_by_id",
            unique=True,
            postgresql_where=text("status IN ('active', 'waiting_for_human')"),
            sqlite_where=text("status IN ('active', 'waiting_for_human')"),
        ),
    )

    project_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False)
    artifact_type: Mapped[str] = mapped_column(String(100), nullable=False)
    workflow_area: Mapped[str] = mapped_column(String(50), nullable=False)
    step_key: Mapped[str | None] = mapped_column(String(100), nullable=True)
    status: Mapped[AgentSessionStatus] = enum_column(
        AgentSessionStatus, nullable=False, default=AgentSessionStatus.ACTIVE
    )
    interrupt_type: Mapped[AgentSessionInterruptType | None] = enum_column(AgentSessionInterruptType, nullable=True)
    missing_context: Mapped[Any | None] = jsonb_column(nullable=True)
    graph_checkpoint: Mapped[Any] = jsonb_column(nullable=False, default=dict)
    focused_artifact_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("artifacts.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    agent_role: Mapped[str | None] = mapped_column(String(100), nullable=True)
    provider_config_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("llm_provider_configs.id"),
        nullable=True,
    )
    created_by_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    # Cursor additive cho control plane. Chỉ admission service được tăng giá trị này dưới row lock.
    turn_sequence: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    # Chỉ turn này được phép chiếm checkpoint của session. Admission/drain thay đổi nó dưới row lock.
    active_turn_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True, index=True)

    messages: Mapped[list["AgentMessage"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
    )
    runs: Mapped[list["AgentRun"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
    )
    turns: Mapped[list["AgentTurnEnvelope"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        foreign_keys="AgentTurnEnvelope.session_id",
    )


class AgentTurnEnvelope(AuditMixin, Base):
    """Biên logical turn bất biến; state thực thi nằm ở bảng tách riêng."""

    __tablename__ = "agent_turn_envelopes"
    __table_args__ = (
        UniqueConstraint("session_id", "session_sequence", name="uq_agent_turn_envelopes_session_sequence"),
        Index("ix_agent_turn_envelopes_session_id", "session_id"),
        Index("ix_agent_turn_envelopes_correlation_id", "correlation_id"),
    )

    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("agent_sessions.id"), nullable=False)
    session_sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    original_trigger_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, unique=True)
    actor_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    message_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("agent_messages.id"), nullable=True
    )
    cohort: Mapped[Any] = jsonb_column(nullable=False, default=dict)
    correlation_id: Mapped[str] = mapped_column(String(64), nullable=False)

    session: Mapped["AgentSession"] = relationship(
        back_populates="turns", foreign_keys=[session_id]
    )
    triggers: Mapped[list["AgentTurnTrigger"]] = relationship(
        back_populates="turn", cascade="all, delete-orphan", foreign_keys="AgentTurnTrigger.turn_id"
    )
    execution_state: Mapped["TurnExecutionState"] = relationship(
        back_populates="turn", cascade="all, delete-orphan", uselist=False
    )


class AgentTurnTrigger(AuditMixin, Base):
    __tablename__ = "agent_turn_triggers"
    __table_args__ = (
        UniqueConstraint(
            "session_id", "trigger_type", "idempotency_key_hash", name="uq_agent_turn_trigger_idempotency"
        ),
        Index("ix_agent_turn_triggers_session_id", "session_id"),
        Index("ix_agent_turn_triggers_turn_id", "turn_id"),
    )

    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("agent_sessions.id"), nullable=False)
    turn_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("agent_turn_envelopes.id"), nullable=True
    )
    trigger_type: Mapped[AgentTurnTriggerType] = enum_column(AgentTurnTriggerType, nullable=False)
    idempotency_key_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    actor_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    message_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("agent_messages.id"), nullable=True
    )

    turn: Mapped["AgentTurnEnvelope | None"] = relationship(back_populates="triggers", foreign_keys=[turn_id])


class TurnExecutionState(AuditMixin, Base):
    __tablename__ = "turn_execution_states"
    __table_args__ = (Index("ix_turn_execution_states_status", "status"),)

    turn_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("agent_turn_envelopes.id"), nullable=False, unique=True
    )
    status: Mapped[TurnExecutionStatus] = enum_column(
        TurnExecutionStatus, nullable=False, default=TurnExecutionStatus.PENDING
    )
    owner_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    ownership_generation: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    lease_expires_at: Mapped[Any | None] = mapped_column(DateTime(timezone=True), nullable=True)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    transition_version: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")

    turn: Mapped["AgentTurnEnvelope"] = relationship(back_populates="execution_state")


class AgentMessage(AuditMixin, Base):
    __tablename__ = "agent_messages"
    __table_args__ = (
        Index("ix_agent_messages_session_id", "session_id"),
        Index("ix_agent_messages_role", "role"),
    )

    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("agent_sessions.id"), nullable=False)
    role: Mapped[AgentMessageRole] = enum_column(AgentMessageRole, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # Structured envelope (kind/locale/options/blocks/queued). Nullable + additive: content stays
    # the mandatory fallback so legacy clients keep working when payload is None.
    payload: Mapped[Any | None] = jsonb_column(nullable=True)

    session: Mapped[AgentSession] = relationship(back_populates="messages")


class AgentRun(AuditMixin, Base):
    __tablename__ = "agent_runs"
    __table_args__ = (
        Index("ix_agent_runs_session_id", "session_id"),
        Index("ix_agent_runs_provider_config_id", "provider_config_id"),
    )

    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("agent_sessions.id"), nullable=False)
    analysis_result: Mapped[Any] = jsonb_column(nullable=False, default=dict)
    provider_config_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("llm_provider_configs.id"),
        nullable=True,
    )
    model_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    token_usage: Mapped[Any | None] = jsonb_column(nullable=True)  # {"input": int, "output": int, "total": int}
    latency_ms: Mapped[int | None] = mapped_column(nullable=True)

    session: Mapped[AgentSession] = relationship(back_populates="runs")
    tool_calls: Mapped[list["AgentToolCall"]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
    )


class AgentToolCall(AuditMixin, Base):
    __tablename__ = "agent_tool_calls"
    __table_args__ = (
        Index("ix_agent_tool_calls_run_id", "run_id"),
        Index("ix_agent_tool_calls_status", "status"),
        Index("ix_agent_tool_calls_created_artifact_id", "created_artifact_id"),
        Index("ix_agent_tool_calls_created_version_id", "created_version_id"),
    )

    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("agent_runs.id"), nullable=False)
    tool_name: Mapped[str] = mapped_column(String(100), nullable=False)
    input_snapshot: Mapped[Any] = jsonb_column(nullable=False, default=dict)
    status: Mapped[AgentToolCallStatus] = enum_column(
        AgentToolCallStatus, nullable=False, default=AgentToolCallStatus.PROPOSED
    )
    created_artifact_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("artifacts.id"), nullable=True
    )
    created_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("artifact_versions.id"), nullable=True
    )
    resolved_at: Mapped[Any] = mapped_column(DateTime(timezone=True), nullable=True)

    run: Mapped[AgentRun] = relationship(back_populates="tool_calls")
