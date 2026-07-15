import enum
import uuid
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Index, Integer, LargeBinary, String, Text, UniqueConstraint, func, text
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


class DraftCommandEffectState(enum.StrEnum):
    PENDING = "pending"
    COMMITTED = "committed"
    FAILED = "failed"


class AgentTurnJobStatus(enum.StrEnum):
    QUEUED = "queued"
    CLAIMED = "claimed"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"


class AgentTurnEventType(enum.StrEnum):
    """Outbox event vocabulary. Only the values an actual writer emits belong here — see
    `emit_turn_event` (`app/graphs/analysis/turn_event_log.py`) for `CHECKPOINT_APPENDED` and
    `turn_outcome_projector.py` for `OUTCOME_COMMITTED`; do not add a value with no writer.
    """

    CHECKPOINT_APPENDED = "checkpoint_appended"
    OUTCOME_COMMITTED = "outcome_committed"


class TurnOutcomeType(enum.StrEnum):
    """Terminal/transition vocabulary for a logical turn's outcome (ADR 0001).

    Only the terminal values (`COMPLETED`, `TERMINAL_FAILURE`, `RECOVERABLE_FAILURE`, `CANCELLED`)
    are ever committed today, through `project_terminal_outcome` — the non-terminal values are
    defined for the full outcome vocabulary but have no writer yet.
    """

    CONTINUE = "continue"
    WAIT_INPUT = "wait_input"
    WAIT_APPROVAL = "wait_approval"
    DIRECT_RESPONSE = "direct_response"
    COMPLETED = "completed"
    RECOVERABLE_FAILURE = "recoverable_failure"
    TERMINAL_FAILURE = "terminal_failure"
    CANCELLED = "cancelled"


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
    # Session-grain checkpoint cohort. Set once at session creation from the checkpoint-history
    # feature flag at that instant and never re-read live afterward — a LangGraph checkpoint is
    # keyed by thread_id (= session_id), not by turn, so a session must stay on the same reader for
    # its whole lifetime even if the flag flips mid-session. "v1" reads/writes `graph_checkpoint`
    # through `AgentSessionCheckpointer`; "v2" reads/writes the `agent_checkpoints` history table
    # through `AgentCheckpointHistorySaver`. `graph_checkpoint` is left unused (stays `{}`) for v2
    # sessions rather than dual-written.
    checkpoint_version: Mapped[str] = mapped_column(String(8), nullable=False, default="v1", server_default="v1")
    # Monotonic per-session outbox cursor. Only `emit_turn_event` increments it, in the same
    # transaction as the `AgentTurnEvent` row it writes — never a second, independent counter.
    event_cursor: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")

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
        UniqueConstraint("tool_call_id", name="uq_agent_turn_trigger_approval_tool_call"),
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
    # Approval identity is server-side: one logical approval turn per proposed tool call.
    tool_call_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("agent_tool_calls.id"), nullable=True
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


class AgentTurnJob(AuditMixin, Base):
    """Durable enqueue/claim record for a turn's execution, one row per turn (no independent
    identity or sequence of its own — `turn_id` unique enforces "at most one job per turn").

    `lease_generation` is not an independently mutable counter: every claim/renew/reclaim writes it
    in lockstep with `TurnExecutionState.ownership_generation` inside the same row-locked
    transaction, so the execution-state generation stays the single source of truth for the fence
    and this column is always a mirror of it, never a value that can drift apart.

    `cohort` is copied from the admitted envelope at enqueue time and never re-read from live
    settings during claim/renew/reclaim/complete — the same snapshot-at-admission discipline every
    other turn-scoped cohort field in this module already follows.
    """

    __tablename__ = "agent_turn_jobs"
    __table_args__ = (
        UniqueConstraint("turn_id", name="uq_agent_turn_jobs_turn_id"),
        Index("ix_agent_turn_jobs_status", "status"),
        Index("ix_agent_turn_jobs_lease_expires_at", "lease_expires_at"),
    )

    turn_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("agent_turn_envelopes.id"), nullable=False, unique=True
    )
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    scheduled_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    status: Mapped[AgentTurnJobStatus] = enum_column(
        AgentTurnJobStatus, nullable=False, default=AgentTurnJobStatus.QUEUED
    )
    lease_owner: Mapped[str | None] = mapped_column(String(128), nullable=True)
    lease_generation: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    lease_expires_at: Mapped[Any | None] = mapped_column(DateTime(timezone=True), nullable=True)
    heartbeat_at: Mapped[Any | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    cohort: Mapped[Any] = jsonb_column(nullable=False, default=dict)
    expected_transition_version: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")


class DraftCommandLedger(AuditMixin, Base):
    """Idempotency/effect ledger for the command boundary shared by write_draft, finalize,
    create_artifact_link and propose_retirement.

    `logical_command_id` is the business identity (turn + action type + canonical intent + expected
    base version) — not the provider tool-call ID, which only rides `tool_call_id` for correlation.
    The unique constraint on `logical_command_id` is the sole idempotency invariant, mirroring
    `AgentTurnTrigger.idempotency_key_hash`'s unique-constraint-is-the-source-of-truth pattern.
    Only reachable when the admitting turn's cohort has `command_handlers_enabled`; legacy cohorts
    never write here.
    """

    __tablename__ = "agent_draft_commands"
    __table_args__ = (
        UniqueConstraint("logical_command_id", name="uq_agent_draft_commands_logical_command_id"),
        Index("ix_agent_draft_commands_turn_id", "turn_id"),
    )

    turn_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("agent_turn_envelopes.id"), nullable=False
    )
    logical_command_id: Mapped[str] = mapped_column(String(64), nullable=False)
    action_type: Mapped[str] = mapped_column(String(50), nullable=False)
    tool_call_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("agent_tool_calls.id"), nullable=True
    )
    artifact_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("artifacts.id"), nullable=True)
    effect_state: Mapped[DraftCommandEffectState] = enum_column(
        DraftCommandEffectState, nullable=False, default=DraftCommandEffectState.COMMITTED
    )
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")


class TurnOutcome(AuditMixin, Base):
    """Committed terminal-outcome audit row.

    Written only by `project_terminal_outcome` (`app/graphs/analysis/turn_outcome_projector.py`),
    and only when the admitting turn's cohort snapshot has `turn_outcomes_enabled=True` — same
    snapshot-at-admission contract as `command_handlers_enabled`. A turn without this
    flag, or a terminal write with no turn context at all (e.g. lazy session expiry), never gets a
    row here; the compatibility `AgentSession.status` write still happens either way.
    """

    __tablename__ = "agent_turn_outcomes"
    __table_args__ = (
        UniqueConstraint("turn_id", name="uq_agent_turn_outcomes_turn_id"),
        Index("ix_agent_turn_outcomes_turn_id", "turn_id"),
        Index("ix_agent_turn_outcomes_session_id", "session_id"),
    )

    turn_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("agent_turn_envelopes.id"), nullable=False
    )
    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("agent_sessions.id"), nullable=False)
    outcome_type: Mapped[TurnOutcomeType] = enum_column(TurnOutcomeType, nullable=False)
    reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    committed_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class AgentCheckpoint(AuditMixin, Base):
    """Checkpoint v2 history row — one row per LangGraph checkpoint ever appended for a session on
    the `checkpoint_version == "v2"` cohort, written only by `AgentCheckpointHistorySaver`
    (`app/graphs/checkpointer.py`).

    `parent_checkpoint_id` must equal the session's current head `checkpoint_id` at append time (or
    both be null for a session's first checkpoint) AND `ownership_generation` must match the
    admitting turn's live `TurnExecutionState.ownership_generation` — both checked inside the same
    transaction as the insert by the saver; a mismatch raises rather than silently overwriting or
    forking history, so `alist()` walking this table by `created_at`/`parent_checkpoint_id` is always
    a single linear chain per session, never a fork. `turn_id` is nullable only because a v2
    session's very first checkpoint could in principle be written before any turn concept applies;
    every v2 session in practice always has one, since only new-cohort sessions ever land on v2.
    `session_sequence` is copied from the admitting turn's `AgentTurnEnvelope.session_sequence` at
    write time for audit correlation only — it is not this table's identity or ordering key
    (`created_at`/`parent_checkpoint_id` are). `pending_writes` holds the same task/channel/value
    shape `AgentSessionCheckpointer` (v1) stores, scoped to this row's own `checkpoint_id` rather than
    a shared session-wide blob.
    """

    __tablename__ = "agent_checkpoints"
    __table_args__ = (
        UniqueConstraint("session_id", "checkpoint_id", name="uq_agent_checkpoints_session_checkpoint"),
        Index("ix_agent_checkpoints_session_created_at", "session_id", "created_at"),
        Index("ix_agent_checkpoints_session_parent", "session_id", "parent_checkpoint_id"),
        Index("ix_agent_checkpoints_turn_id", "turn_id"),
    )

    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("agent_sessions.id"), nullable=False)
    turn_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("agent_turn_envelopes.id"), nullable=True
    )
    checkpoint_id: Mapped[str] = mapped_column(String(64), nullable=False)
    parent_checkpoint_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    session_sequence: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    ownership_generation: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    serde_type: Mapped[str] = mapped_column(String(32), nullable=False)
    data: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    # Named checkpoint_metadata (not metadata) because `metadata` is reserved by
    # DeclarativeBase for the schema MetaData collection; the underlying column stays "metadata".
    checkpoint_metadata: Mapped[Any] = jsonb_column("metadata", nullable=False, default=dict)
    new_versions: Mapped[Any] = jsonb_column(nullable=False, default=dict)
    pending_writes: Mapped[Any] = jsonb_column(nullable=False, default=list)


class AgentTurnEvent(AuditMixin, Base):
    """Outbox/event log row — an audited side effect of an already-committed transition, written
    only by `emit_turn_event` (`app/graphs/analysis/turn_event_log.py`) inside the same transaction
    as the transition it records, and only for sessions on the `checkpoint_version == "v2"` cohort.

    `session_sequence` is copied from `AgentSession.event_cursor` at write time and is this table's
    ordering AND dedup key (the unique constraint below) — a race that computes the same next-cursor
    value twice is rejected by the constraint and swallowed as a no-op by the writer, never
    duplicated or reordered. This table is a projection/audit surface, not a second source of truth:
    reconciliation and resume decisions read committed `TurnOutcome`/checkpoint state, never this
    table alone. `payload` is redacted before insert (never at read time) using the same
    allowlist-by-key approach `turn_audit.py` already applies to `AgentRun.analysis_result`.
    """

    __tablename__ = "agent_turn_events"
    __table_args__ = (
        UniqueConstraint("session_id", "session_sequence", name="uq_agent_turn_events_session_sequence"),
        Index("ix_agent_turn_events_session_id", "session_id"),
        Index("ix_agent_turn_events_turn_id", "turn_id"),
    )

    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("agent_sessions.id"), nullable=False)
    turn_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("agent_turn_envelopes.id"), nullable=True
    )
    session_sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    event_type: Mapped[AgentTurnEventType] = enum_column(AgentTurnEventType, nullable=False)
    parent_checkpoint_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    payload: Mapped[Any] = jsonb_column(nullable=False, default=dict)


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
        Index("ix_agent_runs_turn_id", "turn_id"),
    )

    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("agent_sessions.id"), nullable=False)
    # Attribution only, never identity: lets on-call join an LLM attempt back to its logical turn
    # (and from there to trigger/command/outcome/event) without redefining what a turn is. Nullable
    # because a run predating this column, or one recorded with no turn context available, has no
    # value to backfill.
    turn_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("agent_turn_envelopes.id"), nullable=True
    )
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
