"""add agent turn control plane

Revision ID: b5c6d7e8f9a0
Revises: a4b5c6d7e8f9
Create Date: 2026-07-14
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "b5c6d7e8f9a0"
down_revision: Union[str, None] = "a4b5c6d7e8f9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("agent_sessions", sa.Column("turn_sequence", sa.Integer(), server_default="0", nullable=False))
    # Tạo enum đúng một lần. `op.create_table()` không được phép thử tạo lại type vừa tạo,
    # nếu không PostgreSQL báo DuplicateObject trên database sạch.
    trigger_type = postgresql.ENUM(
        "user_message", "approval", "cancel", "retry", name="agentturntriggertype", create_type=False
    )
    execution_status = postgresql.ENUM(
        "pending", "running", "waiting", "terminal", name="turnexecutionstatus", create_type=False
    )
    trigger_type.create(op.get_bind(), checkfirst=True)
    execution_status.create(op.get_bind(), checkfirst=True)
    op.create_table(
        "agent_turn_envelopes",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("session_sequence", sa.Integer(), nullable=False),
        sa.Column("original_trigger_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("actor_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("message_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("cohort", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("correlation_id", sa.String(length=64), nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["agent_sessions.id"]),
        sa.ForeignKeyConstraint(["actor_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["message_id"], ["agent_messages.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("session_id", "session_sequence", name="uq_agent_turn_envelopes_session_sequence"),
        sa.UniqueConstraint("original_trigger_id"),
    )
    op.create_index("ix_agent_turn_envelopes_session_id", "agent_turn_envelopes", ["session_id"])
    op.create_index("ix_agent_turn_envelopes_correlation_id", "agent_turn_envelopes", ["correlation_id"])
    op.add_column(
        "agent_sessions",
        sa.Column("active_turn_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_index("ix_agent_sessions_active_turn_id", "agent_sessions", ["active_turn_id"])
    op.create_table(
        "agent_turn_triggers",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("turn_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("trigger_type", trigger_type, nullable=False),
        sa.Column("idempotency_key_hash", sa.String(length=64), nullable=True),
        sa.Column("actor_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("message_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.ForeignKeyConstraint(["session_id"], ["agent_sessions.id"]),
        sa.ForeignKeyConstraint(["turn_id"], ["agent_turn_envelopes.id"]),
        sa.ForeignKeyConstraint(["actor_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["message_id"], ["agent_messages.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("session_id", "trigger_type", "idempotency_key_hash", name="uq_agent_turn_trigger_idempotency"),
    )
    op.create_index("ix_agent_turn_triggers_session_id", "agent_turn_triggers", ["session_id"])
    op.create_index("ix_agent_turn_triggers_turn_id", "agent_turn_triggers", ["turn_id"])
    op.create_table(
        "turn_execution_states",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("turn_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("status", execution_status, nullable=False),
        sa.Column("owner_id", sa.String(length=128), nullable=True),
        sa.Column("ownership_generation", sa.Integer(), server_default="0", nullable=False),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("attempt", sa.Integer(), server_default="0", nullable=False),
        sa.Column("transition_version", sa.Integer(), server_default="0", nullable=False),
        sa.ForeignKeyConstraint(["turn_id"], ["agent_turn_envelopes.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("turn_id"),
    )
    op.create_index("ix_turn_execution_states_status", "turn_execution_states", ["status"])


def downgrade() -> None:
    op.drop_index("ix_turn_execution_states_status", table_name="turn_execution_states")
    op.drop_table("turn_execution_states")
    op.drop_index("ix_agent_turn_triggers_turn_id", table_name="agent_turn_triggers")
    op.drop_index("ix_agent_turn_triggers_session_id", table_name="agent_turn_triggers")
    op.drop_table("agent_turn_triggers")
    op.drop_index("ix_agent_sessions_active_turn_id", table_name="agent_sessions")
    op.drop_column("agent_sessions", "active_turn_id")
    op.drop_index("ix_agent_turn_envelopes_correlation_id", table_name="agent_turn_envelopes")
    op.drop_index("ix_agent_turn_envelopes_session_id", table_name="agent_turn_envelopes")
    op.drop_table("agent_turn_envelopes")
    op.drop_column("agent_sessions", "turn_sequence")
    postgresql.ENUM(name="turnexecutionstatus").drop(op.get_bind(), checkfirst=True)
    postgresql.ENUM(name="agentturntriggertype").drop(op.get_bind(), checkfirst=True)
