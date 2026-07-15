"""add checkpoint v2 history and turn events

Revision ID: d02fa1bc91a3
Revises: d38efc70b4f0
Create Date: 2026-07-16
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "d02fa1bc91a3"
down_revision: Union[str, None] = "d38efc70b4f0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "agent_sessions",
        sa.Column("checkpoint_version", sa.String(length=8), server_default="v1", nullable=False),
    )
    op.add_column(
        "agent_sessions",
        sa.Column("event_cursor", sa.Integer(), server_default="0", nullable=False),
    )

    event_type = postgresql.ENUM(
        "checkpoint_appended",
        "outcome_committed",
        name="agentturneventtype",
        create_type=False,
    )
    event_type.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "agent_checkpoints",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("turn_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("checkpoint_id", sa.String(length=64), nullable=False),
        sa.Column("parent_checkpoint_id", sa.String(length=64), nullable=True),
        sa.Column("session_sequence", sa.Integer(), server_default="0", nullable=False),
        sa.Column("ownership_generation", sa.Integer(), server_default="0", nullable=False),
        sa.Column("serde_type", sa.String(length=32), nullable=False),
        sa.Column("data", sa.LargeBinary(), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), server_default="{}", nullable=False),
        sa.Column("new_versions", postgresql.JSONB(), server_default="{}", nullable=False),
        sa.Column("pending_writes", postgresql.JSONB(), server_default="[]", nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["agent_sessions.id"]),
        sa.ForeignKeyConstraint(["turn_id"], ["agent_turn_envelopes.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_unique_constraint(
        "uq_agent_checkpoints_session_checkpoint", "agent_checkpoints", ["session_id", "checkpoint_id"]
    )
    op.create_index(
        "ix_agent_checkpoints_session_created_at", "agent_checkpoints", ["session_id", "created_at"]
    )
    op.create_index(
        "ix_agent_checkpoints_session_parent", "agent_checkpoints", ["session_id", "parent_checkpoint_id"]
    )
    op.create_index("ix_agent_checkpoints_turn_id", "agent_checkpoints", ["turn_id"])

    op.create_table(
        "agent_turn_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("turn_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("session_sequence", sa.Integer(), nullable=False),
        sa.Column("event_type", event_type, nullable=False),
        sa.Column("parent_checkpoint_id", sa.String(length=64), nullable=True),
        sa.Column("payload", postgresql.JSONB(), server_default="{}", nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["agent_sessions.id"]),
        sa.ForeignKeyConstraint(["turn_id"], ["agent_turn_envelopes.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_unique_constraint(
        "uq_agent_turn_events_session_sequence", "agent_turn_events", ["session_id", "session_sequence"]
    )
    op.create_index("ix_agent_turn_events_session_id", "agent_turn_events", ["session_id"])
    op.create_index("ix_agent_turn_events_turn_id", "agent_turn_events", ["turn_id"])


def downgrade() -> None:
    op.drop_index("ix_agent_turn_events_turn_id", table_name="agent_turn_events")
    op.drop_index("ix_agent_turn_events_session_id", table_name="agent_turn_events")
    op.drop_constraint("uq_agent_turn_events_session_sequence", "agent_turn_events", type_="unique")
    op.drop_table("agent_turn_events")

    op.drop_index("ix_agent_checkpoints_turn_id", table_name="agent_checkpoints")
    op.drop_index("ix_agent_checkpoints_session_parent", table_name="agent_checkpoints")
    op.drop_index("ix_agent_checkpoints_session_created_at", table_name="agent_checkpoints")
    op.drop_constraint("uq_agent_checkpoints_session_checkpoint", "agent_checkpoints", type_="unique")
    op.drop_table("agent_checkpoints")

    postgresql.ENUM(name="agentturneventtype").drop(op.get_bind(), checkfirst=True)

    op.drop_column("agent_sessions", "event_cursor")
    op.drop_column("agent_sessions", "checkpoint_version")
