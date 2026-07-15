"""add turn outcomes

Revision ID: b26dfb69a3e9
Revises: e5f6a7b8c9d0
Create Date: 2026-07-15
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "b26dfb69a3e9"
down_revision: Union[str, None] = "e5f6a7b8c9d0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    outcome_type = postgresql.ENUM(
        "continue",
        "wait_input",
        "wait_approval",
        "direct_response",
        "completed",
        "recoverable_failure",
        "terminal_failure",
        "cancelled",
        name="turnoutcometype",
        create_type=False,
    )
    outcome_type.create(op.get_bind(), checkfirst=True)
    op.create_table(
        "agent_turn_outcomes",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("turn_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("outcome_type", outcome_type, nullable=False),
        sa.Column("reason", sa.String(length=255), nullable=True),
        sa.Column("committed_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["turn_id"], ["agent_turn_envelopes.id"]),
        sa.ForeignKeyConstraint(["session_id"], ["agent_sessions.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_agent_turn_outcomes_turn_id", "agent_turn_outcomes", ["turn_id"])
    op.create_index("ix_agent_turn_outcomes_session_id", "agent_turn_outcomes", ["session_id"])


def downgrade() -> None:
    op.drop_index("ix_agent_turn_outcomes_session_id", table_name="agent_turn_outcomes")
    op.drop_index("ix_agent_turn_outcomes_turn_id", table_name="agent_turn_outcomes")
    op.drop_table("agent_turn_outcomes")
    postgresql.ENUM(name="turnoutcometype").drop(op.get_bind(), checkfirst=True)
