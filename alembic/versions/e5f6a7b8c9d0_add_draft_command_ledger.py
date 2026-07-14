"""add draft command ledger

Revision ID: e5f6a7b8c9d0
Revises: b5c6d7e8f9a0
Create Date: 2026-07-15
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "e5f6a7b8c9d0"
down_revision: Union[str, None] = "b5c6d7e8f9a0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    effect_state = postgresql.ENUM(
        "pending", "committed", "failed", name="draftcommandeffectstate", create_type=False
    )
    effect_state.create(op.get_bind(), checkfirst=True)
    op.create_table(
        "agent_draft_commands",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("turn_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("logical_command_id", sa.String(length=64), nullable=False),
        sa.Column("action_type", sa.String(length=50), nullable=False),
        sa.Column("tool_call_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("artifact_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("effect_state", effect_state, nullable=False),
        sa.Column("attempt", sa.Integer(), server_default="0", nullable=False),
        sa.ForeignKeyConstraint(["turn_id"], ["agent_turn_envelopes.id"]),
        sa.ForeignKeyConstraint(["tool_call_id"], ["agent_tool_calls.id"]),
        sa.ForeignKeyConstraint(["artifact_id"], ["artifacts.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("logical_command_id", name="uq_agent_draft_commands_logical_command_id"),
    )
    op.create_index("ix_agent_draft_commands_turn_id", "agent_draft_commands", ["turn_id"])


def downgrade() -> None:
    op.drop_index("ix_agent_draft_commands_turn_id", table_name="agent_draft_commands")
    op.drop_table("agent_draft_commands")
    postgresql.ENUM(name="draftcommandeffectstate").drop(op.get_bind(), checkfirst=True)
