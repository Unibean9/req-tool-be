"""add approval turn identity and terminal outcome fence

Revision ID: c27dfb69a3e9
Revises: b26dfb69a3e9
Create Date: 2026-07-15
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "c27dfb69a3e9"
down_revision: Union[str, None] = "b26dfb69a3e9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "agent_turn_triggers",
        sa.Column("tool_call_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_agent_turn_triggers_tool_call_id",
        "agent_turn_triggers",
        "agent_tool_calls",
        ["tool_call_id"],
        ["id"],
    )
    op.create_unique_constraint(
        "uq_agent_turn_trigger_approval_tool_call", "agent_turn_triggers", ["tool_call_id"]
    )
    # This migration may be applied to a database where the expand-side outcome table
    # already received rows.  Keep one deterministic historical row per turn before the
    # exactly-once fence is installed, so upgrade does not fail on legacy duplicates.
    op.execute(
        """
        DELETE FROM agent_turn_outcomes AS duplicate
        USING agent_turn_outcomes AS winner
        WHERE duplicate.turn_id = winner.turn_id
          AND (duplicate.created_at, duplicate.id) > (winner.created_at, winner.id)
        """
    )
    op.create_unique_constraint("uq_agent_turn_outcomes_turn_id", "agent_turn_outcomes", ["turn_id"])


def downgrade() -> None:
    op.drop_constraint("uq_agent_turn_outcomes_turn_id", "agent_turn_outcomes", type_="unique")
    op.drop_constraint("uq_agent_turn_trigger_approval_tool_call", "agent_turn_triggers", type_="unique")
    op.drop_constraint("fk_agent_turn_triggers_tool_call_id", "agent_turn_triggers", type_="foreignkey")
    op.drop_column("agent_turn_triggers", "tool_call_id")
