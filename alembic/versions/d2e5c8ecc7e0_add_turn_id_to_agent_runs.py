"""add turn_id attribution column to agent_runs

Revision ID: d2e5c8ecc7e0
Revises: d02fa1bc91a3
Create Date: 2026-07-16
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "d2e5c8ecc7e0"
down_revision: Union[str, None] = "d02fa1bc91a3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "agent_runs",
        sa.Column("turn_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_agent_runs_turn_id_agent_turn_envelopes",
        "agent_runs",
        "agent_turn_envelopes",
        ["turn_id"],
        ["id"],
    )
    op.create_index("ix_agent_runs_turn_id", "agent_runs", ["turn_id"])


def downgrade() -> None:
    op.drop_index("ix_agent_runs_turn_id", table_name="agent_runs")
    op.drop_constraint("fk_agent_runs_turn_id_agent_turn_envelopes", "agent_runs", type_="foreignkey")
    op.drop_column("agent_runs", "turn_id")
