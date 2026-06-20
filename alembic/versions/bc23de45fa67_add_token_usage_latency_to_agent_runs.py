"""add token_usage latency_ms to agent_runs

Revision ID: bc23de45fa67
Revises: ab12cd34ef56
Create Date: 2026-06-20

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "bc23de45fa67"
down_revision: Union[str, None] = "ab12cd34ef56"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "agent_runs",
        sa.Column("token_usage", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.add_column("agent_runs", sa.Column("latency_ms", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("agent_runs", "latency_ms")
    op.drop_column("agent_runs", "token_usage")
