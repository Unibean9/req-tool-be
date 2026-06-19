"""add agent_role to agent_sessions

Revision ID: a8b9c0d1e2f3
Revises: f6a7b8c9d0e1
Create Date: 2026-06-19

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "a8b9c0d1e2f3"
down_revision: Union[str, None] = "f6a7b8c9d0e1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("agent_sessions", sa.Column("agent_role", sa.String(100), nullable=True))


def downgrade() -> None:
    op.drop_column("agent_sessions", "agent_role")
