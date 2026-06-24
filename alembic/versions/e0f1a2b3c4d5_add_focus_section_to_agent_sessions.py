"""add focus_section to agent_sessions

Revision ID: e0f1a2b3c4d5
Revises: d0e1f2a3b4c5
Create Date: 2026-06-22

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "e0f1a2b3c4d5"
down_revision: Union[str, None] = "d0e1f2a3b4c5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("agent_sessions", sa.Column("focus_section", sa.String(length=100), nullable=True))


def downgrade() -> None:
    op.drop_column("agent_sessions", "focus_section")
