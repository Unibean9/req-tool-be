"""add proxy llm provider types

Revision ID: e1f2a3b4c5d8
Revises: e1f2a3b4c5d7
Create Date: 2026-06-29 14:39:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "e1f2a3b4c5d8"
down_revision: str | Sequence[str] | None = "e1f2a3b4c5d7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(sa.text("ALTER TYPE providertype ADD VALUE IF NOT EXISTS 'deepseek'"))
    op.execute(sa.text("ALTER TYPE providertype ADD VALUE IF NOT EXISTS 'mistral'"))
    op.execute(sa.text("ALTER TYPE providertype ADD VALUE IF NOT EXISTS 'openrouter'"))


def downgrade() -> None:
    pass
