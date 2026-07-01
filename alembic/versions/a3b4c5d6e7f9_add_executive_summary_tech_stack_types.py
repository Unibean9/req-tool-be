"""add executive_summary and tech_stack artifact types

Revision ID: a3b4c5d6e7f9
Revises: e1f2a3b4c5d8
Create Date: 2026-07-01

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "a3b4c5d6e7f9"
down_revision: Union[str, None] = "e1f2a3b4c5d8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # PG forbids USING a value added via ALTER TYPE ADD VALUE in the same transaction that added it,
    # so this migration only ADDs the two new values with no same-transaction index/constraint use.
    op.execute(sa.text("ALTER TYPE artifacttype ADD VALUE IF NOT EXISTS 'executive_summary'"))
    op.execute(sa.text("ALTER TYPE artifacttype ADD VALUE IF NOT EXISTS 'tech_stack'"))


def downgrade() -> None:
    # PG enum values cannot be dropped in place. No-op.
    pass
