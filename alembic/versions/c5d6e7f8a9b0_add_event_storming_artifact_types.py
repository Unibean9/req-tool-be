"""add event storming artifact types

Revision ID: c5d6e7f8a9b0
Revises: b4c5d6e7f8a9
Create Date: 2026-07-04

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "c5d6e7f8a9b0"
down_revision: Union[str, None] = "b4c5d6e7f8a9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # PG forbids USING a value added via ALTER TYPE ADD VALUE in the same transaction that added it,
    # so this migration only ADDs the five new values with no same-transaction index/constraint use.
    op.execute(sa.text("ALTER TYPE artifacttype ADD VALUE IF NOT EXISTS 'event_storming'"))
    op.execute(sa.text("ALTER TYPE artifacttype ADD VALUE IF NOT EXISTS 'domain_event'"))
    op.execute(sa.text("ALTER TYPE artifacttype ADD VALUE IF NOT EXISTS 'actor_command'"))
    op.execute(sa.text("ALTER TYPE artifacttype ADD VALUE IF NOT EXISTS 'policy'"))
    op.execute(sa.text("ALTER TYPE artifacttype ADD VALUE IF NOT EXISTS 'aggregate'"))


def downgrade() -> None:
    # PG enum values cannot be dropped in place. No-op.
    pass
