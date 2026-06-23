"""add requirements artifact type

Revision ID: f0a1b2c3d4e6
Revises: e0f1a2b3c4d5
Create Date: 2026-06-23

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "f0a1b2c3d4e6"
down_revision: Union[str, None] = "e0f1a2b3c4d5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # PG forbids USING a value added via ALTER TYPE ADD VALUE in the same transaction that added it.
    # So this migration only ADDs 'requirements'; the partial unique index that references it lives in
    # the later recreate migration (f1a2b3c4d5e7), where 'requirements' is a CREATE TYPE member and is
    # therefore usable in that transaction.
    op.execute(sa.text("ALTER TYPE artifacttype ADD VALUE IF NOT EXISTS 'requirements'"))


def downgrade() -> None:
    # PG enum values cannot be dropped in place; the recreate migration owns the value set. No-op.
    pass
