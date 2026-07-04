"""add project executive_summary column

Revision ID: b4c5d6e7f8a9
Revises: a3b4c5d6e7f9
Create Date: 2026-07-04

Promotes the executive summary from an elicited BRD artifact to a synthesized
project-level field. Scoped to the single new nullable column; no enum or
other schema changes.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "b4c5d6e7f8a9"
down_revision: Union[str, None] = "a3b4c5d6e7f9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("projects", sa.Column("executive_summary", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("projects", "executive_summary")
