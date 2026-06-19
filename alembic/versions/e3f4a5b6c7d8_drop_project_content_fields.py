"""drop project content fields

Revision ID: e3f4a5b6c7d8
Revises: d1e2f3a4b5c6
Create Date: 2026-06-18

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "e3f4a5b6c7d8"
down_revision: Union[str, None] = "d1e2f3a4b5c6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    for column_name in (
        "context",
        "problems",
        "proposed_solutions",
        "start_date",
        "end_date",
        "budget",
        "executive_summary",
        "roi_notes",
    ):
        op.execute(sa.text(f'ALTER TABLE projects DROP COLUMN IF EXISTS "{column_name}"'))


def downgrade() -> None:
    op.add_column("projects", sa.Column("roi_notes", sa.Text(), nullable=True))
    op.add_column("projects", sa.Column("executive_summary", sa.Text(), nullable=True))
    op.add_column("projects", sa.Column("budget", sa.Numeric(12, 2), nullable=True))
    op.add_column("projects", sa.Column("end_date", sa.Date(), nullable=True))
    op.add_column("projects", sa.Column("start_date", sa.Date(), nullable=True))
    op.add_column("projects", sa.Column("proposed_solutions", sa.JSON(), nullable=True))
    op.add_column("projects", sa.Column("problems", sa.JSON(), nullable=True))
    op.add_column("projects", sa.Column("context", sa.Text(), nullable=True))
