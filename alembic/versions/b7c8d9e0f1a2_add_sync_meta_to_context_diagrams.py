"""add sync_meta to project_context_diagrams

Revision ID: b7c8d9e0f1a2
Revises: 916870f4ab8d
Create Date: 2026-05-24

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b7c8d9e0f1a2"
down_revision: Union[str, None] = "916870f4ab8d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "project_context_diagrams",
        sa.Column("sync_meta", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("project_context_diagrams", "sync_meta")
