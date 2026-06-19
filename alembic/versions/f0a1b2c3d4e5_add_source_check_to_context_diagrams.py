"""add source check constraint to project_context_diagrams

Revision ID: f0a1b2c3d4e5
Revises: e9f0a1b2c3d4
Create Date: 2026-05-24

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "f0a1b2c3d4e5"
down_revision: Union[str, None] = "e9f0a1b2c3d4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_check_constraint(
        "ck_project_context_diagrams_source",
        "project_context_diagrams",
        "source IN ('manual', 'generated', 'synced')",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_project_context_diagrams_source",
        "project_context_diagrams",
        type_="check",
    )
