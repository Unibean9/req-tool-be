"""add strong_model_name to llm_provider_configs

Revision ID: c9d0e1f2a3b4
Revises: bc23de45fa67
Create Date: 2026-06-20

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "c9d0e1f2a3b4"
down_revision: Union[str, None] = "bc23de45fa67"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("llm_provider_configs", sa.Column("strong_model_name", sa.String(length=255), nullable=True))


def downgrade() -> None:
    op.drop_column("llm_provider_configs", "strong_model_name")
