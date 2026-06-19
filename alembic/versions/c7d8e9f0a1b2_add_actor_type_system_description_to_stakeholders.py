"""add actor_type and system_description to stakeholders

Revision ID: c7d8e9f0a1b2
Revises: b6c7d8e9f0a1
Create Date: 2026-05-23

"""
import warnings
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "c7d8e9f0a1b2"
down_revision: Union[str, None] = "b6c7d8e9f0a1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()

    warnings.warn(
        "\n[MIGRATION WARNING] Updating stakeholders.actor_type from is_business_actor:\n"
        "  - is_business_actor=TRUE  → actor_type='business_actor'\n"
        "  - is_business_actor=FALSE → actor_type='none'\n"
        "  Downgrade sẽ DROP column actor_type nhưng KHÔNG khôi phục is_business_actor về trạng thái gốc.",
        stacklevel=2,
    )

    conn.execute(sa.text("DROP TYPE IF EXISTS actortype"))
    conn.execute(sa.text(
        "CREATE TYPE actortype AS ENUM ('none', 'business_actor', 'other_actor')"
    ))

    op.add_column("stakeholders", sa.Column(
        "actor_type",
        postgresql.ENUM(name="actortype", create_type=False),
        nullable=False,
        server_default="none",
    ))
    op.alter_column("stakeholders", "actor_type", server_default=None)

    op.add_column("stakeholders", sa.Column("system_description", sa.Text(), nullable=True))

    conn.execute(sa.text(
        "UPDATE stakeholders SET actor_type = 'business_actor' WHERE is_business_actor = TRUE"
    ))


def downgrade() -> None:
    op.drop_column("stakeholders", "system_description")
    op.drop_column("stakeholders", "actor_type")
    conn = op.get_bind()
    conn.execute(sa.text("DROP TYPE IF EXISTS actortype"))
