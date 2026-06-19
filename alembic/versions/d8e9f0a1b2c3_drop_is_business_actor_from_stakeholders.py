"""drop is_business_actor from stakeholders

Revision ID: d8e9f0a1b2c3
Revises: c7d8e9f0a1b2
Create Date: 2026-05-23

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "d8e9f0a1b2c3"
down_revision: Union[str, None] = "c7d8e9f0a1b2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_column("stakeholders", "is_business_actor")


def downgrade() -> None:
    op.add_column("stakeholders", sa.Column(
        "is_business_actor", sa.Boolean(), nullable=False, server_default=sa.false()
    ))
    op.alter_column("stakeholders", "is_business_actor", server_default=None)

    conn = op.get_bind()
    conn.execute(sa.text(
        "UPDATE stakeholders SET is_business_actor = TRUE WHERE actor_type = 'business_actor'"
    ))
