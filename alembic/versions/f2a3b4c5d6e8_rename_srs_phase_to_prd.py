"""Rename workflow phase srs to prd

Revision ID: f2a3b4c5d6e8
Revises: f1a2b3c4d5e7
Create Date: 2026-06-23

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "f2a3b4c5d6e8"
down_revision: Union[str, None] = "f1a2b3c4d5e7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute(sa.text("ALTER TYPE workflowstepphase RENAME TO workflowstepphase_old"))
        op.execute(sa.text("CREATE TYPE workflowstepphase AS ENUM ('brd', 'prd', 'delivery')"))
        op.execute(
            sa.text(
                "ALTER TABLE workflow_steps "
                "ALTER COLUMN phase TYPE workflowstepphase "
                "USING (CASE WHEN phase::text = 'srs' THEN 'prd' ELSE phase::text END)::workflowstepphase"
            )
        )
        op.execute(sa.text("DROP TYPE workflowstepphase_old"))
    else:
        op.execute(sa.text("UPDATE workflow_steps SET phase = 'prd' WHERE phase = 'srs'"))


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute(sa.text("ALTER TYPE workflowstepphase RENAME TO workflowstepphase_old"))
        op.execute(sa.text("CREATE TYPE workflowstepphase AS ENUM ('brd', 'srs', 'delivery')"))
        op.execute(
            sa.text(
                "ALTER TABLE workflow_steps "
                "ALTER COLUMN phase TYPE workflowstepphase "
                "USING (CASE WHEN phase::text = 'prd' THEN 'srs' ELSE phase::text END)::workflowstepphase"
            )
        )
        op.execute(sa.text("DROP TYPE workflowstepphase_old"))
    else:
        op.execute(sa.text("UPDATE workflow_steps SET phase = 'srs' WHERE phase = 'prd'"))
