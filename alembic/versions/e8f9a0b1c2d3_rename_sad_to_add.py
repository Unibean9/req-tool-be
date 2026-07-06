"""rename sad artifact type to add

Revision ID: e8f9a0b1c2d3
Revises: d7e8f9a0b2
Create Date: 2026-07-06 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "e8f9a0b1c2d3"
down_revision: str | Sequence[str] | None = "d7e8f9a0b2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

def upgrade() -> None:
    bind = op.get_bind()
    op.drop_index("uq_artifacts_project_sad", table_name="artifacts")
    if bind.dialect.name == "postgresql":
        op.execute(sa.text("ALTER TYPE artifacttype RENAME VALUE 'sad' TO 'add'"))
    else:
        op.execute(sa.text("UPDATE artifacts SET type = 'add' WHERE type = 'sad'"))
    op.execute(sa.text("UPDATE agent_sessions SET artifact_type = 'add' WHERE artifact_type = 'sad'"))
    op.create_index(
        "uq_artifacts_project_add",
        "artifacts",
        ["project_id"],
        unique=True,
        postgresql_where=sa.text("type = 'add' AND status != 'archived'"),
        sqlite_where=sa.text("type = 'add' AND status != 'archived'"),
    )


def downgrade() -> None:
    bind = op.get_bind()
    op.drop_index("uq_artifacts_project_add", table_name="artifacts")
    if bind.dialect.name == "postgresql":
        op.execute(sa.text("ALTER TYPE artifacttype RENAME VALUE 'add' TO 'sad'"))
    else:
        op.execute(sa.text("UPDATE artifacts SET type = 'sad' WHERE type = 'add'"))
    op.execute(sa.text("UPDATE agent_sessions SET artifact_type = 'sad' WHERE artifact_type = 'add'"))
    op.create_index(
        "uq_artifacts_project_sad",
        "artifacts",
        ["project_id"],
        unique=True,
        postgresql_where=sa.text("type = 'sad' AND status != 'archived'"),
        sqlite_where=sa.text("type = 'sad' AND status != 'archived'"),
    )
