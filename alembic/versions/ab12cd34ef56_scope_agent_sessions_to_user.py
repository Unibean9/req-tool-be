"""scope agent sessions to user

Revision ID: ab12cd34ef56
Revises: a8b9c0d1e2f3
Create Date: 2026-06-19

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "ab12cd34ef56"
down_revision: Union[str, None] = "a8b9c0d1e2f3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_index("uq_agent_sessions_active_project_artifact_type", table_name="agent_sessions")
    op.create_index(
        "uq_agent_sessions_active_project_artifact_type_user",
        "agent_sessions",
        ["project_id", "artifact_type", "created_by_id"],
        unique=True,
        postgresql_where=sa.text("status IN ('active', 'waiting_for_human')"),
    )


def downgrade() -> None:
    op.drop_index("uq_agent_sessions_active_project_artifact_type_user", table_name="agent_sessions")
    op.create_index(
        "uq_agent_sessions_active_project_artifact_type",
        "agent_sessions",
        ["project_id", "artifact_type"],
        unique=True,
        postgresql_where=sa.text("status IN ('active', 'waiting_for_human')"),
    )
