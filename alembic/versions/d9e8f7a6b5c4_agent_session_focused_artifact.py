"""replace focus section with focused artifact

Revision ID: d9e8f7a6b5c4
Revises: d3e4f5a6b7c8
Create Date: 2026-06-23

Deploy order: deploy and verify compatible code first, then run this migration.
ACTIVE checkpoints are intentionally cleared and are not restored on downgrade.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "d9e8f7a6b5c4"
down_revision: Union[str, None] = "d3e4f5a6b7c8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "agent_sessions",
        sa.Column("focused_artifact_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_agent_sessions_focused_artifact_id",
        "agent_sessions",
        "artifacts",
        ["focused_artifact_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_agent_sessions_focused_artifact_id",
        "agent_sessions",
        ["focused_artifact_id"],
    )
    op.execute(
        sa.text(
            "UPDATE agent_sessions SET graph_checkpoint = '{}' "
            "WHERE status = 'active'"
        )
    )
    op.drop_column("agent_sessions", "focus_section")


def downgrade() -> None:
    op.add_column(
        "agent_sessions",
        sa.Column("focus_section", sa.String(length=100), nullable=True),
    )
    op.drop_index("ix_agent_sessions_focused_artifact_id", table_name="agent_sessions")
    op.drop_constraint(
        "fk_agent_sessions_focused_artifact_id",
        "agent_sessions",
        type_="foreignkey",
    )
    op.drop_column("agent_sessions", "focused_artifact_id")
