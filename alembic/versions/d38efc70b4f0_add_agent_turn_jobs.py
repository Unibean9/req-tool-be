"""add agent turn jobs

Revision ID: d38efc70b4f0
Revises: c27dfb69a3e9
Create Date: 2026-07-15
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "d38efc70b4f0"
down_revision: Union[str, None] = "c27dfb69a3e9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    job_status = postgresql.ENUM(
        "queued",
        "claimed",
        "running",
        "succeeded",
        "failed",
        "dead_letter",
        name="agentturnjobstatus",
        create_type=False,
    )
    job_status.create(op.get_bind(), checkfirst=True)
    op.create_table(
        "agent_turn_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("turn_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("attempt", sa.Integer(), server_default="0", nullable=False),
        sa.Column("scheduled_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("status", job_status, nullable=False),
        sa.Column("lease_owner", sa.String(length=128), nullable=True),
        sa.Column("lease_generation", sa.Integer(), server_default="0", nullable=False),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("cohort", postgresql.JSONB(), server_default="{}", nullable=False),
        sa.Column("expected_transition_version", sa.Integer(), server_default="0", nullable=False),
        sa.ForeignKeyConstraint(["turn_id"], ["agent_turn_envelopes.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_unique_constraint("uq_agent_turn_jobs_turn_id", "agent_turn_jobs", ["turn_id"])
    op.create_index("ix_agent_turn_jobs_status", "agent_turn_jobs", ["status"])
    op.create_index("ix_agent_turn_jobs_lease_expires_at", "agent_turn_jobs", ["lease_expires_at"])


def downgrade() -> None:
    op.drop_index("ix_agent_turn_jobs_lease_expires_at", table_name="agent_turn_jobs")
    op.drop_index("ix_agent_turn_jobs_status", table_name="agent_turn_jobs")
    op.drop_constraint("uq_agent_turn_jobs_turn_id", "agent_turn_jobs", type_="unique")
    op.drop_table("agent_turn_jobs")
    postgresql.ENUM(name="agentturnjobstatus").drop(op.get_bind(), checkfirst=True)
