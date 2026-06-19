"""add agent tables

Revision ID: f6a7b8c9d0e1
Revises: e4f5a6b7c8d9
Create Date: 2026-06-19

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "f6a7b8c9d0e1"
down_revision: Union[str, None] = "e4f5a6b7c8d9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


ENUMS = {
    "agentsessionstatus": ("active", "waiting_for_human", "completed", "failed"),
    "agentsessioninterrupttype": ("ask_human", "propose_artifacts"),
    "agentmessagerole": ("agent", "user"),
    "agenttoolcallstatus": ("proposed", "approved", "rejected", "executed", "superseded"),
}


def pg_enum(name: str) -> postgresql.ENUM:
    return postgresql.ENUM(*ENUMS[name], name=name, create_type=False)


def create_enum_types() -> None:
    for name, values in ENUMS.items():
        op.execute(sa.text(f'DROP TYPE IF EXISTS "{name}" CASCADE'))
        quoted_values = ", ".join(f"'{value}'" for value in values)
        op.execute(sa.text(f'CREATE TYPE "{name}" AS ENUM ({quoted_values})'))


def drop_enum_types() -> None:
    for name in reversed(ENUMS):
        op.execute(sa.text(f'DROP TYPE IF EXISTS "{name}" CASCADE'))


def audit_columns() -> list[sa.Column]:
    return [
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    ]


def upgrade() -> None:
    create_enum_types()

    op.create_table(
        "agent_sessions",
        *audit_columns(),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("artifact_type", sa.String(100), nullable=False),
        sa.Column("workflow_area", sa.String(50), nullable=False),
        sa.Column("step_key", sa.String(100), nullable=True),
        sa.Column("status", pg_enum("agentsessionstatus"), nullable=False),
        sa.Column("interrupt_type", pg_enum("agentsessioninterrupttype"), nullable=True),
        sa.Column("missing_context", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("graph_checkpoint", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("provider_config_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("created_by_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
        sa.ForeignKeyConstraint(["provider_config_id"], ["llm_provider_configs.id"]),
        sa.ForeignKeyConstraint(["created_by_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_agent_sessions_project_id", "agent_sessions", ["project_id"])
    op.create_index("ix_agent_sessions_artifact_type", "agent_sessions", ["artifact_type"])
    op.create_index("ix_agent_sessions_workflow_area", "agent_sessions", ["workflow_area"])
    op.create_index("ix_agent_sessions_status", "agent_sessions", ["status"])
    op.create_index("ix_agent_sessions_provider_config_id", "agent_sessions", ["provider_config_id"])
    op.create_index("ix_agent_sessions_created_by_id", "agent_sessions", ["created_by_id"])
    op.create_index(
        "uq_agent_sessions_active_project_artifact_type",
        "agent_sessions",
        ["project_id", "artifact_type"],
        unique=True,
        postgresql_where=sa.text("status IN ('active', 'waiting_for_human')"),
    )

    op.create_table(
        "agent_messages",
        *audit_columns(),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("role", pg_enum("agentmessagerole"), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["agent_sessions.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_agent_messages_session_id", "agent_messages", ["session_id"])
    op.create_index("ix_agent_messages_role", "agent_messages", ["role"])

    op.create_table(
        "agent_runs",
        *audit_columns(),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("analysis_result", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("provider_config_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("model_name", sa.String(255), nullable=True),
        sa.ForeignKeyConstraint(["session_id"], ["agent_sessions.id"]),
        sa.ForeignKeyConstraint(["provider_config_id"], ["llm_provider_configs.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_agent_runs_session_id", "agent_runs", ["session_id"])
    op.create_index("ix_agent_runs_provider_config_id", "agent_runs", ["provider_config_id"])

    op.create_table(
        "agent_tool_calls",
        *audit_columns(),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("tool_name", sa.String(100), nullable=False),
        sa.Column("input_snapshot", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("status", pg_enum("agenttoolcallstatus"), nullable=False),
        sa.Column("created_artifact_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("created_version_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["run_id"], ["agent_runs.id"]),
        sa.ForeignKeyConstraint(["created_artifact_id"], ["artifacts.id"]),
        sa.ForeignKeyConstraint(["created_version_id"], ["artifact_versions.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_agent_tool_calls_run_id", "agent_tool_calls", ["run_id"])
    op.create_index("ix_agent_tool_calls_status", "agent_tool_calls", ["status"])
    op.create_index("ix_agent_tool_calls_created_artifact_id", "agent_tool_calls", ["created_artifact_id"])
    op.create_index("ix_agent_tool_calls_created_version_id", "agent_tool_calls", ["created_version_id"])

    op.create_foreign_key(
        "fk_artifact_versions_agent_run",
        "artifact_versions",
        "agent_runs",
        ["agent_run_id"],
        ["id"],
    )
    op.create_foreign_key(
        "fk_artifact_versions_tool_call",
        "artifact_versions",
        "agent_tool_calls",
        ["tool_call_id"],
        ["id"],
    )
    op.create_foreign_key(
        "fk_artifact_versions_provider_config",
        "artifact_versions",
        "llm_provider_configs",
        ["provider_config_id"],
        ["id"],
    )


def downgrade() -> None:
    op.drop_constraint("fk_artifact_versions_provider_config", "artifact_versions", type_="foreignkey")
    op.drop_constraint("fk_artifact_versions_tool_call", "artifact_versions", type_="foreignkey")
    op.drop_constraint("fk_artifact_versions_agent_run", "artifact_versions", type_="foreignkey")
    op.drop_table("agent_tool_calls")
    op.drop_table("agent_runs")
    op.drop_table("agent_messages")
    op.drop_table("agent_sessions")
    drop_enum_types()
