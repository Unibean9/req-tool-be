"""them bang artifact repository

Revision ID: d1e2f3a4b5c6
Revises: c0d1e2f3a4b5
Create Date: 2026-06-18

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "d1e2f3a4b5c6"
down_revision: Union[str, None] = "c0d1e2f3a4b5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


ENUMS = {
    "artifactstatus": ("draft", "needs_clarification", "accepted", "rejected", "archived"),
    "artifactpriority": ("must", "should", "could", "wont"),
    "artifacttype": (
        "research_output",
        "intent",
        "problem",
        "goal",
        "stakeholder",
        "capability",
        "domain_entity",
        "business_rule",
        "constraint",
        "assumption",
        "risk",
        "open_question",
        "functional_requirement",
        "non_functional_requirement",
        "use_case",
        "epic",
        "story",
        "acceptance_criteria",
    ),
    "changesource": ("manual", "import", "ai_generation", "tool_call", "system"),
    "versionstatus": ("draft", "proposed", "accepted", "rejected", "archived"),
    "sourcetype": ("markdown_upload", "text_paste", "repo_file", "url", "external_doc"),
    "relationtype": (
        "derives_from",
        "decomposes_to",
        "satisfies",
        "supports",
        "depends_on",
        "blocks",
        "conflicts_with",
        "clarifies",
        "constrains",
        "informs",
        "validates",
    ),
    "evidencesourcetype": ("chat", "repo_file", "document", "url", "user_input", "ai_output"),
    "reviewstatus": ("approved", "rejected", "changes_requested"),
    "workflowrunstatus": ("draft", "active", "completed", "archived"),
    "workflowstepkey": (
        "intent_vision",
        "capability_map",
        "domain_model",
        "requirements_spec",
        "realization_backlog",
    ),
    "workflowstepphase": ("brd", "srs", "delivery"),
    "workflowstepstatus": (
        "pending",
        "ready",
        "in_progress",
        "waiting_review",
        "approved",
        "rejected",
        "skipped",
    ),
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
        "source_documents",
        *audit_columns(),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("uploaded_by_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("source_type", pg_enum("sourcetype"), nullable=False),
        sa.Column("locator", sa.String(512), nullable=True),
        sa.Column("content_text", sa.Text(), nullable=True),
        sa.Column("content_hash", sa.String(128), nullable=True),
        sa.Column("mime_type", sa.String(255), nullable=True),
        sa.Column("size_bytes", sa.Integer(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
        sa.ForeignKeyConstraint(["uploaded_by_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_source_documents_project_id", "source_documents", ["project_id"])
    op.create_index("ix_source_documents_uploaded_by_id", "source_documents", ["uploaded_by_id"])

    op.create_table(
        "workflow_runs",
        *audit_columns(),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("status", pg_enum("workflowrunstatus"), nullable=False),
        sa.Column("current_step_key", pg_enum("workflowstepkey"), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("created_by_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
        sa.ForeignKeyConstraint(["created_by_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_workflow_runs_project_id", "workflow_runs", ["project_id"])
    op.create_index("ix_workflow_runs_created_by_id", "workflow_runs", ["created_by_id"])

    op.create_table(
        "artifacts",
        *audit_columns(),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("step_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("current_version_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("type", pg_enum("artifacttype"), nullable=False),
        sa.Column("status", pg_enum("artifactstatus"), nullable=False),
        sa.Column("priority", pg_enum("artifactpriority"), nullable=True),
        sa.Column("code", sa.String(100), nullable=True),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("confidence", sa.Numeric(5, 2), nullable=True),
        sa.Column("nfr_category", sa.String(100), nullable=True),
        sa.Column("stakeholder_role", sa.String(100), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("created_by_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
        sa.ForeignKeyConstraint(["run_id"], ["workflow_runs.id"]),
        sa.ForeignKeyConstraint(["created_by_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_artifacts_project_id", "artifacts", ["project_id"])
    op.create_index("ix_artifacts_run_id", "artifacts", ["run_id"])
    op.create_index("ix_artifacts_step_id", "artifacts", ["step_id"])
    op.create_index("ix_artifacts_type", "artifacts", ["type"])
    op.create_index("ix_artifacts_status", "artifacts", ["status"])
    op.create_index("ix_artifacts_priority", "artifacts", ["priority"])
    op.create_index("ix_artifacts_code", "artifacts", ["code"])
    op.create_index("ix_artifacts_nfr_category", "artifacts", ["nfr_category"])
    op.create_index("ix_artifacts_stakeholder_role", "artifacts", ["stakeholder_role"])
    op.create_index("ix_artifacts_created_by_id", "artifacts", ["created_by_id"])

    op.create_table(
        "artifact_versions",
        *audit_columns(),
        sa.Column("artifact_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("version_number", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("change_summary", sa.Text(), nullable=True),
        sa.Column("status", pg_enum("versionstatus"), nullable=False),
        sa.Column("change_source", pg_enum("changesource"), nullable=False),
        sa.Column("parent_version_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("created_by_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("source_document_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("agent_run_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("tool_call_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("provider_config_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("model_name", sa.String(255), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.ForeignKeyConstraint(["artifact_id"], ["artifacts.id"]),
        sa.ForeignKeyConstraint(["parent_version_id"], ["artifact_versions.id"]),
        sa.ForeignKeyConstraint(["created_by_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["source_document_id"], ["source_documents.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("artifact_id", "version_number", name="uq_artifact_versions_artifact_version"),
    )
    op.create_index("ix_artifact_versions_artifact_id", "artifact_versions", ["artifact_id"])
    op.create_index("ix_artifact_versions_parent_version_id", "artifact_versions", ["parent_version_id"])
    op.create_index("ix_artifact_versions_created_by_id", "artifact_versions", ["created_by_id"])
    op.create_index("ix_artifact_versions_source_document_id", "artifact_versions", ["source_document_id"])
    op.create_index("ix_artifact_versions_agent_run_id", "artifact_versions", ["agent_run_id"])
    op.create_index("ix_artifact_versions_tool_call_id", "artifact_versions", ["tool_call_id"])
    op.create_index("ix_artifact_versions_provider_config_id", "artifact_versions", ["provider_config_id"])
    op.create_foreign_key(
        "fk_artifact_current_version",
        "artifacts",
        "artifact_versions",
        ["current_version_id"],
        ["id"],
    )

    op.create_table(
        "artifact_links",
        *audit_columns(),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_artifact_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("target_artifact_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("relation_type", pg_enum("relationtype"), nullable=False),
        sa.Column("created_by_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
        sa.ForeignKeyConstraint(["source_artifact_id"], ["artifacts.id"]),
        sa.ForeignKeyConstraint(["target_artifact_id"], ["artifacts.id"]),
        sa.ForeignKeyConstraint(["created_by_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("source_artifact_id", "target_artifact_id", "relation_type", name="uq_artifact_links_pair_type"),
    )
    op.create_index("ix_artifact_links_project_id", "artifact_links", ["project_id"])
    op.create_index("ix_artifact_links_source_artifact_id", "artifact_links", ["source_artifact_id"])
    op.create_index("ix_artifact_links_target_artifact_id", "artifact_links", ["target_artifact_id"])
    op.create_index("ix_artifact_links_relation_type", "artifact_links", ["relation_type"])
    op.create_index("ix_artifact_links_created_by_id", "artifact_links", ["created_by_id"])

    op.create_table(
        "artifact_evidence",
        *audit_columns(),
        sa.Column("artifact_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("artifact_version_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("source_document_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("source_type", pg_enum("evidencesourcetype"), nullable=False),
        sa.Column("locator", sa.String(255), nullable=False),
        sa.Column("excerpt", sa.Text(), nullable=True),
        sa.Column("confidence", sa.Numeric(5, 2), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.ForeignKeyConstraint(["artifact_id"], ["artifacts.id"]),
        sa.ForeignKeyConstraint(["artifact_version_id"], ["artifact_versions.id"]),
        sa.ForeignKeyConstraint(["source_document_id"], ["source_documents.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_artifact_evidence_artifact_id", "artifact_evidence", ["artifact_id"])
    op.create_index("ix_artifact_evidence_artifact_version_id", "artifact_evidence", ["artifact_version_id"])
    op.create_index("ix_artifact_evidence_source_document_id", "artifact_evidence", ["source_document_id"])

    op.create_table(
        "artifact_reviews",
        *audit_columns(),
        sa.Column("artifact_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("artifact_version_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("reviewed_by_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("review_status", pg_enum("reviewstatus"), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["artifact_id"], ["artifacts.id"]),
        sa.ForeignKeyConstraint(["artifact_version_id"], ["artifact_versions.id"]),
        sa.ForeignKeyConstraint(["reviewed_by_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_artifact_reviews_artifact_id", "artifact_reviews", ["artifact_id"])
    op.create_index("ix_artifact_reviews_artifact_version_id", "artifact_reviews", ["artifact_version_id"])
    op.create_index("ix_artifact_reviews_reviewed_by_id", "artifact_reviews", ["reviewed_by_id"])

    op.create_table(
        "workflow_steps",
        *audit_columns(),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("step_key", pg_enum("workflowstepkey"), nullable=False),
        sa.Column("phase", pg_enum("workflowstepphase"), nullable=False),
        sa.Column("status", pg_enum("workflowstepstatus"), nullable=False),
        sa.Column("input_snapshot", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("output_snapshot", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("approved_by_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["workflow_runs.id"]),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
        sa.ForeignKeyConstraint(["approved_by_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", "step_key", name="uq_workflow_steps_run_key"),
    )
    op.create_index("ix_workflow_steps_run_id", "workflow_steps", ["run_id"])
    op.create_index("ix_workflow_steps_project_id", "workflow_steps", ["project_id"])
    op.create_index("ix_workflow_steps_approved_by_id", "workflow_steps", ["approved_by_id"])
    op.create_foreign_key(
        "fk_artifact_step",
        "artifacts",
        "workflow_steps",
        ["step_id"],
        ["id"],
    )


def downgrade() -> None:
    op.drop_constraint("fk_artifact_step", "artifacts", type_="foreignkey")
    op.drop_table("workflow_steps")
    op.drop_table("artifact_reviews")
    op.drop_table("artifact_evidence")
    op.drop_table("artifact_links")
    op.drop_constraint("fk_artifact_current_version", "artifacts", type_="foreignkey")
    op.drop_table("artifact_versions")
    op.drop_table("artifacts")
    op.drop_table("workflow_runs")
    op.drop_table("source_documents")
    drop_enum_types()
