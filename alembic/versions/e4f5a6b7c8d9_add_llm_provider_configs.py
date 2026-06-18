"""add llm provider configs

Revision ID: e4f5a6b7c8d9
Revises: e3f4a5b6c7d8
Create Date: 2026-06-19

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "e4f5a6b7c8d9"
down_revision: Union[str, None] = "e3f4a5b6c7d8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(sa.text("DROP TYPE IF EXISTS providertype CASCADE"))
    op.execute(sa.text("DROP TYPE IF EXISTS llmproviderstatus CASCADE"))
    op.execute(sa.text("CREATE TYPE providertype AS ENUM ('openai_compatible', 'azure_openai', 'anthropic', 'bedrock', 'gemini')"))
    op.execute(sa.text("CREATE TYPE llmproviderstatus AS ENUM ('draft', 'active', 'error', 'disabled')"))
    op.create_table(
        "llm_provider_configs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("org_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("provider_type", postgresql.ENUM(name="providertype", create_type=False), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("base_url", sa.String(512), nullable=True),
        sa.Column("model_name", sa.String(255), nullable=True),
        sa.Column("secret_ref", sa.String(255), nullable=True),
        sa.Column("encrypted_api_key", sa.Text(), nullable=True),
        sa.Column("status", postgresql.ENUM(name="llmproviderstatus", create_type=False), nullable=False),
        sa.Column("is_default", sa.Boolean(), nullable=False, server_default=sa.text("FALSE")),
        sa.Column("last_checked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_check_error", sa.Text(), nullable=True),
        sa.CheckConstraint("(org_id IS NULL) != (project_id IS NULL)", name="ck_llm_provider_scope_xor"),
        sa.CheckConstraint("(secret_ref IS NULL) != (encrypted_api_key IS NULL)", name="ck_llm_provider_secret_xor"),
        sa.ForeignKeyConstraint(["org_id"], ["organizations.id"]),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_llm_provider_configs_org_id", "llm_provider_configs", ["org_id"])
    op.create_index("ix_llm_provider_configs_project_id", "llm_provider_configs", ["project_id"])
    op.create_index("ix_llm_provider_configs_provider_type", "llm_provider_configs", ["provider_type"])
    op.create_index("ix_llm_provider_configs_status", "llm_provider_configs", ["status"])
    op.create_index("ix_llm_provider_configs_is_default", "llm_provider_configs", ["is_default"])
    op.create_index(
        "uq_llm_provider_default_project",
        "llm_provider_configs",
        ["project_id"],
        unique=True,
        postgresql_where=sa.text("project_id IS NOT NULL AND is_default = TRUE"),
    )
    op.create_index(
        "uq_llm_provider_default_org",
        "llm_provider_configs",
        ["org_id"],
        unique=True,
        postgresql_where=sa.text("org_id IS NOT NULL AND is_default = TRUE"),
    )


def downgrade() -> None:
    op.drop_table("llm_provider_configs")
    op.execute(sa.text("DROP TYPE IF EXISTS llmproviderstatus"))
    op.execute(sa.text("DROP TYPE IF EXISTS providertype"))
