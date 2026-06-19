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
    op.execute(sa.text("CREATE TYPE providertype AS ENUM ('bedrock', 'openai', 'google', 'anthropic')"))
    op.execute(sa.text("CREATE TYPE llmproviderstatus AS ENUM ('draft', 'active', 'error', 'disabled')"))
    op.create_table(
        "llm_provider_configs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("provider_type", postgresql.ENUM(name="providertype", create_type=False), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("base_url", sa.String(512), nullable=True),
        sa.Column("region", sa.String(64), nullable=True),
        sa.Column("model_name", sa.String(255), nullable=True),
        sa.Column("encrypted_api_key", sa.Text(), nullable=True),
        sa.Column("encrypted_secret_key", sa.Text(), nullable=True),
        sa.Column("status", postgresql.ENUM(name="llmproviderstatus", create_type=False), nullable=False),
        sa.Column("is_default", sa.Boolean(), nullable=False, server_default=sa.text("FALSE")),
        sa.Column("last_checked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_check_error", sa.Text(), nullable=True),
        sa.CheckConstraint("encrypted_api_key IS NOT NULL", name="ck_llm_provider_api_key_required"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_llm_provider_configs_user_id", "llm_provider_configs", ["user_id"])
    op.create_index("ix_llm_provider_configs_provider_type", "llm_provider_configs", ["provider_type"])
    op.create_index("ix_llm_provider_configs_status", "llm_provider_configs", ["status"])
    op.create_index("ix_llm_provider_configs_is_default", "llm_provider_configs", ["is_default"])
    op.create_index(
        "uq_llm_provider_default_user",
        "llm_provider_configs",
        ["user_id"],
        unique=True,
        postgresql_where=sa.text("is_default = TRUE"),
    )


def downgrade() -> None:
    op.drop_table("llm_provider_configs")
    op.execute(sa.text("DROP TYPE IF EXISTS llmproviderstatus"))
    op.execute(sa.text("DROP TYPE IF EXISTS providertype"))
