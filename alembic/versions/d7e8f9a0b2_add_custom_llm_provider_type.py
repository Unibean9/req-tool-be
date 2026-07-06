"""replace proxy llm provider types with custom

Revision ID: d7e8f9a0b2
Revises: c6d7e8f9a0b1
Create Date: 2026-07-05 12:45:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "d7e8f9a0b2"
down_revision: str | Sequence[str] | None = "c6d7e8f9a0b1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        sa.text(
            """
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1
                    FROM llm_provider_configs
                    WHERE provider_type::text IN ('deepseek', 'openrouter')
                ) THEN
                    RAISE EXCEPTION
                        'Cannot remove legacy LLM provider enum values while deepseek/openrouter rows exist';
                END IF;
            END $$;
            """
        )
    )
    op.execute(sa.text("ALTER TYPE providertype RENAME TO providertype_old"))
    op.execute(
        sa.text(
            "CREATE TYPE providertype AS ENUM "
            "('bedrock', 'openai', 'google', 'anthropic', 'mistral', 'custom')"
        )
    )
    op.execute(
        sa.text(
            "ALTER TABLE llm_provider_configs "
            "ALTER COLUMN provider_type TYPE providertype "
            "USING provider_type::text::providertype"
        )
    )
    op.execute(sa.text("DROP TYPE providertype_old"))


def downgrade() -> None:
    op.execute(sa.text("ALTER TYPE providertype RENAME TO providertype_new"))
    op.execute(
        sa.text(
            "CREATE TYPE providertype AS ENUM "
            "('bedrock', 'openai', 'google', 'anthropic', 'deepseek', 'mistral', 'openrouter')"
        )
    )
    op.execute(
        sa.text(
            "ALTER TABLE llm_provider_configs "
            "ALTER COLUMN provider_type TYPE providertype "
            "USING provider_type::text::providertype"
        )
    )
    op.execute(sa.text("DROP TYPE providertype_new"))
