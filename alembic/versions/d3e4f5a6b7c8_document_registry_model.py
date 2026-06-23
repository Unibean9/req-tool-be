"""document registry parent-child artifact model

Revision ID: d3e4f5a6b7c8
Revises: f2a3b4c5d6e8, f0a1b2c3d4e5, a318c2b350cc
Create Date: 2026-06-23

This pre-production migration intentionally does not backfill existing artifact rows.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "d3e4f5a6b7c8"
down_revision: Union[str, tuple[str, ...], None] = (
    "f2a3b4c5d6e8",
    "f0a1b2c3d4e5",
    "a318c2b350cc",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

NEW_ENUM_VALUES = (
    "vision_objectives",
    "problem_statement",
    "stakeholder_register",
    "scope_capabilities",
    "business_rules",
    "constraints_assumptions",
    "risks_issues",
    "prd",
    "sad",
    "component",
    "interface",
    "tech_decision",
)


def upgrade() -> None:
    bind = op.get_bind()
    op.execute(sa.text("DROP INDEX IF EXISTS uq_artifacts_project_requirements"))

    if bind.dialect.name == "postgresql":
        with op.get_context().autocommit_block():
            for value in NEW_ENUM_VALUES:
                op.execute(sa.text(f"ALTER TYPE artifacttype ADD VALUE IF NOT EXISTS '{value}'"))
            op.execute(sa.text("ALTER TYPE artifacttype RENAME VALUE 'requirements' TO 'brd'"))

    op.add_column(
        "artifacts",
        sa.Column("parent_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_artifacts_parent_id",
        "artifacts",
        "artifacts",
        ["parent_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_index("ix_artifacts_parent_id", "artifacts", ["parent_id"])

    for container_type in ("brd", "prd", "sad"):
        op.create_index(
            f"uq_artifacts_project_{container_type}",
            "artifacts",
            ["project_id"],
            unique=True,
            postgresql_where=sa.text(f"type = '{container_type}'"),
            sqlite_where=sa.text(f"type = '{container_type}'"),
        )

    op.execute(
        sa.text(
            "UPDATE agent_sessions SET artifact_type = 'brd' "
            "WHERE artifact_type = 'requirements'"
        )
    )


def downgrade() -> None:
    for container_type in ("sad", "prd", "brd"):
        op.drop_index(f"uq_artifacts_project_{container_type}", table_name="artifacts")
    op.drop_index("ix_artifacts_parent_id", table_name="artifacts")
    op.drop_constraint("fk_artifacts_parent_id", "artifacts", type_="foreignkey")
    op.drop_column("artifacts", "parent_id")
    raise NotImplementedError(
        "Artifact enum rename/additions are intentionally not reversed; recreate the pre-production database"
    )
