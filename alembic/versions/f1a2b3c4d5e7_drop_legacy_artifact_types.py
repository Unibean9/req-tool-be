"""drop legacy artifact types

Revision ID: f1a2b3c4d5e7
Revises: f0a1b2c3d4e6
Create Date: 2026-06-23

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "f1a2b3c4d5e7"
down_revision: Union[str, None] = "f0a1b2c3d4e6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

NEW_VALUES = (
    "requirements",
    "domain_entity",
    "functional_requirement",
    "non_functional_requirement",
    "use_case",
    "epic",
    "story",
    "acceptance_criteria",
)

LEGACY_REQUIREMENTS_VALUES = (
    "research_output",
    "intent",
    "problem",
    "goal",
    "stakeholder",
    "capability",
    "business_rule",
    "constraint",
    "assumption",
    "risk",
    "open_question",
)


def upgrade() -> None:
    op.drop_index("ix_artifacts_nfr_category", table_name="artifacts")
    op.drop_index("ix_artifacts_stakeholder_role", table_name="artifacts")
    op.drop_column("artifacts", "nfr_category")
    op.drop_column("artifacts", "stakeholder_role")

    values_sql = ", ".join(f"'{value}'" for value in NEW_VALUES)
    legacy_values_sql = ", ".join(f"'{value}'" for value in LEGACY_REQUIREMENTS_VALUES)
    duplicate_projects = op.get_bind().execute(
        sa.text(
            "SELECT project_id "
            "FROM artifacts "
            f"WHERE type::text IN ({legacy_values_sql}) "
            "GROUP BY project_id "
            "HAVING COUNT(*) > 1"
        )
    ).fetchall()
    if duplicate_projects:
        project_ids = ", ".join(str(row.project_id) for row in duplicate_projects)
        raise RuntimeError(
            "Không thể tự động gộp nhiều artifact legacy thành một requirements artifact "
            f"cho project: {project_ids}"
        )

    op.execute(sa.text("ALTER TYPE artifacttype RENAME TO artifacttype_old"))
    op.execute(sa.text(f"CREATE TYPE artifacttype AS ENUM ({values_sql})"))
    op.execute(
        sa.text(
            "ALTER TABLE artifacts "
            "ALTER COLUMN type TYPE artifacttype "
            f"USING (CASE WHEN type::text IN ({legacy_values_sql}) "
            "THEN 'requirements' ELSE type::text END)::artifacttype"
        )
    )
    op.execute(sa.text("DROP TYPE artifacttype_old"))

    # 'requirements' is a CREATE TYPE member of the new type (not an ADD VALUE), so it is usable in
    # this same transaction. This partial unique index is the race guard for find-or-create — one
    # requirements artifact per project.
    op.create_index(
        "uq_artifacts_project_requirements",
        "artifacts",
        ["project_id"],
        unique=True,
        postgresql_where=sa.text("type = 'requirements'"),
        sqlite_where=sa.text("type = 'requirements'"),
    )


def downgrade() -> None:
    raise NotImplementedError("Dropping legacy artifact types is destructive and not safely reversible")
