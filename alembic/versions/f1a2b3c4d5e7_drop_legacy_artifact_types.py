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


def upgrade() -> None:
    op.drop_index("ix_artifacts_nfr_category", table_name="artifacts")
    op.drop_index("ix_artifacts_stakeholder_role", table_name="artifacts")
    op.drop_column("artifacts", "nfr_category")
    op.drop_column("artifacts", "stakeholder_role")

    values_sql = ", ".join(f"'{value}'" for value in NEW_VALUES)
    # Any row holding a to-be-dropped legacy type would make the USING cast below fail with
    # "invalid input value for enum". The artifacts table is greenfield in this deployment, so this
    # is a no-op here; it runs defensively so the cast is safe on any DB (this migration is already
    # destructive and irreversible by design).
    op.execute(sa.text(f"DELETE FROM artifacts WHERE type::text NOT IN ({values_sql})"))
    op.execute(sa.text("ALTER TYPE artifacttype RENAME TO artifacttype_old"))
    op.execute(sa.text(f"CREATE TYPE artifacttype AS ENUM ({values_sql})"))
    op.execute(
        sa.text(
            "ALTER TABLE artifacts "
            "ALTER COLUMN type TYPE artifacttype "
            "USING type::text::artifacttype"
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
