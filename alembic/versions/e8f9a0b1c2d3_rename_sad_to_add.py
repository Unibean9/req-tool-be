"""rename sad artifact type to add

Revision ID: e8f9a0b1c2d3
Revises: d7e8f9a0b2
Create Date: 2026-07-06 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "e8f9a0b1c2d3"
down_revision: str | Sequence[str] | None = "d7e8f9a0b2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_ARTIFACT_TYPES_WITH_ADD = (
    "brd",
    "prd",
    "add",
    "executive_summary",
    "vision_objectives",
    "problem_statement",
    "stakeholder_register",
    "scope_capabilities",
    "business_rules",
    "constraints_assumptions",
    "risks_issues",
    "domain_entity",
    "functional_requirement",
    "non_functional_requirement",
    "use_case",
    "component",
    "interface",
    "tech_decision",
    "tech_stack",
    "event_storming",
    "domain_event",
    "actor_command",
    "policy",
    "aggregate",
    "epic",
    "story",
    "acceptance_criteria",
)

_ARTIFACT_TYPES_WITH_SAD = tuple("sad" if item == "add" else item for item in _ARTIFACT_TYPES_WITH_ADD)


def _create_artifact_type(values: tuple[str, ...]) -> None:
    quoted = ", ".join(f"'{value}'" for value in values)
    op.execute(sa.text(f"CREATE TYPE artifacttype AS ENUM ({quoted})"))


def upgrade() -> None:
    op.drop_index("uq_artifacts_project_sad", table_name="artifacts")
    op.execute(sa.text("ALTER TYPE artifacttype RENAME TO artifacttype_old"))
    _create_artifact_type(_ARTIFACT_TYPES_WITH_ADD)
    op.execute(
        sa.text(
            "ALTER TABLE artifacts "
            "ALTER COLUMN type TYPE artifacttype "
            "USING (CASE WHEN type::text = 'sad' THEN 'add' ELSE type::text END)::artifacttype"
        )
    )
    op.execute(sa.text("DROP TYPE artifacttype_old"))
    op.execute(sa.text("UPDATE agent_sessions SET artifact_type = 'add' WHERE artifact_type = 'sad'"))
    op.create_index(
        "uq_artifacts_project_add",
        "artifacts",
        ["project_id"],
        unique=True,
        postgresql_where=sa.text("type = 'add' AND status != 'archived'"),
        sqlite_where=sa.text("type = 'add' AND status != 'archived'"),
    )


def downgrade() -> None:
    op.drop_index("uq_artifacts_project_add", table_name="artifacts")
    op.execute(sa.text("ALTER TYPE artifacttype RENAME TO artifacttype_new"))
    _create_artifact_type(_ARTIFACT_TYPES_WITH_SAD)
    op.execute(
        sa.text(
            "ALTER TABLE artifacts "
            "ALTER COLUMN type TYPE artifacttype "
            "USING (CASE WHEN type::text = 'add' THEN 'sad' ELSE type::text END)::artifacttype"
        )
    )
    op.execute(sa.text("DROP TYPE artifacttype_new"))
    op.execute(sa.text("UPDATE agent_sessions SET artifact_type = 'sad' WHERE artifact_type = 'add'"))
    op.create_index(
        "uq_artifacts_project_sad",
        "artifacts",
        ["project_id"],
        unique=True,
        postgresql_where=sa.text("type = 'sad' AND status != 'archived'"),
        sqlite_where=sa.text("type = 'sad' AND status != 'archived'"),
    )
