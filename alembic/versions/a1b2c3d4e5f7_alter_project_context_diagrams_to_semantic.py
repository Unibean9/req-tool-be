"""alter project_context_diagrams to semantic model

Revision ID: a1b2c3d4e5f7
Revises: f0a1b2c3d4e5
Create Date: 2026-05-24

"""
from alembic import op
import sqlalchemy as sa

revision = "a1b2c3d4e5f7"
down_revision = "f0a1b2c3d4e5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint(
        "ck_project_context_diagrams_source",
        "project_context_diagrams",
        type_="check",
    )
    op.drop_column("project_context_diagrams", "nodes")
    op.drop_column("project_context_diagrams", "edges")
    op.drop_column("project_context_diagrams", "source")

    op.add_column(
        "project_context_diagrams",
        sa.Column("stakeholder_ids", sa.JSON(), nullable=False, server_default="[]"),
    )
    op.add_column(
        "project_context_diagrams",
        sa.Column("flows", sa.JSON(), nullable=False, server_default="[]"),
    )
    op.add_column(
        "project_context_diagrams",
        sa.Column("layout", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("project_context_diagrams", "layout")
    op.drop_column("project_context_diagrams", "flows")
    op.drop_column("project_context_diagrams", "stakeholder_ids")

    op.add_column(
        "project_context_diagrams",
        sa.Column("nodes", sa.JSON(), nullable=False, server_default="[]"),
    )
    op.add_column(
        "project_context_diagrams",
        sa.Column("edges", sa.JSON(), nullable=False, server_default="[]"),
    )
    op.add_column(
        "project_context_diagrams",
        sa.Column("source", sa.Text(), nullable=False, server_default="manual"),
    )
    op.create_check_constraint(
        "ck_project_context_diagrams_source",
        "project_context_diagrams",
        "source IN ('manual', 'generated', 'synced')",
    )
