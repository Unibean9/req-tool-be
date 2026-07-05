"""singleton unique indexes exclude archived artifacts

Revision ID: c6d7e8f9a0b1
Revises: c5d6e7f8a9b0
Create Date: 2026-07-04

Soft-delete/retirement sets a singleton artifact (brd/prd/sad) to ARCHIVED while keeping its row.
The original partial unique indexes matched only on ``type``, so an archived singleton still
occupied the unique slot and blocked recreating that document type. Recreate each index with an
added ``status != 'archived'`` predicate so an archived singleton frees the slot.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "c6d7e8f9a0b1"
down_revision: Union[str, tuple[str, ...], None] = "c5d6e7f8a9b0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_CONTAINER_TYPES = ("brd", "prd", "sad")


def upgrade() -> None:
    for container_type in _CONTAINER_TYPES:
        op.drop_index(f"uq_artifacts_project_{container_type}", table_name="artifacts")
        op.create_index(
            f"uq_artifacts_project_{container_type}",
            "artifacts",
            ["project_id"],
            unique=True,
            postgresql_where=sa.text(f"type = '{container_type}' AND status != 'archived'"),
            sqlite_where=sa.text(f"type = '{container_type}' AND status != 'archived'"),
        )


def downgrade() -> None:
    for container_type in _CONTAINER_TYPES:
        op.drop_index(f"uq_artifacts_project_{container_type}", table_name="artifacts")
        op.create_index(
            f"uq_artifacts_project_{container_type}",
            "artifacts",
            ["project_id"],
            unique=True,
            postgresql_where=sa.text(f"type = '{container_type}'"),
            sqlite_where=sa.text(f"type = '{container_type}'"),
        )
