"""rename swimlane to activity in project_flows

Revision ID: 916870f4ab8d
Revises: a1b2c3d4e5f7
Create Date: 2026-05-24 14:39:53.287513

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '916870f4ab8d'
down_revision: Union[str, None] = 'a1b2c3d4e5f7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE project_flows RENAME COLUMN swimlane TO activity")


def downgrade() -> None:
    op.execute("ALTER TABLE project_flows RENAME COLUMN activity TO swimlane")
