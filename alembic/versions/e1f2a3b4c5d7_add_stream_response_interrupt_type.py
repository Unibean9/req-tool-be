"""add stream_response to agentsessioninterrupttype enum

Revision ID: e1f2a3b4c5d7
Revises: d9e8f7a6b5c4
Create Date: 2026-06-25

Deploy order: run migration before deploying code that sets STREAM_RESPONSE.
Rollback: ALTER TYPE ... ADD VALUE is non-transactional on PG < 14; downgrade is intentionally
a no-op — the value stays in the enum but no code will write it after rollback.
"""

from typing import Sequence, Union

from alembic import op

revision: str = "e1f2a3b4c5d7"
down_revision: Union[str, None] = "d9e8f7a6b5c4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TYPE agentsessioninterrupttype ADD VALUE IF NOT EXISTS 'stream_response'")


def downgrade() -> None:
    # ALTER TYPE ... DROP VALUE does not exist in PostgreSQL; leaving this as a no-op is the
    # only safe choice. The value remains in the DB enum but no application code will write it.
    pass
