"""add turn_failed to agentsessionstatus enum

Revision ID: 1c77f288f547
Revises: e8f9a0b1c2d3
Create Date: 2026-07-08

Deploy order: run migration before deploying code that sets TURN_FAILED.
Rollback: ALTER TYPE ... ADD VALUE is non-transactional on PG < 14; downgrade is intentionally
a no-op — the value stays in the enum but no code will write it after rollback. Unlike the
stream_response precedent, a TURN_FAILED session can persist indefinitely (it is a resumable
resting state, not a transient interrupt), so if the *code* is rolled back while rows sit in
'turn_failed', deserializing that value into the older Python enum fails on session load. Before
rolling back the code, run `UPDATE agent_sessions SET status='failed' WHERE status='turn_failed'`,
or explicitly accept that risk of failed session loads until those rows are cleared.
"""

from typing import Sequence, Union

from alembic import op

revision: str = "1c77f288f547"
down_revision: Union[str, None] = "e8f9a0b1c2d3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TYPE agentsessionstatus ADD VALUE IF NOT EXISTS 'turn_failed'")


def downgrade() -> None:
    # ALTER TYPE ... DROP VALUE does not exist in PostgreSQL; leaving this as a no-op is the
    # only safe choice. The value remains in the DB enum but no application code will write it.
    pass
