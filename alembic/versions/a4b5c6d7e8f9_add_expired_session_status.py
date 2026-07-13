"""add expired to agentsessionstatus enum

Revision ID: a4b5c6d7e8f9
Revises: 1c77f288f547
Create Date: 2026-07-11

Deploy order: run migration before deploying code that sets EXPIRED.
Rollback: ALTER TYPE ... ADD VALUE is non-transactional on PG < 14; downgrade is intentionally
a no-op — the value stays in the enum but no code will write it after rollback. Like the
turn_failed precedent, an EXPIRED session can persist indefinitely, so if the *code* is rolled
back while rows sit in 'expired', deserializing that value into the older Python enum fails on
session load. Before rolling back the code, run
`UPDATE agent_sessions SET status='failed' WHERE status='expired'`, or explicitly accept that
risk of failed session loads until those rows are cleared.
"""

from typing import Sequence, Union

from alembic import op

revision: str = "a4b5c6d7e8f9"
down_revision: Union[str, None] = "1c77f288f547"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TYPE agentsessionstatus ADD VALUE IF NOT EXISTS 'expired'")


def downgrade() -> None:
    # ALTER TYPE ... DROP VALUE does not exist in PostgreSQL; leaving this as a no-op is the
    # only safe choice. The value remains in the DB enum but no application code will write it.
    pass
