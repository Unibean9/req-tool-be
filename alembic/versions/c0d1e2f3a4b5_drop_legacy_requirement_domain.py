"""xoa bang legacy requirement domain

Revision ID: c0d1e2f3a4b5
Revises: b7c8d9e0f1a2
Create Date: 2026-06-18

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "c0d1e2f3a4b5"
down_revision: Union[str, None] = "b7c8d9e0f1a2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


LEGACY_TABLES = [
    "sync_logs",
    "sync_queue",
    "github_items",
    "project_context_diagrams",
    "project_out_of_scope",
    "project_business_requirements",
    "project_goal_objectives",
    "project_constraints",
    "project_flow_action_rules",
    "project_flow_actions",
    "story_estimates",
    "nfr_feature_links",
    "nfrs",
    "project_rules",
    "project_flows",
    "project_goals",
    "stakeholders",
    "acceptance_criteria",
    "tasks",
    "stories",
    "features",
    "epics",
    "close_reasons",
    "actors",
    "github_connections",
]

LEGACY_TYPES = [
    "sync_queue_status",
    "sync_operation",
    "sync_log_status",
    "constraintseverity",
    "constrainttype",
    "goalpriority",
    "ruletype",
    "nfrcategory",
    "influencelevel",
    "actortype",
    "close_reason_enum",
    "item_status",
    "item_type",
    "priority",
]


def upgrade() -> None:
    conn = op.get_bind()
    for table_name in LEGACY_TABLES:
        conn.execute(sa.text(f'DROP TABLE IF EXISTS "{table_name}" CASCADE'))
    for type_name in LEGACY_TYPES:
        conn.execute(sa.text(f'DROP TYPE IF EXISTS "{type_name}" CASCADE'))


def downgrade() -> None:
    raise NotImplementedError("Migration cleanup một chiều; dùng migration lịch sử trước đó để dựng lại domain cũ.")
