"""Phase 5 — a resumed legacy checkpoint's state-field assumptions/open-questions migrate to nodes.

Only fires when the keys are still present in the state dict (LangGraph drops a channel removed from
the schema, so on a real resume this is a no-op — documented in evidence/phase-05-divergence-audit.md).
This exercises the migration contract at the node boundary.
"""

import pytest

from app.graphs.decision_graph import derive_assumptions, derive_open_questions
from app.graphs.nodes import orchestrator_node


@pytest.mark.asyncio
async def test_orchestrator_migrates_legacy_state_fields_into_decision_nodes():
    state = {
        "decision_nodes": {},
        "artifact_type": "brd",
        "assumptions": [{"statement": "users have smartphones", "source": "model"}],
        "open_questions": [{"question": "which region launches first?", "domain": "scope"}],
    }

    update = await orchestrator_node(state, {})

    nodes = update["decision_nodes"]
    assert derive_assumptions(nodes) == [{"statement": "users have smartphones", "status": "needs_confirmation"}]
    assert derive_open_questions(nodes) == [{"question": "which region launches first?", "status": "proposed"}]


@pytest.mark.asyncio
async def test_orchestrator_no_migration_update_without_legacy_fields():
    state = {"decision_nodes": {}, "artifact_type": "brd"}

    update = await orchestrator_node(state, {})

    # No legacy fields -> orchestrator does not emit a decision_nodes update from migration.
    assert "decision_nodes" not in update
