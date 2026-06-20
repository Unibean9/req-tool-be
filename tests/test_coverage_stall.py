"""Tests for the slot-coverage root-cause fix.

Two defects produced the verbatim-repeated question observed in production:
  - the slot directive listed opaque slot keys with no description/rubric, so the
    LLM graded every slot 'empty' forever -> coverage stuck at 0.0;
  - nothing detected the stall, so the deterministic coverage hint re-asked the
    same weakest slot every turn until the turn cap.

These tests lock the fix: the directive now carries descriptions + a grading
rubric + an instruction to credit the latest answer, and a stall counter both
relaxes the gate and rewrites the hint once coverage stops advancing.
"""

import uuid

import pytest
from sqlalchemy import select

from app.graphs.slot_schema import COVERAGE_STALL_LIMIT, SLOT_DESCRIPTIONS
from app.models.agent import AgentRun
from tests.conftest import TestSessionFactory
from tests.helpers import create_org, create_project, make_auth_headers
from tests.test_graph_nodes import _config, _make_agent_session, _session_factory, _state


# ---------------------------------------------------------------------------
# Fix 1 — slot directive carries meaning (descriptions + rubric + credit answer)
# ---------------------------------------------------------------------------

def test_slot_directive_lists_descriptions_and_rubric():
    from app.graphs.nodes import _build_slot_directive

    directive = _build_slot_directive(_state(artifact_type="intent"))

    # Each required slot must appear with its human description, not just the key.
    assert SLOT_DESCRIPTIONS["why_now"] in directive
    assert SLOT_DESCRIPTIONS["sponsor"] in directive
    # Explicit grading rubric so the model stops defaulting everything to 'empty'.
    assert "filled" in directive and "partial" in directive and "empty" in directive
    # Must credit the latest user answer instead of re-grading it 'empty'.
    assert "mới nhất" in directive


def test_slot_directive_empty_for_non_brd():
    from app.graphs.nodes import _build_slot_directive

    assert _build_slot_directive(_state(artifact_type="functional_requirement")) == ""


# ---------------------------------------------------------------------------
# Fix 2 — stall detection relaxes the gate and rewrites the hint
# ---------------------------------------------------------------------------

def test_route_gate_blocks_below_stall_limit():
    from app.graphs.nodes import route_node

    state = _state(artifact_type="intent", turn_count=2, analysis_result={"next_action": "propose"})
    state["coverage_complete"] = False
    state["coverage_stall_count"] = 0

    assert route_node(state) == "ask_human"


def test_route_gate_relaxes_after_stall():
    from app.graphs.nodes import route_node

    state = _state(artifact_type="intent", turn_count=2, analysis_result={"next_action": "propose"})
    state["coverage_complete"] = False
    state["coverage_stall_count"] = COVERAGE_STALL_LIMIT

    assert route_node(state) == "confirm"


def test_coverage_hint_breaks_loop_on_stall():
    from app.graphs.nodes import _build_coverage_hint

    state = _state(artifact_type="intent")
    state["coverage_complete"] = False
    state["slot_coverage"] = {
        "why_now": "empty",
        "sponsor": "empty",
        "expected_outcome": "empty",
        "success_state": "empty",
    }
    state["coverage_stall_count"] = COVERAGE_STALL_LIMIT

    hint = _build_coverage_hint(state)

    # Stall hint must steer AWAY from re-asking the same slot.
    assert "không tăng" in hint
    assert "propose" in hint


def test_coverage_hint_normal_below_stall():
    """Below the stall limit the hint still pins the weakest slot (existing behaviour)."""
    from app.graphs.nodes import _build_coverage_hint

    state = _state(artifact_type="intent")
    state["coverage_complete"] = False
    state["slot_coverage"] = {"why_now": "empty", "sponsor": "empty"}
    state["coverage_stall_count"] = 0

    hint = _build_coverage_hint(state)

    assert "không tăng" not in hint
    assert SLOT_DESCRIPTIONS["why_now"] in hint


def test_coverage_hint_rotates_off_last_asked_slot():
    """Root-cause regression: re-pinning the slot we just asked reproduced the question
    verbatim. The hint must skip last_asked_slot and advance to the next weak slot."""
    from app.graphs.nodes import _build_coverage_hint

    state = _state(artifact_type="intent")
    state["coverage_complete"] = False
    state["coverage_stall_count"] = 0
    state["slot_coverage"] = {
        "why_now": "empty",
        "sponsor": "empty",
        "expected_outcome": "empty",
        "success_state": "empty",
    }
    state["last_asked_slot"] = "why_now"

    hint = _build_coverage_hint(state)

    # Must move on to the next slot, not re-ask the one already asked.
    assert SLOT_DESCRIPTIONS["sponsor"] in hint
    assert SLOT_DESCRIPTIONS["why_now"] not in hint


# ---------------------------------------------------------------------------
# Fix 2 — analyze_node maintains the stall counter
# ---------------------------------------------------------------------------

async def _run_analyze(client, db_session, slot_assessment, prev_ratio, prev_stall, last_asked=None):
    from app.graphs.nodes import analyze_node

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])
    agent_session = await _make_agent_session(client, db_session, project_id)

    mock_llm = _AsyncLLM(slot_assessment)
    state = _state(artifact_type="intent")
    state["coverage_ratio"] = prev_ratio
    state["coverage_stall_count"] = prev_stall
    state["last_asked_slot"] = last_asked
    config = _config(str(agent_session.id), str(project_id), mock_llm)
    config["configurable"]["session_factory"] = _session_factory()

    result = await analyze_node(state, config)
    return result, agent_session


class _AsyncLLM:
    def __init__(self, slot_assessment):
        self._slot_assessment = slot_assessment

    async def generate(self, **kwargs):
        return (
            {
                "next_action": "ask",
                "confidence": 0.3,
                "gaps": [],
                "message": "Câu hỏi tiếp theo?",
                "slot_assessment": self._slot_assessment,
            },
            None,
        )


@pytest.mark.asyncio
async def test_analyze_increments_stall_when_coverage_flat(client, db_session):
    all_empty = {"why_now": "empty", "sponsor": "empty", "expected_outcome": "empty", "success_state": "empty"}

    result, _ = await _run_analyze(client, db_session, all_empty, prev_ratio=0.0, prev_stall=0)

    assert result["coverage_ratio"] == 0.0
    assert result["coverage_stall_count"] == 1


@pytest.mark.asyncio
async def test_analyze_resets_stall_when_coverage_improves(client, db_session):
    one_filled = {"why_now": "filled", "sponsor": "empty", "expected_outcome": "empty", "success_state": "empty"}

    result, _ = await _run_analyze(client, db_session, one_filled, prev_ratio=0.0, prev_stall=2)

    assert result["coverage_ratio"] > 0.0
    assert result["coverage_stall_count"] == 0


@pytest.mark.asyncio
async def test_analyze_credits_last_asked_slot_when_regraded_empty(client, db_session):
    """Regression: the user answered the slot we just asked, but the model still grades it
    'empty'. analyze_node must credit it 'partial' so coverage advances and the next hint
    rotates on — otherwise the same question repeats."""
    all_empty = {"why_now": "empty", "sponsor": "empty", "expected_outcome": "empty", "success_state": "empty"}

    result, _ = await _run_analyze(
        client, db_session, all_empty, prev_ratio=0.0, prev_stall=0, last_asked="why_now"
    )

    assert result["slot_coverage"]["why_now"] == "partial"
    assert result["coverage_ratio"] == pytest.approx(0.125)


@pytest.mark.asyncio
async def test_analyze_stall_zero_when_no_slot_assessment(client, db_session):
    """No slot_assessment -> fail-open coverage -> stall counter resets to 0."""
    result, _ = await _run_analyze(client, db_session, None, prev_ratio=0.0, prev_stall=2)

    assert result["coverage_complete"] is None
    assert result["coverage_stall_count"] == 0
