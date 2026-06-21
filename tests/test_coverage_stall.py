"""Tests for the slot-coverage root-cause fix (tool-loop world).

Two defects produced the verbatim-repeated question observed in production:
  - the slot directive listed opaque slot keys with no description/rubric, so the
    LLM graded every slot 'empty' forever -> coverage stuck at 0.0;
  - nothing detected the stall, so the deterministic coverage hint re-asked the
    same weakest slot every turn until the turn cap.

These tests lock the fix: the directive now carries descriptions + a grading
rubric + an instruction to credit the latest answer, and a stall counter rewrites
the hint once coverage stops advancing.

The agent is now a PURE LangGraph tool-loop: analyze_node always emits a tool
selection (TOOL_SELECTION_SCHEMA) and route_node no longer vetoes on coverage.
The coverage STALL COUNTER and HINT logic are unchanged and still tested here; the
removed route_node floor/stall ROUTING branch is no longer exercised.
"""

import uuid

import pytest

from app.graphs.slot_schema import COVERAGE_STALL_LIMIT, SLOT_DESCRIPTIONS
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


def test_slot_directive_is_rubric_not_script():
    """The directive is a reference rubric, not a sequential march.

    The old timing sentence forced 'only keep asking while slots are empty/partial; propose
    when most are filled' — that scaffolding is what produced robotic, off-focus questions.
    The directive must drop it and frame the slots as a reference rubric the LLM consults,
    letting the LLM choose ask->propose timing from full context.
    """
    from app.graphs.nodes import _build_slot_directive

    directive = _build_slot_directive(_state(artifact_type="intent"))

    # The hard sequential-timing sentence must be gone.
    assert "Chỉ tiếp tục hỏi khi còn slot" not in directive
    # Reference-rubric framing must be present.
    assert "rubric" in directive.lower() or "tham chiếu" in directive


def test_slot_directive_emit_mandate_unchanged():
    """compute_coverage still depends on the LLM emitting slot_assessment every turn."""
    from app.graphs.nodes import _build_slot_directive

    directive = _build_slot_directive(_state(artifact_type="intent"))

    assert "slot_assessment" in directive


# ---------------------------------------------------------------------------
# Fix 2 — stall detection rewrites the coverage hint
# ---------------------------------------------------------------------------

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
    # Gap-inventory lists ALL weak slots, not a single pinned one.
    assert SLOT_DESCRIPTIONS["why_now"] in hint
    assert SLOT_DESCRIPTIONS["sponsor"] in hint


def test_coverage_hint_lists_all_weak_slots():
    """The hint is a gap inventory, not a pinned single question.

    It must list every empty/partial slot with its description and invite the LLM to pick the
    angle that fits the conversation — instead of dictating 'next ask about X'.
    """
    from app.graphs.nodes import _build_coverage_hint

    state = _state(artifact_type="intent")
    state["coverage_complete"] = False
    state["coverage_stall_count"] = 0
    state["slot_coverage"] = {
        "why_now": "empty",
        "sponsor": "empty",
        "expected_outcome": "partial",
        "success_state": "empty",
    }

    hint = _build_coverage_hint(state)

    # Every weak slot appears with its human description.
    for slot in ("why_now", "sponsor", "expected_outcome", "success_state"):
        assert SLOT_DESCRIPTIONS[slot] in hint
    # The old pinned-question phrasing must be gone.
    assert "tiếp theo cần hỏi về" not in hint
    # LLM is invited to choose the angle, not marched through a checklist.
    assert "angle" in hint or "góc độ" in hint


def test_coverage_hint_excludes_last_asked_slot():
    """The slot asked last turn must not reappear in the gap list (anti-repeat exclusion)."""
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

    # A not-yet-asked slot is offered.
    assert SLOT_DESCRIPTIONS["sponsor"] in hint
    # The just-asked slot is dropped entirely from the inventory.
    assert SLOT_DESCRIPTIONS["why_now"] not in hint


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
    """Tool-loop analyst stub: scripts an ask_user selection that reports slot_assessment.

    analyze_node reads slot_assessment off the returned dict to compute coverage and maintain the
    stall counter; the named tool only drives the emitted AIMessage (irrelevant to these asserts).
    """

    def __init__(self, slot_assessment):
        self._slot_assessment = slot_assessment

    async def generate(self, **kwargs):
        return (
            {
                "tool": "ask_user",
                "message": "Câu hỏi tiếp theo?",
                "confidence": 0.3,
                "gaps": [],
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
async def test_original_scenario_no_consecutive_repeat_under_chronic_undergrade(client, db_session):
    """Reproduce the production transcript (SLR/PRISMA "why now" loop).

    Worst case: the model chronically grades every slot 'empty' even after the user answers —
    exactly the condition that produced 3 verbatim-repeated questions. Drive analyze_node turn
    by turn (threading state as LangGraph would) and assert the deterministic hint never steers
    toward the same slot twice in a row, and the loop escapes via the stall break.

    Pre-fix this loop pinned why_now every turn (no rotation, no credit) -> the asserted list
    would be [why_now, why_now, ...] and fail. The fix makes it rotate and terminate.
    """
    from app.graphs.nodes import _coverage_hint_target, analyze_node

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])
    agent_session = await _make_agent_session(client, db_session, project_id)

    all_empty = {"why_now": "empty", "sponsor": "empty", "expected_outcome": "empty", "success_state": "empty"}
    mock_llm = _AsyncLLM(all_empty)
    config = _config(str(agent_session.id), str(project_id), mock_llm)
    config["configurable"]["session_factory"] = _session_factory()

    state = _state(artifact_type="intent")
    pinned_sequence: list[str | None] = []
    stall_broke = False
    for _ in range(6):
        # Slot this turn's hint steers the question toward (None -> no pin / stall break).
        pinned_sequence.append(_coverage_hint_target(state))
        result = await analyze_node(state, config)
        state = {**state, **result}
        if (state.get("coverage_stall_count") or 0) >= COVERAGE_STALL_LIMIT:
            stall_broke = True
            break

    asked = [slot for slot in pinned_sequence if slot is not None]
    # No two consecutive turns ask about the same slot -> no verbatim repeated question.
    assert asked, pinned_sequence
    assert all(a != b for a, b in zip(asked, asked[1:], strict=False)), pinned_sequence
    # The chronic-undergrade loop is bounded: it escapes via the stall break instead of nagging.
    assert stall_broke


@pytest.mark.asyncio
async def test_analyze_stall_zero_when_no_slot_assessment(client, db_session):
    """No slot_assessment -> fail-open coverage -> stall counter resets to 0."""
    result, _ = await _run_analyze(client, db_session, None, prev_ratio=0.0, prev_stall=2)

    assert result["coverage_complete"] is None
    assert result["coverage_stall_count"] == 0
