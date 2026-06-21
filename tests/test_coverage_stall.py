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
# Fix 2 — stall detection relaxes the gate and rewrites the hint
# ---------------------------------------------------------------------------

def test_route_propose_past_floor_honoured_without_override():
    """Phase 0: past the tier-1 floor, propose is honoured regardless of an affirmative token.

    Pre-Phase-0 this returned ask_human (tier-2 required an explicit user request). Tier-2 is
    removed, so coverage incompleteness no longer vetoes routing once past the floor.
    """
    from app.graphs.nodes import route_node

    state = _state(artifact_type="intent", turn_count=2, analysis_result={"next_action": "propose"})
    state["coverage_complete"] = False
    state["coverage_stall_count"] = 0
    # Past the floor (1 slot filled); no affirmative token -> pre-Phase-0 tier-2 would have blocked.
    state["slot_coverage"] = {"why_now": "filled", "sponsor": "empty"}
    state["messages"] = [{"role": "user", "content": "tôi cần thêm thông tin"}]

    assert route_node(state) == "confirm"


def test_route_gate_blocks_at_zero_filled_even_with_user_override():
    """Hard floor: 0 slot filled (propose-from-greeting) -> chặn kể cả user override."""
    from app.graphs.nodes import route_node

    state = _state(artifact_type="intent", turn_count=2, analysis_result={"next_action": "propose"})
    state["coverage_complete"] = False
    state["coverage_stall_count"] = 0
    state["slot_coverage"] = {"why_now": "empty", "sponsor": "empty"}
    state["messages"] = [{"role": "user", "content": "cứ tạo đi"}]

    assert route_node(state) == "ask_human"


def test_route_propose_past_floor_no_affirmative_goes_confirm():
    """M9 (Phase 0): once past the tier-1 floor, a model 'propose' is honoured even when
    coverage is incomplete and the user message carries no affirmative token.

    Pre-Phase-0 the tier-2 soft gate forced this back to ask_human (it required an explicit
    user request via _user_requests_propose). Phase 0 removes tier-2: coverage is a prompt
    signal, not a routing veto. The graph must not override the model's judgement here.
    """
    from app.graphs.nodes import route_node

    state = _state(artifact_type="problem", turn_count=2, analysis_result={"next_action": "propose"})
    state["coverage_complete"] = False
    state["coverage_stall_count"] = 0
    # required slots for 'problem': who/obstacle/root_cause/frequency/impact.
    # who+obstacle filled -> past the floor (>=1 filled); the rest empty -> coverage incomplete.
    state["slot_coverage"] = {
        "who": "filled",
        "obstacle": "filled",
        "root_cause": "empty",
        "frequency": "empty",
        "impact": "empty",
    }
    # No affirmative token: pre-Phase-0 tier-2 would have blocked on this.
    state["messages"] = [{"role": "user", "content": "tôi cần thêm thông tin"}]

    assert route_node(state) == "confirm"


def test_user_requests_propose_detects_signal():
    from app.graphs.nodes import _user_requests_propose

    yes = _state(artifact_type="intent")
    yes["messages"] = [{"role": "user", "content": "cứ tạo đi"}]
    assert _user_requests_propose(yes) is True

    no = _state(artifact_type="intent")
    no["messages"] = [{"role": "user", "content": "tôi cần thêm thông tin"}]
    assert _user_requests_propose(no) is False


def test_below_minimum_floor():
    from app.graphs.nodes import _below_minimum_floor

    assert _below_minimum_floor({"slot_coverage": {"why_now": "empty", "sponsor": "empty"}}) is True
    assert _below_minimum_floor({"slot_coverage": {"why_now": "filled", "sponsor": "empty"}}) is False
    # None coverage (non-BRD) -> fail-open, not below floor.
    assert _below_minimum_floor({"slot_coverage": None}) is False


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
    assert all(a != b for a, b in zip(asked, asked[1:])), pinned_sequence
    # The chronic-undergrade loop is bounded: it escapes via the stall break instead of nagging.
    assert stall_broke


@pytest.mark.asyncio
async def test_analyze_stall_zero_when_no_slot_assessment(client, db_session):
    """No slot_assessment -> fail-open coverage -> stall counter resets to 0."""
    result, _ = await _run_analyze(client, db_session, None, prev_ratio=0.0, prev_stall=2)

    assert result["coverage_complete"] is None
    assert result["coverage_stall_count"] == 0
