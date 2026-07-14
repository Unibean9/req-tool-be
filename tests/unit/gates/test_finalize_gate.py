"""Finalize gate: graph view non-empty AND critique_rounds > 0 AND quality gate passed AND candidate_readiness sufficient.

Also covers the blocker-question edge case: a resurfaced (blocker-resolved-but-unanswered) parked
open_question must never gate finalize, since no tool exists to clear it.
"""

import hashlib
from unittest.mock import patch

import pytest

from app.graphs.agent_tools import _finalize_impl, current_draft_body, get_available_tools
from app.graphs.decision_graph import create_node, render_view
from app.schemas.artifact_synthesis import ArtifactReadinessState


def _names(state):
    return {t.name for t in get_available_tools(state)}


def _hash(body: str) -> str:
    return hashlib.md5(body.encode()).hexdigest()[:8]


def _state_with_graph(statement: str = "A draft") -> dict:
    return {
        "messages": [],
        "user_confirmed": True,
        "artifact_type": "brd",
        "decision_nodes": {
            "N1": create_node(
                kind="objective",
                statement=statement,
                origin={"source": "test"},
                status="confirmed",
            )
        },
    }


def _current_body(state: dict) -> str:
    return render_view(state["decision_nodes"], state["artifact_type"])


def _passing_state(draft: str = "A draft", critique_rounds: int = 1) -> dict:
    """A state where every finalize-gate condition is satisfied (pass gate + fresh hash + sufficient readiness)."""
    state = {
        **_state_with_graph(draft),
        "critique_rounds": critique_rounds,
        "quality_report": {"quality_gate_result": "pass", "blocking_issues": []},
        "candidate_readiness": {"state": ArtifactReadinessState.SUFFICIENT, "score": 1.0, "gaps": []},
    }
    state["last_critiqued_draft_hash"] = _hash(_current_body(state))
    return state


def test_finalize_not_available_when_critique_rounds_zero():
    state = _passing_state(critique_rounds=0)
    assert "finalize" not in _names(state)


def test_finalize_available_when_critique_rounds_positive():
    assert "finalize" in _names(_passing_state())


def test_finalize_not_available_without_decision_graph_view():
    state = _passing_state()
    state["decision_nodes"] = {}
    assert "finalize" not in _names(state)


def test_finalize_uses_db_draft_without_graph_view():
    draft = "Draft tai tu DB"
    state = {
        "messages": [],
        "user_confirmed": True,
        "artifact_type": "brd",
        "decision_nodes": {},
        "draft_body": draft,
        "critique_rounds": 1,
        "quality_report": {"quality_gate_result": "pass", "blocking_issues": []},
        "last_critiqued_draft_hash": _hash(draft),
        "candidate_readiness": {"state": ArtifactReadinessState.SUFFICIENT, "score": 1.0, "gaps": []},
    }
    names = _names(state)
    assert "finalize" in names
    assert "run_readiness_check" in names


def test_finalize_not_available_when_gate_fails():
    state = _passing_state()
    state["quality_report"] = {"quality_gate_result": "fail", "blocking_issues": ["missing metric"]}
    assert "finalize" not in _names(state)


def test_finalize_not_available_when_draft_edited_after_critique():
    state = _passing_state()
    state["decision_nodes"] = {
        "N1": create_node(
            kind="objective",
            statement="Draft da sua sau critique",
            origin={"source": "test"},
            status="confirmed",
        )
    }
    assert "finalize" not in _names(state)


def test_finalize_not_available_on_stale_hash_at_rounds_cap():
    from app.graphs.agent_tools import CRITIQUE_ROUNDS_MAX

    state = _passing_state(critique_rounds=CRITIQUE_ROUNDS_MAX)
    # Stale hash: the rounds cap no longer opens finalize unconditionally — the escape hatch is gone.
    state["last_critiqued_draft_hash"] = "deadbeef"
    assert "finalize" not in _names(state)


def test_finalize_available_at_rounds_cap_when_hash_matches():
    from app.graphs.agent_tools import CRITIQUE_ROUNDS_MAX

    # Matching hash exactly at the cap (no edit since the last critique) is still the normal,
    # non-grace path — removing the bypass branch must not accidentally block it.
    state = _passing_state(critique_rounds=CRITIQUE_ROUNDS_MAX)
    assert "finalize" in _names(state)


def _working_session_factory():
    """A session_factory whose `async with ... as db` yields a db stub good enough for finalize's
    predecessor check + session-row status update (no real DB needed)."""

    class _Session:
        status = None
        interrupt_type = None

    session_row = _Session()

    class _Result:
        def scalar_one(self_inner):
            return session_row

        def scalar(self_inner):
            return 1

    class _DB:
        async def execute(self_inner, *a, **k):
            return _Result()

        async def commit(self_inner):
            return None

    def _factory():
        class _Ctx:
            async def __aenter__(self_inner):
                return _DB()

            async def __aexit__(self_inner, *a):
                return False

        return _Ctx()

    return _factory, session_row


@pytest.mark.asyncio
@pytest.mark.golden
async def test_finalize_interrupt_triggers_when_available():
    """_finalize_impl interrupts for human confirmation (the approval step) rather than erroring."""
    factory, session_row = _working_session_factory()
    config = {"configurable": {"session_factory": factory, "thread_id": "00000000-0000-0000-0000-000000000001"}}
    state = {
        **_passing_state("draft"),
        "critique_rounds": 1,
    }

    with patch("app.graphs.agent_tools.interrupt") as mock_interrupt:
        await _finalize_impl("Session complete.", state, config, "call_1")

    mock_interrupt.assert_called_once()
    assert session_row.status is not None


@pytest.mark.asyncio
async def test_finalize_hard_blocks_when_gate_fails():
    """Even if reached directly, _finalize_impl refuses a failing gate — ToolMessage, no interrupt."""
    config = {"configurable": {"session_factory": None, "thread_id": "00000000-0000-0000-0000-000000000001"}}
    state = {
        **_state_with_graph("draft"),
        "critique_rounds": 1,
        "quality_report": {"quality_gate_result": "fail", "blocking_issues": ["missing measurement criteria"]},
        "candidate_readiness": {"state": ArtifactReadinessState.SUFFICIENT, "score": 1.0, "gaps": []},
    }

    with patch("app.graphs.agent_tools.interrupt") as mock_interrupt:
        command = await _finalize_impl("Session complete.", state, config, "call_1")

    mock_interrupt.assert_not_called()
    msg = command.update["messages"][0]
    assert command.update["tool_errors"][0]["code"] == "finalize_gate_blocked"
    assert msg.status == "error"
    assert "Cannot finalize" in msg.content
    assert "missing measurement criteria" in msg.content


@pytest.mark.asyncio
async def test_finalize_hard_blocks_without_current_draft_body():
    config = {"configurable": {"session_factory": None, "thread_id": "00000000-0000-0000-0000-000000000001"}}
    state = {
        "critique_rounds": 2,
        "quality_report": {"quality_gate_result": "pass", "blocking_issues": []},
        "last_critiqued_draft_hash": _hash("draft"),
    }

    with patch("app.graphs.agent_tools.interrupt") as mock_interrupt:
        command = await _finalize_impl("Session complete.", state, config, "call_1")

    mock_interrupt.assert_not_called()
    assert command.update["tool_errors"][0]["code"] == "finalize_gate_blocked"
    assert "Cannot finalize" in command.update["messages"][0].content


@pytest.mark.asyncio
async def test_current_draft_body_uses_focused_artifact_body_without_graph():
    state = {
        "focused_artifact_id": "00000000-0000-0000-0000-000000000001",
        "draft_body": "stale",
    }
    config = {"configurable": {"session_factory": None}}
    assert await current_draft_body(state, config) == "stale"


@pytest.mark.asyncio
async def test_finalize_hard_blocks_when_focused_artifact_was_not_critiqued():
    config = {"configurable": {"session_factory": None, "thread_id": "00000000-0000-0000-0000-000000000001"}}
    state = {
        **_state_with_graph("Draft item B"),
        "critique_rounds": 1,
        "quality_report": {"quality_gate_result": "pass", "blocking_issues": []},
        "last_critiqued_draft_hash": _hash("Draft section A"),
        "candidate_readiness": {"state": ArtifactReadinessState.SUFFICIENT, "score": 1.0, "gaps": []},
    }

    with patch("app.graphs.agent_tools.interrupt") as mock_interrupt:
        command = await _finalize_impl("Session complete.", state, config, "call_1")

    mock_interrupt.assert_not_called()
    msg = command.update["messages"][0]
    assert "Cannot finalize" in msg.content
    assert "current draft" in msg.content


# ---------------------------------------------------------------------------
# A resurfaced (blocker-resolved-but-unanswered) parked open_question never gates finalize: the
# tools that once cleared a blocker (update_decision_node/dismiss_question) no longer exist, so
# finalize must not wait on one regardless of its status.
# ---------------------------------------------------------------------------


def _state_with_blocker(extra_nodes: dict | None = None) -> dict:
    nodes = {
        "N1": create_node(
            kind="objective", statement="A draft", origin={"source": "test"}, status="confirmed", node_id="N1"
        ),
        "Q1": create_node(
            kind="open_question",
            statement="Confirm rollout timeline",
            origin={"source": "test"},
            status="parked",
            blocks=["N1"],
            node_id="Q1",
        ),
    }
    nodes.update(extra_nodes or {})
    state = {
        "messages": [],
        "user_confirmed": True,
        "artifact_type": "brd",
        "decision_nodes": nodes,
        "critique_rounds": 1,
        "quality_report": {"quality_gate_result": "pass", "blocking_issues": []},
        "candidate_readiness": {"state": ArtifactReadinessState.SUFFICIENT, "score": 1.0, "gaps": []},
    }
    state["last_critiqued_draft_hash"] = _hash(_current_body(state))
    return state


@pytest.mark.asyncio
async def test_finalize_passes_with_unresolved_blocker_question():
    """Anti-wedge: a parked open_question with an active blocker must never gate finalize, since
    no tool remains to clear it."""
    factory, _ = _working_session_factory()
    config = {"configurable": {"session_factory": factory, "thread_id": "00000000-0000-0000-0000-000000000001"}}
    state = _state_with_blocker()

    with patch("app.graphs.agent_tools.interrupt") as mock_interrupt:
        command = await _finalize_impl("Session complete.", state, config, "call_1")

    mock_interrupt.assert_called_once()
    assert "tool_errors" not in command.update


@pytest.mark.asyncio
async def test_finalize_passes_after_answering_and_confirming_blocker():
    factory, _ = _working_session_factory()
    config = {"configurable": {"session_factory": factory, "thread_id": "00000000-0000-0000-0000-000000000001"}}
    state = _state_with_blocker()
    state["decision_nodes"]["Q1"]["status"] = "confirmed"
    state["last_critiqued_draft_hash"] = _hash(_current_body(state))

    with patch("app.graphs.agent_tools.interrupt") as mock_interrupt:
        command = await _finalize_impl("Session complete.", state, config, "call_1")

    mock_interrupt.assert_called_once()
    assert "tool_errors" not in command.update


@pytest.mark.asyncio
async def test_finalize_not_gated_by_non_blocker_parked_question():
    factory, _ = _working_session_factory()
    config = {"configurable": {"session_factory": factory, "thread_id": "00000000-0000-0000-0000-000000000001"}}
    extra = {
        "Q2": create_node(
            kind="open_question",
            statement="Nice-to-have follow-up",
            origin={"source": "test"},
            status="parked",
            node_id="Q2",
        )
    }
    state = _state_with_blocker(extra_nodes=extra)
    state["decision_nodes"]["Q1"]["status"] = "confirmed"
    state["last_critiqued_draft_hash"] = _hash(_current_body(state))

    with patch("app.graphs.agent_tools.interrupt") as mock_interrupt:
        command = await _finalize_impl("Session complete.", state, config, "call_1")

    mock_interrupt.assert_called_once()
    assert "tool_errors" not in command.update


# ---------------------------------------------------------------------------
# Required parity + anti-wedge tests: empty decision_nodes must behave exactly like
# flag-off did before this phase; a resurfaced parked open_question (built via the same
# create_node shape notes/sweep use) must never wedge finalize nor render the removed
# "DRAFT IN PROGRESS" block.
# ---------------------------------------------------------------------------


def test_parity_empty_decision_nodes_matches_prior_flag_off_behavior():
    """Empty decision_nodes: prompt carries no decision-view block, and the tool menu/finalize
    gate behave exactly as flag-off already did before this phase's removal."""
    from app.graphs.nodes import _build_tool_selection_prompt

    draft = "Draft tai tu DB"
    state = {
        "messages": [],
        "user_confirmed": True,
        "artifact_type": "brd",
        "decision_nodes": {},
        "draft_body": draft,
        "critique_rounds": 1,
        "quality_report": {"quality_gate_result": "pass", "blocking_issues": []},
        "last_critiqued_draft_hash": _hash(draft),
        "candidate_readiness": {"state": ArtifactReadinessState.SUFFICIENT, "score": 1.0, "gaps": []},
    }

    prompt = _build_tool_selection_prompt(state, [])
    assert "DRAFT IN PROGRESS" not in prompt

    names = _names(state)
    assert "finalize" in names
    assert "run_readiness_check" in names


@pytest.mark.asyncio
async def test_anti_wedge_assumption_plus_resurfaced_question_finalizes_without_draft_view():
    """A confirmed assumption node plus a resurfaced parked open_question (blocker resolved, same
    create_node shape notes/completeness_sweep produce) must not show the removed "DRAFT IN
    PROGRESS" block and must not block finalize."""
    from app.graphs.nodes import _build_tool_selection_prompt

    nodes = {
        "A1": create_node(
            kind="assumption",
            statement="Sinh vien co the truy cap internet on dinh",
            origin={"source": "test"},
            status="confirmed",
            node_id="A1",
        ),
        "N1": create_node(
            kind="objective", statement="A draft", origin={"source": "test"}, status="confirmed", node_id="N1"
        ),
        "Q1": create_node(
            kind="open_question",
            statement="Confirm rollout timeline",
            origin={"source": "test"},
            status="parked",
            blocks=["N1"],
            node_id="Q1",
        ),
    }

    prompt = _build_tool_selection_prompt({"artifact_type": "brd", "decision_nodes": nodes}, [])
    assert "DRAFT IN PROGRESS" not in prompt

    state = {
        "messages": [],
        "user_confirmed": True,
        "artifact_type": "brd",
        "decision_nodes": nodes,
        "critique_rounds": 1,
        "quality_report": {"quality_gate_result": "pass", "blocking_issues": []},
        "candidate_readiness": {"state": ArtifactReadinessState.SUFFICIENT, "score": 1.0, "gaps": []},
    }
    state["last_critiqued_draft_hash"] = _hash(_current_body(state))

    factory, _ = _working_session_factory()
    config = {"configurable": {"session_factory": factory, "thread_id": "00000000-0000-0000-0000-000000000001"}}

    with patch("app.graphs.agent_tools.interrupt") as mock_interrupt:
        command = await _finalize_impl("Session complete.", state, config, "call_1")

    mock_interrupt.assert_called_once()
    assert "tool_errors" not in command.update
