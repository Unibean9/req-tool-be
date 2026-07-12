"""Explore Tools + Dynamic Gating.

Covers the note scratchpad tool (`note`) and `get_available_tools(state)`, the state-driven menu
that decides which tools the loop may pick each turn.
"""

import hashlib

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from app.graphs.agent_tools import get_available_tools
from app.graphs.decision_graph import create_node, render_view


def _names(tools):
    return [t.name for t in tools]


def _hash(body: str) -> str:
    return hashlib.md5(body.encode()).hexdigest()[:8]


def _pass_gate(draft: str) -> dict:
    """Quality-report + hash + readiness keys that satisfy the finalize gate for `draft`."""
    from app.schemas.artifact_synthesis import ArtifactReadinessState
    return {
        "quality_report": {"quality_gate_result": "pass", "blocking_issues": []},
        "last_critiqued_draft_hash": _hash(draft),
        "candidate_readiness": {"state": ArtifactReadinessState.SUFFICIENT, "score": 1.0, "gaps": []},
    }


def _draft_state(statement: str = "Mot content draft") -> dict:
    nodes = {
        "N1": create_node(
            kind="objective",
            statement=statement,
            origin={"source": "test"},
            status="confirmed",
        )
    }
    return {"artifact_type": "brd", "decision_nodes": nodes}


def _draft_body(state: dict) -> str:
    return render_view(state["decision_nodes"], state["artifact_type"])


def _note_turn(call_id: str):
    """An AIMessage choosing a note tool — one note turn in the loop's history."""
    return AIMessage(content="", tool_calls=[{"id": call_id, "name": "note", "args": {"content": "x"}}])


# ---------------------------------------------------------------------------
# T1 — finalize gated when there is no draft body
# ---------------------------------------------------------------------------

def test_finalize_not_available_without_draft_body():
    assert "finalize" not in _names(get_available_tools({"messages": []}))
    assert "finalize" not in _names(get_available_tools({"messages": [], "decision_nodes": {}}))


# ---------------------------------------------------------------------------
# T2 — finalize available once graph view is non-empty AND a critique has run
# ---------------------------------------------------------------------------

def test_finalize_available_after_write_draft():
    # finalize now also requires critique_rounds > 0 AND a passing, current quality gate (spec §15.1).
    state = {**_draft_state(), "messages": [], "user_confirmed": True, "critique_rounds": 1}
    state.update(_pass_gate(_draft_body(state)))
    assert "finalize" in _names(get_available_tools(state))


def test_finalize_hidden_when_gate_fails():
    state = {**_draft_state(), "messages": [], "user_confirmed": True, "critique_rounds": 1}
    draft = _draft_body(state)
    state = {
        **state,
        "messages": [],
        "user_confirmed": True,
        "critique_rounds": 1,
        "quality_report": {"quality_gate_result": "fail", "blocking_issues": ["x"]},
        "last_critiqued_draft_hash": _hash(draft),
    }
    assert "finalize" not in _names(get_available_tools(state))


def test_finalize_hidden_when_draft_stale():
    original = _draft_state("Mot content draft")
    edited = _draft_state("Mot content draft da sua")
    state = {
        **edited,
        "messages": [],
        "user_confirmed": True,
        "critique_rounds": 1,
        **_pass_gate(_draft_body(original)),
    }
    assert "finalize" not in _names(get_available_tools(state))


def test_readiness_check_gated_on_critique_rounds():
    no_critique = {**_draft_state(), "messages": [], "user_confirmed": True, "critique_rounds": 0}
    assert "run_readiness_check" not in _names(get_available_tools(no_critique))
    with_critique = {**_draft_state(), "messages": [], "user_confirmed": True, "critique_rounds": 1}
    assert "run_readiness_check" in _names(get_available_tools(with_critique))


# ---------------------------------------------------------------------------
# T3 — note tool is no longer capped by a step-limit gate
# ---------------------------------------------------------------------------

def test_note_tool_remains_available_after_many_notes():
    messages = [_note_turn(f"c{i}") for i in range(5)]
    names = _names(get_available_tools({"messages": messages}))

    assert "note" in names


def test_write_note_available_with_no_note_history():
    messages = []
    names = _names(get_available_tools({"messages": messages}))
    assert "note" in names


# ---------------------------------------------------------------------------
# T5 — respond is always offered and breaks the note streak (a user-facing pause)
# ---------------------------------------------------------------------------

def _respond_turn(call_id: str):
    return AIMessage(
        content="", tool_calls=[{"id": call_id, "name": "respond", "args": {"message": "x", "mode": "critique"}}]
    )


def test_respond_always_available():
    assert "respond" in _names(get_available_tools({"messages": []}))


# ---------------------------------------------------------------------------
# M2 — read_artifact offered in both phases (read-only, never gated)
# ---------------------------------------------------------------------------

def test_read_artifact_available_in_intent_and_artifact_phase():
    # Intent phase (user_confirmed None) and artifact phase both offer side-effect-free reads.
    assert "read_artifact" in _names(get_available_tools({"messages": []}))
    assert "read_artifact" in _names(get_available_tools({"messages": [], "user_confirmed": True}))
    assert "read_source_documents" in _names(get_available_tools({"messages": []}))
    assert "read_source_documents" in _names(get_available_tools({"messages": [], "user_confirmed": True}))


def test_respond_resets_note_step_limit():
    messages = [_note_turn(f"c{i}") for i in range(5)] + [_respond_turn("r1")]
    names = _names(get_available_tools({"messages": messages}))
    assert "note" in names


def test_note_step_limit_does_not_block_run_critique():
    messages = [_note_turn(f"c{i}") for i in range(5)]
    names = _names(get_available_tools({**_draft_state(), "messages": messages, "user_confirmed": True}))
    assert "run_critique" in names


# ---------------------------------------------------------------------------
# T4 — write_note appends a ToolMessage to messages (decision 3: no notes field)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_write_note_appends_to_messages():
    from app.graphs.agent_tools import _write_note_impl

    command = await _write_note_impl("Assumption X may be wrong because...", {}, "call_1", "explore_note")

    appended = command.update["messages"]
    assert len(appended) == 1
    msg = appended[0]
    assert isinstance(msg, ToolMessage)
    assert "Assumption X" in msg.content
    assert msg.tool_call_id == "call_1"
