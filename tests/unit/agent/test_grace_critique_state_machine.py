"""Anti-wedge state-machine tests for the grace critique round.

From `(critique_rounds >= max, draft hash stale)` there is always a path to a
terminal action — finalize opens on a passing grace round, escalate is
recommended and finalize stays blocked on a failing one — and there is no
infinite `run_critique` availability at the cap without a new edit.
"""

import hashlib

import pytest

from app.graphs.agent_tools import CRITIQUE_ROUNDS_MAX, _cached_draft_body, _finalize_gate_open, _run_critique_impl
from app.graphs.gating.menu_rules import RunCritiqueMenuRule
from app.graphs.gating.verdict import VerdictKind
from app.schemas.artifact_synthesis import ArtifactReadinessState
from tests.factories import _draft_state, _scripted_client


def _hash(state: dict) -> str:
    return hashlib.md5(_cached_draft_body(state).encode()).hexdigest()[:8]


def _at_cap_stale_state() -> dict:
    """A draft edited after the last critique (hash mismatch), sitting at the rounds cap."""
    state = _draft_state()
    state["critique_rounds"] = CRITIQUE_ROUNDS_MAX
    state["candidate_readiness"] = {"state": ArtifactReadinessState.SUFFICIENT, "score": 1.0, "gaps": []}
    state["last_critiqued_draft_hash"] = "deadbeef"  # deliberately does not match the current draft
    return state


@pytest.mark.asyncio
async def test_grace_round_pass_opens_finalize():
    """Test A: a passing grace round re-anchors the hash and opens finalize normally — via the
    ordinary hash-match path, not a rounds-based bypass."""
    state = _at_cap_stale_state()
    assert RunCritiqueMenuRule().evaluate({"name": "run_critique"}, state).kind is VerdictKind.ALLOW

    config = {"configurable": {"llm_client": _scripted_client(0.9, ["nit nho"], [])}}
    command = await _run_critique_impl("completeness", state, config, "c1")

    state["quality_report"] = command.update["quality_report"]
    state["last_critiqued_draft_hash"] = command.update["last_critiqued_draft_hash"]
    state["critique_rounds"] = command.update["critique_rounds"]

    assert state["quality_report"]["quality_gate_result"] == "pass"
    assert state["last_critiqued_draft_hash"] == _hash(state)
    assert _finalize_gate_open(state) is True


@pytest.mark.asyncio
async def test_grace_round_fail_escalates_then_edit_reopens_grace_round():
    """Test B: a failing grace round recommends escalate and keeps finalize blocked; run_critique
    is gated off again (hash now matches the failed grace round) until the draft is edited again,
    which opens a fresh grace round — not a dead end."""
    state = _at_cap_stale_state()
    config = {"configurable": {"llm_client": _scripted_client(0.4, ["missing metric"], ["them KPI"])}}
    command = await _run_critique_impl("completeness", state, config, "c1")

    state["quality_report"] = command.update["quality_report"]
    state["last_critiqued_draft_hash"] = command.update["last_critiqued_draft_hash"]
    state["critique_rounds"] = command.update["critique_rounds"]

    assert state["quality_report"]["recommended_next_action"] == "escalate"
    assert _finalize_gate_open(state) is False

    assert RunCritiqueMenuRule().evaluate({"name": "run_critique"}, state).kind is VerdictKind.DENY

    # Editing the draft again (new statement -> new hash) opens a fresh grace round.
    state["decision_nodes"]["N1"]["statement"] = "Increase retention by 40% instead."
    assert RunCritiqueMenuRule().evaluate({"name": "run_critique"}, state).kind is VerdictKind.ALLOW


@pytest.mark.asyncio
async def test_no_grace_round_without_a_new_edit():
    """Test C: at the cap with the hash matching (no edit since the last critique), run_critique
    stays unavailable via both the menu rule and the tool implementation's own cap guard — no
    infinite-loop availability when nothing changed."""
    state = _draft_state()
    state["critique_rounds"] = CRITIQUE_ROUNDS_MAX
    state["last_critiqued_draft_hash"] = _hash(state)

    assert RunCritiqueMenuRule().evaluate({"name": "run_critique"}, state).kind is VerdictKind.DENY

    config = {"configurable": {"llm_client": None}}
    command = await _run_critique_impl("completeness", state, config, "call_1")
    assert command.update["tool_errors"][0]["code"] == "tool_not_available"
