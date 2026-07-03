"""Session phase state machine: transitions, legacy derivation, and per-phase tool gating."""

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from app.graphs.agent_tools import _phase_signals, current_session_phase, get_available_tools
from app.graphs.analysis.tool_gating import _gate_selected_tools, gate_model_selection
from app.graphs.session_phase import (
    DRAFT,
    ELICIT,
    FINALIZE,
    INTENT,
    REVIEW,
    IllegalPhaseTransition,
    PhaseSignals,
    derive_phase,
    phase_allows,
    transition,
)
from app.graphs.state import build_initial_workflow_state


def _signals(**overrides) -> PhaseSignals:
    base = dict(user_confirmed=False, has_draft=False, has_evidence=False, critique_started=False, finalize_open=False)
    base.update(overrides)
    return PhaseSignals(**base)


def _state(**overrides):
    state = build_initial_workflow_state(artifact_type="vision_objectives", workflow_area="analysis", step_key=None)
    state.update(overrides)
    return state


# --- derivation -----------------------------------------------------------


def test_derive_phase_covers_all_flag_combinations():
    assert derive_phase(_signals()) == INTENT
    assert derive_phase(_signals(user_confirmed=True)) == ELICIT
    assert derive_phase(_signals(user_confirmed=True, has_evidence=True)) == DRAFT
    assert derive_phase(_signals(user_confirmed=True, has_draft=True)) == DRAFT
    assert derive_phase(_signals(user_confirmed=True, has_draft=True, critique_started=True)) == REVIEW
    assert (
        derive_phase(_signals(user_confirmed=True, has_draft=True, critique_started=True, finalize_open=True))
        == FINALIZE
    )


def test_legacy_checkpoint_derives_phase_on_first_transition():
    """current=None (checkpoint created before session_phase existed) adopts the derived phase."""
    assert transition(None, _signals()) == INTENT
    assert transition(None, _signals(user_confirmed=True, has_draft=True, critique_started=True)) == REVIEW


def test_legacy_checkpoint_fixtures_resume_with_correct_phase():
    """Fixture states shaped like current-format checkpoints (pre-phase) derive correctly."""
    fresh = _state()
    fresh.pop("session_phase")
    assert current_session_phase(fresh) == INTENT

    confirmed = _state(user_confirmed=True)
    confirmed.pop("session_phase")
    assert current_session_phase(confirmed) == ELICIT

    reviewing = _state(user_confirmed=True, draft_body="## Vision\nx", critique_rounds=1)
    reviewing.pop("session_phase")
    assert current_session_phase(reviewing) == REVIEW


# --- transitions ----------------------------------------------------------


def test_legal_forward_transitions():
    assert transition(INTENT, _signals(user_confirmed=True)) == ELICIT
    assert transition(ELICIT, _signals(user_confirmed=True, has_evidence=True)) == DRAFT
    assert transition(DRAFT, _signals(user_confirmed=True, has_draft=True, critique_started=True)) == REVIEW
    assert (
        transition(REVIEW, _signals(user_confirmed=True, has_draft=True, critique_started=True, finalize_open=True))
        == FINALIZE
    )


def test_first_pass_critique_jumps_draft_to_finalize():
    """A critique that passes on the first round opens finalize the same turn -> DRAFT skips REVIEW."""
    assert (
        transition(DRAFT, _signals(user_confirmed=True, has_draft=True, critique_started=True, finalize_open=True))
        == FINALIZE
    )


def test_review_returns_to_draft_on_revision():
    """Critique state cleared (revision restarts) → REVIEW legally regresses to DRAFT."""
    assert transition(REVIEW, _signals(user_confirmed=True, has_draft=True)) == DRAFT


def test_illegal_transition_raises():
    with pytest.raises(IllegalPhaseTransition):
        transition(INTENT, _signals(user_confirmed=True, has_draft=True, critique_started=True))  # intent → review


def test_self_loop_is_always_legal():
    assert transition(REVIEW, _signals(user_confirmed=True, has_draft=True, critique_started=True)) == REVIEW


# --- single writer --------------------------------------------------------


def test_session_phase_single_writer_is_orchestrator():
    """No module outside nodes.py assigns state['session_phase'] (grep-test per the plan)."""
    app_dir = Path(__file__).parents[2] / "app"
    offenders = []
    for path in app_dir.rglob("*.py"):
        if "__pycache__" in str(path):
            continue
        source = path.read_text(encoding="utf-8")
        if '"session_phase":' in source or "['session_phase'] =" in source or '["session_phase"] =' in source:
            # build_initial_workflow_state's None default and nodes' orchestrator update are the
            # only allowed graph-state assignment sites. agent_event_service builds the read-side
            # SSE snapshot dict (key derived from the checkpoint), which is a projection, not a write.
            # turn_audit records session_phase into the AgentRun.analysis_result audit dict (so the
            # eval can group per-phase prompt tokens) — also a projection, never a WorkflowState write.
            if path.name not in {"nodes.py", "state.py", "agent_event_service.py", "turn_audit.py"}:
                offenders.append(path.name)
    assert offenders == [], f"session_phase assigned outside orchestrator/state defaults: {offenders}"


# --- menu gating ----------------------------------------------------------


def test_intent_phase_menu_excludes_draft_and_quality_tools():
    state = _state(session_phase=INTENT)
    names = {t.name for t in get_available_tools(state)}
    assert "write_draft" not in names
    assert "confirm_intent" in names
    assert "elicit" in names


def test_review_phase_menu_excludes_elicitation():
    state = _state(
        session_phase=REVIEW,
        user_confirmed=True,
        draft_body="## Vision\nx",
        critique_rounds=1,
    )
    names = {t.name for t in get_available_tools(state)}
    assert "elicit" not in names
    assert "web_search" not in names
    assert "write_draft" in names  # revision stays possible


def test_gate_rejects_out_of_phase_selection_with_phase_named(caplog):
    import logging

    state = _state(session_phase=REVIEW, user_confirmed=True)
    with caplog.at_level(logging.INFO, logger="app.graphs.analysis.tool_gating"):
        kept = _gate_selected_tools(state, [{"name": "elicit", "args": {}}, {"name": "respond", "args": {"message": "x"}}])
    assert [item["name"] for item in kept] == ["respond"]
    assert any("dropped_out_of_phase_tool" in r.getMessage() for r in caplog.records)


def test_gate_model_selection_reports_out_of_phase_feedback():
    state = _state(session_phase=FINALIZE, user_confirmed=True)
    message = AIMessage(
        content="",
        tool_calls=[{"id": "c1", "name": "elicit", "args": {"technique": "5_whys", "seed": "x"}}],
    )
    _model, gated, dropped, feedback, out_of_phase = gate_model_selection(state, message)
    assert gated == []
    assert out_of_phase == ["elicit"]
    assert feedback["out_of_phase_tools"] == {"phase": FINALIZE, "dropped": ["elicit"]}
    assert "dropped_tools" not in feedback  # phase drops are reported via their own notice


def test_phase_allows_unset_phase_blocks_nothing():
    assert phase_allows(None, "write_draft")
    assert phase_allows("", "finalize")


# --- signals --------------------------------------------------------------


def test_phase_signals_reads_draft_and_evidence_sources():
    state = _state(user_confirmed=True, session_elicit_count=1)
    signals = _phase_signals(state)
    assert signals.user_confirmed is True
    assert signals.has_evidence is True
    assert signals.has_draft is False
    assert derive_phase(signals) == DRAFT


def test_db_loaded_draft_counts_for_review_menu():
    state = _state(
        session_phase=REVIEW,
        user_confirmed=True,
        decision_nodes={},
        draft_body="## Vision\nExisting DB draft",
        critique_rounds=1,
    )

    signals = _phase_signals(state)
    names = {tool.name for tool in get_available_tools(state)}

    assert signals.has_draft is True
    assert "run_critique" in names
    assert "run_readiness_check" in names
