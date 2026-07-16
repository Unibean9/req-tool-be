"""Golden differential: legacy menu verdict vs `CapabilityResolver` decision for the read-only pilot
capabilities (`run_critique`, `run_readiness_check`, `recommend_next_workflow`), reusing the state
fixtures from `test_menu_gating_matrix.py`'s hand-computed matrix. 0 high-severity mismatch is the
enforce-readiness bar.
"""

import pytest

from app.graphs import gating
from app.graphs.agent_tools import current_session_phase
from app.graphs.gating import Mode, menu_rules
from app.graphs.gating.capability_resolver import CapabilityResolver
from app.graphs.gating.decision_projection import compare_decision
from app.graphs.gating.workflow_snapshot import ActorContextRef, TurnCohortRef, build_workflow_snapshot
from tests.unit.gates import _menu_gating_states as states

_PILOT_CAPABILITIES = ("run_critique", "run_readiness_check", "recommend_next_workflow")

_VALID_COHORT = TurnCohortRef(turn_id="turn-1", policy_resolver_mode="shadow", execution_mode="inline")
_VALID_ACTOR = ActorContextRef(actor_id="actor-1", correlation_id="corr-1")

_resolver = CapabilityResolver()


def _legacy_verdicts(state):
    menu_rules.ensure_menu_rules_registered()
    phase = current_session_phase(state)
    return {
        name: gating.check({"name": name, "phase": phase}, state, Mode.MENU) for name in _PILOT_CAPABILITIES
    }


def _assert_zero_high_severity_mismatch(state):
    legacy = _legacy_verdicts(state)
    snapshot = build_workflow_snapshot(state, turn_cohort=_VALID_COHORT, actor_context=_VALID_ACTOR)
    for name in _PILOT_CAPABILITIES:
        decision = _resolver.resolve(name, snapshot, evaluation_context="menu")
        mismatch = compare_decision(legacy[name], decision)
        assert mismatch is None or mismatch.severity == "accepted", (
            f"{name}: high-severity mismatch legacy_allow={legacy[name].is_allow} "
            f"resolver={decision.allowed} reason={decision.reason}"
        )


def test_golden_no_draft_no_phase():
    _assert_zero_high_severity_mismatch(states.NO_DRAFT_NO_PHASE)


def test_golden_has_draft_critique_zero():
    _assert_zero_high_severity_mismatch(states.HAS_DRAFT_CRITIQUE_ZERO)


def test_golden_has_draft_critique_gt_zero_gate_closed():
    _assert_zero_high_severity_mismatch(states.HAS_DRAFT_CRITIQUE_GT_ZERO_GATE_CLOSED)


def test_golden_has_draft_critique_gt_zero_gate_open():
    _assert_zero_high_severity_mismatch(states.has_draft_critique_gt_zero_gate_open())


def test_golden_critique_rounds_at_max():
    _assert_zero_high_severity_mismatch(states.critique_rounds_at_max())


def test_golden_coverage_signal_without_draft():
    _assert_zero_high_severity_mismatch(states.COVERAGE_SIGNAL_WITHOUT_DRAFT)


def test_golden_phase_excludes_tool():
    from app.graphs.session_phase import INTENT

    _assert_zero_high_severity_mismatch(states.phase_excludes_tool(INTENT))


@pytest.mark.parametrize("name", _PILOT_CAPABILITIES)
def test_valid_cohort_matches_legacy_exactly_no_accepted_exception(name):
    """With a valid cohort, resolver/legacy must agree exactly (no "accepted" cohort-missing
    exception can mask a real policy divergence for these three capabilities)."""
    state = states.has_draft_critique_gt_zero_gate_open()
    legacy = _legacy_verdicts(state)[name]
    snapshot = build_workflow_snapshot(state, turn_cohort=_VALID_COHORT, actor_context=_VALID_ACTOR)
    decision = _resolver.resolve(name, snapshot, evaluation_context="menu")
    assert decision.allowed == legacy.is_allow
