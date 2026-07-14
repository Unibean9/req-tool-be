"""`agent_policy_resolver_mode` behavior at the `get_available_tools` call site:

- legacy (default): the menu is byte-identical to the pre-Phase-3 behavior.
- shadow: still byte-identical — shadow only compares/projects, never mutates the menu.
- enforce: still byte-identical here because turn admission is not yet threaded through
  WorkflowState — every call site passes turn_cohort=None, so enforce fails closed to legacy.
  Mutating tools (write_draft, finalize) are never touched by any mode.
"""

import hashlib

import pytest

from app.config import settings
from app.graphs.agent_tools import get_available_tools
from app.graphs.gating import pilot_runtime
from app.graphs.gating.verdict import Verdict
from app.graphs.gating.workflow_snapshot import ActorContextRef, TurnCohortRef
from app.schemas.artifact_synthesis import ArtifactReadinessState

_STATES = [
    {},
    {"user_confirmed": True, "draft_body": "A draft", "critique_rounds": 0},
    {"user_confirmed": True, "draft_body": "A draft", "critique_rounds": 1},
    {
        "user_confirmed": True,
        "draft_body": "A draft",
        "critique_rounds": 1,
        "quality_report": {"quality_gate_result": "pass", "blocking_issues": []},
        "candidate_readiness": {"state": ArtifactReadinessState.SUFFICIENT, "score": 1.0, "gaps": []},
        "last_critiqued_draft_hash": hashlib.md5(b"A draft").hexdigest()[:8],
    },
]


def _names(state):
    return {t.name for t in get_available_tools(state)}


@pytest.mark.parametrize("mode", ["legacy", "shadow", "enforce"])
def test_menu_unchanged_across_resolver_modes(monkeypatch, mode):
    baseline = [_names(state) for state in _STATES]
    monkeypatch.setattr(settings, "agent_policy_resolver_mode", mode)
    under_mode = [_names(state) for state in _STATES]
    assert under_mode == baseline


def test_mutating_tools_never_gated_by_resolver_in_any_mode(monkeypatch):
    state = {"user_confirmed": True, "draft_body": "A draft", "critique_rounds": 1}
    for mode in ("legacy", "shadow", "enforce"):
        monkeypatch.setattr(settings, "agent_policy_resolver_mode", mode)
        names = _names(state)
        assert "write_draft" in names  # unaffected by resolver mode


def test_shadow_admitted_turn_stays_shadow_under_global_enforce(monkeypatch):
    """A turn admitted while the global mode was "shadow" must never be enforced just because the
    global setting is later flipped to "enforce" — the turn finishes with its own recorded adapter.
    Only a cohort recorded under policy_resolver_mode="enforce" may substitute the legacy verdict."""
    monkeypatch.setattr(settings, "agent_policy_resolver_mode", "enforce")
    state = {"user_confirmed": True, "draft_body": "A draft", "critique_rounds": 1}
    cohort = TurnCohortRef(turn_id="turn-1", policy_resolver_mode="shadow", execution_mode="inline")
    actor = ActorContextRef(actor_id="user-1", correlation_id="corr-1")
    legacy_verdict = Verdict.deny("legacy_forced_deny_for_test")
    verdicts = {"run_critique": legacy_verdict}

    pilot_runtime.evaluate_pilot_capabilities(
        state, verdicts, turn_cohort=cohort, actor_context=actor
    )

    assert verdicts["run_critique"] is legacy_verdict  # untouched: shadow-admitted, not substituted


def test_enforce_admitted_turn_is_substituted_under_global_enforce(monkeypatch):
    """Contrast case: a cohort actually recorded under policy_resolver_mode="enforce" is eligible
    for substitution when the global mode is also "enforce"."""
    monkeypatch.setattr(settings, "agent_policy_resolver_mode", "enforce")
    state = {"user_confirmed": True, "draft_body": "A draft", "critique_rounds": 1}
    cohort = TurnCohortRef(turn_id="turn-2", policy_resolver_mode="enforce", execution_mode="inline")
    actor = ActorContextRef(actor_id="user-1", correlation_id="corr-1")
    legacy_verdict = Verdict.deny("legacy_forced_deny_for_test")
    verdicts = {"run_critique": legacy_verdict}

    pilot_runtime.evaluate_pilot_capabilities(
        state, verdicts, turn_cohort=cohort, actor_context=actor
    )

    assert verdicts["run_critique"] is not legacy_verdict
    assert verdicts["run_critique"].is_allow  # resolver allows run_critique for this state
