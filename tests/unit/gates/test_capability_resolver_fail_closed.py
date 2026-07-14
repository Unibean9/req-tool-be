"""Missing/forged/stale turn cohort or ActorContext must fail closed to a deny — the resolver must
never open a read-only pilot capability just because the workflow facts alone would satisfy it.
"""

from app.graphs.gating.capability_resolver import COHORT_OR_ACTOR_MISSING_REASON, CapabilityResolver
from app.graphs.gating.workflow_snapshot import ActorContextRef, TurnCohortRef, build_workflow_snapshot

_resolver = CapabilityResolver()

# A state that would satisfy every read-only pilot capability's legacy condition, so any
# fail-closed denial below is attributable only to the cohort/actor guard.
_WIDE_OPEN_STATE = {
    "user_confirmed": True,
    "draft_body": "A draft",
    "critique_rounds": 1,
    "section_coverage": {"a": "filled", "b": "filled"},
}

_VALID_COHORT = TurnCohortRef(turn_id="turn-1", policy_resolver_mode="shadow", execution_mode="inline")
_VALID_ACTOR = ActorContextRef(actor_id="actor-1", correlation_id="corr-1")


def _assert_denied_for_every_pilot_capability(snapshot):
    for capability in ("run_critique", "run_readiness_check", "recommend_next_workflow"):
        decision = _resolver.resolve(capability, snapshot)
        assert decision.allowed is False
        assert decision.reason == COHORT_OR_ACTOR_MISSING_REASON


def test_missing_cohort_and_actor_fail_closed():
    snapshot = build_workflow_snapshot(_WIDE_OPEN_STATE, turn_cohort=None, actor_context=None)
    _assert_denied_for_every_pilot_capability(snapshot)


def test_missing_actor_only_fails_closed():
    snapshot = build_workflow_snapshot(_WIDE_OPEN_STATE, turn_cohort=_VALID_COHORT, actor_context=None)
    _assert_denied_for_every_pilot_capability(snapshot)


def test_missing_cohort_only_fails_closed():
    snapshot = build_workflow_snapshot(_WIDE_OPEN_STATE, turn_cohort=None, actor_context=_VALID_ACTOR)
    _assert_denied_for_every_pilot_capability(snapshot)


def test_forged_empty_actor_id_fails_closed():
    forged_actor = ActorContextRef(actor_id="", correlation_id="corr-1")
    snapshot = build_workflow_snapshot(_WIDE_OPEN_STATE, turn_cohort=_VALID_COHORT, actor_context=forged_actor)
    _assert_denied_for_every_pilot_capability(snapshot)


def test_forged_empty_turn_id_fails_closed():
    forged_cohort = TurnCohortRef(turn_id="", policy_resolver_mode="shadow", execution_mode="inline")
    snapshot = build_workflow_snapshot(_WIDE_OPEN_STATE, turn_cohort=forged_cohort, actor_context=_VALID_ACTOR)
    _assert_denied_for_every_pilot_capability(snapshot)


def test_stale_cohort_admitted_under_legacy_mode_fails_closed():
    """A cohort recorded at admission time with policy_resolver_mode="legacy" must stay fail-closed
    even though this snapshot is being resolved now (e.g. the process-wide setting flipped to
    shadow/enforce after this turn was admitted) — an admitted turn keeps its own adapter."""
    stale_cohort = TurnCohortRef(turn_id="turn-1", policy_resolver_mode="legacy", execution_mode="inline")
    snapshot = build_workflow_snapshot(_WIDE_OPEN_STATE, turn_cohort=stale_cohort, actor_context=_VALID_ACTOR)
    _assert_denied_for_every_pilot_capability(snapshot)


def test_non_pilot_capability_always_denied_regardless_of_cohort():
    snapshot = build_workflow_snapshot(_WIDE_OPEN_STATE, turn_cohort=_VALID_COHORT, actor_context=_VALID_ACTOR)
    for capability in ("write_draft", "finalize", "create_artifact_link", "propose_retirement"):
        decision = _resolver.resolve(capability, snapshot)
        assert decision.allowed is False
        assert decision.reason == "capability_not_migrated"
        assert decision.effect_class == "mutating"
        assert decision.pilot_eligible is False
