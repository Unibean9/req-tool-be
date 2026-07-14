"""Decision projection: one log line per (turn_identity, evaluation_context, capability); the
comparison itself is pure and never mutates the resolver's decision or the legacy verdict.
"""

from app.graphs.gating import decision_projection
from app.graphs.gating.capability_resolver import CapabilityDecision
from app.graphs.gating.decision_projection import compare_decision, project_decision
from app.graphs.gating.verdict import Verdict


def _decision(allowed: bool, reason: str | None = None) -> CapabilityDecision:
    return CapabilityDecision(
        capability="run_critique",
        evaluation_context="menu",
        allowed=allowed,
        reason=reason,
        effect_class="read_only",
        pilot_eligible=True,
        snapshot_version="v1",
    )


def test_compare_decision_returns_none_on_agreement():
    assert compare_decision(Verdict.allow(), _decision(True)) is None
    assert compare_decision(Verdict.deny("x"), _decision(False, "x")) is None


def test_compare_decision_classifies_cohort_missing_as_accepted():
    mismatch = compare_decision(Verdict.allow(), _decision(False, "turn_cohort_or_actor_context_missing"))
    assert mismatch is not None
    assert mismatch.severity == "accepted"


def test_compare_decision_classifies_other_disagreement_as_high():
    mismatch = compare_decision(Verdict.allow(), _decision(False, "run_critique_unavailable"))
    assert mismatch is not None
    assert mismatch.severity == "high"


def test_project_decision_logs_once_per_turn_context_capability(monkeypatch):
    decision_projection.reset_projection_dedupe()
    calls: list[tuple] = []
    monkeypatch.setattr(
        decision_projection,
        "log_gate_decision",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    project_decision(_decision(True), turn_identity="turn-1", correlation_id="corr-1")
    project_decision(_decision(True), turn_identity="turn-1", correlation_id="corr-1")
    assert len(calls) == 1

    project_decision(_decision(True), turn_identity="turn-2", correlation_id="corr-2")
    assert len(calls) == 2
    decision_projection.reset_projection_dedupe()
