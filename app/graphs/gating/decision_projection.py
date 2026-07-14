"""Decision projection / audit adapter — the ONLY place a `CapabilityDecision` gets logged.

This module consumes `CapabilityDecision` (and, for shadow comparison, the legacy `Verdict` it is
compared against) as input; the dependency direction is one-way — `capability_resolver.py` must
never import this module or `logging`. Keeping observability here (instead of inside the
evaluator) is what makes the resolver pure: audit is a projection of an already-computed decision,
not a side effect the decision depends on.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.graphs.gate_logging import log_gate_decision
from app.graphs.gating.capability_resolver import CapabilityDecision
from app.graphs.gating.verdict import Verdict

# Mismatches whose sole cause is the resolver's fail-closed turn-cohort/actor-context guard are an
# accepted, expected difference while turn admission is not yet threaded through `WorkflowState` —
# every current production call site builds a snapshot with turn_cohort=None, so legacy-allows /
# resolver-denies-for-that-reason is the universal case, not a policy bug. Any other reason
# disagreeing with legacy is high-severity and must drive golden-matrix fixes before enforce.
_ACCEPTED_MISMATCH_REASONS = frozenset({"turn_cohort_or_actor_context_missing"})


@dataclass(frozen=True)
class ShadowMismatch:
    capability: str
    evaluation_context: str
    legacy_allowed: bool
    resolver_allowed: bool
    resolver_reason: str | None
    severity: str  # "accepted" | "high"


def compare_decision(legacy: Verdict, decision: CapabilityDecision) -> ShadowMismatch | None:
    """Pure comparison — no logging, no I/O. Returns `None` when legacy and resolver agree."""
    if legacy.is_allow == decision.allowed:
        return None
    severity = "accepted" if decision.reason in _ACCEPTED_MISMATCH_REASONS else "high"
    return ShadowMismatch(
        capability=decision.capability,
        evaluation_context=decision.evaluation_context,
        legacy_allowed=legacy.is_allow,
        resolver_allowed=decision.allowed,
        resolver_reason=decision.reason,
        severity=severity,
    )


# Process-lifetime dedupe key set: (turn_identity, evaluation_context, capability). Keeps
# `project_decision` from logging the same tuple twice — "one decision projection per
# turn/context" — without needing a DB/cache dependency in this leaf module. Turn admission is not
# yet threaded through `WorkflowState`, so callers pass the best turn-identity proxy they have
# (real `turn_id` once admission is wired; `last_agent_run_id` today) — see `pilot_runtime.py`.
_projected_keys: set[tuple[str, str, str]] = set()


def reset_projection_dedupe() -> None:
    """Test-only: clear the process-lifetime dedupe set between test cases."""
    _projected_keys.clear()


def project_decision(
    decision: CapabilityDecision,
    *,
    turn_identity: str | None,
    correlation_id: str | None,
    mismatch: ShadowMismatch | None = None,
) -> None:
    """Record one decision projection per (turn_identity, evaluation_context, capability)."""
    key = (turn_identity or "no_turn", decision.evaluation_context, decision.capability)
    if key in _projected_keys:
        return
    _projected_keys.add(key)

    extra: dict[str, object] = {
        "capability": decision.capability,
        "context": decision.evaluation_context,
        "pilot_eligible": decision.pilot_eligible,
        "turn_identity": turn_identity or "",
        "correlation_id": correlation_id or "",
    }
    if mismatch is not None:
        extra["mismatch_severity"] = mismatch.severity
        extra["legacy_allowed"] = mismatch.legacy_allowed
        extra["resolver_allowed"] = mismatch.resolver_allowed
    log_gate_decision(
        "capability_resolver",
        "allowed" if decision.allowed else "denied",
        reason=decision.reason,
        extra=extra,
    )
