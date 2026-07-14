"""`CapabilityResolver`: pure `WorkflowSnapshot -> CapabilityDecision` evaluator.

Purity invariant (locked by `tests/unit/gates/test_capability_resolver_purity.py`): this module
must never import `logging`, a DB session, or a tool-handler module (`app.graphs.agent_tools` or
anything under it). Audit/telemetry for a `CapabilityDecision` lives entirely in
`decision_projection.py`, which consumes this module's output — never the reverse.

Only the read-only pilot capabilities declared in `capability_manifest.READ_ONLY_PILOT_CAPABILITIES`
are ever evaluated as "allowed"; every other capability (including every mutating one) resolves to
`allowed=False, reason="capability_not_migrated"` so nothing outside the pilot can be opened by this
resolver, in any mode.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from app.graphs.gating.capability_manifest import MUTATING_CAPABILITIES, READ_ONLY_PILOT_CAPABILITIES
from app.graphs.gating.workflow_snapshot import WorkflowSnapshot
from app.graphs.session_phase import phase_allows

MENU = "menu"
DISPATCH = "dispatch"
HANDLER = "handler"

COHORT_OR_ACTOR_MISSING_REASON = "turn_cohort_or_actor_context_missing"


@dataclass(frozen=True)
class CapabilityDecision:
    capability: str
    evaluation_context: str  # "menu" | "dispatch" | "handler"
    allowed: bool
    reason: str | None
    effect_class: str  # "read_only" | "mutating" | "unclassified"
    pilot_eligible: bool
    snapshot_version: str


def _cohort_is_valid(snapshot: WorkflowSnapshot) -> bool:
    """Fail-closed guard: a capability can only be resolved authoritatively when the turn cohort
    and actor context are both present, non-empty, and were admitted under a non-legacy resolver
    mode. A cohort admitted while the resolver mode was "legacy" stays fail-closed even if the
    process-wide setting later flips — an admitted turn finishes with its own recorded adapter,
    not whatever the global setting has since become."""
    cohort = snapshot.turn_cohort
    actor = snapshot.actor_context
    if cohort is None or actor is None:
        return False
    if not cohort.turn_id or not actor.actor_id or not actor.correlation_id:
        return False
    if cohort.policy_resolver_mode not in ("shadow", "enforce"):
        return False
    return True


def _evaluate_run_critique(snapshot: WorkflowSnapshot) -> tuple[bool, str | None]:
    if not snapshot.has_draft:
        return False, "run_critique_unavailable"
    if snapshot.critique_rounds < snapshot.critique_rounds_max:
        return True, None
    if snapshot.draft_hash_stale:
        return True, None
    return False, "run_critique_unavailable"


def _evaluate_run_readiness_check(snapshot: WorkflowSnapshot) -> tuple[bool, str | None]:
    if snapshot.has_draft and snapshot.critique_rounds > 0:
        return True, None
    return False, "run_readiness_check_unavailable"


def _evaluate_recommend_next_workflow(snapshot: WorkflowSnapshot) -> tuple[bool, str | None]:
    if snapshot.has_draft or snapshot.sections_with_signal >= 2:
        return True, None
    return False, "recommend_next_workflow_unavailable"


_PILOT_EVALUATORS: dict[str, Callable[[WorkflowSnapshot], tuple[bool, str | None]]] = {
    "run_critique": _evaluate_run_critique,
    "run_readiness_check": _evaluate_run_readiness_check,
    "recommend_next_workflow": _evaluate_recommend_next_workflow,
}


class CapabilityResolver:
    """Stateless — safe to share a single instance process-wide."""

    def resolve(
        self,
        capability: str,
        snapshot: WorkflowSnapshot,
        *,
        evaluation_context: str = MENU,
    ) -> CapabilityDecision:
        pilot_eligible = capability in READ_ONLY_PILOT_CAPABILITIES
        if not pilot_eligible:
            effect_class = "mutating" if capability in MUTATING_CAPABILITIES else "unclassified"
            return CapabilityDecision(
                capability=capability,
                evaluation_context=evaluation_context,
                allowed=False,
                reason="capability_not_migrated",
                effect_class=effect_class,
                pilot_eligible=False,
                snapshot_version=snapshot.version,
            )

        if not _cohort_is_valid(snapshot):
            return CapabilityDecision(
                capability=capability,
                evaluation_context=evaluation_context,
                allowed=False,
                reason=COHORT_OR_ACTOR_MISSING_REASON,
                effect_class="read_only",
                pilot_eligible=True,
                snapshot_version=snapshot.version,
            )

        if not phase_allows(snapshot.phase, capability):
            return CapabilityDecision(
                capability=capability,
                evaluation_context=evaluation_context,
                allowed=False,
                reason="phase_excludes_tool",
                effect_class="read_only",
                pilot_eligible=True,
                snapshot_version=snapshot.version,
            )

        allowed, reason = _PILOT_EVALUATORS[capability](snapshot)
        return CapabilityDecision(
            capability=capability,
            evaluation_context=evaluation_context,
            allowed=allowed,
            reason=reason,
            effect_class="read_only",
            pilot_eligible=True,
            snapshot_version=snapshot.version,
        )
