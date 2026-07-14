"""Wiring layer: runs `CapabilityResolver` alongside the legacy menu verdicts for the read-only
pilot capabilities, per `settings.agent_policy_resolver_mode`.

- `legacy`: no-op (the default; existing regression suite stays green because nothing here runs).
- `shadow`: resolve + compare + project for telemetry only — `verdicts` is never mutated.
- `enforce`: resolve, project, and substitute the resolver's decision into `verdicts` ONLY for a
  pilot capability whose snapshot carries a valid (non-missing, non-legacy-admitted) turn cohort +
  actor context; otherwise fail closed to the untouched legacy verdict already in `verdicts`.

This module is the boundary that is allowed to know about both the pure resolver and the logging
projection adapter — `capability_resolver.py` stays ignorant of both `logging` and this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.config import settings
from app.graphs.gating.capability_manifest import READ_ONLY_PILOT_CAPABILITIES
from app.graphs.gating.capability_resolver import (
    COHORT_OR_ACTOR_MISSING_REASON,
    CapabilityDecision,
    CapabilityResolver,
)
from app.graphs.gating.decision_projection import compare_decision, project_decision
from app.graphs.gating.verdict import Verdict
from app.graphs.gating.workflow_snapshot import ActorContextRef, TurnCohortRef, build_workflow_snapshot

_resolver = CapabilityResolver()


def _turn_identity(state: Mapping[str, Any], turn_cohort: TurnCohortRef | None) -> str | None:
    if turn_cohort is not None:
        return turn_cohort.turn_id
    # Best available turn-identity proxy until turn admission is threaded through WorkflowState —
    # changes once per analyze turn, same lifetime "one decision projection per turn" is meant to key on.
    return state.get("last_agent_run_id")


def evaluate_pilot_capabilities(
    state: Mapping[str, Any],
    verdicts: dict[str, Verdict],
    *,
    evaluation_context: str = "menu",
    turn_cohort: TurnCohortRef | None = None,
    actor_context: ActorContextRef | None = None,
) -> None:
    """Shadow-compare (and, in enforce mode, selectively substitute) the pilot capabilities present
    in `verdicts`. Mutates `verdicts` in place — ONLY ever for capabilities in the read-only pilot
    allowlist, and ONLY in `enforce` mode with a valid cohort/actor. Mutating tools are never keys
    considered here regardless of mode (they are simply never in `READ_ONLY_PILOT_CAPABILITIES`).

    A no-op when `agent_policy_resolver_mode == "legacy"` (the default) — no runtime behavior
    changes and nothing gets logged.
    """
    mode = settings.agent_policy_resolver_mode
    if mode == "legacy":
        return

    snapshot = build_workflow_snapshot(state, turn_cohort=turn_cohort, actor_context=actor_context)
    turn_identity = _turn_identity(state, turn_cohort)
    correlation_id = actor_context.correlation_id if actor_context is not None else None

    for capability in READ_ONLY_PILOT_CAPABILITIES:
        if capability not in verdicts:
            continue
        legacy_verdict = verdicts[capability]
        decision: CapabilityDecision = _resolver.resolve(
            capability, snapshot, evaluation_context=evaluation_context
        )
        mismatch = compare_decision(legacy_verdict, decision)
        project_decision(
            decision,
            turn_identity=turn_identity,
            correlation_id=correlation_id,
            mismatch=mismatch,
        )
        # Substitution requires BOTH the global mode and the turn's own recorded admission mode to
        # be "enforce" — a turn admitted under "shadow" must stay shadow-compared even if the
        # global setting is later flipped to "enforce" (a turn finishes with its admitted adapter,
        # not whatever the global setting has since become). `_cohort_is_valid` alone accepts
        # "shadow" or "enforce" cohorts because a shadow-admitted cohort must still produce a
        # comparable decision here — the stricter check belongs at the substitution boundary, not
        # inside cohort validity.
        turn_is_enforce_admitted = turn_cohort is not None and turn_cohort.policy_resolver_mode == "enforce"
        if mode == "enforce" and turn_is_enforce_admitted and decision.reason != COHORT_OR_ACTOR_MISSING_REASON:
            verdicts[capability] = (
                Verdict.allow() if decision.allowed else Verdict.deny(decision.reason or "resolver_denied")
            )
        # else: shadow mode, enforce without a valid cohort, or enforce with a shadow-admitted turn —
        # fail closed to the untouched legacy verdict already in `verdicts`.
