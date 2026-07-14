"""`WorkflowSnapshot`: a pure, immutable normalization of the workflow facts `CapabilityResolver`
decides over.

Pure module: no DB session import, no `logging` import, no tool-handler import. Recomputes phase
and finalize-gate facts directly against `state` via `PhaseSignals`/`derive_phase`, rather than
calling `agent_tools.current_session_phase`/`agent_tools._phase_signals` — both of those trigger
`log_gate_decision` as a side effect (via `_finalize_gate_open`), so building a snapshot must never
reuse them or it would inflate the legacy gate's log volume purely as a side effect of shadow
comparison. The duplication here is deliberate and temporary: it is the pure-authority replacement
this phase is migrating toward, not a new legacy-parallel rule copy.

`turn_cohort`/`actor_context` are the only fields a caller supplies out of band (quoted, never
derived, from `AgentTurnEnvelope` — see Phase 2). Everything else is derived from `state`. Turn
admission is not yet threaded through `WorkflowState`, so every current call site builds a snapshot
with both left `None`; `CapabilityResolver` must fail closed on that, never guess a capability open.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from app.config import settings
from app.documents.registry import status_score
from app.graphs.decision_graph import render_view
from app.graphs.lifecycle_context import lifecycle_tool_block_reason
from app.graphs.session_phase import PhaseSignals, derive_phase
from app.schemas.artifact_synthesis import ArtifactReadinessState

SNAPSHOT_VERSION = "v1"


@dataclass(frozen=True)
class ActorContextRef:
    """Immutable reference to the acting principal, quoted from `AgentTurnEnvelope`."""

    actor_id: str
    correlation_id: str


@dataclass(frozen=True)
class TurnCohortRef:
    """Immutable reference to the admitted turn's cohort, quoted from `AgentTurnEnvelope.cohort`."""

    turn_id: str
    policy_resolver_mode: str | None
    execution_mode: str | None
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkflowSnapshot:
    """Everything `CapabilityResolver` needs to decide a read-only pilot capability."""

    version: str
    phase: str | None
    has_draft: bool
    critique_rounds: int
    critique_rounds_max: int
    draft_hash_stale: bool
    finalize_gate_open: bool
    sections_with_signal: int
    lifecycle_block_reason: Mapping[str, str | None]
    turn_cohort: TurnCohortRef | None
    actor_context: ActorContextRef | None


def _draft_body(state: Mapping[str, Any]) -> str:
    decision_nodes = state.get("decision_nodes") or {}
    artifact_type = state.get("artifact_type") or "brd"
    if decision_nodes:
        return render_view(decision_nodes, artifact_type)
    return str(state.get("draft_body") or "")


def _draft_hash_stale(state: Mapping[str, Any], draft_body: str) -> bool:
    current_hash = hashlib.md5(draft_body.encode()).hexdigest()[:8]
    return current_hash != state.get("last_critiqued_draft_hash")


def _finalize_gate_open(state: Mapping[str, Any], draft_hash_stale: bool) -> bool:
    report = state.get("quality_report")
    if not report or report.get("quality_gate_result") != "pass":
        return False
    readiness = state.get("candidate_readiness")
    if not isinstance(readiness, dict) or readiness.get("state") != ArtifactReadinessState.SUFFICIENT:
        return False
    return not draft_hash_stale


def build_workflow_snapshot(
    state: Mapping[str, Any],
    *,
    turn_cohort: TurnCohortRef | None = None,
    actor_context: ActorContextRef | None = None,
) -> WorkflowSnapshot:
    """Build an immutable `WorkflowSnapshot` from current server facts.

    `turn_cohort`/`actor_context` must come from an admitted `AgentTurnEnvelope`; a caller that has
    none (turn admission not enabled, or the turn is unknown) passes `None` for both rather than
    fabricating placeholder values — `CapabilityResolver` treats `None` as fail-closed input.
    """
    draft_body = _draft_body(state)
    has_draft = bool(draft_body.strip())
    critique_rounds = int(state.get("critique_rounds") or 0)
    critique_started = critique_rounds > 0 or bool(state.get("quality_report"))
    draft_hash_stale = _draft_hash_stale(state, draft_body)
    finalize_gate_open = has_draft and critique_started and _finalize_gate_open(state, draft_hash_stale)

    signals = PhaseSignals(
        user_confirmed=state.get("user_confirmed") is not None,
        has_draft=has_draft,
        has_evidence=bool(state.get("decision_nodes")) or bool(state.get("session_elicit_count") or 0),
        critique_started=critique_started,
        finalize_open=finalize_gate_open,
    )
    phase = state.get("session_phase") or derive_phase(signals)

    coverage = state.get("section_coverage") or {}
    sections_with_signal = sum(1 for v in coverage.values() if status_score(v) > 0.0)

    lifecycle_reasons = {
        "write_draft": lifecycle_tool_block_reason(state, "write_draft", {}),
    }

    return WorkflowSnapshot(
        version=SNAPSHOT_VERSION,
        phase=phase,
        has_draft=has_draft,
        critique_rounds=critique_rounds,
        critique_rounds_max=settings.max_critique_rounds,
        draft_hash_stale=draft_hash_stale,
        finalize_gate_open=finalize_gate_open,
        sections_with_signal=sections_with_signal,
        lifecycle_block_reason=lifecycle_reasons,
        turn_cohort=turn_cohort,
        actor_context=actor_context,
    )
