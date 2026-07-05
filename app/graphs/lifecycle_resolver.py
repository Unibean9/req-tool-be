"""Pure artifact lifecycle resolver.

The resolver consumes only already-loaded snapshots. Callers own database lookups,
including the per-id predecessor lookup required for `based_on` entries.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


class LifecycleState(enum.StrEnum):
    MISSING = "missing"
    BLOCKED = "blocked"
    IN_PROGRESS = "in_progress"
    CURRENT = "current"
    STALE = "stale"
    ORPHAN = "orphan"


class LifecycleAction(enum.StrEnum):
    CREATE = "create"
    ELICIT = "elicit"
    ASK = "ask"
    AMEND = "amend"
    RECONCILE = "reconcile"
    RETIRE = "retire"
    RELINK = "relink"


ALLOWED_ACTIONS_BY_STATE: dict[LifecycleState, frozenset[LifecycleAction]] = {
    LifecycleState.MISSING: frozenset({LifecycleAction.CREATE}),
    LifecycleState.BLOCKED: frozenset({LifecycleAction.ELICIT, LifecycleAction.ASK}),
    LifecycleState.IN_PROGRESS: frozenset({LifecycleAction.AMEND}),
    LifecycleState.CURRENT: frozenset(),
    LifecycleState.STALE: frozenset({LifecycleAction.RECONCILE}),
    LifecycleState.ORPHAN: frozenset({LifecycleAction.RETIRE, LifecycleAction.RELINK}),
}


@dataclass(frozen=True, slots=True)
class ArtifactLifecycleSnapshot:
    artifact_type: str
    artifact_id: str | None = None
    status: Any | None = None
    current_version_id: str | None = None
    current_version_status: Any | None = None
    based_on: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StructuralPredecessorSnapshot:
    artifact_type: str
    artifact_id: str | None = None
    status: Any | None = None
    current_version_id: str | None = None
    found: bool = True


@dataclass(frozen=True, slots=True)
class BasedOnPredecessorSnapshot:
    artifact_id: str
    artifact_type: str | None = None
    status: Any | None = None
    current_version_id: str | None = None
    found: bool = True
    retired: bool = False


@dataclass(frozen=True, slots=True)
class LifecycleVerdict:
    artifact_type: str
    state: LifecycleState
    reason: str
    allowed_actions: frozenset[LifecycleAction]
    artifact_id: str | None = None
    blockers: tuple[str, ...] = ()


def resolve_lifecycle(
    artifact_type: str,
    artifact: ArtifactLifecycleSnapshot | None,
    *,
    required_predecessors: Sequence[StructuralPredecessorSnapshot] = (),
    based_on_predecessors: Mapping[str, BasedOnPredecessorSnapshot] | None = None,
) -> LifecycleVerdict:
    """Resolve one artifact's lifecycle state from in-memory facts only."""

    blockers = _blocking_required_predecessors(required_predecessors)
    if artifact is None:
        if blockers:
            return _verdict(
                artifact_type,
                LifecycleState.BLOCKED,
                f"Blocked by unaccepted predecessor(s): {', '.join(blockers)}.",
                blockers=blockers,
            )
        return _verdict(artifact_type, LifecycleState.MISSING, f"Artifact type {artifact_type} is not created yet.")

    artifact_id = _string_or_none(artifact.artifact_id)
    if not _string_or_none(artifact.current_version_id):
        if blockers:
            return _verdict(
                artifact_type,
                LifecycleState.BLOCKED,
                f"Blocked by unaccepted predecessor(s): {', '.join(blockers)}.",
                artifact_id=artifact_id,
                blockers=blockers,
            )
        return _verdict(
            artifact_type,
            LifecycleState.MISSING,
            f"Artifact {artifact_id or artifact_type} has no current version.",
            artifact_id=artifact_id,
        )

    if _artifact_status(artifact.status) != "accepted":
        return _verdict(
            artifact_type,
            LifecycleState.IN_PROGRESS,
            f"Artifact {artifact_id or artifact_type} has a draft version that is not accepted yet.",
            artifact_id=artifact_id,
        )

    based_on = _normalize_based_on(artifact.based_on)
    predecessor_map = based_on_predecessors or {}
    orphan_reason = _orphan_reason(based_on, predecessor_map)
    if orphan_reason:
        return _verdict(artifact_type, LifecycleState.ORPHAN, orphan_reason, artifact_id=artifact_id)

    stale_reason = _stale_reason(based_on, predecessor_map)
    if stale_reason:
        return _verdict(artifact_type, LifecycleState.STALE, stale_reason, artifact_id=artifact_id)

    if blockers:
        return _verdict(
            artifact_type,
            LifecycleState.BLOCKED,
            f"Blocked by unaccepted predecessor(s): {', '.join(blockers)}.",
            artifact_id=artifact_id,
            blockers=blockers,
        )

    if based_on:
        reason = "Artifact is current; based_on predecessors match live versions."
    else:
        reason = "Artifact is current; it has no based_on predecessors."
    return _verdict(artifact_type, LifecycleState.CURRENT, reason, artifact_id=artifact_id)


def render_lifecycle_log_line(verdict: LifecycleVerdict, *, session_id: str | None = None) -> str:
    """Render a verdict in the same key=value shape as gate decision logs."""

    parts = [
        "gate=lifecycle_resolver",
        f"verdict={verdict.state.value}",
        f"state={verdict.state.value}",
        f"artifact_type={verdict.artifact_type!r}",
    ]
    if verdict.artifact_id:
        parts.append(f"artifact_id={verdict.artifact_id!r}")
    if session_id:
        parts.append(f"session_id={session_id}")
    if verdict.reason:
        parts.append(f"reason={verdict.reason!r}")
    actions = ",".join(action.value for action in sorted(verdict.allowed_actions, key=lambda item: item.value))
    parts.append(f"allowed_actions={actions!r}")
    if verdict.blockers:
        parts.append(f"blockers={','.join(verdict.blockers)!r}")
    return " ".join(parts)


def _verdict(
    artifact_type: str,
    state: LifecycleState,
    reason: str,
    *,
    artifact_id: str | None = None,
    blockers: Sequence[str] = (),
) -> LifecycleVerdict:
    return LifecycleVerdict(
        artifact_type=artifact_type,
        artifact_id=artifact_id,
        state=state,
        reason=reason,
        allowed_actions=ALLOWED_ACTIONS_BY_STATE[state],
        blockers=tuple(blockers),
    )


def _blocking_required_predecessors(predecessors: Sequence[StructuralPredecessorSnapshot]) -> tuple[str, ...]:
    blockers: list[str] = []
    for predecessor in predecessors:
        if not predecessor.found:
            blockers.append(predecessor.artifact_type)
            continue
        if _artifact_status(predecessor.status) != "accepted":
            blockers.append(predecessor.artifact_type)
            continue
        if not _string_or_none(predecessor.current_version_id):
            blockers.append(predecessor.artifact_type)
    return tuple(blockers)


def _orphan_reason(
    based_on: Mapping[str, str],
    predecessors: Mapping[str, BasedOnPredecessorSnapshot],
) -> str | None:
    for predecessor_id in sorted(based_on):
        predecessor = predecessors.get(predecessor_id)
        if predecessor is None:
            return f"Artifact is orphaned; predecessor {predecessor_id} was not supplied to the resolver."
        label = _predecessor_label(predecessor_id, predecessor)
        if not predecessor.found:
            return f"Artifact is orphaned; predecessor {label} was not found."
        if predecessor.retired or _artifact_status(predecessor.status) == "archived":
            return f"Artifact is orphaned; predecessor {label} is retired."
        if not _string_or_none(predecessor.current_version_id):
            return f"Artifact is orphaned; predecessor {label} has no live current version."
    return None


def _stale_reason(
    based_on: Mapping[str, str],
    predecessors: Mapping[str, BasedOnPredecessorSnapshot],
) -> str | None:
    for predecessor_id, based_on_version_id in sorted(based_on.items()):
        predecessor = predecessors[predecessor_id]
        live_version_id = _string_or_none(predecessor.current_version_id)
        if live_version_id != based_on_version_id:
            label = _predecessor_label(predecessor_id, predecessor)
            return (
                f"Artifact is stale; predecessor {label} moved from version "
                f"{based_on_version_id} to {live_version_id}."
            )
    return None


def _normalize_based_on(based_on: Mapping[str, str]) -> dict[str, str]:
    return {_string_or_none(key) or "": _string_or_none(value) or "" for key, value in based_on.items()}


def _predecessor_label(predecessor_id: str, predecessor: BasedOnPredecessorSnapshot) -> str:
    if predecessor.artifact_type:
        return f"{predecessor.artifact_type} {predecessor_id}"
    return predecessor_id


def _artifact_status(value: Any | None) -> str | None:
    if value is None:
        return None
    enum_value = getattr(value, "value", None)
    if enum_value is not None:
        return str(enum_value)
    return str(value)


def _string_or_none(value: Any | None) -> str | None:
    if value is None:
        return None
    return str(value)
