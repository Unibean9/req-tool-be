"""Per-turn context loading: focus reconciliation, artifact reads, coverage, decision view."""

import uuid
from dataclasses import dataclass
from typing import Any

from langchain_core.runnables import RunnableConfig
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.documents.registry import children_of, container_for, output_contract
from app.graphs.gate_logging import log_gate_decision
from app.graphs.lifecycle_resolver import (
    ArtifactLifecycleSnapshot,
    BasedOnPredecessorSnapshot,
    StructuralPredecessorSnapshot,
    resolve_lifecycle,
)
from app.graphs.policy import ancestor_types
from app.graphs.state import WorkflowState
from app.graphs.tools import read_artifacts, read_current_body
from app.models.agent import AgentSession
from app.models.artifact import Artifact, ArtifactType, ArtifactVersion
from app.services.document_service import DocumentService

_KNOWN_ARTIFACT_TYPES = frozenset(item.value for item in ArtifactType)


async def _document_coverage(
    *,
    db,
    project_id: uuid.UUID,
    artifact_type: str,
    focused_artifact_id: uuid.UUID | None,
) -> dict[str, Any]:
    return await DocumentService(db).document_coverage(
        project_id=project_id,
        artifact_type=artifact_type,
        focused_artifact_id=focused_artifact_id,
    )


@dataclass
class TurnContext:
    effective_state: WorkflowState
    focus_reset_update: dict[str, Any]
    artifacts: list[dict[str, Any]]
    lifecycle_reports: list[dict[str, Any]]
    artifact_history: list[dict[str, Any]]
    coverage: dict[str, Any]
    draft_body: str | None
    previous_draft_body: str | None


def _dedupe_types(types: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in types:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _same_container_source_types(artifact_type: str) -> list[str]:
    container = container_for(artifact_type)
    if container is None:
        return []
    try:
        return [item for item in children_of(container) if item != artifact_type]
    except ValueError:
        return []


def _context_artifact_types(artifact_type: str) -> list[str]:
    """Artifact rows worth exposing as source candidates for this turn.

    Predecessors remain the authoritative finalize/session dependency gate. Same-container document
    items are context only: they let the model resolve references like "based on Executive Summary"
    to an existing artifact id before asking the user to paste content.
    """
    return _dedupe_types(
        [
            artifact_type,
            *ancestor_types(artifact_type),
            *_same_container_source_types(artifact_type),
        ]
    )


def _enum_value(value: Any) -> str | None:
    if value is None:
        return None
    enum_value = getattr(value, "value", None)
    return str(enum_value if enum_value is not None else value)


def _artifact_type(value: Any) -> str:
    return str(_enum_value(value) or "")


def _current_metadata(artifact: Artifact | None) -> dict[str, Any]:
    if artifact is None or artifact.current_version is None:
        return {}
    metadata = artifact.current_version.extra_metadata or {}
    return metadata if isinstance(metadata, dict) else {}


def _based_on_from_artifact(artifact: Artifact | None) -> dict[str, str]:
    metadata = _current_metadata(artifact)
    based_on = metadata.get("based_on")
    if not isinstance(based_on, dict):
        return {}
    return {str(key): str(value) for key, value in based_on.items() if key and value}


def _report_from_verdict(verdict, title: str | None = None) -> dict[str, Any]:
    report = {
        "artifact_type": verdict.artifact_type,
        "artifact_id": verdict.artifact_id,
        "state": verdict.state.value,
        "reason": verdict.reason,
        "allowed_actions": sorted(action.value for action in verdict.allowed_actions),
        "blockers": list(verdict.blockers),
    }
    if title:
        report["title"] = title
    return report


async def _based_on_predecessors(
    db,
    *,
    project_id: uuid.UUID,
    based_on: dict[str, str],
) -> dict[str, BasedOnPredecessorSnapshot]:
    if not based_on:
        return {}
    predecessor_ids: list[uuid.UUID] = []
    for raw_id in sorted(based_on):
        try:
            predecessor_ids.append(uuid.UUID(str(raw_id)))
        except (TypeError, ValueError):
            continue
    if not predecessor_ids:
        return {}

    rows = (
        (
            await db.execute(
                select(Artifact)
                .where(Artifact.project_id == project_id)
                .where(Artifact.id.in_(predecessor_ids))
                .options(selectinload(Artifact.current_version))
            )
        )
        .scalars()
        .all()
    )
    rows_by_id = {str(row.id): row for row in rows}
    snapshots: dict[str, BasedOnPredecessorSnapshot] = {}
    for raw_id in sorted(based_on):
        row = rows_by_id.get(str(raw_id))
        if row is None:
            snapshots[str(raw_id)] = BasedOnPredecessorSnapshot(artifact_id=str(raw_id), found=False)
            continue
        snapshots[str(raw_id)] = BasedOnPredecessorSnapshot(
            artifact_id=str(row.id),
            artifact_type=_artifact_type(row.type),
            status=_enum_value(row.status),
            current_version_id=str(row.current_version_id) if row.current_version_id else None,
            found=True,
        )
    return snapshots


def _structural_predecessors(
    artifact_type: str,
    artifacts_by_type: dict[str, list[Artifact]],
) -> list[StructuralPredecessorSnapshot]:
    snapshots: list[StructuralPredecessorSnapshot] = []
    for predecessor_type in ancestor_types(artifact_type):
        accepted = next(
            (
                item
                for item in artifacts_by_type.get(predecessor_type, [])
                if _enum_value(item.status) == "accepted" and item.current_version_id is not None
            ),
            None,
        )
        if accepted is None:
            snapshots.append(StructuralPredecessorSnapshot(artifact_type=predecessor_type, found=False))
            continue
        snapshots.append(
            StructuralPredecessorSnapshot(
                artifact_type=predecessor_type,
                artifact_id=str(accepted.id),
                status=_enum_value(accepted.status),
                current_version_id=str(accepted.current_version_id) if accepted.current_version_id else None,
                found=True,
            )
        )
    return snapshots


async def _load_lifecycle_reports(
    db,
    *,
    project_id: uuid.UUID,
    context_types: list[str],
    session_id: str,
) -> list[dict[str, Any]]:
    # Only DB-enum types can be queried; a synthetic/unknown type would raise a LookupError on the
    # IN clause. Mirrors read_artifacts' known-type guard so the report set matches the loaded rows.
    known_types = [item for item in context_types if item in _KNOWN_ARTIFACT_TYPES]
    if not known_types:
        return []
    rows = (
        (
            await db.execute(
                select(Artifact)
                .where(Artifact.project_id == project_id)
                .where(Artifact.type.in_(known_types))
                .options(selectinload(Artifact.current_version))
                .order_by(Artifact.created_at, Artifact.id)
            )
        )
        .scalars()
        .all()
    )
    artifacts_by_type: dict[str, list[Artifact]] = {}
    for row in rows:
        artifacts_by_type.setdefault(_artifact_type(row.type), []).append(row)

    reports: list[dict[str, Any]] = []
    for artifact_type in known_types:
        structural = _structural_predecessors(artifact_type, artifacts_by_type)
        artifacts = artifacts_by_type.get(artifact_type) or [None]
        for artifact in artifacts:
            based_on = _based_on_from_artifact(artifact)
            predecessors = await _based_on_predecessors(db, project_id=project_id, based_on=based_on)
            snapshot = (
                None
                if artifact is None
                else ArtifactLifecycleSnapshot(
                    artifact_type=artifact_type,
                    artifact_id=str(artifact.id),
                    status=_enum_value(artifact.status),
                    current_version_id=str(artifact.current_version_id) if artifact.current_version_id else None,
                    based_on=based_on,
                )
            )
            verdict = resolve_lifecycle(
                artifact_type,
                snapshot,
                required_predecessors=structural,
                based_on_predecessors=predecessors,
            )
            report = _report_from_verdict(verdict, title=artifact.title if artifact is not None else None)
            log_gate_decision(
                "lifecycle_report",
                verdict.state.value,
                reason=verdict.reason,
                session_id=session_id,
                extra={
                    "artifact_type": artifact_type,
                    "artifact_id": verdict.artifact_id,
                    "allowed_actions": report["allowed_actions"],
                },
            )
            reports.append(report)
    return reports


async def _load_artifact_history(db, *, artifact_ids: list[str]) -> list[dict[str, Any]]:
    parsed_ids: list[uuid.UUID] = []
    for raw_id in artifact_ids:
        try:
            parsed_ids.append(uuid.UUID(str(raw_id)))
        except (TypeError, ValueError):
            continue
    if not parsed_ids:
        return []
    rows = (
        (
            await db.execute(
                select(ArtifactVersion, Artifact.type)
                .join(Artifact, ArtifactVersion.artifact_id == Artifact.id)
                .where(ArtifactVersion.artifact_id.in_(parsed_ids))
                .order_by(ArtifactVersion.artifact_id, ArtifactVersion.version_number.desc())
            )
        )
        .all()
    )
    seen_by_artifact: dict[str, int] = {}
    history: list[dict[str, Any]] = []
    for version, artifact_type in rows:
        artifact_id = str(version.artifact_id)
        if seen_by_artifact.get(artifact_id, 0) >= 3:
            continue
        seen_by_artifact[artifact_id] = seen_by_artifact.get(artifact_id, 0) + 1
        history.append(
            {
                "artifact_id": artifact_id,
                "artifact_type": _artifact_type(artifact_type),
                "version_id": str(version.id),
                "version_number": version.version_number,
                "change_source": _enum_value(version.change_source),
            }
        )
    return history


async def load_turn_context(state: WorkflowState, config: RunnableConfig) -> TurnContext:
    """Load everything analyze_node needs from the DB for this turn (one session scope)."""
    cfg = config["configurable"]
    session_factory = cfg["session_factory"]
    session_id = uuid.UUID(cfg["thread_id"])
    project_id = uuid.UUID(cfg["project_id"])

    effective_state: WorkflowState = state
    focus_reset_update: dict[str, Any] = {}

    # Context for the analyst = artifacts of the current type, its full transitive ancestry, and
    # same-container document items. Ancestors preserve dependency provenance; same-container rows
    # let the model resolve user references like "based on Executive Summary" before asking for
    # pasted content. Dedup keeps this token-light because read_artifacts returns title-only rows.
    artifact_type = state["artifact_type"]
    context_types = _context_artifact_types(artifact_type)
    async with session_factory() as db:
        db_focused_artifact_id = (
            await db.execute(select(AgentSession.focused_artifact_id).where(AgentSession.id == session_id))
        ).scalar_one_or_none()
        state_focused_artifact_id = (
            uuid.UUID(str(state["focused_artifact_id"])) if state.get("focused_artifact_id") else None
        )
        if db_focused_artifact_id != state_focused_artifact_id:
            focus_reset_update = {
                "focused_artifact_id": (str(db_focused_artifact_id) if db_focused_artifact_id is not None else None),
                "critique_rounds": 0,
                "quality_report": None,
                "last_critiqued_draft_hash": None,
                "candidate_readiness": None,
                "feedback_summary": None,
                "verification_status": None,
                "latest_checked_revision": None,
                "source_context": [{"__reset__": True}],
            }
            effective_state = {**state, **focus_reset_update}

        # Batched into one query for the whole context type set instead of one round trip per type.
        artifacts = await read_artifacts(
            db=db,
            project_id=project_id,
            artifact_type=context_types,
            context={"workflow_area": effective_state["workflow_area"]},
        )
        lifecycle_reports = await _load_lifecycle_reports(
            db,
            project_id=project_id,
            context_types=context_types,
            session_id=str(session_id),
        )
        artifact_history = await _load_artifact_history(
            db,
            artifact_ids=[str(item.get("artifact_id")) for item in lifecycle_reports if item.get("artifact_id")],
        )
        # Load the current draft body for this artifact_type so the analyst can mine
        # the delta instead of re-asking what the draft already records (M7/M8).
        draft = await read_current_body(
            db=db,
            project_id=project_id,
            artifact_type=artifact_type,
            artifact_id=db_focused_artifact_id,
        )
        coverage = await _document_coverage(
            db=db,
            project_id=project_id,
            artifact_type=artifact_type,
            focused_artifact_id=db_focused_artifact_id,
        )
        previous_accepted = sum(
            1 for value in (effective_state.get("section_coverage") or {}).values() if value == "filled"
        )
        current_accepted = sum(1 for value in (coverage.get("section_coverage") or {}).values() if value == "filled")
        if (
            coverage["coverage_complete"]
            or current_accepted > previous_accepted
            or effective_state.get("section_coverage") is None
        ):
            coverage["section_coverage_stall_count"] = 0
        else:
            coverage["section_coverage_stall_count"] = (effective_state.get("section_coverage_stall_count") or 0) + 1
        effective_state = {**effective_state, **coverage}
        effective_state = {
            **effective_state,
            "lifecycle_reports": lifecycle_reports,
            "artifact_history": artifact_history,
        }
    # Captured before the freshly-read draft below shadows it: state["draft_body"] on entry is
    # exactly what this node persisted as "draft_body" last turn (see the result dict further
    # down), so it doubles as the "last sent to the model" snapshot the diff needs at zero cost.
    previous_draft_body = state.get("draft_body")
    draft_body = draft["body"] if draft else None

    return TurnContext(
        effective_state=effective_state,
        focus_reset_update=focus_reset_update,
        artifacts=artifacts,
        lifecycle_reports=lifecycle_reports,
        artifact_history=artifact_history,
        coverage=coverage,
        draft_body=draft_body,
        previous_draft_body=previous_draft_body,
    )


def _missing_required_headings(artifact_type: str, body: str) -> list[str]:
    try:
        contract = output_contract(artifact_type)
    except ValueError:
        return []
    return [heading for heading in contract.required_headings if heading not in body]
