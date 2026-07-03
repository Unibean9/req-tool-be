"""Per-turn context loading: focus reconciliation, artifact reads, coverage, decision view."""

import hashlib
import uuid
from dataclasses import dataclass
from typing import Any

from langchain_core.runnables import RunnableConfig
from sqlalchemy import select

from app.config import settings
from app.documents.registry import children_of, container_for, output_contract
from app.graphs.policy import ancestor_types
from app.graphs.state import WorkflowState
from app.graphs.tools import read_artifacts, read_current_body
from app.models.agent import AgentSession
from app.services.document_service import DocumentService


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
            }
            effective_state = {**state, **focus_reset_update}

        # Batched into one query for the whole context type set instead of one round trip per type.
        artifacts = await read_artifacts(
            db=db,
            project_id=project_id,
            artifact_type=context_types,
            context={"workflow_area": effective_state["workflow_area"]},
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
    # Captured before the freshly-read draft below shadows it: state["draft_body"] on entry is
    # exactly what this node persisted as "draft_body" last turn (see the result dict further
    # down), so it doubles as the "last sent to the model" snapshot the diff needs at zero cost.
    previous_draft_body = state.get("draft_body")
    draft_body = draft["body"] if draft else None

    return TurnContext(
        effective_state=effective_state,
        focus_reset_update=focus_reset_update,
        artifacts=artifacts,
        coverage=coverage,
        draft_body=draft_body,
        previous_draft_body=previous_draft_body,
    )


# P7: cross-turn cache for the rendered decision-view block, keyed by (session_id, content
# fingerprint). The fingerprint is an md5 of the serialized decision_nodes, so ANY change to the
# graph produces a different key — invalidation is automatic and foolproof, no separate "invalidate
# on mutation" bookkeeping is needed. In-process, module-level dict only: it does NOT survive process
# restarts and does NOT help across multiple worker processes; it only skips redundant re-renders for
# a session that stays warm within one process's lifetime. Bounded via simple FIFO eviction so a
# long-running process can't grow this unboundedly across many sessions.
_DECISION_VIEW_CACHE: dict[tuple[str, str], str] = {}
_DECISION_VIEW_CACHE_MAX_ENTRIES = 512


def _decision_nodes_fingerprint(decision_nodes: dict[str, Any]) -> str:
    import json

    return hashlib.md5(json.dumps(decision_nodes, sort_keys=True).encode("utf-8")).hexdigest()


def _build_decision_view_block(state: WorkflowState, session_id: str | None = None) -> str:
    """Rendered decision-graph view shown as the live draft target.

    Cross-turn cached per session_id when provided (P7) — render_view is skipped when this session's
    decision_nodes are byte-identical to the last time this block was rendered for it. session_id is
    optional so callers without a session context (e.g. unit tests) still get correct, uncached output.
    """
    decision_nodes = state.get("decision_nodes") or {}
    if not decision_nodes:
        return ""

    cache_key = (session_id, _decision_nodes_fingerprint(decision_nodes)) if session_id else None
    if cache_key is not None and cache_key in _DECISION_VIEW_CACHE:
        return _DECISION_VIEW_CACHE[cache_key]

    from app.graphs.decision_graph import render_view

    view = render_view(decision_nodes, state.get("artifact_type") or "brd").strip()
    block = f"\n\nDRAFT IN PROGRESS (incrementally updated - reflects clarified points):\n{view}" if view else ""

    if cache_key is not None:
        if len(_DECISION_VIEW_CACHE) >= _DECISION_VIEW_CACHE_MAX_ENTRIES:
            _DECISION_VIEW_CACHE.pop(next(iter(_DECISION_VIEW_CACHE)))
        _DECISION_VIEW_CACHE[cache_key] = block
    return block


def _missing_required_headings(artifact_type: str, body: str) -> list[str]:
    try:
        contract = output_contract(artifact_type)
    except ValueError:
        return []
    return [heading for heading in contract.required_headings if heading not in body]


def _decision_view_can_hide_draft(state: WorkflowState, decision_view_block: str, draft_body: str | None) -> bool:
    if not decision_view_block or not settings.decision_graph_enabled:
        return False
    if not draft_body:
        return True
    return not _missing_required_headings(state.get("artifact_type") or "brd", decision_view_block)
