"""write_draft / finalize — the propose and done branches of the tool loop.

write_draft proposes an artifact draft (deterministic + readiness gates, then a PROPOSE_ARTIFACTS
interrupt); finalize closes the session behind the quality gate with a HITL confirmation. The
draft-cache (current_draft_body/_cached_draft_body/artifact_stage) and the pure-state gates
(_draft_hash_stale/_finalize_gate_open/_phase_signals/current_session_phase) stay in the coordinator;
this module reaches them — plus interrupt and log_gate_decision, which tests patch on the coordinator
namespace — through the module reference at call time, so the split changes no behavior and no test.
"""

import uuid
from typing import Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.documents.registry import output_contract
from app.graphs import agent_tools
from app.graphs.agent_tools._shared import (
    RecoverableToolError,
    _missing_required_arg_update,
    _recoverable_tool_update,
)
from app.graphs.decision_graph import (
    render_node_map,
    render_view,
    scan_parked_questions,
    synthesis_assumption_signals,
)
from app.graphs.gating import dispatch_rules
from app.graphs.policy import ancestor_types
from app.graphs.state import WorkflowState
from app.graphs.validators import validate_proposal
from app.models.agent import (
    AgentSession,
    AgentSessionInterruptType,
    AgentSessionStatus,
    AgentToolCall,
    AgentToolCallStatus,
)
from app.models.artifact import Artifact, ArtifactStatus, ArtifactVersion
from app.models.project import Project
from app.schemas.artifact_synthesis import ArtifactSynthesisMetadata, evaluate_candidate_readiness
from app.services.document_service import DocumentService


def _missing_required_headings(artifact_type: str, body: str) -> list[str]:
    try:
        contract = output_contract(artifact_type)
    except ValueError:
        return []
    return [heading for heading in contract.required_headings if heading not in str(body or "")]


def _has_complete_required_headings(artifact_type: str, body: str) -> bool:
    return bool(str(body or "").strip()) and not _missing_required_headings(artifact_type, body)


def _resolve_proposed_body(state: WorkflowState, body: str) -> str:
    """Pick the proposal body that is safe to canonicalize and persist."""
    decision_nodes = state.get("decision_nodes") or {}
    if not decision_nodes:
        return body
    artifact_type = state.get("artifact_type") or "brd"
    rendered = render_view(decision_nodes, artifact_type)
    if not _missing_required_headings(artifact_type, rendered):
        return rendered
    if _has_complete_required_headings(artifact_type, body):
        return body
    draft_body = state.get("draft_body") or ""
    if _has_complete_required_headings(artifact_type, draft_body):
        return draft_body
    return body


def _proposal_based_on(state: WorkflowState, artifact_type: str) -> dict[str, str]:
    predecessor_types = set(ancestor_types(artifact_type))
    based_on: dict[str, str] = {}
    for item in state.get("turn_context_artifacts") or []:
        artifact_id = str(item.get("id") or "").strip()
        artifact_type_value = str(item.get("type") or "").strip()
        version_id = str(item.get("current_version_id") or "").strip()
        if artifact_id and version_id and artifact_type_value in predecessor_types:
            based_on[artifact_id] = version_id
    return based_on


def _proposal_lifecycle_metadata(state: WorkflowState, artifact_type: str) -> dict[str, Any]:
    return {
        "based_on": _proposal_based_on(state, artifact_type),
        "decision_node_map": render_node_map(state.get("decision_nodes") or {}, artifact_type),
    }


AUTO_EVIDENCE_EXCERPT_MAX_CHARS = 1200


def _evidence_excerpt(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if len(text) > AUTO_EVIDENCE_EXCERPT_MAX_CHARS:
        return text[:AUTO_EVIDENCE_EXCERPT_MAX_CHARS]
    return text


def _proposal_source_evidence(state: WorkflowState, based_on: dict[str, str]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in state.get("source_context") or []:
        if not isinstance(item, dict):
            continue
        excerpt = _evidence_excerpt(item.get("excerpt"))
        if not excerpt:
            continue
        source_document_id = str(item.get("source_document_id") or "").strip()
        if source_document_id:
            key = ("source_document", source_document_id)
            if key in seen:
                continue
            seen.add(key)
            evidence.append(
                {
                    "source_document_id": source_document_id,
                    "source_type": "document",
                    "locator": f"source_document:{source_document_id}",
                    "excerpt": excerpt,
                    "confidence": 1.0,
                    "metadata": {
                        "source_kind": "source_document",
                        "title": item.get("title"),
                        "source_locator": item.get("locator"),
                    },
                }
            )
            continue

        predecessor_artifact_id = str(item.get("artifact_id") or "").strip()
        predecessor_version_id = str(item.get("predecessor_version_id") or "").strip()
        if not predecessor_artifact_id or not predecessor_version_id:
            continue
        if based_on.get(predecessor_artifact_id) != predecessor_version_id:
            continue
        key = ("predecessor_version", predecessor_version_id)
        if key in seen:
            continue
        seen.add(key)
        evidence.append(
            {
                "source_type": "ai_output",
                "locator": f"artifact_version:{predecessor_version_id}",
                "excerpt": excerpt,
                "confidence": 1.0,
                "metadata": {
                    "source_kind": "predecessor_version",
                    "predecessor_artifact_id": predecessor_artifact_id,
                    "predecessor_version_id": predecessor_version_id,
                    "title": item.get("title"),
                },
            }
        )
    return evidence


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        normalized = str(item or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


async def _stale_predecessor_warnings(
    db: AsyncSession,
    project_id: uuid.UUID | None,
    based_on: dict[str, str],
    predecessor_types: dict[str, str],
) -> list[str]:
    """Advisory, read-only check of each `based_on` predecessor's current version/status.

    Mirrors the 3 staleness reasons the approval-time guard uses, without locking any row — this is
    a non-blocking heads-up surfaced early so the reviewer and the model both see it sooner; approval
    remains the sole authority that can actually block a stale predecessor.
    """
    if not based_on:
        return []
    warnings: list[str] = []
    for predecessor_id, based_on_version_id in based_on.items():
        try:
            artifact_id = uuid.UUID(predecessor_id)
        except (TypeError, ValueError):
            continue
        query = select(Artifact).where(Artifact.id == artifact_id)
        if project_id is not None:
            query = query.where(Artifact.project_id == project_id)
        predecessor = (await db.execute(query)).scalar_one_or_none()
        if predecessor is None:
            artifact_type = predecessor_types.get(predecessor_id) or "unknown"
            warnings.append(f"stale_predecessor:{artifact_type}:missing_predecessor")
            continue
        current_version_id = str(predecessor.current_version_id) if predecessor.current_version_id else None
        if predecessor.status == ArtifactStatus.ARCHIVED:
            reason = "retired_predecessor"
        elif current_version_id != based_on_version_id:
            reason = "predecessor_version_changed"
        else:
            continue
        warnings.append(f"stale_predecessor:{predecessor.type.value}:{reason}")
    return warnings


def _cold_start_draft_blocked(state: WorkflowState) -> bool:
    if state.get("decision_nodes"):
        return False
    if (state.get("session_elicit_count") or 0) > 0:
        return False
    # turn_count resets to 0 on every human resume, so it cannot represent session depth.
    # user_confirmed=True means confirm_intent ran — at least one round of Q&A happened.
    if state.get("user_confirmed") is not None:
        return False
    return True


async def _write_draft_impl(
    title: str,
    body: str,
    state: WorkflowState,
    config: RunnableConfig,
    tool_call_id: str,
    curation_action: str | None = None,
    curation_justification: str | None = None,
):
    body = _resolve_proposed_body(state, body)
    if not str(body or "").strip():
        return _missing_required_arg_update("write_draft", "body", tool_call_id)
    lifecycle_verdict = dispatch_rules.LifecycleWriteDraftBlockRule().evaluate(
        {
            "name": "write_draft",
            "args": {"curation_action": curation_action, "curation_justification": curation_justification},
        },
        state,
    )
    if not lifecycle_verdict.is_allow:
        lifecycle_block = lifecycle_verdict.reason
        return _recoverable_tool_update(
            RecoverableToolError(
                code=lifecycle_block,
                message=f"Cannot write_draft: lifecycle state blocks this proposal ({lifecycle_block}).",
                recovery="Follow the Situation Report allowed actions before proposing a draft.",
            ),
            tool_call_id,
        )
    cold_start_verdict = dispatch_rules.ColdStartDraftBlockRule().evaluate({"name": "write_draft"}, state)
    if not cold_start_verdict.is_allow:
        return _recoverable_tool_update(
            RecoverableToolError(
                code="cold_start_requires_elicitation",
                message=(
                    "Cannot write_draft immediately from a thin cold start; use elicit/web_search "
                    "or create a decision_node first to record rationale, assumptions, and open questions."
                ),
            ),
            tool_call_id,
        )

    cfg = config["configurable"]
    session_factory = cfg["session_factory"]
    session_id = uuid.UUID(cfg["thread_id"])
    if not state.get("last_agent_run_id"):
        raise RuntimeError("write_draft requires last_agent_run_id in state — analyze_node must run first")
    run_id = uuid.UUID(state["last_agent_run_id"])
    focused_artifact_id = state.get("focused_artifact_id")
    if not focused_artifact_id:
        return _recoverable_tool_update(
            RecoverableToolError(
                code="missing_focused_artifact",
                message=(
                    "Cannot write_draft because state is missing focused_artifact_id; "
                    "select the document item to write before creating a proposal."
                ),
                user_fixable=True,
            ),
            tool_call_id,
        )
    tool_key = f"write_draft:{focused_artifact_id}"
    project_id = uuid.UUID(str(cfg["project_id"])) if cfg.get("project_id") else None

    async with session_factory() as db:
        try:
            focused = await DocumentService(db).get_document_item_artifact(
                artifact_id=uuid.UUID(str(focused_artifact_id)),
                project_id=project_id,
            )
        except ValueError as exc:
            raise RuntimeError("write_draft focused artifact must be an existing document item") from exc
        # Idempotency on (run_id, tool_name): a resume re-executes this body, so skip if the
        # proposed write already exists for this run. tool_name discriminates it from the enum
        # path's "create_artifact" rows — no new column, no migration (R3).
        existing_tool_call = (
            await db.execute(
                select(AgentToolCall).where(
                    AgentToolCall.run_id == run_id,
                    AgentToolCall.tool_name == tool_key,
                )
            )
        ).scalar_one_or_none()
        if existing_tool_call:
            readiness = dict((existing_tool_call.input_snapshot or {}).get("candidate_readiness") or {})
            # Resume-of-same-run: reuse the warnings already baked into the persisted snapshot on the
            # first write instead of re-querying — consistent with what was actually recorded and adds
            # no extra DB read on this path.
            existing_metadata = (existing_tool_call.input_snapshot or {}).get("synthesis_metadata") or {}
            stale_warnings = [
                warning
                for warning in existing_metadata.get("deterministic_warnings") or []
                if str(warning).startswith("stale_predecessor:")
            ]
        else:
            # Deterministic quality gate before the LLM path: violations block the proposal (no
            # PROPOSE_ARTIFACTS interrupt) and are returned for the model's repair loop; warnings ride
            # the synthesis metadata for the human reviewer (FE snapshot) and the model (write_draft
            # ToolMessage). Escape hatch: enforce_deterministic_gate.
            gate = validate_proposal(focused.type.value, {"title": title, "body": body})
            if settings.enforce_deterministic_gate and gate.violations:
                agent_tools.log_gate_decision(
                    "deterministic_proposal",
                    "blocked",
                    reason="; ".join(gate.violations),
                    session_id=str(session_id),
                )
                return _recoverable_tool_update(
                    RecoverableToolError(
                        code="deterministic_gate_failed",
                        message=(
                            "Draft blocked by the deterministic quality gate. Fix these before proposing: "
                            + "; ".join(gate.violations)
                        ),
                        user_fixable=True,
                    ),
                    tool_call_id,
                )
            agent_tools.log_gate_decision(
                "deterministic_proposal",
                "pass",
                reason=f"{len(gate.warnings)} warning(s)" if gate.warnings else None,
                session_id=str(session_id),
            )
            # Single source of truth: assumptions/open-questions are derived from the
            # decision graph only — no parallel state fields, no key-fact reconciliation shim.
            graph_confirmed, graph_pending = synthesis_assumption_signals(state.get("decision_nodes") or {})
            lifecycle_metadata = _proposal_lifecycle_metadata(state, focused.type.value)
            based_on = (
                lifecycle_metadata.get("based_on") if isinstance(lifecycle_metadata.get("based_on"), dict) else {}
            )
            source_evidence = _proposal_source_evidence(state, based_on)
            if curation_action or curation_justification:
                lifecycle_metadata["curation_decision"] = {
                    "action": str(curation_action or "").strip().upper(),
                    "justification": str(curation_justification or "").strip(),
                }
            predecessor_types = {
                str(item.get("id") or "").strip(): str(item.get("type") or "").strip()
                for item in state.get("turn_context_artifacts") or []
            }
            stale_warnings = await _stale_predecessor_warnings(db, project_id, based_on, predecessor_types)
            metadata = ArtifactSynthesisMetadata(
                artifact_type=focused.type.value,
                focused_artifact_id=focused.id,
                base_version_id=focused.current_version_id,
                evidence_refs=[f"agent_run:{run_id}", f"tool_call:{tool_call_id}"],
                inference_level="medium",
                confirmed_assumptions=_dedupe_keep_order(graph_confirmed),
                pending_assumptions=_dedupe_keep_order(graph_pending),
                deterministic_warnings=[*gate.warnings, *stale_warnings],
            )
            candidate_readiness = evaluate_candidate_readiness(
                artifact_type=focused.type.value,
                body=body,
                synthesis_metadata=metadata,
            )
            if not candidate_readiness.can_persist:
                reject_streak = (state.get("readiness_reject_streak") or 0) + 1
                reason = "; ".join(candidate_readiness.blocking_reasons) or "candidate is not persistable"
                agent_tools.log_gate_decision(
                    "candidate_readiness_propose",
                    "blocked",
                    reason=reason,
                    session_id=str(session_id),
                )
                if reject_streak >= 2:
                    recovery = (
                        "Do not retry write_draft again. Call ask_user, presenting these blocking "
                        f"reasons/missing items to the human so they can supply what's missing: {reason}"
                    )
                else:
                    recovery = f"Repair the draft body to address: {reason}"
                return _recoverable_tool_update(
                    RecoverableToolError(
                        code="candidate_readiness_not_ready",
                        message="Draft blocked: it would not pass the approval readiness check.",
                        recovery=recovery,
                        user_fixable=True,
                    ),
                    tool_call_id,
                    extra_update={"readiness_reject_streak": reject_streak},
                )
            agent_tools.log_gate_decision(
                "candidate_readiness_propose",
                "pass",
                session_id=str(session_id),
            )
            readiness = candidate_readiness.model_dump(mode="json")
            input_snapshot = {
                "artifact_type": focused.type.value,
                "focused_artifact_id": str(focused.id),
                "base_version_id": str(focused.current_version_id) if focused.current_version_id else None,
                "title": title,
                "body": body,
                "synthesis_metadata": metadata.model_dump(mode="json"),
                "lifecycle_metadata": lifecycle_metadata,
                "candidate_readiness": readiness,
            }
            if source_evidence:
                input_snapshot["source_evidence"] = source_evidence
            db.add(
                AgentToolCall(
                    run_id=run_id,
                    tool_name=tool_key,
                    input_snapshot=input_snapshot,
                    status=AgentToolCallStatus.PROPOSED,
                )
            )
        session_row = (await db.execute(select(AgentSession).where(AgentSession.id == session_id))).scalar_one()
        session_row.status = AgentSessionStatus.WAITING_FOR_HUMAN
        session_row.interrupt_type = AgentSessionInterruptType.PROPOSE_ARTIFACTS
        await db.commit()

    agent_tools.interrupt({"type": "propose_artifacts", "tool_name": "write_draft"})
    message_content = title
    if stale_warnings:
        message_content = title + "\n" + "\n".join(stale_warnings)
    return Command(
        update={
            "messages": [ToolMessage(content=message_content, tool_call_id=tool_call_id)],
            "draft_body": body,
            "candidate_readiness": readiness,
        }
    )


@tool
async def write_draft(
    title: Annotated[str, "Short title for the proposed artifact."],
    body: Annotated[
        str,
        "Full draft body in Markdown following the artifact's output contract (required headings); "
        "mark inferred / missing / needs_confirmation parts explicitly. Not a transcript or form dump. "
        "Used when the decision graph is still partial; a complete graph-rendered view takes precedence.",
    ],
    state: Annotated[dict, InjectedState],
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
    curation_action: Annotated[
        str | None,
        "Required only for STALE artifacts: ADD | UPDATE | SUPERSEDE | RETIRE | NOOP.",
    ] = None,
    curation_justification: Annotated[
        str | None,
        "Required only for STALE artifacts: short reason explaining the chosen curation action.",
    ] = None,
) -> Command:
    """Propose an artifact draft and pause for the user to review it.

    Use once enough confirmed information exists to produce a structured draft. Available only in the
    artifact phase (after confirm_intent). This is the propose/approval gate: when decision nodes
    exist and cover the contract, the proposal is the view rendered from the graph (record content via
    the decision-node tools, not here). Without a complete graph, supply the body — it grows
    incrementally, never rewritten from scratch.
    """
    return await _write_draft_impl(title, body, state, config, tool_call_id, curation_action, curation_justification)


def _open_blocker_questions(state: WorkflowState) -> list[dict[str, Any]]:
    """Parked open_questions whose blockers are resolved but the question itself is still unanswered.

    These are the resurfaced/actionable questions (see scan_parked_questions); dismissed or answered
    nodes are excluded by construction (dismiss sets status to "dismissed", never "parked").

    Short-circuits when the decision graph is disabled: the only tools that clear a blocker
    (update_decision_node / dismiss_question) are gated on the same flag, so gating finalize while
    they are unavailable would wedge the session with no path out.
    """
    if not settings.decision_graph_enabled:
        return []
    return scan_parked_questions(state.get("decision_nodes") or {})


# --- Executive summary synthesis (BRD finalize) -----------------------------
# executive_summary is no longer elicited; it is synthesized from the BRD's
# vision/problem/scope at finalize and promoted to a project field.
_EXEC_SUMMARY_SOURCES = ("problem_statement", "vision_objectives", "scope_capabilities")


def synthesize_executive_summary(sources: dict[str, str]) -> str:
    """Assemble a one-paragraph executive summary from BRD source sections.

    Deterministic template over problem/vision/scope, isolated so it can be
    swapped for an LLM synthesis and stubbed in tests. The finalize hook calls
    it exactly once (see `_persist_executive_summary_draft`), so even a
    non-deterministic implementation yields a stable persisted value across an
    interrupt/resume cycle.
    """
    labels = {
        "problem_statement": "Problem",
        "vision_objectives": "Vision & objectives",
        "scope_capabilities": "Scope",
    }
    parts = [
        f"{labels[key]}: {(sources.get(key) or '').strip()}"
        for key in _EXEC_SUMMARY_SOURCES
        if (sources.get(key) or "").strip()
    ]
    return " ".join(parts)


async def _load_accepted_bodies(db, project_id, types: tuple[str, ...]) -> dict[str, str]:
    rows = (
        await db.execute(
            select(Artifact.type, ArtifactVersion.body)
            .join(ArtifactVersion, Artifact.current_version_id == ArtifactVersion.id)
            .where(
                Artifact.project_id == project_id,
                Artifact.type.in_(list(types)),
                Artifact.status == ArtifactStatus.ACCEPTED,
            )
        )
    ).all()
    return {getattr(t, "value", t): (body or "") for t, body in rows}


async def _persist_executive_summary_draft(db, project_id) -> str | None:
    """Resume-safe synthesis: compute and persist the executive summary once.

    If the project already carries an executive_summary — persisted on a prior
    execution of this finalize node, before its interrupt — reuse it verbatim and
    never recompute, so a non-deterministic synthesis cannot drift across resume.
    The caller commits within its own DB block.
    """
    project = (await db.execute(select(Project).where(Project.id == project_id))).scalar_one_or_none()
    if project is None:
        return None
    if project.executive_summary:
        return project.executive_summary
    sources = await _load_accepted_bodies(db, project_id, _EXEC_SUMMARY_SOURCES)
    # Via the coordinator reference so a test patching agent_tools.synthesize_executive_summary
    # (the re-exported name) still intercepts the resume-stability call.
    draft = agent_tools.synthesize_executive_summary(sources)
    if not draft.strip():
        return None
    project.executive_summary = draft
    return draft


async def _apply_executive_summary_resume(project_id, resume, session_factory) -> None:
    """Apply the user's finalize confirmation to the synthesized draft.

    Resume payload shape (BRD finalize only): {"executive_summary_action": "edit",
    "executive_summary": "<text>"} overwrites; {"executive_summary_action": "reject"}
    clears the field; anything else (approve/default) keeps the persisted draft.
    """
    if not isinstance(resume, dict):
        return
    action = resume.get("executive_summary_action")
    if action not in ("edit", "reject"):
        return
    async with session_factory() as db:
        project = (await db.execute(select(Project).where(Project.id == project_id))).scalar_one_or_none()
        if project is None:
            return
        if action == "edit":
            new_text = str(resume.get("executive_summary") or "").strip()
            if new_text:
                project.executive_summary = new_text
        else:  # reject
            project.executive_summary = None
        await db.commit()


async def _finalize_impl(summary: str, state: WorkflowState, config: RunnableConfig, tool_call_id: str):
    if not str(summary or "").strip():
        return _missing_required_arg_update("finalize", "summary", tool_call_id)

    # Hard-block: even if the menu gate is bypassed, never finalize over a failing quality gate.
    # A missing report counts as "fail" — finalize requires a passing critique to exist.
    report = state.get("quality_report")
    if (
        not (await agent_tools.current_draft_body(state, config)).strip()
        or not report
        or report.get("quality_gate_result") != "pass"
        or not agent_tools._finalize_gate_open(state)
    ):
        blocking = (report or {}).get("blocking_issues") or []
        detail = "; ".join(blocking) if blocking else "no valid critique for the current draft"
        return _recoverable_tool_update(
            RecoverableToolError(
                code="finalize_gate_blocked",
                message=f"Cannot finalize: quality gate has not passed ({detail}).",
            ),
            tool_call_id,
        )

    # Loop-closure gate (pure state read, so it runs before any DB work): unresolved blocker-class
    # parked questions must be answered or explicitly dismissed before finalize.
    blockers = _open_blocker_questions(state)
    if blockers:
        offenders = "; ".join(f"{node['id']} ({node.get('statement', '')})" for node in blockers)
        return _recoverable_tool_update(
            RecoverableToolError(
                code="finalize_blocker_unresolved",
                message=f"Cannot finalize: {len(blockers)} blocker question(s) are unresolved: {offenders}.",
                recovery=(
                    "Resolve each (answer + confirm its blocker) or call dismiss_question with a reason, "
                    "then finalize."
                ),
            ),
            tool_call_id,
        )

    cfg = config["configurable"]
    session_factory = cfg["session_factory"]
    session_id = uuid.UUID(cfg["thread_id"])
    artifact_type = state.get("artifact_type") or ""
    project_id_raw = cfg.get("project_id")
    finalize_project_id = uuid.UUID(str(project_id_raw)) if project_id_raw else None
    exec_summary_draft: str | None = None

    async with session_factory() as db:
        if finalize_project_id is not None:
            project_id = finalize_project_id
            missing_predecessors: list[str] = []
            for pred_type in ancestor_types(artifact_type):
                count = (
                    await db.execute(
                        select(func.count(Artifact.id)).where(
                            Artifact.project_id == project_id,
                            Artifact.type == pred_type,
                            Artifact.status == ArtifactStatus.ACCEPTED,
                        )
                    )
                ).scalar() or 0
                if count == 0:
                    missing_predecessors.append(pred_type)
            if missing_predecessors:
                return _recoverable_tool_update(
                    RecoverableToolError(
                        code="finalize_predecessor_blocked",
                        message=(
                            "Cannot finalize: predecessor artifact is not accepted yet "
                            f"({', '.join(missing_predecessors)})."
                        ),
                    ),
                    tool_call_id,
                )

            # Finalizing the BRD container synthesizes the executive summary from
            # vision/problem/scope and persists it (once) BEFORE the interrupt, so
            # the resume path reuses the confirmed draft instead of recomputing.
            if artifact_type == "brd":
                exec_summary_draft = await _persist_executive_summary_draft(db, project_id)

        session_row = (await db.execute(select(AgentSession).where(AgentSession.id == session_id))).scalar_one()
        session_row.status = AgentSessionStatus.WAITING_FOR_HUMAN
        session_row.interrupt_type = AgentSessionInterruptType.ASK_HUMAN
        await db.commit()

    interrupt_payload: dict[str, Any] = {"type": "finalize", "message": summary}
    if exec_summary_draft:
        interrupt_payload["executive_summary"] = exec_summary_draft
    resume = agent_tools.interrupt(interrupt_payload)

    # Apply the user's confirmation of the synthesized executive summary (edit/reject);
    # approve/default keeps the persisted draft. Runs once (finalize resumes once).
    if artifact_type == "brd" and finalize_project_id is not None:
        await _apply_executive_summary_resume(finalize_project_id, resume, session_factory)

    return Command(update={"messages": [ToolMessage(content=summary, tool_call_id=tool_call_id)]})


@tool
async def finalize(
    summary: Annotated[str, "Closing summary of what was accomplished, in the user's locale."],
    state: Annotated[dict, InjectedState],
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Finalize the working session and pause for the user to confirm completion.

    Use only after a passing quality critique exists for the current draft (run_critique). It is the
    completion gate, not a substitute for write_draft. Blocked if the quality gate has not passed.
    """
    return await _finalize_impl(summary, state, config, tool_call_id)
