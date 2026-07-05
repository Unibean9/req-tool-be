"""Native LangGraph tools — the sole analyst dispatch path.

`analyze_node` binds these directly via provider tool-calling; the model picks among them and the
parallel ToolNode runs the choice. Each tool is a thin `@tool` over a plain async impl so the impls
stay unit-testable without a Runtime.

Idempotency on resume — LangGraph re-executes a ToolNode body from the top when its interrupt is
resumed: ask_user keys its message insert on the per-invocation ToolCall.id; approval proposals key
their AgentToolCall rows on (run_id, proposal-specific tool_name), reusing the existing
AgentToolCall.tool_name column (no migration). finalize has no insert to dedup — its only DB write is
an idempotent-by-value session status update — so it needs no key.
"""

import hashlib
import json
import logging
import re
import uuid
from typing import Annotated, Any, Literal

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command, interrupt
from sqlalchemy import exists, func, select

from app.config import settings
from app.documents.registry import children_of, output_contract, status_score
from app.graphs import interrupts
from app.graphs.decision_graph import (
    VALID_KINDS,
    VALID_STATUSES,
    create_node,
    dismiss_node,
    get_dependents,
    impact,
    infer_cascade_mode,
    render_node_map,
    render_view,
    scan_parked_questions,
    supersede_node,
    synthesis_assumption_signals,
    update_node,
)
from app.graphs.gate_logging import log_gate_decision
from app.graphs.lifecycle_context import filter_lifecycle_menu_tools, lifecycle_tool_block_reason
from app.graphs.note_parser import extract_structured_objects
from app.graphs.policy import ancestor_types
from app.graphs.session_phase import PhaseSignals, derive_phase, phase_allows
from app.graphs.state import QualityReport, WorkflowState
from app.graphs.tools import read_current_body
from app.graphs.tools import read_source_documents as read_source_documents_query
from app.graphs.validators import validate_proposal
from app.models.agent import (
    AgentSession,
    AgentSessionInterruptType,
    AgentSessionStatus,
    AgentToolCall,
    AgentToolCallStatus,
)
from app.models.artifact import Artifact, ArtifactLink, ArtifactStatus, ArtifactVersion, RelationType
from app.models.project import Project
from app.schemas.artifact import ArtifactLinkCreateRequest
from app.schemas.artifact_synthesis import (
    ArtifactReadinessState,
    ArtifactSynthesisMetadata,
    evaluate_candidate_readiness,
)
from app.services.document_service import DocumentService

logger = logging.getLogger(__name__)


class RecoverableToolError(Exception):
    def __init__(
        self, *, code: str, message: str, user_fixable: bool = False, recovery: str | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.user_fixable = user_fixable
        # A short, code-specific imperative telling the model how to recover. Optional: absent
        # recovery keeps the legacy behavior (the message alone), so old checkpoints are unchanged.
        self.recovery = recovery


def _recoverable_tool_update(exc: RecoverableToolError, tool_call_id: str) -> Command:
    logger.info(
        "tool-error code=%s classification=recoverable user_fixable=%s message=%s",
        exc.code,
        exc.user_fixable,
        exc.message,
    )
    entry: dict[str, Any] = {
        "code": exc.code,
        "classification": "recoverable",
        "user_fixable": exc.user_fixable,
        "message": exc.message,
    }
    tool_message = exc.message
    if exc.recovery:
        entry["recovery"] = exc.recovery
        tool_message = f"{exc.message} {exc.recovery}"
    return Command(
        update={
            "tool_errors": [entry],
            "messages": [ToolMessage(content=tool_message, tool_call_id=tool_call_id, status="error")],
        }
    )


def _missing_required_arg_update(tool_name: str, arg_name: str, tool_call_id: str) -> Command:
    return _recoverable_tool_update(
        RecoverableToolError(
            code="missing_required_arg",
            message=f"Cannot {tool_name}: missing required field '{arg_name}'.",
            recovery=f"Provide '{arg_name}' and call {tool_name} again.",
        ),
        tool_call_id,
    )


def _tool_not_available_update(tool_name: str, message: str, tool_call_id: str) -> Command:
    return _recoverable_tool_update(
        RecoverableToolError(
            code="tool_not_available",
            message=f"Cannot {tool_name}: {message}",
            recovery="This tool is not offered in the current phase; pick from the tools offered this turn.",
        ),
        tool_call_id,
    )


def _tool_is_available(state: WorkflowState, tool_name: str) -> bool:
    return any(t.name == tool_name for t in get_available_tools(state))


# ---------------------------------------------------------------------------
# Artifact lifecycle — single source for the draft body and a derived stage
# ---------------------------------------------------------------------------


async def current_draft_body(
    state: WorkflowState,
    config: RunnableConfig | None = None,
) -> str:
    """Canonical draft body for the whole tool-loop.

    Every read site — the critique target, the finalize-gate hash, has-draft checks — MUST route
    through here. The decision graph wins when it has nodes; a DB-loaded draft is the fallback for
    resumed document sessions whose graph has not been rebuilt yet.
    """
    _ = config
    return _current_draft_body_sync(state)


def _cached_draft_body(state: WorkflowState) -> str:
    """Synchronous draft view used by menu construction and finalize hashing."""
    return _current_draft_body_sync(state)


def _current_draft_body_sync(state: WorkflowState) -> str:
    decision_nodes = state.get("decision_nodes") or {}
    artifact_type = state.get("artifact_type") or "brd"
    if decision_nodes:
        return render_view(decision_nodes, artifact_type)
    return str(state.get("draft_body") or "")


async def artifact_stage(
    state: WorkflowState,
    config: RunnableConfig | None = None,
) -> str:
    """Derived lifecycle stage, read-only, computed from existing state — no new persisted field.

    "empty" -> no draft yet; "drafting" -> a draft exists but no critique has scored it;
    "critiqued" -> at least one critique round but the gate has not passed; "gate_passed" -> the
    latest quality report passed. One shared vocabulary for the menu and validation to reason over.
    """
    if not (await current_draft_body(state, config)).strip():
        return "empty"
    report = state.get("quality_report")
    if report and report.get("quality_gate_result") == "pass":
        return "gate_passed"
    if (state.get("critique_rounds") or 0) > 0:
        return "critiqued"
    return "drafting"


# ---------------------------------------------------------------------------
# Shared audit helper for interaction tools (ask_user / respond / confirm_intent)
# ---------------------------------------------------------------------------


async def _audit_interaction_tool_call(
    state: WorkflowState,
    config: RunnableConfig,
    *,
    tool_name: str,
    message: str,
) -> None:
    """Best-effort AgentToolCall row for interaction tools (ask_user/respond/confirm_intent)."""
    run_id_raw = state.get("last_agent_run_id")
    if not run_id_raw:
        return
    try:
        session_factory = config["configurable"]["session_factory"]
        async with session_factory() as db:
            already = (
                await db.execute(
                    select(
                        exists().where(
                            AgentToolCall.run_id == uuid.UUID(str(run_id_raw)),
                            AgentToolCall.tool_name == tool_name,
                        )
                    )
                )
            ).scalar()
            if not already:
                db.add(
                    AgentToolCall(
                        run_id=uuid.UUID(str(run_id_raw)),
                        tool_name=tool_name,
                        input_snapshot={"message": message},
                        status=AgentToolCallStatus.EXECUTED,
                    )
                )
                await db.commit()
    except Exception as exc:  # noqa: BLE001
        logger.warning("interaction tool audit persist failed (%s): %s", tool_name, exc)


# ---------------------------------------------------------------------------
# ask_user — parity for the `ask` enum branch
# ---------------------------------------------------------------------------


# Batched question form: up to 3 related, typed facets in one interrupt.
_MAX_BATCH_QUESTIONS = 3
_BATCH_QUESTION_TYPES = frozenset({"choice", "text", "confirm"})


def _normalize_batch_questions(questions: Any) -> list[dict[str, Any]]:
    """Keep at most 3 well-formed questions; drop malformed entries rather than erroring the turn.

    A valid entry has a non-empty `prompt` and a `type` in {choice, text, confirm} (defaulting to
    `text`); `choice` entries keep their string `options`. Anything else is silently dropped so a
    partially-malformed batch still asks its good questions instead of failing the whole tool call.
    """
    if not isinstance(questions, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in questions:
        if not isinstance(item, dict):
            continue
        prompt = str(item.get("prompt") or "").strip()
        if not prompt:
            continue
        q_type = str(item.get("type") or "text").strip().lower()
        if q_type not in _BATCH_QUESTION_TYPES:
            q_type = "text"
        entry: dict[str, Any] = {"prompt": prompt, "type": q_type}
        if q_type == "choice":
            entry["options"] = [str(opt).strip() for opt in (item.get("options") or []) if str(opt).strip()]
        normalized.append(entry)
        if len(normalized) >= _MAX_BATCH_QUESTIONS:
            break
    return normalized


def _render_batched_question_text(message: str, questions: list[dict[str, Any]]) -> str:
    """Joined-text fallback: the header (if any) followed by each numbered question, so a client
    that renders only free text still surfaces every facet."""
    lines: list[str] = []
    header = str(message or "").strip()
    if header:
        lines.append(header)
    for index, question in enumerate(questions, start=1):
        line = f"{index}. {question['prompt']}"
        if question["type"] == "choice" and question.get("options"):
            line += f" ({' / '.join(question['options'])})"
        lines.append(line)
    return "\n".join(lines)


async def _ask_user_impl(
    message: str,
    state: WorkflowState,
    config: RunnableConfig,
    tool_call_id: str,
    questions: Any = None,
):
    batch = _normalize_batch_questions(questions)
    if not str(message or "").strip() and not batch:
        return _missing_required_arg_update("ask_user", "message", tool_call_id)
    content = _render_batched_question_text(message, batch) if batch else message

    # ToolCall.id is the correct idempotency key here: inside the ToolNode body
    # state["last_agent_run_id"] still belongs to the prior analyze_node, not this invocation.
    # interrupt_kind="stream_response": session stays ACTIVE so the conversation resume path applies
    # (not the approval-gate path). The graph still halts via interrupt() — only the DB fields differ.
    await _audit_interaction_tool_call(state, config, tool_name=f"ask_user:{tool_call_id}", message=content)
    user_content = await interrupts._save_and_interrupt_ask(
        state, config, content, run_id=tool_call_id, interrupt_kind="stream_response", questions=batch or None
    )
    return Command(
        update={
            "messages": [
                ToolMessage(content=user_content, tool_call_id=tool_call_id),
                {"role": "user", "content": user_content},
            ]
        }
    )


@tool
async def ask_user(
    message: Annotated[
        str,
        "A short header/lead-in, written in the user's locale. May be empty when `questions` is given.",
    ],
    state: Annotated[dict, InjectedState],
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
    questions: Annotated[
        list[dict] | None,
        "Optional batch of up to 3 RELATED questions asked in one turn. Each item: "
        "{prompt: str, type: 'choice'|'text'|'confirm', options?: [str] for choice}. Batch related "
        "facets of one topic; ask serially (single question, empty list) only when one answer "
        "determines the next question.",
    ] = None,
) -> Command:
    """Ask the user for information and pause for their reply.

    Use when you need information you do not have and cannot reasonably infer. Do NOT use to deliver
    an opinion or assessment (use respond) or to present a prepared draft (use write_draft /
    confirm_intent). Keep to ONE topic: either one focused question, or up to 3 related facets of
    the same topic via `questions` — never an open-ended checklist.
    """
    return await _ask_user_impl(message, state, config, tool_call_id, questions)


# ---------------------------------------------------------------------------
# confirm_intent — one-shot intent phase transition
# ---------------------------------------------------------------------------


async def _confirm_intent_impl(
    summary: str,
    state: WorkflowState,
    config: RunnableConfig,
    tool_call_id: str,
) -> Command:
    if not str(summary or "").strip():
        return _missing_required_arg_update("confirm_intent", "summary", tool_call_id)
    if state.get("user_confirmed") is not None:
        return _tool_not_available_update(
            "confirm_intent",
            "intent is already confirmed; use ask_user/respond/write_draft for the current phase.",
            tool_call_id,
        )

    # interrupt_kind="stream_response" keeps the session ACTIVE (D4): the user can reply, and the
    # next turn sees user_confirmed=True — which unlocks the artifact tool menu in get_available_tools.
    # kind="assessment": this is a surfaced intent summary, not a clarifying question.
    await _audit_interaction_tool_call(state, config, tool_name=f"confirm_intent:{tool_call_id}", message=summary)
    user_content = await interrupts._save_and_interrupt_ask(
        state, config, summary, run_id=tool_call_id, kind="assessment", interrupt_kind="stream_response"
    )
    return Command(
        update={
            "user_confirmed": True,
            "messages": [
                ToolMessage(content=user_content, tool_call_id=tool_call_id),
                {"role": "user", "content": user_content},
            ],
        }
    )


@tool
async def confirm_intent(
    summary: Annotated[
        str,
        "A short restatement of the user's goal/intent for them to confirm or correct, in their locale.",
    ],
    state: Annotated[dict, InjectedState],
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Present a concise summary of the user's intent and pause for confirmation.

    Use in the intent phase, before any artifact work, to confirm you understood what they want to
    build. This is the one-shot gate into the artifact phase. Not for clarifying questions (use
    ask_user) and not for presenting a full draft (use write_draft).
    """
    return await _confirm_intent_impl(summary, state, config, tool_call_id)


# ---------------------------------------------------------------------------
# write_draft — parity for the `propose` enum branch
# ---------------------------------------------------------------------------


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
    lifecycle_block = lifecycle_tool_block_reason(
        state,
        "write_draft",
        {"curation_action": curation_action, "curation_justification": curation_justification},
    )
    if lifecycle_block:
        log_gate_decision("lifecycle_tool_impl", "blocked", reason=lifecycle_block)
        return _recoverable_tool_update(
            RecoverableToolError(
                code=lifecycle_block,
                message=f"Cannot write_draft: lifecycle state blocks this proposal ({lifecycle_block}).",
                recovery="Follow the Situation Report allowed actions before proposing a draft.",
            ),
            tool_call_id,
        )
    if _cold_start_draft_blocked(state):
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

    async with session_factory() as db:
        try:
            focused = await DocumentService(db).get_document_item_artifact(
                artifact_id=uuid.UUID(str(focused_artifact_id)),
                project_id=uuid.UUID(str(cfg["project_id"])) if cfg.get("project_id") else None,
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
        else:
            # Deterministic quality gate before the LLM path: violations block the proposal (no
            # PROPOSE_ARTIFACTS interrupt) and are returned for the model's repair loop; warnings ride
            # the synthesis metadata for the critique judge. Escape hatch: enforce_deterministic_gate.
            gate = validate_proposal(focused.type.value, {"title": title, "body": body})
            if settings.enforce_deterministic_gate and gate.violations:
                log_gate_decision(
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
            log_gate_decision(
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
            metadata = ArtifactSynthesisMetadata(
                artifact_type=focused.type.value,
                focused_artifact_id=focused.id,
                base_version_id=focused.current_version_id,
                evidence_refs=[f"agent_run:{run_id}", f"tool_call:{tool_call_id}"],
                inference_level="medium",
                confirmed_assumptions=_dedupe_keep_order(graph_confirmed),
                pending_assumptions=_dedupe_keep_order(graph_pending),
                deterministic_warnings=gate.warnings,
            )
            readiness = evaluate_candidate_readiness(
                artifact_type=focused.type.value,
                body=body,
                synthesis_metadata=metadata,
            ).model_dump(mode="json")
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

    interrupt({"type": "propose_artifacts", "tool_name": "write_draft"})
    return Command(
        update={
            "messages": [ToolMessage(content=title, tool_call_id=tool_call_id)],
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


# ---------------------------------------------------------------------------
# finalize — parity for the `done` enum branch, with a HITL confirmation gate
# ---------------------------------------------------------------------------


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
    draft = synthesize_executive_summary(sources)
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
        not (await current_draft_body(state, config)).strip()
        or not report
        or report.get("quality_gate_result") != "pass"
        or not _finalize_gate_open(state)
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
    resume = interrupt(interrupt_payload)

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


# ---------------------------------------------------------------------------
# critique_note / explore_note — scratchpad notes
# (no interrupt, no DB, no approval)
# ---------------------------------------------------------------------------
# Splitting the former single write_note into two named angles makes the analytical move a
# first-class menu choice, so the analyst can record a critique and an exploration angle in the
# same turn rather than committing to one operating mode.


async def _write_note_impl(content: str, state: WorkflowState, tool_call_id: str, tool_name: str):
    if not str(content or "").strip():
        return _missing_required_arg_update(tool_name, "content", tool_call_id)
    if not _tool_is_available(state, tool_name):
        return _tool_not_available_update(
            tool_name,
            "note step limit reached; ask the user, respond, or switch tools instead of adding more notes.",
            tool_call_id,
        )

    # The note text lives in the message history (decision 3): no `notes` state field, no DB row.
    # Beyond that, tagged lines are parsed into structured objects. risks/key_facts carry additive
    # reducers, so emit ONLY the new entries — returning prior+new would re-append every existing
    # entry through the reducer. ASSUMPTION:/OPEN_QUESTION: lines instead create decision nodes
    # because the graph is the single source of truth for assumptions/open-questions — assumptions
    # land needs_confirmation (agent-authored, not user-confirmed), open_questions proposed.
    extracted = extract_structured_objects(content)
    update: dict[str, Any] = {"messages": [ToolMessage(content=content, tool_call_id=tool_call_id)]}
    for bucket in ("risks", "key_facts"):
        if extracted[bucket]:
            update[bucket] = extracted[bucket]

    new_nodes: dict[str, Any] = {}
    origin = _node_origin(state, None)
    for assumption in extracted["assumptions"]:
        node = create_node(
            kind="assumption", statement=assumption["statement"], origin=origin, status="needs_confirmation"
        )
        new_nodes[node["id"]] = node
    for question in extracted["open_questions"]:
        node = create_node(kind="open_question", statement=question["question"], origin=origin, status="proposed")
        new_nodes[node["id"]] = node
    if new_nodes:
        update["decision_nodes"] = {**(state.get("decision_nodes") or {}), **new_nodes}
    return Command(update=update)


@tool
async def critique_note(
    content: Annotated[
        str, "The critique. Prefix tagged lines (ASSUMPTION: / RISK: / OPEN_QUESTION:) to record structured items."
    ],  # noqa: E501
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Critique note — silent scratchpad, no user interrupt and no approval.

    Use to think critically (probe weaknesses, risky assumptions, contradictions) before asking or
    drafting. Not shown to the user — use respond to surface a critique to them.
    """
    return await _write_note_impl(content, state, tool_call_id, "critique_note")


@tool
async def explore_note(
    content: Annotated[
        str, "The exploration. Prefix tagged lines (ASSUMPTION: / RISK: / OPEN_QUESTION:) to record structured items."
    ],  # noqa: E501
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Exploration note — silent scratchpad, no user interrupt and no approval.

    Use to broaden the perspective: raise angles or options not yet considered. Not shown to the user.
    """
    return await _write_note_impl(content, state, tool_call_id, "explore_note")


# ---------------------------------------------------------------------------
# decision graph — create / update / supersede nodes (flag-gated)
# ---------------------------------------------------------------------------
# The decision graph is the source of truth for the artifact view. These three
# tools mutate decision_nodes in state (no DB, no interrupt) via Command.update — the whole dict is
# replaced each call because LangGraph does not merge nested state. All writes are behind
# DECISION_GRAPH_ENABLED so an in-progress graph model never leaks into a persisted checkpoint.
_TOOL_EDITABLE_STATUSES = VALID_STATUSES - {"superseded"}


def _decision_graph_off_update(tool_name: str, tool_call_id: str) -> Command:
    logger.warning("tool=%s skipped: DECISION_GRAPH_ENABLED is off", tool_name)
    return _tool_not_available_update(
        tool_name, "decision graph is disabled (DECISION_GRAPH_ENABLED=false)", tool_call_id
    )


def _node_origin(state: WorkflowState, technique: str | None) -> dict[str, Any]:
    return {"turn": state.get("turn_count") or 0, "by": "agent", "technique": technique, "source": None}


def _section_content(statement: str | None, fields: dict[str, str] | None) -> str:
    """Text a section write exposes to validation: the statement plus any table-cell values."""
    parts = [str(statement or "")]
    if fields:
        parts.extend(str(value) for value in fields.values())
    return "\n".join(part for part in parts if part)


def _section_findings_update(state: WorkflowState, node: dict[str, Any]) -> dict[str, Any]:
    """Validate the section a decision-node write touched and record findings.

    Never blocks the write. Keyed by section heading, replace-on-write per key: a passing section
    stores [] so a re-validation clears a prior defect through the merge reducer. A node with no
    section has nothing to validate, so the findings map is left untouched.
    """
    section = node.get("section")
    if not section:
        return {}
    from app.graphs.analysis.section_validation import validate_section

    findings = validate_section(
        state.get("artifact_type") or "brd", section, _section_content(node.get("statement"), node.get("fields"))
    )
    return {"section_findings": {**(state.get("section_findings") or {}), section: findings}}


async def _create_decision_node_impl(
    kind: str,
    statement: str,
    depends_on: list[str] | None,
    technique: str | None,
    state: WorkflowState,
    tool_call_id: str,
    node_id: str | None = None,
    status: str | None = None,
    blocks: list[str] | None = None,
    section: str | None = None,
    fields: dict[str, str] | None = None,
) -> Command:
    if not settings.decision_graph_enabled:
        return _decision_graph_off_update("create_decision_node", tool_call_id)
    if kind not in VALID_KINDS:
        return _tool_not_available_update(
            "create_decision_node", f"invalid kind '{kind}'; choose one of {sorted(VALID_KINDS)}", tool_call_id
        )
    if not str(statement or "").strip():
        return _missing_required_arg_update("create_decision_node", "statement", tool_call_id)
    if status is not None and status not in _TOOL_EDITABLE_STATUSES:
        return _tool_not_available_update(
            "create_decision_node",
            f"invalid status '{status}'; choose one of {sorted(_TOOL_EDITABLE_STATUSES)}",
            tool_call_id,
        )

    nodes_state = state.get("decision_nodes") or {}
    unknown = [dep for dep in (depends_on or []) if dep not in nodes_state]
    if unknown:
        return _tool_not_available_update(
            "create_decision_node", f"depends_on points to missing nodes: {unknown}", tool_call_id
        )
    node = create_node(
        kind=kind,
        statement=statement,
        origin=_node_origin(state, technique),
        depends_on=depends_on,
        node_id=node_id,
        status=status or "proposed",
        blocks=blocks,
        section=section,
        fields=fields,
    )
    if node["id"] in nodes_state:
        return _tool_not_available_update(
            "create_decision_node",
            f"node_id '{node['id']}' already exists; use update/supersede instead of overwriting",
            tool_call_id,
        )
    updated = {**nodes_state, node["id"]: node}
    return Command(
        update={
            "decision_nodes": updated,
            "messages": [ToolMessage(content=f"node {node['id']} ({kind})", tool_call_id=tool_call_id)],
            **_section_findings_update(state, node),
        }
    )


async def _update_decision_node_impl(
    node_id: str,
    status: str | None,
    statement: str | None,
    state: WorkflowState,
    tool_call_id: str,
    section: str | None = None,
    fields: dict[str, str] | None = None,
) -> Command:
    if not settings.decision_graph_enabled:
        return _decision_graph_off_update("update_decision_node", tool_call_id)
    nodes_state = state.get("decision_nodes") or {}
    if node_id not in nodes_state:
        return _tool_not_available_update("update_decision_node", f"node '{node_id}' does not exist", tool_call_id)
    if nodes_state[node_id].get("status") == "superseded":
        return _tool_not_available_update(
            "update_decision_node",
            f"node '{node_id}' is superseded; use the replacement node for further edits",
            tool_call_id,
        )
    if status is not None and status not in _TOOL_EDITABLE_STATUSES:
        return _tool_not_available_update(
            "update_decision_node",
            f"invalid status '{status}'; choose one of {sorted(_TOOL_EDITABLE_STATUSES)}",
            tool_call_id,
        )

    updates = {
        k: v
        for k, v in {
            "status": status,
            "statement": statement,
            "section": section,
            "fields": fields,
        }.items()
        if v is not None
    }
    if not updates:
        return _missing_required_arg_update("update_decision_node", "status or statement", tool_call_id)
    updated = update_node(nodes_state, node_id, **updates)
    return Command(
        update={
            "decision_nodes": updated,
            "messages": [ToolMessage(content=f"node {node_id} updated", tool_call_id=tool_call_id)],
            **_section_findings_update(state, updated[node_id]),
        }
    )


async def _supersede_decision_node_impl(
    node_id: str,
    new_statement: str,
    cascade_mode: str | None,
    state: WorkflowState,
    tool_call_id: str,
) -> Command:
    if not settings.decision_graph_enabled:
        return _decision_graph_off_update("supersede_decision_node", tool_call_id)
    nodes_state = state.get("decision_nodes") or {}
    if node_id not in nodes_state:
        return _tool_not_available_update("supersede_decision_node", f"node '{node_id}' does not exist", tool_call_id)
    if not str(new_statement or "").strip():
        return _missing_required_arg_update("supersede_decision_node", "new_statement", tool_call_id)
    if cascade_mode is not None and cascade_mode not in {"reconfirm", "abandon"}:
        return _tool_not_available_update(
            "supersede_decision_node", f"invalid cascade_mode '{cascade_mode}'", tool_call_id
        )

    resolved_mode = cascade_mode or infer_cascade_mode(nodes_state, node_id)
    rippled = [d for d in get_dependents(nodes_state, node_id) if nodes_state[d].get("status") != "superseded"]
    updated = supersede_node(nodes_state, node_id, new_statement, _node_origin(state, None), resolved_mode)
    new_id = updated[node_id]["superseded_by"]
    summary = f"superseded {node_id} → {new_id}; cascade={resolved_mode}; dependents={rippled or 'none'}"
    return Command(
        update={
            "decision_nodes": updated,
            "messages": [ToolMessage(content=summary, tool_call_id=tool_call_id)],
            **_section_findings_update(state, updated[new_id]),
        }
    )


@tool
async def create_decision_node(
    kind: Annotated[str, "objective | scope | assumption | decision | risk | open_question | fact"],
    statement: Annotated[str, "The decision/objective/etc. as a single clear statement, in the user's locale."],
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    depends_on: Annotated[list[str] | None, "ids of nodes this one builds on; [] for a root."] = None,
    technique: Annotated[str | None, "Elicitation technique that produced this node, for provenance."] = None,
    node_id: Annotated[str | None, "Optional stable short id (e.g. N1); auto-generated if omitted."] = None,
    status: Annotated[
        str | None,
        "Optional initial status: proposed|confirmed|inferred|needs_confirmation|parked. Defaults to proposed.",
    ] = None,
    blocks: Annotated[
        list[str] | None,
        "For parked open_question nodes: ids currently blocked by this question.",
    ] = None,
    section: Annotated[
        str | None,
        "Target heading in the output contract, e.g. '## Objectives' or '## Success Metrics'.",
    ] = None,
    fields: Annotated[
        dict[str, str] | None,
        "Column values when the section renders as a table; keys must match the output contract table_columns.",
    ] = None,
) -> Command:
    """Record a new decision-graph node (objective, decision, risk, ...) with provenance.

    Use to capture a piece of analysis as durable state instead of prose in a draft. Each node keeps
    why it exists (origin) and what it builds on (depends_on) so later reversals can ripple correctly.
    """
    return await _create_decision_node_impl(
        kind, statement, depends_on, technique, state, tool_call_id, node_id, status, blocks, section, fields
    )


@tool
async def update_decision_node(
    node_id: Annotated[str, "id of the node to update."],
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    status: Annotated[str | None, "New status: proposed|confirmed|inferred|needs_confirmation|parked."] = None,
    statement: Annotated[str | None, "Revised statement (a local edit, NOT a reversal)."] = None,
    section: Annotated[
        str | None,
        "New target heading in the output contract, e.g. '## Objectives' or '## Success Metrics'.",
    ] = None,
    fields: Annotated[
        dict[str, str] | None,
        "New column values for the section table; keys must match the output contract table_columns.",
    ] = None,
) -> Command:
    """Update a node's status or statement in place — a local edit that does not rewrite history.

    Use to confirm a proposed node or refine its wording. To reverse a decision (keep history + ripple
    to dependents) use supersede_decision_node instead.
    """
    return await _update_decision_node_impl(node_id, status, statement, state, tool_call_id, section, fields)


@tool
async def supersede_decision_node(
    node_id: Annotated[str, "id of the node being reversed."],
    new_statement: Annotated[str, "The replacement decision/objective, in the user's locale."],
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    cascade_mode: Annotated[
        str | None,
        "How dependents ripple: 'reconfirm' (local edit, dependents need re-confirm) or 'abandon' "
        "(direction reversal, dependents parked). Omit to infer: a root decision node or one with "
        "many dependents → abandon; otherwise reconfirm. Pass explicitly when you know the intent.",
    ] = None,
) -> Command:
    """Reverse a decision without destroying history and ripple the change to dependents.

    Creates a new node that supersedes the old one (old → superseded, kept for provenance) and marks
    every dependent stale (reconfirm) or parked (abandon). Use for a genuine change of direction; use
    update_decision_node for a local wording/status edit.
    """
    return await _supersede_decision_node_impl(node_id, new_statement, cascade_mode, state, tool_call_id)


async def _dismiss_question_impl(
    node_id: str,
    reason: str,
    state: WorkflowState,
    tool_call_id: str,
) -> Command:
    if not settings.decision_graph_enabled:
        return _decision_graph_off_update("dismiss_question", tool_call_id)
    if not str(node_id or "").strip():
        return _missing_required_arg_update("dismiss_question", "node_id", tool_call_id)
    if not str(reason or "").strip():
        return _missing_required_arg_update("dismiss_question", "reason", tool_call_id)

    nodes_state = state.get("decision_nodes") or {}
    node = nodes_state.get(node_id)
    if node is None or node.get("kind") != "open_question":
        return _recoverable_tool_update(
            RecoverableToolError(
                code="dismiss_target_invalid",
                message=f"Cannot dismiss '{node_id}': not a known open_question node.",
                recovery="Pass the id of a parked open_question from the feedback signals.",
            ),
            tool_call_id,
        )

    origin = _node_origin(state, None)
    updated = dismiss_node(nodes_state, node_id, reason, origin)
    # SECURITY: `reason` is agent/user-supplied text persisted as data on the node's audit trail
    # (dismiss_node). The confirmation below is a fixed string — the reason is never echoed back as
    # an instruction.
    return Command(
        update={
            "decision_nodes": updated,
            "messages": [ToolMessage(content=f"Dismissed {node_id}.", tool_call_id=tool_call_id)],
        }
    )


@tool
async def dismiss_question(
    node_id: Annotated[str, "id of the parked open_question node to dismiss."],
    reason: Annotated[str, "Substantive reason for dismissing the question, for the audit trail."],
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Dismiss a resurfaced open_question with an auditable reason instead of answering it.

    Use when a parked question is no longer worth pursuing (superseded by other decisions, out of
    scope, etc.) — not an escape hatch for work you should still do. The dismissal (reason + turn) is
    recorded on the node and the question is removed from the resurfacing/blocker set.
    """
    return await _dismiss_question_impl(node_id, reason, state, tool_call_id)


def _config_ids(config: RunnableConfig) -> tuple[uuid.UUID | None, uuid.UUID | None, Any]:
    cfg = config.get("configurable") or {}
    project_raw = cfg.get("project_id")
    thread_raw = cfg.get("thread_id")
    project_id = uuid.UUID(str(project_raw)) if project_raw is not None else None
    session_id = uuid.UUID(str(thread_raw)) if thread_raw is not None else None
    return project_id, session_id, cfg.get("session_factory")


async def _session_user_id(session_factory, session_id: uuid.UUID | None) -> uuid.UUID | None:
    if session_factory is None or session_id is None:
        return None
    async with session_factory() as db:
        return await db.scalar(select(AgentSession.created_by_id).where(AgentSession.id == session_id))


def _proposal_run_id(state: WorkflowState, tool_call_id: str, tool_name: str) -> uuid.UUID:
    run_id_raw = state.get("last_agent_run_id")
    if run_id_raw:
        return uuid.UUID(str(run_id_raw))
    # record_run_and_dispatch uses "{run_uuid}-{index}" ids. This fallback keeps proposal tools
    # idempotent even if a legacy caller did not thread last_agent_run_id into state.
    try:
        return uuid.UUID(str(tool_call_id)[:36])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{tool_name} requires last_agent_run_id in state — analyze_node must run first") from exc


async def _save_approval_proposal(
    *,
    config: RunnableConfig,
    run_id: uuid.UUID,
    session_id: uuid.UUID,
    tool_name: str,
    input_snapshot: dict[str, Any],
) -> None:
    session_factory = config["configurable"]["session_factory"]
    async with session_factory() as db:
        existing = (
            await db.execute(
                select(AgentToolCall).where(
                    AgentToolCall.run_id == run_id,
                    AgentToolCall.tool_name == tool_name,
                )
            )
        ).scalar_one_or_none()
        if existing is None:
            db.add(
                AgentToolCall(
                    run_id=run_id,
                    tool_name=tool_name,
                    input_snapshot=input_snapshot,
                    status=AgentToolCallStatus.PROPOSED,
                )
            )
        session_row = (await db.execute(select(AgentSession).where(AgentSession.id == session_id))).scalar_one()
        session_row.status = AgentSessionStatus.WAITING_FOR_HUMAN
        session_row.interrupt_type = AgentSessionInterruptType.PROPOSE_ARTIFACTS
        await db.commit()


async def _load_artifact_links(config: RunnableConfig) -> list[dict[str, str]]:
    project_id, _session_id, session_factory = _config_ids(config)
    if project_id is None or session_factory is None:
        return []
    async with session_factory() as db:
        rows = (
            (
                await db.execute(
                    select(ArtifactLink).where(ArtifactLink.project_id == project_id).order_by(ArtifactLink.created_at)
                )
            )
            .scalars()
            .all()
        )
    return [
        {
            "source_id": str(row.source_artifact_id),
            "target_id": str(row.target_artifact_id),
            "relation_type": row.relation_type.value,
        }
        for row in rows
    ]


async def _read_artifact_graph_impl(config: RunnableConfig, tool_call_id: str) -> Command:
    links = await _load_artifact_links(config)
    return Command(update={"messages": [ToolMessage(content=json.dumps({"links": links}), tool_call_id=tool_call_id)]})


@tool("read_artifact_graph")
async def read_artifact_graph_tool(
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Read artifact-link graph for the current project. Read-only, non-interrupting."""
    return await _read_artifact_graph_impl(config, tool_call_id)


async def _create_artifact_link_impl(
    source_artifact_id: str,
    target_artifact_id: str,
    relation_type: str,
    state: WorkflowState,
    config: RunnableConfig,
    tool_call_id: str,
) -> Command:
    project_id, session_id, session_factory = _config_ids(config)
    if project_id is None or session_id is None or session_factory is None:
        return _tool_not_available_update("create_artifact_link", "missing project/session context", tool_call_id)
    try:
        body = ArtifactLinkCreateRequest(
            source_artifact_id=uuid.UUID(str(source_artifact_id)),
            target_artifact_id=uuid.UUID(str(target_artifact_id)),
            relation_type=RelationType(str(relation_type)),
        )
    except (ValueError, TypeError) as exc:
        return _tool_not_available_update("create_artifact_link", f"invalid input: {exc}", tool_call_id)
    run_id = _proposal_run_id(state, tool_call_id, "create_artifact_link")
    proposal_tool_name = (
        f"create_artifact_link:{body.source_artifact_id}:{body.target_artifact_id}:{body.relation_type.value}"
    )
    await _save_approval_proposal(
        config=config,
        run_id=run_id,
        session_id=session_id,
        tool_name=proposal_tool_name,
        input_snapshot={
            "source_artifact_id": str(body.source_artifact_id),
            "target_artifact_id": str(body.target_artifact_id),
            "relation_type": body.relation_type.value,
            "metadata": body.metadata,
        },
    )
    interrupt({"type": "propose_artifacts", "tool_name": "create_artifact_link"})
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=(
                        "create_artifact_link proposal is waiting for human approval; "
                        "the link is not committed until approved."
                    ),
                    tool_call_id=tool_call_id,
                )
            ]
        }
    )


@tool("create_artifact_link")
async def create_artifact_link_tool(
    source_artifact_id: Annotated[str, "Source artifact UUID."],
    target_artifact_id: Annotated[str, "Target artifact UUID."],
    relation_type: Annotated[str, "RelationType value, e.g. derives_from, depends_on, satisfies."],
    state: Annotated[dict, InjectedState],
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Propose an artifact dependency link and pause for human approval.

    The link is not committed until the proposal is approved, so do not expect read_artifact_graph to
    show it in the same turn.
    """
    return await _create_artifact_link_impl(
        source_artifact_id,
        target_artifact_id,
        relation_type,
        state,
        config,
        tool_call_id,
    )


async def _propose_retirement_impl(
    artifact_id: str,
    reason: str,
    state: WorkflowState,
    config: RunnableConfig,
    tool_call_id: str,
    superseded_by_artifact_id: str | None = None,
) -> Command:
    project_id, session_id, session_factory = _config_ids(config)
    if project_id is None or session_id is None or session_factory is None:
        return _tool_not_available_update("propose_retirement", "missing project/session context", tool_call_id)
    if not str(reason or "").strip():
        return _missing_required_arg_update("propose_retirement", "reason", tool_call_id)
    try:
        retired_id = uuid.UUID(str(artifact_id))
        superseded_by_id = (
            uuid.UUID(str(superseded_by_artifact_id)) if str(superseded_by_artifact_id or "").strip() else None
        )
    except (TypeError, ValueError) as exc:
        return _tool_not_available_update("propose_retirement", f"invalid input: {exc}", tool_call_id)
    run_id = _proposal_run_id(state, tool_call_id, "propose_retirement")
    proposal_tool_name = f"propose_retirement:{retired_id}"
    await _save_approval_proposal(
        config=config,
        run_id=run_id,
        session_id=session_id,
        tool_name=proposal_tool_name,
        input_snapshot={
            "artifact_id": str(retired_id),
            "reason": str(reason).strip(),
            "superseded_by_artifact_id": str(superseded_by_id) if superseded_by_id else None,
        },
    )
    interrupt({"type": "propose_artifacts", "tool_name": "propose_retirement"})
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=(
                        "propose_retirement is waiting for human approval; "
                        "the artifact is not archived until approved."
                    ),
                    tool_call_id=tool_call_id,
                )
            ]
        }
    )


@tool("propose_retirement")
async def propose_retirement_tool(
    artifact_id: Annotated[str, "Artifact UUID to archive/retire."],
    reason: Annotated[str, "Concise audit reason for retiring this artifact."],
    state: Annotated[dict, InjectedState],
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
    superseded_by_artifact_id: Annotated[str | None, "Optional replacement artifact UUID."] = None,
) -> Command:
    """Propose retiring an artifact and pause for human approval.

    Approval archives the artifact, records the optional superseded_by reference, and rejects the
    proposal if live downstream dependents still use the artifact.
    """
    return await _propose_retirement_impl(
        artifact_id,
        reason,
        state,
        config,
        tool_call_id,
        superseded_by_artifact_id,
    )


async def _run_impact_analysis_impl(
    change_description: str,
    state: WorkflowState,
    config: RunnableConfig,
    tool_call_id: str,
    changed_artifact_id: str | None = None,
) -> Command:
    if not str(change_description or "").strip():
        return _missing_required_arg_update("run_impact_analysis", "change_description", tool_call_id)
    nodes_state = state.get("decision_nodes") or {}
    links = await _load_artifact_links(config)
    result = impact(change_description, nodes_state, links, changed_artifact_id=changed_artifact_id)
    affected = result["affected_node_ids"]
    feedback = dict(state.get("feedback_summary") or {})
    if affected:
        feedback["stale_warning"] = f"{len(affected)} node need reconfirmation due to change: {', '.join(affected)}"
        feedback["impact_result"] = {
            "affected_node_ids": affected,
            "stale_artifact_ids": result["stale_artifact_ids"],
        }
    return Command(
        update={
            "decision_nodes": result["decision_nodes"],
            "feedback_summary": feedback,
            "messages": [
                ToolMessage(
                    content=json.dumps(
                        {
                            "affected_node_ids": affected,
                            "stale_artifact_ids": result["stale_artifact_ids"],
                        },
                        ensure_ascii=False,
                    ),
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


@tool("run_impact_analysis")
async def run_impact_analysis(
    change_description: Annotated[str, "User-described change that may affect existing nodes."],
    state: Annotated[dict, InjectedState],
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
    changed_artifact_id: Annotated[str | None, "Artifact UUID where the change originated, if known."] = None,
) -> Command:
    """Mark exactly affected decision nodes stale; do not rewrite them silently."""
    return await _run_impact_analysis_impl(change_description, state, config, tool_call_id, changed_artifact_id)


# ---------------------------------------------------------------------------
# respond — user-facing critique/exploration (mode-bearing, interrupting)
# ---------------------------------------------------------------------------
# The note tools are silent scratchpad; respond is the outward voice for a non-question turn. It
# lets the analyst deliver a critique or an exploration TO the user and pause for their reaction,
# so the agent is not forced to phrase every proactive turn as an ask_user (the Q&A-bias fix).


async def _respond_impl(message: str, mode: str, state: WorkflowState, config: RunnableConfig, tool_call_id: str):
    if not str(message or "").strip():
        return _missing_required_arg_update("respond", "message", tool_call_id)

    # Reuses the ask_user persist+interrupt path (idempotency keyed on ToolCall.id). respond is a
    # conversational pause like ask_user/confirm_intent, so it keeps the session ACTIVE with a
    # STREAM_RESPONSE interrupt instead of entering the approval-gate WAITING state.
    await _audit_interaction_tool_call(state, config, tool_name=f"respond:{tool_call_id}", message=message)
    user_content = await interrupts._save_and_interrupt_ask(
        state, config, message, run_id=tool_call_id, kind="assessment", mode=mode, interrupt_kind="stream_response"
    )
    return Command(
        update={
            "messages": [
                ToolMessage(content=user_content, tool_call_id=tool_call_id),
                {"role": "user", "content": user_content},
            ]
        }
    )


@tool
async def respond(
    message: Annotated[str, "The assessment to deliver, in the user's locale — a complete thought, not a question."],
    mode: Annotated[str, "Operating angle: 'critique' or 'structuring'."],
    state: Annotated[dict, InjectedState],
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Share an assessment with the user — a critique or exploration, NOT a question — and pause for their reaction.

    Use to deliver a proactive opinion or analysis instead of phrasing every turn as a question. Use
    ask_user when you actually need an answer; use the note tools to think without interrupting.
    """  # noqa: E501
    return await _respond_impl(message, mode, state, config, tool_call_id)


# ---------------------------------------------------------------------------
# run_critique — formal judge call over the current draft (mode-targeted, non-interrupting)
# ---------------------------------------------------------------------------
# Unlike critique_note (silent scratchpad), run_critique invokes the production judge in
# critique.py, records a quality_report, and increments critique_rounds. It does not interrupt —
# the analyst surfaces the result to the user via `respond` on a later turn.


async def _run_critique_impl(
    target: str,  # noqa: ARG001 — kept for schema parity; the judge scores the loaded draft body
    mode: str,
    state: WorkflowState,
    config: RunnableConfig,
    tool_call_id: str,
):
    from app.graphs.critique import _invoke_judge

    if not str(mode or "").strip():
        return _missing_required_arg_update("run_critique", "mode", tool_call_id)
    if (state.get("critique_rounds") or 0) >= CRITIQUE_ROUNDS_MAX:
        return _tool_not_available_update(
            "run_critique",
            "critique round limit reached; revise, respond/escalate, or finalize if the gate has passed.",
            tool_call_id,
        )

    cfg = config["configurable"]
    llm_client = cfg.get("strong_llm_client") or cfg.get("llm_client")
    # Source of truth for both the critique target and the hash: see current_draft_body. The
    # finalize gate reads the same helper, so the scored body and the gate body can never diverge.
    body = await current_draft_body(state, config)
    if not body.strip():
        return _tool_not_available_update(
            "run_critique",
            "no current draft to critique; write_draft or load an artifact first.",
            tool_call_id,
        )
    judged = await _invoke_judge(body, mode, llm_client)

    threshold = settings.critique_score_threshold
    score = judged["score"]
    findings = judged["findings"]
    suggestions = judged["suggestions"]
    # Gate result is derived from score, NOT from blocking_issues emptiness — the no-LLM degraded
    # path (score=0.0, findings=[]) must still "fail" so the loop can never finalize without a real
    # critique. This is fail-safe by design, not a bug.
    quality_gate_result = "fail" if score < threshold else "pass"
    blocking_issues = findings if quality_gate_result == "fail" else []
    non_blocking_warnings = findings if quality_gate_result == "pass" else []
    revision_plan = suggestions if quality_gate_result == "fail" else []

    rounds_after = (state.get("critique_rounds") or 0) + 1
    # A passing gate steers to finalize. A failing gate steers to revise while rounds remain; once
    # the rounds cap is reached and the gate still fails the loop has no auto-recovery (run_critique
    # is gated off, finalize is blocked), so it must escalate — hand the decision to the user rather
    # than revise silently forever. "re_critique" is never recommended (it would be a dead signal).
    if quality_gate_result == "pass":
        recommended_next_action = "finalize"
    elif rounds_after >= CRITIQUE_ROUNDS_MAX:
        recommended_next_action = "escalate"
    else:
        recommended_next_action = "revise"

    log_gate_decision(
        "critique",
        quality_gate_result,
        score=score,
        reason=recommended_next_action,
        session_id=cfg.get("thread_id"),
    )
    report: QualityReport = {
        "mode": judged["mode"],
        "score": score,
        "findings": findings,
        "suggestions": suggestions,
        "blocking_issues": blocking_issues,
        "non_blocking_warnings": non_blocking_warnings,
        "revision_plan": revision_plan,
        "quality_gate_result": quality_gate_result,
        "recommended_next_action": recommended_next_action,
    }
    draft_hash = hashlib.md5(body.encode()).hexdigest()[:8]
    summary = f"critique[{report['mode']}] score={report['score']:.2f} gate={quality_gate_result}"
    return Command(
        update={
            "quality_report": report,
            "last_critiqued_draft_hash": draft_hash,
            "critique_rounds": rounds_after,
            "messages": [ToolMessage(content=summary, tool_call_id=tool_call_id)],
        }
    )


@tool
async def run_critique(
    target: Annotated[str, "Cosmetic label for the target; the judge always scores the current draft body."],
    mode: Annotated[str, "Critique dimension, e.g. 'completeness', 'clarity', 'feasibility'."],
    state: Annotated[dict, InjectedState],
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Run a formal quality critique over the current draft along one mode and record the report.

    Use after a draft exists to score it before finalizing. Does not interrupt — surface the result
    to the user with respond on a later turn. Gated off after the critique-rounds cap is reached.
    """
    return await _run_critique_impl(target, mode, state, config, tool_call_id)


# ---------------------------------------------------------------------------
# recommend_next_workflow — read-only analysis tool (no interrupt; audits to AgentToolCall)
# ---------------------------------------------------------------------------

# First four sections define a product brief; all seven define a PRD (addendum §6).
_BRIEF_SECTIONS = ("vision_objectives", "problem_statement", "stakeholder_register", "scope_capabilities")


def _compute_recommendation(section_coverage: dict[str, str] | None, planning_track: str) -> dict[str, Any]:
    """Pure workflow-selection rule over 7-section coverage. No DB, no state mutation.

    Derives the artifact chain inline from the latest section_coverage (never the possibly-stale
    state["artifact_chain"]). On the quick track, never escalates past readiness_check.
    """
    cov = section_coverage or {}
    scores = {section: status_score(cov.get(section)) for section in children_of("brd")}
    brief_score = sum(scores[s] for s in _BRIEF_SECTIONS) / len(_BRIEF_SECTIONS)
    prd_score = sum(scores.values()) / len(scores)
    missing = [section for section, score in scores.items() if score == 0.0]

    if prd_score >= 0.7:
        recommended, reason = "readiness_check", "PRD coverage is near-complete; assess readiness next."
    elif brief_score >= 0.6:
        recommended, reason = "prd", "Product-brief sections are solid; expand into a PRD."
    else:
        recommended, reason = "brief", "Early signal captured; consolidate into a product brief."

    # Quick track guards against over-planning a small idea.
    if planning_track == "quick" and recommended == "architecture_readiness":
        recommended = "readiness_check"

    missing_count = len(missing)
    confidence = "low" if missing_count >= 4 else ("medium" if missing_count >= 1 else "high")

    return {
        "recommended_next_workflow": recommended,
        "reason": reason,
        "required_inputs": list(missing),
        "blocking_gaps": list(missing),
        "confidence": confidence,
    }


async def _recommend_next_workflow_impl(
    current_artifact_type: str,  # noqa: ARG001 — kept for schema parity; recommendation is coverage-driven
    planning_track: str,
    state: WorkflowState,
    config: RunnableConfig,
    tool_call_id: str,
):
    # Read-only and side-effect-free, so no availability gate: calling it early just yields a weaker
    # recommendation, which is valid feedback. When to call is a prompt hint, not a safety invariant.
    result = _compute_recommendation(state.get("section_coverage"), planning_track or "quick")

    # Audit: reuse AgentToolCall.input_snapshot for the result blob (no output_snapshot column).
    # Best-effort — a DB failure must not deny the recommendation to the user.
    if not state.get("last_agent_run_id"):
        raise RuntimeError("recommend_next_workflow requires last_agent_run_id in state — analyze_node must run first")
    run_id = uuid.UUID(state["last_agent_run_id"])
    cfg = config["configurable"]
    session_factory = cfg["session_factory"]
    try:
        async with session_factory() as db:
            already = (
                await db.execute(
                    select(
                        exists().where(
                            AgentToolCall.run_id == run_id,
                            AgentToolCall.tool_name == "recommend_next_workflow",
                        )
                    )
                )
            ).scalar()
            if not already:
                db.add(
                    AgentToolCall(
                        run_id=run_id,
                        tool_name="recommend_next_workflow",
                        input_snapshot=result,
                        status=AgentToolCallStatus.PROPOSED,
                    )
                )
                await db.commit()
    except Exception as exc:  # noqa: BLE001 — audit is best-effort; never block the recommendation
        logger.warning("recommend_next_workflow audit persist failed: %s", exc)

    method_profile = dict(state.get("method_profile") or {})
    method_profile["recommended_next_workflow"] = result["recommended_next_workflow"]
    summary = f"recommend_next_workflow -> {result['recommended_next_workflow']} ({result['confidence']})"
    return Command(
        update={
            "method_profile": method_profile,
            "messages": [ToolMessage(content=summary, tool_call_id=tool_call_id)],
        }
    )


@tool
async def recommend_next_workflow(
    current_artifact_type: Annotated[
        str,
        "The artifact type currently in focus (cosmetic; recommendation is coverage-driven).",
    ],
    planning_track: Annotated[str, "Planning depth: 'quick' | 'standard' | 'enterprise'."],
    state: Annotated[dict, InjectedState],
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Recommend the next planning workflow from current coverage; records an audit entry.

    Use when the user asks what to do next, or when coverage suggests advancing the artifact chain.
    Read-only and non-interrupting.
    """
    return await _recommend_next_workflow_impl(current_artifact_type, planning_track, state, config, tool_call_id)


# ---------------------------------------------------------------------------
# run_readiness_check — 10-dimension readiness assessment (no interrupt; audits to AgentToolCall)
# ---------------------------------------------------------------------------


async def _run_readiness_check_impl(
    target: str,  # noqa: ARG001 — kept for schema parity; the check is coverage-driven
    state: WorkflowState,
    config: RunnableConfig,
    tool_call_id: str,
):
    # No availability gate, same rationale as _recommend_next_workflow_impl: an early call returns a
    # low readiness score, which is valid feedback rather than a safety error.
    from app.graphs.readiness import compute_readiness_score

    report = compute_readiness_score(state.get("section_coverage"), state)

    if not state.get("last_agent_run_id"):
        raise RuntimeError("run_readiness_check requires last_agent_run_id in state — analyze_node must run first")
    run_id = uuid.UUID(state["last_agent_run_id"])
    cfg = config["configurable"]
    session_factory = cfg["session_factory"]
    try:
        async with session_factory() as db:
            already = (
                await db.execute(
                    select(
                        exists().where(
                            AgentToolCall.run_id == run_id,
                            AgentToolCall.tool_name == "run_readiness_check",
                        )
                    )
                )
            ).scalar()
            if not already:
                db.add(
                    AgentToolCall(
                        run_id=run_id,
                        tool_name="run_readiness_check",
                        input_snapshot=report,
                        status=AgentToolCallStatus.PROPOSED,
                    )
                )
                await db.commit()
    except Exception as exc:  # noqa: BLE001 — audit is best-effort; never block the check
        logger.warning("run_readiness_check audit persist failed: %s", exc)

    readiness = dict(state.get("readiness") or {})
    readiness["requirements_ready"] = report["ready"]
    readiness["blocking_gaps"] = report["blocking_gaps"]
    readiness["recommended_next_step"] = report["recommended_next_step"]
    summary = f"run_readiness_check -> ready={report['ready']} score={report['readiness_score']:.2f}"
    log_gate_decision(
        "readiness",
        "ready" if report["ready"] else "not_ready",
        score=report["readiness_score"],
        reason=report["recommended_next_step"],
        session_id=cfg.get("thread_id"),
    )
    return Command(
        update={
            "readiness": readiness,
            "messages": [ToolMessage(content=summary, tool_call_id=tool_call_id)],
        }
    )


@tool
async def run_readiness_check(
    target: Annotated[str, "Cosmetic label; the check is coverage-driven."],
    state: Annotated[dict, InjectedState],
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Assess readiness to advance the planning lifecycle across 10 dimensions; records an audit entry.

    Use after at least one critique round to check whether the draft is ready to progress. Read-only
    and non-interrupting.
    """
    return await _run_readiness_check_impl(target, state, config, tool_call_id)


# ---------------------------------------------------------------------------
# read_artifact — side-effect-free body read by id (no interrupt, no DB write)
# ---------------------------------------------------------------------------
# Lets the model pull a sibling/ancestor artifact's body mid-session instead of re-asking the user
# for content already recorded. Loops back through analyze_node like the note tools — appends a
# ToolMessage, never interrupts. Project-scoped via read_current_body's project_id filter so a session
# can only read its own project's artifacts.

# Cap a single read so a large body cannot dominate the analyze prompt; the head is enough to orient,
# and a focused draft is reached through write_draft/current_draft_body, not this tool.
READ_ARTIFACT_MAX_CHARS = 8000
READ_SOURCE_DOCUMENT_MAX_CHARS = 8000
READ_SOURCE_DOCUMENT_MAX_ITEMS = 3


def _read_artifact_source_context(result: dict[str, Any], excerpt: str) -> list[dict[str, Any]]:
    version_id = str(result.get("current_version_id") or "").strip()
    artifact_id = str(result.get("artifact_id") or "").strip()
    if not version_id or not artifact_id or not excerpt.strip():
        return []
    return [
        {
            "source_kind": "predecessor_version",
            "artifact_id": artifact_id,
            "predecessor_version_id": version_id,
            "title": result.get("title"),
            "locator": f"artifact_version:{version_id}",
            "excerpt": excerpt,
            "truncated": bool(result.get("truncated")),
        }
    ]


def _source_document_source_context(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for document in documents:
        excerpt = str(document.get("excerpt") or "").strip()
        source_document_id = str(document.get("id") or "").strip()
        if not source_document_id or not excerpt:
            continue
        entries.append(
            {
                "source_kind": "source_document",
                "source_document_id": source_document_id,
                "title": document.get("title"),
                "locator": document.get("locator"),
                "excerpt": excerpt,
                "truncated": bool(document.get("truncated")),
            }
        )
    return entries


async def _read_artifact_impl(artifact_id: str, config: RunnableConfig, tool_call_id: str):
    cfg = config["configurable"]
    project_id_raw = cfg.get("project_id")
    session_factory = cfg.get("session_factory")
    try:
        target_id = uuid.UUID(str(artifact_id))
    except (ValueError, TypeError):
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"read_artifact: invalid id ({artifact_id!r}).",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    result = None
    if session_factory is not None and project_id_raw is not None:
        async with session_factory() as db:
            result = await read_current_body(
                db=db,
                project_id=uuid.UUID(str(project_id_raw)),
                artifact_id=target_id,
            )

    if result is None:
        content = f"read_artifact: artifact not found {artifact_id} (or has no content yet) in project."
        source_context = []
    else:
        body = result["body"] or ""
        excerpt = body[:READ_ARTIFACT_MAX_CHARS]
        if len(body) > READ_ARTIFACT_MAX_CHARS:
            body = excerpt + "\n\n…(remaining content truncated)"
            result = {**result, "truncated": True}
        else:
            body = excerpt
        content = f"# {result['title']}\n\n{body}"
        source_context = _read_artifact_source_context(result, excerpt)
    update: dict[str, Any] = {"messages": [ToolMessage(content=content, tool_call_id=tool_call_id)]}
    if source_context:
        update["source_context"] = source_context
    return Command(update=update)


@tool
async def read_artifact(
    id: Annotated[str, "The artifact id (UUID) to read — a sibling or ancestor in this project."],
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Read the current body of another artifact in this project by its id.

    Use to pull context from a sibling or ancestor artifact (e.g. the parent BRD) instead of asking
    the user for content that already exists. Read-only and non-interrupting; the body is returned to
    you, not shown to the user.
    """
    return await _read_artifact_impl(id, config, tool_call_id)


def _normalize_source_document_ids(ids: Any) -> tuple[bool, list[uuid.UUID], list[str]]:
    if ids is None:
        return False, [], []
    if isinstance(ids, str):
        raw_items = [ids]
    else:
        try:
            raw_items = list(ids or [])
        except TypeError:
            raw_items = [ids]
    parsed: list[uuid.UUID] = []
    invalid: list[str] = []
    for raw in raw_items[:READ_SOURCE_DOCUMENT_MAX_ITEMS]:
        try:
            parsed.append(uuid.UUID(str(raw)))
        except (TypeError, ValueError):
            invalid.append(str(raw))
    return True, parsed, invalid


async def _read_source_documents_impl(ids: Any, config: RunnableConfig, tool_call_id: str) -> Command:
    cfg = config["configurable"]
    project_id_raw = cfg.get("project_id")
    session_factory = cfg.get("session_factory")
    ids_supplied, parsed_ids, invalid_ids = _normalize_source_document_ids(ids)
    documents: list[dict[str, Any]] = []
    if session_factory is not None and project_id_raw is not None and (parsed_ids or not ids_supplied):
        async with session_factory() as db:
            documents = await read_source_documents_query(
                db=db,
                project_id=uuid.UUID(str(project_id_raw)),
                source_document_ids=parsed_ids or None,
                limit=READ_SOURCE_DOCUMENT_MAX_ITEMS,
                max_chars=READ_SOURCE_DOCUMENT_MAX_CHARS,
            )
    payload = {
        "documents": documents,
        "invalid_ids": invalid_ids,
        "strategy": {
            "max_items": READ_SOURCE_DOCUMENT_MAX_ITEMS,
            "max_chars_per_document": READ_SOURCE_DOCUMENT_MAX_CHARS,
            "mode": "bounded_excerpt",
        },
    }
    source_context = _source_document_source_context(documents)
    update: dict[str, Any] = {
        "messages": [ToolMessage(content=json.dumps(payload, ensure_ascii=False), tool_call_id=tool_call_id)]
    }
    if source_context:
        update["source_context"] = source_context
    return Command(update=update)


@tool("read_source_documents")
async def read_source_documents_tool(
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
    ids: Annotated[
        list[str] | None,
        "Optional source document UUIDs. Omit to read the latest project source documents as bounded excerpts.",
    ] = None,
) -> Command:
    """Read bounded stored source-document text for this project.

    Use before drafting when the user references uploaded/pasted source documents. The returned
    excerpts are available to generation and are snapshotted as evidence only if a draft is proposed.
    """
    return await _read_source_documents_impl(ids, config, tool_call_id)


# ---------------------------------------------------------------------------
# get_available_tools — state-driven gate over the tool-loop
# ---------------------------------------------------------------------------

# After this many run_critique calls the formal judge is gated off the menu so the loop cannot
# spin on critique forever (spec §5.5). write_draft / ask_user stay available regardless.
# Sourced from config (max_critique_rounds) so the reflection-round cap is a single tunable.
CRITIQUE_ROUNDS_MAX = settings.max_critique_rounds

# Cap on LLM judge calls the orchestrator's diagnosis step may spend per turn escalating a
# heuristic high-risk classification.
DIAGNOSIS_JUDGE_CALLS_MAX = settings.max_diagnosis_judge_calls


# ---------------------------------------------------------------------------
# Elicitation surface — BMAD technique scaffolds + external knowledge
# ---------------------------------------------------------------------------

ELICIT_TECHNIQUES = (
    "5_whys",
    "reverse",
    "moscow",
    "first_principles",
    "comparable_products",
    "pre_mortem",
    "tree_of_thought",
    "socratic_questioning",
    "challenge_assumptions",
    "event_storming",
)


def _duckduckgo_search(query: str) -> list[dict]:
    """Keyless DuckDuckGo HTML scrape. Best-effort; web_search wraps this in graceful fallback."""

    import httpx

    resp = httpx.get(
        "https://html.duckduckgo.com/html/",
        params={"q": query},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=10.0,
    )
    resp.raise_for_status()
    results = []
    for match in re.finditer(r'result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', resp.text):
        title = re.sub(r"<[^>]+>", "", match.group(2)).strip()
        results.append({"title": title, "snippet": title, "url": match.group(1)})
    return results[:8]


def _default_search_client():
    """Resolve the configured search client, or None when search is disabled (CI default)."""
    if settings.search_provider == "duckduckgo":
        return _duckduckgo_search
    return None


def web_search(query: str, *, client=None) -> dict:
    """Run a web search, degrading gracefully when no provider is available.

    Returns {"results": [...], "source": "web_search"} on success, or
    {"results": [], "error": "search_unavailable"} when no client is configured or the call fails —
    never raises, so elicit() can fall back to model knowledge. Each result is {title, snippet, url}.
    """
    search = client or _default_search_client()
    if search is None:
        return {"results": [], "error": "search_unavailable"}
    try:
        raw = search(query)
    except Exception:
        return {"results": [], "error": "search_unavailable"}
    return {"results": list(raw), "source": "web_search"}


def _elicit_5_whys(seed: str) -> dict:
    chain = [{"depth": 1, "prompt": f"Why '{seed}' happen?"}] + [
        {"depth": d, "prompt": "Why does the upper-level cause exist?"} for d in range(2, 6)
    ]
    return {
        "technique": "5_whys",
        "seed": seed,
        "chain": chain,
        "root_cause": "Follow the why-chain to the final layer, identify the root cause, then record it as a node.",
    }


def _elicit_reverse(seed: str) -> dict:
    failure_modes = [
        {
            "mode": f"Fastest way to make '{seed}' fail completely",
            "mitigation_hint": "Invert into a required success condition.",
        },
        {
            "mode": "Implicit assumption breaks in reality",
            "mitigation_hint": "List assumptions and attach each to a validation.",
        },
        {
            "mode": "External dependency is not ready in time",
            "mitigation_hint": "Identify a fallback or reduce dependency.",
        },
    ]
    return {"technique": "reverse", "seed": seed, "failure_modes": failure_modes}


def _elicit_moscow(seed: str) -> dict:
    return {
        "technique": "moscow",
        "seed": seed,
        "must": [f"(Required for v1) core item for: {seed}"],
        "should": [],
        "could": [],
        "wont": [f"(Excluded from v1) deferred item for: {seed}"],
    }


def _elicit_first_principles(seed: str) -> dict:
    return {
        "technique": "first_principles",
        "seed": seed,
        "fundamentals": [
            f"Undeniable first-principle fact about '{seed}'",
            "Real physical/economic constraint (not a design convention)",
        ],
        "rebuilt_approach": "Rebuild a minimal solution from first principles without design assumptions.",
    }


def _elicit_comparable_products(seed: str, search_client) -> dict:
    res = web_search(seed, client=search_client)
    results = res.get("results") or []
    if res.get("error") or not results:
        return {
            "technique": "comparable_products",
            "seed": seed,
            "products": [
                {
                    "name": f"(Comparable product for {seed})",
                    "model": "Reference model to validate",
                    "relevance": "Fill when real data is available.",
                }
            ],
            "source": "model_knowledge",
            "evidence_source": "model_knowledge",
        }
    products = [
        {"name": r.get("title", ""), "model": r.get("snippet", ""), "relevance": f"Related to: {seed}"} for r in results
    ]
    return {
        "technique": "comparable_products",
        "seed": seed,
        "products": products,
        "source": "web_search",
        "evidence_source": "web",
    }


def _elicit_pre_mortem(seed: str) -> dict:
    return {
        "technique": "pre_mortem",
        "seed": seed,
        "premise": f"Imagine '{seed}' has already failed six months from now.",
        "failure_causes": [
            {"cause": "Most likely cause of failure", "prevention_hint": "Turn into a guarded precondition."},
            {"cause": "Second most likely cause of failure", "prevention_hint": "Turn into a guarded precondition."},
            {"cause": "Overlooked/silent cause of failure", "prevention_hint": "Turn into a monitored assumption."},
        ],
    }


def _elicit_tree_of_thought(seed: str) -> dict:
    return {
        "technique": "tree_of_thought",
        "seed": seed,
        "branches": [
            {"path": "Conservative approach", "outcome_hint": "Lower risk, slower/narrower payoff."},
            {"path": "Aggressive approach", "outcome_hint": "Higher risk, faster/broader payoff."},
            {"path": "Hybrid approach", "outcome_hint": "Combine strengths, watch for added complexity."},
        ],
        "evaluation_hint": "Score each branch against feasibility and value, then pick or merge.",
    }


def _elicit_socratic_questioning(seed: str) -> dict:
    return {
        "technique": "socratic_questioning",
        "seed": seed,
        "questions": [
            {"probe": f"What does '{seed}' actually mean in this context?", "targets": "clarification"},
            {"probe": "What evidence supports this being true?", "targets": "assumption"},
            {"probe": "What would change if this were false?", "targets": "implication"},
        ],
    }


def _elicit_challenge_assumptions(seed: str) -> dict:
    return {
        "technique": "challenge_assumptions",
        "seed": seed,
        "assumptions_to_challenge": [
            {
                "assumption": f"Implicit assumption behind '{seed}'",
                "counter_argument": "State the strongest case against it.",
                "revised_statement": "Rewrite the requirement so it holds even if the assumption is false.",
            },
        ],
    }


def elicit(technique: str, seed: str, *, search_client=None) -> dict:
    """Apply a BMAD elicitation technique to a seed, returning a structured frame.

    Reasoning techniques (5_whys/reverse/moscow/first_principles) return a deterministic frame for
    the agent to fill; comparable_products pulls real external knowledge via web_search and falls
    back to model knowledge when search is unavailable.
    """
    if technique not in ELICIT_TECHNIQUES:
        raise ValueError(f"unknown elicit technique {technique!r}; expected one of {ELICIT_TECHNIQUES}")
    if technique == "5_whys":
        return _elicit_5_whys(seed)
    if technique == "reverse":
        return _elicit_reverse(seed)
    if technique == "moscow":
        return _elicit_moscow(seed)
    if technique == "first_principles":
        return _elicit_first_principles(seed)
    if technique == "comparable_products":
        return _elicit_comparable_products(seed, search_client)
    if technique == "pre_mortem":
        return _elicit_pre_mortem(seed)
    if technique == "tree_of_thought":
        return _elicit_tree_of_thought(seed)
    if technique == "socratic_questioning":
        return _elicit_socratic_questioning(seed)
    return _elicit_challenge_assumptions(seed)


@tool("web_search")
async def web_search_tool(
    query: Annotated[str, "External knowledge search query."],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Search the web for external knowledge (comparable products, industry standards). Returns structured results.

    When no provider is configured or the call fails, returns an empty result with an error field — never
    interrupts the tool loop.
    """
    result = web_search(query)
    return Command(
        update={"messages": [ToolMessage(content=json.dumps(result, ensure_ascii=False), tool_call_id=tool_call_id)]}
    )


@tool("elicit")
async def elicit_tool(
    technique: Annotated[
        Literal[
            "5_whys",
            "reverse",
            "moscow",
            "first_principles",
            "comparable_products",
            "pre_mortem",
            "tree_of_thought",
            "socratic_questioning",
            "challenge_assumptions",
        ],
        "BMAD elicitation technique applied to the seed.",
    ],
    seed: Annotated[str, "Seed/topic to apply the technique to."],
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Apply a BMAD elicitation technique to a seed and return a structured frame to reason over and record as nodes.

    comparable_products fetches real external knowledge via web_search (falls back to model knowledge).
    Each successful call increments session_elicit_count so policy knows the cold start has been explored.
    """
    try:
        result = elicit(technique, seed)
    except ValueError as exc:
        return _recoverable_tool_update(
            RecoverableToolError(code="elicit_unknown_technique", message=str(exc), user_fixable=True),
            tool_call_id,
        )
    # Emit a DELTA (+1), not an absolute count: the channel uses an additive reducer so two elicits in
    # one turn accumulate correctly. Returning state+1 from both (the same pre-turn snapshot) would
    # either collide (no reducer) or double-count (absolute + add). _ = state kept for signature parity.
    _ = state
    return Command(
        update={
            "session_elicit_count": 1,
            "messages": [ToolMessage(content=json.dumps(result, ensure_ascii=False), tool_call_id=tool_call_id)],
        }
    )


def get_all_analyzer_tools() -> list:
    """Full tool registry for the ToolNode and the analyze schema; availability lives in the prompt + tool guards."""
    return [
        ask_user,
        respond,
        write_draft,
        finalize,
        critique_note,
        explore_note,
        read_artifact,
        read_source_documents_tool,
        run_critique,
        recommend_next_workflow,
        run_readiness_check,
        confirm_intent,
        elicit_tool,
        web_search_tool,
        create_decision_node,
        update_decision_node,
        supersede_decision_node,
        dismiss_question,
        run_impact_analysis,
        read_artifact_graph_tool,
        create_artifact_link_tool,
        propose_retirement_tool,
    ]


def _finalize_gate_open(state: WorkflowState) -> bool:
    """Quality side of the finalize gate: gate passed AND the scored draft is still current.

    The current draft body comes from `current_draft_body` — the SAME helper `_run_critique_impl`
    writes the hash from — so the gate body can never diverge from the scored body. Escape hatch: at
    the rounds cap a passing gate finalizes regardless of hash, so an edit after the final critique
    cannot wedge the loop (run_critique is capped, finalize would otherwise be stuck on a stale hash).
    """
    report = state.get("quality_report")
    if not report or report.get("quality_gate_result") != "pass":
        log_gate_decision("finalize", "blocked", reason="critique_not_passed")
        return False
    readiness = state.get("candidate_readiness")
    if not isinstance(readiness, dict) or readiness.get("state") != ArtifactReadinessState.SUFFICIENT:
        log_gate_decision("finalize", "blocked", reason="readiness_not_sufficient")
        return False
    if (state.get("critique_rounds") or 0) >= CRITIQUE_ROUNDS_MAX:
        log_gate_decision("finalize", "open", reason="rounds_cap")
        return True
    current_hash = hashlib.md5(_cached_draft_body(state).encode()).hexdigest()[:8]
    if current_hash != state.get("last_critiqued_draft_hash"):
        log_gate_decision("finalize", "blocked", reason="stale_draft")
        return False
    log_gate_decision("finalize", "open")
    return True


def _phase_signals(state: WorkflowState) -> PhaseSignals:
    """State-derived facts the session-phase machine transitions on (single computation site)."""
    has_draft = bool(_cached_draft_body(state).strip())
    critique_started = bool(state.get("critique_rounds") or 0) or bool(state.get("quality_report"))
    return PhaseSignals(
        user_confirmed=state.get("user_confirmed") is not None,
        has_draft=has_draft,
        has_evidence=bool(state.get("decision_nodes")) or bool(state.get("session_elicit_count") or 0),
        critique_started=critique_started,
        # Only consult the finalize gate when it can matter — it logs each evaluation.
        finalize_open=bool(has_draft and critique_started and _finalize_gate_open(state)),
    )


def current_session_phase(state: WorkflowState) -> str:
    """The phase gating reads: orchestrator's persisted value, derived on the fly for legacy
    checkpoints (resume paths can re-enter the ToolNode before orchestrator has ever run)."""
    return state.get("session_phase") or derive_phase(_phase_signals(state))


def get_available_tools(state: WorkflowState) -> list:
    """Tools the loop may pick this turn, gated on state.

    The menu is intentionally broad. Two gate layers: the session-phase menu (see
    app/graphs/session_phase.py — the single authority for which loop tools a phase offers) and
    the draft/quality conditions (`finalize` needs a passing current critique, critique/readiness
    tools need a rendered draft). POLICY in policy.py is documentation-only for loop tools.
    """
    tools = [
        ask_user,
        respond,
        write_draft,
        critique_note,
        explore_note,
        confirm_intent,
        read_artifact,
        read_source_documents_tool,
        read_artifact_graph_tool,
        create_artifact_link_tool,
        propose_retirement_tool,
        run_impact_analysis,
        elicit_tool,
        web_search_tool,
    ]
    has_draft = bool(_cached_draft_body(state).strip())
    critique_rounds = state.get("critique_rounds") or 0
    if has_draft and critique_rounds > 0 and _finalize_gate_open(state):
        tools.append(finalize)
    if has_draft and critique_rounds < CRITIQUE_ROUNDS_MAX:
        tools.append(run_critique)
    # recommend_next_workflow: available once there is a draft, or once >= 2 sections have any
    # coverage (lets the quick track recommend early, before a draft exists).
    coverage = state.get("section_coverage") or {}
    sections_with_signal = sum(1 for v in coverage.values() if status_score(v) > 0.0)
    if has_draft or sections_with_signal >= 2:
        tools.append(recommend_next_workflow)
    # run_readiness_check needs an artifact AND a prior critique round — readiness is meaningless
    # before any quality signal exists. Routes through current_draft_body (has_draft) so a DB-loaded
    # draft qualifies, same as finalize.
    if has_draft and critique_rounds > 0:
        tools.append(run_readiness_check)
    tools.extend(_decision_graph_menu(state))
    phase = current_session_phase(state)
    phase_filtered = [t for t in tools if phase_allows(phase, t.name)]
    return filter_lifecycle_menu_tools(phase_filtered, state)


def _decision_graph_menu(state: WorkflowState) -> list:
    """Decision-graph tools available this turn — empty when the feature flag is off.

    create is always offered (nodes can be created from a fresh graph); update/supersede only once at
    least one node exists.
    """
    if not settings.decision_graph_enabled:
        return []
    menu = [create_decision_node]
    decision_nodes = state.get("decision_nodes") or {}
    if decision_nodes:
        menu.extend([update_decision_node, supersede_decision_node])
    if any(n.get("kind") == "open_question" and n.get("status") == "parked" for n in decision_nodes.values()):
        menu.append(dismiss_question)
    return menu
