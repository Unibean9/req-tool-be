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
import logging

from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from app.config import settings
from app.graphs import gating
from app.graphs.agent_tools._shared import (
    RecoverableToolError,
    _missing_required_arg_update,
    _node_origin,
    _recoverable_tool_update,
    _tool_not_available_update,
)
from app.graphs.decision_graph import render_view
from app.graphs.gate_logging import log_gate_decision
from app.graphs.gating import Mode, menu_rules
from app.graphs.session_phase import PhaseSignals, derive_phase
from app.graphs.state import WorkflowState
from app.schemas.artifact_synthesis import ArtifactReadinessState

logger = logging.getLogger(__name__)


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


# interaction tools moved to agent_tools.interaction; re-exported for stable imports.
# artifact-link tools moved to agent_tools.artifact_links; re-exported for stable imports.
from app.graphs.agent_tools.artifact_links import (  # noqa: E402
    _config_ids,
    _create_artifact_link_impl,
    _load_artifact_links,
    _proposal_run_id,
    _propose_retirement_impl,
    _read_artifact_graph_impl,
    _run_impact_analysis_impl,
    _save_approval_proposal,
    _session_user_id,
    create_artifact_link_tool,
    propose_retirement_tool,
    read_artifact_graph_tool,
    run_impact_analysis,
)

# artifact-read tools moved to agent_tools.artifact_read; re-exported for stable imports.
from app.graphs.agent_tools.artifact_read import (  # noqa: E402
    READ_ARTIFACT_MAX_CHARS,
    READ_SOURCE_DOCUMENT_MAX_CHARS,
    READ_SOURCE_DOCUMENT_MAX_ITEMS,
    _normalize_source_document_ids,
    _read_artifact_impl,
    _read_artifact_source_context,
    _read_source_documents_impl,
    _source_document_source_context,
    read_artifact,
    read_source_documents_tool,
)

# run_critique moved to agent_tools.critique_tool; re-exported for stable imports.
from app.graphs.agent_tools.critique_tool import (  # noqa: E402
    CRITIQUE_ROUNDS_MAX,
    _run_critique_impl,
    run_critique,
)

# draft-lifecycle tools moved to agent_tools.draft_lifecycle; re-exported for stable imports.
# _cold_start_draft_blocked is re-exported because gating/dispatch_rules reaches it via
# agent_tools._cold_start_draft_blocked. Draft-cache + gates stay defined in this module below.
from app.graphs.agent_tools.draft_lifecycle import (  # noqa: E402
    AUTO_EVIDENCE_EXCERPT_MAX_CHARS,
    _apply_executive_summary_resume,
    _cold_start_draft_blocked,
    _dedupe_keep_order,
    _evidence_excerpt,
    _finalize_impl,
    _has_complete_required_headings,
    _load_accepted_bodies,
    _missing_required_headings,
    _persist_executive_summary_draft,
    _proposal_based_on,
    _proposal_lifecycle_metadata,
    _proposal_source_evidence,
    _resolve_proposed_body,
    _stale_predecessor_warnings,
    _write_draft_impl,
    finalize,
    synthesize_executive_summary,
    write_draft,
)
from app.graphs.agent_tools.interaction import (  # noqa: E402
    _BATCH_QUESTION_TYPES,
    _MAX_BATCH_QUESTIONS,
    _ask_user_impl,
    _audit_interaction_tool_call,
    _confirm_intent_impl,
    _normalize_batch_questions,
    _render_batched_question_text,
    _respond_impl,
    ask_user,
    confirm_intent,
    respond,
)

# notes moved to agent_tools.notes; re-exported for stable imports.
from app.graphs.agent_tools.notes import (  # noqa: E402
    _write_note_impl,
    critique_note,
    explore_note,
    note,
)

# workflow/readiness moved to agent_tools.workflow; re-exported for stable imports.
from app.graphs.agent_tools.workflow import (  # noqa: E402
    _BRIEF_SECTIONS,
    _compute_recommendation,
    _recommend_next_workflow_impl,
    _run_readiness_check_impl,
    recommend_next_workflow,
    run_readiness_check,
)

# ---------------------------------------------------------------------------
# get_available_tools — state-driven gate over the tool-loop
# ---------------------------------------------------------------------------

# After this many run_critique calls the formal judge is gated off the menu so the loop cannot
# spin on critique forever (spec §5.5). write_draft / ask_user stay available regardless.
# Sourced from config (max_critique_rounds) so the reflection-round cap is a single tunable.
# Cap on LLM judge calls the orchestrator's diagnosis step may spend per turn escalating a
# heuristic high-risk classification.
DIAGNOSIS_JUDGE_CALLS_MAX = settings.max_diagnosis_judge_calls


# Elicitation surface moved to agent_tools.elicitation; re-exported for stable imports.
from app.graphs.agent_tools.elicitation import (  # noqa: E402
    ELICIT_TECHNIQUES,
    _default_search_client,
    _duckduckgo_search,
    elicit,
    elicit_tool,
    web_search,
    web_search_tool,
)


def get_all_analyzer_tools() -> list:
    """Full tool registry for the ToolNode and the analyze schema; availability lives in the prompt + tool guards.

    `critique_note`/`explore_note` stay in this registry (no longer on the menu, see
    `get_available_tools`) only so `ToolNode` can re-execute an old tool_call when resuming a
    checkpoint created before the two tools were merged into `note`.
    """
    return [
        ask_user,
        respond,
        write_draft,
        finalize,
        note,
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
        run_impact_analysis,
        read_artifact_graph_tool,
        create_artifact_link_tool,
        propose_retirement_tool,
    ]


def _draft_hash_stale(state: WorkflowState) -> bool:
    """True iff the current draft body's hash differs from the hash of the last-critiqued draft.

    Pure function of state (no I/O), shared by `_finalize_gate_open`, the `run_critique` menu rule,
    and `_run_critique_impl`'s cap guard so the three can never drift apart on what counts as an
    edit since the last critique.
    """
    current_hash = hashlib.md5(_cached_draft_body(state).encode()).hexdigest()[:8]
    return current_hash != state.get("last_critiqued_draft_hash")


def _finalize_gate_open(state: WorkflowState) -> bool:
    """Quality side of the finalize gate: gate passed AND the scored draft is still current.

    The current draft body comes from `current_draft_body` — the SAME helper `_run_critique_impl`
    writes the hash from — so the gate body can never diverge from the scored body. The hash must
    always match the last-critiqued hash for finalize to open; there is no rounds-based exception —
    an edit after the last critique always re-blocks finalize until the draft is re-scored.
    """
    report = state.get("quality_report")
    if not report or report.get("quality_gate_result") != "pass":
        log_gate_decision("finalize", "blocked", reason="critique_not_passed")
        return False
    readiness = state.get("candidate_readiness")
    if not isinstance(readiness, dict) or readiness.get("state") != ArtifactReadinessState.SUFFICIENT:
        log_gate_decision("finalize", "blocked", reason="readiness_not_sufficient")
        return False
    if _draft_hash_stale(state):
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

    Single chokepoint: every candidate tool in the fixed universe below is offered to
    `gating.check(..., Mode.MENU)`, which runs the registered per-call rules in order — the
    draft/quality/decision-graph rules first, then a combined session-phase + artifact-lifecycle
    rule last (see app/graphs/gating/menu_rules.py). That last rule is silent in menu-mode; a
    dispatch-mode counterpart logs its decisions once dispatch-time gating is wired through the
    same rule. POLICY in policy.py governs repository tools via `@governed` and does not gate loop
    tools.
    """
    candidates = [
        ask_user,
        respond,
        write_draft,
        note,
        confirm_intent,
        read_artifact,
        read_source_documents_tool,
        read_artifact_graph_tool,
        create_artifact_link_tool,
        propose_retirement_tool,
        run_impact_analysis,
        elicit_tool,
        web_search_tool,
        finalize,
        run_critique,
        recommend_next_workflow,
        run_readiness_check,
    ]
    menu_rules.ensure_menu_rules_registered()
    # Computed once (not per candidate tool): current_session_phase can invoke _finalize_gate_open
    # (via _phase_signals), which logs — a call per candidate would multiply that logging.
    phase = current_session_phase(state)
    return [
        tool
        for tool in candidates
        if gating.check({"name": tool.name, "phase": phase}, state, Mode.MENU).is_allow
    ]


__all__ = [
    'AUTO_EVIDENCE_EXCERPT_MAX_CHARS',
    'CRITIQUE_ROUNDS_MAX',
    'DIAGNOSIS_JUDGE_CALLS_MAX',
    'ELICIT_TECHNIQUES',
    'READ_ARTIFACT_MAX_CHARS',
    'READ_SOURCE_DOCUMENT_MAX_CHARS',
    'READ_SOURCE_DOCUMENT_MAX_ITEMS',
    'RecoverableToolError',
    '_BATCH_QUESTION_TYPES',
    '_BRIEF_SECTIONS',
    '_MAX_BATCH_QUESTIONS',
    '_apply_executive_summary_resume',
    '_ask_user_impl',
    '_audit_interaction_tool_call',
    '_cached_draft_body',
    '_cold_start_draft_blocked',
    '_compute_recommendation',
    '_config_ids',
    '_confirm_intent_impl',
    '_create_artifact_link_impl',
    '_current_draft_body_sync',
    '_dedupe_keep_order',
    '_default_search_client',
    '_draft_hash_stale',
    '_duckduckgo_search',
    '_evidence_excerpt',
    '_finalize_gate_open',
    '_finalize_impl',
    '_has_complete_required_headings',
    '_load_accepted_bodies',
    '_load_artifact_links',
    '_missing_required_arg_update',
    '_missing_required_headings',
    '_node_origin',
    '_normalize_batch_questions',
    '_normalize_source_document_ids',
    '_persist_executive_summary_draft',
    '_phase_signals',
    '_proposal_based_on',
    '_proposal_lifecycle_metadata',
    '_proposal_run_id',
    '_proposal_source_evidence',
    '_propose_retirement_impl',
    '_read_artifact_graph_impl',
    '_read_artifact_impl',
    '_read_artifact_source_context',
    '_read_source_documents_impl',
    '_recommend_next_workflow_impl',
    '_recoverable_tool_update',
    '_render_batched_question_text',
    '_resolve_proposed_body',
    '_respond_impl',
    '_run_critique_impl',
    '_run_impact_analysis_impl',
    '_run_readiness_check_impl',
    '_save_approval_proposal',
    '_session_user_id',
    '_source_document_source_context',
    '_stale_predecessor_warnings',
    '_tool_is_available',
    '_tool_not_available_update',
    '_write_draft_impl',
    '_write_note_impl',
    'artifact_stage',
    'ask_user',
    'confirm_intent',
    'create_artifact_link_tool',
    'critique_note',
    'current_draft_body',
    'current_session_phase',
    'elicit',
    'elicit_tool',
    'explore_note',
    'finalize',
    'get_all_analyzer_tools',
    'get_available_tools',
    'interrupt',
    'log_gate_decision',
    'logger',
    'note',
    'propose_retirement_tool',
    'read_artifact',
    'read_artifact_graph_tool',
    'read_source_documents_tool',
    'recommend_next_workflow',
    'respond',
    'run_critique',
    'run_impact_analysis',
    'run_readiness_check',
    'synthesize_executive_summary',
    'web_search',
    'web_search_tool',
    'write_draft',
]
