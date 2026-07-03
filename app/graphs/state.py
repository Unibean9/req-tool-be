import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import add_messages


class AssumptionObject(TypedDict):
    """A captured assumption (spec §5.4): what is being relied on and how confident we are."""

    statement: str
    source: str
    confidence: str
    impact: str
    owner: str
    status: str


class RiskObject(TypedDict):
    """A captured risk: an adverse event with its likelihood and mitigation."""

    statement: str
    likelihood: str
    impact: str
    mitigation: str
    owner: str
    status: str


class OpenQuestionObject(TypedDict):
    """An unresolved question blocking or informing the requirements."""

    question: str
    domain: str
    decision_needed: str
    status: str


class KeyFactObject(TypedDict):
    """A key fact about the project: a confirmed datum that must survive conversation compression."""

    statement: str
    source: str
    turn: str


class DecisionNode(TypedDict):
    """A node in the decision graph — the unit of truth for an artifact.

    Lifecycle preserves history: changing a decision creates a new node that supersedes the old one
    (never deleted). depends_on edges enable backward ripple when a node is superseded. The artifact
    view (BRD/PRD) is a derived projection rendered from active nodes, never the source.
    """

    id: str
    kind: str  # objective | scope | assumption | decision | risk | open_question | fact
    statement: str
    status: str  # proposed | confirmed | inferred | needs_confirmation | parked | superseded | dismissed
    origin: dict[str, Any]  # {turn, by, technique, source} — why this node exists
    depends_on: list[str]
    supersedes: str | None
    superseded_by: str | None
    blocks: list[str]
    answer: str | None
    # Which output-contract required_heading this node renders under (e.g. "## Objectives"). The 7 kinds
    # don't map 1:1 to contract sections (a vision and an objective are both kind=objective), so the
    # node carries its target section explicitly. None → render_view falls back to a kind heuristic.
    section: str | None
    # Column values for a table section, keyed by the contract's table_columns (e.g.
    # {"goal": ..., "metric": ..., "target": ...}). A single free-text statement cannot fill an N-column
    # table; fields carries that structure. A section renders as a table when its nodes carry fields.
    fields: dict[str, str] | None
    # Audit trail set only when status becomes "dismissed": {reason, turn, dismissed_by}. Absent on
    # every other node (backward-compatible: legacy checkpoints have no dismissed nodes at all).
    dismissal: dict[str, Any] | None


def merge_decision_nodes(
    left: dict[str, "DecisionNode"] | None, right: dict[str, "DecisionNode"] | None
) -> dict[str, "DecisionNode"]:
    """Merge decision-graph updates per node-id instead of replacing the whole dict.

    Two decision tools in one turn each receive the pre-turn snapshot via InjectedState and each return
    the full graph; a plain-replace channel would let the second clobber the first's new node. Per-id
    merge keeps both: a key present only in `left` survives because `right` (built from the same
    snapshot) simply does not mention it. The graph is non-destructive — no tool ever removes a node —
    so a merge can never resurrect deleted history.
    """
    if not left:
        return right or {}
    if not right:
        return left
    return {**left, **right}


def merge_section_findings(
    left: dict[str, list] | None, right: dict[str, list] | None
) -> dict[str, list]:
    """Merge section-finding updates per section-key, mirroring merge_decision_nodes.

    Two section-writing tools in one turn each build from the same pre-turn snapshot and return the
    full dict; per-key union keeps both writes. A passing section is stored as [] (never dropped) so
    clearing a defect propagates rather than being lost by omission.
    """
    if not left:
        return right or {}
    if not right:
        return left
    return {**left, **right}


class MethodProfile(TypedDict):
    """BMAD method profile (addendum §8): which planning workflow the project is in."""

    method: str
    planning_track: str
    project_type: str
    current_workflow: str
    recommended_next_workflow: str | None


class ArtifactChain(TypedDict):
    """Status of each BMAD planning artifact stage (addendum §8)."""

    brainstorming: str
    product_brief: str
    prd: str


class Readiness(TypedDict):
    """Readiness assessment across the planning lifecycle (addendum §8)."""

    requirements_ready: bool
    architecture_needed: str
    implementation_ready: bool
    blocking_gaps: list[str]
    recommended_next_step: str | None


class QualityReport(TypedDict):
    """Reflection feedback contract — the structured verdict run_critique writes to state.

    Four base fields mirror the judge output (critique.py); the five derived fields are
    post-classified in Python by `_run_critique_impl`. The gate result is derived from `score`
    against `settings.critique_score_threshold`, never from whether `blocking_issues` is empty —
    so the no-LLM degraded path (score=0.0, findings=[]) still yields "fail" (fail-safe).
    """

    mode: str
    score: float
    findings: list[str]
    suggestions: list[str]
    blocking_issues: list[str]
    non_blocking_warnings: list[str]
    revision_plan: list[str]
    quality_gate_result: str  # "pass" | "fail" — derived from score, not from blocking_issues
    recommended_next_action: str  # "finalize" | "revise" | "escalate" (cap reached, gate still fails)


# Defaults for a fresh session — a small idea starts on the quick track at brainstorm.
DEFAULT_METHOD_PROFILE: MethodProfile = {
    "method": "bmad_inspired",
    "planning_track": "quick",
    "project_type": "unknown",
    "current_workflow": "brainstorm",
    "recommended_next_workflow": None,
}

DEFAULT_ARTIFACT_CHAIN: ArtifactChain = {
    "brainstorming": "missing",
    "product_brief": "missing",
    "prd": "missing",
}

DEFAULT_READINESS: Readiness = {
    "requirements_ready": False,
    "architecture_needed": "unknown",
    "implementation_ready": False,
    "blocking_gaps": [],
    "recommended_next_step": None,
}


class WorkflowState(TypedDict):
    artifact_type: str
    workflow_area: str
    step_key: str | None
    messages: Annotated[list[dict[str, Any]], add_messages]
    conversation_summary: str
    analysis_result: dict[str, Any] | None
    pending_tool_call_ids: list[str]
    last_agent_run_id: str | None
    turn_count: int
    missing_context: list[str]
    user_confirmed: bool | None
    critique_rounds: int
    quality_report: QualityReport | None
    # MD5(8) of the draft body the last run_critique scored — lets the finalize gate detect a draft
    # edited after critique (stale report) and force a re-critique. Source body is always
    # `current_draft_body(state)`.
    last_critiqued_draft_hash: str | None
    locale: str | None
    # Triage channels: the entry node classifies each fresh turn as "converse" (greeting/smalltalk)
    # or "work" (requirements analysis) and, for converse, stages the reply for converse_node.
    turn_type: str | None
    triage_reply: str | None
    section_coverage: dict[str, str] | None
    coverage_complete: bool | None
    section_coverage_stall_count: int | None
    # Structured analytical objects extracted from note tools (spec §7.1). Accumulate across turns;
    # populated by the note parser. assumptions and open_questions are derived from the decision graph;
    # see decision_graph.derive_assumptions / derive_open_questions.
    # Additive reducers keep same-turn parallel note writes from colliding.
    risks: Annotated[list[RiskObject], operator.add]
    # Confirmed facts that must survive conversation compression. Never included in summarize_node
    # compression — they are the ground truth the analyst builds on.
    key_facts: Annotated[list[KeyFactObject], operator.add]
    # Exact document item this session reads and writes.
    focused_artifact_id: str | None
    # Persisted draft body loaded from the DB each analyze turn. The decision graph renders the live
    # draft view; this field stays as DB context for document workflows.
    draft_body: str | None
    candidate_readiness: dict[str, Any] | None
    # Append-only recoverable tool error log. Multiple tool failures can be returned in one ToolNode
    # superstep, so this channel needs the same additive reducer discipline as note outputs.
    tool_errors: Annotated[list[dict[str, Any]], operator.add]
    feedback_summary: dict[str, Any] | None
    verification_status: dict[str, Any] | None
    latest_checked_revision: str | None
    # BMAD method layer (addendum §8) — sits above the 7-section engine; analyze_node assigns
    # workflow_mode / planning_track each turn.
    method_profile: MethodProfile
    artifact_chain: ArtifactChain
    readiness: Readiness
    # Multi-angle mode steering. A one-shot hint set by the user to switch the
    # agent to critique/explore/etc.; analyze_node consumes it and clears it the same turn.
    mode_hint: str | None
    # Count of successful elicit() calls this session. Drives the cold-start hard gate: a fresh
    # project must run at least one elicitation before write_draft is offered. Each elicit emits a +1
    # delta; the additive reducer accumulates them so two elicits in one turn don't collide
    # (INVALID_CONCURRENT_GRAPH_UPDATE). Persists across resume within a session.
    session_elicit_count: Annotated[int, operator.add]
    # Decision graph keyed by node id — source of truth for the artifact. Old sessions missing this key
    # default to {} on load without migration or crash. The merge reducer keeps concurrent same-turn
    # node writes from clobbering each other (see merge_decision_nodes).
    decision_nodes: Annotated[dict[str, DecisionNode], merge_decision_nodes]
    # Bounded fingerprint history of dispatched tool calls (name + sorted-args), newest last. Populated
    # once per turn in analyze_node at the same site dispatched_tools is built; read by route_node to
    # detect N consecutive identical calls and exit the analyze/tools cycle early. Plain replace-on-write
    # (like turn_count) — no special merge needed since it's written at a single site per turn.
    recent_tool_calls: list[str]
    # System-selected thinking mode for this turn ("structuring" | "challenging" | "synthesizing" |
    # "risk_probing"), written every turn by orchestrator_node's heuristic diagnosis step and read by
    # analyze_node's prompt builder. Plain replace-on-write, mirroring mode_hint's placement — the
    # system-driven sibling of the user-driven mode_hint steer. None when adaptive diagnosis is
    # disabled (enable_adaptive_diagnosis=False).
    thinking_mode: str | None
    # Diagnosis result backing thinking_mode: {"risk_level": "low"|"high", "signals": [...]}. Plain
    # replace-on-write, fully overwritten each turn — no accumulation semantics.
    diagnosis_signal: dict[str, Any] | None
    # Count of LLM judge calls spent escalating a high-risk diagnosis, session-lifetime. Plain
    # replace-on-write; compared against DIAGNOSIS_JUDGE_CALLS_MAX to gate further escalation.
    diagnosis_judge_calls_used: int
    # Explicit workflow position: "intent"|"elicit"|"draft"|"review"|"finalize" (session_phase.py).
    # SINGLE WRITER: only orchestrator_node assigns it (via session_phase.transition); readers fall
    # back to on-the-fly derivation for checkpoints created before this field existed.
    session_phase: str | None
    # Count of model tool selections rejected by the per-phase gate, session-lifetime. Written at a
    # single site per turn (analyze result); read by the behavior eval's out_of_phase metric.
    out_of_phase_tool_calls: int
    # Section-scoped structural findings from the last write of each section, keyed by the section
    # heading. A passing section stores [] (not absent) so a re-validation can clear a
    # prior defect through the union merge. Two decision tools in one turn each return the full dict
    # built from the same snapshot, so per-key merge keeps both — mirroring decision_nodes.
    section_findings: Annotated[dict[str, list[dict[str, Any]]], merge_section_findings]


def build_initial_workflow_state(
    *,
    artifact_type: str,
    workflow_area: str,
    step_key: str | None,
    messages: list[dict[str, Any]] | None = None,
    missing_context: list[str] | None = None,
    focused_artifact_id: Any = None,
    mode_hint: str | None = None,
) -> WorkflowState:
    """Build the canonical initial state for every graph entry point."""
    return {
        "artifact_type": artifact_type,
        "workflow_area": workflow_area,
        "step_key": step_key,
        "messages": list(messages or []),
        "conversation_summary": "",
        "analysis_result": None,
        "pending_tool_call_ids": [],
        "last_agent_run_id": None,
        "turn_count": 0,
        "missing_context": list(missing_context or []),
        "user_confirmed": None,
        "critique_rounds": 0,
        "quality_report": None,
        "last_critiqued_draft_hash": None,
        "locale": None,
        "turn_type": None,
        "triage_reply": None,
        "section_coverage": None,
        "coverage_complete": None,
        "section_coverage_stall_count": None,
        "risks": [],
        "key_facts": [],
        "focused_artifact_id": str(focused_artifact_id) if focused_artifact_id is not None else None,
        "draft_body": None,
        "candidate_readiness": None,
        "tool_errors": [],
        "feedback_summary": None,
        "verification_status": None,
        "latest_checked_revision": None,
        "method_profile": dict(DEFAULT_METHOD_PROFILE),
        "artifact_chain": dict(DEFAULT_ARTIFACT_CHAIN),
        "readiness": dict(DEFAULT_READINESS),
        "mode_hint": mode_hint,
        "session_elicit_count": 0,
        "decision_nodes": {},
        "recent_tool_calls": [],
        "thinking_mode": None,
        "diagnosis_signal": None,
        "diagnosis_judge_calls_used": 0,
        "session_phase": None,
        "out_of_phase_tool_calls": 0,
        "section_findings": {},
    }
