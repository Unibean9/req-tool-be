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
    recommended_next_action: str  # "finalize" | "revise"


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
    # `draft_body or working_draft or ""` (Phase 3).
    last_critiqued_draft_hash: str | None
    locale: str | None
    # Triage channels: the entry node classifies each fresh turn as "converse" (greeting/smalltalk)
    # or "work" (requirements analysis) and, for converse, stages the reply for converse_node.
    turn_type: str | None
    triage_reply: str | None
    section_coverage: dict[str, str] | None
    coverage_ratio: float | None
    coverage_complete: bool | None
    section_coverage_stall_count: int | None
    # Structured analytical objects extracted from note tools (spec §7.1). Accumulate across turns;
    # populated by the note parser, queried by validators (Phase 5) and the finalize gate.
    assumptions: list[AssumptionObject]
    risks: list[RiskObject]
    open_questions: list[OpenQuestionObject]
    # Persisted draft body loaded from the DB each analyze turn — lets run_critique target the
    # confirmed artifact even when no in-session working_draft exists yet.
    draft_body: str | None
    working_draft: str | None
    # BMAD method layer (addendum §8) — sits above the 7-section engine; analyze_node assigns
    # workflow_mode / planning_track each turn. Independent of active_mode.
    method_profile: MethodProfile
    artifact_chain: ArtifactChain
    readiness: Readiness
    # Multi-angle mode steering. A one-shot hint set by the user to switch the
    # agent to critique/explore/etc.; analyze_node consumes it and clears it the same turn.
    mode_hint: str | None
