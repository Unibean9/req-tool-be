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
    quality_report: dict[str, Any] | None
    locale: str | None
    intent: str | None
    section_coverage: dict[str, str] | None
    coverage_ratio: float | None
    coverage_complete: bool | None
    section_coverage_stall_count: int | None
    # Structured analytical objects extracted from note tools (spec §7.1). Accumulate across turns;
    # populated by the note parser, queried by validators (Phase 5) and the finalize gate.
    assumptions: list[AssumptionObject]
    risks: list[RiskObject]
    open_questions: list[OpenQuestionObject]
    working_draft: str | None
    # Multi-angle mode steering. A one-shot hint set by the user to switch the
    # agent to critique/explore/etc.; analyze_node consumes it and clears it the same turn.
    mode_hint: str | None
