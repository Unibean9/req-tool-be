from typing import Annotated, Any, TypedDict

from langgraph.graph import add_messages


class WorkflowState(TypedDict):
    artifact_type: str
    workflow_area: str
    step_key: str | None
    messages: Annotated[list[dict[str, Any]], add_messages]
    analysis_result: dict[str, Any] | None
    pending_tool_call_ids: list[str]
    last_agent_run_id: str | None
    turn_count: int
    missing_context: list[str]
    user_confirmed: bool | None
    critique_rounds: int
    quality_report: dict[str, Any] | None

