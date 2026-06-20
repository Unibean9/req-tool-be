from app.graphs.state import WorkflowState


def test_workflow_state_accepts_slot_fields():
    state: WorkflowState = {
        "artifact_type": "problem",
        "workflow_area": "analysis",
        "step_key": None,
        "messages": [],
        "conversation_summary": "",
        "analysis_result": None,
        "pending_tool_call_ids": [],
        "last_agent_run_id": None,
        "turn_count": 0,
        "missing_context": [],
        "user_confirmed": None,
        "critique_rounds": 0,
        "quality_report": None,
        "locale": None,
        "intent": None,
        "slot_coverage": None,
        "coverage_ratio": None,
        "coverage_complete": None,
    }

    assert state["slot_coverage"] is None
    assert state["coverage_ratio"] is None
    assert state["coverage_complete"] is None
    assert "slot_coverage" in WorkflowState.__annotations__
    assert "coverage_ratio" in WorkflowState.__annotations__
    assert "coverage_complete" in WorkflowState.__annotations__
