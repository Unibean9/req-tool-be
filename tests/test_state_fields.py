from app.graphs.state import WorkflowState


def test_workflow_state_accepts_section_fields():
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
        "last_critiqued_draft_hash": None,
        "locale": None,
        "turn_type": None,
        "triage_reply": None,
        "section_coverage": None,
        "section_assessment": None,
        "coverage_ratio": None,
        "coverage_complete": None,
        "section_coverage_stall_count": None,
        "assumptions": [],
        "risks": [],
        "open_questions": [],
        "sections_body": {"vision_objectives": "Mục tiêu"},
        "focus_section": "vision_objectives",
        "draft_body": None,
        "working_draft": None,
        "method_profile": {
            "method": "bmad_inspired",
            "planning_track": "quick",
            "project_type": "unknown",
            "current_workflow": "brainstorm",
            "recommended_next_workflow": None,
        },
        "artifact_chain": {
            "brainstorming": "missing",
            "product_brief": "missing",
            "prd": "missing",
        },
        "readiness": {
            "requirements_ready": False,
            "architecture_needed": "unknown",
            "implementation_ready": False,
            "blocking_gaps": [],
            "recommended_next_step": None,
        },
        "mode_hint": None,
    }

    assert state["section_coverage"] is None
    assert state["section_assessment"] is None
    assert state["coverage_ratio"] is None
    assert state["coverage_complete"] is None
    assert state["sections_body"]["vision_objectives"] == "Mục tiêu"
    assert state["focus_section"] == "vision_objectives"


def test_state_has_section_coverage_field():
    assert "section_coverage" in WorkflowState.__annotations__
    assert "section_assessment" in WorkflowState.__annotations__
    assert "section_coverage_stall_count" in WorkflowState.__annotations__
    assert "sections_body" in WorkflowState.__annotations__
    assert "focus_section" in WorkflowState.__annotations__


def test_state_no_longer_has_slot_coverage_field():
    assert "slot_coverage" not in WorkflowState.__annotations__
    assert "last_asked_slot" not in WorkflowState.__annotations__
    assert "coverage_stall_count" not in WorkflowState.__annotations__
