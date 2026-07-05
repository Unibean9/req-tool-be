from app.graphs.state import WorkflowState, build_initial_workflow_state


def test_workflow_state_accepts_document_focus_fields():
    state: WorkflowState = build_initial_workflow_state(
        artifact_type="problem",
        workflow_area="analysis",
        step_key=None,
        focused_artifact_id="00000000-0000-0000-0000-000000000001",
    )

    assert state["section_coverage"] is None
    assert state["coverage_complete"] is None
    assert state["focused_artifact_id"] is not None
    assert "assumptions" not in state
    assert "open_questions" not in state


def test_state_has_key_facts_field():
    assert "key_facts" in WorkflowState.__annotations__


def test_initial_state_seeds_key_facts_as_empty_list():
    state = build_initial_workflow_state(
        artifact_type="brd", workflow_area="analysis", step_key=None
    )
    assert state["key_facts"] == []


def test_state_has_section_coverage_field():
    assert "section_coverage" in WorkflowState.__annotations__
    assert "section_coverage_stall_count" in WorkflowState.__annotations__
    assert "focused_artifact_id" in WorkflowState.__annotations__
    assert "turn_context_artifacts" in WorkflowState.__annotations__
    assert "lifecycle_reports" in WorkflowState.__annotations__
    assert "artifact_history" in WorkflowState.__annotations__
    assert "candidate_readiness" in WorkflowState.__annotations__
    assert "tool_errors" in WorkflowState.__annotations__
    assert "feedback_summary" in WorkflowState.__annotations__
    assert "verification_status" in WorkflowState.__annotations__
    assert "latest_checked_revision" in WorkflowState.__annotations__
    assert "analysis_frame" not in WorkflowState.__annotations__
    assert "working_draft" not in WorkflowState.__annotations__
    assert "section_assessment" not in WorkflowState.__annotations__
    assert "coverage_ratio" not in WorkflowState.__annotations__
    assert "sections_body" not in WorkflowState.__annotations__
    assert "focus_section" not in WorkflowState.__annotations__


def test_state_no_longer_has_slot_coverage_field():
    assert "slot_coverage" not in WorkflowState.__annotations__
    assert "last_asked_slot" not in WorkflowState.__annotations__
    assert "coverage_stall_count" not in WorkflowState.__annotations__


def test_initial_state_seeds_turn_context_artifacts_empty():
    state = build_initial_workflow_state(
        artifact_type="brd",
        workflow_area="analysis",
        step_key=None,
    )

    assert state["turn_context_artifacts"] == []
    assert state["lifecycle_reports"] == []
    assert state["artifact_history"] == []


def test_initial_workflow_state_seeds_governance_fields():
    state = build_initial_workflow_state(
        artifact_type="vision_objectives",
        workflow_area="analysis",
        step_key=None,
        messages=[{"role": "user", "content": "Create vision"}],
        missing_context=["brd"],
        focused_artifact_id="00000000-0000-0000-0000-000000000001",
        mode_hint="critique",
    )

    assert state["messages"] == [{"role": "user", "content": "Create vision"}]
    assert state["missing_context"] == ["brd"]
    assert state["focused_artifact_id"] == "00000000-0000-0000-0000-000000000001"
    assert state["mode_hint"] == "critique"
    assert state["critique_rounds"] == 0
    assert state["quality_report"] is None
    assert state["last_critiqued_draft_hash"] is None
    assert state["candidate_readiness"] is None
    assert "working_draft" not in state
    assert state["tool_errors"] == []
    assert state["feedback_summary"] is None
    assert state["verification_status"] is None
    assert state["latest_checked_revision"] is None
