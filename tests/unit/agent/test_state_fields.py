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


def test_workflow_state_annotations_reflect_current_schema():
    # Merges the separate has_key_facts_field / has_section_coverage_field /
    # no_longer_has_slot_coverage_field checks: all are static TypedDict annotation membership
    # assertions on the same WorkflowState schema.
    present_fields = [
        "key_facts",
        "section_coverage",
        "section_coverage_stall_count",
        "focused_artifact_id",
        "turn_context_artifacts",
        "lifecycle_reports",
        "artifact_history",
        "candidate_readiness",
        "tool_errors",
        "feedback_summary",
        "verification_status",
        "latest_checked_revision",
    ]
    for field in present_fields:
        assert field in WorkflowState.__annotations__

    removed_fields = [
        "analysis_frame",
        "working_draft",
        "section_assessment",
        "coverage_ratio",
        "sections_body",
        "focus_section",
        "slot_coverage",
        "last_asked_slot",
        "coverage_stall_count",
    ]
    for field in removed_fields:
        assert field not in WorkflowState.__annotations__


def test_initial_workflow_state_seeds_default_list_fields():
    state = build_initial_workflow_state(
        artifact_type="brd",
        workflow_area="analysis",
        step_key=None,
    )

    assert state["key_facts"] == []
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
