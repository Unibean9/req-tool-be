"""Foundation: verify rubric move + state/config extension."""


def test_rubric_state_and_settings_foundation():
    from app.config import settings
    from app.graphs.rubric import RUBRIC_CRITERIA, render_criteria_block
    from app.graphs.state import WorkflowState
    from tests.eval.rubric import RUBRIC_CRITERIA as reexported_rubric_criteria

    # 8 base (29148 + INVEST/SMART) + 3 business dimensions.
    assert len(RUBRIC_CRITERIA) == 11
    assert isinstance(render_criteria_block(), str)
    assert reexported_rubric_criteria is not None

    state: WorkflowState = {
        "artifact_type": "story",
        "workflow_area": "x",
        "step_key": None,
        "messages": [],
        "analysis_result": None,
        "pending_tool_call_ids": [],
        "last_agent_run_id": None,
        "turn_count": 0,
        "missing_context": [],
        "user_confirmed": None,
        "critique_rounds": 0,
        "quality_report": None,
    }
    assert state["critique_rounds"] == 0
    assert state["quality_report"] is None

    assert settings.max_critique_rounds == 2
    assert settings.critique_score_threshold == 0.7
