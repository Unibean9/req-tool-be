"""Foundation: verify rubric move + state/config extension."""


def test_rubric_importable_from_app():
    from app.graphs.rubric import RUBRIC_CRITERIA, render_criteria_block

    assert len(RUBRIC_CRITERIA) == 8
    assert isinstance(render_criteria_block(), str)


def test_rubric_reexport_from_tests():
    from tests.eval.rubric import RUBRIC_CRITERIA

    assert RUBRIC_CRITERIA is not None


def test_state_has_critique_fields():
    from app.graphs.state import WorkflowState

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


def test_settings_defaults():
    from app.config import settings

    assert settings.max_critique_rounds == 2
    assert settings.critique_score_threshold == 0.7
