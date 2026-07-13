"""Executive-summary synthesis + project-schema exposure."""

from app.graphs.agent_tools import synthesize_executive_summary
from app.schemas.project import ProjectResponse


def test_synthesize_includes_all_present_sources():
    summary = synthesize_executive_summary(
        {
            "vision_objectives": "Increase retention by 20%.",
            "problem_statement": "Onboarding is slow.",
            "scope_capabilities": "Dashboard and approval flow.",
        }
    )
    assert "Increase retention by 20%." in summary
    assert "Onboarding is slow." in summary
    assert "Dashboard and approval flow." in summary


def test_synthesize_empty_when_no_sources():
    assert synthesize_executive_summary({}) == ""
    assert synthesize_executive_summary({"vision_objectives": "   "}) == ""


def test_project_response_exposes_executive_summary():
    import uuid
    from datetime import datetime

    base = {
        "id": uuid.uuid4(),
        "org_id": uuid.uuid4(),
        "name": "P",
        "slug": "p",
        "description": None,
        "created_at": datetime(2026, 7, 4),
    }
    assert ProjectResponse.model_validate(base).executive_summary is None
    assert ProjectResponse.model_validate({**base, "executive_summary": "S"}).executive_summary == "S"
