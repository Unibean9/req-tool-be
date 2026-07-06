"""BRD-finalize executive-summary synthesis, resume-safety, and export."""

import pytest
from sqlalchemy import select

from app.graphs import agent_tools
from app.graphs.agent_tools import (
    _apply_executive_summary_resume,
    _persist_executive_summary_draft,
)
from app.models.artifact import Artifact, ArtifactStatus, ArtifactVersion, ChangeSource, VersionStatus
from app.models.project import Project
from app.services.export_service import render_brd
from tests.factories import _project, _session_factory


async def _accepted_item(db, project_id, item_type, body):
    art = Artifact(
        project_id=project_id, type=item_type, status=ArtifactStatus.ACCEPTED, title=item_type, extra_metadata={}
    )
    db.add(art)
    await db.flush()
    ver = ArtifactVersion(
        artifact_id=art.id,
        version_number=1,
        title=item_type,
        body=body,
        status=VersionStatus.ACCEPTED,
        change_source=ChangeSource.MANUAL,
        extra_metadata={},
    )
    db.add(ver)
    await db.flush()
    art.current_version_id = ver.id
    await db.flush()
    return art


@pytest.mark.asyncio
async def test_persist_synthesizes_and_stores_executive_summary(client, db_session):
    project_id = await _project(client)
    await _accepted_item(db_session, project_id, "vision_objectives", "Increase retention by 20%.")
    await _accepted_item(db_session, project_id, "problem_statement", "Onboarding is slow.")
    await _accepted_item(db_session, project_id, "scope_capabilities", "Dashboard and approval flow.")
    await db_session.commit()

    draft = await _persist_executive_summary_draft(db_session, project_id)
    await db_session.commit()

    assert draft and "Increase retention by 20%." in draft
    stored = (await db_session.execute(select(Project.executive_summary).where(Project.id == project_id))).scalar_one()
    assert stored == draft


@pytest.mark.asyncio
async def test_persist_returns_none_without_sources(client, db_session):
    project_id = await _project(client)
    draft = await _persist_executive_summary_draft(db_session, project_id)
    assert draft is None


@pytest.mark.asyncio
async def test_persist_is_resume_stable(client, db_session, monkeypatch):
    """Guard against the interrupt/resume re-execution hazard: synthesis runs once.

    A stub that returns different text on each call must never change the persisted
    value on a second (resume) call — the helper reuses the stored draft.
    """
    project_id = await _project(client)
    await _accepted_item(db_session, project_id, "vision_objectives", "V")
    await db_session.commit()

    calls = {"n": 0}

    def _unstable(_sources):
        calls["n"] += 1
        return f"draft-{calls['n']}"

    monkeypatch.setattr(agent_tools, "synthesize_executive_summary", _unstable)

    first = await _persist_executive_summary_draft(db_session, project_id)
    await db_session.commit()
    second = await _persist_executive_summary_draft(db_session, project_id)
    await db_session.commit()

    assert first == "draft-1"
    assert second == "draft-1"  # resume reuses the persisted draft, does not recompute
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_apply_resume_edit_and_reject(client, db_session):
    project_id = await _project(client)
    project = (await db_session.execute(select(Project).where(Project.id == project_id))).scalar_one()
    project.executive_summary = "auto draft"
    await db_session.commit()

    await _apply_executive_summary_resume(
        project_id, {"executive_summary_action": "edit", "executive_summary": "user text"}, _session_factory()
    )
    edited = (await db_session.execute(select(Project.executive_summary).where(Project.id == project_id))).scalar_one()
    assert edited == "user text"

    await _apply_executive_summary_resume(project_id, {"executive_summary_action": "reject"}, _session_factory())
    rejected = (
        await db_session.execute(select(Project.executive_summary).where(Project.id == project_id))
    ).scalar_one()
    assert rejected is None


@pytest.mark.asyncio
async def test_apply_resume_approve_keeps_draft(client, db_session):
    project_id = await _project(client)
    project = (await db_session.execute(select(Project).where(Project.id == project_id))).scalar_one()
    project.executive_summary = "auto draft"
    await db_session.commit()

    await _apply_executive_summary_resume(project_id, {}, _session_factory())
    kept = (await db_session.execute(select(Project.executive_summary).where(Project.id == project_id))).scalar_one()
    assert kept == "auto draft"


@pytest.mark.asyncio
async def test_brd_export_renders_executive_summary_first_no_risks(client, db_session):
    project_id = await _project(client)
    project = (await db_session.execute(select(Project).where(Project.id == project_id))).scalar_one()
    project.executive_summary = "This project lifts retention."
    await db_session.commit()

    text = await render_brd(project_id, db_session)

    assert "## Executive Summary" in text
    assert "This project lifts retention." in text
    assert text.index("## Executive Summary") < text.index("## Vision and Objectives")
    assert "## Risks and Issues" not in text
