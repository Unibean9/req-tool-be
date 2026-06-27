import uuid

import pytest
from sqlalchemy import select

from app.models.artifact import WorkflowRun, WorkflowRunStatus, WorkflowStep, WorkflowStepStatus
from app.services.workflow_service import WorkflowService
from tests.conftest import BASE
from tests.helpers import create_org, create_project, make_auth_headers


@pytest.mark.asyncio
async def test_create_workflow_run_auto_seeds_five_pending_steps(client, db_session):
    headers, project = await _project_context(client)

    resp = await client.post(
        f"{BASE}/projects/{project['id']}/workflow-runs",
        json={"name": "Requirements analysis flow"},
        headers=headers,
    )

    assert resp.status_code == 201, resp.text
    run = resp.json()["data"]
    steps = (
        await db_session.execute(select(WorkflowStep).where(WorkflowStep.run_id == uuid.UUID(run["id"])))
    ).scalars().all()
    assert len(steps) == 5
    assert {step.status.value for step in steps} == {"pending"}


@pytest.mark.asyncio
async def test_workflow_step_seed_has_correct_keys_and_phase_mapping(client):
    headers, project = await _project_context(client)

    resp = await client.post(
        f"{BASE}/projects/{project['id']}/workflow-runs",
        json={"name": "Requirements analysis flow"},
        headers=headers,
    )

    assert resp.status_code == 201, resp.text
    steps = {step["step_key"]: step["phase"] for step in resp.json()["data"]["steps"]}
    assert steps == {
        "intent_vision": "brd",
        "capability_map": "brd",
        "domain_model": "prd",
        "requirements_spec": "prd",
        "realization_backlog": "delivery",
    }


@pytest.mark.asyncio
async def test_update_step_status_internal_persists_status(client, db_session):
    headers, project = await _project_context(client)
    create_resp = await client.post(
        f"{BASE}/projects/{project['id']}/workflow-runs",
        json={"name": "Requirements analysis flow"},
        headers=headers,
    )
    step_id = create_resp.json()["data"]["steps"][0]["id"]

    updated = await WorkflowService(db_session).update_step_status(
        project_id=uuid.UUID(project["id"]),
        step_id=uuid.UUID(step_id),
        status=WorkflowStepStatus.IN_PROGRESS,
    )

    assert updated.status == WorkflowStepStatus.IN_PROGRESS
    stored = await db_session.get(WorkflowStep, uuid.UUID(step_id))
    assert stored.status == WorkflowStepStatus.IN_PROGRESS


@pytest.mark.asyncio
async def test_create_workflow_run_rejects_duplicate_active_run(client):
    headers, project = await _project_context(client)
    first = await client.post(
        f"{BASE}/projects/{project['id']}/workflow-runs",
        json={"name": "First flow"},
        headers=headers,
    )
    second = await client.post(
        f"{BASE}/projects/{project['id']}/workflow-runs",
        json={"name": "Second flow"},
        headers=headers,
    )

    assert first.status_code == 201, first.text
    assert second.status_code == 409
    assert "active" in second.json()["detail"]


@pytest.mark.asyncio
async def test_get_current_workflow_run_returns_active_or_404(client, db_session):
    headers, project = await _project_context(client)
    missing = await client.get(f"{BASE}/projects/{project['id']}/workflow-runs/current", headers=headers)
    create_resp = await client.post(
        f"{BASE}/projects/{project['id']}/workflow-runs",
        json={"name": "Current flow"},
        headers=headers,
    )
    current = await client.get(f"{BASE}/projects/{project['id']}/workflow-runs/current", headers=headers)

    assert missing.status_code == 404
    assert current.status_code == 200, current.text
    assert current.json()["data"]["id"] == create_resp.json()["data"]["id"]
    assert await db_session.scalar(select(WorkflowRun.status).where(WorkflowRun.id == uuid.UUID(current.json()["data"]["id"]))) == WorkflowRunStatus.ACTIVE


@pytest.mark.asyncio
async def test_get_workflow_steps_returns_current_run_steps(client):
    headers, project = await _project_context(client)
    await client.post(
        f"{BASE}/projects/{project['id']}/workflow-runs",
        json={"name": "Current flow"},
        headers=headers,
    )

    resp = await client.get(f"{BASE}/projects/{project['id']}/workflow-steps", headers=headers)

    assert resp.status_code == 200, resp.text
    steps = resp.json()["data"]
    assert len(steps) == 5
    assert {"step_key", "phase", "status"}.issubset(steps[0])


@pytest.mark.asyncio
async def test_get_workflow_progress_returns_status_summary(client, db_session):
    headers, project = await _project_context(client)
    create_resp = await client.post(
        f"{BASE}/projects/{project['id']}/workflow-runs",
        json={"name": "Current flow"},
        headers=headers,
    )
    await WorkflowService(db_session).update_step_status(
        project_id=uuid.UUID(project["id"]),
        step_id=uuid.UUID(create_resp.json()["data"]["steps"][0]["id"]),
        status=WorkflowStepStatus.IN_PROGRESS,
    )

    resp = await client.get(f"{BASE}/projects/{project['id']}/workflow-progress", headers=headers)

    assert resp.status_code == 200, resp.text
    progress = resp.json()["data"]
    assert progress["run_id"] == create_resp.json()["data"]["id"]
    assert progress["step_counts"]["pending"] == 4
    assert progress["step_counts"]["in_progress"] == 1


@pytest.mark.asyncio
async def test_workflow_endpoints_reject_non_project_member(client):
    owner_headers, project = await _project_context(client)
    outsider_headers = await make_auth_headers(client)
    await client.post(
        f"{BASE}/projects/{project['id']}/workflow-runs",
        json={"name": "Current flow"},
        headers=owner_headers,
    )

    create_resp = await client.post(
        f"{BASE}/projects/{project['id']}/workflow-runs",
        json={"name": "No permission"},
        headers=outsider_headers,
    )
    current_resp = await client.get(f"{BASE}/projects/{project['id']}/workflow-runs/current", headers=outsider_headers)
    steps_resp = await client.get(f"{BASE}/projects/{project['id']}/workflow-steps", headers=outsider_headers)
    progress_resp = await client.get(f"{BASE}/projects/{project['id']}/workflow-progress", headers=outsider_headers)

    assert create_resp.status_code == 403
    assert current_resp.status_code == 403
    assert steps_resp.status_code == 403
    assert progress_resp.status_code == 403


async def _project_context(client):
    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    return headers, project
