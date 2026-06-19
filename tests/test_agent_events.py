import uuid

import pytest

from app.core.security import decode_token
from app.models.agent import (
    AgentMessage,
    AgentMessageRole,
    AgentRun,
    AgentSession,
    AgentSessionInterruptType,
    AgentSessionStatus,
    AgentToolCall,
    AgentToolCallStatus,
)
from app.models.organization import OrgMember
from app.services.agent_event_service import AgentEventService
from tests.conftest import BASE
from tests.helpers import create_org, create_project, make_auth_headers


def _user_id(headers: dict) -> uuid.UUID:
    token = headers["Authorization"].removeprefix("Bearer ")
    return uuid.UUID(decode_token(token)["sub"])


async def _project_id(client):
    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    return uuid.UUID(project["id"])


async def _project_with_owner(client):
    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    return headers, org, project


@pytest.mark.asyncio
async def test_agent_event_snapshot_contains_safe_session_messages_and_tool_calls(client, db_session):
    project_id = await _project_id(client)
    owner_id = uuid.uuid4()
    session = AgentSession(
        project_id=project_id,
        artifact_type="goal",
        workflow_area="analysis",
        graph_checkpoint={"secret": "khong-duoc-leak"},
        status=AgentSessionStatus.WAITING_FOR_HUMAN,
        interrupt_type=AgentSessionInterruptType.PROPOSE_ARTIFACTS,
        created_by_id=owner_id,
    )
    db_session.add(session)
    await db_session.flush()

    db_session.add(
        AgentMessage(
            session_id=session.id,
            role=AgentMessageRole.AGENT,
            content="Cần duyệt artifact đề xuất.",
        )
    )
    run = AgentRun(session_id=session.id, analysis_result={"confidence": 0.9})
    db_session.add(run)
    await db_session.flush()
    db_session.add(
        AgentToolCall(
            run_id=run.id,
            tool_name="create_artifact",
            input_snapshot={"artifact_type": "goal", "title": "Mục tiêu"},
            status=AgentToolCallStatus.PROPOSED,
        )
    )
    await db_session.flush()

    snapshot = await AgentEventService(db_session).build_snapshot(
        project_id=project_id,
        session_id=session.id,
        user_id=owner_id,
    )

    assert snapshot["type"] == "snapshot"
    assert snapshot["session"]["id"] == session.id
    assert snapshot["session"]["created_by_id"] == owner_id
    assert "graph_checkpoint" not in snapshot["session"]
    assert snapshot["messages"][0]["content"] == "Cần duyệt artifact đề xuất."
    assert snapshot["tool_calls"][0]["tool_name"] == "create_artifact"


@pytest.mark.asyncio
async def test_agent_event_snapshot_rejects_non_owner(client, db_session):
    project_id = await _project_id(client)
    session = AgentSession(
        project_id=project_id,
        artifact_type="goal",
        workflow_area="analysis",
        graph_checkpoint={},
        status=AgentSessionStatus.WAITING_FOR_HUMAN,
        created_by_id=uuid.uuid4(),
    )
    db_session.add(session)
    await db_session.flush()

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        await AgentEventService(db_session).build_snapshot(
            project_id=project_id,
            session_id=session.id,
            user_id=uuid.uuid4(),
        )

    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_agent_events_route_streams_initial_snapshot_for_owner(client, db_session):
    headers, _, project = await _project_with_owner(client)
    owner_id = _user_id(headers)
    session = AgentSession(
        project_id=uuid.UUID(project["id"]),
        artifact_type="goal",
        workflow_area="analysis",
        graph_checkpoint={"hidden": True},
        status=AgentSessionStatus.COMPLETED,
        created_by_id=owner_id,
    )
    db_session.add(session)
    await db_session.flush()

    async with client.stream(
        "GET",
        f"{BASE}/projects/{project['id']}/agent-sessions/{session.id}/events",
        headers=headers,
    ) as resp:
        body = await resp.aread()

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    text = body.decode()
    assert "event: snapshot" in text
    assert "graph_checkpoint" not in text


@pytest.mark.asyncio
async def test_agent_events_route_rejects_project_member_who_is_not_session_owner(client, db_session):
    owner_headers, org, project = await _project_with_owner(client)
    owner_id = _user_id(owner_headers)
    member_headers = await make_auth_headers(client)
    member_id = _user_id(member_headers)
    db_session.add(OrgMember(org_id=uuid.UUID(org["id"]), user_id=member_id, role="member"))

    session = AgentSession(
        project_id=uuid.UUID(project["id"]),
        artifact_type="goal",
        workflow_area="analysis",
        graph_checkpoint={},
        status=AgentSessionStatus.WAITING_FOR_HUMAN,
        created_by_id=owner_id,
    )
    db_session.add(session)
    await db_session.flush()

    resp = await client.get(
        f"{BASE}/projects/{project['id']}/agent-sessions/{session.id}/events",
        headers=member_headers,
    )

    assert resp.status_code == 404
