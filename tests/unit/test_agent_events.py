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
from app.services.agent_event_service import AgentEventService, _ui_status
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
async def test_agent_event_snapshot_hides_internal_audit_tool_calls(client, db_session):
    project_id = await _project_id(client)
    owner_id = uuid.uuid4()
    session = AgentSession(
        project_id=project_id,
        artifact_type="goal",
        workflow_area="analysis",
        graph_checkpoint={},
        status=AgentSessionStatus.WAITING_FOR_HUMAN,
        interrupt_type=AgentSessionInterruptType.PROPOSE_ARTIFACTS,
        created_by_id=owner_id,
    )
    db_session.add(session)
    await db_session.flush()
    run = AgentRun(session_id=session.id, analysis_result={})
    db_session.add(run)
    await db_session.flush()
    db_session.add_all(
        [
            AgentToolCall(
                run_id=run.id,
                tool_name="write_draft:section-1",
                input_snapshot={"title": "Bản nháp", "body": "Nội dung"},
                status=AgentToolCallStatus.PROPOSED,
            ),
            AgentToolCall(
                run_id=run.id,
                tool_name="recommend_next_workflow",
                input_snapshot={"recommended_next_workflow": "prd"},
                status=AgentToolCallStatus.PROPOSED,
            ),
        ]
    )
    await db_session.flush()

    snapshot = await AgentEventService(db_session).build_snapshot(
        project_id=project_id, session_id=session.id, user_id=owner_id
    )

    assert [tc["tool_name"] for tc in snapshot["tool_calls"]] == ["write_draft:section-1"]


# ---------------------------------------------------------------------------
# ui_status derived in the snapshot (S1)
# ---------------------------------------------------------------------------

def test_ui_status_function_unit():
    assert _ui_status(AgentSessionStatus.ACTIVE, None) == "processing"
    assert _ui_status(AgentSessionStatus.ACTIVE, AgentSessionInterruptType.STREAM_RESPONSE) == "waiting_input"
    assert _ui_status(AgentSessionStatus.WAITING_FOR_HUMAN, AgentSessionInterruptType.PROPOSE_ARTIFACTS) == "waiting_approval"
    assert _ui_status(AgentSessionStatus.WAITING_FOR_HUMAN, AgentSessionInterruptType.ASK_HUMAN) == "waiting_input"
    assert _ui_status(AgentSessionStatus.WAITING_FOR_HUMAN, None) == "waiting_input"
    assert _ui_status(AgentSessionStatus.FAILED, None) == "error"
    assert _ui_status(AgentSessionStatus.COMPLETED, None) == "idle"


async def _snapshot_for(db_session, project_id, *, status, interrupt_type=None):
    owner_id = uuid.uuid4()
    session = AgentSession(
        project_id=project_id,
        artifact_type="goal",
        workflow_area="analysis",
        graph_checkpoint={"secret": "khong-duoc-leak"},
        status=status,
        interrupt_type=interrupt_type,
        created_by_id=owner_id,
    )
    db_session.add(session)
    await db_session.flush()
    return await AgentEventService(db_session).build_snapshot(
        project_id=project_id, session_id=session.id, user_id=owner_id
    )


@pytest.mark.asyncio
async def test_ui_status_active(client, db_session):
    project_id = await _project_id(client)
    snapshot = await _snapshot_for(db_session, project_id, status=AgentSessionStatus.ACTIVE)
    assert snapshot["session"]["ui_status"] == "processing"


@pytest.mark.asyncio
async def test_ui_status_stream_response(client, db_session):
    project_id = await _project_id(client)
    snapshot = await _snapshot_for(
        db_session, project_id,
        status=AgentSessionStatus.ACTIVE,
        interrupt_type=AgentSessionInterruptType.STREAM_RESPONSE,
    )
    assert snapshot["session"]["ui_status"] == "waiting_input"


@pytest.mark.asyncio
async def test_ui_status_waiting_approval(client, db_session):
    project_id = await _project_id(client)
    snapshot = await _snapshot_for(
        db_session, project_id,
        status=AgentSessionStatus.WAITING_FOR_HUMAN,
        interrupt_type=AgentSessionInterruptType.PROPOSE_ARTIFACTS,
    )
    assert snapshot["session"]["ui_status"] == "waiting_approval"


@pytest.mark.asyncio
async def test_ui_status_waiting_input(client, db_session):
    project_id = await _project_id(client)
    snapshot = await _snapshot_for(
        db_session, project_id,
        status=AgentSessionStatus.WAITING_FOR_HUMAN,
        interrupt_type=AgentSessionInterruptType.ASK_HUMAN,
    )
    assert snapshot["session"]["ui_status"] == "waiting_input"


@pytest.mark.asyncio
async def test_ui_status_failed(client, db_session):
    project_id = await _project_id(client)
    snapshot = await _snapshot_for(db_session, project_id, status=AgentSessionStatus.FAILED)
    assert snapshot["session"]["ui_status"] == "error"


@pytest.mark.asyncio
async def test_ui_status_completed(client, db_session):
    project_id = await _project_id(client)
    snapshot = await _snapshot_for(db_session, project_id, status=AgentSessionStatus.COMPLETED)
    assert snapshot["session"]["ui_status"] == "idle"


@pytest.mark.asyncio
async def test_snapshot_no_graph_checkpoint_after_ui_status_added(client, db_session):
    project_id = await _project_id(client)
    snapshot = await _snapshot_for(
        db_session, project_id,
        status=AgentSessionStatus.WAITING_FOR_HUMAN,
        interrupt_type=AgentSessionInterruptType.ASK_HUMAN,
    )
    assert "graph_checkpoint" not in snapshot["session"]


# ---------------------------------------------------------------------------
# Phase 5 — payload exposed in snapshot (S6, S7)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_snapshot_messages_include_payload(client, db_session):
    project_id = await _project_id(client)
    owner_id = uuid.uuid4()
    session = AgentSession(
        project_id=project_id,
        artifact_type="goal",
        workflow_area="analysis",
        graph_checkpoint={"secret": "leak"},
        status=AgentSessionStatus.WAITING_FOR_HUMAN,
        interrupt_type=AgentSessionInterruptType.ASK_HUMAN,
        created_by_id=owner_id,
    )
    db_session.add(session)
    await db_session.flush()
    db_session.add(
        AgentMessage(
            session_id=session.id,
            role=AgentMessageRole.AGENT,
            content="Mục tiêu là gì?",
            payload={"kind": "question", "locale": "vi", "options": [], "blocks": []},
        )
    )
    await db_session.flush()

    snapshot = await AgentEventService(db_session).build_snapshot(
        project_id=project_id, session_id=session.id, user_id=owner_id
    )

    assert snapshot["messages"][0]["payload"]["kind"] == "question"


@pytest.mark.asyncio
async def test_snapshot_mixed_legacy_and_new_messages(client, db_session):
    project_id = await _project_id(client)
    owner_id = uuid.uuid4()
    session = AgentSession(
        project_id=project_id,
        artifact_type="goal",
        workflow_area="analysis",
        graph_checkpoint={},
        status=AgentSessionStatus.WAITING_FOR_HUMAN,
        created_by_id=owner_id,
    )
    db_session.add(session)
    await db_session.flush()
    db_session.add_all([
        AgentMessage(session_id=session.id, role=AgentMessageRole.USER, content="legacy", payload=None),
        AgentMessage(
            session_id=session.id,
            role=AgentMessageRole.AGENT,
            content="mới",
            payload={"kind": "question", "locale": "vi"},
        ),
    ])
    await db_session.flush()

    snapshot = await AgentEventService(db_session).build_snapshot(
        project_id=project_id, session_id=session.id, user_id=owner_id
    )

    assert snapshot["messages"][0]["payload"] is None
    assert snapshot["messages"][1]["payload"]["kind"] == "question"


@pytest.mark.asyncio
async def test_snapshot_with_payload_still_no_graph_checkpoint(client, db_session):
    project_id = await _project_id(client)
    owner_id = uuid.uuid4()
    session = AgentSession(
        project_id=project_id,
        artifact_type="goal",
        workflow_area="analysis",
        graph_checkpoint={"secret": "leak"},
        status=AgentSessionStatus.WAITING_FOR_HUMAN,
        created_by_id=owner_id,
    )
    db_session.add(session)
    await db_session.flush()
    db_session.add(
        AgentMessage(
            session_id=session.id,
            role=AgentMessageRole.AGENT,
            content="x",
            payload={"kind": "question", "locale": "vi"},
        )
    )
    await db_session.flush()

    snapshot = await AgentEventService(db_session).build_snapshot(
        project_id=project_id, session_id=session.id, user_id=owner_id
    )

    assert "graph_checkpoint" not in snapshot["session"]


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
