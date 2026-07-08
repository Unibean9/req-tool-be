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
from app.models.artifact import Artifact, ArtifactStatus, ArtifactType
from app.models.organization import OrgMember
from app.services.agent_event_service import AgentEventService, _ui_status
from tests.conftest import BASE, TestSessionFactory
from tests.factories import _project
from tests.helpers import create_org, create_project, make_auth_headers


def _user_id(headers: dict) -> uuid.UUID:
    token = headers["Authorization"].removeprefix("Bearer ")
    return uuid.UUID(decode_token(token)["sub"])


async def _project_with_owner(client):
    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    return headers, org, project


@pytest.mark.asyncio
async def test_agent_event_snapshot_contains_safe_session_messages_and_tool_calls(client, db_session):
    project_id = await _project(client)
    owner_id = uuid.uuid4()
    session = AgentSession(
        project_id=project_id,
        artifact_type="goal",
        workflow_area="analysis",
        graph_checkpoint={"secret": "must-not-leak"},
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
            content="Artifact proposal needs approval.",
        )
    )
    run = AgentRun(session_id=session.id, analysis_result={"confidence": 0.9})
    db_session.add(run)
    await db_session.flush()
    db_session.add(
        AgentToolCall(
            run_id=run.id,
            tool_name="create_artifact",
            input_snapshot={
                "artifact_type": "goal",
                "title": "Goal",
                "synthesis_metadata": {
                    "contract_version": "2026-06-23",
                    "evidence_refs": ["agent_run:1"],
                },
                "candidate_readiness": {"state": "sufficient"},
            },
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
    assert snapshot["messages"][0]["content"] == "Artifact proposal needs approval."
    assert snapshot["tool_calls"][0]["tool_name"] == "create_artifact"
    public_snapshot = snapshot["tool_calls"][0]["input_snapshot"]
    assert "contract_version" not in public_snapshot["synthesis_metadata"]
    assert public_snapshot["synthesis_metadata"]["evidence_refs"] == ["agent_run:1"]
    assert public_snapshot["candidate_readiness"] == {"state": "sufficient"}


@pytest.mark.asyncio
async def test_agent_event_snapshot_refreshes_status_committed_by_graph_session(client, db_session):
    project_id = await _project(client)
    owner_id = uuid.uuid4()
    session = AgentSession(
        project_id=project_id,
        artifact_type="goal",
        workflow_area="analysis",
        graph_checkpoint={},
        status=AgentSessionStatus.ACTIVE,
        created_by_id=owner_id,
    )
    db_session.add(session)
    await db_session.commit()

    service = AgentEventService(db_session)
    first = await service.build_snapshot(project_id=project_id, session_id=session.id, user_id=owner_id)

    async with TestSessionFactory() as graph_db:
        row = await graph_db.get(AgentSession, session.id)
        row.status = AgentSessionStatus.ACTIVE
        row.interrupt_type = AgentSessionInterruptType.STREAM_RESPONSE
        await graph_db.commit()

    second = await service.build_snapshot(project_id=project_id, session_id=session.id, user_id=owner_id)

    assert first["session"]["ui_status"] == "processing"
    assert second["session"]["interrupt_type"] == AgentSessionInterruptType.STREAM_RESPONSE
    assert second["session"]["ui_status"] == "waiting_input"


@pytest.mark.asyncio
async def test_agent_event_snapshot_hides_internal_audit_tool_calls(client, db_session):
    project_id = await _project(client)
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
                input_snapshot={"title": "Draft", "body": "Content"},
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
    assert _ui_status(AgentSessionStatus.TURN_FAILED, None) == "error"
    assert _ui_status(AgentSessionStatus.COMPLETED, None) == "idle"


async def _snapshot_for(db_session, project_id, *, status, interrupt_type=None):
    owner_id = uuid.uuid4()
    session = AgentSession(
        project_id=project_id,
        artifact_type="goal",
        workflow_area="analysis",
        graph_checkpoint={"secret": "must-not-leak"},
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
    project_id = await _project(client)
    snapshot = await _snapshot_for(db_session, project_id, status=AgentSessionStatus.ACTIVE)
    assert snapshot["session"]["ui_status"] == "processing"


@pytest.mark.asyncio
async def test_ui_status_stream_response(client, db_session):
    project_id = await _project(client)
    snapshot = await _snapshot_for(
        db_session, project_id,
        status=AgentSessionStatus.ACTIVE,
        interrupt_type=AgentSessionInterruptType.STREAM_RESPONSE,
    )
    assert snapshot["session"]["ui_status"] == "waiting_input"


@pytest.mark.asyncio
async def test_ui_status_waiting_approval(client, db_session):
    project_id = await _project(client)
    snapshot = await _snapshot_for(
        db_session, project_id,
        status=AgentSessionStatus.WAITING_FOR_HUMAN,
        interrupt_type=AgentSessionInterruptType.PROPOSE_ARTIFACTS,
    )
    assert snapshot["session"]["ui_status"] == "waiting_approval"


@pytest.mark.asyncio
async def test_ui_status_waiting_input(client, db_session):
    project_id = await _project(client)
    snapshot = await _snapshot_for(
        db_session, project_id,
        status=AgentSessionStatus.WAITING_FOR_HUMAN,
        interrupt_type=AgentSessionInterruptType.ASK_HUMAN,
    )
    assert snapshot["session"]["ui_status"] == "waiting_input"


@pytest.mark.asyncio
async def test_ui_status_failed(client, db_session):
    project_id = await _project(client)
    snapshot = await _snapshot_for(db_session, project_id, status=AgentSessionStatus.FAILED)
    assert snapshot["session"]["ui_status"] == "error"


@pytest.mark.asyncio
async def test_ui_status_turn_failed(client, db_session):
    """TURN_FAILED maps to 'error' like FAILED, not the 'idle' fallthrough — otherwise a resumable
    failed turn would look like nothing happened even though the error message is in the transcript."""
    project_id = await _project(client)
    snapshot = await _snapshot_for(db_session, project_id, status=AgentSessionStatus.TURN_FAILED)
    assert snapshot["session"]["ui_status"] == "error"


@pytest.mark.asyncio
async def test_ui_status_completed(client, db_session):
    project_id = await _project(client)
    snapshot = await _snapshot_for(db_session, project_id, status=AgentSessionStatus.COMPLETED)
    assert snapshot["session"]["ui_status"] == "idle"


@pytest.mark.asyncio
async def test_snapshot_no_graph_checkpoint_after_ui_status_added(client, db_session):
    project_id = await _project(client)
    snapshot = await _snapshot_for(
        db_session, project_id,
        status=AgentSessionStatus.WAITING_FOR_HUMAN,
        interrupt_type=AgentSessionInterruptType.ASK_HUMAN,
    )
    assert "graph_checkpoint" not in snapshot["session"]


class _NeverDisconnectedRequest:
    """Stays connected so the stream loop must close on its own via the status check."""

    async def is_disconnected(self) -> bool:
        return False


@pytest.mark.asyncio
async def test_stream_closes_on_turn_failed(client, db_session):
    """The SSE stream must close on TURN_FAILED exactly as it does on FAILED — the turn is over and
    the client should reconnect only after sending its next message, not poll a dead turn forever."""
    project_id = await _project(client)
    owner_id = uuid.uuid4()
    session = AgentSession(
        project_id=project_id,
        artifact_type="goal",
        workflow_area="analysis",
        graph_checkpoint={},
        status=AgentSessionStatus.TURN_FAILED,
        created_by_id=owner_id,
    )
    db_session.add(session)
    await db_session.flush()

    frames = []
    async for frame in AgentEventService(db_session).stream_session_events(
        project_id=project_id,
        session_id=session.id,
        user_id=owner_id,
        request=_NeverDisconnectedRequest(),
        interval_seconds=0.01,
        heartbeat_seconds=1.0,
    ):
        frames.append(frame)

    assert any("event: stream_closed" in f and '"status":"turn_failed"' in f for f in frames)


# ---------------------------------------------------------------------------
# payload exposed in snapshot (S6, S7)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_snapshot_messages_include_payload(client, db_session):
    project_id = await _project(client)
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
            content="Goal la gi?",
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
    project_id = await _project(client)
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
            content="new",
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
    project_id = await _project(client)
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
    project_id = await _project(client)
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


# ---------------------------------------------------------------------------
# _document_for_session — container detection is registry-driven, not a
# hardcoded {"brd", "prd", "add"} set. event_storming must resolve the same
# way brd/prd/add already do.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_document_for_session_resolves_event_storming_via_artifact_type(client, db_session):
    project_id = await _project(client)
    session = AgentSession(
        project_id=project_id,
        artifact_type="event_storming",
        workflow_area="analysis",
        graph_checkpoint={},
    )
    db_session.add(session)
    await db_session.flush()

    document = await AgentEventService(db_session)._document_for_session(session, project_id)

    assert document is not None
    assert document.document_type == ArtifactType.EVENT_STORMING


@pytest.mark.asyncio
async def test_document_for_session_resolves_event_storming_via_focused_container(client, db_session):
    project_id = await _project(client)
    container = Artifact(
        project_id=project_id,
        type=ArtifactType.EVENT_STORMING,
        status=ArtifactStatus.DRAFT,
        title="Event Storming",
        extra_metadata={},
    )
    db_session.add(container)
    await db_session.flush()
    session = AgentSession(
        project_id=project_id,
        artifact_type="goal",
        workflow_area="analysis",
        graph_checkpoint={},
        focused_artifact_id=container.id,
    )
    db_session.add(session)
    await db_session.flush()

    document = await AgentEventService(db_session)._document_for_session(session, project_id)

    assert document is not None
    assert document.document_type == ArtifactType.EVENT_STORMING


@pytest.mark.asyncio
@pytest.mark.parametrize("container_type", ["brd", "prd", "add"])
async def test_document_for_session_regression_via_artifact_type(client, db_session, container_type):
    project_id = await _project(client)
    session = AgentSession(
        project_id=project_id,
        artifact_type=container_type,
        workflow_area="analysis",
        graph_checkpoint={},
    )
    db_session.add(session)
    await db_session.flush()

    document = await AgentEventService(db_session)._document_for_session(session, project_id)

    assert document is not None
    assert document.document_type == ArtifactType(container_type)


@pytest.mark.asyncio
@pytest.mark.parametrize("container_type", ["brd", "prd", "add"])
async def test_document_for_session_regression_via_focused_container(client, db_session, container_type):
    project_id = await _project(client)
    container = Artifact(
        project_id=project_id,
        type=ArtifactType(container_type),
        status=ArtifactStatus.DRAFT,
        title=container_type.upper(),
        extra_metadata={},
    )
    db_session.add(container)
    await db_session.flush()
    session = AgentSession(
        project_id=project_id,
        artifact_type="goal",
        workflow_area="analysis",
        graph_checkpoint={},
        focused_artifact_id=container.id,
    )
    db_session.add(session)
    await db_session.flush()

    document = await AgentEventService(db_session)._document_for_session(session, project_id)

    assert document is not None
    assert document.document_type == ArtifactType(container_type)
