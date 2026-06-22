import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, StatementError

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
from tests.helpers import create_org, create_project, make_auth_headers


async def create_project_id(client) -> uuid.UUID:
    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    return uuid.UUID(project["id"])


@pytest.mark.asyncio
async def test_agent_session_can_be_saved_and_loaded(client, db_session):
    project_id = await create_project_id(client)
    session = AgentSession(
        project_id=project_id,
        artifact_type="goal",
        workflow_area="analysis",
        status=AgentSessionStatus.ACTIVE,
        interrupt_type=AgentSessionInterruptType.ASK_HUMAN,
        missing_context=["intent", "problem"],
        graph_checkpoint={"checkpoint": "value"},
        focus_section="vision_objectives",
    )
    db_session.add(session)
    await db_session.flush()

    loaded = await db_session.get(AgentSession, session.id)

    assert loaded is not None
    assert loaded.artifact_type == "goal"
    assert loaded.workflow_area == "analysis"
    assert loaded.status == AgentSessionStatus.ACTIVE
    assert loaded.interrupt_type == AgentSessionInterruptType.ASK_HUMAN
    assert loaded.missing_context == ["intent", "problem"]
    assert loaded.graph_checkpoint == {"checkpoint": "value"}
    assert loaded.focus_section == "vision_objectives"


@pytest.mark.asyncio
async def test_agent_message_and_tool_call_cascade_from_session(client, db_session):
    project_id = await create_project_id(client)
    session = AgentSession(
        project_id=project_id,
        artifact_type="intent",
        workflow_area="analysis",
        status=AgentSessionStatus.ACTIVE,
    )
    message = AgentMessage(role=AgentMessageRole.AGENT, content="Cần làm rõ phạm vi.")
    run = AgentRun(analysis_result={"confidence": "high", "next_action": "propose"})
    tool_call = AgentToolCall(
        tool_name="create_artifact",
        input_snapshot={"artifact_type": "intent", "title": "Tầm nhìn"},
        status=AgentToolCallStatus.PROPOSED,
    )
    run.tool_calls.append(tool_call)
    session.messages.append(message)
    session.runs.append(run)
    db_session.add(session)
    await db_session.flush()

    session_id = session.id
    message_id = message.id
    run_id = run.id
    tool_call_id = tool_call.id
    await db_session.delete(session)
    await db_session.flush()

    assert await db_session.get(AgentSession, session_id) is None
    assert await db_session.get(AgentMessage, message_id) is None
    assert await db_session.get(AgentRun, run_id) is None
    assert await db_session.get(AgentToolCall, tool_call_id) is None


@pytest.mark.asyncio
async def test_agent_session_status_rejects_unknown_value(client, db_session):
    project_id = await create_project_id(client)
    session = AgentSession(
        project_id=project_id,
        artifact_type="risk",
        workflow_area="analysis",
        status="paused",
    )
    db_session.add(session)

    with pytest.raises(StatementError):
        await db_session.flush()


@pytest.mark.asyncio
async def test_one_active_agent_session_per_project_artifact_type_and_user(client, db_session):
    project_id = await create_project_id(client)
    owner_id = uuid.uuid4()
    db_session.add(
        AgentSession(
            project_id=project_id,
            artifact_type="problem",
            workflow_area="analysis",
            status=AgentSessionStatus.ACTIVE,
            created_by_id=owner_id,
        )
    )
    await db_session.flush()

    db_session.add(
        AgentSession(
            project_id=project_id,
            artifact_type="problem",
            workflow_area="analysis",
            status=AgentSessionStatus.WAITING_FOR_HUMAN,
            created_by_id=owner_id,
        )
    )

    with pytest.raises(IntegrityError):
        await db_session.flush()


@pytest.mark.asyncio
async def test_active_agent_sessions_with_same_artifact_type_are_allowed_for_different_users(client, db_session):
    project_id = await create_project_id(client)
    db_session.add(
        AgentSession(
            project_id=project_id,
            artifact_type="problem",
            workflow_area="analysis",
            status=AgentSessionStatus.ACTIVE,
            created_by_id=uuid.uuid4(),
        )
    )
    db_session.add(
        AgentSession(
            project_id=project_id,
            artifact_type="problem",
            workflow_area="analysis",
            status=AgentSessionStatus.WAITING_FOR_HUMAN,
            created_by_id=uuid.uuid4(),
        )
    )

    await db_session.flush()

    result = await db_session.execute(
        select(AgentSession).where(
            AgentSession.project_id == project_id,
            AgentSession.artifact_type == "problem",
        )
    )
    assert len(result.scalars().all()) == 2


@pytest.mark.asyncio
async def test_completed_agent_session_does_not_block_new_session(client, db_session):
    project_id = await create_project_id(client)
    db_session.add(
        AgentSession(
            project_id=project_id,
            artifact_type="capability",
            workflow_area="planning",
            status=AgentSessionStatus.COMPLETED,
        )
    )
    db_session.add(
        AgentSession(
            project_id=project_id,
            artifact_type="capability",
            workflow_area="planning",
            status=AgentSessionStatus.ACTIVE,
        )
    )
    await db_session.flush()

    result = await db_session.execute(
        select(AgentSession).where(
            AgentSession.project_id == project_id,
            AgentSession.artifact_type == "capability",
        )
    )

    assert len(result.scalars().all()) == 2
