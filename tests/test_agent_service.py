import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cryptography.fernet import Fernet
from fastapi import HTTPException
from sqlalchemy import select

from app.config import settings
from app.core import crypto
from app.core.crypto import encrypt_token
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
from app.models.artifact import Artifact, ArtifactVersion
from app.models.llm_provider import LLMProviderConfig, ProviderType
from tests.helpers import create_org, create_project, make_auth_headers


def _mock_graph():
    g = MagicMock()
    g.ainvoke = AsyncMock(return_value={})
    return g


async def _setup(client):
    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    return uuid.UUID(project["id"])


# Use patch to suppress background tasks in all tests — avoids concurrent session access.
@pytest.fixture(autouse=True)
def _no_background_tasks():
    real_create_task = asyncio.create_task

    def _side_effect(coro, *args, **kwargs):
        qualname = getattr(coro, "__qualname__", "")
        if "AgentService" in qualname or "AsyncMockMixin" in qualname:
            coro.close()
            return MagicMock()
        return real_create_task(coro, *args, **kwargs)

    with patch("app.services.agent_service.asyncio.create_task") as mock_ct:
        mock_ct.side_effect = _side_effect
        yield mock_ct


def _make_service(db_session, graph=None, session_factory=None):
    from app.services.agent_service import AgentService

    if session_factory is None:
        from contextlib import asynccontextmanager
        @asynccontextmanager
        async def _sf():
            yield db_session
        session_factory = _sf

    return AgentService(db=db_session, graph=graph or _mock_graph(), session_factory=session_factory)


# ---------------------------------------------------------------------------
# create_session — predecessor checks
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_session_goal_without_predecessors_returns_missing_context(client, db_session):
    project_id = await _setup(client)
    svc = _make_service(db_session)

    result = await svc.create_session(project_id=project_id, artifact_type="goal")

    assert set(result["missing_context"]) == {"intent", "problem"}


@pytest.mark.asyncio
async def test_resolve_llm_client_passes_bedrock_secret_key(db_session, monkeypatch):
    original_key = settings.encryption_key
    original_previous = settings.encryption_key_previous
    monkeypatch.setattr(settings, "encryption_key", Fernet.generate_key().decode())
    monkeypatch.setattr(settings, "encryption_key_previous", "")
    crypto._get_fernet.cache_clear()

    try:
        captured = {}
        sentinel = object()

        def fake_create(**kwargs):
            captured.update(kwargs)
            return sentinel

        monkeypatch.setattr("app.services.llm_clients.LLMClientFactory.create", fake_create)
        config = LLMProviderConfig(
            user_id=uuid.uuid4(),
            provider_type=ProviderType.BEDROCK,
            name="Bedrock",
            encrypted_api_key=encrypt_token("AKIATEST"),
            encrypted_secret_key=encrypt_token("aws-secret"),
            region="us-east-1",
            model_name="amazon.nova-lite-v1:0",
        )
        db_session.add(config)
        await db_session.flush()

        client = await _make_service(db_session)._resolve_llm_client(config.id)

        assert client is sentinel
        assert captured["provider_type"] == ProviderType.BEDROCK
        assert captured["api_key"] == "AKIATEST"
        assert captured["secret_key"] == "aws-secret"
        assert captured["region"] == "us-east-1"
        assert captured["model"] == "amazon.nova-lite-v1:0"
    finally:
        monkeypatch.setattr(settings, "encryption_key", original_key)
        monkeypatch.setattr(settings, "encryption_key_previous", original_previous)
        crypto._get_fernet.cache_clear()


@pytest.mark.asyncio
async def test_create_session_goal_with_intent_missing_problem(client, db_session):
    project_id = await _setup(client)

    db_session.add(Artifact(project_id=project_id, type="intent", title="Intent", extra_metadata={}, status="draft"))
    await db_session.flush()

    svc = _make_service(db_session)
    result = await svc.create_session(project_id=project_id, artifact_type="goal")

    assert result["missing_context"] == ["problem"]


@pytest.mark.asyncio
async def test_create_session_intent_has_no_predecessors(client, db_session):
    project_id = await _setup(client)
    svc = _make_service(db_session)

    result = await svc.create_session(project_id=project_id, artifact_type="intent")

    assert result["missing_context"] == []


# ---------------------------------------------------------------------------
# create_session — duplicate → 409
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_session_duplicate_active_raises_409(client, db_session):
    project_id = await _setup(client)
    svc = _make_service(db_session)

    await svc.create_session(project_id=project_id, artifact_type="intent")

    with pytest.raises(HTTPException) as exc:
        await _make_service(db_session).create_session(project_id=project_id, artifact_type="intent")

    assert exc.value.status_code == 409
    assert "session_id" in exc.value.detail


# ---------------------------------------------------------------------------
# handle_user_message
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handle_user_message_ask_human_resumes_graph(client, db_session, _no_background_tasks):
    project_id = await _setup(client)
    graph = _mock_graph()
    svc = _make_service(db_session, graph)

    session = AgentSession(
        project_id=project_id, artifact_type="goal", workflow_area="analysis",
        graph_checkpoint={}, status=AgentSessionStatus.WAITING_FOR_HUMAN,
        interrupt_type=AgentSessionInterruptType.ASK_HUMAN,
    )
    db_session.add(session)
    await db_session.flush()

    await svc.handle_user_message(project_id=project_id, session_id=session.id, content="Thêm thông tin")

    # Background task was scheduled through _run_graph so resume failures update session status/messages.
    _no_background_tasks.assert_called_once()
    scheduled = _no_background_tasks.call_args.args[0]
    assert scheduled.cr_code.co_name == "_run_graph"


@pytest.mark.asyncio
async def test_handle_user_message_propose_artifacts_raises_400(client, db_session):
    project_id = await _setup(client)
    svc = _make_service(db_session)

    session = AgentSession(
        project_id=project_id, artifact_type="goal", workflow_area="analysis",
        graph_checkpoint={}, status=AgentSessionStatus.WAITING_FOR_HUMAN,
        interrupt_type=AgentSessionInterruptType.PROPOSE_ARTIFACTS,
    )
    db_session.add(session)
    await db_session.flush()

    with pytest.raises(HTTPException) as exc:
        await svc.handle_user_message(project_id=project_id, session_id=session.id, content="ok")

    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_run_graph_failure_marks_session_failed_and_saves_agent_message(client, db_session):
    project_id = await _setup(client)
    graph = _mock_graph()
    graph.ainvoke = AsyncMock(side_effect=RuntimeError("provider rejected request"))
    svc = _make_service(db_session, graph)

    session = AgentSession(
        project_id=project_id,
        artifact_type="research_output",
        workflow_area="analysis",
        graph_checkpoint={},
        status=AgentSessionStatus.ACTIVE,
    )
    db_session.add(session)
    await db_session.flush()

    await svc._run_graph(
        session_id=session.id,
        project_id=project_id,
        artifact_type="research_output",
        step_key="intent_vision",
        workflow_area="analysis",
        agent_role=None,
        missing_context=[],
        llm_client=AsyncMock(),
        initial_state=None,
        resume_command=None,
    )

    updated = (await db_session.execute(select(AgentSession).where(AgentSession.id == session.id))).scalar_one()
    messages = (
        await db_session.execute(select(AgentMessage).where(AgentMessage.session_id == session.id))
    ).scalars().all()
    assert updated.status == AgentSessionStatus.FAILED
    assert len(messages) == 1
    assert messages[0].role == AgentMessageRole.AGENT
    assert "provider rejected request" in messages[0].content


@pytest.mark.asyncio
async def test_run_graph_resume_failure_marks_session_failed_and_saves_agent_message(client, db_session):
    project_id = await _setup(client)
    graph = _mock_graph()
    graph.ainvoke = AsyncMock(side_effect=RuntimeError("resume rejected request"))
    svc = _make_service(db_session, graph)

    session = AgentSession(
        project_id=project_id,
        artifact_type="research_output",
        workflow_area="analysis",
        graph_checkpoint={},
        status=AgentSessionStatus.ACTIVE,
    )
    db_session.add(session)
    await db_session.flush()

    from langgraph.types import Command

    await svc._run_graph(
        session_id=session.id,
        project_id=project_id,
        artifact_type="research_output",
        step_key="intent_vision",
        workflow_area="analysis",
        agent_role=None,
        missing_context=[],
        llm_client=AsyncMock(),
        initial_state=None,
        resume_command=Command(resume={"content": "Thêm thông tin"}),
    )

    updated = (await db_session.execute(select(AgentSession).where(AgentSession.id == session.id))).scalar_one()
    messages = (
        await db_session.execute(select(AgentMessage).where(AgentMessage.session_id == session.id))
    ).scalars().all()
    assert updated.status == AgentSessionStatus.FAILED
    assert len(messages) == 1
    assert messages[0].role == AgentMessageRole.AGENT
    assert "resume rejected request" in messages[0].content


# ---------------------------------------------------------------------------
# approve_tool_call — batch logic
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_approve_tool_call_batch_first_does_not_resume(client, db_session, _no_background_tasks):
    project_id = await _setup(client)
    svc = _make_service(db_session)

    _, _, tc1, _ = await _make_propose_session(db_session, project_id)

    await svc.approve_tool_call(project_id=project_id, tool_call_id=tc1.id, created_by_id=None)

    # create_task not called (batch still has pending)
    _no_background_tasks.assert_not_called()


@pytest.mark.asyncio
async def test_approve_tool_call_batch_all_approved_resumes_once(client, db_session, _no_background_tasks):
    project_id = await _setup(client)
    svc = _make_service(db_session)

    _, _, tc1, tc2 = await _make_propose_session(db_session, project_id)

    await svc.approve_tool_call(project_id=project_id, tool_call_id=tc1.id, created_by_id=None)
    await svc.approve_tool_call(project_id=project_id, tool_call_id=tc2.id, created_by_id=None)

    assert _no_background_tasks.call_count == 1
    scheduled = _no_background_tasks.call_args.args[0]
    assert scheduled.cr_code.co_name == "_run_graph"


@pytest.mark.asyncio
async def test_approve_tool_call_cross_project_returns_404(client, db_session):
    project_id_a = await _setup(client)
    project_id_b = await _setup(client)

    _, _, tc, _ = await _make_propose_session(db_session, project_id_a)

    with pytest.raises(HTTPException) as exc:
        await _make_service(db_session).approve_tool_call(
            project_id=project_id_b, tool_call_id=tc.id, created_by_id=None
        )

    assert exc.value.status_code == 404


# ---------------------------------------------------------------------------
# ArtifactVersion traceability
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_approve_tool_call_sets_artifact_version_traceability(client, db_session):
    project_id = await _setup(client)
    svc = _make_service(db_session)

    session, run, tc, _ = await _make_propose_session(db_session, project_id)

    await svc.approve_tool_call(project_id=project_id, tool_call_id=tc.id, created_by_id=None)

    updated_tc = (await db_session.execute(select(AgentToolCall).where(AgentToolCall.id == tc.id))).scalar_one()
    assert updated_tc.status == AgentToolCallStatus.EXECUTED
    assert updated_tc.created_artifact_id is not None
    assert updated_tc.created_version_id is not None

    version = (await db_session.execute(select(ArtifactVersion).where(ArtifactVersion.id == updated_tc.created_version_id))).scalar_one()
    assert version.agent_run_id == run.id
    assert version.tool_call_id == tc.id


# ---------------------------------------------------------------------------
# reject_tool_call
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reject_tool_call_batch_all_rejected_resumes(client, db_session, _no_background_tasks):
    project_id = await _setup(client)
    svc = _make_service(db_session)

    _, _, tc1, tc2 = await _make_propose_session(db_session, project_id)

    await svc.reject_tool_call(project_id=project_id, tool_call_id=tc1.id)
    await svc.reject_tool_call(project_id=project_id, tool_call_id=tc2.id)

    assert _no_background_tasks.call_count == 1


# ---------------------------------------------------------------------------
# request_edit
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_request_edit_supersedes_tool_call_and_resumes_when_last(client, db_session, _no_background_tasks):
    """request_edit resumes the graph only when it supersedes the LAST proposed tool call."""
    project_id = await _setup(client)
    svc = _make_service(db_session)

    # Session with a single proposed tool call — edit on it should trigger resume.
    session, run, tc = await _make_single_propose_session(db_session, project_id)

    await svc.request_edit(project_id=project_id, tool_call_id=tc.id, note="Cần chỉnh sửa")

    updated = (await db_session.execute(select(AgentToolCall).where(AgentToolCall.id == tc.id))).scalar_one()
    assert updated.status == AgentToolCallStatus.SUPERSEDED
    _no_background_tasks.assert_called_once()


@pytest.mark.asyncio
async def test_request_edit_does_not_resume_when_others_still_proposed(client, db_session, _no_background_tasks):
    """request_edit on one of many proposed tool calls must NOT trigger graph resume."""
    project_id = await _setup(client)
    svc = _make_service(db_session)

    _, _, tc1, _ = await _make_propose_session(db_session, project_id)

    await svc.request_edit(project_id=project_id, tool_call_id=tc1.id, note="Cần chỉnh sửa")

    _no_background_tasks.assert_not_called()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _make_single_propose_session(db_session, project_id):
    session = AgentSession(
        project_id=project_id, artifact_type="goal", workflow_area="analysis",
        graph_checkpoint={}, status=AgentSessionStatus.WAITING_FOR_HUMAN,
        interrupt_type=AgentSessionInterruptType.PROPOSE_ARTIFACTS,
    )
    db_session.add(session)
    await db_session.flush()

    run = AgentRun(session_id=session.id, analysis_result={})
    db_session.add(run)
    await db_session.flush()

    tc = AgentToolCall(
        run_id=run.id, tool_name="create_artifact",
        input_snapshot={"artifact_type": "goal", "title": "Mục tiêu", "body": "Mô tả"},
        status=AgentToolCallStatus.PROPOSED,
    )
    db_session.add(tc)
    await db_session.flush()
    return session, run, tc


async def _make_propose_session(db_session, project_id):
    session = AgentSession(
        project_id=project_id, artifact_type="goal", workflow_area="analysis",
        graph_checkpoint={}, status=AgentSessionStatus.WAITING_FOR_HUMAN,
        interrupt_type=AgentSessionInterruptType.PROPOSE_ARTIFACTS,
    )
    db_session.add(session)
    await db_session.flush()

    run = AgentRun(session_id=session.id, analysis_result={})
    db_session.add(run)
    await db_session.flush()

    tc1 = AgentToolCall(
        run_id=run.id, tool_name="create_artifact",
        input_snapshot={"artifact_type": "goal", "title": "Mục tiêu A", "body": "Mô tả"},
        status=AgentToolCallStatus.PROPOSED,
    )
    tc2 = AgentToolCall(
        run_id=run.id, tool_name="create_artifact",
        input_snapshot={"artifact_type": "goal", "title": "Mục tiêu B", "body": "Mô tả"},
        status=AgentToolCallStatus.PROPOSED,
    )
    db_session.add(tc1)
    db_session.add(tc2)
    await db_session.flush()
    return session, run, tc1, tc2
