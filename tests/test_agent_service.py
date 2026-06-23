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
from app.models.artifact import (
    Artifact,
    ArtifactStatus,
    ArtifactType,
    ArtifactVersion,
)
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

    assert result["missing_context"] == []


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

        client, strong_client = await _make_service(db_session)._resolve_llm_client(config.id)

        assert client is sentinel
        assert strong_client is None
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
async def test_resolve_llm_client_returns_strong_when_configured(db_session, monkeypatch):
    original_key = settings.encryption_key
    original_previous = settings.encryption_key_previous
    monkeypatch.setattr(settings, "encryption_key", Fernet.generate_key().decode())
    monkeypatch.setattr(settings, "encryption_key_previous", "")
    crypto._get_fernet.cache_clear()

    try:
        created = []

        def fake_create(**kwargs):
            client = object()
            created.append((kwargs, client))
            return client

        monkeypatch.setattr("app.services.llm_clients.LLMClientFactory.create", fake_create)
        config = LLMProviderConfig(
            user_id=uuid.uuid4(),
            provider_type=ProviderType.BEDROCK,
            name="Bedrock",
            encrypted_api_key=encrypt_token("AKIATEST"),
            encrypted_secret_key=encrypt_token("aws-secret"),
            region="us-east-1",
            model_name="amazon.nova-lite-v1:0",
            strong_model_name="anthropic.claude-3-5-sonnet-20241022-v2:0",
        )
        db_session.add(config)
        await db_session.flush()

        default_client, strong_client = await _make_service(db_session)._resolve_llm_client(config.id)

        assert default_client is created[0][1]
        assert strong_client is created[1][1]
        assert [item[0]["model"] for item in created] == [
            "amazon.nova-lite-v1:0",
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
        ]
        assert all(item[0]["api_key"] == "AKIATEST" for item in created)
        assert all(item[0]["secret_key"] == "aws-secret" for item in created)
        assert all(item[0]["region"] == "us-east-1" for item in created)
    finally:
        monkeypatch.setattr(settings, "encryption_key", original_key)
        monkeypatch.setattr(settings, "encryption_key_previous", original_previous)
        crypto._get_fernet.cache_clear()


@pytest.mark.asyncio
async def test_resolve_llm_client_strong_none_when_unset(db_session, monkeypatch):
    original_key = settings.encryption_key
    original_previous = settings.encryption_key_previous
    monkeypatch.setattr(settings, "encryption_key", Fernet.generate_key().decode())
    monkeypatch.setattr(settings, "encryption_key_previous", "")
    crypto._get_fernet.cache_clear()

    try:
        created = []

        def fake_create(**kwargs):
            client = object()
            created.append((kwargs, client))
            return client

        monkeypatch.setattr("app.services.llm_clients.LLMClientFactory.create", fake_create)
        config = LLMProviderConfig(
            user_id=uuid.uuid4(),
            provider_type=ProviderType.OPENAI,
            name="OpenAI",
            encrypted_api_key=encrypt_token("sk-test"),
            model_name="gpt-4o-mini",
        )
        db_session.add(config)
        await db_session.flush()

        default_client, strong_client = await _make_service(db_session)._resolve_llm_client(config.id)

        assert default_client is created[0][1]
        assert strong_client is None
        assert len(created) == 1
    finally:
        monkeypatch.setattr(settings, "encryption_key", original_key)
        monkeypatch.setattr(settings, "encryption_key_previous", original_previous)
        crypto._get_fernet.cache_clear()


def test_make_config_exposes_strong_llm_client(db_session):
    svc = _make_service(db_session)
    session_id = uuid.uuid4()
    project_id = uuid.uuid4()
    default_client = object()
    strong_client = object()

    config = svc._make_config(
        session_id,
        project_id,
        default_client,
        agent_role="analyst",
        strong_llm_client=strong_client,
    )

    assert config["configurable"]["llm_client"] is default_client
    assert config["configurable"]["strong_llm_client"] is strong_client


@pytest.mark.asyncio
async def test_create_session_goal_with_intent_missing_problem(client, db_session):
    project_id = await _setup(client)
    svc = _make_service(db_session)
    result = await svc.create_session(project_id=project_id, artifact_type="goal")

    assert result["missing_context"] == []


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
    owner_id = uuid.uuid4()
    svc = _make_service(db_session)

    await svc.create_session(project_id=project_id, artifact_type="intent", created_by_id=owner_id)

    with pytest.raises(HTTPException) as exc:
        await _make_service(db_session).create_session(
            project_id=project_id,
            artifact_type="intent",
            created_by_id=owner_id,
        )

    assert exc.value.status_code == 409
    assert "session_id" in exc.value.detail


@pytest.mark.asyncio
async def test_create_session_same_artifact_allowed_for_different_users(client, db_session):
    project_id = await _setup(client)
    svc = _make_service(db_session)

    await svc.create_session(project_id=project_id, artifact_type="intent", created_by_id=uuid.uuid4())
    result = await svc.create_session(project_id=project_id, artifact_type="intent", created_by_id=uuid.uuid4())

    assert result["session_id"]


# ---------------------------------------------------------------------------
# handle_user_message
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handle_user_message_ask_human_resumes_graph(client, db_session, _no_background_tasks):
    project_id = await _setup(client)
    owner_id = uuid.uuid4()
    graph = _mock_graph()
    svc = _make_service(db_session, graph)

    session = AgentSession(
        project_id=project_id, artifact_type="goal", workflow_area="analysis",
        graph_checkpoint={}, status=AgentSessionStatus.WAITING_FOR_HUMAN,
        interrupt_type=AgentSessionInterruptType.ASK_HUMAN,
        created_by_id=owner_id,
    )
    db_session.add(session)
    await db_session.flush()

    await svc.handle_user_message(
        project_id=project_id,
        session_id=session.id,
        content="Thêm thông tin",
        user_id=owner_id,
    )

    # Background task was scheduled through _run_graph so resume failures update session status/messages.
    _no_background_tasks.assert_called_once()
    scheduled = _no_background_tasks.call_args.args[0]
    assert scheduled.cr_code.co_name == "_run_graph"


@pytest.mark.asyncio
async def test_resume_command_uses_keyed_form_for_single_interrupt(db_session):
    from langgraph.types import Interrupt

    from app.graphs.checkpointer import AgentSessionCheckpointer

    # 32-char hex to match the xxh3_128_hexdigest format LangGraph's interrupt() produces
    INTERRUPT_ID = "a1b2c3d4e5f6789012345678901234ab"

    svc = _make_service(db_session)
    session = AgentSession(
        id=uuid.uuid4(),
        project_id=uuid.uuid4(),
        artifact_type="goal",
        workflow_area="analysis",
        graph_checkpoint={},
        status=AgentSessionStatus.WAITING_FOR_HUMAN,
        interrupt_type=AgentSessionInterruptType.ASK_HUMAN,
    )
    checker = AgentSessionCheckpointer(session_id=str(session.id), session_factory=svc.session_factory)
    session.graph_checkpoint = {
        "pending_writes": [
            checker._dump_pending_write(
                "task-1",
                "__interrupt__",
                [Interrupt(value={"type": "ask_human", "message": "Tiếp tục?"}, id=INTERRUPT_ID)],
            )
        ]
    }

    command = svc._resume_command(session, {"content": "Có"})

    assert command.resume == {INTERRUPT_ID: {"content": "Có"}}


@pytest.mark.asyncio
async def test_resume_command_keys_all_interrupt_ids_when_multiple_pending(db_session):
    # When more than one interrupt is pending, LangGraph rejects an unkeyed
    # Command(resume=...) ("you must specify the interrupt id when resuming").
    # Keying only the latest would leave the other pending and trigger that error
    # on the next resume — so the reply must address EVERY pending interrupt.
    from langgraph.types import Interrupt

    from app.graphs.checkpointer import AgentSessionCheckpointer

    # 32-char hex IDs matching the xxh3_128_hexdigest format LangGraph's interrupt() produces
    INTERRUPT_ID_1 = "a1b2c3d4e5f6789012345678901234ab"
    INTERRUPT_ID_2 = "b2c3d4e5f6789012345678901234abcd"

    svc = _make_service(db_session)
    session = AgentSession(
        id=uuid.uuid4(),
        project_id=uuid.uuid4(),
        artifact_type="goal",
        workflow_area="analysis",
        graph_checkpoint={},
        status=AgentSessionStatus.WAITING_FOR_HUMAN,
        interrupt_type=AgentSessionInterruptType.ASK_HUMAN,
    )
    checker = AgentSessionCheckpointer(session_id=str(session.id), session_factory=svc.session_factory)
    session.graph_checkpoint = {
        "pending_writes": [
            checker._dump_pending_write(
                "task-1",
                "__interrupt__",
                [Interrupt(value={"type": "ask_human", "message": "Lần 1?"}, id=INTERRUPT_ID_1)],
            ),
            checker._dump_pending_write(
                "task-2",
                "__interrupt__",
                [Interrupt(value={"type": "ask_human", "message": "Lần 2?"}, id=INTERRUPT_ID_2)],
            ),
        ]
    }

    command = svc._resume_command(session, {"content": "Có"})

    assert command.resume == {
        INTERRUPT_ID_1: {"content": "Có"},
        INTERRUPT_ID_2: {"content": "Có"},
    }


@pytest.mark.asyncio
async def test_handle_user_message_rejects_non_owner(client, db_session):
    project_id = await _setup(client)
    owner_id = uuid.uuid4()
    svc = _make_service(db_session)

    session = AgentSession(
        project_id=project_id,
        artifact_type="goal",
        workflow_area="analysis",
        graph_checkpoint={},
        status=AgentSessionStatus.WAITING_FOR_HUMAN,
        interrupt_type=None,
        created_by_id=owner_id,
    )
    db_session.add(session)
    await db_session.flush()

    with pytest.raises(HTTPException) as exc:
        await svc.handle_user_message(
            project_id=project_id,
            session_id=session.id,
            content="Xin chào",
            user_id=uuid.uuid4(),
        )

    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_handle_user_message_when_active_returns_200_and_queues(client, db_session, _no_background_tasks):
    """S2: sending a message while the session is ACTIVE queues it instead of returning 400."""
    project_id = await _setup(client)
    svc = _make_service(db_session)

    session = AgentSession(
        project_id=project_id, artifact_type="goal", workflow_area="analysis",
        graph_checkpoint={}, status=AgentSessionStatus.ACTIVE,
    )
    db_session.add(session)
    await db_session.flush()

    msg = await svc.handle_user_message(project_id=project_id, session_id=session.id, content="tạo đi")

    assert msg.role == AgentMessageRole.USER
    assert msg.payload["queued"] is True
    _no_background_tasks.assert_not_called()


@pytest.mark.asyncio
async def test_handle_user_message_propose_artifacts_returns_200_and_queues(client, db_session, _no_background_tasks):
    """S2 without carve-out: PROPOSE_ARTIFACTS also queues instead of returning 400."""
    project_id = await _setup(client)
    svc = _make_service(db_session)

    session = AgentSession(
        project_id=project_id, artifact_type="goal", workflow_area="analysis",
        graph_checkpoint={}, status=AgentSessionStatus.WAITING_FOR_HUMAN,
        interrupt_type=AgentSessionInterruptType.PROPOSE_ARTIFACTS,
    )
    db_session.add(session)
    await db_session.flush()

    msg = await svc.handle_user_message(project_id=project_id, session_id=session.id, content="ok tạo đi")

    assert msg.payload["queued"] is True
    _no_background_tasks.assert_not_called()


@pytest.mark.asyncio
async def test_queued_message_not_a_second_graph_task(client, db_session, _no_background_tasks):
    """Queuing while ACTIVE must not spawn a second graph task."""
    project_id = await _setup(client)
    svc = _make_service(db_session)

    session = AgentSession(
        project_id=project_id, artifact_type="goal", workflow_area="analysis",
        graph_checkpoint={}, status=AgentSessionStatus.ACTIVE,
    )
    db_session.add(session)
    await db_session.flush()

    await svc.handle_user_message(project_id=project_id, session_id=session.id, content="hello")

    _no_background_tasks.assert_not_called()


@pytest.mark.asyncio
async def test_drain_queue_processes_queued_messages_after_completed(client, db_session, _no_background_tasks):
    """After a COMPLETED turn, the queued message is dequeued and a new turn is scheduled."""
    project_id = await _setup(client)
    svc = _make_service(db_session)

    session = AgentSession(
        project_id=project_id, artifact_type="goal", workflow_area="analysis",
        graph_checkpoint={}, status=AgentSessionStatus.COMPLETED,
    )
    db_session.add(session)
    await db_session.flush()
    queued = AgentMessage(
        session_id=session.id, role=AgentMessageRole.USER, content="tạo đi", payload={"queued": True}
    )
    db_session.add(queued)
    await db_session.flush()

    await svc._drain_queue(
        session_id=session.id, project_id=project_id, artifact_type="goal", step_key=None,
        workflow_area="analysis", agent_role=None, missing_context=[], llm_client=AsyncMock(),
    )

    await db_session.refresh(queued)
    assert queued.payload["queued"] is False
    _no_background_tasks.assert_called_once()
    scheduled = _no_background_tasks.call_args.args[0]
    assert scheduled.cr_code.co_name == "_run_graph"


@pytest.mark.asyncio
async def test_drain_queue_does_not_fire_after_waiting_for_human(client, db_session, _no_background_tasks):
    """A WAITING_FOR_HUMAN turn must not drain; the queued message stays queued."""
    project_id = await _setup(client)
    svc = _make_service(db_session)

    session = AgentSession(
        project_id=project_id, artifact_type="goal", workflow_area="analysis",
        graph_checkpoint={}, status=AgentSessionStatus.WAITING_FOR_HUMAN,
        interrupt_type=AgentSessionInterruptType.ASK_HUMAN,
    )
    db_session.add(session)
    await db_session.flush()
    queued = AgentMessage(
        session_id=session.id, role=AgentMessageRole.USER, content="tạo đi", payload={"queued": True}
    )
    db_session.add(queued)
    await db_session.flush()

    await svc._drain_queue(
        session_id=session.id, project_id=project_id, artifact_type="goal", step_key=None,
        workflow_area="analysis", agent_role=None, missing_context=[], llm_client=AsyncMock(),
    )

    await db_session.refresh(queued)
    assert queued.payload["queued"] is True
    _no_background_tasks.assert_not_called()


@pytest.mark.asyncio
async def test_run_graph_timeout_sets_session_failed(client, db_session, monkeypatch):
    """An ainvoke timeout is caught inside _run_graph and marks the session FAILED."""
    project_id = await _setup(client)

    async def _slow(*args, **kwargs):
        await asyncio.sleep(5)

    graph = _mock_graph()
    graph.ainvoke = _slow
    svc = _make_service(db_session, graph)
    monkeypatch.setattr(settings, "agent_turn_timeout_seconds", 0.01)

    session = AgentSession(
        project_id=project_id, artifact_type="goal", workflow_area="analysis",
        graph_checkpoint={}, status=AgentSessionStatus.ACTIVE,
    )
    db_session.add(session)
    await db_session.flush()

    await svc._run_graph(
        session_id=session.id, project_id=project_id, artifact_type="goal", step_key=None,
        workflow_area="analysis", agent_role=None, missing_context=[], llm_client=AsyncMock(),
        initial_state=None, resume_command=None,
    )

    updated = (await db_session.execute(select(AgentSession).where(AgentSession.id == session.id))).scalar_one()
    messages = (
        await db_session.execute(select(AgentMessage).where(AgentMessage.session_id == session.id))
    ).scalars().all()
    assert updated.status == AgentSessionStatus.FAILED
    assert any(m.role == AgentMessageRole.AGENT for m in messages)


@pytest.mark.asyncio
async def test_run_graph_timeout_does_not_affect_normal_flow(client, db_session, monkeypatch):
    """A normal ainvoke return under a wide timeout marks the session COMPLETED, not FAILED."""
    project_id = await _setup(client)
    graph = _mock_graph()
    svc = _make_service(db_session, graph)
    monkeypatch.setattr(settings, "agent_turn_timeout_seconds", 90.0)

    session = AgentSession(
        project_id=project_id, artifact_type="goal", workflow_area="analysis",
        graph_checkpoint={}, status=AgentSessionStatus.ACTIVE,
    )
    db_session.add(session)
    await db_session.flush()

    await svc._run_graph(
        session_id=session.id, project_id=project_id, artifact_type="goal", step_key=None,
        workflow_area="analysis", agent_role=None, missing_context=[], llm_client=AsyncMock(),
        initial_state=None, resume_command=None,
    )

    updated = (await db_session.execute(select(AgentSession).where(AgentSession.id == session.id))).scalar_one()
    assert updated.status == AgentSessionStatus.COMPLETED


@pytest.mark.asyncio
async def test_run_graph_with_initial_state_none_has_locale_and_intent(client, db_session):
    """Fallback state with initial_state=None includes locale and intent for graph reads."""
    project_id = await _setup(client)
    graph = _mock_graph()
    svc = _make_service(db_session, graph)

    session = AgentSession(
        project_id=project_id, artifact_type="goal", workflow_area="analysis",
        graph_checkpoint={}, status=AgentSessionStatus.ACTIVE,
    )
    db_session.add(session)
    await db_session.flush()

    await svc._run_graph(
        session_id=session.id, project_id=project_id, artifact_type="goal", step_key=None,
        workflow_area="analysis", agent_role=None, missing_context=[], llm_client=AsyncMock(),
        initial_state=None, resume_command=None,
    )

    passed_state = graph.ainvoke.call_args.args[0]
    assert "locale" in passed_state and passed_state["locale"] is None
    assert passed_state["section_coverage"] is None
    assert passed_state["coverage_complete"] is None
    assert passed_state["section_coverage_stall_count"] is None
    assert passed_state["focused_artifact_id"] is None


@pytest.mark.asyncio
async def test_drain_json_path_query_correctness(client, db_session):
    """JSON-path query payload.queued==True returns exactly one row for sqlite/postgres correctness."""
    project_id = await _setup(client)
    session = AgentSession(
        project_id=project_id, artifact_type="goal", workflow_area="analysis",
        graph_checkpoint={}, status=AgentSessionStatus.COMPLETED,
    )
    db_session.add(session)
    await db_session.flush()
    db_session.add_all([
        AgentMessage(session_id=session.id, role=AgentMessageRole.USER, content="a", payload={"queued": True}),
        AgentMessage(session_id=session.id, role=AgentMessageRole.USER, content="b", payload={"queued": False}),
    ])
    await db_session.flush()

    rows = (
        await db_session.execute(
            select(AgentMessage).where(
                AgentMessage.session_id == session.id,
                AgentMessage.payload["queued"].as_boolean().is_(True),
            )
        )
    ).scalars().all()

    assert len(rows) == 1
    assert rows[0].content == "a"


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


@pytest.mark.asyncio
async def test_approve_tool_call_rejects_non_owner(client, db_session):
    project_id = await _setup(client)
    owner_id = uuid.uuid4()
    _, _, tc, _ = await _make_propose_session(db_session, project_id, created_by_id=owner_id)

    with pytest.raises(HTTPException) as exc:
        await _make_service(db_session).approve_tool_call(
            project_id=project_id,
            tool_call_id=tc.id,
            created_by_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
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
    assert version.body == "Mô tả"


@pytest.mark.asyncio
async def test_approve_tool_call_rejects_missing_focused_artifact(client, db_session):
    project_id = await _setup(client)
    svc = _make_service(db_session)
    session, run, tc = await _make_single_propose_session(db_session, project_id)
    tc.input_snapshot = {"artifact_type": "unknown_type", "title": "Mục tiêu", "body": "Mô tả"}
    await db_session.flush()

    with pytest.raises(HTTPException) as exc:
        await svc.approve_tool_call(project_id=project_id, tool_call_id=tc.id, created_by_id=None)

    assert exc.value.status_code == 422
    versions = (
        await db_session.execute(
            select(ArtifactVersion).where(ArtifactVersion.agent_run_id == run.id)
        )
    ).scalars().all()
    assert versions == []


@pytest.mark.asyncio
async def test_approve_tool_call_updates_each_focused_document_item(
    client, db_session, _no_background_tasks
):
    project_id = await _setup(client)
    svc = _make_service(db_session)
    session, run, tc1, tc2 = await _make_propose_session(db_session, project_id)
    tc1.input_snapshot = {
        "artifact_type": "vision_objectives",
        "title": "Vision",
        "body": "Section A",
        "focused_artifact_id": tc1.input_snapshot["focused_artifact_id"],
    }
    tc2.input_snapshot = {
        "artifact_type": "problem_statement",
        "title": "Problem",
        "body": "Section B",
        "focused_artifact_id": tc2.input_snapshot["focused_artifact_id"],
    }
    await db_session.flush()

    await svc.approve_tool_call(project_id=project_id, tool_call_id=tc1.id, created_by_id=None)
    await svc.approve_tool_call(project_id=project_id, tool_call_id=tc2.id, created_by_id=None)

    artifacts = (await db_session.execute(select(Artifact).where(Artifact.project_id == project_id))).scalars().all()
    children = [artifact for artifact in artifacts if artifact.parent_id is not None]
    assert {artifact.type.value for artifact in children} == {
        "vision_objectives",
        "problem_statement",
    }
    bodies = {}
    for artifact in children:
        version = await db_session.get(ArtifactVersion, artifact.current_version_id)
        bodies[artifact.type.value] = version.body
    assert bodies == {
        "vision_objectives": "Section A",
        "problem_statement": "Section B",
    }


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

async def _make_single_propose_session(db_session, project_id, created_by_id=None):
    _, focused, _ = await _make_brd_items(db_session, project_id)
    session = AgentSession(
        project_id=project_id, artifact_type="vision_objectives", workflow_area="analysis",
        graph_checkpoint={}, status=AgentSessionStatus.WAITING_FOR_HUMAN,
        interrupt_type=AgentSessionInterruptType.PROPOSE_ARTIFACTS,
        created_by_id=created_by_id,
        focused_artifact_id=focused.id,
    )
    db_session.add(session)
    await db_session.flush()

    run = AgentRun(session_id=session.id, analysis_result={})
    db_session.add(run)
    await db_session.flush()

    tc = AgentToolCall(
        run_id=run.id, tool_name="create_artifact",
        input_snapshot={
            "artifact_type": "vision_objectives",
            "title": "Mục tiêu",
            "body": "Mô tả",
            "focused_artifact_id": str(focused.id),
        },
        status=AgentToolCallStatus.PROPOSED,
    )
    db_session.add(tc)
    await db_session.flush()
    return session, run, tc


async def _make_propose_session(db_session, project_id, created_by_id=None):
    _, focused_a, focused_b = await _make_brd_items(db_session, project_id)
    session = AgentSession(
        project_id=project_id, artifact_type="vision_objectives", workflow_area="analysis",
        graph_checkpoint={}, status=AgentSessionStatus.WAITING_FOR_HUMAN,
        interrupt_type=AgentSessionInterruptType.PROPOSE_ARTIFACTS,
        created_by_id=created_by_id,
        focused_artifact_id=focused_a.id,
    )
    db_session.add(session)
    await db_session.flush()

    run = AgentRun(session_id=session.id, analysis_result={})
    db_session.add(run)
    await db_session.flush()

    tc1 = AgentToolCall(
        run_id=run.id, tool_name="create_artifact",
        input_snapshot={
            "artifact_type": "vision_objectives",
            "title": "Mục tiêu A",
            "body": "Mô tả",
            "focused_artifact_id": str(focused_a.id),
        },
        status=AgentToolCallStatus.PROPOSED,
    )
    tc2 = AgentToolCall(
        run_id=run.id, tool_name="create_artifact",
        input_snapshot={
            "artifact_type": "problem_statement",
            "title": "Mục tiêu B",
            "body": "Mô tả",
            "focused_artifact_id": str(focused_b.id),
        },
        status=AgentToolCallStatus.PROPOSED,
    )
    db_session.add(tc1)
    db_session.add(tc2)
    await db_session.flush()
    return session, run, tc1, tc2


async def _make_brd_items(db_session, project_id):
    parent = Artifact(
        project_id=project_id,
        type=ArtifactType.BRD,
        status=ArtifactStatus.DRAFT,
        title="BRD",
        extra_metadata={},
    )
    db_session.add(parent)
    await db_session.flush()
    focused_a = Artifact(
        project_id=project_id,
        parent_id=parent.id,
        type=ArtifactType.VISION_OBJECTIVES,
        status=ArtifactStatus.DRAFT,
        title="Vision",
        extra_metadata={},
    )
    focused_b = Artifact(
        project_id=project_id,
        parent_id=parent.id,
        type=ArtifactType.PROBLEM_STATEMENT,
        status=ArtifactStatus.DRAFT,
        title="Problem",
        extra_metadata={},
    )
    db_session.add_all([focused_a, focused_b])
    await db_session.flush()
    return parent, focused_a, focused_b


# ---------------------------------------------------------------------------
# Regression: create_session does not run the graph immediately; the first message starts it.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_session_does_not_start_graph(client, db_session, _no_background_tasks):
    """create_session creates a WAITING_FOR_HUMAN session without scheduling _run_graph."""
    project_id = await _setup(client)
    svc = _make_service(db_session)

    await svc.create_session(project_id=project_id, artifact_type="intent")

    _no_background_tasks.assert_not_called()

    session = (await db_session.execute(
        select(AgentSession).where(AgentSession.project_id == project_id)
    )).scalar_one()
    assert session.status == AgentSessionStatus.WAITING_FOR_HUMAN
    assert session.interrupt_type is None


@pytest.mark.asyncio
async def test_first_user_message_starts_graph_fresh(client, db_session, _no_background_tasks):
    """The first user message with interrupt_type=None invokes the graph with initial_state."""
    project_id = await _setup(client)
    graph = _mock_graph()
    svc = _make_service(db_session, graph)

    session = AgentSession(
        project_id=project_id, artifact_type="intent", workflow_area="analysis",
        graph_checkpoint={}, status=AgentSessionStatus.WAITING_FOR_HUMAN,
        interrupt_type=None,
    )
    db_session.add(session)
    await db_session.flush()

    await svc.handle_user_message(project_id=project_id, session_id=session.id, content="Xin chào")

    _no_background_tasks.assert_called_once()
    scheduled = _no_background_tasks.call_args.args[0]
    assert scheduled.cr_code.co_name == "_run_graph"
