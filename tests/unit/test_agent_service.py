import asyncio
import logging
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cryptography.fernet import Fernet
from fastapi import HTTPException
from sqlalchemy import select

from app.config import settings
from app.core import crypto
from app.core.crypto import encrypt_token
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
from app.models.artifact import (
    Artifact,
    ArtifactEvidence,
    ArtifactLink,
    ArtifactStatus,
    ArtifactType,
    ArtifactVersion,
    ChangeSource,
    RelationType,
    SourceDocument,
    SourceType,
    VersionStatus,
)
from app.models.llm_provider import LLMProviderConfig, LLMProviderStatus, ProviderType
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


async def _setup_with_user(client):
    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    token = headers["Authorization"].removeprefix("Bearer ")
    return uuid.UUID(project["id"]), uuid.UUID(decode_token(token)["sub"])


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
            status=LLMProviderStatus.ACTIVE,
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
            status=LLMProviderStatus.ACTIVE,
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
            status=LLMProviderStatus.ACTIVE,
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


@pytest.mark.asyncio
async def test_resolve_llm_client_rejects_unchecked_config(db_session, monkeypatch):
    original_key = settings.encryption_key
    original_previous = settings.encryption_key_previous
    monkeypatch.setattr(settings, "encryption_key", Fernet.generate_key().decode())
    monkeypatch.setattr(settings, "encryption_key_previous", "")
    crypto._get_fernet.cache_clear()

    try:
        config = LLMProviderConfig(
            user_id=uuid.uuid4(),
            provider_type=ProviderType.OPENAI,
            name="OpenAI",
            encrypted_api_key=encrypt_token("sk-test"),
            model_name="gpt-4o-mini",
            status=LLMProviderStatus.DRAFT,
        )
        db_session.add(config)
        await db_session.flush()

        with pytest.raises(HTTPException) as exc_info:
            await _make_service(db_session)._resolve_llm_client(config.id)

        assert exc_info.value.status_code == 422
        assert exc_info.value.detail == "LLM provider config must pass health check before use"
    finally:
        monkeypatch.setattr(settings, "encryption_key", original_key)
        monkeypatch.setattr(settings, "encryption_key_previous", original_previous)
        crypto._get_fernet.cache_clear()


@pytest.mark.asyncio
async def test_handle_user_message_rejects_unchecked_config_before_session_side_effects(
    client,
    db_session,
    monkeypatch,
):
    project_id = await _setup(client)
    original_key = settings.encryption_key
    original_previous = settings.encryption_key_previous
    monkeypatch.setattr(settings, "encryption_key", Fernet.generate_key().decode())
    monkeypatch.setattr(settings, "encryption_key_previous", "")
    crypto._get_fernet.cache_clear()

    try:
        config = LLMProviderConfig(
            user_id=uuid.uuid4(),
            provider_type=ProviderType.OPENAI,
            name="OpenAI",
            encrypted_api_key=encrypt_token("sk-test"),
            model_name="gpt-4o-mini",
            status=LLMProviderStatus.DRAFT,
        )
        db_session.add(config)
        await db_session.flush()

        svc = _make_service(db_session)
        session_payload = await svc.create_session(
            project_id=project_id,
            artifact_type="goal",
            provider_config_id=config.id,
        )
        session_id = uuid.UUID(session_payload["session_id"])

        with pytest.raises(HTTPException) as exc_info:
            await svc.handle_user_message(project_id=project_id, session_id=session_id, content="Xin chao")

        session = await svc.get_session(project_id=project_id, session_id=session_id)
        messages = (
            await db_session.execute(
                select(AgentMessage).where(
                    AgentMessage.session_id == session_id,
                    AgentMessage.role == AgentMessageRole.USER,
                )
            )
        ).scalars().all()

        assert exc_info.value.status_code == 422
        assert session.status == AgentSessionStatus.WAITING_FOR_HUMAN
        assert messages == []
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


@pytest.mark.asyncio
async def test_create_session_warns_when_predecessor_not_accepted(client, db_session):
    project_id = await _setup(client)
    db_session.add(
        Artifact(
            project_id=project_id,
            type=ArtifactType.BRD,
            status=ArtifactStatus.DRAFT,
            title="Draft BRD",
            extra_metadata={},
        )
    )
    await db_session.flush()

    result = await _make_service(db_session).create_session(project_id=project_id, artifact_type="prd")

    assert result["missing_context"] == ["brd"]


@pytest.mark.asyncio
async def test_create_session_allows_accepted_predecessor(client, db_session):
    project_id = await _setup(client)
    db_session.add(
        Artifact(
            project_id=project_id,
            type=ArtifactType.BRD,
            status=ArtifactStatus.ACCEPTED,
            title="BRD da accepted",
            extra_metadata={},
        )
    )
    await db_session.flush()

    result = await _make_service(db_session).create_session(project_id=project_id, artifact_type="prd")

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
        content="Them thong tin",
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
                [Interrupt(value={"type": "ask_human", "message": "Tiep tuc?"}, id=INTERRUPT_ID)],
            )
        ]
    }

    command = svc._resume_command(session, {"content": "Co"})

    assert command.resume == {INTERRUPT_ID: {"content": "Co"}}
    # Every human resume resets the per-request silent-loop counter so conversations are unbounded,
    # and the per-turn diagnosis judge budget so a later turn can escalate again.
    assert command.update == {"turn_count": 0, "diagnosis_judge_calls_used": 0}


@pytest.mark.asyncio
async def test_resume_command_merges_turn_count_reset_with_state_update(db_session):
    svc = _make_service(db_session)
    session = AgentSession(
        id=uuid.uuid4(), project_id=uuid.uuid4(), artifact_type="goal",
        workflow_area="analysis", graph_checkpoint={},
        status=AgentSessionStatus.WAITING_FOR_HUMAN, interrupt_type=AgentSessionInterruptType.ASK_HUMAN,
    )

    command = svc._resume_command(session, {"content": "Co"}, state_update={"mode_hint": "critique"})

    assert command.update == {"turn_count": 0, "diagnosis_judge_calls_used": 0, "mode_hint": "critique"}


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
                [Interrupt(value={"type": "ask_human", "message": "Lan 1?"}, id=INTERRUPT_ID_1)],
            ),
            checker._dump_pending_write(
                "task-2",
                "__interrupt__",
                [Interrupt(value={"type": "ask_human", "message": "Lan 2?"}, id=INTERRUPT_ID_2)],
            ),
        ]
    }

    command = svc._resume_command(session, {"content": "Co"})

    assert command.resume == {
        INTERRUPT_ID_1: {"content": "Co"},
        INTERRUPT_ID_2: {"content": "Co"},
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
            content="Hello",
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

    msg = await svc.handle_user_message(project_id=project_id, session_id=session.id, content="tao di")

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

    msg = await svc.handle_user_message(project_id=project_id, session_id=session.id, content="ok tao di")

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
        session_id=session.id, role=AgentMessageRole.USER, content="tao di", payload={"queued": True}
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
        session_id=session.id, role=AgentMessageRole.USER, content="tao di", payload={"queued": True}
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
async def test_run_graph_turn_cap_marks_failed_not_completed(client, db_session):
    """A graph that ENDs with an undispatched tool_call (route_node hit the turn cap before the
    pending ask_user ran) must be marked FAILED, not COMPLETED — it did not finish, it ran out of turns."""
    from langchain_core.messages import AIMessage

    project_id = await _setup(client)
    graph = _mock_graph()
    # No __interrupt__ (the tool never ran), but the last message still carries tool_calls.
    graph.ainvoke = AsyncMock(return_value={
        "messages": [AIMessage(content="", tool_calls=[
            {"id": "r:0", "name": "ask_user", "args": {"message": "Anything else?"}}
        ])],
    })
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

    updated = (await db_session.execute(select(AgentSession).where(AgentSession.id == session.id))).scalar_one()
    messages = (
        await db_session.execute(select(AgentMessage).where(AgentMessage.session_id == session.id))
    ).scalars().all()
    assert updated.status == AgentSessionStatus.FAILED
    assert any(m.role == AgentMessageRole.AGENT for m in messages)


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
        resume_command=Command(resume={"content": "Them thong tin"}),
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
async def test_approve_tool_call_batch_all_approved_completes_without_resume(client, db_session, _no_background_tasks):
    project_id = await _setup(client)
    svc = _make_service(db_session)

    session, _, tc1, tc2 = await _make_propose_session(db_session, project_id)
    call_count_before = _no_background_tasks.call_count

    await svc.approve_tool_call(project_id=project_id, tool_call_id=tc1.id, created_by_id=None)
    await svc.approve_tool_call(project_id=project_id, tool_call_id=tc2.id, created_by_id=None)

    await db_session.refresh(session)
    assert session.status == AgentSessionStatus.COMPLETED
    assert session.interrupt_type is None
    assert _no_background_tasks.call_count == call_count_before


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
    assert version.body == _vision_body()


@pytest.mark.asyncio
async def test_approve_tool_call_persists_synthesis_metadata_and_parent_version(client, db_session):
    from app.models.artifact import ChangeSource, VersionStatus

    project_id = await _setup(client)
    svc = _make_service(db_session)
    session, run, tc = await _make_single_propose_session(db_session, project_id)
    focused = await db_session.get(Artifact, uuid.UUID(tc.input_snapshot["focused_artifact_id"]))
    old_version = ArtifactVersion(
        artifact_id=focused.id,
        version_number=1,
        title="Vision cu",
        body="Body cu",
        status=VersionStatus.DRAFT,
        change_source=ChangeSource.MANUAL,
        extra_metadata={},
    )
    db_session.add(old_version)
    await db_session.flush()
    focused.current_version_id = old_version.id
    tc.input_snapshot = {
        **tc.input_snapshot,
        "base_version_id": str(old_version.id),
        "synthesis_metadata": {
            "artifact_type": "vision_objectives",
            "focused_artifact_id": str(focused.id),
            "base_version_id": str(old_version.id),
            "evidence_refs": [f"agent_run:{run.id}"],
            "inference_level": "medium",
            "confirmed_assumptions": ["Retention metric confirmed"],
            "pending_assumptions": [],
            "synthesis_source": "bmad_synthesis",
        },
        "candidate_readiness": _sufficient_readiness(),
    }
    await db_session.flush()

    updated_tc = await svc.approve_tool_call(project_id=project_id, tool_call_id=tc.id, created_by_id=None)

    version = await db_session.get(ArtifactVersion, updated_tc.created_version_id)
    assert version.parent_version_id == old_version.id
    assert version.extra_metadata["synthesis_source"] == "bmad_synthesis"
    assert version.extra_metadata["pending_assumptions"] == []
    assert version.extra_metadata["focused_artifact_id"] == str(focused.id)


@pytest.mark.asyncio
async def test_approve_tool_call_persists_lifecycle_metadata(client, db_session):
    from app.models.artifact import ChangeSource, VersionStatus

    project_id = await _setup(client)
    svc = _make_service(db_session)
    _session, _run, tc = await _make_single_propose_session(db_session, project_id)
    predecessor = (
        await db_session.execute(
            select(Artifact).where(
                Artifact.project_id == project_id,
                Artifact.type == ArtifactType.PROBLEM_STATEMENT,
            )
        )
    ).scalar_one()
    predecessor.status = ArtifactStatus.ACCEPTED
    predecessor_version = ArtifactVersion(
        artifact_id=predecessor.id,
        version_number=1,
        title="Problem",
        body="Problem body",
        status=VersionStatus.DRAFT,
        change_source=ChangeSource.MANUAL,
        extra_metadata={},
    )
    db_session.add(predecessor_version)
    await db_session.flush()
    predecessor.current_version_id = predecessor_version.id
    tc.input_snapshot = {
        **tc.input_snapshot,
        "lifecycle_metadata": {
            "based_on": {str(predecessor.id): str(predecessor_version.id)},
            "decision_node_map": {
                "N1": {"section": "## Vision", "rendered_tag": None},
            },
        },
    }
    await db_session.flush()

    updated_tc = await svc.approve_tool_call(project_id=project_id, tool_call_id=tc.id, created_by_id=None)

    version = await db_session.get(ArtifactVersion, updated_tc.created_version_id)
    assert version.extra_metadata["based_on"] == {str(predecessor.id): str(predecessor_version.id)}
    assert version.extra_metadata["decision_node_map"] == {
        "N1": {"section": "## Vision", "rendered_tag": None},
    }
    assert version.extra_metadata["synthesis_source"] == "bmad_synthesis"


@pytest.mark.asyncio
async def test_approve_tool_call_persists_auto_source_evidence_after_approval(client, db_session):
    project_id = await _setup(client)
    svc = _make_service(db_session)
    _session, _run, tc = await _make_single_propose_session(db_session, project_id)
    predecessor = (
        await db_session.execute(
            select(Artifact).where(
                Artifact.project_id == project_id,
                Artifact.type == ArtifactType.PROBLEM_STATEMENT,
            )
        )
    ).scalar_one()
    predecessor_version = ArtifactVersion(
        artifact_id=predecessor.id,
        version_number=1,
        title="Problem",
        body="Problem body",
        status=VersionStatus.DRAFT,
        change_source=ChangeSource.MANUAL,
        extra_metadata={},
    )
    source_document = SourceDocument(
        project_id=project_id,
        title="Interview notes",
        source_type=SourceType.TEXT_PASTE,
        content_text="Users need monthly reports.",
        extra_metadata={},
    )
    db_session.add_all([predecessor_version, source_document])
    await db_session.flush()
    predecessor.current_version_id = predecessor_version.id
    tc.input_snapshot = {
        **tc.input_snapshot,
        "source_evidence": [
            {
                "source_document_id": str(source_document.id),
                "source_type": "document",
                "locator": f"source_document:{source_document.id}",
                "excerpt": "Users need monthly reports.",
                "confidence": 1.0,
                "metadata": {"source_kind": "source_document"},
            },
            {
                "source_type": "ai_output",
                "locator": f"artifact_version:{predecessor_version.id}",
                "excerpt": "Problem body",
                "confidence": 1.0,
                "metadata": {
                    "source_kind": "predecessor_version",
                    "predecessor_artifact_id": str(predecessor.id),
                    "predecessor_version_id": str(predecessor_version.id),
                },
            },
            {
                "source_type": "chat",
                "locator": "bare_claim",
                "excerpt": "A bare claim without a source id.",
                "metadata": {},
            },
        ],
    }
    await db_session.flush()

    updated_tc = await svc.approve_tool_call(project_id=project_id, tool_call_id=tc.id, created_by_id=None)

    rows = (
        await db_session.execute(
            select(ArtifactEvidence).where(ArtifactEvidence.artifact_id == updated_tc.created_artifact_id)
        )
    ).scalars().all()
    assert len(rows) == 2
    assert {row.artifact_version_id for row in rows} == {updated_tc.created_version_id}
    document_row = next(row for row in rows if row.source_document_id == source_document.id)
    predecessor_row = next(row for row in rows if row.source_document_id is None)
    assert document_row.excerpt == "Users need monthly reports."
    assert predecessor_row.extra_metadata["predecessor_version_id"] == str(predecessor_version.id)
    assert predecessor_row.excerpt == "Problem body"


@pytest.mark.asyncio
async def test_reject_tool_call_does_not_persist_auto_source_evidence(client, db_session):
    project_id = await _setup(client)
    svc = _make_service(db_session)
    _session, _run, tc = await _make_single_propose_session(db_session, project_id)
    source_document = SourceDocument(
        project_id=project_id,
        title="Interview notes",
        source_type=SourceType.TEXT_PASTE,
        content_text="Users need monthly reports.",
        extra_metadata={},
    )
    db_session.add(source_document)
    await db_session.flush()
    tc.input_snapshot = {
        **tc.input_snapshot,
        "source_evidence": [
            {
                "source_document_id": str(source_document.id),
                "source_type": "document",
                "locator": f"source_document:{source_document.id}",
                "excerpt": "Users need monthly reports.",
                "metadata": {"source_kind": "source_document"},
            }
        ],
    }
    await db_session.flush()

    await svc.reject_tool_call(project_id=project_id, tool_call_id=tc.id)

    focused_artifact_id = uuid.UUID(tc.input_snapshot["focused_artifact_id"])
    rows = (
        await db_session.execute(select(ArtifactEvidence).where(ArtifactEvidence.artifact_id == focused_artifact_id))
    ).scalars().all()
    assert rows == []


@pytest.mark.asyncio
async def test_approve_tool_call_recovers_stale_base_version_in_loop_and_preserves_http_409(
    client, db_session, _no_background_tasks
):
    from app.models.artifact import ChangeSource, VersionStatus

    project_id = await _setup(client)
    svc = _make_service(db_session)
    session, run, tc = await _make_single_propose_session(db_session, project_id)
    focused = await db_session.get(Artifact, uuid.UUID(tc.input_snapshot["focused_artifact_id"]))
    old_version = ArtifactVersion(
        artifact_id=focused.id,
        version_number=1,
        title="Vision cu",
        body="Body cu",
        status=VersionStatus.DRAFT,
        change_source=ChangeSource.MANUAL,
        extra_metadata={},
    )
    current_version = ArtifactVersion(
        artifact_id=focused.id,
        version_number=2,
        title="Vision new",
        body="Body new",
        status=VersionStatus.DRAFT,
        change_source=ChangeSource.MANUAL,
        extra_metadata={},
    )
    db_session.add_all([old_version, current_version])
    await db_session.flush()
    focused.current_version_id = current_version.id
    tc.input_snapshot = {
        **tc.input_snapshot,
        "base_version_id": str(old_version.id),
        "synthesis_metadata": {
            **tc.input_snapshot["synthesis_metadata"],
            "base_version_id": str(old_version.id),
        },
    }
    await db_session.flush()
    captured = {}
    original_resume_command = svc._resume_command

    def _capture_resume_command(session_row, resume, *, state_update=None):
        captured["state_update"] = state_update
        return original_resume_command(session_row, resume, state_update=state_update)

    svc._resume_command = _capture_resume_command
    _no_background_tasks.reset_mock()

    with pytest.raises(HTTPException) as exc:
        await svc.approve_tool_call(
            project_id=project_id,
            tool_call_id=tc.id,
            created_by_id=None,
            _llm_client=AsyncMock(),
        )

    assert exc.value.status_code == 409
    assert exc.value.detail["base_version_id"] == str(old_version.id)
    assert exc.value.detail["current_version_id"] == str(current_version.id)
    await db_session.refresh(session)
    await db_session.refresh(tc)
    assert tc.status == AgentToolCallStatus.SUPERSEDED
    assert tc.created_version_id is None
    assert session.status == AgentSessionStatus.ACTIVE
    assert _no_background_tasks.call_count == 1
    stale = captured["state_update"]["feedback_summary"]["stale_base_version"]
    assert stale["base_version_id"] == str(old_version.id)
    assert stale["current_version_id"] == str(current_version.id)
    assert stale["artifact_id"] == str(focused.id)


@pytest.mark.asyncio
async def test_approve_tool_call_supersedes_and_resumes_when_lifecycle_predecessor_changed(
    client, db_session, _no_background_tasks
):
    from app.models.artifact import ChangeSource, VersionStatus

    project_id = await _setup(client)
    svc = _make_service(db_session)
    session, run, tc = await _make_single_propose_session(db_session, project_id)
    predecessor = (
        await db_session.execute(
            select(Artifact).where(
                Artifact.project_id == project_id,
                Artifact.type == ArtifactType.PROBLEM_STATEMENT,
            )
        )
    ).scalar_one()
    predecessor.status = ArtifactStatus.ACCEPTED
    old_version = ArtifactVersion(
        artifact_id=predecessor.id,
        version_number=1,
        title="Problem old",
        body="Old problem",
        status=VersionStatus.DRAFT,
        change_source=ChangeSource.MANUAL,
        extra_metadata={},
    )
    db_session.add(old_version)
    await db_session.flush()
    current_version = ArtifactVersion(
        artifact_id=predecessor.id,
        version_number=2,
        title="Problem current",
        body="Current problem",
        status=VersionStatus.DRAFT,
        change_source=ChangeSource.MANUAL,
        parent_version_id=old_version.id,
        extra_metadata={},
    )
    db_session.add(current_version)
    await db_session.flush()
    predecessor.current_version_id = current_version.id
    tc.input_snapshot = {
        **tc.input_snapshot,
        "lifecycle_metadata": {
            "based_on": {str(predecessor.id): str(old_version.id)},
            "decision_node_map": {},
        },
    }
    await db_session.flush()

    captured = {}
    original_resume_command = svc._resume_command

    def _capture_resume_command(session_row, resume, *, state_update=None):
        captured["state_update"] = state_update
        return original_resume_command(session_row, resume, state_update=state_update)

    svc._resume_command = _capture_resume_command
    _no_background_tasks.reset_mock()

    with pytest.raises(HTTPException) as exc:
        await svc.approve_tool_call(
            project_id=project_id,
            tool_call_id=tc.id,
            created_by_id=None,
            _llm_client=AsyncMock(),
        )

    assert exc.value.status_code == 409
    assert exc.value.detail["stale_predecessors"][0]["artifact_id"] == str(predecessor.id)
    await db_session.refresh(session)
    await db_session.refresh(tc)
    assert tc.status == AgentToolCallStatus.SUPERSEDED
    assert tc.created_version_id is None
    assert session.status == AgentSessionStatus.ACTIVE
    assert _no_background_tasks.call_count == 1
    rejection = captured["state_update"]["feedback_summary"]["lifecycle_persist_rejection"]
    assert rejection["stale_predecessors"] == [
        {
            "artifact_id": str(predecessor.id),
            "based_on_version_id": str(old_version.id),
            "current_version_id": str(current_version.id),
            "reason": "predecessor_version_changed",
        }
    ]
    versions = (
        await db_session.execute(select(ArtifactVersion).where(ArtifactVersion.agent_run_id == run.id))
    ).scalars().all()
    assert versions == []


@pytest.mark.asyncio
async def test_guard_lifecycle_predecessors_locks_in_canonical_sorted_order(client, db_session):
    """Deadlock-avoidance regression (plan F4): predecessor rows must be locked FOR UPDATE in a
    canonical artifact_id order. The guard appends stale predecessors in lock-acquisition order, so a
    sorted stale_predecessors output proves the lock order is canonical regardless of based_on order."""
    from app.models.artifact import ChangeSource, VersionStatus

    project_id = await _setup(client)
    svc = _make_service(db_session)

    based_on: dict[str, str] = {}
    for index in range(3):
        art = Artifact(
            project_id=project_id,
            type=ArtifactType.FUNCTIONAL_REQUIREMENT,
            status=ArtifactStatus.ACCEPTED,
            title=f"Pred {index}",
            extra_metadata={},
        )
        db_session.add(art)
        await db_session.flush()
        current = ArtifactVersion(
            artifact_id=art.id,
            version_number=1,
            title=f"Pred {index}",
            body="Body",
            status=VersionStatus.ACCEPTED,
            change_source=ChangeSource.MANUAL,
            extra_metadata={},
        )
        db_session.add(current)
        await db_session.flush()
        art.current_version_id = current.id
        # based_on points at a different (stale) version id, so every predecessor is stale.
        based_on[str(art.id)] = str(uuid.uuid4())
    await db_session.flush()

    # Feed based_on in reverse-sorted order to prove the guard reorders it canonically.
    reversed_based_on = dict(sorted(based_on.items(), key=lambda kv: kv[0], reverse=True))
    rejection = await svc._guard_lifecycle_predecessors(
        project_id, {"lifecycle_metadata": {"based_on": reversed_based_on}}
    )

    assert rejection is not None
    locked_order = [item["artifact_id"] for item in rejection["stale_predecessors"]]
    assert locked_order == sorted(based_on)


@pytest.mark.asyncio
async def test_approve_tool_call_recovers_incomplete_candidate_readiness_and_preserves_http_422(
    client, db_session, _no_background_tasks, caplog
):
    project_id = await _setup(client)
    svc = _make_service(db_session)
    session, run, tc = await _make_single_propose_session(db_session, project_id)
    tc.input_snapshot = {
        **tc.input_snapshot,
        "body": "## Vision\nIncrease retention.",
        "synthesis_metadata": {
            **tc.input_snapshot["synthesis_metadata"],
            "pending_assumptions": ["Target retention 15%"],
        },
        "candidate_readiness": {
            "state": "well_structured_but_incomplete",
            "can_persist": False,
            "missing": ["target"],
            "needs_confirmation": [],
            "inferred": [],
            "blocking_reasons": ["Missing target needing confirmation"],
        },
    }
    await db_session.flush()
    captured = {}
    original_resume_command = svc._resume_command

    def _capture_resume_command(session_row, resume, *, state_update=None):
        captured["state_update"] = state_update
        return original_resume_command(session_row, resume, state_update=state_update)

    svc._resume_command = _capture_resume_command
    _no_background_tasks.reset_mock()

    with caplog.at_level(logging.INFO, logger="app.graphs.gate_logging"):
        with pytest.raises(HTTPException) as exc:
            await svc.approve_tool_call(
                project_id=project_id,
                tool_call_id=tc.id,
                created_by_id=None,
                _llm_client=AsyncMock(),
            )

    assert exc.value.status_code == 422
    assert exc.value.detail["state"] == "poorly_structured"
    await db_session.refresh(session)
    await db_session.refresh(tc)
    focused = await db_session.get(Artifact, uuid.UUID(tc.input_snapshot["focused_artifact_id"]))
    assert tc.status == AgentToolCallStatus.SUPERSEDED
    assert tc.created_version_id is None
    assert session.status == AgentSessionStatus.ACTIVE
    assert focused.status == ArtifactStatus.DRAFT
    assert _no_background_tasks.call_count == 1
    feedback = captured["state_update"]["feedback_summary"]["candidate_readiness_rejection"]
    assert feedback["state"] == "poorly_structured"
    assert feedback["focused_artifact_id"] == str(focused.id)
    assert any(
        "gate=in_loop_feedback_recovery verdict=seeded" in record.getMessage()
        and "candidate_readiness_rejection" in record.getMessage()
        for record in caplog.records
    )
    versions = (
        await db_session.execute(select(ArtifactVersion).where(ArtifactVersion.agent_run_id == run.id))
    ).scalars().all()
    assert versions == []


@pytest.mark.asyncio
async def test_feedback_loop_many_edits_only_persists_final_ready_version(client, db_session, _no_background_tasks):
    project_id = await _setup(client)
    svc = _make_service(db_session)
    session, run, tc = await _make_single_propose_session(db_session, project_id)
    focused_artifact_id = uuid.UUID(tc.input_snapshot["focused_artifact_id"])

    for index in range(5):
        if index > 0:
            tc = AgentToolCall(
                run_id=run.id,
                tool_name="create_artifact",
                input_snapshot={
                    **tc.input_snapshot,
                    "candidate_readiness": {
                        "state": "well_structured_but_incomplete",
                        "can_persist": False,
                        "missing": [f"gap-{index}"],
                        "needs_confirmation": [],
                        "inferred": [],
                        "blocking_reasons": [f"Feedback nho {index} chua du readiness"],
                    },
                },
                status=AgentToolCallStatus.PROPOSED,
            )
            db_session.add(tc)
            await db_session.flush()
        await svc.request_edit(project_id=project_id, tool_call_id=tc.id, note=f"Chinh nho {index}")

    versions_before_approve = (
        await db_session.execute(select(ArtifactVersion).where(ArtifactVersion.artifact_id == focused_artifact_id))
    ).scalars().all()
    assert versions_before_approve == []

    final_tc = AgentToolCall(
        run_id=run.id,
        tool_name="create_artifact",
        input_snapshot={
            **tc.input_snapshot,
            "candidate_readiness": _sufficient_readiness(),
        },
        status=AgentToolCallStatus.PROPOSED,
    )
    db_session.add(final_tc)
    await db_session.flush()

    await svc.approve_tool_call(project_id=project_id, tool_call_id=final_tc.id, created_by_id=None)

    versions_after_approve = (
        await db_session.execute(select(ArtifactVersion).where(ArtifactVersion.artifact_id == focused_artifact_id))
    ).scalars().all()
    assert len(versions_after_approve) == 1
    assert final_tc.created_version_id == versions_after_approve[0].id


@pytest.mark.asyncio
async def test_approve_tool_call_rejects_unmarked_pending_assumptions(client, db_session):
    project_id = await _setup(client)
    svc = _make_service(db_session)
    session, run, tc = await _make_single_propose_session(db_session, project_id)
    tc.input_snapshot = {
        **tc.input_snapshot,
        "body": "\n\n".join(
            [
                "## Vision\nIncrease retention.",
                "## Objectives\n- Improve activation.",
                "## Success Metrics\n- Retention target 15%.",
            ]
        ),
        "synthesis_metadata": {
            **tc.input_snapshot["synthesis_metadata"],
            "confirmed_assumptions": ["Students use study groups weekly"],
            "pending_assumptions": ["Target retention 15%"],
        },
    }
    await db_session.flush()

    with pytest.raises(HTTPException) as exc_info:
        await svc.approve_tool_call(project_id=project_id, tool_call_id=tc.id, created_by_id=None)

    assert exc_info.value.status_code == 422
    assert "not ready enough to persist" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_approve_tool_call_persists_marked_confirmation_candidate_readiness(client, db_session):
    project_id = await _setup(client)
    svc = _make_service(db_session)
    session, run, tc = await _make_single_propose_session(db_session, project_id)
    tc.input_snapshot = {
        **tc.input_snapshot,
        "body": "\n\n".join(
            [
                "## Vision\nTang retention.",
                "## Objectives\n- Cai thien activation.",
                "## Success Metrics\n- Retention target 15% (agent-inferred, needs confirmation).",
            ]
        ),
        "synthesis_metadata": {
            **tc.input_snapshot["synthesis_metadata"],
            "pending_assumptions": ["Target retention 15%"],
        },
        "candidate_readiness": {
            "state": "needs_confirmation",
            "can_persist": True,
            "missing": [],
            "needs_confirmation": ["Target retention 15%"],
            "inferred": ["Retention target 15% (agent-inferred, needs confirmation)."],
            "blocking_reasons": [],
        },
    }
    await db_session.flush()

    await svc.approve_tool_call(project_id=project_id, tool_call_id=tc.id, created_by_id=None)

    versions = (
        await db_session.execute(select(ArtifactVersion).where(ArtifactVersion.agent_run_id == run.id))
    ).scalars().all()
    assert len(versions) == 1
    assert tc.created_version_id == versions[0].id


@pytest.mark.asyncio
async def test_approve_tool_call_recomputes_readiness_even_when_snapshot_claims_sufficient(client, db_session):
    project_id = await _setup(client)
    svc = _make_service(db_session)
    session, run, tc = await _make_single_propose_session(db_session, project_id)
    tc.input_snapshot = {
        **tc.input_snapshot,
        "body": "Content missing heading",
        "candidate_readiness": _sufficient_readiness(),
    }
    await db_session.flush()

    with pytest.raises(HTTPException) as exc:
        await svc.approve_tool_call(project_id=project_id, tool_call_id=tc.id, created_by_id=None)

    assert exc.value.status_code == 422
    assert exc.value.detail["state"] == "poorly_structured"
    versions = (
        await db_session.execute(select(ArtifactVersion).where(ArtifactVersion.agent_run_id == run.id))
    ).scalars().all()
    assert versions == []


@pytest.mark.asyncio
async def test_approve_tool_call_retry_does_not_create_duplicate_version(
    client, db_session, _no_background_tasks
):
    project_id = await _setup(client)
    svc = _make_service(db_session)
    session, run, tc = await _make_single_propose_session(db_session, project_id)
    focused_artifact_id = uuid.UUID(tc.input_snapshot["focused_artifact_id"])

    first = await svc.approve_tool_call(
        project_id=project_id, tool_call_id=tc.id, created_by_id=None
    )
    call_count_after_first = _no_background_tasks.call_count
    second = await svc.approve_tool_call(
        project_id=project_id, tool_call_id=tc.id, created_by_id=None
    )

    versions = (
        await db_session.execute(
            select(ArtifactVersion).where(ArtifactVersion.artifact_id == focused_artifact_id)
        )
    ).scalars().all()
    assert first.created_artifact_id == second.created_artifact_id == focused_artifact_id
    assert first.created_version_id == second.created_version_id
    assert len(versions) == 1
    assert _no_background_tasks.call_count == call_count_after_first


@pytest.mark.asyncio
async def test_approve_tool_call_reuses_existing_version_for_partial_retry(
    client, db_session, _no_background_tasks
):
    from app.models.artifact import ChangeSource, VersionStatus

    project_id = await _setup(client)
    svc = _make_service(db_session)
    session, run, tc = await _make_single_propose_session(db_session, project_id)
    focused_artifact_id = uuid.UUID(tc.input_snapshot["focused_artifact_id"])
    existing = ArtifactVersion(
        artifact_id=focused_artifact_id,
        version_number=1,
        title="Goal",
        body="Mo ta",
        status=VersionStatus.DRAFT,
        change_source=ChangeSource.AI_GENERATION,
        agent_run_id=run.id,
        tool_call_id=tc.id,
        extra_metadata={},
    )
    db_session.add(existing)
    await db_session.flush()

    updated = await svc.approve_tool_call(
        project_id=project_id, tool_call_id=tc.id, created_by_id=None
    )

    versions = (
        await db_session.execute(
            select(ArtifactVersion).where(ArtifactVersion.artifact_id == focused_artifact_id)
        )
    ).scalars().all()
    focused = await db_session.get(Artifact, focused_artifact_id)
    assert updated.created_version_id == existing.id
    assert focused.current_version_id == existing.id
    assert focused.status == ArtifactStatus.ACCEPTED
    assert len(versions) == 1


@pytest.mark.asyncio
async def test_approve_last_required_child_accepts_parent_container(client, db_session):
    from app.documents.registry import children_of, get_config
    from app.models.artifact import ChangeSource, VersionStatus

    project_id = await _setup(client)
    svc = _make_service(db_session)
    parent = Artifact(
        project_id=project_id,
        type=ArtifactType.BRD,
        status=ArtifactStatus.DRAFT,
        title="BRD",
        extra_metadata={},
    )
    db_session.add(parent)
    await db_session.flush()
    children = []
    for item_type in children_of("brd"):
        child = Artifact(
            project_id=project_id,
            parent_id=parent.id,
            type=ArtifactType(item_type),
            status=ArtifactStatus.DRAFT,
            title=get_config(item_type).label,
            extra_metadata={},
        )
        children.append(child)
    db_session.add_all(children)
    await db_session.flush()
    for child in children[:-1]:
        version = ArtifactVersion(
            artifact_id=child.id,
            version_number=1,
            title=child.title,
            body=f"{child.type.value} body",
            status=VersionStatus.DRAFT,
            change_source=ChangeSource.MANUAL,
            extra_metadata={},
        )
        db_session.add(version)
        await db_session.flush()
        child.current_version_id = version.id

    session = AgentSession(
        project_id=project_id, artifact_type=children[-1].type.value, workflow_area="analysis",
        graph_checkpoint={}, status=AgentSessionStatus.WAITING_FOR_HUMAN,
        interrupt_type=AgentSessionInterruptType.PROPOSE_ARTIFACTS,
        focused_artifact_id=children[-1].id,
    )
    db_session.add(session)
    await db_session.flush()
    run = AgentRun(session_id=session.id, analysis_result={})
    db_session.add(run)
    await db_session.flush()
    tc = AgentToolCall(
        run_id=run.id,
        tool_name="create_artifact",
        input_snapshot={
            "artifact_type": children[-1].type.value,
            "title": children[-1].title,
            "body": _risks_body(),
            "focused_artifact_id": str(children[-1].id),
            "candidate_readiness": _sufficient_readiness(),
        },
        status=AgentToolCallStatus.PROPOSED,
    )
    db_session.add(tc)
    await db_session.flush()

    await svc.approve_tool_call(project_id=project_id, tool_call_id=tc.id, created_by_id=None)

    await db_session.refresh(parent)
    assert parent.status == ArtifactStatus.ACCEPTED


@pytest.mark.asyncio
async def test_approve_tool_call_rejects_missing_focused_artifact(client, db_session):
    project_id = await _setup(client)
    svc = _make_service(db_session)
    session, run, tc = await _make_single_propose_session(db_session, project_id)
    tc.input_snapshot = {"artifact_type": "unknown_type", "title": "Goal", "body": "Mo ta"}
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
        "body": _vision_body(),
        "focused_artifact_id": tc1.input_snapshot["focused_artifact_id"],
        "candidate_readiness": _sufficient_readiness(),
    }
    tc2.input_snapshot = {
        "artifact_type": "problem_statement",
        "title": "Problem",
        "body": _problem_body(),
        "focused_artifact_id": tc2.input_snapshot["focused_artifact_id"],
        "candidate_readiness": _sufficient_readiness(),
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
        "vision_objectives": _vision_body(),
        "problem_statement": _problem_body(),
    }


@pytest.mark.asyncio
async def test_approve_create_artifact_link_tool_call_executes_after_approval(client, db_session):
    project_id, user_id = await _setup_with_user(client)
    svc = _make_service(db_session)
    source = Artifact(
        project_id=project_id,
        type=ArtifactType.FUNCTIONAL_REQUIREMENT,
        status=ArtifactStatus.DRAFT,
        title="Source",
        created_by_id=user_id,
    )
    target = Artifact(
        project_id=project_id,
        type=ArtifactType.EPIC,
        status=ArtifactStatus.DRAFT,
        title="Target",
        created_by_id=user_id,
    )
    db_session.add_all([source, target])
    await db_session.flush()
    _session, _run, tc = await _make_public_tool_call_session(
        db_session,
        project_id,
        user_id,
        f"create_artifact_link:{source.id}:{target.id}:derives_from",
        {
            "source_artifact_id": str(source.id),
            "target_artifact_id": str(target.id),
            "relation_type": "derives_from",
            "metadata": {"reason": "trace"},
        },
    )

    updated = await svc.approve_tool_call(project_id=project_id, tool_call_id=tc.id, created_by_id=user_id)

    link = (await db_session.execute(select(ArtifactLink).where(ArtifactLink.project_id == project_id))).scalar_one()
    assert updated.status == AgentToolCallStatus.EXECUTED
    assert updated.input_snapshot["created_link_id"] == str(link.id)
    assert link.source_artifact_id == source.id
    assert link.target_artifact_id == target.id
    assert link.relation_type == RelationType.DERIVES_FROM


@pytest.mark.asyncio
async def test_approve_propose_retirement_archives_leaf_and_records_superseded_by(client, db_session):
    project_id, user_id = await _setup_with_user(client)
    svc = _make_service(db_session)
    retired = Artifact(
        project_id=project_id,
        type=ArtifactType.EPIC,
        status=ArtifactStatus.ACCEPTED,
        title="Old epic",
        created_by_id=user_id,
    )
    replacement = Artifact(
        project_id=project_id,
        type=ArtifactType.EPIC,
        status=ArtifactStatus.ACCEPTED,
        title="New epic",
        created_by_id=user_id,
    )
    db_session.add_all([retired, replacement])
    await db_session.flush()
    version = ArtifactVersion(
        artifact_id=retired.id,
        version_number=1,
        title="Old epic",
        body="Old body",
        status=VersionStatus.ACCEPTED,
        change_source=ChangeSource.MANUAL,
        created_by_id=user_id,
        extra_metadata={},
    )
    db_session.add(version)
    await db_session.flush()
    retired.current_version_id = version.id
    await db_session.flush()
    _session, _run, tc = await _make_public_tool_call_session(
        db_session,
        project_id,
        user_id,
        f"propose_retirement:{retired.id}",
        {
            "artifact_id": str(retired.id),
            "reason": "Superseded by a cleaner epic",
            "superseded_by_artifact_id": str(replacement.id),
        },
    )

    updated = await svc.approve_tool_call(project_id=project_id, tool_call_id=tc.id, created_by_id=user_id)

    await db_session.refresh(retired)
    await db_session.refresh(version)
    assert updated.status == AgentToolCallStatus.EXECUTED
    assert updated.created_artifact_id == retired.id
    assert retired.status == ArtifactStatus.ARCHIVED
    assert retired.extra_metadata["superseded_by"] == str(replacement.id)
    assert retired.extra_metadata["retirement"]["source"] == "agent_retirement"
    assert version.status == VersionStatus.ARCHIVED


@pytest.mark.asyncio
async def test_approve_propose_retirement_blocks_live_downstream_link(client, db_session):
    project_id, user_id = await _setup_with_user(client)
    svc = _make_service(db_session)
    dependent = Artifact(
        project_id=project_id,
        type=ArtifactType.FUNCTIONAL_REQUIREMENT,
        status=ArtifactStatus.ACCEPTED,
        title="Dependent",
        created_by_id=user_id,
    )
    retired = Artifact(
        project_id=project_id,
        type=ArtifactType.EPIC,
        status=ArtifactStatus.ACCEPTED,
        title="Retired",
        created_by_id=user_id,
    )
    db_session.add_all([dependent, retired])
    await db_session.flush()
    db_session.add(
        ArtifactLink(
            project_id=project_id,
            source_artifact_id=dependent.id,
            target_artifact_id=retired.id,
            relation_type=RelationType.SUPPORTS,
        )
    )
    await db_session.flush()
    _session, _run, tc = await _make_public_tool_call_session(
        db_session,
        project_id,
        user_id,
        f"propose_retirement:{retired.id}",
        {"artifact_id": str(retired.id), "reason": "No longer valid"},
    )

    with pytest.raises(HTTPException) as exc:
        await svc.approve_tool_call(project_id=project_id, tool_call_id=tc.id, created_by_id=user_id)

    assert exc.value.status_code == 409
    assert str(dependent.id) in exc.value.detail["artifact_ids"]
    await db_session.refresh(tc)
    await db_session.refresh(retired)
    assert tc.status == AgentToolCallStatus.PROPOSED
    assert retired.status == ArtifactStatus.ACCEPTED


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


@pytest.mark.asyncio
async def test_reject_tool_call_retry_is_idempotent(client, db_session, _no_background_tasks):
    project_id = await _setup(client)
    svc = _make_service(db_session)

    _, _, tc = await _make_single_propose_session(db_session, project_id)

    first = await svc.reject_tool_call(project_id=project_id, tool_call_id=tc.id)
    call_count_after_first = _no_background_tasks.call_count
    second = await svc.reject_tool_call(project_id=project_id, tool_call_id=tc.id)

    assert first.id == second.id == tc.id
    assert second.status == AgentToolCallStatus.REJECTED
    assert second.created_artifact_id is None
    assert second.created_version_id is None
    assert _no_background_tasks.call_count == call_count_after_first


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

    await svc.request_edit(project_id=project_id, tool_call_id=tc.id, note="Can chinh sua")

    updated = (await db_session.execute(select(AgentToolCall).where(AgentToolCall.id == tc.id))).scalar_one()
    assert updated.status == AgentToolCallStatus.SUPERSEDED
    _no_background_tasks.assert_called_once()


@pytest.mark.asyncio
async def test_request_edit_does_not_resume_when_others_still_proposed(client, db_session, _no_background_tasks):
    """request_edit on one of many proposed tool calls must NOT trigger graph resume."""
    project_id = await _setup(client)
    svc = _make_service(db_session)

    _, _, tc1, _ = await _make_propose_session(db_session, project_id)

    await svc.request_edit(project_id=project_id, tool_call_id=tc1.id, note="Can chinh sua")

    _no_background_tasks.assert_not_called()


@pytest.mark.asyncio
async def test_request_edit_recovers_stale_base_version_in_loop(client, db_session, _no_background_tasks):
    """request_edit resumes the graph, so a stale base is pulled into the loop (seeded into
    feedback_summary for the resumed turn to re-read + rebase) instead of a terminal 409."""
    from unittest.mock import AsyncMock

    from app.models.artifact import ChangeSource, VersionStatus

    project_id = await _setup(client)
    svc = _make_service(db_session)
    session, run, tc = await _make_single_propose_session(db_session, project_id)
    focused = await db_session.get(Artifact, uuid.UUID(tc.input_snapshot["focused_artifact_id"]))
    old_version = ArtifactVersion(
        artifact_id=focused.id,
        version_number=1,
        title="V1",
        body="Body 1",
        status=VersionStatus.DRAFT,
        change_source=ChangeSource.MANUAL,
        extra_metadata={},
    )
    current_version = ArtifactVersion(
        artifact_id=focused.id,
        version_number=2,
        title="V2",
        body="Body 2",
        status=VersionStatus.DRAFT,
        change_source=ChangeSource.MANUAL,
        parent_version_id=old_version.id,
        extra_metadata={},
    )
    db_session.add_all([old_version, current_version])
    await db_session.flush()
    focused.current_version_id = current_version.id
    tc.input_snapshot = {**tc.input_snapshot, "base_version_id": str(old_version.id)}
    await db_session.flush()

    svc._check_and_resume = AsyncMock()
    await svc.request_edit(project_id=project_id, tool_call_id=tc.id, note="Sua lai")

    await db_session.refresh(tc)
    assert tc.status == AgentToolCallStatus.SUPERSEDED
    svc._check_and_resume.assert_awaited_once()
    seed = svc._check_and_resume.await_args.kwargs["state_update"]
    stale = seed["feedback_summary"]["stale_base_version"]
    assert stale["current_version_id"] == str(current_version.id)
    assert stale["artifact_id"] == str(focused.id)


@pytest.mark.asyncio
async def test_request_edit_without_stale_seeds_no_state_update(client, db_session, _no_background_tasks):
    """Non-stale request_edit preserves the legacy resume: no stale seed threaded to the resume."""
    from unittest.mock import AsyncMock

    project_id = await _setup(client)
    svc = _make_service(db_session)
    session, run, tc = await _make_single_propose_session(db_session, project_id)

    svc._check_and_resume = AsyncMock()
    await svc.request_edit(project_id=project_id, tool_call_id=tc.id, note="Sua lai")

    svc._check_and_resume.assert_awaited_once()
    assert svc._check_and_resume.await_args.kwargs["state_update"] is None


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
            "title": "Goal",
            "body": _vision_body(),
            "focused_artifact_id": str(focused.id),
            "synthesis_metadata": _synthesis_metadata("vision_objectives", focused.id),
            "candidate_readiness": _sufficient_readiness(),
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
            "title": "Goal A",
            "body": _vision_body(),
            "focused_artifact_id": str(focused_a.id),
            "synthesis_metadata": _synthesis_metadata("vision_objectives", focused_a.id),
            "candidate_readiness": _sufficient_readiness(),
        },
        status=AgentToolCallStatus.PROPOSED,
    )
    tc2 = AgentToolCall(
        run_id=run.id, tool_name="create_artifact",
        input_snapshot={
            "artifact_type": "problem_statement",
            "title": "Goal B",
            "body": _problem_body(),
            "focused_artifact_id": str(focused_b.id),
            "synthesis_metadata": _synthesis_metadata("problem_statement", focused_b.id),
            "candidate_readiness": _sufficient_readiness(),
        },
        status=AgentToolCallStatus.PROPOSED,
    )
    db_session.add(tc1)
    db_session.add(tc2)
    await db_session.flush()
    return session, run, tc1, tc2


async def _make_public_tool_call_session(db_session, project_id, user_id, tool_name, input_snapshot):
    session = AgentSession(
        project_id=project_id,
        artifact_type="epic",
        workflow_area="analysis",
        graph_checkpoint={},
        status=AgentSessionStatus.WAITING_FOR_HUMAN,
        interrupt_type=AgentSessionInterruptType.PROPOSE_ARTIFACTS,
        created_by_id=user_id,
    )
    db_session.add(session)
    await db_session.flush()
    run = AgentRun(session_id=session.id, analysis_result={})
    db_session.add(run)
    await db_session.flush()
    tc = AgentToolCall(
        run_id=run.id,
        tool_name=tool_name,
        input_snapshot=input_snapshot,
        status=AgentToolCallStatus.PROPOSED,
    )
    db_session.add(tc)
    await db_session.flush()
    return session, run, tc


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


def _vision_body():
    return "\n\n".join(
        [
            "## Vision\nTang retention.",
            "## Objectives\n- Cai thien activation.",
            "## Success Metrics\n- Retention target 15%.",
        ]
    )


def _problem_body():
    return "\n\n".join(
        [
            "## Problem Statement\nUsers do not see value early.",
            "## Affected Users\nNew users.",
            "## Impact\nRetention thap.",
            "## Root Cause / Contributing Factors\nOnboarding lacks guidance.",
        ]
    )


def _risks_body():
    # constraints_assumptions is the last BRD child and absorbed risks_issues, so its
    # contract now requires the constraints headings plus ## Risks / ## Mitigation Plan.
    return "\n\n".join(
        [
            "## Constraints\n- Must integrate with existing infrastructure.",
            "## Assumptions\n- Current auth provider stays.",
            "## Validation Plan\n- Confirm provider SLA before build.",
            "## Risks\n- Low adoption risk.",
            "## Mitigation Plan\n- Measure baseline and test new onboarding.",
        ]
    )


def _synthesis_metadata(artifact_type, focused_artifact_id):
    return {
        "artifact_type": artifact_type,
        "focused_artifact_id": str(focused_artifact_id),
        "base_version_id": None,
        "evidence_refs": [],
        "inference_level": "medium",
        "confirmed_assumptions": [],
        "pending_assumptions": [],
        "synthesis_source": "bmad_synthesis",
    }


def _sufficient_readiness():
    return {
        "state": "sufficient",
        "can_persist": True,
        "missing": [],
        "needs_confirmation": [],
        "inferred": [],
        "blocking_reasons": [],
    }


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

    await svc.handle_user_message(project_id=project_id, session_id=session.id, content="Hello")

    _no_background_tasks.assert_called_once()
    scheduled = _no_background_tasks.call_args.args[0]
    assert scheduled.cr_code.co_name == "_run_graph"


# ---------------------------------------------------------------------------
# _document_type_for_session — container detection is registry-driven, not a
# hardcoded {"brd", "prd", "sad"} set. event_storming must resolve the same
# way brd/prd/sad already do.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_document_type_for_session_resolves_event_storming_via_artifact_type(client, db_session):
    project_id = await _setup(client)
    svc = _make_service(db_session)
    session = AgentSession(
        project_id=project_id,
        artifact_type="event_storming",
        workflow_area="analysis",
        graph_checkpoint={},
    )
    db_session.add(session)
    await db_session.flush()

    document_type = await svc._document_type_for_session(session)

    assert document_type == "event_storming"


@pytest.mark.asyncio
async def test_document_type_for_session_resolves_event_storming_via_focused_container(client, db_session):
    project_id = await _setup(client)
    svc = _make_service(db_session)
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

    document_type = await svc._document_type_for_session(session)

    assert document_type == "event_storming"


@pytest.mark.asyncio
@pytest.mark.parametrize("container_type", ["brd", "prd", "sad"])
async def test_document_type_for_session_regression_via_artifact_type(client, db_session, container_type):
    project_id = await _setup(client)
    svc = _make_service(db_session)
    session = AgentSession(
        project_id=project_id,
        artifact_type=container_type,
        workflow_area="analysis",
        graph_checkpoint={},
    )
    db_session.add(session)
    await db_session.flush()

    document_type = await svc._document_type_for_session(session)

    assert document_type == container_type


@pytest.mark.asyncio
@pytest.mark.parametrize("container_type", ["brd", "prd", "sad"])
async def test_document_type_for_session_regression_via_focused_container(client, db_session, container_type):
    project_id = await _setup(client)
    svc = _make_service(db_session)
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

    document_type = await svc._document_type_for_session(session)

    assert document_type == container_type
