"""Dormant command-boundary mechanism on
`_execute_create_artifact_link`/`_execute_retirement`.

This mechanism is not reachable from `approve_tool_call` (the only production caller) today —
the REST approval endpoint threads no turn context yet (threading it end-to-end is deliberately
excluded as riskier, separate work). These tests exercise the mechanism the only
way it is currently reachable: direct keyword-argument injection into the private execution
methods, proving the fence/duplicate/ledger machinery itself is correct and ready to be wired up
by a future increment.
"""

import uuid

import pytest
from sqlalchemy import select

from app.models.agent import (
    AgentTurnEnvelope,
    DraftCommandLedger,
    TurnExecutionState,
    TurnExecutionStatus,
)
from app.models.artifact import Artifact, ArtifactStatus, ArtifactType, ArtifactVersion, ChangeSource, VersionStatus
from app.models.user import User
from tests.helpers import create_org, create_project, make_auth_headers


async def _setup_with_user(client):
    from app.core.security import decode_token

    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    token = headers["Authorization"].removeprefix("Bearer ")
    return uuid.UUID(project["id"]), uuid.UUID(decode_token(token)["sub"])


def _make_service(db_session):
    from app.services.agent_service import AgentService

    return AgentService(db_session, graph=None, session_factory=None)


async def _seed_turn(db_session, *, owner_id: str = "owner-a", generation: int = 1):
    user = User(email=f"link-cmd-turn-{uuid.uuid4()}@example.com", hashed_password="hash")
    db_session.add(user)
    await db_session.flush()
    envelope = AgentTurnEnvelope(
        session_id=uuid.uuid4(),
        session_sequence=1,
        original_trigger_id=uuid.uuid4(),
        actor_id=user.id,
        cohort={"command_handlers_enabled": True},
        correlation_id=str(uuid.uuid4()),
    )
    db_session.add(envelope)
    await db_session.flush()
    db_session.add(
        TurnExecutionState(
            turn_id=envelope.id,
            status=TurnExecutionStatus.RUNNING,
            owner_id=owner_id,
            ownership_generation=generation,
        )
    )
    await db_session.commit()
    return envelope


@pytest.mark.asyncio
async def test_create_artifact_link_records_ledger_row_when_turn_context_present(client, db_session):
    project_id, user_id = await _setup_with_user(client)
    svc = _make_service(db_session)
    source = Artifact(
        project_id=project_id, type=ArtifactType.FUNCTIONAL_REQUIREMENT, status=ArtifactStatus.DRAFT,
        title="Source", created_by_id=user_id,
    )
    target = Artifact(
        project_id=project_id, type=ArtifactType.EPIC, status=ArtifactStatus.DRAFT,
        title="Target", created_by_id=user_id,
    )
    db_session.add_all([source, target])
    await db_session.flush()
    envelope = await _seed_turn(db_session)

    link = await svc._execute_create_artifact_link(
        project_id=project_id,
        session_id=uuid.uuid4(),
        snapshot={
            "source_artifact_id": str(source.id),
            "target_artifact_id": str(target.id),
            "relation_type": "derives_from",
        },
        created_by_id=user_id,
        turn_id=envelope.id,
        turn_owner_id="owner-a",
        turn_ownership_generation=1,
    )

    ledger_rows = (
        await db_session.execute(
            select(DraftCommandLedger).where(DraftCommandLedger.turn_id == envelope.id)
        )
    ).scalars().all()
    assert len(ledger_rows) == 1
    assert ledger_rows[0].action_type == "create_artifact_link"
    assert ledger_rows[0].artifact_id == link.id


@pytest.mark.asyncio
async def test_create_artifact_link_duplicate_retry_reuses_committed_link(client, db_session):
    project_id, user_id = await _setup_with_user(client)
    svc = _make_service(db_session)
    source = Artifact(
        project_id=project_id, type=ArtifactType.FUNCTIONAL_REQUIREMENT, status=ArtifactStatus.DRAFT,
        title="Source", created_by_id=user_id,
    )
    target = Artifact(
        project_id=project_id, type=ArtifactType.EPIC, status=ArtifactStatus.DRAFT,
        title="Target", created_by_id=user_id,
    )
    db_session.add_all([source, target])
    await db_session.flush()
    envelope = await _seed_turn(db_session)
    snapshot = {
        "source_artifact_id": str(source.id),
        "target_artifact_id": str(target.id),
        "relation_type": "derives_from",
    }
    kwargs = dict(
        project_id=project_id,
        session_id=uuid.uuid4(),
        snapshot=snapshot,
        created_by_id=user_id,
        turn_id=envelope.id,
        turn_owner_id="owner-a",
        turn_ownership_generation=1,
    )

    first = await svc._execute_create_artifact_link(**kwargs)
    second = await svc._execute_create_artifact_link(**kwargs)

    assert first.id == second.id
    ledger_rows = (
        await db_session.execute(
            select(DraftCommandLedger).where(DraftCommandLedger.turn_id == envelope.id)
        )
    ).scalars().all()
    assert len(ledger_rows) == 1


@pytest.mark.asyncio
async def test_create_artifact_link_stale_fence_is_rejected(client, db_session):
    project_id, user_id = await _setup_with_user(client)
    svc = _make_service(db_session)
    source = Artifact(
        project_id=project_id, type=ArtifactType.FUNCTIONAL_REQUIREMENT, status=ArtifactStatus.DRAFT,
        title="Source", created_by_id=user_id,
    )
    target = Artifact(
        project_id=project_id, type=ArtifactType.EPIC, status=ArtifactStatus.DRAFT,
        title="Target", created_by_id=user_id,
    )
    db_session.add_all([source, target])
    await db_session.flush()
    envelope = await _seed_turn(db_session, owner_id="owner-a", generation=1)

    turn_state = (
        await db_session.execute(select(TurnExecutionState).where(TurnExecutionState.turn_id == envelope.id))
    ).scalar_one()
    turn_state.owner_id = "owner-b"
    turn_state.ownership_generation = 2
    await db_session.commit()

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        await svc._execute_create_artifact_link(
            project_id=project_id,
            session_id=uuid.uuid4(),
            snapshot={
                "source_artifact_id": str(source.id),
                "target_artifact_id": str(target.id),
                "relation_type": "derives_from",
            },
            created_by_id=user_id,
            turn_id=envelope.id,
            turn_owner_id="owner-a",
            turn_ownership_generation=1,
        )
    assert exc_info.value.status_code == 409
    ledger_rows = (
        await db_session.execute(
            select(DraftCommandLedger).where(DraftCommandLedger.turn_id == envelope.id)
        )
    ).scalars().all()
    assert ledger_rows == []


@pytest.mark.asyncio
async def test_execute_retirement_records_ledger_row_when_turn_context_present(client, db_session):
    project_id, user_id = await _setup_with_user(client)
    svc = _make_service(db_session)
    retired = Artifact(
        project_id=project_id, type=ArtifactType.EPIC, status=ArtifactStatus.ACCEPTED,
        title="Old epic", created_by_id=user_id,
    )
    db_session.add(retired)
    await db_session.flush()
    version = ArtifactVersion(
        artifact_id=retired.id, version_number=1, title="Old epic", body="Old body",
        status=VersionStatus.ACCEPTED, change_source=ChangeSource.MANUAL, created_by_id=user_id,
        extra_metadata={},
    )
    db_session.add(version)
    await db_session.flush()
    retired.current_version_id = version.id
    await db_session.flush()
    envelope = await _seed_turn(db_session)

    archived = await svc._execute_retirement(
        project_id=project_id,
        session_id=uuid.uuid4(),
        snapshot={"artifact_id": str(retired.id), "reason": "Superseded"},
        created_by_id=user_id,
        turn_id=envelope.id,
        turn_owner_id="owner-a",
        turn_ownership_generation=1,
    )

    ledger_rows = (
        await db_session.execute(
            select(DraftCommandLedger).where(DraftCommandLedger.turn_id == envelope.id)
        )
    ).scalars().all()
    assert len(ledger_rows) == 1
    assert ledger_rows[0].action_type == "propose_retirement"
    assert ledger_rows[0].artifact_id == archived.id


@pytest.mark.asyncio
async def test_execute_retirement_no_turn_context_uses_fully_legacy_path(client, db_session):
    """Default behavior — no kwargs passed — is byte-identical to the pre-increment code path
    (this is the shape `approve_tool_call` actually calls today)."""
    project_id, user_id = await _setup_with_user(client)
    svc = _make_service(db_session)
    retired = Artifact(
        project_id=project_id, type=ArtifactType.EPIC, status=ArtifactStatus.ACCEPTED,
        title="Old epic", created_by_id=user_id,
    )
    db_session.add(retired)
    await db_session.flush()
    version = ArtifactVersion(
        artifact_id=retired.id, version_number=1, title="Old epic", body="Old body",
        status=VersionStatus.ACCEPTED, change_source=ChangeSource.MANUAL, created_by_id=user_id,
        extra_metadata={},
    )
    db_session.add(version)
    await db_session.flush()
    retired.current_version_id = version.id
    await db_session.flush()

    ledger_count_before = len((await db_session.execute(select(DraftCommandLedger))).scalars().all())

    archived = await svc._execute_retirement(
        project_id=project_id,
        session_id=uuid.uuid4(),
        snapshot={"artifact_id": str(retired.id), "reason": "Superseded"},
        created_by_id=user_id,
    )

    assert archived.status == ArtifactStatus.ARCHIVED
    ledger_count_after = len((await db_session.execute(select(DraftCommandLedger))).scalars().all())
    assert ledger_count_after == ledger_count_before
