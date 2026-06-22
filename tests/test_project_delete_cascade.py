"""Regression test for delete project: a project with full child data can be deleted.

Enable foreign key enforcement on SQLite (off by default) to mimic PostgreSQL — this lets the test
catch delete-ordering bugs that cause FK violations, not just missing-table bugs.
"""

import uuid

import pytest
import pytest_asyncio
from sqlalchemy import event, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.agent import AgentRun, AgentSession, AgentSessionStatus, AgentToolCall, AgentToolCallStatus
from app.models.artifact import (
    Artifact,
    ArtifactEvidence,
    ArtifactLink,
    ArtifactReview,
    ArtifactStatus,
    ArtifactType,
    ArtifactVersion,
    ChangeSource,
    EvidenceSourceType,
    RelationType,
    ReviewStatus,
    SourceDocument,
    SourceType,
    VersionStatus,
)
from app.models.base import Base
from app.models.organization import Organization
from app.models.project import Project
from app.models.user import User
from app.services.project_service import ProjectService


@pytest_asyncio.fixture
async def fk_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", connect_args={"check_same_thread": False})

    @event.listens_for(engine.sync_engine, "connect")
    def _enable_fk(dbapi_conn, _record):
        dbapi_conn.execute("PRAGMA foreign_keys=ON")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session
    await engine.dispose()


async def _seed_full_project(session: AsyncSession, org_id: uuid.UUID, user_id: uuid.UUID) -> uuid.UUID:
    project = Project(org_id=org_id, name="Full", slug=f"full-{uuid.uuid4().hex[:6]}")
    session.add(project)
    await session.flush()

    doc = SourceDocument(project_id=project.id, title="Doc", source_type=SourceType.TEXT_PASTE, extra_metadata={})
    session.add(doc)

    artifact_a = Artifact(project_id=project.id, type=ArtifactType.GOAL, status=ArtifactStatus.ACCEPTED, title="A", extra_metadata={})
    artifact_b = Artifact(project_id=project.id, type=ArtifactType.PROBLEM, status=ArtifactStatus.DRAFT, title="B", extra_metadata={})
    session.add_all([artifact_a, artifact_b])
    await session.flush()

    session_row = AgentSession(project_id=project.id, artifact_type="goal", workflow_area="brd", status=AgentSessionStatus.COMPLETED, graph_checkpoint={})
    session.add(session_row)
    await session.flush()
    run = AgentRun(session_id=session_row.id, analysis_result={})
    session.add(run)
    await session.flush()
    tool_call = AgentToolCall(run_id=run.id, tool_name="create_artifact", input_snapshot={}, status=AgentToolCallStatus.EXECUTED)
    session.add(tool_call)
    await session.flush()

    v1 = ArtifactVersion(
        artifact_id=artifact_a.id, version_number=1, title="A", body="v1",
        status=VersionStatus.DRAFT, change_source=ChangeSource.AI_GENERATION,
        agent_run_id=run.id, tool_call_id=tool_call.id, source_document_id=doc.id, extra_metadata={},
    )
    session.add(v1)
    await session.flush()
    v2 = ArtifactVersion(
        artifact_id=artifact_a.id, version_number=2, title="A", body="v2",
        status=VersionStatus.DRAFT, change_source=ChangeSource.MANUAL,
        parent_version_id=v1.id, extra_metadata={},
    )
    session.add(v2)
    await session.flush()

    artifact_a.current_version_id = v2.id
    tool_call.created_artifact_id = artifact_a.id
    tool_call.created_version_id = v1.id

    session.add(ArtifactLink(
        project_id=project.id, source_artifact_id=artifact_a.id, target_artifact_id=artifact_b.id,
        relation_type=RelationType.DERIVES_FROM, extra_metadata={},
    ))
    session.add(ArtifactEvidence(
        artifact_id=artifact_a.id, artifact_version_id=v1.id, source_document_id=doc.id,
        source_type=EvidenceSourceType.DOCUMENT, locator="p1", extra_metadata={},
    ))
    session.add(ArtifactReview(
        artifact_id=artifact_a.id, artifact_version_id=v1.id, review_status=ReviewStatus.APPROVED,
    ))
    await session.flush()
    return project.id


@pytest.mark.asyncio
async def test_delete_project_cascades_full_subtree(fk_session):
    user = User(email=f"u-{uuid.uuid4().hex[:8]}@x.com", hashed_password="x", full_name="U", is_active=True)
    fk_session.add(user)
    await fk_session.flush()
    org = Organization(name="Org", slug=f"org-{uuid.uuid4().hex[:6]}", owner_id=user.id)
    fk_session.add(org)
    await fk_session.flush()

    target_id = await _seed_full_project(fk_session, org.id, user.id)
    sibling_id = await _seed_full_project(fk_session, org.id, user.id)
    await fk_session.flush()

    await ProjectService(fk_session).delete(org.id, target_id)
    await fk_session.flush()

    async def count(model, project_col) -> int:
        return (await fk_session.execute(select(func.count()).select_from(model).where(project_col == target_id))).scalar()

    assert (await fk_session.execute(select(func.count()).select_from(Project).where(Project.id == target_id))).scalar() == 0
    assert await count(Artifact, Artifact.project_id) == 0
    assert await count(ArtifactLink, ArtifactLink.project_id) == 0
    assert await count(SourceDocument, SourceDocument.project_id) == 0
    assert await count(AgentSession, AgentSession.project_id) == 0

    # The sibling project is unaffected.
    assert (await fk_session.execute(select(func.count()).select_from(Project).where(Project.id == sibling_id))).scalar() == 1
    assert (await fk_session.execute(select(func.count()).select_from(Artifact).where(Artifact.project_id == sibling_id))).scalar() == 2
