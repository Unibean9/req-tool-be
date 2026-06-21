import uuid

import pytest

from app.models import Project, User
from app.models.artifact import Artifact, ArtifactPriority, ArtifactStatus, ArtifactType, RelationType
from app.models.organization import Organization, OrgMember
from app.services.artifact_service import ArtifactService


@pytest.mark.asyncio
async def test_create_link_rejects_self_link(db_session):
    user, project, _other_project = await _seed_link_projects(db_session)
    source = Artifact(
        project_id=project.id,
        type=ArtifactType.GOAL,
        status=ArtifactStatus.DRAFT,
        priority=ArtifactPriority.MUST,
        title="Nguon",
    )
    db_session.add(source)
    await db_session.flush()

    service = ArtifactService(db_session)

    with pytest.raises(ValueError, match="không được liên kết chính nó"):
        await service.create_link(
            project_id=project.id,
            source_artifact_id=source.id,
            target_artifact_id=source.id,
            relation_type=RelationType.DERIVES_FROM,
            created_by_id=user.id,
        )


@pytest.mark.asyncio
async def test_create_link_rejects_cross_project_target(db_session):
    user, project, other_project = await _seed_link_projects(db_session)
    source = Artifact(
        project_id=project.id,
        type=ArtifactType.GOAL,
        status=ArtifactStatus.DRAFT,
        priority=ArtifactPriority.MUST,
        title="Nguon",
    )
    other_project_target = Artifact(
        project_id=other_project.id,
        type=ArtifactType.CAPABILITY,
        status=ArtifactStatus.DRAFT,
        priority=ArtifactPriority.SHOULD,
        title="Dich sai project",
    )
    db_session.add_all([source, other_project_target])
    await db_session.flush()

    service = ArtifactService(db_session)

    with pytest.raises(ValueError, match="cùng một dự án"):
        await service.create_link(
            project_id=project.id,
            source_artifact_id=source.id,
            target_artifact_id=other_project_target.id,
            relation_type=RelationType.DERIVES_FROM,
            created_by_id=user.id,
        )


@pytest.mark.asyncio
async def test_create_link_rejects_non_project_member(db_session):
    _user, project, _other_project = await _seed_link_projects(db_session)
    outsider = User(email=f"outsider-{uuid.uuid4()}@example.com", hashed_password="x", full_name="Outsider")
    source = Artifact(
        project_id=project.id,
        type=ArtifactType.GOAL,
        status=ArtifactStatus.DRAFT,
        priority=ArtifactPriority.MUST,
        title="Nguon",
    )
    target = Artifact(
        project_id=project.id,
        type=ArtifactType.CAPABILITY,
        status=ArtifactStatus.DRAFT,
        priority=ArtifactPriority.SHOULD,
        title="Dich hop le",
    )
    db_session.add_all([outsider, source, target])
    await db_session.flush()

    service = ArtifactService(db_session)

    with pytest.raises(PermissionError, match="không có quyền truy cập dự án"):
        await service.create_link(
            project_id=project.id,
            source_artifact_id=source.id,
            target_artifact_id=target.id,
            relation_type=RelationType.DERIVES_FROM,
            created_by_id=outsider.id,
        )


@pytest.mark.asyncio
async def test_create_link_rejects_reverse_duplicate_relation(db_session):
    user, project, _other_project = await _seed_link_projects(db_session)
    source = Artifact(
        project_id=project.id,
        type=ArtifactType.GOAL,
        status=ArtifactStatus.DRAFT,
        priority=ArtifactPriority.MUST,
        title="Nguon",
    )
    target = Artifact(
        project_id=project.id,
        type=ArtifactType.CAPABILITY,
        status=ArtifactStatus.DRAFT,
        priority=ArtifactPriority.SHOULD,
        title="Dich hop le",
    )
    db_session.add_all([source, target])
    await db_session.flush()

    service = ArtifactService(db_session)
    await service.create_link(
        project_id=project.id,
        source_artifact_id=source.id,
        target_artifact_id=target.id,
        relation_type=RelationType.DERIVES_FROM,
        created_by_id=user.id,
    )

    with pytest.raises(ValueError, match="đã tồn tại"):
        await service.create_link(
            project_id=project.id,
            source_artifact_id=target.id,
            target_artifact_id=source.id,
            relation_type=RelationType.DERIVES_FROM,
            created_by_id=user.id,
        )


@pytest.mark.asyncio
async def test_create_link_rejects_reverse_duplicate_even_with_different_relation(db_session):
    user, project, _other_project = await _seed_link_projects(db_session)
    source = Artifact(
        project_id=project.id,
        type=ArtifactType.GOAL,
        status=ArtifactStatus.DRAFT,
        priority=ArtifactPriority.MUST,
        title="Nguon",
    )
    target = Artifact(
        project_id=project.id,
        type=ArtifactType.CAPABILITY,
        status=ArtifactStatus.DRAFT,
        priority=ArtifactPriority.SHOULD,
        title="Dich hop le",
    )
    db_session.add_all([source, target])
    await db_session.flush()

    service = ArtifactService(db_session)
    await service.create_link(
        project_id=project.id,
        source_artifact_id=source.id,
        target_artifact_id=target.id,
        relation_type=RelationType.DERIVES_FROM,
        created_by_id=user.id,
    )

    with pytest.raises(ValueError, match="đã tồn tại"):
        await service.create_link(
            project_id=project.id,
            source_artifact_id=target.id,
            target_artifact_id=source.id,
            relation_type=RelationType.SATISFIES,
            created_by_id=user.id,
        )


@pytest.mark.asyncio
async def test_create_link_accepts_same_project_artifacts(db_session):
    user, project, _other_project = await _seed_link_projects(db_session)
    source = Artifact(
        project_id=project.id,
        type=ArtifactType.GOAL,
        status=ArtifactStatus.DRAFT,
        priority=ArtifactPriority.MUST,
        title="Nguon",
    )
    target = Artifact(
        project_id=project.id,
        type=ArtifactType.CAPABILITY,
        status=ArtifactStatus.DRAFT,
        priority=ArtifactPriority.SHOULD,
        title="Dich hop le",
    )
    db_session.add_all([source, target])
    await db_session.flush()

    service = ArtifactService(db_session)
    link = await service.create_link(
        project_id=project.id,
        source_artifact_id=source.id,
        target_artifact_id=target.id,
        relation_type=RelationType.DERIVES_FROM,
        created_by_id=user.id,
    )

    assert link.project_id == project.id
    assert link.source_artifact_id == source.id
    assert link.target_artifact_id == target.id


async def _seed_link_projects(db_session):
    user = User(email=f"artifact-{uuid.uuid4()}@example.com", hashed_password="x", full_name="Artifact Repository")
    db_session.add(user)
    await db_session.flush()

    org_a = Organization(name="Org A", slug=f"org-a-{uuid.uuid4()}", owner_id=user.id)
    org_b = Organization(name="Org B", slug=f"org-b-{uuid.uuid4()}", owner_id=user.id)
    db_session.add_all([org_a, org_b])
    await db_session.flush()
    db_session.add_all(
        [
            OrgMember(org_id=org_a.id, user_id=user.id, role="owner"),
            OrgMember(org_id=org_b.id, user_id=user.id, role="owner"),
        ]
    )
    project = Project(org_id=org_a.id, name="Project A", slug=f"project-a-{uuid.uuid4()}")
    other_project = Project(org_id=org_b.id, name="Project B", slug=f"project-b-{uuid.uuid4()}")
    db_session.add_all([project, other_project])
    await db_session.flush()
    return user, project, other_project
