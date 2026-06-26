import uuid

import pytest

from app.models.artifact import Artifact, ArtifactStatus, ArtifactType, RelationType
from app.models.organization import Organization, OrgMember
from app.models.project import Project
from app.models.user import User
from app.services.artifact_service import ArtifactService


@pytest.mark.asyncio
async def test_create_artifact_link_rejects_transitive_cycle(db_session):
    user = User(email=f"cycle-{uuid.uuid4().hex}@example.com", hashed_password="x", full_name="Cycle User")
    db_session.add(user)
    await db_session.flush()
    org = Organization(name="Cycle Org", slug=f"cycle-{uuid.uuid4().hex}", owner_id=user.id)
    db_session.add(org)
    await db_session.flush()
    db_session.add(OrgMember(org_id=org.id, user_id=user.id, role="owner"))
    project = Project(org_id=org.id, name="Cycle Project", slug=f"cycle-{uuid.uuid4().hex}")
    db_session.add(project)
    await db_session.flush()
    project_id = project.id
    artifacts = []
    for title, artifact_type in (
        ("BRD", ArtifactType.BRD),
        ("PRD", ArtifactType.PRD),
        ("SAD", ArtifactType.SAD),
    ):
        artifact = Artifact(
            project_id=project_id,
            type=artifact_type,
            status=ArtifactStatus.DRAFT,
            title=title,
            created_by_id=user.id,
        )
        db_session.add(artifact)
        artifacts.append(artifact)
    await db_session.flush()

    service = ArtifactService(db_session)
    await service.create_link(
        project_id=project_id,
        source_artifact_id=artifacts[0].id,
        target_artifact_id=artifacts[1].id,
        relation_type=RelationType.DEPENDS_ON,
        created_by_id=user.id,
    )
    await service.create_link(
        project_id=project_id,
        source_artifact_id=artifacts[1].id,
        target_artifact_id=artifacts[2].id,
        relation_type=RelationType.DEPENDS_ON,
        created_by_id=user.id,
    )

    with pytest.raises(ValueError, match="cycle"):
        await service.create_link(
            project_id=project_id,
            source_artifact_id=artifacts[2].id,
            target_artifact_id=artifacts[0].id,
            relation_type=RelationType.DEPENDS_ON,
            created_by_id=user.id,
        )
