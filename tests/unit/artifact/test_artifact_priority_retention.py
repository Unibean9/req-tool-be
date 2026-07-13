import uuid

import pytest
from sqlalchemy import select

from app.models.artifact import Artifact, ArtifactPriority, ArtifactStatus, ArtifactType


@pytest.mark.asyncio
async def test_wont_artifacts_are_retained_when_other_priorities_are_deleted(db_session):
    project_id = uuid.uuid4()
    wont = Artifact(
        project_id=project_id,
        type=ArtifactType.FUNCTIONAL_REQUIREMENT,
        status=ArtifactStatus.DRAFT,
        priority=ArtifactPriority.WONT,
        title="Khong lam trong phase nay",
    )
    must = Artifact(
        project_id=project_id,
        type=ArtifactType.FUNCTIONAL_REQUIREMENT,
        status=ArtifactStatus.DRAFT,
        priority=ArtifactPriority.MUST,
        title="Bat buoc",
    )
    db_session.add_all([wont, must])
    await db_session.flush()

    await db_session.delete(must)
    await db_session.flush()

    result = await db_session.execute(
        select(Artifact).where(
            Artifact.project_id == project_id,
            Artifact.priority == ArtifactPriority.WONT,
        )
    )
    assert result.scalar_one().id == wont.id
