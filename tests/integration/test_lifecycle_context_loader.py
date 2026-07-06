import uuid

import pytest

from app.graphs.analysis.context_loader import _load_lifecycle_reports
from app.models.artifact import Artifact, ArtifactStatus, ArtifactType, ArtifactVersion, ChangeSource, VersionStatus
from tests.helpers import create_org, create_project, make_auth_headers


@pytest.mark.asyncio
async def test_lifecycle_report_marks_missing_based_on_predecessor_as_orphan(client, db_session):
    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])

    focused = Artifact(
        project_id=project_id,
        type=ArtifactType.VISION_OBJECTIVES,
        status=ArtifactStatus.ACCEPTED,
        title="Vision",
        extra_metadata={},
    )
    db_session.add(focused)
    await db_session.flush()
    missing_predecessor_id = uuid.uuid4()
    version = ArtifactVersion(
        artifact_id=focused.id,
        version_number=1,
        title="Vision",
        body="## Vision\nGrow retention.",
        status=VersionStatus.DRAFT,
        change_source=ChangeSource.AI_GENERATION,
        extra_metadata={"based_on": {str(missing_predecessor_id): str(uuid.uuid4())}},
    )
    db_session.add(version)
    await db_session.flush()
    focused.current_version_id = version.id
    await db_session.flush()

    reports = await _load_lifecycle_reports(
        db_session,
        project_id=project_id,
        context_types=["vision_objectives"],
        session_id="session-1",
    )

    assert reports[0]["state"] == "orphan"
    assert reports[0]["artifact_id"] == str(focused.id)
    assert str(missing_predecessor_id) in reports[0]["reason"]
