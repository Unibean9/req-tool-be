"""Regression guard: reserving a document-item slot must not create a placeholder version.

Frontend bootstraps a `focused_artifact_id` for the agent by POSTing an empty
"(Workbench slot...)" body with change_source=system before the agent even runs.
If that call creates a real ArtifactVersion, the item ends up with two versions
after the agent's draft is approved: a meaningless placeholder (v1) and the real
approved content (v2), even though the user only ever approved one draft.
"""

import uuid

import pytest

from app.core.security import create_access_token, hash_password
from app.models.artifact import ChangeSource
from app.models.user import User
from app.schemas.document import DocumentItemWriteRequest
from app.services.document_service import DocumentService
from tests.helpers import create_org, create_project


async def _make_user(db_session) -> User:
    user = User(
        email=f"bootstrap-{uuid.uuid4().hex[:8]}@example.com",
        hashed_password=hash_password("Secret123!"),
        full_name="Bootstrap Tester",
        is_active=True,
    )
    db_session.add(user)
    await db_session.flush()
    return user


@pytest.mark.asyncio
async def test_system_bootstrap_slot_reservation_creates_no_version(client, db_session):
    user = await _make_user(db_session)
    headers = {"Authorization": f"Bearer {create_access_token(str(user.id), user.role)}"}
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])

    service = DocumentService(db_session)
    await service.create_document(project_id=project_id, document_type="brd", created_by_id=user.id)

    bootstrapped = await service.upsert_item(
        project_id=project_id,
        document_type="brd",
        item_type="problem_statement",
        body=DocumentItemWriteRequest(
            title="Problem Statement",
            body="(Workbench slot — content will be drafted here.)",
            change_source=ChangeSource.SYSTEM,
            change_summary="Reserved section slot for workbench",
        ),
        created_by_id=user.id,
    )

    assert bootstrapped.artifact_id is not None
    assert bootstrapped.current_version_id is None
    assert bootstrapped.versions == []


@pytest.mark.asyncio
async def test_repeated_bootstrap_before_first_draft_stays_versionless(client, db_session):
    """Reopening the section (double-submit/retry) re-issues the same slot-reservation
    upsert before any agent draft exists. It must not create a placeholder version either."""
    user = await _make_user(db_session)
    headers = {"Authorization": f"Bearer {create_access_token(str(user.id), user.role)}"}
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])

    service = DocumentService(db_session)
    await service.create_document(project_id=project_id, document_type="brd", created_by_id=user.id)

    bootstrap_body = DocumentItemWriteRequest(
        title="Problem Statement",
        body="(Workbench slot — content will be drafted here.)",
        change_source=ChangeSource.SYSTEM,
        change_summary="Reserved section slot for workbench",
    )

    first = await service.upsert_item(
        project_id=project_id,
        document_type="brd",
        item_type="problem_statement",
        body=bootstrap_body,
        created_by_id=user.id,
    )
    second = await service.upsert_item(
        project_id=project_id,
        document_type="brd",
        item_type="problem_statement",
        body=bootstrap_body,
        created_by_id=user.id,
    )

    assert first.artifact_id == second.artifact_id
    assert second.current_version_id is None
    assert second.versions == []


@pytest.mark.asyncio
async def test_first_agent_approval_after_bootstrap_yields_single_version(client, db_session):
    user = await _make_user(db_session)
    headers = {"Authorization": f"Bearer {create_access_token(str(user.id), user.role)}"}
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    project_id = uuid.UUID(project["id"])

    service = DocumentService(db_session)
    await service.create_document(project_id=project_id, document_type="brd", created_by_id=user.id)

    bootstrapped = await service.upsert_item(
        project_id=project_id,
        document_type="brd",
        item_type="problem_statement",
        body=DocumentItemWriteRequest(
            title="Problem Statement",
            body="(Workbench slot — content will be drafted here.)",
            change_source=ChangeSource.SYSTEM,
            change_summary="Reserved section slot for workbench",
        ),
        created_by_id=user.id,
    )

    _artifact, version = await service.create_item_version(
        artifact_id=bootstrapped.artifact_id,
        project_id=project_id,
        title="Phát biểu vấn đề - Hệ thống POS quán cà phê",
        body="Nội dung draft do agent tạo.",
        created_by_id=user.id,
        change_source=ChangeSource.AI_GENERATION,
        mark_accepted=True,
    )

    assert version.version_number == 1

    item = await service.get_item(project_id=project_id, document_type="brd", item_type="problem_statement")
    assert [v.version_number for v in item.versions] == [1]
