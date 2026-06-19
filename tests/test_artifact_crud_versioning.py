import uuid

import pytest
from sqlalchemy import func, select

from app.models.artifact import Artifact, ArtifactEvidence, ArtifactReview, ArtifactVersion, EvidenceSourceType, RelationType
from tests.conftest import BASE
from tests.helpers import create_org, create_project, make_auth_headers


@pytest.mark.asyncio
async def test_create_artifact_creates_identity_and_first_version(client, db_session):
    headers, project = await _project_context(client)

    resp = await client.post(
        f"{BASE}/projects/{project['id']}/artifacts",
        json={
            "type": "functional_requirement",
            "title": "Đăng nhập bằng email",
            "body": "Người dùng đăng nhập bằng email và mật khẩu.",
            "priority": "must",
            "metadata": {"nguon": "phong_van"},
        },
        headers=headers,
    )

    assert resp.status_code == 201, resp.text
    artifact = resp.json()["data"]
    assert artifact["current_version_id"] is not None
    assert artifact["current_version"]["version_number"] == 1
    assert artifact["current_version"]["title"] == "Đăng nhập bằng email"
    assert artifact["current_version"]["body"] == "Người dùng đăng nhập bằng email và mật khẩu."

    version = await db_session.get(ArtifactVersion, uuid.UUID(artifact["current_version_id"]))
    assert version.artifact_id.hex == artifact["id"].replace("-", "")
    assert version.version_number == 1


@pytest.mark.asyncio
async def test_update_artifact_creates_immutable_version_and_preserves_old_content(client, db_session):
    headers, project = await _project_context(client)
    artifact = await _create_artifact(client, headers, project["id"], title="Phiên bản 1", body="Nội dung cũ")
    old_version_id = artifact["current_version_id"]

    resp = await client.patch(
        f"{BASE}/projects/{project['id']}/artifacts/{artifact['id']}",
        json={"title": "Phiên bản 2", "body": "Nội dung mới", "change_summary": "Cập nhật nội dung"},
        headers=headers,
    )

    assert resp.status_code == 200, resp.text
    updated = resp.json()["data"]
    assert updated["current_version_id"] != old_version_id
    assert updated["current_version"]["version_number"] == 2
    assert updated["current_version"]["parent_version_id"] == old_version_id
    assert updated["current_version"]["title"] == "Phiên bản 2"

    old_version = await db_session.get(ArtifactVersion, uuid.UUID(old_version_id))
    assert old_version.title == "Phiên bản 1"
    assert old_version.body == "Nội dung cũ"


@pytest.mark.asyncio
async def test_review_approve_and_reject_create_review_rows(client, db_session):
    headers, project = await _project_context(client)
    artifact = await _create_artifact(client, headers, project["id"])
    version_id = artifact["current_version_id"]

    approve = await client.post(
        f"{BASE}/projects/{project['id']}/artifacts/{artifact['id']}/versions/{version_id}/review",
        json={"review_status": "approved", "comment": "Đạt yêu cầu"},
        headers=headers,
    )
    reject = await client.post(
        f"{BASE}/projects/{project['id']}/artifacts/{artifact['id']}/versions/{version_id}/review",
        json={"review_status": "rejected", "comment": "Cần sửa lại"},
        headers=headers,
    )

    assert approve.status_code == 201, approve.text
    assert reject.status_code == 201, reject.text
    rows = (
        await db_session.execute(
            select(ArtifactReview)
            .where(ArtifactReview.artifact_version_id == uuid.UUID(version_id))
            .order_by(ArtifactReview.created_at)
        )
    ).scalars().all()
    assert [row.review_status.value for row in rows] == ["approved", "rejected"]
    assert rows[0].reviewed_by_id is not None
    assert rows[0].created_at is not None


@pytest.mark.asyncio
async def test_restore_version_sets_current_version_without_overwriting_history(client, db_session):
    headers, project = await _project_context(client)
    artifact = await _create_artifact(client, headers, project["id"], title="Cũ", body="Nội dung cũ")
    first_version_id = artifact["current_version_id"]
    updated_resp = await client.patch(
        f"{BASE}/projects/{project['id']}/artifacts/{artifact['id']}",
        json={"title": "Mới", "body": "Nội dung mới"},
        headers=headers,
    )
    second_version_id = updated_resp.json()["data"]["current_version_id"]

    restore = await client.post(
        f"{BASE}/projects/{project['id']}/artifacts/{artifact['id']}/versions/{first_version_id}/restore",
        headers=headers,
    )

    assert restore.status_code == 200, restore.text
    assert restore.json()["data"]["current_version_id"] == first_version_id
    versions = (
        await db_session.execute(select(ArtifactVersion).where(ArtifactVersion.artifact_id == uuid.UUID(artifact["id"])))
    ).scalars().all()
    assert {str(version.id) for version in versions} == {first_version_id, second_version_id}


@pytest.mark.asyncio
async def test_source_document_upload_does_not_create_artifact(client, db_session):
    headers, project = await _project_context(client)
    before_count = await db_session.scalar(select(func.count(Artifact.id)))

    resp = await client.post(
        f"{BASE}/projects/{project['id']}/source-documents",
        json={
            "title": "Ghi chú phỏng vấn",
            "source_type": "markdown_upload",
            "content_text": "# Ghi chú\nNgười dùng cần báo cáo.",
            "mime_type": "text/markdown",
        },
        headers=headers,
    )

    assert resp.status_code == 201, resp.text
    assert resp.json()["data"]["content_hash"]
    after_count = await db_session.scalar(select(func.count(Artifact.id)))
    assert after_count == before_count


@pytest.mark.asyncio
async def test_research_output_accepts_optional_research_type(client):
    headers, project = await _project_context(client)

    with_type = await _create_artifact(
        client,
        headers,
        project["id"],
        artifact_type="research_output",
        metadata={"research_type": "interview"},
    )
    without_type = await _create_artifact(client, headers, project["id"], artifact_type="research_output")

    assert with_type["type"] == "research_output"
    assert without_type["type"] == "research_output"


@pytest.mark.asyncio
async def test_artifact_endpoints_reject_non_project_member(client):
    owner_headers, project = await _project_context(client)
    outsider_headers = await make_auth_headers(client)
    artifact = await _create_artifact(client, owner_headers, project["id"])

    create_resp = await client.post(
        f"{BASE}/projects/{project['id']}/artifacts",
        json={"type": "goal", "title": "Không hợp lệ", "body": "Không có quyền"},
        headers=outsider_headers,
    )
    list_resp = await client.get(f"{BASE}/projects/{project['id']}/artifacts", headers=outsider_headers)
    update_resp = await client.patch(
        f"{BASE}/projects/{project['id']}/artifacts/{artifact['id']}",
        json={"body": "Không được sửa"},
        headers=outsider_headers,
    )
    source_resp = await client.post(
        f"{BASE}/projects/{project['id']}/source-documents",
        json={"title": "Không hợp lệ", "source_type": "text_paste", "content_text": "Không có quyền"},
        headers=outsider_headers,
    )

    assert create_resp.status_code == 403
    assert list_resp.status_code == 403
    assert update_resp.status_code == 403
    assert source_resp.status_code == 403


@pytest.mark.asyncio
async def test_list_artifacts_filters_by_type_status_and_priority(client):
    headers, project = await _project_context(client)
    await _create_artifact(client, headers, project["id"], artifact_type="goal", priority="must")
    draft_should = await _create_artifact(
        client,
        headers,
        project["id"],
        artifact_type="functional_requirement",
        priority="should",
    )
    await _create_artifact(client, headers, project["id"], artifact_type="functional_requirement", priority="could")

    resp = await client.get(
        f"{BASE}/projects/{project['id']}/artifacts",
        params={"type": "functional_requirement", "status": "draft", "priority": "should"},
        headers=headers,
    )

    assert resp.status_code == 200, resp.text
    items = resp.json()["data"]
    assert [item["id"] for item in items] == [draft_should["id"]]


@pytest.mark.asyncio
async def test_delete_artifact_removes_own_history_and_links_but_keeps_related_artifact(client, db_session):
    headers, project = await _project_context(client)
    source = await _create_artifact(client, headers, project["id"], title="Nguồn")
    target = await _create_artifact(client, headers, project["id"], title="Đích")
    db_session.add(
        ArtifactEvidence(
            artifact_id=uuid.UUID(source["id"]),
            artifact_version_id=uuid.UUID(source["current_version_id"]),
            source_type=EvidenceSourceType.USER_INPUT,
            locator="chat:1",
        )
    )
    from app.models.artifact import ArtifactLink

    db_session.add(
        ArtifactLink(
            project_id=uuid.UUID(project["id"]),
            source_artifact_id=uuid.UUID(source["id"]),
            target_artifact_id=uuid.UUID(target["id"]),
            relation_type=RelationType.SUPPORTS,
        )
    )
    await db_session.flush()

    resp = await client.delete(f"{BASE}/projects/{project['id']}/artifacts/{source['id']}", headers=headers)

    assert resp.status_code == 204, resp.text
    assert await db_session.get(Artifact, uuid.UUID(source["id"])) is None
    assert await db_session.get(Artifact, uuid.UUID(target["id"])) is not None
    assert (
        await db_session.scalar(
            select(func.count(ArtifactVersion.id)).where(ArtifactVersion.artifact_id == uuid.UUID(source["id"]))
        )
        == 0
    )


@pytest.mark.asyncio
async def test_delete_source_document_returns_conflict_when_evidence_uses_it(client, db_session):
    headers, project = await _project_context(client)
    source_doc_resp = await client.post(
        f"{BASE}/projects/{project['id']}/source-documents",
        json={"title": "Tài liệu nguồn", "source_type": "text_paste", "content_text": "Bằng chứng"},
        headers=headers,
    )
    artifact = await _create_artifact(client, headers, project["id"])
    source_doc = source_doc_resp.json()["data"]
    db_session.add(
        ArtifactEvidence(
            artifact_id=uuid.UUID(artifact["id"]),
            artifact_version_id=uuid.UUID(artifact["current_version_id"]),
            source_document_id=uuid.UUID(source_doc["id"]),
            source_type=EvidenceSourceType.DOCUMENT,
            locator="doc:1",
        )
    )
    await db_session.flush()

    resp = await client.delete(f"{BASE}/projects/{project['id']}/source-documents/{source_doc['id']}", headers=headers)

    assert resp.status_code == 409
    assert artifact["id"] in resp.json()["detail"]["artifact_ids"]


async def _project_context(client):
    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    return headers, project


async def _create_artifact(
    client,
    headers,
    project_id,
    *,
    artifact_type="functional_requirement",
    title="Yêu cầu",
    body="Nội dung yêu cầu",
    priority="must",
    metadata=None,
):
    resp = await client.post(
        f"{BASE}/projects/{project_id}/artifacts",
        json={
            "type": artifact_type,
            "title": title,
            "body": body,
            "priority": priority,
            "metadata": metadata or {},
        },
        headers=headers,
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["data"]
