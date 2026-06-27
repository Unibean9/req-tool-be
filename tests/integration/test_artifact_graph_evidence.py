import uuid

import pytest

from app.models.artifact import ArtifactEvidence
from tests.conftest import BASE
from tests.helpers import create_org, create_project, make_auth_headers


@pytest.mark.asyncio
async def test_graph_endpoint_returns_nodes_links_and_version_references(client):
    headers, project = await _project_context(client)
    req = await _create_artifact(client, headers, project["id"], artifact_type="brd", title="BRD")
    fr = await _create_artifact(
        client,
        headers,
        project["id"],
        artifact_type="functional_requirement",
        title="Requirement chuc nang",
    )
    link = await _create_link(client, headers, project["id"], fr["id"], req["id"], "satisfies")

    resp = await client.get(f"{BASE}/projects/{project['id']}/artifact-graph", headers=headers)

    assert resp.status_code == 200, resp.text
    graph = resp.json()["data"]
    assert {node["id"] for node in graph["nodes"]} == {req["id"], fr["id"]}
    assert graph["links"][0]["id"] == link["id"]
    assert graph["links"][0]["relation_type"] == "satisfies"
    fr_node = next(node for node in graph["nodes"] if node["id"] == fr["id"])
    assert fr_node["current_version"]["id"] == fr["current_version_id"]


@pytest.mark.asyncio
async def test_graph_endpoint_reports_orphan_warning(client):
    headers, project = await _project_context(client)
    artifact = await _create_artifact(client, headers, project["id"], artifact_type="epic")

    resp = await client.get(f"{BASE}/projects/{project['id']}/artifact-graph", headers=headers)

    warnings = resp.json()["data"]["warnings"]
    assert {"type": "orphan_artifact", "artifact_id": artifact["id"]} in warnings


@pytest.mark.asyncio
async def test_graph_endpoint_reports_missing_upstream_trace_warning(client):
    headers, project = await _project_context(client)
    fr = await _create_artifact(client, headers, project["id"], artifact_type="functional_requirement")

    resp = await client.get(f"{BASE}/projects/{project['id']}/artifact-graph", headers=headers)

    warnings = resp.json()["data"]["warnings"]
    assert {"type": "missing_upstream_trace", "artifact_id": fr["id"]} in warnings


@pytest.mark.asyncio
async def test_graph_endpoint_reports_conflicting_artifacts_warning(client):
    headers, project = await _project_context(client)
    first = await _create_artifact(client, headers, project["id"], artifact_type="functional_requirement", title="A")
    second = await _create_artifact(client, headers, project["id"], artifact_type="epic", title="B")
    await _create_link(client, headers, project["id"], first["id"], second["id"], "conflicts_with")

    resp = await client.get(f"{BASE}/projects/{project['id']}/artifact-graph", headers=headers)

    warnings = resp.json()["data"]["warnings"]
    assert {"type": "conflicting_artifacts", "artifact_id": first["id"]} in warnings
    assert {"type": "conflicting_artifacts", "artifact_id": second["id"]} in warnings


@pytest.mark.asyncio
async def test_create_link_rejects_cross_project_self_and_bidirectional_duplicate(client):
    headers, project = await _project_context(client)
    other_headers, other_project = await _project_context(client)
    source = await _create_artifact(client, headers, project["id"], title="Source")
    target = await _create_artifact(client, headers, project["id"], title="Target")
    other = await _create_artifact(client, other_headers, other_project["id"], title="Other project")

    self_link = await client.post(
        f"{BASE}/projects/{project['id']}/artifact-links",
        json={"source_artifact_id": source["id"], "target_artifact_id": source["id"], "relation_type": "derives_from"},
        headers=headers,
    )
    cross_project = await client.post(
        f"{BASE}/projects/{project['id']}/artifact-links",
        json={"source_artifact_id": source["id"], "target_artifact_id": other["id"], "relation_type": "derives_from"},
        headers=headers,
    )
    created = await _create_link(client, headers, project["id"], source["id"], target["id"], "derives_from")
    duplicate = await client.post(
        f"{BASE}/projects/{project['id']}/artifact-links",
        json={"source_artifact_id": target["id"], "target_artifact_id": source["id"], "relation_type": "satisfies"},
        headers=headers,
    )

    assert self_link.status_code == 400
    assert cross_project.status_code == 400
    assert created["relation_type"] == "derives_from"
    assert duplicate.status_code == 400


@pytest.mark.asyncio
async def test_evidence_attach_and_retrieve(client, db_session):
    headers, project = await _project_context(client)
    source_doc = await _create_source_document(client, headers, project["id"])
    artifact = await _create_artifact(client, headers, project["id"])

    create_resp = await client.post(
        f"{BASE}/projects/{project['id']}/artifacts/{artifact['id']}/evidence",
        json={
            "artifact_version_id": artifact["current_version_id"],
            "source_document_id": source_doc["id"],
            "source_type": "document",
            "locator": "doc:section-1",
            "excerpt": "Users need monthly reports.",
            "confidence": 0.9,
            "metadata": {"trang": 1},
        },
        headers=headers,
    )
    list_resp = await client.get(f"{BASE}/projects/{project['id']}/artifacts/{artifact['id']}/evidence", headers=headers)

    assert create_resp.status_code == 201, create_resp.text
    assert list_resp.status_code == 200, list_resp.text
    evidence = list_resp.json()["data"]
    assert len(evidence) == 1
    assert evidence[0]["source_document_id"] == source_doc["id"]
    assert evidence[0]["locator"] == "doc:section-1"
    db_evidence = await db_session.get(ArtifactEvidence, uuid.UUID(evidence[0]["id"]))
    assert db_evidence.artifact_id == uuid.UUID(artifact["id"])


@pytest.mark.asyncio
async def test_source_document_delete_returns_409_when_referenced_by_evidence(client):
    headers, project = await _project_context(client)
    source_doc = await _create_source_document(client, headers, project["id"])
    artifact = await _create_artifact(client, headers, project["id"])
    await client.post(
        f"{BASE}/projects/{project['id']}/artifacts/{artifact['id']}/evidence",
        json={
            "source_document_id": source_doc["id"],
            "source_type": "document",
            "locator": "doc:section-1",
        },
        headers=headers,
    )

    resp = await client.delete(f"{BASE}/projects/{project['id']}/source-documents/{source_doc['id']}", headers=headers)

    assert resp.status_code == 409
    assert artifact["id"] in resp.json()["detail"]["artifact_ids"]


async def _project_context(client):
    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    return headers, project


async def _create_artifact(client, headers, project_id, *, artifact_type="functional_requirement", title="Artifact"):
    resp = await client.post(
        f"{BASE}/projects/{project_id}/artifacts",
        json={"type": artifact_type, "title": title, "body": f"Content {title}", "priority": "must"},
        headers=headers,
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["data"]


async def _create_link(client, headers, project_id, source_id, target_id, relation_type):
    resp = await client.post(
        f"{BASE}/projects/{project_id}/artifact-links",
        json={"source_artifact_id": source_id, "target_artifact_id": target_id, "relation_type": relation_type},
        headers=headers,
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["data"]


async def _create_source_document(client, headers, project_id):
    resp = await client.post(
        f"{BASE}/projects/{project_id}/source-documents",
        json={"title": "Source document", "source_type": "text_paste", "content_text": "Evidence"},
        headers=headers,
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["data"]
