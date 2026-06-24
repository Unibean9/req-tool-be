import pytest

from app.documents.registry import children_of
from tests.conftest import BASE
from tests.helpers import create_org, create_project, make_auth_headers


async def _project_context(client):
    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    return headers, project


@pytest.mark.asyncio
async def test_document_types_require_auth_and_come_from_registry(client):
    unauthorized = await client.get(f"{BASE}/documents/types")
    assert unauthorized.status_code == 401

    headers = await make_auth_headers(client)
    response = await client.get(f"{BASE}/documents/types", headers=headers)
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert [item["artifact_type"] for item in data["containers"]] == ["brd", "prd", "sad"]
    functional = next(item for item in data["items"] if item["artifact_type"] == "functional_requirement")
    assert functional["output_contract"]["format"] == "markdown"
    assert "## Functional Requirement" in functional["output_contract"]["required_headings"]


@pytest.mark.asyncio
async def test_get_brd_returns_seven_empty_slots_before_creation(client):
    headers, project = await _project_context(client)
    response = await client.get(
        f"{BASE}/projects/{project['id']}/documents/brd",
        headers=headers,
    )
    assert response.status_code == 200, response.text
    document = response.json()["data"]
    assert document["artifact_id"] is None
    assert len(document["items"]) == 7
    assert all(item["artifact_id"] is None for item in document["items"])


@pytest.mark.asyncio
async def test_create_brd_item_links_child_to_container(client):
    headers, project = await _project_context(client)
    container_response = await client.post(
        f"{BASE}/projects/{project['id']}/documents/brd",
        headers=headers,
    )
    assert container_response.status_code == 201, container_response.text
    container = container_response.json()["data"]

    item_response = await client.post(
        f"{BASE}/projects/{project['id']}/documents/brd/vision_objectives",
        json={
            "title": "Vision",
            "body": "Tăng retention 20% trong Q4.",
            "status": "accepted",
        },
        headers=headers,
    )
    assert item_response.status_code == 201, item_response.text
    item = item_response.json()["data"]
    assert item["parent_id"] == container["artifact_id"]
    assert item["artifact_type"] == "vision_objectives"
    assert item["current_version"]["body"] == "Tăng retention 20% trong Q4."

    document_response = await client.get(
        f"{BASE}/projects/{project['id']}/documents/brd",
        headers=headers,
    )
    slots = {
        slot["artifact_type"]: slot
        for slot in document_response.json()["data"]["items"]
    }
    assert slots["vision_objectives"]["artifact_id"] == item["artifact_id"]


@pytest.mark.asyncio
async def test_prd_container_and_functional_requirement_flow(client):
    headers, project = await _project_context(client)
    container_response = await client.post(
        f"{BASE}/projects/{project['id']}/documents/prd",
        headers=headers,
    )
    assert container_response.status_code == 201, container_response.text
    container = container_response.json()["data"]
    assert [item["artifact_type"] for item in container["items"]] == [
        "functional_requirement",
        "use_case",
        "non_functional_requirement",
        "acceptance_criteria",
    ]

    item_response = await client.post(
        f"{BASE}/projects/{project['id']}/documents/prd/functional_requirement",
        json={
            "title": "Email login",
            "body": "User can sign in with email and password.",
        },
        headers=headers,
    )
    assert item_response.status_code == 201, item_response.text
    assert item_response.json()["data"]["parent_id"] == container["artifact_id"]


@pytest.mark.asyncio
async def test_manual_accepted_children_auto_accept_container(client):
    headers, project = await _project_context(client)
    container_response = await client.post(
        f"{BASE}/projects/{project['id']}/documents/brd",
        headers=headers,
    )
    assert container_response.status_code == 201, container_response.text

    for item_type in children_of("brd"):
        item_response = await client.post(
            f"{BASE}/projects/{project['id']}/documents/brd/{item_type}",
            json={
                "title": item_type,
                "body": f"{item_type} body",
                "status": "accepted",
            },
            headers=headers,
        )
        assert item_response.status_code == 201, item_response.text

    document_response = await client.get(
        f"{BASE}/projects/{project['id']}/documents/brd",
        headers=headers,
    )
    assert document_response.status_code == 200, document_response.text
    assert document_response.json()["data"]["status"] == "accepted"


@pytest.mark.asyncio
async def test_manual_child_downgrade_reverts_container_to_draft(client):
    headers, project = await _project_context(client)
    container_response = await client.post(
        f"{BASE}/projects/{project['id']}/documents/brd",
        headers=headers,
    )
    assert container_response.status_code == 201, container_response.text

    for item_type in children_of("brd"):
        item_response = await client.post(
            f"{BASE}/projects/{project['id']}/documents/brd/{item_type}",
            json={
                "title": item_type,
                "body": f"{item_type} body",
                "status": "accepted",
            },
            headers=headers,
        )
        assert item_response.status_code == 201, item_response.text

    downgrade_response = await client.post(
        f"{BASE}/projects/{project['id']}/documents/brd/vision_objectives",
        json={
            "title": "Vision draft",
            "body": "Needs revision.",
            "status": "draft",
        },
        headers=headers,
    )
    assert downgrade_response.status_code == 201, downgrade_response.text

    document_response = await client.get(
        f"{BASE}/projects/{project['id']}/documents/brd",
        headers=headers,
    )
    assert document_response.status_code == 200, document_response.text
    assert document_response.json()["data"]["status"] == "draft"
