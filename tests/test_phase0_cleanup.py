import pytest

from tests.conftest import BASE
from tests.helpers import create_org, create_project, make_auth_headers


REMOVED_ENDPOINTS = [
    "/stakeholders",
    "/actors",
    "/nfrs",
    "/requirements/epics",
    "/requirements/features",
    "/requirements/stories",
    "/requirements/tasks",
    "/estimates",
    "/context-diagram",
]


@pytest.mark.asyncio
async def test_health_endpoint_works(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_auth_org_project_foundation_works(client):
    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])

    me_resp = await client.get(f"{BASE}/users/me", headers=headers)
    org_resp = await client.get(f"{BASE}/orgs/me", headers=headers)
    project_resp = await client.get(f"{BASE}/orgs/{org['id']}/projects/{project['id']}", headers=headers)

    assert me_resp.status_code == 200
    assert org_resp.status_code == 200
    assert project_resp.status_code == 200


@pytest.mark.asyncio
async def test_spec_org_project_aliases_work(client):
    headers = await make_auth_headers(client)
    org = await create_org(client, headers)

    orgs_resp = await client.get(f"{BASE}/organizations", headers=headers)
    create_resp = await client.post(
        f"{BASE}/projects",
        json={"org_id": org["id"], "name": "Minimal Project"},
        headers=headers,
    )
    assert orgs_resp.status_code == 200
    assert create_resp.status_code == 201, create_resp.text

    project = create_resp.json()["data"]
    projects_resp = await client.get(f"{BASE}/projects", headers=headers)
    project_resp = await client.get(f"{BASE}/projects/{project['id']}", headers=headers)

    assert projects_resp.status_code == 200
    assert project_resp.status_code == 200
    assert project_resp.json()["data"]["proposed_solutions"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("path", REMOVED_ENDPOINTS)
async def test_removed_legacy_endpoints_return_404(client, path):
    headers = await make_auth_headers(client)
    resp = await client.get(f"{BASE}{path}", headers=headers)
    assert resp.status_code == 404
