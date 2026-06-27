import pytest

from tests.conftest import BASE
from tests.helpers import create_org, create_project, make_auth_headers


@pytest.mark.asyncio
async def test_auth_org_project_foundation_routes_work(client):
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
async def test_top_level_project_routes_are_removed(client):
    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])

    orgs_resp = await client.get(f"{BASE}/organizations", headers=headers)
    projects_resp = await client.get(f"{BASE}/projects", headers=headers)
    create_resp = await client.post(
        f"{BASE}/projects",
        json={"org_id": org["id"], "name": "Minimal Project"},
        headers=headers,
    )
    project_resp = await client.get(f"{BASE}/projects/{project['id']}", headers=headers)

    assert orgs_resp.status_code == 200
    assert projects_resp.status_code == 404
    assert create_resp.status_code == 404
    assert project_resp.status_code == 404


@pytest.mark.asyncio
async def test_project_create_contract_is_name_and_optional_description_only(client):
    headers = await make_auth_headers(client)
    org = await create_org(client, headers)

    minimal_resp = await client.post(
        f"{BASE}/orgs/{org['id']}/projects",
        json={"name": "Minimal project", "description": "Short description"},
        headers=headers,
    )
    obsolete_resp = await client.post(
        f"{BASE}/orgs/{org['id']}/projects",
        json={"name": "Du an sai", "problems": ["Van de nghiep vu"]},
        headers=headers,
    )

    assert minimal_resp.status_code == 201, minimal_resp.text
    project = minimal_resp.json()["data"]
    assert project["name"] == "Minimal project"
    assert project["description"] == "Short description"
    assert "problems" not in project
    assert "context" not in project
    assert obsolete_resp.status_code == 422
