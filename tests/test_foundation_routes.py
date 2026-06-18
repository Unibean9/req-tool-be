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
async def test_top_level_org_project_routes_work(client):
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
    assert "proposed_solutions" not in project_resp.json()["data"]


@pytest.mark.asyncio
async def test_project_create_contract_is_name_and_optional_description_only(client):
    headers = await make_auth_headers(client)
    org = await create_org(client, headers)

    minimal_resp = await client.post(
        f"{BASE}/orgs/{org['id']}/projects",
        json={"name": "Dự án tối giản", "description": "Mô tả ngắn"},
        headers=headers,
    )
    obsolete_resp = await client.post(
        f"{BASE}/orgs/{org['id']}/projects",
        json={"name": "Dự án sai", "problems": ["Vấn đề nghiệp vụ"]},
        headers=headers,
    )

    assert minimal_resp.status_code == 201, minimal_resp.text
    project = minimal_resp.json()["data"]
    assert project["name"] == "Dự án tối giản"
    assert project["description"] == "Mô tả ngắn"
    assert "problems" not in project
    assert "context" not in project
    assert obsolete_resp.status_code == 422
