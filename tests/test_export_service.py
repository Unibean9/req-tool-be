import pytest

from tests.conftest import BASE
from tests.helpers import create_org, create_project, make_auth_headers

REQUIREMENTS_BODY = {
    "vision_objectives": "Tăng tỷ lệ giữ chân người dùng lên 20%.",
    "problem_statement": "Quy trình hiện tại chậm và thiếu minh bạch.",
    "stakeholder_register": "Người dùng cuối, quản trị viên, chủ dự án.",
    "scope_capabilities": "Dashboard theo dõi và luồng phê duyệt.",
    "business_rules": "Mọi phê duyệt phải có người chịu trách nhiệm.",
    "constraints_assumptions": "Tích hợp trong hạ tầng hiện có.",
    "risks_issues": "Thiếu dữ liệu lịch sử để baseline.",
}


@pytest.mark.asyncio
async def test_brd_export_renders_requirements_sections_and_research_basis(client):
    headers, project = await _project_context(client)
    await _requirements_artifact(client, headers, project["id"], REQUIREMENTS_BODY)
    await _source_document(client, headers, project["id"], "Interview", "Người dùng cần dashboard rõ ràng")

    resp = await client.get(f"{BASE}/projects/{project['id']}/exports/brd.md", headers=headers)

    assert resp.status_code == 200, resp.text
    text = resp.text
    for heading in [
        "## Vision and Objectives",
        "## Problem Statement",
        "## Stakeholder Register",
        "## Scope and Capabilities",
        "## Business Rules",
        "## Constraints and Assumptions",
        "## Risks and Issues",
        "## Research Basis",
    ]:
        assert heading in text
    assert "Tăng tỷ lệ giữ chân" in text
    assert "Interview: Người dùng cần dashboard rõ ràng" in text


@pytest.mark.asyncio
async def test_product_brief_renders_first_four_sections_only(client):
    headers, project = await _project_context(client)
    await _requirements_artifact(client, headers, project["id"], REQUIREMENTS_BODY)

    resp = await client.get(f"{BASE}/projects/{project['id']}/exports/product-brief.md", headers=headers)

    assert resp.status_code == 200, resp.text
    text = resp.text
    assert "# Product Brief" in text
    assert "## Vision and Objectives" in text
    assert "## Scope and Capabilities" in text
    assert "## Business Rules" not in text
    assert "Dashboard theo dõi" in text


@pytest.mark.asyncio
async def test_prd_export_empty_project_does_not_crash(client):
    headers, project = await _project_context(client)

    resp = await client.get(f"{BASE}/projects/{project['id']}/exports/prd.md", headers=headers)

    assert resp.status_code == 200, resp.text
    text = resp.text
    assert "# Product Requirements Document" in text
    assert "## Functional Requirement" in text
    assert "_Không có nội dung._" in text


@pytest.mark.asyncio
async def test_srs_export_route_removed(client):
    headers, project = await _project_context(client)

    resp = await client.get(f"{BASE}/projects/{project['id']}/exports/srs.md", headers=headers)

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_export_endpoints_reject_non_project_member(client):
    headers, project = await _project_context(client)
    outsider_headers = await make_auth_headers(client)
    await _requirements_artifact(client, headers, project["id"], REQUIREMENTS_BODY)

    for name in ["brd.md", "prd.md", "product-brief.md"]:
        resp = await client.get(f"{BASE}/projects/{project['id']}/exports/{name}", headers=outsider_headers)
        assert resp.status_code == 403


async def _project_context(client):
    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    return headers, project


async def _requirements_artifact(client, headers, project_id, body):
    create = await client.post(
        f"{BASE}/projects/{project_id}/documents/brd",
        headers=headers,
    )
    assert create.status_code == 201, create.text
    items = []
    for item_type, content in body.items():
        resp = await client.post(
            f"{BASE}/projects/{project_id}/documents/brd/{item_type}",
            json={
                "title": item_type.replace("_", " ").title(),
                "body": content,
                "status": "accepted",
                "priority": "must",
                "metadata": {},
            },
            headers=headers,
        )
        assert resp.status_code == 201, resp.text
        items.append(resp.json()["data"])
    return items


async def _source_document(client, headers, project_id, title, content):
    payload = {
        "title": title,
        "source_type": "text_paste",
        "content_text": content,
        "metadata": {},
    }
    resp = await client.post(f"{BASE}/projects/{project_id}/source-documents", json=payload, headers=headers)
    assert resp.status_code == 201, resp.text
    return resp.json()["data"]
