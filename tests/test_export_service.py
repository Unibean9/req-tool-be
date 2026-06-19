import pytest

from tests.conftest import BASE
from tests.helpers import create_org, create_project, make_auth_headers


@pytest.mark.asyncio
async def test_brd_export_has_business_sections_and_excludes_srs_only_artifacts(client):
    headers, project = await _project_context(client)
    await _artifact(client, headers, project["id"], "goal", "Tăng doanh thu", "Mục tiêu kinh doanh", priority="must")
    await _artifact(client, headers, project["id"], "problem", "Quy trình chậm", "Vấn đề nghiệp vụ")
    await _artifact(client, headers, project["id"], "business_rule", "Quy tắc phê duyệt", "Phải có phê duyệt")
    await _artifact(client, headers, project["id"], "domain_entity", "Đơn hàng", "Entity không thuộc BRD")
    await _artifact(client, headers, project["id"], "functional_requirement", "FR-1", "Không thuộc BRD")

    resp = await client.get(f"{BASE}/projects/{project['id']}/exports/brd.md", headers=headers)

    assert resp.status_code == 200, resp.text
    text = resp.text
    for heading in [
        "## Business Objectives",
        "## Problem Statement",
        "## Stakeholder Register",
        "## Business Scope",
        "## Business Rules",
        "## Constraints and Assumptions",
        "## Risks and Issues",
        "## Research Basis",
    ]:
        assert heading in text
    assert "FR-1" not in text
    assert "Đơn hàng" not in text


@pytest.mark.asyncio
async def test_product_brief_routes_personas_and_groups_research(client):
    headers, project = await _project_context(client)
    await _artifact(
        client,
        headers,
        project["id"],
        "stakeholder",
        "Người dùng cuối",
        "Persona hợp lệ",
        stakeholder_role="user_persona",
    )
    await _artifact(
        client,
        headers,
        project["id"],
        "stakeholder",
        "Giám đốc tài chính",
        "Business stakeholder không thuộc brief",
        stakeholder_role="business_stakeholder",
    )
    await _artifact(
        client,
        headers,
        project["id"],
        "research_output",
        "Phỏng vấn người dùng",
        "Người dùng cần dashboard",
        metadata={"research_type": "interview"},
    )
    await _artifact(client, headers, project["id"], "business_rule", "Quy tắc nội bộ", "Không thuộc brief")
    await _artifact(client, headers, project["id"], "epic", "Epic bán hàng", "Không thuộc brief")

    resp = await client.get(f"{BASE}/projects/{project['id']}/exports/product-brief.md", headers=headers)

    assert resp.status_code == 200, resp.text
    text = resp.text
    assert "## User Personas" in text
    assert "### Research: interview" in text
    assert "Người dùng cuối" in text
    assert "Phỏng vấn người dùng" in text
    assert "Giám đốc tài chính" not in text
    assert "Quy tắc nội bộ" not in text
    assert "Epic bán hàng" not in text


@pytest.mark.asyncio
async def test_srs_export_groups_frs_by_epic_nfr_by_category_and_embeds_acceptance_criteria(client):
    headers, project = await _project_context(client)
    epic = await _artifact(client, headers, project["id"], "epic", "Epic đăng nhập", "Nhóm đăng nhập")
    fr = await _artifact(client, headers, project["id"], "functional_requirement", "Đăng nhập email", "Cho phép đăng nhập")
    ac = await _artifact(client, headers, project["id"], "acceptance_criteria", "AC đăng nhập", "Given email hợp lệ")
    await _link(client, headers, project["id"], fr["id"], epic["id"], "derives_from")
    await _link(client, headers, project["id"], ac["id"], fr["id"], "validates")
    await _artifact(
        client,
        headers,
        project["id"],
        "non_functional_requirement",
        "Phản hồi nhanh",
        "P95 dưới 500ms",
        nfr_category="performance",
    )
    await _artifact(client, headers, project["id"], "problem", "BRD only", "Không thuộc SRS")

    resp = await client.get(f"{BASE}/projects/{project['id']}/exports/srs.md", headers=headers)

    assert resp.status_code == 200, resp.text
    text = resp.text
    assert "### Epic đăng nhập" in text
    assert "Đăng nhập email" in text
    assert "  - AC đăng nhập: Given email hợp lệ" in text
    assert "### NFR: performance" in text
    assert "Phản hồi nhanh" in text
    assert "BRD only" not in text
    assert "## Acceptance Criteria" not in text


@pytest.mark.asyncio
async def test_prd_export_contains_executive_sections_and_traceability(client):
    headers, project = await _project_context(client)
    goal = await _artifact(client, headers, project["id"], "goal", "Tăng chuyển đổi", "Vision")
    capability = await _artifact(client, headers, project["id"], "capability", "Thanh toán nhanh", "Capability")
    fr = await _artifact(client, headers, project["id"], "functional_requirement", "Lưu thẻ", "Requirement")
    await _link(client, headers, project["id"], fr["id"], goal["id"], "satisfies")
    await _link(client, headers, project["id"], capability["id"], goal["id"], "supports")

    resp = await client.get(f"{BASE}/projects/{project['id']}/exports/prd.md", headers=headers)

    assert resp.status_code == 200, resp.text
    text = resp.text
    assert "# Product Requirements Document" in text
    assert "## Executive Summary" in text
    assert "## Requirements Summary" in text
    assert "## Traceability" in text
    assert "Lưu thẻ -> Tăng chuyển đổi (satisfies)" in text


@pytest.mark.asyncio
async def test_export_moscow_ordering_wont_and_status_filters(client):
    headers, project = await _project_context(client)
    await _artifact(client, headers, project["id"], "goal", "Could item", "C", priority="could")
    await _artifact(client, headers, project["id"], "goal", "Must item", "M", priority="must")
    await _artifact(client, headers, project["id"], "goal", "Should item", "S", priority="should")
    await _artifact(client, headers, project["id"], "goal", "Unprioritized item", "U", priority=None)
    await _artifact(client, headers, project["id"], "goal", "Won't item", "W", priority="wont")
    await _artifact(client, headers, project["id"], "goal", "Draft item", "D", status="draft")
    await _artifact(client, headers, project["id"], "goal", "Rejected item", "R", status="rejected")

    default_resp = await client.get(f"{BASE}/projects/{project['id']}/exports/brd.md", headers=headers)
    include_wont_resp = await client.get(
        f"{BASE}/projects/{project['id']}/exports/brd.md",
        params={"include_wont": "true"},
        headers=headers,
    )

    text = default_resp.text
    assert text.index("Must item") < text.index("Should item") < text.index("Could item") < text.index("Unprioritized item")
    assert "Won't item" not in text
    assert "Draft item" not in text
    assert "Rejected item" not in text
    assert "Won't item" in include_wont_resp.text


@pytest.mark.asyncio
async def test_stakeholder_and_constraint_routing_between_brd_srs_and_brief(client):
    headers, project = await _project_context(client)
    await _artifact(
        client,
        headers,
        project["id"],
        "stakeholder",
        "Business Owner",
        "BRD stakeholder",
        stakeholder_role="business_stakeholder",
    )
    await _artifact(
        client,
        headers,
        project["id"],
        "stakeholder",
        "Persona A",
        "SRS persona",
        stakeholder_role="user_persona",
    )
    await _artifact(client, headers, project["id"], "constraint", "Business constraint", "Ràng buộc nghiệp vụ", metadata={"constraint_type": "business"})
    await _artifact(client, headers, project["id"], "constraint", "System constraint", "Ràng buộc hệ thống", metadata={"constraint_type": "system"})
    await _artifact(client, headers, project["id"], "constraint", "Both constraint", "Ràng buộc chung", metadata={"constraint_type": "both"})

    brd = (await client.get(f"{BASE}/projects/{project['id']}/exports/brd.md", headers=headers)).text
    srs = (await client.get(f"{BASE}/projects/{project['id']}/exports/srs.md", headers=headers)).text
    brief = (await client.get(f"{BASE}/projects/{project['id']}/exports/product-brief.md", headers=headers)).text

    assert "Business Owner" in brd
    assert "Persona A" not in brd
    assert "Business Owner" not in srs
    assert "Persona A" in srs
    assert "Business Owner" not in brief
    assert "Persona A" in brief
    assert "Business constraint" in brd
    assert "Business constraint" not in srs
    assert "System constraint" not in brd
    assert "System constraint" in srs
    assert "Both constraint" in brd
    assert "Both constraint" in srs


@pytest.mark.asyncio
async def test_export_endpoints_reject_non_project_member(client):
    headers, project = await _project_context(client)
    outsider_headers = await make_auth_headers(client)
    await _artifact(client, headers, project["id"], "goal", "Mục tiêu", "Nội dung")

    for name in ["brd.md", "srs.md", "prd.md", "product-brief.md"]:
        resp = await client.get(f"{BASE}/projects/{project['id']}/exports/{name}", headers=outsider_headers)
        assert resp.status_code == 403


async def _project_context(client):
    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    return headers, project


async def _artifact(
    client,
    headers,
    project_id,
    artifact_type,
    title,
    body,
    *,
    priority="must",
    status="accepted",
    stakeholder_role=None,
    nfr_category=None,
    metadata=None,
):
    payload = {
        "type": artifact_type,
        "title": title,
        "body": body,
        "status": status,
        "priority": priority,
        "stakeholder_role": stakeholder_role,
        "nfr_category": nfr_category,
        "metadata": metadata or {},
    }
    resp = await client.post(f"{BASE}/projects/{project_id}/artifacts", json=payload, headers=headers)
    assert resp.status_code == 201, resp.text
    return resp.json()["data"]


async def _link(client, headers, project_id, source_id, target_id, relation_type):
    resp = await client.post(
        f"{BASE}/projects/{project_id}/artifact-links",
        json={"source_artifact_id": source_id, "target_artifact_id": target_id, "relation_type": relation_type},
        headers=headers,
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["data"]
