"""UseCaseService.generate_groups / generate_group_use_cases: the split generation flow added to
fix use-case generation timing out on a larger project (the old single call had to enumerate every
group's use cases at once). DB/document-loading is mocked out at the service's own seams
(_project/_locked_payload/_llm_client/_save/_record, and the module-level
load_project_requirements_source) so this exercises the real merge/validation/idempotency logic
without a full project fixture or a real LLM call.

generate_groups calls the model too (an earlier version parsed the Business Capabilities doc with
a regex tied to one fixed "BC-xx" heading/ID convention and silently returned zero groups for any
project using a different one -- exactly what happened on a real project whose capabilities were a
numbered table with "C1"/"C2"/... ids instead).
"""

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.schemas.use_case import UseCaseLevel
from app.services.use_case_service import UseCaseService
from app.use_cases.models import (
    RequirementsSourceSnapshot,
    SourceEvidence,
    StoredComponentSnapshot,
    StoredDocumentSnapshot,
)

PROJECT_ID = uuid.uuid4()
USER_ID = uuid.uuid4()


def _source() -> RequirementsSourceSnapshot:
    # Deliberately NOT the "### BC-xx:" convention -- a numbered-table format like a real project
    # used, to prove generate_groups no longer assumes one fixed heading/ID scheme.
    capability_body = """
## Business Capabilities

### Domain 1: Task Management
| ID | Capability | Goal | User Segment |
|----|-----------|------|--------------|
| C1 | Create Task | Let a member create a task. | Group Member, Group Admin |
""".strip()
    brd = StoredDocumentSnapshot(document_type="brd", label="BRD", components=[])
    prd = StoredDocumentSnapshot(
        document_type="prd",
        label="PRD",
        components=[
            StoredComponentSnapshot(
                document_type="prd", artifact_type="use_case", label="Business Capabilities", body=capability_body
            ),
        ],
    )
    evidence = [
        SourceEvidence(
            evidence_id="entity:prd:use_case:C1:5",
            document_type="prd",
            artifact_type="use_case",
            kind="business_capability",
            locator="prd.use_case:L5",
            excerpt="C1",
            entity_id="C1",
        ),
    ]
    return RequirementsSourceSnapshot(
        project_id=str(PROJECT_ID), brd=brd, prd=prd, components=[*prd.components], evidence=evidence,
        source_hash="test-source",
    )


def _payload_after_generate_groups() -> dict:
    return {
        "projectId": str(PROJECT_ID),
        "projectName": "Task App",
        "actors": [
            {"id": "ACT-GROUP-MEMBER", "name": "Group Member", "kind": "Primary actor"},
            {"id": "ACT-GROUP-ADMIN", "name": "Group Admin", "kind": "Supporting actor"},
        ],
        "useCases": [
            {
                "id": "UC-SUM-TASK-MANAGEMENT",
                "level": "L0",
                "title": "Task Management",
                "primaryActorId": "ACT-GROUP-MEMBER",
                "supportingActorIds": [],
                "subsystem": "Task Management",
                "status": "Inferred",
                "priority": "Should",
                "parentUseCaseId": None,
                "description": "The actor can manage tasks.",
                "precondition": "The project is available to the actor.",
                "sourceTrace": [],
            },
        ],
        "relationships": [],
        "diagrams": [],
        "diagramPlans": [],
        "sourceHash": "test-source",
        "validation": None,
        "generation": {"source": "ai", "relationsGenerated": False},
    }


def _service_with_mocks(payload: dict, *, generate_result):
    service = UseCaseService(db=None)
    project = SimpleNamespace(id=PROJECT_ID, name="Task App")
    record = SimpleNamespace(model_data=None)
    provider = SimpleNamespace(
        id=uuid.uuid4(), provider_type=SimpleNamespace(value="anthropic"), model_name="claude-test"
    )
    fake_client = SimpleNamespace(generate=AsyncMock(return_value=generate_result))

    service._project = AsyncMock(return_value=project)
    service._locked_payload = AsyncMock(return_value=(payload, record))
    service._record = AsyncMock(return_value=record)
    service._llm_client = AsyncMock(return_value=(fake_client, provider))
    service._save = AsyncMock()
    return service


def _body():
    return SimpleNamespace(provider_config_id=None, max_level=UseCaseLevel.L2)


# ---------------------------------------------------------------------------
# generate_groups (Phase 1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_groups_reads_whatever_format_the_capabilities_doc_actually_uses():
    """The whole point of the fix: a numbered-table capability doc (not "### BC-xx:") still
    produces groups, because this is a real (mocked) model call, not a fixed-convention regex."""
    service = UseCaseService(db=None)
    project = SimpleNamespace(id=PROJECT_ID, name="Task App")
    provider = SimpleNamespace(
        id=uuid.uuid4(), provider_type=SimpleNamespace(value="anthropic"), model_name="claude-test"
    )
    draft_response = (
        {
            "actors": [
                {"name": "Group Member", "kind": "human_role", "source_refs": ["entity:prd:use_case:C1:5"]},
                {"name": "Group Admin", "kind": "human_role", "source_refs": ["entity:prd:use_case:C1:5"]},
            ],
            "groups": [
                {
                    "name": "Task Management",
                    "goal": "Let a team track its work.",
                    "user_segment": ["Group Member", "Group Admin"],
                    "source_refs": ["entity:prd:use_case:C1:5"],
                }
            ],
        },
        {"input": 200, "output": 60},
    )
    service._project = AsyncMock(return_value=project)
    service._record = AsyncMock(return_value=SimpleNamespace(model_data=None))
    service._llm_client = AsyncMock(
        return_value=(SimpleNamespace(generate=AsyncMock(return_value=draft_response)), provider)
    )
    service._save = AsyncMock()
    service.db = AsyncMock()

    with patch("app.services.use_case_service.load_project_requirements_source", AsyncMock(return_value=_source())):
        response = await service.generate_groups(project_id=PROJECT_ID, user_id=USER_ID, body=_body())

    groups = [item for item in response.use_cases if item.level == "L0"]
    assert len(groups) == 1
    assert groups[0].title == "Task Management"
    assert {actor.name for actor in response.actors} == {"Group Member", "Group Admin"}
    # Every actor referenced by the group has an association relation.
    assert any(
        r.source_id == groups[0].primary_actor_id and r.target_id == groups[0].id and r.type == "association"
        for r in response.relationships
    )


@pytest.mark.asyncio
async def test_generate_groups_rejects_when_nothing_has_a_user_segment():
    """An earlier deterministic version silently produced an empty (but "successful") model when
    it couldn't parse the source. This version fails loudly instead, so the FE surfaces an error
    rather than showing an empty catalog with no indication anything went wrong."""
    from fastapi import HTTPException

    service = UseCaseService(db=None)
    project = SimpleNamespace(id=PROJECT_ID, name="Task App")
    provider = SimpleNamespace(
        id=uuid.uuid4(), provider_type=SimpleNamespace(value="anthropic"), model_name="claude-test"
    )
    draft_response = (
        {
            "actors": [],
            "groups": [
                {"name": "Orphan Group", "goal": "n/a", "user_segment": [], "source_refs": ["entity:prd:use_case:C1:5"]}
            ],
        },
        {},
    )
    service._project = AsyncMock(return_value=project)
    service._record = AsyncMock(return_value=SimpleNamespace(model_data=None))
    service._llm_client = AsyncMock(
        return_value=(SimpleNamespace(generate=AsyncMock(return_value=draft_response)), provider)
    )
    service._save = AsyncMock()

    with patch("app.services.use_case_service.load_project_requirements_source", AsyncMock(return_value=_source())):
        with pytest.raises(HTTPException) as exc_info:
            await service.generate_groups(project_id=PROJECT_ID, user_id=USER_ID, body=_body())
    assert exc_info.value.status_code == 502


# ---------------------------------------------------------------------------
# generate_group_use_cases (Phase 2)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_group_use_cases_merges_l1_and_l2_drafts_via_local_tags():
    payload = _payload_after_generate_groups()
    draft_response = (
        {
            "use_cases": [
                {
                    "level": "L1",
                    "local_tag": "G1",
                    "parent_local_tag": None,
                    "name": "Create Task",
                    "primary_actor_id": "ACT-GROUP-MEMBER",
                    "secondary_actor_ids": [],
                    "description": "The Group Member can create a task.",
                    "precondition": "The project exists.",
                    "priority": "must",
                    "source_refs": ["entity:prd:use_case:C1:5"],
                },
                {
                    "level": "L2",
                    "local_tag": "G1-A",
                    "parent_local_tag": "G1",
                    "name": "Assign Task Owner",
                    "primary_actor_id": "ACT-GROUP-MEMBER",
                    "secondary_actor_ids": ["ACT-GROUP-ADMIN"],
                    "description": "The Group Member can assign a task owner.",
                    "precondition": "The task exists.",
                    "priority": "should",
                    "source_refs": ["entity:prd:use_case:C1:5"],
                },
            ]
        },
        {"input": 150, "output": 80},
    )
    service = _service_with_mocks(payload, generate_result=draft_response)

    with patch("app.services.use_case_service.load_project_requirements_source", AsyncMock(return_value=_source())):
        response = await service.generate_group_use_cases(
            project_id=PROJECT_ID, user_id=USER_ID, group_id="UC-SUM-TASK-MANAGEMENT", body=_body()
        )

    l1_rows = [item for item in response.use_cases if item.level == "L1"]
    l2_rows = [item for item in response.use_cases if item.level == "L2"]
    assert len(l1_rows) == 1 and l1_rows[0].title == "Create Task"
    assert l1_rows[0].parent_use_case_id == "UC-SUM-TASK-MANAGEMENT"
    assert len(l2_rows) == 1 and l2_rows[0].title == "Assign Task Owner"
    assert l2_rows[0].parent_use_case_id == l1_rows[0].id
    assert l2_rows[0].supporting_actor_ids == ["ACT-GROUP-ADMIN"]
    service._save.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_group_use_cases_drops_l2_whose_parent_tag_does_not_resolve():
    payload = _payload_after_generate_groups()
    draft_response = (
        {
            "use_cases": [
                {
                    "level": "L2",
                    "local_tag": "orphan",
                    "parent_local_tag": "does-not-exist",
                    "name": "Orphan Sub-Use-Case",
                    "primary_actor_id": "ACT-GROUP-MEMBER",
                    "secondary_actor_ids": [],
                    "description": "Should be dropped: parent tag never resolved.",
                    "precondition": "n/a",
                    "priority": "could",
                    "source_refs": ["entity:prd:use_case:C1:5"],
                }
            ]
        },
        {},
    )
    service = _service_with_mocks(payload, generate_result=draft_response)

    with patch("app.services.use_case_service.load_project_requirements_source", AsyncMock(return_value=_source())):
        response = await service.generate_group_use_cases(
            project_id=PROJECT_ID, user_id=USER_ID, group_id="UC-SUM-TASK-MANAGEMENT", body=_body()
        )

    assert [item for item in response.use_cases if item.level == "L2"] == []


@pytest.mark.asyncio
async def test_generate_group_use_cases_drops_drafts_with_unknown_actor():
    payload = _payload_after_generate_groups()
    draft_response = (
        {
            "use_cases": [
                {
                    "level": "L1",
                    "local_tag": "G1",
                    "parent_local_tag": None,
                    "name": "Invented Actor",
                    "primary_actor_id": "ACT-DOES-NOT-EXIST",
                    "secondary_actor_ids": [],
                    "description": "Should be dropped: actor was never given to the model.",
                    "precondition": "n/a",
                    "priority": "could",
                    "source_refs": ["entity:prd:use_case:C1:5"],
                }
            ]
        },
        {},
    )
    service = _service_with_mocks(payload, generate_result=draft_response)

    with patch("app.services.use_case_service.load_project_requirements_source", AsyncMock(return_value=_source())):
        response = await service.generate_group_use_cases(
            project_id=PROJECT_ID, user_id=USER_ID, group_id="UC-SUM-TASK-MANAGEMENT", body=_body()
        )

    assert [item for item in response.use_cases if item.level in {"L1", "L2"}] == []


@pytest.mark.asyncio
async def test_generate_group_use_cases_is_idempotent_per_group():
    """Re-running a group replaces its previously generated L1/L2 rows instead of duplicating
    them."""
    payload = _payload_after_generate_groups()
    # Deliberately NOT "UC-L1-001"/"UC-L2-001" -- that's the exact id _next_use_case_id would
    # mint next, which would make a freshly-generated row LOOK like a leaked stale one by pure id
    # coincidence. Use ids the id scheme would never naturally produce, so a leak is unambiguous.
    payload["useCases"].append(
        {
            "id": "UC-STALE-OLD-L1",
            "level": "L1",
            "title": "Stale Old Draft",
            "primaryActorId": "ACT-GROUP-MEMBER",
            "supportingActorIds": [],
            "subsystem": "Task Management",
            "status": "Inferred",
            "priority": "Could",
            "parentUseCaseId": "UC-SUM-TASK-MANAGEMENT",
            "description": "A previous generation's row that should be replaced, not duplicated.",
            "precondition": "n/a",
            "sourceTrace": [],
        }
    )
    payload["useCases"].append(
        {
            "id": "UC-STALE-OLD-L2",
            "level": "L2",
            "title": "Stale Old Sub-Draft",
            "primaryActorId": "ACT-GROUP-MEMBER",
            "supportingActorIds": [],
            "subsystem": "Task Management",
            "status": "Inferred",
            "priority": "Could",
            "parentUseCaseId": "UC-STALE-OLD-L1",
            "description": "A previous generation's L2 row under the stale L1.",
            "precondition": "n/a",
            "sourceTrace": [],
        }
    )
    payload["relationships"].extend(
        [
            {
                "id": "REL-PART-OF-1",
                "sourceId": "UC-SUM-TASK-MANAGEMENT",
                "targetId": "UC-STALE-OLD-L1",
                "type": "part-of",
            },
            {"id": "REL-PART-OF-2", "sourceId": "UC-STALE-OLD-L1", "targetId": "UC-STALE-OLD-L2", "type": "part-of"},
        ]
    )
    draft_response = (
        {
            "use_cases": [
                {
                    "level": "L1",
                    "local_tag": "G1",
                    "parent_local_tag": None,
                    "name": "Fresh Draft",
                    "primary_actor_id": "ACT-GROUP-MEMBER",
                    "secondary_actor_ids": [],
                    "description": "The new generation's row.",
                    "precondition": "n/a",
                    "priority": "should",
                    "source_refs": ["entity:prd:use_case:C1:5"],
                }
            ]
        },
        {},
    )
    service = _service_with_mocks(payload, generate_result=draft_response)

    with patch("app.services.use_case_service.load_project_requirements_source", AsyncMock(return_value=_source())):
        response = await service.generate_group_use_cases(
            project_id=PROJECT_ID, user_id=USER_ID, group_id="UC-SUM-TASK-MANAGEMENT", body=_body()
        )

    l1_titles = {item.title for item in response.use_cases if item.level == "L1"}
    assert l1_titles == {"Fresh Draft"}
    assert not any(item.level == "L2" and item.title == "Stale Old Sub-Draft" for item in response.use_cases)
    assert not any(item.id in {"UC-STALE-OLD-L1", "UC-STALE-OLD-L2"} for item in response.use_cases)
    assert not any(r.target_id in {"UC-STALE-OLD-L1", "UC-STALE-OLD-L2"} for r in response.relationships)


@pytest.mark.asyncio
async def test_generate_group_use_cases_unknown_group_is_404():
    from fastapi import HTTPException

    payload = _payload_after_generate_groups()
    service = _service_with_mocks(payload, generate_result=({"use_cases": []}, {}))

    with pytest.raises(HTTPException) as exc_info:
        await service.generate_group_use_cases(
            project_id=PROJECT_ID, user_id=USER_ID, group_id="UC-SUM-DOES-NOT-EXIST", body=_body()
        )
    assert exc_info.value.status_code == 404
