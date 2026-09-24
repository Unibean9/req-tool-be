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

A group is a module (UseCaseModule), not a use case -- an earlier version of generate_groups
also minted a redundant "L0" UseCaseEntry per group (id like UC-SUM-*) to stand in for the module,
which made the module indistinguishable from a real use case in the table/diagram. These tests
exercise the current shape: generate_groups returns `modules` (no use_cases at all), and
generate_group_use_cases takes a module id and produces flat use-case rows under it. The canonical
model has no L0/L1/L2 hierarchy -- every draft becomes an independent top-level row, and legacy
level/local_tag/parent_local_tag hints in a draft payload are accepted and silently ignored rather
than used to nest rows.
"""

import uuid
from datetime import UTC, datetime, timedelta
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
        "modules": [
            {"id": "SUB-TASK-MANAGEMENT", "name": "Task Management", "goal": "Let a team track its work."},
        ],
        "useCases": [],
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

    assert response.use_cases == []
    assert len(response.modules) == 1
    assert response.modules[0].name == "Task Management"
    assert response.modules[0].goal == "Let a team track its work."
    assert response.modules[0].id.startswith("SUB-")
    assert {actor.name for actor in response.actors} == {"Group Member", "Group Admin"}


@pytest.mark.asyncio
async def test_generate_groups_returns_an_empty_model_with_a_validation_error_when_nothing_could_be_extracted():
    """Phase 1 never raises for an unparseable source: when the LLM draft comes back empty and the
    deterministic completion fallback also finds nothing to extract, generate_groups still returns
    200 with an empty model. The validation report carries a NO_USE_CASES error so the FE can
    surface it instead of silently showing an empty catalog with no indication anything went wrong."""
    service = UseCaseService(db=None)
    project = SimpleNamespace(id=PROJECT_ID, name="Task App")
    provider = SimpleNamespace(
        id=uuid.uuid4(), provider_type=SimpleNamespace(value="anthropic"), model_name="claude-test"
    )
    draft_response = ({"actors": [], "groups": []}, {})
    empty_source = RequirementsSourceSnapshot(
        project_id=str(PROJECT_ID),
        brd=StoredDocumentSnapshot(document_type="brd", label="BRD", components=[]),
        prd=StoredDocumentSnapshot(document_type="prd", label="PRD", components=[]),
        components=[],
        evidence=[],
        source_hash="empty",
    )
    service._project = AsyncMock(return_value=project)
    service._record = AsyncMock(return_value=SimpleNamespace(model_data=None))
    service._llm_client = AsyncMock(
        return_value=(SimpleNamespace(generate=AsyncMock(return_value=draft_response)), provider)
    )
    service._save = AsyncMock()
    service.db = AsyncMock()

    with patch("app.services.use_case_service.load_project_requirements_source", AsyncMock(return_value=empty_source)):
        response = await service.generate_groups(project_id=PROJECT_ID, user_id=USER_ID, body=_body())

    assert response.modules == []
    assert response.use_cases == []
    assert response.validation is not None
    assert any(issue.code == "NO_USE_CASES" for issue in response.validation.issues)
    assert response.validation.eligible_for_srs is False


# ---------------------------------------------------------------------------
# Split generation's shared "running" marker. generate_groups is the only phase that checks
# it (see UseCaseService._mark_generation_started) -- a second overlapping attempt (a page
# reload plus another click while a previous run is still mid-flight server-side) is
# rejected immediately with 409 instead of queuing behind generate_group_use_cases' /
# generate_relations' row lock and eventually failing as an opaque, unlogged 500. A marker
# that stopped being refreshed (a crashed or --reload-killed request) is treated as
# abandoned rather than locking the project out of generating forever.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_groups_rejects_a_second_call_while_one_is_genuinely_running():
    from fastapi import HTTPException

    now = datetime.now(UTC).isoformat()
    payload = _payload_after_generate_groups()
    payload["generation"] = {"status": "running", "generationId": "gen-1", "startedAt": now, "updatedAt": now}
    service = UseCaseService(db=AsyncMock())
    service._project = AsyncMock(return_value=SimpleNamespace(id=PROJECT_ID, name="Task App"))
    service._record = AsyncMock(return_value=SimpleNamespace(model_data=payload))

    with patch("app.services.use_case_service.load_project_requirements_source", AsyncMock(return_value=_source())):
        with pytest.raises(HTTPException) as exc_info:
            await service.generate_groups(project_id=PROJECT_ID, user_id=USER_ID, body=_body())

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail["code"] == "USE_CASE_GENERATION_IN_PROGRESS"


@pytest.mark.asyncio
async def test_generate_groups_recovers_from_a_stale_running_marker():
    """A marker last touched well past the staleness window (a crashed/killed request, not a
    genuinely slow one) must not permanently lock a project out of generating again."""
    stale = (datetime.now(UTC) - timedelta(minutes=30)).isoformat()
    payload = _payload_after_generate_groups()
    payload["generation"] = {"status": "running", "generationId": "gen-old", "startedAt": stale, "updatedAt": stale}
    provider = SimpleNamespace(
        id=uuid.uuid4(), provider_type=SimpleNamespace(value="anthropic"), model_name="claude-test"
    )
    draft_response = ({"actors": [], "groups": []}, {})
    service = UseCaseService(db=AsyncMock())
    service._project = AsyncMock(return_value=SimpleNamespace(id=PROJECT_ID, name="Task App"))
    service._record = AsyncMock(return_value=SimpleNamespace(model_data=payload))
    service._llm_client = AsyncMock(
        return_value=(SimpleNamespace(generate=AsyncMock(return_value=draft_response)), provider)
    )
    service._save = AsyncMock()

    with patch("app.services.use_case_service.load_project_requirements_source", AsyncMock(return_value=_source())):
        response = await service.generate_groups(project_id=PROJECT_ID, user_id=USER_ID, body=_body())

    assert response.generation is not None
    assert response.generation.get("status") == "running"
    assert response.generation.get("generationId") != "gen-old"


@pytest.mark.asyncio
async def test_generate_relations_completes_gracefully_with_no_use_cases_instead_of_erroring():
    """generate_relations is split generation's last phase, so it is also responsible for
    resolving the running marker to a terminal state -- even when there is nothing to relate
    (every module failed, or the source was empty). Erroring here instead would leave that
    case stuck reporting "running" forever."""
    now = datetime.now(UTC).isoformat()
    payload = _payload_after_generate_groups()
    payload["generation"] = {"status": "running", "generationId": "gen-1", "startedAt": now, "updatedAt": now}
    record = SimpleNamespace(model_data=payload)
    service = UseCaseService(db=AsyncMock())
    service._project = AsyncMock(return_value=SimpleNamespace(id=PROJECT_ID, name="Task App"))
    service._locked_payload = AsyncMock(return_value=(payload, record))
    service._save = AsyncMock()

    response = await service.generate_relations(project_id=PROJECT_ID, user_id=USER_ID, body=_body())

    assert response.generation is not None
    assert response.generation.get("status") == "completed"
    assert response.generation.get("relationsGenerated") is False
    service._save.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_group_use_cases_bounds_the_prompt_to_the_modules_own_source_refs():
    """generate_groups (Phase 1) already tags each module with the evidence it read that module
    from (module["sourceRefs"]). generate_group_use_cases must use that -- not send the whole
    BRD/PRD to every module's request -- both so the prompt stays small and so content from an
    unrelated FR cannot leak into a module it was never attributed to."""
    functional_body = """
## Functional Requirements
| ID | Requirement | Behavior | Priority |
| --- | --- | --- | --- |
| FR-TM01 | Tao task don le | Truong nhom tao task moi. | Must |
| FR-XY99 | Unrelated other domain requirement | Should never appear in this module's prompt. | Must |
""".strip()
    brd = StoredDocumentSnapshot(document_type="brd", label="BRD", components=[])
    prd = StoredDocumentSnapshot(
        document_type="prd",
        label="PRD",
        components=[
            StoredComponentSnapshot(
                document_type="prd", artifact_type="functional_requirement", label="FRs", body=functional_body
            ),
        ],
    )
    source = RequirementsSourceSnapshot(
        project_id=str(PROJECT_ID),
        brd=brd,
        prd=prd,
        components=[*prd.components],
        evidence=[
            SourceEvidence(
                evidence_id="entity:prd:functional_requirement:FR-TM01:5",
                document_type="prd",
                artifact_type="functional_requirement",
                kind="functional_requirement",
                locator="prd.functional_requirement:L5",
                excerpt="FR-TM01",
                entity_id="FR-TM01",
            ),
        ],
        source_hash="test-source",
    )
    payload = _payload_after_generate_groups()
    payload["modules"][0]["sourceRefs"] = ["entity:prd:functional_requirement:FR-TM01:5"]
    service = _service_with_mocks(payload, generate_result=({"use_cases": []}, {}))

    with patch("app.services.use_case_service.load_project_requirements_source", AsyncMock(return_value=source)):
        await service.generate_group_use_cases(
            project_id=PROJECT_ID, user_id=USER_ID, group_id="SUB-TASK-MANAGEMENT", body=_body()
        )

    prompt = service._llm_client.return_value[0].generate.call_args.kwargs["messages"][0]["content"]
    assert "FR-TM01" in prompt
    assert "FR-XY99" not in prompt


# ---------------------------------------------------------------------------
# generate_diagram: its own step, deliberately separate from table generation (no LLM call,
# requires a table to already exist) so the FE can gate a "Generate diagram" action behind
# "Generate table" having produced at least one use case.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_diagram_requires_use_cases_first():
    from fastapi import HTTPException

    payload = _payload_after_generate_groups()
    record = SimpleNamespace(model_data=payload)
    service = UseCaseService(db=AsyncMock())
    service._project = AsyncMock(return_value=SimpleNamespace(id=PROJECT_ID, name="Task App"))
    service._locked_payload = AsyncMock(return_value=(payload, record))

    with pytest.raises(HTTPException) as exc_info:
        await service.generate_diagram(project_id=PROJECT_ID)

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail["code"] == "USE_CASE_TABLE_REQUIRED"


@pytest.mark.asyncio
async def test_generate_diagram_recomputes_layout_when_use_cases_exist():
    payload = _payload_after_generate_groups()
    payload["useCases"] = [
        {
            "id": "UC-TMM-001",
            "name": "Add Team Member",
            "moduleId": "SUB-TASK-MANAGEMENT",
            "primaryActorId": "ACT-GROUP-MEMBER",
            "secondaryActorIds": [],
            "evidence": "explicit",
            "priority": "required",
            "description": "A member is added to the team.",
            "preconditions": [],
            "sourceTrace": [],
        }
    ]
    record = SimpleNamespace(model_data=payload)
    service = UseCaseService(db=AsyncMock())
    service._project = AsyncMock(return_value=SimpleNamespace(id=PROJECT_ID, name="Task App"))
    service._locked_payload = AsyncMock(return_value=(payload, record))
    service._attach_diagram_layout = AsyncMock(return_value=None)
    service._save = AsyncMock()

    response = await service.generate_diagram(project_id=PROJECT_ID)

    service._attach_diagram_layout.assert_awaited_once()
    service._save.assert_awaited_once()
    assert len(response.use_cases) == 1


# ---------------------------------------------------------------------------
# generate_group_use_cases (Phase 2)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_group_use_cases_produces_flat_rows_ignoring_legacy_level_tags():
    """The canonical model has no use-case hierarchy: every draft becomes an independent top-level
    row under the module. Legacy level/local_tag/parent_local_tag hints (from an older provider
    prompt or a stale client) are accepted and silently dropped, not used to nest rows."""
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
            project_id=PROJECT_ID, user_id=USER_ID, group_id="SUB-TASK-MANAGEMENT", body=_body()
        )

    names = {item.name for item in response.use_cases}
    assert names == {"Create Task", "Assign Task Owner"}
    assert all(item.module_id == "SUB-TASK-MANAGEMENT" for item in response.use_cases)
    assert all(item.id.startswith("UC-") for item in response.use_cases)
    assign = next(item for item in response.use_cases if item.name == "Assign Task Owner")
    assert assign.secondary_actor_ids == ["ACT-GROUP-ADMIN"]
    service._save.assert_awaited_once()


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
            project_id=PROJECT_ID, user_id=USER_ID, group_id="SUB-TASK-MANAGEMENT", body=_body()
        )

    assert response.use_cases == []


@pytest.mark.asyncio
async def test_generate_group_use_cases_is_idempotent_per_group():
    """Re-running a module replaces its previously generated rows instead of duplicating them.
    Matched by module id -- there is no module-parent use-case id anymore."""
    payload = _payload_after_generate_groups()
    payload["useCases"].append(
        {
            "id": "UC-STALE-OLD-1",
            "name": "Stale Old Draft",
            "moduleId": "SUB-TASK-MANAGEMENT",
            "primaryActorId": "ACT-GROUP-MEMBER",
            "secondaryActorIds": [],
            "evidence": "inferred",
            "priority": "optional",
            "description": "A previous generation's row that should be replaced, not duplicated.",
            "preconditions": [],
            "sourceTrace": [],
        }
    )
    payload["relationships"].append(
        {"id": "REL-STALE", "sourceId": "UC-STALE-OLD-1", "targetId": "UC-STALE-OLD-1", "type": "include"},
    )
    draft_response = (
        {
            "use_cases": [
                {
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
            project_id=PROJECT_ID, user_id=USER_ID, group_id="SUB-TASK-MANAGEMENT", body=_body()
        )

    names = {item.name for item in response.use_cases}
    assert names == {"Fresh Draft"}
    assert not any(item.id == "UC-STALE-OLD-1" for item in response.use_cases)
    assert not any(
        r.source_id == "UC-STALE-OLD-1" or r.target_id == "UC-STALE-OLD-1" for r in response.relationships
    )


@pytest.mark.asyncio
async def test_generate_group_use_cases_caps_per_module_and_keeps_highest_priority():
    """Concept-stage cap: the table stays small enough to read/diagram at a glance. The budget
    is split evenly across modules (patched to 4 total here, 2 modules -> 2 each) so one module
    cannot exhaust it, and within a module lower-priority drafts are dropped first."""
    payload = _payload_after_generate_groups()
    payload["modules"].append({"id": "SUB-REPORTING", "name": "Reporting", "goal": "See progress."})
    draft_response = (
        {
            "use_cases": [
                {
                    "name": "Nice to Have Export",
                    "primary_actor_id": "ACT-GROUP-MEMBER",
                    "secondary_actor_ids": [],
                    "description": "Lowest priority, should be dropped.",
                    "precondition": "n/a",
                    "priority": "optional",
                    "source_refs": ["entity:prd:use_case:C1:5"],
                },
                {
                    "name": "Create Task",
                    "primary_actor_id": "ACT-GROUP-MEMBER",
                    "secondary_actor_ids": [],
                    "description": "Highest priority, must be kept.",
                    "precondition": "n/a",
                    "priority": "required",
                    "source_refs": ["entity:prd:use_case:C1:5"],
                },
                {
                    "name": "Assign Task Owner",
                    "primary_actor_id": "ACT-GROUP-MEMBER",
                    "secondary_actor_ids": [],
                    "description": "Middle priority, must be kept over optional.",
                    "precondition": "n/a",
                    "priority": "recommended",
                    "source_refs": ["entity:prd:use_case:C1:5"],
                },
            ]
        },
        {},
    )
    service = _service_with_mocks(payload, generate_result=draft_response)

    with (
        patch("app.services.use_case_service._MAX_GENERATED_USE_CASES", 4),
        patch("app.services.use_case_service.load_project_requirements_source", AsyncMock(return_value=_source())),
    ):
        response = await service.generate_group_use_cases(
            project_id=PROJECT_ID, user_id=USER_ID, group_id="SUB-TASK-MANAGEMENT", body=_body()
        )

    names = {item.name for item in response.use_cases}
    assert names == {"Create Task", "Assign Task Owner"}
    assert "Nice to Have Export" not in names


@pytest.mark.asyncio
async def test_generate_group_use_cases_cap_reserves_a_slot_per_actor_before_filling_by_priority():
    """The priority cut must not be allowed to zero out an actor entirely. Here the two
    highest-priority drafts both belong to the same actor and the budget is 2 -- picking purely
    by priority would keep both of that actor's rows and drop the other actor's only row, even
    though it is lower priority, leaving that actor with no use case anywhere in the diagram."""
    payload = _payload_after_generate_groups()
    draft_response = (
        {
            "use_cases": [
                {
                    "name": "Member Primary Required",
                    "primary_actor_id": "ACT-GROUP-MEMBER",
                    "secondary_actor_ids": [],
                    "description": "Highest priority, actor already has one selected.",
                    "precondition": "n/a",
                    "priority": "required",
                    "source_refs": ["entity:prd:use_case:C1:5"],
                },
                {
                    "name": "Member Secondary Recommended",
                    "primary_actor_id": "ACT-GROUP-MEMBER",
                    "secondary_actor_ids": [],
                    "description": "Second-highest priority, but same actor as the row above.",
                    "precondition": "n/a",
                    "priority": "recommended",
                    "source_refs": ["entity:prd:use_case:C1:5"],
                },
                {
                    "name": "Admin Only Optional",
                    "primary_actor_id": "ACT-GROUP-ADMIN",
                    "secondary_actor_ids": [],
                    "description": "Lowest priority, but the only row for this actor -- must survive the cut.",
                    "precondition": "n/a",
                    "priority": "optional",
                    "source_refs": ["entity:prd:use_case:C1:5"],
                },
            ]
        },
        {},
    )
    service = _service_with_mocks(payload, generate_result=draft_response)

    with (
        patch("app.services.use_case_service._MAX_GENERATED_USE_CASES", 2),
        patch("app.services.use_case_service.load_project_requirements_source", AsyncMock(return_value=_source())),
    ):
        response = await service.generate_group_use_cases(
            project_id=PROJECT_ID, user_id=USER_ID, group_id="SUB-TASK-MANAGEMENT", body=_body()
        )

    names = {item.name for item in response.use_cases}
    assert names == {"Member Primary Required", "Admin Only Optional"}
    assert "Member Secondary Recommended" not in names


@pytest.mark.asyncio
async def test_generate_group_use_cases_unknown_group_is_404():
    from fastapi import HTTPException

    payload = _payload_after_generate_groups()
    service = _service_with_mocks(payload, generate_result=({"use_cases": []}, {}))

    with pytest.raises(HTTPException) as exc_info:
        await service.generate_group_use_cases(
            project_id=PROJECT_ID, user_id=USER_ID, group_id="SUB-DOES-NOT-EXIST", body=_body()
        )
    assert exc_info.value.status_code == 404
