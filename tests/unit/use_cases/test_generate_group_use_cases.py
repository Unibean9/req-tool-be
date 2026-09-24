"""UseCaseService.generate_groups (pipeline step 1), generate_diagram and diagram positions. The
later pipeline steps are covered in test_generation_pipeline.py. DB/document-loading is mocked out at the service's own seams
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
exercise the current shape: generate_groups returns `modules` (no use_cases at all).
"""

import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.schemas.use_case import DiagramNodePositionRequest, UseCaseLevel
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
# rejected immediately with 409 instead of queuing behind the later steps'
# row lock and eventually failing as an opaque, unlogged 500. A marker
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



def test_drop_actors_without_use_cases_keeps_only_referenced_actors():
    """Deterministic, not AI-predicted: generate_groups (Phase 1) proposes the actor roster
    before any use case exists, so it cannot know in advance which of them a later module batch
    will actually assign. This runs once the full use-case set is known (see
    _resolve_relationships_with_client, table generation's last step) and removes whichever
    actor never ended up as anyone's primary (or a legacy secondary) actor."""
    payload = {
        "actors": [
            {"id": "ACT-USED-PRIMARY", "name": "Used as primary"},
            {"id": "ACT-USED-SECONDARY", "name": "Used as legacy secondary"},
            {"id": "ACT-UNUSED", "name": "Never assigned to anything"},
        ],
        "useCases": [
            {"id": "UC-1", "primaryActorId": "ACT-USED-PRIMARY", "secondaryActorIds": ["ACT-USED-SECONDARY"]},
        ],
    }

    UseCaseService._drop_actors_without_use_cases(payload)

    assert {actor["id"] for actor in payload["actors"]} == {"ACT-USED-PRIMARY", "ACT-USED-SECONDARY"}




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


def _payload_with_diagram() -> dict:
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
    payload["diagramLayout"] = {
        "engine": "elk",
        "version": "0.11",
        "system": {"id": "SYSTEM", "name": "Task App", "x": 0, "y": 0, "width": 800, "height": 470},
        "nodes": [
            {"id": "ACT-GROUP-MEMBER", "kind": "actor", "name": "Group Member", "x": 0, "y": 97, "width": 160, "height": 110, "side": "left", "manual": False},
            {"id": "UC-TMM-001", "kind": "use_case", "name": "Add Team Member", "x": 328, "y": 116, "width": 224, "height": 72, "manual": False},
        ],
        "edges": [],
        "diagnostics": {"warnings": [], "overlapCount": 0},
    }
    return payload


# ---------------------------------------------------------------------------
# update_diagram_positions: saving positions the user dragged by hand
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_diagram_positions_requires_an_existing_diagram():
    from fastapi import HTTPException

    payload = _payload_after_generate_groups()
    record = SimpleNamespace(model_data=payload)
    service = UseCaseService(db=AsyncMock())
    service._project = AsyncMock(return_value=SimpleNamespace(id=PROJECT_ID, name="Task App"))
    service._locked_payload = AsyncMock(return_value=(payload, record))

    with pytest.raises(HTTPException) as exc_info:
        await service.update_diagram_positions(
            project_id=PROJECT_ID, positions=[DiagramNodePositionRequest(id="ACT-GROUP-MEMBER", x=1, y=2)]
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail["code"] == "USE_CASE_DIAGRAM_REQUIRED"


@pytest.mark.asyncio
async def test_update_diagram_positions_updates_known_nodes_and_marks_manual():
    """Only ids that already exist in the current diagram are touched (an unknown id is
    silently skipped, never used to create a node), and a touched node is marked manual so a
    later regenerate (see _attach_diagram_layout) carries this position forward."""
    payload = _payload_with_diagram()
    record = SimpleNamespace(model_data=payload)
    service = UseCaseService(db=AsyncMock())
    service._project = AsyncMock(return_value=SimpleNamespace(id=PROJECT_ID, name="Task App"))
    service._locked_payload = AsyncMock(return_value=(payload, record))
    service._attach_diagram_layout = AsyncMock(return_value=None)
    service._save = AsyncMock()

    await service.update_diagram_positions(
        project_id=PROJECT_ID,
        positions=[
            DiagramNodePositionRequest(id="ACT-GROUP-MEMBER", x=999, y=888),
            DiagramNodePositionRequest(id="UC-DOES-NOT-EXIST", x=1, y=1),
        ],
    )

    nodes_by_id = {node["id"]: node for node in payload["diagramLayout"]["nodes"]}
    moved = nodes_by_id["ACT-GROUP-MEMBER"]
    assert moved["x"] == 999
    assert moved["y"] == 888
    assert moved["manual"] is True
    untouched = nodes_by_id["UC-TMM-001"]
    assert untouched["manual"] is False
    service._save.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_diagram_positions_errors_when_no_ids_match():
    from fastapi import HTTPException

    payload = _payload_with_diagram()
    record = SimpleNamespace(model_data=payload)
    service = UseCaseService(db=AsyncMock())
    service._project = AsyncMock(return_value=SimpleNamespace(id=PROJECT_ID, name="Task App"))
    service._locked_payload = AsyncMock(return_value=(payload, record))

    with pytest.raises(HTTPException) as exc_info:
        await service.update_diagram_positions(
            project_id=PROJECT_ID, positions=[DiagramNodePositionRequest(id="UC-DOES-NOT-EXIST", x=1, y=1)]
        )

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail["code"] == "USE_CASE_DIAGRAM_POSITIONS_NOT_FOUND"


@pytest.mark.asyncio
async def test_attach_diagram_layout_forwards_manual_positions_to_the_worker():
    """A node already marked manual in the persisted layout must be handed back to the ELK
    worker as manualPosition on the next recompute (table regenerate, Generate diagram, ...),
    or the position the user saved would be silently lost on the very next regenerate."""
    payload = _payload_with_diagram()
    payload["diagramLayout"]["nodes"][0]["manual"] = True
    payload["diagramLayout"]["nodes"][0]["x"] = 555
    payload["diagramLayout"]["nodes"][0]["y"] = 444
    service = UseCaseService(db=AsyncMock())
    fake_layout = {
        "engine": "elk",
        "version": "0.11",
        "system": {"id": "SYSTEM", "name": "Task App", "x": 0, "y": 0, "width": 800, "height": 470},
        "nodes": [],
        "edges": [],
        "diagnostics": {"warnings": [], "overlapCount": 0},
    }
    worker = AsyncMock(return_value=fake_layout)
    with patch("app.services.use_case_service.generate_diagram_layout", worker):
        await service._attach_diagram_layout(payload)

    sent_actors = worker.await_args.args[0]["actors"]
    moved = next(a for a in sent_actors if a["id"] == "ACT-GROUP-MEMBER")
    assert moved["manualPosition"] == {"x": 555, "y": 444}
