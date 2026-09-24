"""Split use-case pipeline after generate_groups: candidates -> selection -> details/relations ->
finalize. Only the DB row, the provider client and source loading are faked; the service's own
read / lock-and-merge / normalise code runs for real against an in-memory row, so these tests
also cover how parallel steps merge their own slice without overwriting each other's."""

import copy
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from app.config import settings
from app.services.use_case_service import UseCaseService
from app.use_cases.models import (
    RequirementsSourceSnapshot,
    SourceEvidence,
    StoredComponentSnapshot,
    StoredDocumentSnapshot,
)

PROJECT_ID = uuid.uuid4()
USER_ID = uuid.uuid4()
REF_CREATE = "entity:prd:functional_requirement:FR-1:3"
REF_ASSIGN = "entity:prd:functional_requirement:FR-2:4"
MODULE_TASKS = "SUB-TASKS"
MODULE_TEAM = "SUB-TEAM"


def _source() -> RequirementsSourceSnapshot:
    body = "| ID | Requirement |\n|----|----|\n| FR-1 | A member creates a task. |\n| FR-2 | An admin assigns a task owner. |"
    component = StoredComponentSnapshot(
        document_type="prd", artifact_type="functional_requirement", label="Functional requirements", body=body
    )
    evidence = [
        SourceEvidence(
            evidence_id=REF_CREATE,
            document_type="prd",
            artifact_type="functional_requirement",
            kind="functional_requirement",
            locator="prd.functional_requirement:L3",
            excerpt="FR-1 A member creates a task.",
            entity_id="FR-1",
        ),
        SourceEvidence(
            evidence_id=REF_ASSIGN,
            document_type="prd",
            artifact_type="functional_requirement",
            kind="functional_requirement",
            locator="prd.functional_requirement:L4",
            excerpt="FR-2 An admin assigns a task owner.",
            entity_id="FR-2",
        ),
    ]
    return RequirementsSourceSnapshot(
        project_id=str(PROJECT_ID),
        brd=StoredDocumentSnapshot(document_type="brd", label="BRD", components=[]),
        prd=StoredDocumentSnapshot(document_type="prd", label="PRD", components=[component]),
        components=[component],
        evidence=evidence,
        source_hash="test-source",
    )


def _running_payload(**generation) -> dict:
    now = datetime.now(UTC).isoformat()
    return {
        "projectId": str(PROJECT_ID),
        "projectName": "Task App",
        "actors": [
            {"id": "ACT-MEMBER", "name": "Member", "kind": "human"},
            {"id": "ACT-ADMIN", "name": "Admin", "kind": "human"},
        ],
        "modules": [
            {"id": MODULE_TASKS, "name": "Tasks", "goal": "Track work.", "sourceRefs": [REF_CREATE]},
            {"id": MODULE_TEAM, "name": "Team", "goal": "Organise people.", "sourceRefs": [REF_ASSIGN]},
        ],
        "useCases": [],
        "relationships": [],
        "generation": {"status": "running", "generationId": "gen-1", "updatedAt": now, **generation},
    }


def _candidate(name, actor="ACT-MEMBER", refs=(REF_CREATE,), priority="required", evidence="explicit"):
    return {
        "name": name,
        "primaryActorId": actor,
        "description": f"{name}.",
        "priority": priority,
        "evidence": evidence,
        "sourceRefs": list(refs),
    }


class _Record:
    """Stands in for the DB row: every read gets a copy, every write replaces the stored value."""

    def __init__(self, store: dict):
        self._store = store
        self.last_generation_error = None

    @property
    def model_data(self):
        return copy.deepcopy(self._store["payload"])

    @model_data.setter
    def model_data(self, value):
        self._store["payload"] = copy.deepcopy(value)


def _service(payload: dict, *, llm_result=None, llm_error=None, llm_side_effect=None):
    store = {"payload": copy.deepcopy(payload)}
    service = UseCaseService(db=AsyncMock())
    service._project = AsyncMock(return_value=SimpleNamespace(id=PROJECT_ID, name="Task App"))
    service._record = AsyncMock(side_effect=lambda *_args, **_kwargs: _Record(store))
    if llm_side_effect is not None:
        generate = AsyncMock(side_effect=llm_side_effect)
    elif llm_error is not None:
        generate = AsyncMock(side_effect=llm_error)
    else:
        generate = AsyncMock(return_value=(llm_result, {}))
    provider = SimpleNamespace(
        id=uuid.uuid4(), provider_type=SimpleNamespace(value="anthropic"), model_name="claude-test"
    )
    service._llm_client = AsyncMock(return_value=(SimpleNamespace(generate=generate), provider))
    return service, store, generate


def _body():
    return SimpleNamespace(provider_config_id=None)


def _patched_source():
    return patch(
        "app.services.use_case_service.load_project_requirements_source", AsyncMock(return_value=_source())
    )


async def _selected(candidates: dict, **generation) -> tuple[UseCaseService, dict]:
    """A store as it looks right after select_use_cases."""

    service, store, _ = _service(_running_payload(candidates=candidates, **generation))
    with _patched_source():
        await service.select_use_cases(project_id=PROJECT_ID)
    return service, store


# ---------------------------------------------------------------------------
# Step 2: candidates
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_candidates_are_stored_per_module_with_a_bounded_output_budget():
    drafts = [
        {"name": f"Create Task {index}", "primary_actor_id": "ACT-MEMBER", "description": "Creates a task.",
         "priority": "required", "evidence": "explicit", "source_refs": [REF_CREATE]}
        for index in range(20)
    ]
    service, store, generate = _service(_running_payload(), llm_result={"candidates": drafts})

    with _patched_source():
        await service.generate_module_candidates(
            project_id=PROJECT_ID, user_id=USER_ID, group_id=MODULE_TASKS, body=_body()
        )

    stored = store["payload"]["generation"]["candidates"][MODULE_TASKS]
    assert len(stored) == 15  # two modules -> per-module shortlist capped at 15
    assert stored[0]["primaryActorId"] == "ACT-MEMBER"
    assert stored[0]["sourceRefs"] == [REF_CREATE]
    assert generate.await_args.kwargs["max_tokens"] == settings.use_case_candidates_max_tokens
    assert store["payload"]["generation"]["status"] == "running"


@pytest.mark.asyncio
async def test_candidates_from_parallel_modules_do_not_overwrite_each_other():
    draft = {"name": "Create Task", "primary_actor_id": "ACT-MEMBER", "description": "Creates a task.",
             "source_refs": [REF_CREATE]}
    service, store, _ = _service(_running_payload(), llm_result={"candidates": [draft]})

    with _patched_source():
        await service.generate_module_candidates(
            project_id=PROJECT_ID, user_id=USER_ID, group_id=MODULE_TASKS, body=_body()
        )
        await service.generate_module_candidates(
            project_id=PROJECT_ID, user_id=USER_ID, group_id=MODULE_TEAM, body=_body()
        )

    assert set(store["payload"]["generation"]["candidates"]) == {MODULE_TASKS, MODULE_TEAM}


@pytest.mark.asyncio
async def test_candidates_require_a_running_generation():
    payload = _running_payload()
    payload["generation"]["status"] = "completed"
    service, _store, _ = _service(payload, llm_result={"candidates": []})

    with _patched_source(), pytest.raises(HTTPException) as exc_info:
        await service.generate_module_candidates(
            project_id=PROJECT_ID, user_id=USER_ID, group_id=MODULE_TASKS, body=_body()
        )

    assert exc_info.value.status_code == 409


@pytest.mark.asyncio
async def test_candidates_failure_fails_the_run():
    """Without every module's shortlist there is nothing complete to select from, so this step
    stays fail-fast -- and must resolve the run marker rather than leave it "running"."""
    service, store, _ = _service(_running_payload(), llm_error=TimeoutError())

    with _patched_source(), pytest.raises(HTTPException) as exc_info:
        await service.generate_module_candidates(
            project_id=PROJECT_ID, user_id=USER_ID, group_id=MODULE_TASKS, body=_body()
        )

    assert exc_info.value.status_code == 504
    assert store["payload"]["generation"]["status"] == "failed"


@pytest.mark.asyncio
async def test_candidates_are_discarded_when_the_run_was_replaced_meanwhile():
    draft = {"name": "Create Task", "primary_actor_id": "ACT-MEMBER", "description": "Creates a task.",
             "source_refs": [REF_CREATE]}
    store_ref: dict = {}

    async def new_run_starts_during_the_call(**_kwargs):
        store_ref["store"]["payload"]["generation"]["generationId"] = "gen-2"
        return {"candidates": [draft]}, {}

    service, store, _ = _service(_running_payload(), llm_side_effect=new_run_starts_during_the_call)
    store_ref["store"] = store

    with _patched_source(), pytest.raises(HTTPException) as exc_info:
        await service.generate_module_candidates(
            project_id=PROJECT_ID, user_id=USER_ID, group_id=MODULE_TASKS, body=_body()
        )

    assert exc_info.value.status_code == 409
    assert "candidates" not in store["payload"]["generation"]


# ---------------------------------------------------------------------------
# Step 3: selection (deterministic)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_selection_keeps_only_candidates_citing_real_source_lines():
    candidates = {
        MODULE_TASKS: [
            _candidate("Create Task"),
            _candidate("Export Report", refs=["entity:prd:functional_requirement:FR-99:9"]),
            _candidate("Archive Task", refs=["the PRD says so"]),
            # The placeholder the old flow silently fell back to for an uncited row.
            _candidate("Send Reminder", refs=["internal"]),
            _candidate("Approve Budget", actor="ACT-UNKNOWN"),
        ],
        MODULE_TEAM: [],
    }

    _service_, store = await _selected(candidates)

    payload = store["payload"]
    assert [row["name"] for row in payload["useCases"]] == ["Create Task"]
    stats = payload["generation"]["selection"]
    assert stats["uncited"] == 3
    assert stats["unknownActor"] == 1
    assert "candidates" not in payload["generation"]


@pytest.mark.asyncio
async def test_selection_starts_rows_without_detail_and_drops_unused_actors():
    _service_, store = await _selected({MODULE_TASKS: [_candidate("Create Task")]})

    payload = store["payload"]
    row = payload["useCases"][0]
    assert row["mainFlow"] == []
    assert payload["generation"]["detailStatus"] == {row["id"]: "pending"}
    assert [actor["id"] for actor in payload["actors"]] == ["ACT-MEMBER"]
    assert payload["generation"]["status"] == "running"


@pytest.mark.asyncio
async def test_selection_caps_the_table_but_covers_every_actor_and_module_first():
    candidates = {
        MODULE_TASKS: [_candidate("Create Task"), _candidate("Edit Task"), _candidate("Close Task")],
        MODULE_TEAM: [_candidate("Assign Owner", actor="ACT-ADMIN", refs=[REF_ASSIGN], priority="optional")],
    }

    with patch("app.services.use_case_service._MAX_GENERATED_USE_CASES", 3):
        _service_, store = await _selected(candidates)

    names = [row["name"] for row in store["payload"]["useCases"]]
    # Pure priority order would pick the three required Tasks rows and leave Admin and the Team
    # module with nothing; coverage comes first, then the best remaining row.
    assert names == ["Create Task", "Edit Task", "Assign Owner"]


@pytest.mark.asyncio
async def test_selection_merges_duplicate_names_across_modules():
    candidates = {
        MODULE_TASKS: [_candidate("Assign Owner", priority="optional")],
        MODULE_TEAM: [_candidate("assign  owner", actor="ACT-ADMIN", refs=[REF_ASSIGN], priority="required")],
    }

    _service_, store = await _selected(candidates)

    rows = store["payload"]["useCases"]
    assert len(rows) == 1
    assert rows[0]["moduleId"] == MODULE_TEAM


# ---------------------------------------------------------------------------
# Step 4: details
# ---------------------------------------------------------------------------


def _detail(use_case_id: str, actor: str = "ACT-MEMBER") -> dict:
    return {
        "use_case_id": use_case_id,
        "trigger": "The member needs a new task.",
        "main_flow": [
            {"step": 1, "participant_type": "actor", "participant_id": actor, "action": "Opens the form."},
            {"step": 2, "participant_type": "system", "action": "Saves the task."},
        ],
    }


async def _two_selected_rows():
    service, store = await _selected(
        {MODULE_TASKS: [_candidate("Create Task")], MODULE_TEAM: [_candidate("Assign Owner", refs=[REF_ASSIGN])]}
    )
    return service, store, [row["id"] for row in store["payload"]["useCases"]]


@pytest.mark.asyncio
async def test_details_fill_only_the_requested_rows():
    _unused, selected_store, (first, second) = await _two_selected_rows()
    service, store, generate = _service(selected_store["payload"], llm_result={"details": [_detail(first)]})

    with _patched_source():
        await service.generate_use_case_details(
            project_id=PROJECT_ID, user_id=USER_ID, use_case_ids=[first], body=_body()
        )

    rows = {row["id"]: row for row in store["payload"]["useCases"]}
    assert [step["action"] for step in rows[first]["mainFlow"]] == ["Opens the form.", "Saves the task."]
    assert rows[second]["mainFlow"] == []
    assert store["payload"]["generation"]["detailStatus"] == {first: "completed", second: "pending"}
    assert store["payload"]["generation"]["status"] == "running"
    assert generate.await_args.kwargs["max_tokens"] == settings.use_case_detail_max_tokens_per_use_case


@pytest.mark.asyncio
async def test_details_missing_from_the_response_are_marked_failed():
    _unused, selected_store, (first, second) = await _two_selected_rows()
    service, store, _ = _service(selected_store["payload"], llm_result={"details": [_detail(first)]})

    with _patched_source():
        await service.generate_use_case_details(
            project_id=PROJECT_ID, user_id=USER_ID, use_case_ids=[first, second], body=_body()
        )

    assert store["payload"]["generation"]["detailStatus"] == {first: "completed", second: "failed"}


@pytest.mark.asyncio
async def test_details_failure_marks_only_those_rows_and_never_the_run():
    _unused, selected_store, (first, second) = await _two_selected_rows()
    service, store, _ = _service(selected_store["payload"], llm_error=TimeoutError())

    with _patched_source(), pytest.raises(HTTPException) as exc_info:
        await service.generate_use_case_details(
            project_id=PROJECT_ID, user_id=USER_ID, use_case_ids=[first], body=_body()
        )

    assert exc_info.value.status_code == 504
    assert store["payload"]["generation"]["detailStatus"] == {first: "failed", second: "pending"}
    assert store["payload"]["generation"]["status"] == "running"


@pytest.mark.asyncio
async def test_details_are_not_written_into_a_row_that_changed_meanwhile():
    _unused, selected_store, (first, _second) = await _two_selected_rows()
    store_ref: dict = {}

    async def row_renamed_during_the_call(**_kwargs):
        row = next(item for item in store_ref["store"]["payload"]["useCases"] if item["id"] == first)
        row["name"] = "Something Else"
        return {"details": [_detail(first)]}, {}

    service, store, _ = _service(selected_store["payload"], llm_side_effect=row_renamed_during_the_call)
    store_ref["store"] = store

    with _patched_source():
        await service.generate_use_case_details(
            project_id=PROJECT_ID, user_id=USER_ID, use_case_ids=[first], body=_body()
        )

    row = next(item for item in store["payload"]["useCases"] if item["id"] == first)
    assert row["mainFlow"] == []
    assert store["payload"]["generation"]["detailStatus"][first] == "pending"


@pytest.mark.asyncio
async def test_retrying_the_last_failed_row_resolves_the_run_to_completed():
    _unused, selected_store, (first, second) = await _two_selected_rows()
    payload = selected_store["payload"]
    payload["generation"].update(
        {"status": "completed_with_errors", "error": "1 use case(s) have no detail yet.",
         "detailStatus": {first: "completed", second: "failed"}}
    )
    service, store, _ = _service(payload, llm_result={"details": [_detail(second)]})

    with _patched_source():
        await service.generate_use_case_details(
            project_id=PROJECT_ID, user_id=USER_ID, use_case_ids=[second], body=_body()
        )

    generation = store["payload"]["generation"]
    assert generation["status"] == "completed"
    assert "error" not in generation


# ---------------------------------------------------------------------------
# Relations (alongside step 4) and step 5: finalize
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_relations_merge_without_finishing_the_run():
    _unused, selected_store, (first, second) = await _two_selected_rows()
    relation = {"kind": "extend", "source_id": second, "target_id": first, "condition": "If an owner is needed.",
                "evidence": [REF_ASSIGN]}
    service, store, generate = _service(selected_store["payload"], llm_result={"relations": [relation]})

    with _patched_source():
        await service.generate_relations(project_id=PROJECT_ID, user_id=USER_ID, body=_body())

    payload = store["payload"]
    assert [(item["sourceId"], item["targetId"]) for item in payload["relationships"]] == [(second, first)]
    assert payload["generation"]["relationsGenerated"] is True
    assert payload["generation"]["status"] == "running"
    assert generate.await_args.kwargs["max_tokens"] == settings.use_case_relations_max_tokens


@pytest.mark.asyncio
async def test_relations_keep_one_relationship_per_pair_with_unique_ids():
    """A provider once returned include AND extend for the same pair: both got the same id
    (React: "two children with the same key"), and the pair was contradictory anyway."""
    _unused, selected_store, (first, second) = await _two_selected_rows()
    relations = [
        {"kind": "extend", "source_id": second, "target_id": first, "condition": "If an owner is needed.",
         "evidence": [REF_ASSIGN]},
        {"kind": "include", "source_id": second, "target_id": first, "evidence": [REF_ASSIGN]},
        {"kind": "extend", "source_id": first, "target_id": second, "condition": "Reverse direction.",
         "evidence": [REF_CREATE]},
    ]
    service, store, _ = _service(selected_store["payload"], llm_result={"relations": relations})

    with _patched_source():
        await service.generate_relations(project_id=PROJECT_ID, user_id=USER_ID, body=_body())

    saved = store["payload"]["relationships"]
    assert [(item["type"], item["sourceId"], item["targetId"]) for item in saved] == [("extend", second, first)]
    assert len({item["id"] for item in saved}) == len(saved)


@pytest.mark.asyncio
async def test_relations_failure_is_recorded_on_the_run_not_fatal():
    _unused, selected_store, _ids = await _two_selected_rows()
    service, store, _ = _service(selected_store["payload"], llm_error=TimeoutError())

    with _patched_source(), pytest.raises(HTTPException):
        await service.generate_relations(project_id=PROJECT_ID, user_id=USER_ID, body=_body())

    assert store["payload"]["generation"]["relationsError"]
    assert store["payload"]["generation"]["status"] == "running"


@pytest.mark.asyncio
async def test_finalize_completes_a_run_with_every_detail_written():
    _unused, selected_store, (first, second) = await _two_selected_rows()
    payload = selected_store["payload"]
    payload["generation"]["detailStatus"] = {first: "completed", second: "completed"}
    service, store, _ = _service(payload)

    with _patched_source():
        await service.finalize_generation(project_id=PROJECT_ID)

    generation = store["payload"]["generation"]
    assert generation["status"] == "completed"
    assert store["payload"]["validation"] is not None


@pytest.mark.asyncio
async def test_finalize_reports_rows_left_without_detail_as_failed():
    _unused, selected_store, (first, second) = await _two_selected_rows()
    payload = selected_store["payload"]
    payload["generation"]["detailStatus"] = {first: "completed", second: "pending"}
    service, store, _ = _service(payload)

    with _patched_source():
        await service.finalize_generation(project_id=PROJECT_ID)

    generation = store["payload"]["generation"]
    assert generation["status"] == "completed_with_errors"
    assert generation["detailStatus"][second] == "failed"
    assert "1 use case(s)" in generation["error"]


@pytest.mark.asyncio
async def test_finalize_leaves_a_run_that_is_not_running_unchanged():
    payload = _running_payload()
    payload["generation"]["status"] = "failed"
    service, store, _ = _service(payload)

    with _patched_source():
        await service.finalize_generation(project_id=PROJECT_ID)

    assert store["payload"]["generation"]["status"] == "failed"
