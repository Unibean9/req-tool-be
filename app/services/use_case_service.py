"""Use-case model persistence, generation, and FE-facing contract mapping."""

from __future__ import annotations

import asyncio
import copy
import json
import re
import uuid
from typing import Any

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.crypto import decrypt_token
from app.models.llm_provider import LLMProviderConfig, LLMProviderStatus
from app.models.project import Project
from app.models.use_case import UseCaseModelRecord
from app.schemas.use_case import (
    ActorCreateRequest,
    ActorKind,
    ActorResponse,
    ActorUpdateRequest,
    RelationshipCreateRequest,
    RelationshipType,
    UseCaseCreateRequest,
    UseCaseGenerateRequest,
    UseCaseLevel,
    UseCaseModelResponse,
    UseCasePlantUmlResponse,
    UseCasePlantUmlUpdateRequest,
    UseCaseRelationshipResponse,
    UseCaseResponse,
    UseCaseUpdateRequest,
    UseCaseValidationResponse,
)
from app.services.llm_clients import LLMClientFactory
from app.use_cases.completion import _ensure_actor_associations, complete_use_case_table
from app.use_cases.models import (
    RequirementsSourceSnapshot,
    UseCaseActor,
    UseCaseEntry,
    UseCaseGroupDetailDraftList,
    UseCaseGroupsDraftList,
    UseCaseModel,
    UseCaseRelation,
    UseCaseRelationshipDraftList,
    UseCaseSubsystem,
)
from app.use_cases.plantuml import render_plantuml
from app.use_cases.rules import validate_use_case_model
from app.use_cases.source_loader import load_project_requirements_source

_LEVEL_ORDER = {"L0": 0, "L1": 1, "L2": 2}
_SLUG_RE = re.compile(r"[^A-Z0-9]+")
_ABSTRACTION_BY_LEVEL = {"L0": "summary", "L1": "user_goal", "L2": "subfunction"}
_STATUS_VALUE = {"Confirmed": "confirmed", "Inferred": "inferred", "Suggested": "suggested"}
_PRIORITY_VALUE = {"Must": "must", "Should": "should", "Could": "could"}


class UseCaseService:
    """Own the aggregate used by the use-case table and editable PlantUML endpoint."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_model(
        self,
        *,
        project_id: uuid.UUID,
        max_level: UseCaseLevel = UseCaseLevel.L2,
        include_actors: bool = True,
        include_relationships: bool = True,
    ) -> UseCaseModelResponse:
        project = await self._project(project_id)
        payload = await self._payload(project)
        return self._response(
            payload,
            max_level=max_level,
            include_actors=include_actors,
            include_relationships=include_relationships,
        )

    async def get_use_case(self, *, project_id: uuid.UUID, use_case_id: str) -> UseCaseResponse:
        project = await self._project(project_id)
        payload = await self._payload(project)
        response = self._response(payload)
        return self._detail_response(response, use_case_id)

    async def create_actor(
        self,
        *,
        project_id: uuid.UUID,
        body: ActorCreateRequest,
    ) -> ActorResponse:
        project = await self._project(project_id)
        payload, record = await self._locked_payload(project)
        actor_id = self._next_id("ACT", body.name, {str(item.get("id")) for item in payload["actors"]})
        actor = {"id": actor_id, "name": body.name.strip(), "kind": body.kind.value}
        payload["actors"].append(actor)
        self._invalidate_manual_state(payload)
        await self._save(record, payload)
        return ActorResponse.model_validate(actor)

    async def update_actor(
        self,
        *,
        project_id: uuid.UUID,
        actor_id: str,
        body: ActorUpdateRequest,
    ) -> ActorResponse:
        project = await self._project(project_id)
        payload, record = await self._locked_payload(project)
        actor = self._find(payload["actors"], actor_id)
        if actor is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Actor not found")
        actor["name"] = body.name.strip()
        self._invalidate_manual_state(payload)
        await self._save(record, payload)
        return ActorResponse.model_validate(actor)

    async def delete_actor(self, *, project_id: uuid.UUID, actor_id: str) -> dict[str, Any]:
        project = await self._project(project_id)
        payload, record = await self._locked_payload(project)
        if self._find(payload["actors"], actor_id) is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Actor not found")
        references = [
            item["id"]
            for item in payload["useCases"]
            if item.get("primaryActorId") == actor_id or actor_id in item.get("supportingActorIds", [])
        ]
        relationship_references = [
            item["id"]
            for item in payload["relationships"]
            if item.get("sourceId") == actor_id or item.get("targetId") == actor_id
        ]
        if references or relationship_references:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                detail={
                    "message": "Actor is still referenced by use cases or relationships",
                    "use_case_ids": references,
                    "relationship_ids": relationship_references,
                },
            )
        payload["actors"] = [item for item in payload["actors"] if item.get("id") != actor_id]
        self._invalidate_manual_state(payload)
        await self._save(record, payload)
        return {"id": actor_id, "deleted": True}

    async def create_use_case(
        self,
        *,
        project_id: uuid.UUID,
        body: UseCaseCreateRequest,
    ) -> UseCaseResponse:
        project = await self._project(project_id)
        payload, record = await self._locked_payload(project)
        self._validate_use_case_refs(
            payload,
            body.primary_actor_id,
            body.supporting_actor_ids,
            body.parent_use_case_id,
            child_level=body.level.value,
        )
        use_case_id = self._next_use_case_id(payload, body.level, body.subsystem)
        item = self._use_case_dict(use_case_id, body)
        payload["useCases"].append(item)
        self._sync_parent_relationship(payload, child_id=use_case_id, parent_id=body.parent_use_case_id)
        self._invalidate_manual_state(payload)
        await self._save(record, payload)
        response = self._response(payload)
        return self._detail_response(response, use_case_id)

    async def update_use_case(
        self,
        *,
        project_id: uuid.UUID,
        use_case_id: str,
        body: UseCaseUpdateRequest,
    ) -> UseCaseResponse:
        project = await self._project(project_id)
        payload, record = await self._locked_payload(project)
        item = self._find(payload["useCases"], use_case_id)
        if item is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Use case not found")
        changes = body.model_dump(exclude_unset=True, by_alias=True)
        nullable_fields = {field for field, value in changes.items() if value is None and field != "parentUseCaseId"}
        if nullable_fields:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                detail=f"Use case fields cannot be null: {', '.join(sorted(nullable_fields))}",
            )
        primary_actor_id = changes.get("primaryActorId", item.get("primaryActorId"))
        supporting_actor_ids = changes.get("supportingActorIds", item.get("supportingActorIds", []))
        supporting_actor_ids = supporting_actor_ids or []
        if "supportingActorIds" in changes:
            changes["supportingActorIds"] = supporting_actor_ids
        parent_id = changes.get("parentUseCaseId", item.get("parentUseCaseId"))
        self._validate_use_case_refs(
            payload,
            primary_actor_id,
            supporting_actor_ids,
            parent_id,
            current_id=use_case_id,
            child_level=changes.get("level", item.get("level")),
        )
        item.update(changes)
        self._sync_parent_relationship(payload, child_id=use_case_id, parent_id=parent_id)
        self._invalidate_manual_state(payload)
        await self._save(record, payload)
        response = self._response(payload)
        return self._detail_response(response, use_case_id)

    async def delete_use_case(self, *, project_id: uuid.UUID, use_case_id: str) -> dict[str, Any]:
        project = await self._project(project_id)
        payload, record = await self._locked_payload(project)
        if self._find(payload["useCases"], use_case_id) is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Use case not found")
        references = [
            relation["id"]
            for relation in payload["relationships"]
            if relation.get("sourceId") == use_case_id or relation.get("targetId") == use_case_id
        ]
        children = [item["id"] for item in payload["useCases"] if item.get("parentUseCaseId") == use_case_id]
        if references or children:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                detail={
                    "message": "Use case is still referenced",
                    "relationship_ids": references,
                    "child_use_case_ids": children,
                },
            )
        payload["useCases"] = [item for item in payload["useCases"] if item.get("id") != use_case_id]
        self._invalidate_manual_state(payload)
        await self._save(record, payload)
        return {"id": use_case_id, "deleted": True}

    async def create_relationship(
        self,
        *,
        project_id: uuid.UUID,
        body: RelationshipCreateRequest,
    ) -> UseCaseRelationshipResponse:
        project = await self._project(project_id)
        payload, record = await self._locked_payload(project)
        self._validate_relationship(payload, body)
        relation_id = self._next_relationship_id(payload, body.source_id, body.target_id)
        relation = {
            "id": relation_id,
            "sourceId": body.source_id,
            "targetId": body.target_id,
            "type": body.type.value,
            "condition": body.condition,
        }
        if body.type == RelationshipType.PART_OF:
            self._sync_parent_relationship(
                payload,
                child_id=body.target_id,
                parent_id=body.source_id,
                relationship_id=relation_id,
            )
        else:
            payload["relationships"].append(relation)
        self._invalidate_manual_state(payload)
        await self._save(record, payload)
        return UseCaseRelationshipResponse.model_validate(relation)

    async def delete_relationship(
        self,
        *,
        project_id: uuid.UUID,
        relationship_id: str,
    ) -> dict[str, Any]:
        project = await self._project(project_id)
        payload, record = await self._locked_payload(project)
        relation = self._find(payload["relationships"], relationship_id)
        if relation is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Relationship not found")
        if relation.get("type") == RelationshipType.PART_OF.value:
            child = self._find(payload["useCases"], relation.get("targetId"))
            if child is not None and child.get("parentUseCaseId") == relation.get("sourceId"):
                child["parentUseCaseId"] = None
        payload["relationships"] = [item for item in payload["relationships"] if item.get("id") != relationship_id]
        self._invalidate_manual_state(payload)
        await self._save(record, payload)
        return {"id": relationship_id, "deleted": True}

    async def generate(
        self,
        *,
        project_id: uuid.UUID,
        user_id: uuid.UUID,
        body: UseCaseGenerateRequest,
    ) -> UseCaseModelResponse:
        project = await self._project(project_id)
        source = await load_project_requirements_source(self.db, project_id=project_id)
        from app.use_cases.harness import UseCaseGenerationHarness

        harness = UseCaseGenerationHarness(source, include_relations=False)
        client, provider = await self._llm_client(user_id=user_id, provider_config_id=body.provider_config_id)
        try:
            raw_result, usage = await asyncio.wait_for(
                client.generate(
                    messages=[{"role": "user", "content": harness.build_user_prompt()}],
                    system=harness.build_system_instruction(),
                    max_tokens=settings.use_case_generation_max_tokens,
                    response_format=harness.response_format(),
                ),
                timeout=settings.use_case_generation_timeout_seconds,
            )
            payload = _decode_llm_payload(raw_result)
            candidate = UseCaseModel.model_validate(payload)
            # The provider may summarize a long PRD and omit entire functional-requirement
            # families.  Complete the table from the stored BRD/PRD registry before validating
            # relations; this keeps the table authoritative while preserving explicit candidate
            # include/extend/generalization relations that can be mapped to it.
            model = complete_use_case_table(source, candidate)
            model, report = harness.parse_and_validate(model)
        except TimeoutError as exc:
            raise HTTPException(status.HTTP_504_GATEWAY_TIMEOUT, detail="Use-case generation timed out") from exc
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status.HTTP_502_BAD_GATEWAY,
                detail={"code": "USE_CASE_GENERATION_FAILED", "message": str(exc)[:500]},
            ) from exc

        response_payload = self._core_to_payload(
            project=project,
            source=source,
            model=model,
            report=report,
            provider=provider,
            usage=usage,
            # The table pass above deliberately withholds include/extend/generalization so the
            # request stays small and fast; the deterministic BRD/PRD floor model can still carry
            # a couple of hardcoded extend relations, so check the actual model rather than assume.
            relations_generated=any(relation.kind != "association" for relation in model.relations),
        )
        record = await self._record(project_id, for_update=True)
        if record is None:
            record = UseCaseModelRecord(project_id=project_id)
            self.db.add(record)
            await self.db.flush()
        record.model_data = response_payload
        record.source_hash = source.source_hash
        record.generated_by_id = user_id
        record.last_generation_error = None
        await self.db.flush()
        return self._response(response_payload, max_level=body.max_level)

    async def generate_groups(
        self,
        *,
        project_id: uuid.UUID,
        user_id: uuid.UUID,
        body: UseCaseGenerateRequest,
    ) -> UseCaseModelResponse:
        """Phase 1 of the split generation flow: extract capability/domain groups (L0) and the
        actor roster only -- not individual use cases -- so the output (and therefore the risk of
        timing out) is a fraction of the size of generating the whole project's use cases in one
        call. Reads the project's Business Capabilities content in whatever format it was
        actually written; an earlier version of this method parsed that content with a regex tied
        to one fixed "BC-xx" heading/ID convention and silently returned zero groups for any
        project that used a different one (a table with its own numbering, in this project's
        case) -- this calls the model instead, the same way the rest of the app already trusts it
        to read free-form BRD/PRD content.

        Each L0 row's id is the "group id" the FE loops over to call generate_group_use_cases for
        that group's own L1/L2 detail.
        """
        project = await self._project(project_id)
        source = await load_project_requirements_source(self.db, project_id=project_id)
        from app.use_cases.harness import UseCaseGroupsHarness

        harness = UseCaseGroupsHarness(source)
        client, provider = await self._llm_client(user_id=user_id, provider_config_id=body.provider_config_id)
        try:
            raw_result, usage = await asyncio.wait_for(
                client.generate(
                    messages=[{"role": "user", "content": harness.build_user_prompt()}],
                    system=harness.build_system_instruction(),
                    max_tokens=settings.use_case_generation_max_tokens,
                    response_format=harness.response_format(),
                ),
                timeout=settings.use_case_generation_timeout_seconds,
            )
            draft_payload = _decode_llm_payload(raw_result)
            drafts = UseCaseGroupsDraftList.model_validate(draft_payload)
        except TimeoutError as exc:
            raise HTTPException(
                status.HTTP_504_GATEWAY_TIMEOUT, detail="Use-case group generation timed out"
            ) from exc
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status.HTTP_502_BAD_GATEWAY,
                detail={"code": "USE_CASE_GROUPS_GENERATION_FAILED", "message": str(exc)[:500]},
            ) from exc

        actor_ids_by_name: dict[str, str] = {}
        actors: list[UseCaseActor] = []

        def actor_id_for(name: str, source_refs: list[str]) -> str:
            key = name.strip().lower()
            if key in actor_ids_by_name:
                return actor_ids_by_name[key]
            new_id = self._next_id("ACT", name, {item.id for item in actors})
            actors.append(
                UseCaseActor(id=new_id, name=name.strip(), kind="human_role", source_refs=source_refs or ["internal"])
            )
            actor_ids_by_name[key] = new_id
            return new_id

        for actor_draft in drafts.actors:
            actor_id_for(actor_draft.name, actor_draft.source_refs)

        subsystems: list[UseCaseSubsystem] = []
        use_cases: list[UseCaseEntry] = []
        for group in drafts.groups:
            if not group.user_segment:
                continue
            primary_id = actor_id_for(group.user_segment[0], group.source_refs)
            secondary_ids = [
                item
                for item in (actor_id_for(role, group.source_refs) for role in group.user_segment[1:])
                if item != primary_id
            ]
            subsystem_id = self._next_id("SUB", group.name, {item.id for item in subsystems})
            subsystems.append(UseCaseSubsystem(id=subsystem_id, name=group.name, source_refs=group.source_refs))
            use_case_id = self._next_id("UC-SUM", group.name, {item.id for item in use_cases})
            use_cases.append(
                UseCaseEntry(
                    id=use_case_id,
                    name=group.name,
                    level="L0",
                    abstraction="summary",
                    primary_actor_id=primary_id,
                    secondary_actor_ids=secondary_ids,
                    subsystem_id=subsystem_id,
                    parent_use_case_id=None,
                    description=f"The actor can {group.goal.rstrip('.').lower()}."[:600],
                    precondition="The project is available to the actor.",
                    priority="should",
                    status="inferred",
                    source_refs=group.source_refs,
                )
            )
        if not use_cases:
            raise HTTPException(
                status.HTTP_502_BAD_GATEWAY,
                detail={
                    "code": "USE_CASE_GROUPS_EMPTY",
                    "message": (
                        "No capability groups with a named user segment could be extracted from the stored "
                        "BRD/PRD. Add or clarify the Business Capabilities content and try again."
                    ),
                },
            )

        model = UseCaseModel(system_name=project.name, actors=actors, subsystems=subsystems, use_cases=use_cases)
        _ensure_actor_associations(model)
        report = validate_use_case_model(model, source)
        response_payload = self._core_to_payload(
            project=project,
            source=source,
            model=model,
            report=report,
            provider=provider,
            usage=usage,
            relations_generated=False,
        )
        record = await self._record(project_id, for_update=True)
        if record is None:
            record = UseCaseModelRecord(project_id=project_id)
            self.db.add(record)
            await self.db.flush()
        record.model_data = response_payload
        record.source_hash = source.source_hash
        record.generated_by_id = user_id
        record.last_generation_error = None
        await self.db.flush()
        return self._response(response_payload, max_level=body.max_level)

    async def generate_relations(
        self,
        *,
        project_id: uuid.UUID,
        user_id: uuid.UUID,
        body: UseCaseGenerateRequest,
    ) -> UseCaseModelResponse:
        """Resolve include/extend/generalization relations for an already-generated table.

        This is the second half of the split generation flow: it never re-derives actors,
        subsystems, or use cases, so the prompt/response stay small compared to a full table
        regeneration.
        """

        project = await self._project(project_id)
        payload, record = await self._locked_payload(project)
        if not payload["useCases"]:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                detail="Generate the use-case table before generating relationships",
            )
        source = await load_project_requirements_source(self.db, project_id=project_id)
        from app.use_cases.harness import UseCaseRelationshipHarness

        harness = UseCaseRelationshipHarness(
            source=source,
            use_cases=[
                {
                    "id": item["id"],
                    "name": item["title"],
                    "subsystem": item["subsystem"],
                    "level": item["level"],
                    "primaryActorId": item["primaryActorId"],
                    "supportingActorIds": item.get("supportingActorIds") or [],
                }
                for item in payload["useCases"]
            ],
            actors=[{"id": item["id"], "name": item["name"]} for item in payload["actors"]],
        )
        client, provider = await self._llm_client(user_id=user_id, provider_config_id=body.provider_config_id)
        try:
            raw_result, usage = await asyncio.wait_for(
                client.generate(
                    messages=[{"role": "user", "content": harness.build_user_prompt()}],
                    system=harness.build_system_instruction(),
                    max_tokens=settings.use_case_generation_max_tokens,
                    response_format=harness.response_format(),
                ),
                timeout=settings.use_case_generation_timeout_seconds,
            )
            draft_payload = _decode_llm_payload(raw_result)
            drafts = UseCaseRelationshipDraftList.model_validate(draft_payload).relations
        except TimeoutError as exc:
            raise HTTPException(status.HTTP_504_GATEWAY_TIMEOUT, detail="Relationship generation timed out") from exc
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status.HTTP_502_BAD_GATEWAY,
                detail={"code": "USE_CASE_RELATIONSHIP_GENERATION_FAILED", "message": str(exc)[:500]},
            ) from exc

        self._apply_relationship_drafts(payload, drafts)
        model = self._rebuild_model_for_render(payload, project.name)
        payload["plantUml"] = {
            "language": "plantuml",
            "source": render_plantuml(model),
            "editable": True,
            "stale": False,
            "generatedFrom": "use-case-table",
        }
        payload["validation"] = None
        payload["generation"] = {
            "source": "ai",
            "providerConfigId": str(provider.id),
            "provider": provider.provider_type.value,
            "model": provider.model_name,
            "usage": usage,
            "relationsGenerated": True,
        }
        await self._save(record, payload)
        return self._response(payload, max_level=body.max_level)

    async def generate_group_use_cases(
        self,
        *,
        project_id: uuid.UUID,
        user_id: uuid.UUID,
        group_id: str,
        body: UseCaseGenerateRequest,
    ) -> UseCaseModelResponse:
        """Phase 2 of the split generation flow: propose one capability group's own L1 user-goal
        use cases (and L2 detail where warranted). The output is scoped to this one group, so it
        stays a fraction of the size of generating every group's use cases in one call -- the
        actual fix for generation timing out on a larger project.

        Idempotent per group: re-running replaces that group's previously generated L1/L2 rows
        instead of accumulating duplicates alongside them.
        """
        project = await self._project(project_id)
        payload, record = await self._locked_payload(project)
        group = self._find(payload["useCases"], group_id)
        if group is None or group.get("level") != "L0":
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Use-case group not found")

        source = await load_project_requirements_source(self.db, project_id=project_id)
        from app.use_cases.harness import UseCaseGroupUseCasesHarness

        actor_ids = {item["id"] for item in payload["actors"]}
        harness = UseCaseGroupUseCasesHarness(
            source=source,
            group={
                "id": group["id"],
                "name": group["title"],
                "goal": group.get("description") or group["title"],
            },
            actors=[{"id": item["id"], "name": item["name"]} for item in payload["actors"]],
        )
        client, provider = await self._llm_client(user_id=user_id, provider_config_id=body.provider_config_id)
        try:
            raw_result, usage = await asyncio.wait_for(
                client.generate(
                    messages=[{"role": "user", "content": harness.build_user_prompt()}],
                    system=harness.build_system_instruction(),
                    max_tokens=settings.use_case_generation_max_tokens,
                    response_format=harness.response_format(),
                ),
                timeout=settings.use_case_generation_timeout_seconds,
            )
            draft_payload = _decode_llm_payload(raw_result)
            drafts = UseCaseGroupDetailDraftList.model_validate(draft_payload).use_cases
        except TimeoutError as exc:
            raise HTTPException(
                status.HTTP_504_GATEWAY_TIMEOUT, detail="Use-case group generation timed out"
            ) from exc
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status.HTTP_502_BAD_GATEWAY,
                detail={"code": "USE_CASE_GROUP_GENERATION_FAILED", "message": str(exc)[:500]},
            ) from exc

        # Idempotent per group: drop this group's previously generated L1 rows and their L2
        # children before merging the fresh drafts, rather than accumulating duplicates.
        existing_l1_ids = {
            item["id"]
            for item in payload["useCases"]
            if item.get("level") == "L1" and item.get("parentUseCaseId") == group_id
        }
        existing_l2_ids = {
            item["id"]
            for item in payload["useCases"]
            if item.get("level") == "L2" and item.get("parentUseCaseId") in existing_l1_ids
        }
        stale_ids = existing_l1_ids | existing_l2_ids
        if stale_ids:
            payload["useCases"] = [item for item in payload["useCases"] if item["id"] not in stale_ids]
            payload["relationships"] = [
                item
                for item in payload["relationships"]
                if item.get("sourceId") not in stale_ids and item.get("targetId") not in stale_ids
            ]

        subsystem = group.get("subsystem") or ""
        tag_to_id: dict[str, str] = {}
        accepted = 0

        def append_use_case(*, level: str, name: str, parent_id: str, draft) -> str:
            new_id = self._next_use_case_id(payload, UseCaseLevel(level), subsystem)
            secondary_ids = [item for item in draft.secondary_actor_ids if item in actor_ids]
            payload["useCases"].append(
                {
                    "id": new_id,
                    "level": level,
                    "title": name,
                    "primaryActorId": draft.primary_actor_id,
                    "supportingActorIds": secondary_ids,
                    "subsystem": subsystem,
                    "status": _status_title("inferred"),
                    "priority": _priority_title(draft.priority),
                    "parentUseCaseId": parent_id,
                    "description": draft.description,
                    "precondition": draft.precondition,
                    "sourceTrace": _source_trace(draft.source_refs, source),
                }
            )
            self._sync_parent_relationship(payload, child_id=new_id, parent_id=parent_id)
            return new_id

        # Two passes: L1 first so its real id exists before an L2 draft resolves its
        # parent_local_tag against it. Never trust an actor/tag id the model invented.
        for draft in drafts:
            if draft.level != "L1" or draft.primary_actor_id not in actor_ids:
                continue
            new_id = append_use_case(level="L1", name=draft.name, parent_id=group_id, draft=draft)
            tag_to_id[draft.local_tag] = new_id
            accepted += 1
        for draft in drafts:
            if draft.level != "L2" or draft.primary_actor_id not in actor_ids:
                continue
            parent_id = tag_to_id.get(draft.parent_local_tag or "")
            if parent_id is None:
                continue
            append_use_case(level="L2", name=draft.name, parent_id=parent_id, draft=draft)
            accepted += 1

        model = self._rebuild_model_for_render(payload, project.name)
        payload["plantUml"] = {
            "language": "plantuml",
            "source": render_plantuml(model),
            "editable": True,
            "stale": False,
            "generatedFrom": "use-case-table",
        }
        payload["validation"] = None
        payload["generation"] = {
            "source": "ai",
            "providerConfigId": str(provider.id),
            "provider": provider.provider_type.value,
            "model": provider.model_name,
            "usage": usage,
            "relationsGenerated": bool((payload.get("generation") or {}).get("relationsGenerated")),
            "lastGroupGenerated": group_id,
            "lastGroupUseCasesAdded": accepted,
        }
        await self._save(record, payload)
        return self._response(payload, max_level=body.max_level)

    def _apply_relationship_drafts(self, payload: dict[str, Any], drafts: list[Any]) -> None:
        """Keep hierarchy links, replace every prior include/extend/generalization/association.

        Each draft is checked with the exact same structural rule ``create_relationship`` uses so
        an AI-proposed relation can never bypass a check a manually created one would fail.
        """

        payload["relationships"] = [
            item for item in payload["relationships"] if item.get("type") == RelationshipType.PART_OF.value
        ]
        seen: set[tuple[str, str, str]] = set()
        for draft in drafts:
            if draft.kind == "association":
                continue
            key = (draft.kind, draft.source_id, draft.target_id)
            if key in seen:
                continue
            try:
                request = RelationshipCreateRequest(
                    sourceId=draft.source_id,
                    targetId=draft.target_id,
                    type=RelationshipType(draft.kind),
                    condition=draft.condition,
                )
                self._validate_relationship(payload, request)
            except HTTPException:
                continue
            relation_id = self._next_relationship_id(payload, draft.source_id, draft.target_id)
            payload["relationships"].append(
                {
                    "id": relation_id,
                    "sourceId": draft.source_id,
                    "targetId": draft.target_id,
                    "type": draft.kind,
                    "condition": draft.condition,
                }
            )
            seen.add(key)

    @staticmethod
    def _rebuild_model_for_render(payload: dict[str, Any], system_name: str) -> UseCaseModel:
        """Reconstruct a renderable ``UseCaseModel`` from the persisted FE-shape aggregate.

        Only ``render_plantuml`` consumes the result, so fields the renderer never reads
        (description, precondition, source_refs) get safe placeholders instead of the original
        evidence, which this flattened aggregate does not retain.
        """

        subsystem_ids: dict[str, str] = {}

        def subsystem_id_for(name: str) -> str:
            key = name or "General"
            if key not in subsystem_ids:
                slug = _SLUG_RE.sub("-", key.upper()).strip("-") or "GEN"
                subsystem_ids[key] = f"SUB-{slug}"
            return subsystem_ids[key]

        actors = [
            UseCaseActor(id=item["id"], name=item["name"], kind="human_role", source_refs=["internal"])
            for item in payload["actors"]
        ]
        use_cases: list[UseCaseEntry] = []
        subsystems: dict[str, UseCaseSubsystem] = {}
        for item in payload["useCases"]:
            subsystem_name = item.get("subsystem") or "General"
            subsystem_id = subsystem_id_for(subsystem_name)
            subsystems.setdefault(
                subsystem_id, UseCaseSubsystem(id=subsystem_id, name=subsystem_name, source_refs=["internal"])
            )
            use_cases.append(
                UseCaseEntry(
                    id=item["id"],
                    name=item["title"],
                    level=item["level"],
                    abstraction=_ABSTRACTION_BY_LEVEL.get(item["level"], "user_goal"),
                    primary_actor_id=item["primaryActorId"],
                    secondary_actor_ids=item.get("supportingActorIds") or [],
                    subsystem_id=subsystem_id,
                    parent_use_case_id=item.get("parentUseCaseId"),
                    description=item.get("description") or "Not specified.",
                    precondition=item.get("precondition") or "Not specified.",
                    priority=_PRIORITY_VALUE.get(item["priority"], "should"),
                    status=_STATUS_VALUE.get(item["status"], "suggested"),
                    source_refs=["internal"],
                )
            )
        relations = [
            UseCaseRelation(
                id=item["id"],
                kind=item["type"],
                source_id=item["sourceId"],
                target_id=item["targetId"],
                condition=item.get("condition"),
            )
            for item in payload["relationships"]
            if item.get("type") != RelationshipType.PART_OF.value
        ]
        return UseCaseModel(
            system_name=system_name,
            actors=actors,
            subsystems=list(subsystems.values()),
            use_cases=use_cases,
            relations=relations,
        )

    async def update_plant_uml(
        self,
        *,
        project_id: uuid.UUID,
        body: UseCasePlantUmlUpdateRequest,
    ) -> UseCasePlantUmlResponse:
        """Persist an edited PlantUML document without regenerating the table.

        The source is intentionally kept as text so a developer can adjust layout, skinparams,
        notes, or labels without changing the source-backed use-case aggregate.  Only the two
        PlantUML document markers are required here; PlantUML itself remains the renderer of the
        document's full syntax.
        """

        source = body.source.strip()
        if not _is_plantuml_source(source):
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="PlantUML source must contain @startuml and @enduml markers",
            )
        project = await self._project(project_id)
        payload, record = await self._locked_payload(project)
        plant_uml = {
            "language": "plantuml",
            "source": source,
            "editable": True,
            "stale": False,
            "generatedFrom": "manual",
        }
        payload["plantUml"] = plant_uml
        payload["generation"] = {"source": "manual-uml"}
        await self._save(record, payload)
        return UseCasePlantUmlResponse.model_validate(plant_uml)

    async def _project(self, project_id: uuid.UUID) -> Project:
        project = await self.db.get(Project, project_id)
        if project is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Project not found")
        return project

    async def _record(self, project_id: uuid.UUID, *, for_update: bool = False) -> UseCaseModelRecord | None:
        query = select(UseCaseModelRecord).where(UseCaseModelRecord.project_id == project_id)
        if for_update:
            query = query.with_for_update()
        return (await self.db.execute(query)).scalar_one_or_none()

    async def _payload(self, project: Project) -> dict[str, Any]:
        record = await self._record(project.id)
        if record is None:
            return _empty_payload(project)
        return _normalise_payload(record.model_data, project)

    async def _locked_payload(self, project: Project) -> tuple[dict[str, Any], UseCaseModelRecord]:
        record = await self._record(project.id, for_update=True)
        if record is None:
            record = UseCaseModelRecord(project_id=project.id, model_data=_empty_payload(project))
            self.db.add(record)
            await self.db.flush()
        return _normalise_payload(record.model_data, project), record

    async def _save(self, record: UseCaseModelRecord, payload: dict[str, Any]) -> None:
        record.model_data = payload
        record.last_generation_error = None
        await self.db.flush()

    async def _llm_client(
        self,
        *,
        user_id: uuid.UUID,
        provider_config_id: uuid.UUID | None,
    ) -> tuple[Any, LLMProviderConfig]:
        query = select(LLMProviderConfig).where(
            LLMProviderConfig.user_id == user_id,
            LLMProviderConfig.status == LLMProviderStatus.ACTIVE,
        )
        if provider_config_id is not None:
            query = query.where(LLMProviderConfig.id == provider_config_id)
        else:
            query = query.order_by(LLMProviderConfig.is_default.desc(), LLMProviderConfig.created_at)
        config = (await self.db.execute(query)).scalars().first()
        if config is None:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail="No active LLM provider is configured")
        api_key = decrypt_token(config.encrypted_api_key) if config.encrypted_api_key else None
        if not api_key:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail="LLM API key cannot be decrypted")
        secret_key = decrypt_token(config.encrypted_secret_key) if config.encrypted_secret_key else None
        client = LLMClientFactory.create(
            provider_type=config.provider_type,
            api_key=api_key,
            secret_key=secret_key,
            model=config.model_name,
            region=config.region,
            base_url=config.base_url,
        )
        return client, config

    @staticmethod
    def _find(items: list[dict[str, Any]], item_id: str | None) -> dict[str, Any] | None:
        if not item_id:
            return None
        return next((item for item in items if item.get("id") == item_id), None)

    @staticmethod
    def _next_id(prefix: str, value: str, existing: set[str]) -> str:
        base = _SLUG_RE.sub("-", value.upper()).strip("-")[:40] or prefix
        candidate = f"{prefix}-{base}"
        suffix = 2
        while candidate in existing:
            candidate = f"{prefix}-{base}-{suffix}"
            suffix += 1
        return candidate

    def _next_use_case_id(self, payload: dict[str, Any], level: UseCaseLevel, subsystem: str) -> str:
        existing = {str(item.get("id")) for item in payload["useCases"]}
        prefix = (
            "L" + level.value[1:]
            if level != UseCaseLevel.L2
            else _subsystem_code(subsystem)
        )
        index = 1
        while True:
            candidate = f"UC-{prefix}-{index:03d}"
            if candidate not in existing:
                return candidate
            index += 1

    def _next_relationship_id(self, payload: dict[str, Any], source_id: str, target_id: str) -> str:
        existing = {str(item.get("id")) for item in payload["relationships"]}
        base = (
            f"REL-{_SLUG_RE.sub('-', source_id.upper()).strip('-')}-"
            f"{_SLUG_RE.sub('-', target_id.upper()).strip('-')}"
        )
        candidate = base
        suffix = 2
        while candidate in existing:
            candidate = f"{base}-{suffix}"
            suffix += 1
        return candidate

    @staticmethod
    def _use_case_dict(use_case_id: str, body: UseCaseCreateRequest) -> dict[str, Any]:
        data = body.model_dump(by_alias=True)
        data["id"] = use_case_id
        data["level"] = body.level.value
        data["status"] = body.status.value
        data["priority"] = body.priority.value
        return data

    def _validate_use_case_refs(
        self,
        payload: dict[str, Any],
        primary_actor_id: str,
        supporting_actor_ids: list[str],
        parent_id: str | None,
        *,
        current_id: str | None = None,
        child_level: str | None = None,
    ) -> None:
        actor_ids = {item.get("id") for item in payload["actors"]}
        if primary_actor_id not in actor_ids or any(item not in actor_ids for item in supporting_actor_ids):
            raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Use case references an unknown actor")
        if primary_actor_id in supporting_actor_ids or len(supporting_actor_ids) != len(set(supporting_actor_ids)):
            raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Use case actor references must be unique")
        if parent_id is not None:
            parent = self._find(payload["useCases"], parent_id)
            if parent_id == current_id or parent is None:
                raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Use case references an invalid parent")
            parent_rank = _LEVEL_ORDER.get(str(parent.get("level")), -1)
            child_rank = _LEVEL_ORDER.get(child_level, 99) if child_level is not None else 99
            if child_level is not None and parent_rank >= child_rank:
                raise HTTPException(
                    status.HTTP_400_BAD_REQUEST,
                    detail="Parent use case must be at a higher abstraction level",
                )
            if current_id is not None:
                seen: set[str] = set()
                ancestor_id: str | None = parent_id
                while ancestor_id is not None:
                    if ancestor_id in seen or ancestor_id == current_id:
                        raise HTTPException(
                            status.HTTP_400_BAD_REQUEST,
                            detail="Use case parent would create a hierarchy cycle",
                        )
                    seen.add(ancestor_id)
                    ancestor = self._find(payload["useCases"], ancestor_id)
                    ancestor_id = ancestor.get("parentUseCaseId") if ancestor is not None else None

    def _validate_relationship(self, payload: dict[str, Any], body: RelationshipCreateRequest) -> None:
        actors = {item.get("id") for item in payload["actors"]}
        use_cases = {item.get("id") for item in payload["useCases"]}
        if body.source_id == body.target_id:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="A relationship cannot point to itself")
        if body.type == RelationshipType.ASSOCIATION and not (
            (body.source_id in actors and body.target_id in use_cases)
            or (body.target_id in actors and body.source_id in use_cases)
        ):
            raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Association requires one actor and one use case")
        if body.type in {
            RelationshipType.PART_OF,
            RelationshipType.INCLUDE,
            RelationshipType.EXTEND,
        } and not (body.source_id in use_cases and body.target_id in use_cases):
            raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Relationship requires two use cases")
        if body.type == RelationshipType.PART_OF:
            parent = self._find(payload["useCases"], body.source_id)
            child = self._find(payload["useCases"], body.target_id)
            parent_rank = _LEVEL_ORDER.get(str(parent.get("level")), -1) if parent else -1
            child_rank = _LEVEL_ORDER.get(str(child.get("level")), 99) if child else 99
            if parent_rank >= child_rank:
                raise HTTPException(
                    status.HTTP_400_BAD_REQUEST,
                    detail="part-of parent must be at a higher abstraction level",
                )
            ancestor_id: str | None = body.source_id
            seen: set[str] = set()
            while ancestor_id is not None:
                if ancestor_id in seen or ancestor_id == body.target_id:
                    raise HTTPException(
                        status.HTTP_400_BAD_REQUEST,
                        detail="part-of relationship would create a hierarchy cycle",
                    )
                seen.add(ancestor_id)
                ancestor = self._find(payload["useCases"], ancestor_id)
                ancestor_id = ancestor.get("parentUseCaseId") if ancestor is not None else None
        if body.type == RelationshipType.EXTEND and not body.condition:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Extend relationship requires a condition")
        if body.type == RelationshipType.GENERALIZATION and not (
            (body.source_id in actors and body.target_id in actors)
            or (body.source_id in use_cases and body.target_id in use_cases)
        ):
            raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Generalization endpoints are invalid")

    @staticmethod
    def _sync_parent_relationship(
        payload: dict[str, Any],
        *,
        child_id: str,
        parent_id: str | None,
        relationship_id: str | None = None,
    ) -> None:
        payload["relationships"] = [
            item
            for item in payload["relationships"]
            if not (item.get("type") == RelationshipType.PART_OF.value and item.get("targetId") == child_id)
        ]
        child = next((item for item in payload["useCases"] if item.get("id") == child_id), None)
        if child is not None:
            child["parentUseCaseId"] = parent_id
        if parent_id is not None:
            relation_id = relationship_id or f"REL-PART-OF-{parent_id}-{child_id}"
            payload["relationships"].append(
                {
                    "id": relation_id,
                    "sourceId": parent_id,
                    "targetId": child_id,
                    "type": RelationshipType.PART_OF.value,
                }
            )

    @staticmethod
    def _invalidate_manual_state(payload: dict[str, Any]) -> None:
        payload["validation"] = None
        payload["generation"] = {"source": "manual"}
        plant_uml = payload.get("plantUml")
        if isinstance(plant_uml, dict):
            plant_uml["stale"] = True

    def _response(
        self,
        payload: dict[str, Any],
        *,
        max_level: UseCaseLevel = UseCaseLevel.L2,
        include_actors: bool = True,
        include_relationships: bool = True,
    ) -> UseCaseModelResponse:
        data = copy.deepcopy(payload)
        if data.get("diagrams"):
            # A structural issue in one diagram must not hide the other diagrams. Generation can
            # persist a render plan for valid diagrams while omitting plans whose edge endpoints
            # are malformed. Reconstruct only the missing plans from the semantic aggregate so
            # the FE can review every level/subsystem; validation remains authoritative for SRS.
            existing_plans = data.get("diagramPlans") or []
            existing_ids = {
                str(item.get("diagramId"))
                for item in existing_plans
                if isinstance(item, dict) and item.get("diagramId")
            }
            missing_plans = [
                item
                for item in _draft_diagram_plans(data)
                if str(item.get("diagramId")) not in existing_ids
            ]
            data["diagramPlans"] = [*existing_plans, *missing_plans]
        # Older plans did not carry the subsystem label. Enrich them from the semantic
        # diagram definitions so each L1 plan is distinguishable in the FE selector.
        diagrams_by_id = {
            str(item.get("id")): item
            for item in data.get("diagrams", [])
            if isinstance(item, dict) and item.get("id")
        }
        for plan in data.get("diagramPlans", []):
            if isinstance(plan, dict) and "subsystem" not in plan:
                diagram = diagrams_by_id.get(str(plan.get("diagramId")), {})
                plan["subsystem"] = diagram.get("subsystem")
        max_rank = _LEVEL_ORDER[max_level.value]
        visible_use_cases = [
            item for item in data.get("useCases", []) if _LEVEL_ORDER.get(str(item.get("level")), 2) <= max_rank
        ]
        visible_ids = {item.get("id") for item in visible_use_cases}
        data["useCases"] = visible_use_cases
        data["actors"] = data.get("actors", []) if include_actors else []
        if include_relationships:
            data["relationships"] = [
                item
                for item in data.get("relationships", [])
                if item.get("sourceId") in visible_ids or item.get("targetId") in visible_ids
            ]
        else:
            data["relationships"] = []
            for diagram in data.get("diagrams", []):
                diagram["relationIds"] = []
            for plan in data.get("diagramPlans", []):
                plan["edges"] = []
        data["diagrams"] = [
            item
            for item in data.get("diagrams", [])
            if _LEVEL_ORDER.get(str(item.get("level")), 2) <= max_rank
        ]
        data["diagramPlans"] = [
            item
            for item in data.get("diagramPlans", [])
            if _LEVEL_ORDER.get(str(item.get("level")), 2) <= max_rank
        ]
        # Keep the overview first, followed by deterministic subsystem diagrams. This makes the
        # FE default view predictable and prevents the first L1 plan from looking like the model
        # lost its L0 overview.
        data["diagrams"].sort(
            key=lambda item: (
                _LEVEL_ORDER.get(str(item.get("level")), 2),
                str(item.get("subsystem") or ""),
                str(item.get("id") or ""),
            )
        )
        data["diagramPlans"].sort(
            key=lambda item: (
                _LEVEL_ORDER.get(str(item.get("level")), 2),
                str(item.get("subsystem") or ""),
                str(item.get("diagramId") or ""),
            )
        )
        response = UseCaseModelResponse.model_validate(data)
        return response

    @staticmethod
    def _detail_response(response: UseCaseModelResponse, use_case_id: str) -> UseCaseResponse:
        row = next((item for item in response.use_cases if item.id == use_case_id), None)
        if row is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Use case not found")
        item = UseCaseResponse.model_validate(row.model_dump(by_alias=True))
        item.relationships = [
            relation
            for relation in response.relationships
            if relation.source_id == use_case_id or relation.target_id == use_case_id
        ]
        return item

    def _core_to_payload(
        self,
        *,
        project: Project,
        source: RequirementsSourceSnapshot,
        model: UseCaseModel,
        report,
        provider: LLMProviderConfig | None,
        usage: dict[str, int] | None,
        relations_generated: bool = True,
    ) -> dict[str, Any]:
        subsystem_names = {item.id: item.name for item in model.subsystems}
        primary_actor_ids = {item.primary_actor_id for item in model.use_cases}
        actors = [
            {
                "id": item.id,
                "name": item.name,
                "kind": ActorKind.PRIMARY.value if item.id in primary_actor_ids else ActorKind.SUPPORTING.value,
            }
            for item in model.actors
        ]
        use_cases = [
            {
                "id": item.id,
                "level": item.level,
                "title": item.name,
                "primaryActorId": item.primary_actor_id,
                "supportingActorIds": item.secondary_actor_ids,
                "subsystem": subsystem_names.get(item.subsystem_id, item.subsystem_id),
                "status": _status_title(item.status),
                "priority": _priority_title(item.priority),
                "parentUseCaseId": item.parent_use_case_id,
                "description": item.description,
                "precondition": item.precondition,
                "sourceTrace": _source_trace(item.source_refs, source),
            }
            for item in model.use_cases
        ]
        relationships: list[dict[str, Any]] = []
        for item in model.use_cases:
            if item.parent_use_case_id:
                relationships.append(
                    {
                        "id": f"REL-PART-OF-{item.parent_use_case_id}-{item.id}",
                        "sourceId": item.parent_use_case_id,
                        "targetId": item.id,
                        "type": RelationshipType.PART_OF.value,
                    }
                )
        for relation in model.relations:
            relationships.append(
                {
                    "id": relation.id,
                    "sourceId": relation.source_id,
                    "targetId": relation.target_id,
                    "type": relation.kind,
                    "condition": relation.condition,
                }
            )
        validation = UseCaseValidationResponse.model_validate(report.model_dump())
        plant_uml = {
            "language": "plantuml",
            "source": render_plantuml(model),
            "editable": True,
            "stale": False,
            "generatedFrom": "use-case-table",
        }
        payload = UseCaseModelResponse(
            projectId=str(project.id),
            projectName=project.name,
            actors=actors,
            useCases=use_cases,
            relationships=relationships,
            diagrams=[],
            diagramPlans=[],
            sourceHash=source.source_hash,
            validation=validation,
            plantUml=plant_uml,
            generation=(
                {
                    "source": "ai",
                    "providerConfigId": str(provider.id),
                    "provider": provider.provider_type.value,
                    "model": provider.model_name,
                    "usage": usage,
                    "relationsGenerated": relations_generated,
                }
                if provider is not None
                else {
                    # Phase 1 (generate_groups) is deterministic -- built straight from the stored
                    # PRD's business-capability/functional-requirement structure, no LLM call, so
                    # there is no provider/usage to report.
                    "source": "deterministic",
                    "relationsGenerated": relations_generated,
                }
            ),
        )
        return payload.model_dump(by_alias=True, mode="json")


def _empty_payload(project: Project) -> dict[str, Any]:
    return UseCaseModelResponse(projectId=str(project.id), projectName=project.name).model_dump(
        by_alias=True, mode="json"
    )


def _draft_diagram_plans(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Build render-only plans from a persisted FE aggregate.

    Generation stores the semantic diagram definitions even when validation blocks SRS
    eligibility.  This fallback deliberately performs only endpoint and notation checks; the
    validation report remains responsible for evidence, language, actor coverage, and level
    correctness.  Invalid edges are skipped so one malformed relation cannot hide the complete
    draft canvas.
    """

    actors = {str(item.get("id")): item for item in payload.get("actors", []) if item.get("id")}
    use_cases = {str(item.get("id")): item for item in payload.get("useCases", []) if item.get("id")}
    relationships = {str(item.get("id")): item for item in payload.get("relationships", []) if item.get("id")}
    plans: list[dict[str, Any]] = []

    for diagram in payload.get("diagrams", []):
        diagram_id = str(diagram.get("id") or "")
        if not diagram_id:
            continue
        actor_ids = [str(item) for item in diagram.get("actorIds", []) if str(item) in actors]
        use_case_ids = [str(item) for item in diagram.get("useCaseIds", []) if str(item) in use_cases]
        node_ids = set(actor_ids) | set(use_case_ids)
        primary_actor_ids = {
            str(use_cases[item].get("primaryActorId"))
            for item in use_case_ids
            if use_cases[item].get("primaryActorId")
        }
        secondary_actor_ids = {
            str(actor_id)
            for item in use_case_ids
            for actor_id in use_cases[item].get("supportingActorIds", [])
        }
        nodes: list[dict[str, Any]] = [
            {
                "id": f"BOUNDARY-{diagram_id}",
                "kind": "system_boundary",
                "label": str(diagram.get("systemBoundary") or "System"),
                "shape": "rectangle",
                "side": "inside",
            }
        ]
        for actor_id in actor_ids:
            actor = actors[actor_id]
            nodes.append(
                {
                    "id": actor_id,
                    "kind": "actor",
                    "label": str(actor.get("name") or actor_id),
                    "shape": "actor",
                    "side": (
                        "right" if actor_id in secondary_actor_ids and actor_id not in primary_actor_ids else "left"
                    ),
                }
            )
        for use_case_id in use_case_ids:
            item = use_cases[use_case_id]
            nodes.append(
                {
                    "id": use_case_id,
                    "kind": "use_case",
                    "label": f"{use_case_id} {item.get('title') or use_case_id}",
                    "shape": "ellipse",
                    "side": "inside",
                }
            )

        edges: list[dict[str, Any]] = []
        for relation_id in diagram.get("relationIds", []):
            relation = relationships.get(str(relation_id))
            if relation is None:
                continue
            kind = str(relation.get("type") or "")
            source_id = str(relation.get("sourceId") or "")
            target_id = str(relation.get("targetId") or "")
            if kind == RelationshipType.PART_OF.value or source_id not in node_ids or target_id not in node_ids:
                continue
            source_is_actor = source_id in actors
            target_is_actor = target_id in actors
            source_is_use_case = source_id in use_cases
            target_is_use_case = target_id in use_cases
            if kind == RelationshipType.ASSOCIATION.value:
                if not ((source_is_actor and target_is_use_case) or (source_is_use_case and target_is_actor)):
                    continue
                line_style, directed, marker, label = "solid", False, "none", None
            elif kind in {RelationshipType.INCLUDE.value, RelationshipType.EXTEND.value}:
                if not (source_is_use_case and target_is_use_case):
                    continue
                line_style, directed, marker, label = "dashed", True, "open_arrow", f"«{kind}»"
            elif kind == RelationshipType.GENERALIZATION.value:
                if not ((source_is_actor and target_is_actor) or (source_is_use_case and target_is_use_case)):
                    continue
                line_style, directed, marker, label = "solid", True, "open_triangle", None
            else:
                continue
            edges.append(
                {
                    "id": str(relation.get("id") or relation_id),
                    "sourceId": source_id,
                    "targetId": target_id,
                    "kind": kind,
                    "lineStyle": line_style,
                    "directed": directed,
                    "marker": marker,
                    "label": label,
                    "condition": relation.get("condition"),
                }
            )
        plans.append(
            {
                "diagramId": diagram_id,
                "level": diagram.get("level"),
                "systemBoundary": str(diagram.get("systemBoundary") or "System"),
                "subsystem": diagram.get("subsystem"),
                "nodes": nodes,
                "edges": edges,
            }
        )
    return plans


def _normalise_payload(raw: Any, project: Project) -> dict[str, Any]:
    payload = copy.deepcopy(raw) if isinstance(raw, dict) else {}
    payload.setdefault("projectId", str(project.id))
    payload["projectName"] = project.name
    for key in ("actors", "useCases", "relationships", "diagrams", "diagramPlans"):
        if not isinstance(payload.get(key), list):
            payload[key] = []
    if not isinstance(payload.get("plantUml"), dict):
        payload["plantUml"] = None
    for item in payload["useCases"]:
        if isinstance(item, dict):
            # Older drafts briefly stored detail-only relationships inside each row.  Keep the
            # aggregate normalized so the list response remains the documented flat shape.
            item.pop("relationships", None)
    return payload


def _status_title(value: str) -> str:
    return {"confirmed": "Confirmed", "inferred": "Inferred", "suggested": "Suggested"}.get(
        value.lower(), value.title()
    )


def _subsystem_code(value: str) -> str:
    """Create the short, stable-looking subsystem prefix used for manually added L2 rows."""

    words = re.findall(r"[A-Z0-9]+", value.upper())
    if not words:
        return "L2"
    if len(words) == 1:
        return words[0][:4] or "L2"
    return "".join(word[0] for word in words)[:4] or "L2"


def _priority_title(value: str) -> str:
    return {"must": "Must", "should": "Should", "could": "Could"}.get(value.lower(), value.title())


def _source_trace(refs: list[str], source: RequirementsSourceSnapshot) -> list[str]:
    evidence = source.evidence_by_id()
    traces: list[str] = []
    for ref in refs:
        item = evidence.get(ref)
        if item is None:
            traces.append(ref)
            continue
        traces.append(f"{item.document_type.upper()} · {item.locator}")
    return traces


def _decode_llm_payload(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    content = getattr(raw, "content", raw)
    if isinstance(content, list):
        content = "".join(str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content)
    if not isinstance(content, str):
        raise ValueError("LLM did not return a JSON object")
    parsed = json.loads(content.strip().removeprefix("```json").removesuffix("```").strip())
    if not isinstance(parsed, dict):
        raise ValueError("LLM response must be a JSON object")
    return parsed


def _is_plantuml_source(source: str) -> bool:
    return bool(
        re.search(r"(?im)^\s*@startuml(?:\s|$)", source)
        and re.search(r"(?im)^\s*@enduml(?:\s|$)", source)
    )
