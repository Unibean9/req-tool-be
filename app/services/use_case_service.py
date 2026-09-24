"""Persistence and generation service for the BRD/PRD-backed use-case model."""

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
    ActorResponse,
    ActorUpdateRequest,
    RelationshipCreateRequest,
    RelationshipType,
    UseCaseCreateRequest,
    UseCaseGenerateRequest,
    UseCaseModelResponse,
    UseCasePlantUmlResponse,
    UseCasePlantUmlUpdateRequest,
    UseCaseRelationshipResponse,
    UseCaseResponse,
    UseCaseUpdateRequest,
    UseCaseValidationResponse,
)
from app.services.llm_clients import LLMClientFactory
from app.use_cases.completion import complete_use_case_table
from app.use_cases.harness import (
    UseCaseGroupsHarness,
    UseCaseGroupUseCasesHarness,
    UseCaseRelationshipHarness,
)
from app.use_cases.models import (
    RequirementsSourceSnapshot,
    UseCaseActor,
    UseCaseEntry,
    UseCaseGroupDetailDraftList,
    UseCaseGroupsDraftList,
    UseCaseModel,
    UseCaseModule,
    UseCaseRelation,
    UseCaseRelationshipDraftList,
    UseCaseSystem,
)
from app.use_cases.plantuml import render_plantuml
from app.use_cases.rules import validate_use_case_model
from app.use_cases.source_loader import canonical_evidence_refs, load_project_requirements_source

_SLUG_RE = re.compile(r"[^A-Z0-9]+")


class UseCaseService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_model(
        self,
        *,
        project_id: uuid.UUID,
        max_level: Any = None,
        include_actors: bool = True,
        include_relationships: bool = True,
    ) -> UseCaseModelResponse:
        _ = max_level
        project = await self._project(project_id)
        payload = await self._payload(project)
        data = copy.deepcopy(payload)
        if not include_actors:
            data["actors"] = []
        if not include_relationships:
            data["relationships"] = []
        return UseCaseModelResponse.model_validate(data)

    async def get_use_case(self, *, project_id: uuid.UUID, use_case_id: str) -> UseCaseResponse:
        project = await self._project(project_id)
        payload = await self._payload(project)
        row = next((item for item in payload.get("useCases", []) if item.get("id") == use_case_id), None)
        if row is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Use case not found")
        related = [
            item
            for item in payload.get("relationships", [])
            if item.get("sourceId") == use_case_id or item.get("targetId") == use_case_id
        ]
        return UseCaseResponse.model_validate({**row, "relationships": related})

    # Actor editing is retained for backwards compatibility; generated use cases themselves are
    # not manually created/edited/deleted by the new UI.
    async def create_actor(self, *, project_id: uuid.UUID, body: ActorCreateRequest) -> ActorResponse:
        project = await self._project(project_id)
        payload, record = await self._locked_payload(project)
        actor_id = self._next_id("ACT", body.name, {str(item.get("id")) for item in payload["actors"]})
        actor = {"id": actor_id, "name": body.name.strip(), "kind": body.kind.value, "side": None}
        payload["actors"].append(actor)
        self._mark_table_changed(payload)
        await self._save(record, payload)
        return ActorResponse.model_validate(actor)

    async def update_actor(self, *, project_id: uuid.UUID, actor_id: str, body: ActorUpdateRequest) -> ActorResponse:
        project = await self._project(project_id)
        payload, record = await self._locked_payload(project)
        actor = self._find(payload["actors"], actor_id)
        if actor is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Actor not found")
        actor["name"] = body.name.strip()
        self._mark_table_changed(payload)
        await self._save(record, payload)
        return ActorResponse.model_validate(actor)

    async def delete_actor(self, *, project_id: uuid.UUID, actor_id: str) -> dict[str, Any]:
        project = await self._project(project_id)
        payload, record = await self._locked_payload(project)
        if self._find(payload["actors"], actor_id) is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Actor not found")
        refs = [
            item["id"]
            for item in payload["useCases"]
            if item.get("primaryActorId") == actor_id or actor_id in item.get("secondaryActorIds", [])
        ]
        if refs:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                detail={"message": "Actor is still referenced by generated use cases", "use_case_ids": refs},
            )
        payload["actors"] = [item for item in payload["actors"] if item.get("id") != actor_id]
        self._mark_table_changed(payload)
        await self._save(record, payload)
        return {"id": actor_id, "deleted": True}

    async def create_use_case(self, *, project_id: uuid.UUID, body: UseCaseCreateRequest) -> UseCaseResponse:
        _ = (project_id, body)
        raise HTTPException(
            status.HTTP_405_METHOD_NOT_ALLOWED,
            detail="Use cases are generated from BRD/PRD; edit PlantUML source instead",
        )

    async def update_use_case(
        self, *, project_id: uuid.UUID, use_case_id: str, body: UseCaseUpdateRequest
    ) -> UseCaseResponse:
        _ = (project_id, use_case_id, body)
        raise HTTPException(
            status.HTTP_405_METHOD_NOT_ALLOWED,
            detail="Use cases are generated from BRD/PRD; edit PlantUML source instead",
        )

    async def delete_use_case(self, *, project_id: uuid.UUID, use_case_id: str) -> dict[str, Any]:
        _ = (project_id, use_case_id)
        raise HTTPException(
            status.HTTP_405_METHOD_NOT_ALLOWED,
            detail="Use cases are generated from BRD/PRD; edit PlantUML source instead",
        )

    async def create_relationship(
        self, *, project_id: uuid.UUID, body: RelationshipCreateRequest
    ) -> UseCaseRelationshipResponse:
        project = await self._project(project_id)
        payload, record = await self._locked_payload(project)
        use_case_ids = {item.get("id") for item in payload["useCases"]}
        if body.source_id not in use_case_ids or body.target_id not in use_case_ids:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, detail="Relationship endpoints must be generated use cases"
            )
        if body.type == RelationshipType.EXTEND and not body.condition:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Extend requires a condition")
        relation = {
            "id": self._next_relationship_id(payload, body.source_id, body.target_id),
            "sourceId": body.source_id,
            "targetId": body.target_id,
            "type": body.type.value,
            "condition": body.condition,
            "reason": body.reason,
            "sourceTrace": [],
        }
        payload["relationships"].append(relation)
        self._refresh_relationship_ids(payload)
        self._mark_table_changed(payload)
        await self._save(record, payload)
        return UseCaseRelationshipResponse.model_validate(relation)

    async def delete_relationship(self, *, project_id: uuid.UUID, relationship_id: str) -> dict[str, Any]:
        project = await self._project(project_id)
        payload, record = await self._locked_payload(project)
        if self._find(payload["relationships"], relationship_id) is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Relationship not found")
        payload["relationships"] = [item for item in payload["relationships"] if item.get("id") != relationship_id]
        self._refresh_relationship_ids(payload)
        self._mark_table_changed(payload)
        await self._save(record, payload)
        return {"id": relationship_id, "deleted": True}

    async def generate(
        self, *, project_id: uuid.UUID, user_id: uuid.UUID, body: UseCaseGenerateRequest
    ) -> UseCaseModelResponse:
        project = await self._project(project_id)
        try:
            source = await load_project_requirements_source(self.db, project_id=project_id)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "code": "USE_CASE_SOURCE_LOAD_FAILED",
                    "message": f"BRD/PRD components could not be loaded: {str(exc)[:400]}",
                },
            ) from exc
        provider = None
        usage: Any = None
        candidate_rows: list[UseCaseEntry] = []
        generation_errors: list[str] = []
        completed_batches = 0
        total_batches = 0

        # Build the source-backed floor before touching the provider.  This keeps generation
        # useful when the provider is slow or returns malformed JSON and prevents a single bad
        # source row from escaping as an unhandled HTTP 500.
        try:
            floor = complete_use_case_table(source)
        except Exception as exc:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "code": "USE_CASE_SOURCE_INVALID",
                    "message": f"Stored BRD/PRD components could not be mapped to a use-case table: {str(exc)[:400]}",
                },
            ) from exc
        try:
            client, provider = await self._llm_client(user_id=user_id, provider_config_id=body.provider_config_id)
            batches = [
                module for module in floor.modules if any(item.module_id == module.id for item in floor.use_cases)
            ]
            total_batches = len(batches)
            batch_semaphore = asyncio.Semaphore(3)

            async def generate_module(module: UseCaseModule) -> tuple[UseCaseModule, list[Any], Any, str | None]:
                rows = [item for item in floor.use_cases if item.module_id == module.id]
                group = module.model_dump(mode="json")
                group["sourceUseCases"] = [
                    item.model_dump(mode="json", by_alias=True) for item in rows
                ]
                context, evidence = _module_generation_context(source, module, rows)
                harness = UseCaseGroupUseCasesHarness(
                    source=source,
                    group=group,
                    actors=[actor.model_dump(mode="json") for actor in floor.actors],
                    source_text=context,
                    evidence_text=evidence,
                )
                try:
                    async with batch_semaphore:
                        raw_result, batch_usage = await asyncio.wait_for(
                            client.generate(
                                messages=[{"role": "user", "content": harness.build_user_prompt()}],
                                system=harness.build_system_instruction(),
                                # A module has a small bounded output; reserving the global token budget
                                # here makes each batch cheaper and less likely to hit a provider limit.
                                max_tokens=min(settings.use_case_generation_max_tokens, 8000),
                                response_format=harness.response_format(),
                            ),
                            timeout=min(settings.use_case_generation_timeout_seconds, 90.0),
                        )
                    drafts = UseCaseGroupDetailDraftList.model_validate(_decode_llm_payload(raw_result)).use_cases
                    return module, drafts, batch_usage, None
                except TimeoutError:
                    return module, [], None, f"{module.name}: generation timed out"
                except Exception as exc:
                    return module, [], None, f"{module.name}: {str(exc)[:240]}"

            # Run module passes concurrently.  Each request carries only the relevant source
            # excerpt, so the total wall time is bounded by the slowest module rather than the
            # size of the complete BRD/PRD snapshot.
            results = await asyncio.gather(*(generate_module(module) for module in batches))
            usages: list[Any] = []
            for module, drafts, batch_usage, error in results:
                if error:
                    generation_errors.append(error)
                    continue
                completed_batches += 1
                if batch_usage is not None:
                    usages.append(batch_usage)
                candidate_rows.extend(_candidate_rows_from_drafts(floor, module, drafts))
            usage = {"batches": usages} if usages else None
        except HTTPException:
            raise
        except Exception as exc:
            generation_errors.append(f"generation setup failed: {str(exc)[:300]}")

        candidate: UseCaseModel | None = None
        if candidate_rows:
            candidate = UseCaseModel(
                system=floor.system,
                modules=floor.modules,
                actors=floor.actors,
                use_cases=candidate_rows,
                relationships=[],
            )
        model = complete_use_case_table(source, candidate)
        generation_error = "; ".join(generation_errors) if generation_errors else None
        if total_batches and completed_batches < total_batches:
            prefix = f"AI details completed for {completed_batches}/{total_batches} modules; "
            generation_error = prefix + (generation_error or "source-backed details retained for failed modules")
        report = validate_use_case_model(model, source)
        payload = self._core_to_payload(
            project=project,
            source=source,
            model=model,
            report=report,
            provider=provider,
            usage=usage,
            generation_error=generation_error,
        )
        payload["generation"] = payload.get("generation") or {}
        payload["generation"].update(
            {
                "batchCount": total_batches,
                "completedBatchCount": completed_batches,
                "generationMode": "module-batch",
            }
        )
        await self._persist_generated(project_id=project_id, user_id=user_id, source=source, payload=payload)
        return UseCaseModelResponse.model_validate(payload)

    async def generate_groups(
        self, *, project_id: uuid.UUID, user_id: uuid.UUID, body: UseCaseGenerateRequest
    ) -> UseCaseModelResponse:
        project = await self._project(project_id)
        source = await load_project_requirements_source(self.db, project_id=project_id)
        provider = None
        usage = None
        try:
            client, provider = await self._llm_client(user_id=user_id, provider_config_id=body.provider_config_id)
            harness = UseCaseGroupsHarness(source)
            raw_result, usage = await asyncio.wait_for(
                client.generate(
                    messages=[{"role": "user", "content": harness.build_user_prompt()}],
                    system=harness.build_system_instruction(),
                    max_tokens=settings.use_case_generation_max_tokens,
                    response_format=harness.response_format(),
                ),
                timeout=settings.use_case_generation_timeout_seconds,
            )
            drafts = UseCaseGroupsDraftList.model_validate(_decode_llm_payload(raw_result))
        except TimeoutError as exc:
            raise HTTPException(
                status.HTTP_504_GATEWAY_TIMEOUT,
                detail={
                    "code": "USE_CASE_GROUPS_GENERATION_TIMEOUT",
                    "message": "Module extraction timed out; retry the smaller group generation flow.",
                },
            ) from exc
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status.HTTP_502_BAD_GATEWAY,
                detail={"code": "USE_CASE_GROUPS_GENERATION_FAILED", "message": str(exc)[:500]},
            ) from exc

        actors: list[UseCaseActor] = []
        actor_ids: set[str] = set()
        for draft in drafts.actors:
            actor_id = self._next_id("ACT", draft.name, actor_ids)
            actor_ids.add(actor_id)
            actors.append(
                UseCaseActor(id=actor_id, name=draft.name.strip(), kind=draft.kind, source_refs=draft.source_refs)
            )
        modules: list[UseCaseModule] = []
        for group in drafts.groups:
            module_id = self._next_id("SUB", group.name, {item.id for item in modules})
            modules.append(
                UseCaseModule(
                    id=module_id, name=group.name.strip(), goal=group.goal.strip(), source_refs=group.source_refs
                )
            )
            for role in group.user_segment:
                if not any(item.name.casefold() == role.casefold() for item in actors):
                    actor_id = self._next_id("ACT", role, actor_ids)
                    actor_ids.add(actor_id)
                    actors.append(
                        UseCaseActor(id=actor_id, name=role.strip(), kind="human", source_refs=group.source_refs)
                    )
        if not modules:
            fallback = complete_use_case_table(source)
            modules = fallback.modules
            actors = fallback.actors
        model = UseCaseModel(
            system=UseCaseSystem(id="SYSTEM", name=project.name),
            modules=modules,
            actors=actors,
            use_cases=[],
            relationships=[],
        )
        report = validate_use_case_model(model, source)
        payload = self._core_to_payload(
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
            record = UseCaseModelRecord(project_id=project_id, model_data=payload)
            self.db.add(record)
            await self.db.flush()
        record.model_data = payload
        record.source_hash = source.source_hash
        record.generated_by_id = user_id
        await self.db.flush()
        return UseCaseModelResponse.model_validate(payload)

    async def generate_group_use_cases(
        self, *, project_id: uuid.UUID, user_id: uuid.UUID, group_id: str, body: UseCaseGenerateRequest
    ) -> UseCaseModelResponse:
        project = await self._project(project_id)
        payload, record = await self._locked_payload(project)
        module = self._find(payload.get("modules", []), group_id)
        if module is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Use-case module not found")
        source = await load_project_requirements_source(self.db, project_id=project_id)
        provider = None
        usage = None
        try:
            client, provider = await self._llm_client(user_id=user_id, provider_config_id=body.provider_config_id)
            harness = UseCaseGroupUseCasesHarness(
                source=source,
                group=module,
                actors=[{"id": item["id"], "name": item["name"]} for item in payload["actors"]],
            )
            raw_result, usage = await asyncio.wait_for(
                client.generate(
                    messages=[{"role": "user", "content": harness.build_user_prompt()}],
                    system=harness.build_system_instruction(),
                    max_tokens=settings.use_case_generation_max_tokens,
                    response_format=harness.response_format(),
                ),
                timeout=settings.use_case_generation_timeout_seconds,
            )
            drafts = UseCaseGroupDetailDraftList.model_validate(_decode_llm_payload(raw_result)).use_cases
        except TimeoutError as exc:
            raise HTTPException(
                status.HTTP_504_GATEWAY_TIMEOUT,
                detail={
                    "code": "USE_CASE_GROUP_GENERATION_TIMEOUT",
                    "message": "This module generation timed out; retry the module.",
                },
            ) from exc
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status.HTTP_502_BAD_GATEWAY,
                detail={"code": "USE_CASE_GROUP_GENERATION_FAILED", "message": str(exc)[:500]},
            ) from exc

        # Idempotent replacement for this module; module rows are never synthetic parent use cases.
        stale = {item["id"] for item in payload["useCases"] if item.get("moduleId") == module["id"]}
        payload["useCases"] = [item for item in payload["useCases"] if item.get("id") not in stale]
        payload["relationships"] = [
            item
            for item in payload["relationships"]
            if item.get("sourceId") not in stale and item.get("targetId") not in stale
        ]
        actor_ids = {item["id"] for item in payload["actors"]}
        next_id = self._next_use_case_id
        accepted = 0
        for draft in drafts:
            if draft.primary_actor_id not in actor_ids:
                continue
            supporting = [
                item for item in draft.secondary_actor_ids if item in actor_ids and item != draft.primary_actor_id
            ]
            use_case_id = next_id(payload, module["name"])
            row = self._draft_row(use_case_id, module["id"], draft, supporting, source)
            payload["useCases"].append(row)
            accepted += 1
        model = self._payload_to_model(payload, project.name, source=source)
        payload["plantUml"] = self._plantuml_payload(model)
        self._sync_actor_sides(payload, model)
        payload["validation"] = UseCaseValidationResponse.model_validate(
            validate_use_case_model(model, source).model_dump()
        ).model_dump(by_alias=True)
        payload["generation"] = {
            "source": "ai",
            "providerConfigId": str(provider.id) if provider else None,
            "provider": provider.provider_type.value if provider else None,
            "model": provider.model_name if provider else None,
            "usage": usage,
            "relationsGenerated": bool((payload.get("generation") or {}).get("relationsGenerated")),
            "lastModuleGenerated": module["id"],
            "lastModuleUseCasesAdded": accepted,
        }
        self._refresh_relationship_ids(payload)
        payload = _normalise_payload(payload, project)
        await self._save(record, payload)
        return UseCaseModelResponse.model_validate(payload)

    async def generate_relations(
        self, *, project_id: uuid.UUID, user_id: uuid.UUID, body: UseCaseGenerateRequest
    ) -> UseCaseModelResponse:
        project = await self._project(project_id)
        payload, record = await self._locked_payload(project)
        if not payload["useCases"]:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                detail={
                    "code": "USE_CASE_TABLE_REQUIRED",
                    "message": "Generate the use-case table before generating relationships.",
                },
            )
        source = await load_project_requirements_source(self.db, project_id=project_id)
        try:
            client, provider = await self._llm_client(user_id=user_id, provider_config_id=body.provider_config_id)
            harness = UseCaseRelationshipHarness(
                source=source,
                use_cases=payload["useCases"],
                actors=payload["actors"],
                source_text=_relationship_generation_context(source),
                evidence_text=_bounded_evidence_index(source, 14000),
            )
            raw_result, usage = await asyncio.wait_for(
                client.generate(
                    messages=[{"role": "user", "content": harness.build_user_prompt()}],
                    system=harness.build_system_instruction(),
                    max_tokens=settings.use_case_generation_max_tokens,
                    response_format=harness.response_format(),
                ),
                timeout=min(settings.use_case_generation_timeout_seconds, 90.0),
            )
            drafts = UseCaseRelationshipDraftList.model_validate(_decode_llm_payload(raw_result)).relations
        except TimeoutError as exc:
            raise HTTPException(
                status.HTTP_504_GATEWAY_TIMEOUT,
                detail={
                    "code": "USE_CASE_RELATION_GENERATION_TIMEOUT",
                    "message": "Relationship generation timed out; retry it.",
                },
            ) from exc
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status.HTTP_502_BAD_GATEWAY,
                detail={"code": "USE_CASE_RELATION_GENERATION_FAILED", "message": str(exc)[:500]},
            ) from exc
        self._apply_relationship_drafts(payload, drafts, source=source)
        self._refresh_relationship_ids(payload)
        model = self._payload_to_model(payload, project.name, source=source)
        payload["plantUml"] = self._plantuml_payload(model)
        self._sync_actor_sides(payload, model)
        payload["validation"] = UseCaseValidationResponse.model_validate(
            validate_use_case_model(model, source).model_dump()
        ).model_dump(by_alias=True)
        payload["generation"] = {
            "source": "ai",
            "providerConfigId": str(provider.id),
            "provider": provider.provider_type.value,
            "model": provider.model_name,
            "usage": usage,
            "relationsGenerated": True,
        }
        payload = _normalise_payload(payload, project)
        await self._save(record, payload)
        return UseCaseModelResponse.model_validate(payload)

    async def update_plant_uml(
        self, *, project_id: uuid.UUID, body: UseCasePlantUmlUpdateRequest
    ) -> UseCasePlantUmlResponse:
        source = body.source.strip()
        if not _is_plantuml_source(source):
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="PlantUML source must contain @startuml and @enduml markers",
            )
        project = await self._project(project_id)
        payload, record = await self._locked_payload(project)
        plant = {"language": "plantuml", "source": source, "editable": True, "stale": False, "generatedFrom": "manual"}
        payload["plantUml"] = plant
        payload["generation"] = {"source": "manual-uml"}
        await self._save(record, payload)
        return UseCasePlantUmlResponse.model_validate(plant)

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
        return _normalise_payload(record.model_data if record is not None else None, project)

    async def _locked_payload(self, project: Project) -> tuple[dict[str, Any], UseCaseModelRecord]:
        record = await self._record(project.id, for_update=True)
        if record is None:
            payload = _empty_payload(project)
            record = UseCaseModelRecord(project_id=project.id, model_data=payload)
            self.db.add(record)
            await self.db.flush()
            return payload, record
        return _normalise_payload(record.model_data, project), record

    async def _save(self, record: UseCaseModelRecord, payload: dict[str, Any]) -> None:
        record.model_data = payload
        record.last_generation_error = None
        await self.db.flush()

    async def _persist_generated(
        self, *, project_id: uuid.UUID, user_id: uuid.UUID, source: RequirementsSourceSnapshot, payload: dict[str, Any]
    ) -> None:
        record = await self._record(project_id, for_update=True)
        if record is None:
            record = UseCaseModelRecord(project_id=project_id, model_data=payload)
            self.db.add(record)
            await self.db.flush()
        record.model_data = payload
        record.source_hash = source.source_hash
        record.generated_by_id = user_id
        generation = payload.get("generation") if isinstance(payload.get("generation"), dict) else {}
        record.last_generation_error = generation.get("error")
        await self.db.flush()

    async def _llm_client(self, *, user_id: uuid.UUID, provider_config_id: uuid.UUID | None):
        query = select(LLMProviderConfig).where(
            LLMProviderConfig.user_id == user_id, LLMProviderConfig.status == LLMProviderStatus.ACTIVE
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
        return LLMClientFactory.create(
            provider_type=config.provider_type,
            api_key=api_key,
            secret_key=secret_key,
            model=config.model_name,
            region=config.region,
            base_url=config.base_url,
        ), config

    def _core_to_payload(
        self,
        *,
        project: Project,
        source: RequirementsSourceSnapshot,
        model: UseCaseModel,
        report,
        provider=None,
        usage=None,
        relations_generated: bool | None = None,
        generation_error: str | None = None,
    ) -> dict[str, Any]:
        system_name = project.name if model.system.name == "Requirements System" else model.system.name
        if model.system.name == "Requirements System":
            model.system = UseCaseSystem(
                id=model.system.id,
                name=system_name,
                description=model.system.description,
                source_refs=model.system.source_refs,
            )
        plant_uml = self._plantuml_payload(model)
        relationships = []
        for relation in model.relationships:
            if relation.kind == "association":
                continue
            relationships.append(
                {
                    "id": relation.id,
                    "sourceId": relation.source_id,
                    "targetId": relation.target_id,
                    "type": relation.kind,
                    "condition": relation.condition,
                    "reason": relation.reason,
                    "confidence": relation.confidence,
                    "reviewState": relation.review_state,
                    "sourceTrace": _source_trace(relation.source_refs, source),
                }
            )
        payload = {
            "projectId": str(project.id),
            "projectName": project.name,
            "system": {
                "id": model.system.id,
                "name": system_name,
                "description": model.system.description,
                "sourceTrace": _source_trace(model.system.source_refs, source),
            },
            "actors": [
                {"id": item.id, "name": item.name, "kind": item.kind, "side": item.side} for item in model.actors
            ],
            "modules": [
                {
                    "id": item.id,
                    "name": item.name,
                    "goal": item.goal,
                    "sourceTrace": _source_trace(item.source_refs, source),
                }
                for item in model.modules
            ],
            "useCases": [_entry_payload(item, source) for item in model.use_cases],
            "relationships": relationships,
            "sourceHash": source.source_hash,
            "validation": UseCaseValidationResponse.model_validate(report.model_dump()).model_dump(by_alias=True),
            "generation": {
                "source": "ai" if provider and not generation_error else "deterministic",
                "providerConfigId": str(provider.id) if provider else None,
                "provider": provider.provider_type.value if provider else None,
                "model": provider.model_name if provider else None,
                "usage": usage,
                "relationsGenerated": bool(any(item["type"] != "association" for item in relationships))
                if relations_generated is None
                else relations_generated,
                **({"error": generation_error} if generation_error else {}),
            },
            "plantUml": plant_uml,
        }
        return _normalise_payload(payload, project)

    @staticmethod
    def _plantuml_payload(model: UseCaseModel) -> dict[str, Any]:
        return {
            "language": "plantuml",
            "source": render_plantuml(model),
            "editable": True,
            "stale": False,
            "generatedFrom": "use-case-table",
        }

    @staticmethod
    def _payload_to_model(
        payload: dict[str, Any], system_name: str, *, source: RequirementsSourceSnapshot | None = None
    ) -> UseCaseModel:
        modules = payload.get("modules", [])
        system_payload = payload.get("system") if isinstance(payload.get("system"), dict) else {}
        canonical_system_name = str(system_payload.get("name") or system_name)
        actors = [
            UseCaseActor(
                id=item["id"],
                name=item["name"],
                kind=_actor_kind(item.get("kind") or item.get("type")),
                side=item.get("side"),
                source_refs=_canonical_source_refs(item.get("sourceTrace"), source),
            )
            for item in payload.get("actors", [])
        ]
        use_cases = []
        for item in payload.get("useCases", []):
            row = dict(item)
            row["source_refs"] = _canonical_source_refs(item.get("sourceTrace"), source)
            related = []
            for link in item.get("relatedRequirements", []) or []:
                link_data = dict(link) if isinstance(link, dict) else {"id": str(link), "type": "functional"}
                link_data["source_refs"] = canonical_evidence_refs(
                    link_data.get("sourceTrace") or link_data.get("source_refs"),
                    source,
                )
                related.append(link_data)
            row["relatedRequirements"] = related
            use_cases.append(UseCaseEntry.model_validate(row))
        relationships = [
            UseCaseRelation.model_validate(
                item | {"source_refs": _canonical_source_refs(item.get("sourceTrace"), source)}
            )
            for item in payload.get("relationships", [])
            if item.get("type") not in {"part-of", "association"}
        ]
        return UseCaseModel(
            system=UseCaseSystem(
                id=str(system_payload.get("id") or "SYSTEM"),
                name=canonical_system_name,
                description=system_payload.get("description"),
                source_refs=_canonical_source_refs(system_payload.get("sourceTrace"), source),
            ),
            modules=[
                UseCaseModule(
                    id=item["id"],
                    name=item["name"],
                    goal=item.get("goal"),
                    source_refs=_canonical_source_refs(item.get("sourceTrace"), source),
                )
                for item in modules
            ],
            actors=actors,
            use_cases=use_cases,
            relationships=relationships,
        )

    def _apply_relationship_drafts(
        self,
        payload: dict[str, Any],
        drafts: list[Any],
        *,
        source: RequirementsSourceSnapshot | None = None,
    ) -> None:
        use_case_ids = {item.get("id") for item in payload["useCases"]}
        # Actor associations are derived from primaryActorId/secondaryActorIds and are not stored
        # in the semantic relationship collection.
        payload["relationships"] = []
        existing: set[tuple[str, str, str]] = set()
        accepted: list[dict[str, Any]] = []
        for draft in drafts:
            if (
                draft.source_id not in use_case_ids
                or draft.target_id not in use_case_ids
                or draft.source_id == draft.target_id
            ):
                continue
            if draft.kind == "extend" and not draft.condition:
                continue
            key = (draft.kind, draft.source_id, draft.target_id)
            if key in existing:
                continue
            source_refs = canonical_evidence_refs(draft.evidence, source, fallback_internal=False)
            # A semantic relationship must be traceable to the stored BRD/PRD
            # snapshot.  Provider prose such as ``UC-... description`` is
            # resolved by the canonicalizer; unresolved citations are dropped
            # instead of becoming validation errors in the saved model.
            if not source_refs:
                continue
            accepted.append(
                {
                    "id": self._next_relationship_id(payload, draft.source_id, draft.target_id),
                    "sourceId": draft.source_id,
                    "targetId": draft.target_id,
                    "type": draft.kind,
                    "condition": draft.condition,
                    "reason": draft.reason,
                    "confidence": draft.confidence,
                    "reviewState": "review_required"
                    if draft.confidence is not None and draft.confidence < 0.8
                    else "accepted",
                    "sourceTrace": _source_trace(source_refs, source),
                }
            )
            existing.add(key)

        # ``include`` denotes reusable behavior shared by at least two base
        # use cases.  Drop singleton candidates before validation/rendering so
        # an over-eager provider cannot leave a review warning in the saved
        # model.
        include_target_counts: dict[str, int] = {}
        for relation in accepted:
            if relation["type"] == "include":
                target = relation["targetId"]
                include_target_counts[target] = include_target_counts.get(target, 0) + 1
        payload["relationships"] = [
            relation
            for relation in accepted
            if relation["type"] != "include" or include_target_counts.get(relation["targetId"], 0) >= 2
        ]

    @staticmethod
    def _draft_row(
        use_case_id: str, module_id: str, draft: Any, supporting: list[str], source: RequirementsSourceSnapshot
    ) -> dict[str, Any]:
        related_requirements = [
            link.model_copy(
                update={"source_refs": canonical_evidence_refs(link.source_refs, source)}
            )
            for link in draft.related_requirements
        ]
        entry = UseCaseEntry.model_validate(
            {
                "id": use_case_id,
                "name": draft.name,
                "module_id": module_id,
                "primary_actor_id": draft.primary_actor_id,
                "secondary_actor_ids": supporting,
                "description": draft.description,
                "trigger": draft.trigger,
                "preconditions": draft.preconditions,
                "main_flow": draft.main_flow,
                "alternative_flows": draft.alternative_flows,
                "exception_flows": draft.exception_flows,
                "postconditions_success": draft.postconditions_success,
                "postconditions_failure": draft.postconditions_failure,
                "business_rules": draft.business_rules,
                "related_requirements": related_requirements,
                "priority": draft.priority,
                "evidence": draft.evidence,
                "source_refs": canonical_evidence_refs(draft.source_refs, source, fallback_internal=True),
                "note": draft.note,
            }
        )
        return _entry_payload(entry, source)

    @staticmethod
    def _next_use_case_id(payload: dict[str, Any], module_name: str) -> str:
        existing = {str(item.get("id")) for item in payload.get("useCases", [])}
        prefix = _subsystem_code(module_name)
        index = 1
        while f"UC-{prefix}-{index:03d}" in existing:
            index += 1
        return f"UC-{prefix}-{index:03d}"

    @staticmethod
    def _next_id(prefix: str, value: str, existing: set[str]) -> str:
        base = _SLUG_RE.sub("-", value.upper()).strip("-")[:50] or prefix
        candidate = f"{prefix}-{base}"
        suffix = 2
        while candidate in existing:
            candidate = f"{prefix}-{base}-{suffix}"
            suffix += 1
        return candidate

    @staticmethod
    def _next_relationship_id(payload: dict[str, Any], source_id: str, target_id: str) -> str:
        existing = {str(item.get("id")) for item in payload.get("relationships", [])}
        base = (
            f"REL-{_SLUG_RE.sub('-', source_id.upper()).strip('-')}-{_SLUG_RE.sub('-', target_id.upper()).strip('-')}"
        )
        candidate = base
        suffix = 2
        while candidate in existing:
            candidate = f"{base}-{suffix}"
            suffix += 1
        return candidate

    @staticmethod
    def _find(items: list[dict[str, Any]], item_id: str | None) -> dict[str, Any] | None:
        return next((item for item in items if item.get("id") == item_id), None) if item_id else None

    @staticmethod
    def _mark_table_changed(payload: dict[str, Any]) -> None:
        if isinstance(payload.get("plantUml"), dict):
            payload["plantUml"]["stale"] = True
        payload["validation"] = None
        payload["generation"] = {"source": "manual"}

    @staticmethod
    def _refresh_relationship_ids(payload: dict[str, Any]) -> None:
        by_id = {item.get("id"): item for item in payload.get("useCases", [])}
        for item in by_id.values():
            item["relationshipIds"] = []
        for relation in payload.get("relationships", []):
            relation_id = relation.get("id")
            for endpoint in (relation.get("sourceId"), relation.get("targetId")):
                if endpoint in by_id and relation_id:
                    by_id[endpoint].setdefault("relationshipIds", []).append(relation_id)

    @staticmethod
    def _sync_actor_sides(payload: dict[str, Any], model: UseCaseModel) -> None:
        side_by_id = {actor.id: actor.side for actor in model.actors}
        for actor in payload.get("actors", []):
            actor["side"] = side_by_id.get(actor.get("id"))


def _empty_payload(project: Project) -> dict[str, Any]:
    return UseCaseModelResponse(
        projectId=str(project.id),
        projectName=project.name,
        system={"id": "SYSTEM", "name": project.name, "description": None, "sourceTrace": []},
    ).model_dump(by_alias=True, mode="json")


def _normalise_payload(raw: Any, project: Project) -> dict[str, Any]:
    raw = copy.deepcopy(raw) if isinstance(raw, dict) else {}
    modules_raw = raw.get("modules") if isinstance(raw.get("modules"), list) else []
    old_rows = raw.get("useCases") if isinstance(raw.get("useCases"), list) else []
    modules: list[dict[str, Any]] = []
    module_by_key: dict[str, dict[str, Any]] = {}
    for item in modules_raw:
        if not isinstance(item, dict):
            continue
        module_id = str(item.get("id") or "")
        name = str(item.get("name") or item.get("title") or "General")
        if not module_id:
            module_id = _module_id(name)
        module = {
            "id": module_id,
            "name": name,
            "goal": item.get("goal"),
            "sourceTrace": item.get("sourceTrace") or item.get("source_trace") or [],
        }
        if module_id not in module_by_key:
            modules.append(module)
            module_by_key[module_id] = module
        module_by_key.setdefault(name.casefold(), module)
    for row in old_rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("level", "")).upper() == "L0":
            name = str(row.get("subsystem") or row.get("title") or "General")
            module_id = _module_id(name)
            if module_id not in module_by_key:
                module = {
                    "id": module_id,
                    "name": name,
                    "goal": row.get("description"),
                    "sourceTrace": row.get("sourceTrace") or [],
                }
                modules.append(module)
                module_by_key[module_id] = module
            module_by_key.setdefault(name.casefold(), module_by_key[module_id])
    rows: list[dict[str, Any]] = []
    for row in old_rows:
        if not isinstance(row, dict) or str(row.get("level", "")).upper() == "L0":
            continue
        name = str(row.get("subsystem") or "General")
        module_id = row.get("moduleId") or row.get("module_id") or _module_id(name)
        module = module_by_key.get(module_id) or module_by_key.get(name.casefold())
        if module is None:
            module = {"id": module_id, "name": name, "goal": None, "sourceTrace": row.get("sourceTrace") or []}
            modules.append(module)
            module_by_key[module_id] = module
        data = dict(row)
        data.update(
            {
                "module_id": module["id"],
                "source_refs": data.get("sourceTrace") or data.get("source_refs") or ["internal"],
            }
        )
        try:
            entry = UseCaseEntry.model_validate(data)
        except Exception:
            continue
        rows.append(_entry_payload(entry, None))
    actors = []
    for item in raw.get("actors", []) if isinstance(raw.get("actors"), list) else []:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        actors.append(
            {
                "id": item["id"],
                "name": item.get("name") or item["id"],
                "kind": _actor_kind(item.get("kind") or item.get("type")),
                "side": item.get("side"),
            }
        )
    relationships = []
    for item in raw.get("relationships", []) if isinstance(raw.get("relationships"), list) else []:
        if not isinstance(item, dict) or item.get("type") in {"part-of", "association"}:
            continue
        rel = {
            "id": item.get("id") or "REL-UNKNOWN",
            "sourceId": item.get("sourceId") or item.get("source_id"),
            "targetId": item.get("targetId") or item.get("target_id"),
            "type": item.get("type") or item.get("kind"),
            "condition": item.get("condition"),
            "reason": item.get("reason"),
            "confidence": item.get("confidence"),
            "reviewState": item.get("reviewState") or item.get("review_state"),
            "sourceTrace": item.get("sourceTrace") or [],
        }
        if (
            rel["sourceId"]
            and rel["targetId"]
            and rel["type"] in {"association", "include", "extend", "generalization"}
        ):
            relationships.append(rel)
    system_raw = raw.get("system")
    if isinstance(system_raw, str):
        system_raw = {"id": "SYSTEM", "name": system_raw}
    if not isinstance(system_raw, dict):
        system_raw = {"id": "SYSTEM", "name": project.name}
    output = {
        "projectId": str(project.id),
        "projectName": project.name,
        "system": {
            "id": system_raw.get("id") or "SYSTEM",
            "name": system_raw.get("name") or project.name,
            "description": system_raw.get("description"),
            "sourceTrace": system_raw.get("sourceTrace") or system_raw.get("source_refs") or [],
        },
        "actors": actors,
        "modules": modules,
        "useCases": rows,
        "relationships": relationships,
        "sourceHash": raw.get("sourceHash"),
        "validation": raw.get("validation"),
        "generation": raw.get("generation"),
        "plantUml": raw.get("plantUml") if isinstance(raw.get("plantUml"), dict) else None,
    }
    UseCaseService._refresh_relationship_ids(output)
    try:
        return UseCaseModelResponse.model_validate(output).model_dump(by_alias=True, mode="json")
    except Exception:
        # An old/corrupt persisted draft should still leave the FE with an empty valid model; the
        # next generation replaces it from the current BRD/PRD snapshot.
        return _empty_payload(project) | {
            "actors": actors,
            "modules": modules,
            "useCases": rows,
            "relationships": relationships,
            "plantUml": output["plantUml"],
        }


def _entry_payload(item: UseCaseEntry, source: RequirementsSourceSnapshot | None) -> dict[str, Any]:
    refs = item.source_refs or ["internal"]
    source_trace = _source_trace(refs, source)
    return {
        "id": item.id,
        "name": item.name,
        "moduleId": item.module_id,
        "primaryActorId": item.primary_actor_id,
        "secondaryActorIds": item.secondary_actor_ids,
        "relationshipIds": item.relationship_ids,
        "evidence": item.evidence,
        "priority": item.priority,
        "description": item.description,
        "trigger": item.trigger,
        "preconditions": item.preconditions,
        "mainFlow": [_flow_step_payload(step) for step in item.main_flow],
        "alternativeFlows": [_flow_payload(flow) for flow in item.alternative_flows],
        "exceptionFlows": [_flow_payload(flow) for flow in item.exception_flows],
        "postconditionsSuccess": item.postconditions_success,
        "postconditionsFailure": item.postconditions_failure,
        "businessRules": item.business_rules,
        "relatedRequirements": [
            {
                "id": link.id,
                "type": link.type,
                "title": link.title,
                "sourceTrace": _source_trace(link.source_refs, source),
            }
            for link in item.related_requirements
        ],
        "sourceTrace": source_trace,
        "note": item.note,
    }


def _flow_step_payload(step: Any) -> dict[str, Any]:
    return {
        "step": step.step,
        "participantType": step.participant_type,
        "participantId": step.participant_id,
        "action": step.action,
    }


def _flow_payload(flow: Any) -> dict[str, Any]:
    return {
        "flowType": flow.flow_type,
        "label": flow.label,
        "branchAtStep": flow.branch_at_step,
        "steps": [_flow_step_payload(step) for step in flow.steps],
    }


def _source_trace(refs: list[str], source: RequirementsSourceSnapshot | None) -> list[str]:
    if source is None:
        return list(refs)
    evidence = source.evidence_by_id()
    return [
        f"{evidence[ref].document_type.upper()} · {evidence[ref].locator}" if ref in evidence else ref for ref in refs
    ]


def _canonical_source_refs(values: Any, source: RequirementsSourceSnapshot | None) -> list[str]:
    return canonical_evidence_refs(values, source, fallback_internal=True)


def _actor_kind(value: Any) -> str:
    value = str(value or "human").lower()
    return {"primary actor": "human", "supporting actor": "human", "human_role": "human", "time": "scheduler"}.get(
        value, value if value in {"human", "external_system", "scheduler"} else "human"
    )


def _module_id(name: str) -> str:
    slug = _SLUG_RE.sub("-", str(name).upper()).strip("-") or "GENERAL"
    return f"SUB-{slug[:70]}"


def _subsystem_code(name: str) -> str:
    words = re.findall(r"[A-Z0-9]+", name.upper())
    return ("".join(item[0] for item in words) if len(words) > 1 else (words[0][:6] if words else "GEN"))[:12]


def _decode_llm_payload(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    content = getattr(raw, "content", raw)
    if isinstance(content, list):
        content = "".join(str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content)
    if not isinstance(content, str):
        raise ValueError("LLM did not return JSON")
    cleaned = content.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I | re.S).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Could not parse JSON from LLM response") from None
        parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("LLM response must be a JSON object")
    return parsed


def _module_generation_context(
    source: RequirementsSourceSnapshot, module: UseCaseModule, rows: list[UseCaseEntry]
) -> tuple[str, str]:
    """Build a bounded, source-backed prompt for one module batch.

    The complete source remains available to the deterministic parser and traceability checks.
    The provider only needs the module's requirement rows, the stakeholder roster, and the nearby
    BRD rules for detail enrichment.  Keeping this excerpt bounded is what makes batching useful
    for large projects.
    """

    requirement_ids = {
        link.id
        for row in rows
        for link in row.related_requirements
        if link.id
    }
    module_words = [word for word in re.findall(r"[\wÀ-ỹ]+", module.name.casefold()) if len(word) > 2]
    blocks: list[str] = [
        f"module_id: {module.id}",
        f"module_name: {module.name}",
        f"module_goal: {module.goal or '(not specified)'}",
        "source_use_cases:",
        json.dumps([row.model_dump(mode="json", by_alias=True) for row in rows], ensure_ascii=False, indent=2),
    ]

    for component in source.components:
        lines = component.body.splitlines()
        selected: list[str] = []
        if component.artifact_type == "functional_requirement":
            selected = [
                line
                for line in lines
                if any(code.casefold() in line.casefold() for code in requirement_ids)
            ]
            if selected:
                selected.insert(0, "| ID | Requirement | Behavior | Priority |")
        elif component.artifact_type == "stakeholder_register":
            # Actor labels and responsibilities are useful to every module, but the assumptions
            # tables below the roster are not needed in every batch.
            selected = [
                line
                for line in lines
                if line.strip().startswith("|") or line.lstrip().startswith("#")
            ][:45]
        elif component.artifact_type in {"business_rules", "scope_capabilities"}:
            selected = [
                line
                for line in lines
                if any(token.casefold() in line.casefold() for token in (*requirement_ids, *module_words))
            ][:60]
        elif component.artifact_type in {"problem_statement", "vision_objectives"}:
            selected = lines[:18]
        elif component.artifact_type == "use_case":
            selected = [
                line
                for line in lines
                if any(token.casefold() in line.casefold() for token in module_words)
            ][:35]
        if selected:
            block = "\n".join(
                (
                    f"--- {component.document_type}.{component.artifact_type} ---",
                    "\n".join(selected),
                )
            )
            blocks.append(block[:9000])

    context = "\n\n".join(blocks)
    if len(context) > 30000:
        context = context[:29950].rstrip() + "\n[context bounded by backend]"

    refs = set(module.source_refs)
    for row in rows:
        refs.update(row.source_refs)
        for link in row.related_requirements:
            refs.update(link.source_refs)
    evidence_lines = ["evidence_id | kind | locator | excerpt", "---|---|---|---"]
    for item in source.evidence:
        if item.evidence_id not in refs and item.kind not in {"actor", "business_rule", "scope"}:
            continue
        excerpt = " ".join(item.excerpt.split()).replace("|", "\\|")[:180]
        evidence_lines.append(f"{item.evidence_id} | {item.kind} | {item.locator} | {excerpt}")
        if len("\n".join(evidence_lines)) > 12000:
            break
    return context, "\n".join(evidence_lines)


def _relationship_generation_context(source: RequirementsSourceSnapshot) -> str:
    """Return a bounded BRD/PRD excerpt for relationship resolution."""

    blocks: list[str] = []
    for component in source.components:
        if component.artifact_type not in {
            "functional_requirement",
            "business_rules",
            "scope_capabilities",
            "stakeholder_register",
        }:
            continue
        lines = component.body.splitlines()
        if component.artifact_type == "stakeholder_register":
            lines = [line for line in lines if line.strip().startswith("|") or line.lstrip().startswith("#")][:45]
        else:
            lines = lines[:120]
        blocks.append(f"--- {component.document_type}.{component.artifact_type} ---\n" + "\n".join(lines))
    text = "\n\n".join(blocks)
    return text[:28000].rstrip() + ("\n[context bounded by backend]" if len(text) > 28000 else "")


def _bounded_evidence_index(source: RequirementsSourceSnapshot, limit: int) -> str:
    """Render a bounded evidence index without duplicating every component body."""

    lines = ["evidence_id | document | component | kind | locator | excerpt", "---|---|---|---|---|---"]
    for item in source.evidence:
        excerpt = " ".join(item.excerpt.split()).replace("|", "\\|")[:180]
        lines.append(
            f"{item.evidence_id} | {item.document_type} | {item.artifact_type} | "
            f"{item.kind} | {item.locator} | {excerpt}"
        )
        if len("\n".join(lines)) >= limit:
            break
    return "\n".join(lines)


def _candidate_rows_from_drafts(
    floor: UseCaseModel, module: UseCaseModule, drafts: list[Any]
) -> list[UseCaseEntry]:
    """Map a module batch back onto the deterministic source-backed use-case IDs."""

    rows = [item for item in floor.use_cases if item.module_id == module.id]
    by_id = {item.id: item for item in rows}
    by_name = {_model_name_key(item.name): item for item in rows}
    actor_ids = {item.id for item in floor.actors}
    output: list[UseCaseEntry] = []
    seen: set[str] = set()
    for draft in drafts:
        target = None
        if draft.local_tag:
            target = by_id.get(draft.local_tag)
        if target is None:
            target = by_name.get(_model_name_key(draft.name))
        if target is None or target.id in seen:
            continue
        data = target.model_dump()
        for field_name in (
            "name",
            "description",
            "trigger",
            "preconditions",
            "main_flow",
            "alternative_flows",
            "exception_flows",
            "postconditions_success",
            "postconditions_failure",
            "business_rules",
            "related_requirements",
            "priority",
            "evidence",
            "note",
        ):
            value = getattr(draft, field_name)
            if value not in (None, [], ""):
                data[field_name] = value
        if draft.primary_actor_id in actor_ids:
            data["primary_actor_id"] = draft.primary_actor_id
        data["secondary_actor_ids"] = [item for item in draft.secondary_actor_ids if item in actor_ids]
        data["source_refs"] = draft.source_refs or target.source_refs
        try:
            output.append(UseCaseEntry.model_validate(data))
            seen.add(target.id)
        except Exception:
            # A malformed detail draft must not discard the source-backed row or turn generation
            # into a 500.  The deterministic row remains in the final model.
            continue
    return output


def _model_name_key(value: str) -> str:
    return re.sub(r"[^\wÀ-ỹ]+", " ", value.casefold()).strip()


def _is_plantuml_source(source: str) -> bool:
    return bool(re.search(r"(?im)^\s*@startuml(?:\s|$)", source) and re.search(r"(?im)^\s*@enduml(?:\s|$)", source))
