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
    UseCaseRelationshipResponse,
    UseCaseResponse,
    UseCaseUpdateRequest,
    UseCaseValidationResponse,
)
from app.services.llm_clients import LLMClientFactory
from app.use_cases.diagram import build_diagram_render_plan
from app.use_cases.models import RequirementsSourceSnapshot, UseCaseModel
from app.use_cases.source_loader import load_project_requirements_source

_LEVEL_ORDER = {"L0": 0, "L1": 1, "L2": 2}
_SLUG_RE = re.compile(r"[^A-Z0-9]+")


class UseCaseService:
    """Own the aggregate used by the use-case table and diagram endpoints."""

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

        harness = UseCaseGenerationHarness(source)
        client, provider = await self._llm_client(user_id=user_id, provider_config_id=body.provider_config_id)
        try:
            raw_result, usage = await asyncio.wait_for(
                client.generate(
                    messages=[{"role": "user", "content": harness.build_user_prompt()}],
                    system=harness.build_system_instruction(),
                    max_tokens=settings.analyze_max_tokens,
                    response_format=harness.response_format(),
                ),
                timeout=settings.use_case_generation_timeout_seconds,
            )
            payload = _decode_llm_payload(raw_result)
            model, report = harness.parse_and_validate(payload)
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

    def _response(
        self,
        payload: dict[str, Any],
        *,
        max_level: UseCaseLevel = UseCaseLevel.L2,
        include_actors: bool = True,
        include_relationships: bool = True,
    ) -> UseCaseModelResponse:
        data = copy.deepcopy(payload)
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
        provider: LLMProviderConfig,
        usage: dict[str, int] | None,
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
        diagrams = [
            {
                "id": item.id,
                "level": item.level,
                "systemBoundary": item.system_boundary,
                "subsystem": subsystem_names.get(item.subsystem_id) if item.subsystem_id else None,
                "actorIds": item.actor_ids,
                "useCaseIds": item.use_case_ids,
                "relationIds": item.relation_ids,
            }
            for item in model.diagrams
        ]
        diagram_plans: list[dict[str, Any]] = []
        if not report.errors:
            for item in model.diagrams:
                try:
                    plan = build_diagram_render_plan(model, item.id, require_confirmed=False)
                except ValueError:
                    continue
                diagram_plans.append(plan.model_dump(by_alias=True))
        validation = UseCaseValidationResponse.model_validate(report.model_dump())
        payload = UseCaseModelResponse(
            projectId=str(project.id),
            projectName=project.name,
            actors=actors,
            useCases=use_cases,
            relationships=relationships,
            diagrams=diagrams,
            diagramPlans=diagram_plans,
            sourceHash=source.source_hash,
            validation=validation,
            generation={
                "source": "ai",
                "providerConfigId": str(provider.id),
                "provider": provider.provider_type.value,
                "model": provider.model_name,
                "usage": usage,
            },
        )
        return payload.model_dump(by_alias=True, mode="json")


def _empty_payload(project: Project) -> dict[str, Any]:
    return UseCaseModelResponse(projectId=str(project.id), projectName=project.name).model_dump(
        by_alias=True, mode="json"
    )


def _normalise_payload(raw: Any, project: Project) -> dict[str, Any]:
    payload = copy.deepcopy(raw) if isinstance(raw, dict) else {}
    payload.setdefault("projectId", str(project.id))
    payload["projectName"] = project.name
    for key in ("actors", "useCases", "relationships", "diagrams", "diagramPlans"):
        if not isinstance(payload.get(key), list):
            payload[key] = []
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
