"""Persistence and generation service for the BRD/PRD-backed use-case model."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import math
import re
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException, status
from pydantic import ValidationError
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
    DiagramNodePositionRequest,
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
    UseCaseCandidatesHarness,
    UseCaseDetailHarness,
    UseCaseGroupsHarness,
    UseCaseGroupUseCasesHarness,
    UseCaseRelationshipHarness,
)
from app.use_cases.layout import DiagramLayoutError, generate_diagram_layout
from app.use_cases.models import (
    RequirementsSourceSnapshot,
    UseCaseActor,
    UseCaseCandidateDraftList,
    UseCaseDetailDraftList,
    UseCaseEntry,
    UseCaseGroupDetailDraftList,
    UseCaseGroupsDraftList,
    UseCaseModel,
    UseCaseModule,
    UseCaseRelation,
    UseCaseRelationshipDraftList,
    UseCaseSystem,
)
from app.use_cases.rules import validate_use_case_model
from app.use_cases.source_loader import canonical_evidence_refs, load_project_requirements_source

logger = logging.getLogger(__name__)

_SLUG_RE = re.compile(r"[^A-Z0-9]+")
# Split generation runs several requests (module extraction, then one per module, then
# relationships) that each refresh a heartbeat on the persisted "running" marker. A marker
# untouched for longer than this is treated as abandoned (a crashed/killed request, a dropped
# connection) rather than a real in-progress run, so a new generation is allowed to start
# instead of being stuck behind a 409 forever. It is well above any single phase's own
# provider-call timeout so a slow-but-healthy run is never mistaken for a dead one.
_GENERATION_STALE_AFTER_SECONDS = 600.0


def _generation_heartbeat_is_stale(generation: dict[str, Any]) -> bool:
    marker = generation.get("updatedAt") or generation.get("startedAt")
    if not isinstance(marker, str) or not marker:
        return True
    try:
        touched = datetime.fromisoformat(marker)
    except ValueError:
        return True
    if touched.tzinfo is None:
        touched = touched.replace(tzinfo=UTC)
    return (datetime.now(UTC) - touched).total_seconds() > _GENERATION_STALE_AFTER_SECONDS


# Concept-stage cap: keep the generated table small enough to read and diagram at a glance
# instead of exhaustively enumerating every requirement family.
_MAX_GENERATED_USE_CASES = 20
_PRIORITY_SCORE = {"required": 30, "recommended": 20, "optional": 10}


def _candidate_limit(payload: dict[str, Any]) -> int:
    """Per-module shortlist size: about twice the final budget across all modules, so selection
    has real choices without any module spending output tokens on a long list."""

    modules = len(payload.get("modules") or []) or 1
    return max(6, min(15, math.ceil(2 * _MAX_GENERATED_USE_CASES / modules)))


def _select_candidates(
    payload: dict[str, Any], candidates_by_module: dict[str, list[dict[str, Any]]], source: RequirementsSourceSnapshot
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Pick at most _MAX_GENERATED_USE_CASES candidates, deterministically.

    Hard filters: at least one citation that resolves to a real evidence line in the stored
    BRD/PRD, and a known actor. Then, by score (priority, explicit over inferred, citation count):
    first one per actor, then one per module, then the best of the rest. Ties go to the
    candidate its module ranked higher, so equal scores spread across modules.
    """

    evidence_ids = set(source.evidence_by_id())
    actor_ids = {item["id"] for item in payload.get("actors", [])}
    module_order = {item["id"]: index for index, item in enumerate(payload.get("modules", []))}
    stats = {"proposed": 0, "uncited": 0, "unknownActor": 0, "duplicate": 0, "selected": 0}
    by_name: dict[str, dict[str, Any]] = {}
    for module in payload.get("modules", []):
        for rank, item in enumerate(candidates_by_module.get(module["id"]) or []):
            stats["proposed"] += 1
            refs = [ref for ref in canonical_evidence_refs(item.get("sourceRefs"), source) if ref in evidence_ids]
            if not refs:
                stats["uncited"] += 1
                continue
            if item.get("primaryActorId") not in actor_ids:
                stats["unknownActor"] += 1
                continue
            score = (
                _PRIORITY_SCORE.get(item.get("priority"), 0)
                + (5 if item.get("evidence") == "explicit" else 0)
                + min(len(refs), 3)
            )
            candidate = {**item, "moduleId": module["id"], "sourceRefs": refs, "score": score, "rank": rank}
            key = _model_name_key(item.get("name") or "")
            if key in by_name:
                stats["duplicate"] += 1
                if by_name[key]["score"] >= score:
                    continue
            by_name[key] = candidate

    pool = sorted(by_name.values(), key=lambda item: (-item["score"], item["rank"], module_order[item["moduleId"]]))
    chosen: list[dict[str, Any]] = []
    taken: set[int] = set()

    def fill(unique_by: str | None) -> None:
        covered = {item[unique_by] for item in chosen} if unique_by else set()
        for index, candidate in enumerate(pool):
            if len(chosen) >= _MAX_GENERATED_USE_CASES:
                return
            if index in taken or (unique_by and candidate[unique_by] in covered):
                continue
            if unique_by:
                covered.add(candidate[unique_by])
            chosen.append(candidate)
            taken.add(index)

    fill("primaryActorId")
    fill("moduleId")
    fill(None)
    chosen.sort(key=lambda item: (module_order[item["moduleId"]], -item["score"], item["rank"]))
    stats["selected"] = len(chosen)
    return chosen, stats


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
            detail="Use cases are generated from BRD/PRD; regenerate the model after source changes",
        )

    async def update_use_case(
        self, *, project_id: uuid.UUID, use_case_id: str, body: UseCaseUpdateRequest
    ) -> UseCaseResponse:
        _ = (project_id, use_case_id, body)
        raise HTTPException(
            status.HTTP_405_METHOD_NOT_ALLOWED,
            detail="Use cases are generated from BRD/PRD; regenerate the model after source changes",
        )

    async def delete_use_case(self, *, project_id: uuid.UUID, use_case_id: str) -> dict[str, Any]:
        _ = (project_id, use_case_id)
        raise HTTPException(
            status.HTTP_405_METHOD_NOT_ALLOWED,
            detail="Use cases are generated from BRD/PRD; regenerate the model after source changes",
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
        client = None
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
        generation_id, generation_started_at = await self._mark_generation_started(
            project=project, source=source, user_id=user_id
        )
        try:
            client, provider = await self._llm_client(user_id=user_id, provider_config_id=body.provider_config_id)
            batches = [
                module for module in floor.modules if any(item.module_id == module.id for item in floor.use_cases)
            ]
            total_batches = len(batches)

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
                    raw_result, batch_usage = await asyncio.wait_for(
                        client.generate(
                            messages=[{"role": "user", "content": harness.build_user_prompt()}],
                            system=harness.build_system_instruction(),
                            # A module has a small bounded output; reserving the global token budget
                            # here makes each batch cheaper and less likely to hit a provider limit.
                            max_tokens=min(settings.use_case_generation_max_tokens, 8000),
                            response_format=harness.response_format(),
                        ),
                        timeout=settings.use_case_generation_batch_timeout_seconds,
                    )
                    drafts = UseCaseGroupDetailDraftList.model_validate(_decode_llm_payload(raw_result)).use_cases
                    return module, drafts, batch_usage, None
                except TimeoutError:
                    return module, [], None, f"{module.name}: generation timed out"
                except Exception as exc:
                    return module, [], None, f"{module.name}: {str(exc)[:240]}"

            usages: list[Any] = []
            # Bedrock Claude requests are deliberately serialized.  Sending several 15k-token
            # structured-output calls at once causes provider queuing and turns otherwise valid
            # batches into simultaneous deadline failures.  Persist the counter after every
            # module so a reload can show real progress while the request is still running.
            for module in batches:
                module, drafts, batch_usage, error = await generate_module(module)
                if error:
                    generation_errors.append(error)
                else:
                    completed_batches += 1
                    if batch_usage is not None:
                        usages.append(batch_usage)
                    candidate_rows.extend(_candidate_rows_from_drafts(floor, module, drafts))
                await self._update_generation_batch_progress(
                    project=project,
                    generation_id=generation_id,
                    batch_count=total_batches,
                    completed_batch_count=completed_batches,
                )
            usage = {"batches": usages} if usages else None
        except HTTPException as exc:
            await self._mark_generation_failed(
                project=project,
                generation_id=generation_id,
                message=_http_exception_message(exc),
            )
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
        payload["generation"] = payload.get("generation") or {}
        payload["generation"].update(
            {
                "generationId": generation_id,
                "startedAt": generation_started_at,
                "batchCount": total_batches,
                "completedBatchCount": completed_batches,
                "generationMode": "module-batch",
                "stages": {
                    "source": "completed",
                    "table": "completed",
                    "relationships": "pending",
                    "validation": "pending",
                    "layout": "pending",
                    "persist": "pending",
                },
            }
        )
        if client is not None and model.use_cases:
            payload["generation"]["stages"]["relationships"] = "running"
        else:
            payload["generation"]["stages"]["relationships"] = "skipped"
        await self._persist_generation_progress(project=project, generation_id=generation_id, payload=payload)
        if client is not None and model.use_cases:
            try:
                relation_usage = await self._resolve_relationships_with_client(
                    payload=payload,
                    source=source,
                    client=client,
                )
                model = self._payload_to_model(payload, project.name, source=source)
                report = validate_use_case_model(model, source)
                payload["validation"] = UseCaseValidationResponse.model_validate(
                    report.model_dump()
                ).model_dump(by_alias=True)
                payload["generation"]["relationsGenerated"] = True
                payload["generation"]["relationshipUsage"] = relation_usage
                payload["generation"]["stages"]["relationships"] = "completed"
            except TimeoutError:
                generation_errors.append("relationship generation timed out")
                payload["generation"]["stages"]["relationships"] = "failed"
            except Exception as exc:
                generation_errors.append(f"relationship generation failed: {str(exc)[:240]}")
                payload["generation"]["stages"]["relationships"] = "failed"
        payload["generation"]["stages"]["validation"] = "completed"
        payload["generation"]["stages"]["layout"] = "running"
        await self._persist_generation_progress(project=project, generation_id=generation_id, payload=payload)

        layout_error = await self._attach_diagram_layout(payload)
        if layout_error:
            generation_errors.append(layout_error)
            payload["generation"]["stages"]["layout"] = "failed"
        else:
            payload["generation"]["stages"]["layout"] = "completed"

        if total_batches and completed_batches < total_batches:
            prefix = f"AI details completed for {completed_batches}/{total_batches} modules; "
            generation_errors.insert(0, prefix.rstrip("; "))
        if generation_errors:
            payload["generation"]["error"] = "; ".join(dict.fromkeys(generation_errors))
        else:
            payload["generation"].pop("error", None)
        payload["generation"].update(
            {
                "generationId": generation_id,
                "status": "completed_with_errors" if generation_errors else "completed",
                "completedAt": _generation_timestamp(),
            }
        )
        payload["generation"]["stages"]["persist"] = "completed"
        await self._persist_generated(project_id=project_id, user_id=user_id, source=source, payload=payload)
        return UseCaseModelResponse.model_validate(payload)

    async def _resolve_relationships_with_client(
        self,
        *,
        payload: dict[str, Any],
        source: RequirementsSourceSnapshot,
        client: Any,
    ) -> Any:
        harness = UseCaseRelationshipHarness(
            source=source,
            use_cases=payload.get("useCases", []),
            actors=payload.get("actors", []),
            source_text=_relationship_generation_context(source),
            evidence_text=_bounded_evidence_index(source, 14000),
        )
        raw_result, usage = await asyncio.wait_for(
            client.generate(
                messages=[{"role": "user", "content": harness.build_user_prompt()}],
                system=harness.build_system_instruction(),
                max_tokens=settings.use_case_relations_max_tokens,
                response_format=harness.response_format(),
            ),
            timeout=settings.use_case_generation_batch_timeout_seconds,
        )
        drafts = UseCaseRelationshipDraftList.model_validate(_decode_llm_payload(raw_result)).relations
        self._apply_relationship_drafts(payload, drafts, source=source)
        self._refresh_relationship_ids(payload)
        # For the monolithic /generate this is the last step, so the full use-case set is known
        # and unused actors can be dropped deterministically. (The split pipeline already does
        # this in select_use_cases and only merges relationships back from this payload.)
        self._drop_actors_without_use_cases(payload)
        return usage

    @staticmethod
    def _drop_actors_without_use_cases(payload: dict[str, Any]) -> None:
        used_actor_ids: set[str] = set()
        for item in payload.get("useCases", []):
            if item.get("primaryActorId"):
                used_actor_ids.add(item["primaryActorId"])
            used_actor_ids.update(item.get("secondaryActorIds") or [])
        payload["actors"] = [actor for actor in payload.get("actors", []) if actor.get("id") in used_actor_ids]

    async def _attach_diagram_layout(self, payload: dict[str, Any]) -> str | None:
        try:
            # Carry forward any actor/use-case position the user dragged and explicitly saved
            # (see update_diagram_positions) into this recompute, so a later table/diagram
            # regenerate does not wipe it out -- only an id with no prior manual node (new,
            # never-positioned-by-hand) gets a fresh auto-computed position. The layout worker
            # itself decides how to reconcile a manual position with everything else (see
            # diagram_layout/layout.mjs); this just needs to still exist as an id in the current
            # model, or it is silently dropped, matching normal idempotent-replace behaviour.
            previous_nodes = (payload.get("diagramLayout") or {}).get("nodes") or []
            manual_positions = {
                node["id"]: {"x": node["x"], "y": node["y"]}
                for node in previous_nodes
                if node.get("manual")
                and node.get("id")
                and isinstance(node.get("x"), int | float)
                and isinstance(node.get("y"), int | float)
            }

            def _with_manual_position(item: dict[str, Any]) -> dict[str, Any]:
                position = manual_positions.get(item.get("id"))
                return {**item, "manualPosition": position} if position else item

            actors_input = [_with_manual_position(actor) for actor in payload.get("actors", [])]
            use_cases_input = [_with_manual_position(item) for item in payload.get("useCases", [])]
            layout = await generate_diagram_layout(
                {
                    "systemName": (payload.get("system") or {}).get("name") or payload.get("projectName"),
                    "modules": payload.get("modules", []),
                    "actors": actors_input,
                    "useCases": use_cases_input,
                    "relationships": payload.get("relationships", []),
                }
            )
            payload["diagramLayout"] = layout
            side_by_id = {
                item.get("id"): item.get("side")
                for item in layout.get("nodes", [])
                if item.get("kind") == "actor" and item.get("side") in {"left", "right"}
            }
            for actor in payload.get("actors", []):
                if actor.get("id") in side_by_id:
                    actor["side"] = side_by_id[actor["id"]]
            return None
        except DiagramLayoutError as exc:
            payload["diagramLayout"] = None
            return f"diagram layout failed: {str(exc)[:240]}"
        except Exception as exc:
            # Layout is a derived presentation artifact.  A worker/runtime problem must leave
            # the source-backed table available instead of turning the whole generation request
            # into an unrelated 500. The type name is kept because some errors have no message.
            logger.exception("Diagram layout failed")
            payload["diagramLayout"] = None
            return f"diagram layout failed: {type(exc).__name__}: {str(exc)[:240]}"

    async def generate_groups(
        self, *, project_id: uuid.UUID, user_id: uuid.UUID, body: UseCaseGenerateRequest
    ) -> UseCaseModelResponse:
        project = await self._project(project_id)
        source = await load_project_requirements_source(self.db, project_id=project_id)
        # This is the entry point of a split-generation run: mark it running (and reject a
        # second overlapping attempt with a clean 409) before doing anything else, mirroring
        # the monolithic /generate route. The later pipeline steps refresh
        # this same marker rather than re-checking it, since they only ever run as this run's
        # own later phases.
        generation_id, _ = await self._mark_generation_started(project=project, source=source, user_id=user_id)
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
            # The overall run is not done -- the later pipeline steps still
            # have to run -- so this stays "running", not the terminal status _core_to_payload
            # would otherwise leave unset.
            self._touch_generation(payload, generation_id, status="running")
            layout_error = await self._attach_diagram_layout(payload)
            if layout_error:
                payload["generation"]["error"] = layout_error
        except TimeoutError as exc:
            await self._mark_generation_failed(
                project=project, generation_id=generation_id, message="Module extraction timed out."
            )
            raise HTTPException(
                status.HTTP_504_GATEWAY_TIMEOUT,
                detail={
                    "code": "USE_CASE_GROUPS_GENERATION_TIMEOUT",
                    "message": "Module extraction timed out; retry the smaller group generation flow.",
                },
            ) from exc
        except HTTPException as exc:
            await self._mark_generation_failed(
                project=project, generation_id=generation_id, message=_http_exception_message(exc)
            )
            raise
        except Exception as exc:
            logger.exception("Module extraction failed for project %s", project_id)
            await self._mark_generation_failed(
                project=project,
                generation_id=generation_id,
                message=f"Module extraction failed: {str(exc)[:400]}",
            )
            raise HTTPException(
                status.HTTP_502_BAD_GATEWAY,
                detail={"code": "USE_CASE_GROUPS_GENERATION_FAILED", "message": str(exc)[:500]},
            ) from exc

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

    # ------------------------------------------------------------------------------------------
    # Split pipeline, after generate_groups (step 1):
    #   2. generate_module_candidates  -- one call per module, in parallel: names + citations only
    #   3. select_use_cases            -- deterministic, no LLM: keeps the best cited candidates
    #   4. generate_use_case_details   -- small batches, in parallel: flows for selected rows only
    #      generate_relations          -- alongside step 4; needs only names/descriptions
    #   5. finalize_generation         -- deterministic terminal step: validation + final status
    # Each LLM step reads without a row lock, calls the provider, then locks only to merge its own
    # slice back in -- so parallel requests do not queue behind each other's provider calls.
    # ------------------------------------------------------------------------------------------

    async def generate_module_candidates(
        self, *, project_id: uuid.UUID, user_id: uuid.UUID, group_id: str, body: UseCaseGenerateRequest
    ) -> UseCaseModelResponse:
        project = await self._project(project_id)
        payload = await self._payload(project)
        generation_id = self._running_generation_id(payload)
        module = self._find(payload.get("modules", []), group_id)
        if module is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Use-case module not found")
        source = await load_project_requirements_source(self.db, project_id=project_id)
        limit = _candidate_limit(payload)
        try:
            client, _provider = await self._llm_client(user_id=user_id, provider_config_id=body.provider_config_id)
            await self._release_read_transaction()
            # Bound the prompt to what generate_groups attributed to this module; a module whose
            # refs resolve to no code falls back to the full source rather than an empty one.
            match_codes = _match_codes_from_refs(source, module.get("sourceRefs") or [])
            bounded_context: tuple[str, str] | None = None
            if match_codes:
                module_obj = UseCaseModule.model_validate(
                    {
                        "id": module["id"],
                        "name": module["name"],
                        "goal": module.get("goal"),
                        "source_refs": module.get("sourceRefs") or [],
                    }
                )
                bounded_context = _module_generation_context(source, module_obj, [], extra_match_codes=match_codes)
            harness = UseCaseCandidatesHarness(
                source=source,
                group=module,
                actors=[{"id": item["id"], "name": item["name"]} for item in payload["actors"]],
                limit=limit,
                source_text=bounded_context[0] if bounded_context else None,
                evidence_text=bounded_context[1] if bounded_context else None,
            )
            raw_result, _usage = await asyncio.wait_for(
                client.generate(
                    messages=[{"role": "user", "content": harness.build_user_prompt()}],
                    system=harness.build_system_instruction(),
                    max_tokens=settings.use_case_candidates_max_tokens,
                    response_format=harness.response_format(),
                ),
                timeout=settings.use_case_generation_batch_timeout_seconds,
            )
            drafts = UseCaseCandidateDraftList.model_validate(_decode_llm_payload(raw_result)).candidates
        except TimeoutError as exc:
            await self._mark_generation_failed(
                project=project, generation_id=generation_id, message="Use-case candidate generation timed out."
            )
            raise HTTPException(
                status.HTTP_504_GATEWAY_TIMEOUT,
                detail={
                    "code": "USE_CASE_CANDIDATES_TIMEOUT",
                    "message": "Proposing use cases for this module timed out; regenerate the table.",
                },
            ) from exc
        except HTTPException as exc:
            await self._mark_generation_failed(
                project=project, generation_id=generation_id, message=_http_exception_message(exc)
            )
            raise
        except Exception as exc:
            logger.exception("Use-case candidate generation failed for %s/%s", project_id, group_id)
            await self._mark_generation_failed(
                project=project,
                generation_id=generation_id,
                message=f"Use-case candidate generation failed: {str(exc)[:400]}",
            )
            raise HTTPException(
                status.HTTP_502_BAD_GATEWAY,
                detail={"code": "USE_CASE_CANDIDATES_FAILED", "message": str(exc)[:500]},
            ) from exc

        candidates = [
            {
                "name": draft.name.strip(),
                "primaryActorId": draft.primary_actor_id,
                "description": draft.description.strip(),
                "priority": draft.priority,
                "evidence": draft.evidence,
                "sourceRefs": list(draft.source_refs),
            }
            for draft in drafts[:limit]
        ]

        def record_candidates(fresh: dict[str, Any]) -> None:
            fresh["generation"].setdefault("candidates", {})[module["id"]] = candidates

        payload = await self._merge_into_run(project, generation_id, record_candidates)
        return UseCaseModelResponse.model_validate(payload)

    async def select_use_cases(self, *, project_id: uuid.UUID) -> UseCaseModelResponse:
        """Deterministic step: turn every module's candidates into the final (at most 20) rows.

        This is where "only from the BRD/PRD" is enforced rather than requested: a candidate whose
        citations do not resolve to a real evidence line in the stored source is dropped here, no
        matter how the provider phrased it. Rows start with no flows (detailStatus "pending");
        generate_use_case_details fills them in.
        """

        project = await self._project(project_id)
        payload, record = await self._locked_payload(project)
        generation_id = self._running_generation_id(payload)
        source = await load_project_requirements_source(self.db, project_id=project_id)
        generation = payload["generation"]
        selected, stats = _select_candidates(payload, generation.get("candidates") or {}, source)

        modules_by_id = {item["id"]: item for item in payload["modules"]}
        payload["useCases"] = []
        payload["relationships"] = []
        for candidate in selected:
            module = modules_by_id[candidate["moduleId"]]
            entry = UseCaseEntry.model_validate(
                {
                    "id": self._next_use_case_id(payload, module["name"]),
                    "name": candidate["name"],
                    "module_id": module["id"],
                    "primary_actor_id": candidate["primaryActorId"],
                    "description": candidate["description"],
                    "priority": candidate["priority"],
                    "evidence": candidate["evidence"],
                    "source_refs": candidate["sourceRefs"],
                }
            )
            payload["useCases"].append(_entry_payload(entry, source))
        # The final use-case set is known from here on, so an actor nobody initiates anything as
        # can be dropped now instead of waiting for a later step.
        self._drop_actors_without_use_cases(payload)

        generation.pop("candidates", None)
        generation.pop("relationsError", None)
        generation["selection"] = stats
        generation["detailStatus"] = {item["id"]: "pending" for item in payload["useCases"]}
        generation["relationsGenerated"] = False
        payload["diagramLayout"] = None
        self._touch_generation(payload, generation_id, status="running")
        payload = _normalise_payload(payload, project)
        await self._save(record, payload)
        return UseCaseModelResponse.model_validate(payload)

    async def generate_use_case_details(
        self,
        *,
        project_id: uuid.UUID,
        user_id: uuid.UUID,
        use_case_ids: list[str],
        body: UseCaseGenerateRequest,
    ) -> UseCaseModelResponse:
        """Write flows for a few already-selected rows. Also the per-row "retry" after a run.

        A failure only marks these rows' detailStatus "failed" -- it never fails the run, since
        every other row is independent of them and already (or still being) written.
        """

        project = await self._project(project_id)
        payload = await self._payload(project)
        rows = [self._find(payload["useCases"], use_case_id) for use_case_id in use_case_ids]
        if any(row is None for row in rows):
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Use case not found")
        # A row is identified by id + name when merging back: a table regenerated meanwhile can
        # reuse the same id for a different use case, which must not receive this detail.
        expected_names = {row["id"]: row["name"] for row in rows}
        source = await load_project_requirements_source(self.db, project_id=project_id)
        try:
            client, _provider = await self._llm_client(user_id=user_id, provider_config_id=body.provider_config_id)
            await self._release_read_transaction()
            refs = {ref for row in rows for ref in _canonical_source_refs(row.get("sourceTrace"), source)}
            source_text, evidence_text = _use_case_detail_context(source, refs)
            actor_names = {item["id"]: item["name"] for item in payload["actors"]}
            harness = UseCaseDetailHarness(
                source=source,
                use_cases=[
                    {
                        "use_case_id": row["id"],
                        "name": row["name"],
                        "primary_actor_id": row["primaryActorId"],
                        "primary_actor": actor_names.get(row["primaryActorId"]),
                        "description": row["description"],
                        "priority": row["priority"],
                        "evidence": row["evidence"],
                    }
                    for row in rows
                ],
                actors=[{"id": item["id"], "name": item["name"]} for item in payload["actors"]],
                source_text=source_text,
                evidence_text=evidence_text,
            )
            raw_result, _usage = await asyncio.wait_for(
                client.generate(
                    messages=[{"role": "user", "content": harness.build_user_prompt()}],
                    system=harness.build_system_instruction(),
                    max_tokens=settings.use_case_detail_max_tokens_per_use_case * len(rows),
                    response_format=harness.response_format(),
                ),
                timeout=settings.use_case_generation_batch_timeout_seconds,
            )
            drafts = UseCaseDetailDraftList.model_validate(_decode_llm_payload(raw_result)).details
        except TimeoutError as exc:
            await self._mark_details_failed(project, expected_names)
            raise HTTPException(
                status.HTTP_504_GATEWAY_TIMEOUT,
                detail={"code": "USE_CASE_DETAIL_TIMEOUT", "message": "Writing the use-case detail timed out."},
            ) from exc
        except HTTPException:
            await self._mark_details_failed(project, expected_names)
            raise
        except Exception as exc:
            logger.exception("Use-case detail generation failed for %s/%s", project_id, use_case_ids)
            await self._mark_details_failed(project, expected_names)
            raise HTTPException(
                status.HTTP_502_BAD_GATEWAY,
                detail={"code": "USE_CASE_DETAIL_FAILED", "message": str(exc)[:500]},
            ) from exc

        drafts_by_id = {draft.use_case_id: draft for draft in drafts}
        fresh, record = await self._locked_payload(project)
        generation = fresh["generation"] if isinstance(fresh.get("generation"), dict) else {}
        statuses = generation.setdefault("detailStatus", {})
        for row in fresh["useCases"]:
            if expected_names.get(row["id"]) != row["name"]:
                continue
            draft = drafts_by_id.get(row["id"])
            applied = draft is not None and self._apply_detail_draft(row, draft, source)
            statuses[row["id"]] = "completed" if applied else "failed"
        fresh["generation"] = generation
        if generation.get("status") == "running":
            self._touch_generation(fresh, generation.get("generationId"), status="running")
        else:
            self._clear_resolved_errors(generation)
        fresh = _normalise_payload(fresh, project)
        await self._save(record, fresh)
        return UseCaseModelResponse.model_validate(fresh)

    async def generate_relations(
        self, *, project_id: uuid.UUID, user_id: uuid.UUID, body: UseCaseGenerateRequest
    ) -> UseCaseModelResponse:
        """Resolve include/extend/generalization for the current rows.

        Runs alongside the detail batches (it only needs names/actors/descriptions, which exist
        from selection on). Not the run's terminal step -- finalize_generation is -- so a failure
        is recorded on the run as relationsError instead of failing it.
        """

        project = await self._project(project_id)
        snapshot = await self._payload(project)
        if not snapshot["useCases"]:
            return UseCaseModelResponse.model_validate(snapshot)
        source = await load_project_requirements_source(self.db, project_id=project_id)
        try:
            client, _provider = await self._llm_client(user_id=user_id, provider_config_id=body.provider_config_id)
            await self._release_read_transaction()
            await self._resolve_relationships_with_client(payload=snapshot, source=source, client=client)
        except TimeoutError as exc:
            await self._record_relations_error(project, "Relationship generation timed out.")
            raise HTTPException(
                status.HTTP_504_GATEWAY_TIMEOUT,
                detail={
                    "code": "USE_CASE_RELATION_GENERATION_TIMEOUT",
                    "message": "Relationship generation timed out; retry it.",
                },
            ) from exc
        except HTTPException as exc:
            await self._record_relations_error(project, _http_exception_message(exc))
            raise
        except Exception as exc:
            logger.exception("Relationship generation failed for project %s", project_id)
            await self._record_relations_error(project, f"Relationship generation failed: {str(exc)[:400]}")
            raise HTTPException(
                status.HTTP_502_BAD_GATEWAY,
                detail={"code": "USE_CASE_RELATION_GENERATION_FAILED", "message": str(exc)[:500]},
            ) from exc

        payload, record = await self._locked_payload(project)
        use_case_ids = {item["id"] for item in payload["useCases"]}
        payload["relationships"] = [
            item
            for item in snapshot["relationships"]
            if item.get("sourceId") in use_case_ids and item.get("targetId") in use_case_ids
        ]
        self._refresh_relationship_ids(payload)
        generation = payload["generation"] if isinstance(payload.get("generation"), dict) else {}
        generation["relationsGenerated"] = True
        generation.pop("relationsError", None)
        payload["generation"] = generation
        if generation.get("status") == "running":
            self._touch_generation(payload, generation.get("generationId"), status="running")
        else:
            self._clear_resolved_errors(generation)
        payload = _normalise_payload(payload, project)
        await self._save(record, payload)
        return UseCaseModelResponse.model_validate(payload)

    async def finalize_generation(self, *, project_id: uuid.UUID) -> UseCaseModelResponse:
        """Terminal step: validate and resolve the run to completed / completed_with_errors.

        Called by the FE once every detail batch and the relationships call have settled,
        whether or not they succeeded. Idempotent: a run that is no longer running is returned
        unchanged.
        """

        project = await self._project(project_id)
        payload, record = await self._locked_payload(project)
        generation = payload.get("generation") if isinstance(payload.get("generation"), dict) else {}
        if generation.get("status") != "running":
            return UseCaseModelResponse.model_validate(payload)
        source = await load_project_requirements_source(self.db, project_id=project_id)
        statuses = generation.get("detailStatus") or {}
        missing = [item["id"] for item in payload["useCases"] if statuses.get(item["id"]) != "completed"]
        for use_case_id in missing:
            statuses[use_case_id] = "failed"
        generation["detailStatus"] = statuses
        problems: list[str] = []
        if missing:
            problems.append(f"{len(missing)} use case(s) have no detail yet; retry them from the table.")
        if generation.get("relationsError"):
            problems.append(f"Relationships were not generated: {generation['relationsError']}")
        if problems:
            generation["error"] = " ".join(problems)
        else:
            generation.pop("error", None)

        model = self._payload_to_model(payload, project.name, source=source)
        self._sync_actor_sides(payload, model)
        payload["validation"] = UseCaseValidationResponse.model_validate(
            validate_use_case_model(model, source).model_dump()
        ).model_dump(by_alias=True)
        generation["completedAt"] = _generation_timestamp()
        payload["generation"] = generation
        self._touch_generation(
            payload, generation.get("generationId"), status="completed_with_errors" if problems else "completed"
        )
        payload = _normalise_payload(payload, project)
        await self._save(record, payload)
        return UseCaseModelResponse.model_validate(payload)

    @staticmethod
    def _running_generation_id(payload: dict[str, Any]) -> str:
        generation = payload.get("generation") if isinstance(payload.get("generation"), dict) else {}
        generation_id = generation.get("generationId")
        if generation.get("status") != "running" or not generation_id:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                detail={
                    "code": "USE_CASE_GENERATION_NOT_RUNNING",
                    "message": "No table generation is running for this project.",
                },
            )
        return generation_id

    async def _release_read_transaction(self) -> None:
        # Nothing is written before the provider call, so end the read transaction here: the
        # pooled connection goes back instead of sitting idle for the whole (parallel) call.
        await self.db.commit()

    async def _merge_into_run(
        self, project: Project, generation_id: str, merge: Any
    ) -> dict[str, Any]:
        payload, record = await self._locked_payload(project)
        generation = payload.get("generation") if isinstance(payload.get("generation"), dict) else {}
        if generation.get("generationId") != generation_id or generation.get("status") != "running":
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                detail={
                    "code": "USE_CASE_GENERATION_SUPERSEDED",
                    "message": "This generation run is no longer active; its result was discarded.",
                },
            )
        merge(payload)
        self._touch_generation(payload, generation_id, status="running")
        payload = _normalise_payload(payload, project)
        await self._save(record, payload)
        return payload

    async def _mark_details_failed(self, project: Project, expected_names: dict[str, str]) -> None:
        # Commits itself: the request is about to raise, and get_db rolls back on an exception.
        payload, record = await self._locked_payload(project)
        generation = payload["generation"] if isinstance(payload.get("generation"), dict) else {}
        statuses = generation.setdefault("detailStatus", {})
        for row in payload["useCases"]:
            if expected_names.get(row["id"]) == row["name"]:
                statuses[row["id"]] = "failed"
        payload["generation"] = generation
        if generation.get("status") == "running":
            self._touch_generation(payload, generation.get("generationId"), status="running")
        await self._save(record, payload)
        await self.db.commit()

    async def _record_relations_error(self, project: Project, message: str) -> None:
        payload, record = await self._locked_payload(project)
        generation = payload["generation"] if isinstance(payload.get("generation"), dict) else {}
        generation["relationsError"] = message[:500]
        payload["generation"] = generation
        await self._save(record, payload)
        await self.db.commit()

    @staticmethod
    def _clear_resolved_errors(generation: dict[str, Any]) -> None:
        """A retry after the run can resolve the last failure; reflect that in the run status."""

        if generation.get("status") != "completed_with_errors" or generation.get("relationsError"):
            return
        if any(value != "completed" for value in (generation.get("detailStatus") or {}).values()):
            return
        generation["status"] = "completed"
        generation.pop("error", None)

    @staticmethod
    def _apply_detail_draft(row: dict[str, Any], draft: Any, source: RequirementsSourceSnapshot) -> bool:
        patched = {
            **row,
            "trigger": draft.trigger,
            "preconditions": draft.preconditions,
            "mainFlow": [_flow_step_payload(step) for step in draft.main_flow],
            "alternativeFlows": [_flow_payload(flow) for flow in draft.alternative_flows],
            "exceptionFlows": [_flow_payload(flow) for flow in draft.exception_flows],
            "postconditionsSuccess": draft.postconditions_success,
            "postconditionsFailure": draft.postconditions_failure,
            "businessRules": draft.business_rules,
            "relatedRequirements": [
                {
                    "id": link.id,
                    "type": link.type,
                    "title": link.title,
                    "sourceTrace": _source_trace(canonical_evidence_refs(link.source_refs, source), source),
                }
                for link in draft.related_requirements
            ],
        }
        # Same validation _normalise_payload applies on save -- which silently drops a row that
        # fails it. Checking first keeps the row (flow-less) and reports it as failed instead.
        try:
            UseCaseEntry.model_validate(
                {**patched, "module_id": patched["moduleId"], "source_refs": patched.get("sourceTrace") or ["internal"]}
            )
        except ValidationError:
            return False
        row.update(patched)
        return True

    async def generate_diagram(self, *, project_id: uuid.UUID) -> UseCaseModelResponse:
        """Recompute the ELK diagram layout from the current, already-generated table.

        This is deliberately its own step (no LLM call, no use-case generation) so the FE can
        gate it behind an explicit "table exists" check and offer it as a fast, separate action
        -- including as a manual refresh after a use case/actor/relationship is edited by hand,
        which does not otherwise keep the layout in sync.
        """

        project = await self._project(project_id)
        payload, record = await self._locked_payload(project)
        if not payload["useCases"]:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                detail={
                    "code": "USE_CASE_TABLE_REQUIRED",
                    "message": "Generate the use-case table before generating the diagram.",
                },
            )
        layout_error = await self._attach_diagram_layout(payload)
        if layout_error:
            raise HTTPException(
                status.HTTP_502_BAD_GATEWAY,
                detail={"code": "USE_CASE_DIAGRAM_GENERATION_FAILED", "message": layout_error},
            )
        payload = _normalise_payload(payload, project)
        await self._save(record, payload)
        return UseCaseModelResponse.model_validate(payload)

    async def update_diagram_positions(
        self, *, project_id: uuid.UUID, positions: list[DiagramNodePositionRequest]
    ) -> UseCaseModelResponse:
        """Persist actor/use-case positions the user dragged by hand.

        Only updates nodes that already exist in the current diagram (an id that does not
        resolve is silently skipped, never used to create one) and marks them `manual` so
        _attach_diagram_layout carries the position forward on every later table/diagram
        regenerate instead of recomputing it fresh like an untouched node.
        """

        project = await self._project(project_id)
        payload, record = await self._locked_payload(project)
        layout = payload.get("diagramLayout")
        if not isinstance(layout, dict) or not layout.get("nodes"):
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                detail={
                    "code": "USE_CASE_DIAGRAM_REQUIRED",
                    "message": "Generate the diagram before saving manual positions.",
                },
            )
        nodes_by_id = {node.get("id"): node for node in layout["nodes"] if node.get("id")}
        updated = 0
        for position in positions:
            node = nodes_by_id.get(position.id)
            if node is None or node.get("kind") not in {"actor", "use_case"}:
                continue
            node["x"] = position.x
            node["y"] = position.y
            node["manual"] = True
            updated += 1
        if not updated:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "code": "USE_CASE_DIAGRAM_POSITIONS_NOT_FOUND",
                    "message": "None of the given node ids exist in the current diagram.",
                },
            )
        payload["diagramLayout"] = layout
        layout_error = await self._attach_diagram_layout(payload)
        if layout_error:
            raise HTTPException(
                status.HTTP_502_BAD_GATEWAY,
                detail={"code": "USE_CASE_DIAGRAM_GENERATION_FAILED", "message": layout_error},
            )
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

    async def _mark_generation_started(
        self, *, project: Project, source: RequirementsSourceSnapshot, user_id: uuid.UUID
    ) -> tuple[str, str]:
        payload, record = await self._locked_payload(project)
        previous_generation = payload.get("generation") if isinstance(payload.get("generation"), dict) else {}
        if previous_generation.get("status") == "running" and not _generation_heartbeat_is_stale(previous_generation):
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                detail={
                    "code": "USE_CASE_GENERATION_IN_PROGRESS",
                    "message": "A use-case model generation is already running for this project.",
                },
            )
        generation_id = str(uuid.uuid4())
        started_at = _generation_timestamp()
        payload["sourceHash"] = source.source_hash
        payload["diagramLayout"] = None
        payload["generation"] = {
            "source": "ai",
            "status": "running",
            "generationId": generation_id,
            "startedAt": started_at,
            "updatedAt": started_at,
            "batchCount": None,
            "completedBatchCount": 0,
            "generationMode": "module-batch",
            "stages": {
                "source": "completed",
                "table": "running",
                "relationships": "pending",
                "validation": "pending",
                "layout": "pending",
                "persist": "pending",
            },
        }
        record.model_data = payload
        record.source_hash = source.source_hash
        record.generated_by_id = user_id
        record.last_generation_error = None
        await self.db.flush()
        # Make the running marker visible to a fresh browser tab/reload while the LLM request is
        # still in progress. The final generation payload is committed by _persist_generated.
        await self.db.commit()
        return generation_id, started_at

    @staticmethod
    def _touch_generation(payload: dict[str, Any], generation_id: str | None, *, status: str) -> None:
        """Refresh the running/terminal marker split generation's phases share.

        Each phase (module extraction, then one call per module, then relationships) persists
        its own payload independently; without this, a phase that does not know to carry the
        marker forward would silently drop it, making the project look idle to a concurrent
        request even while a phase is genuinely still in flight.
        """

        generation = payload.get("generation") if isinstance(payload.get("generation"), dict) else {}
        if generation_id:
            generation["generationId"] = generation_id
        generation["status"] = status
        generation["updatedAt"] = _generation_timestamp()
        payload["generation"] = generation

    async def _mark_generation_failed(self, *, project: Project, generation_id: str, message: str) -> None:
        record = await self._record(project.id, for_update=True)
        if record is None:
            return
        payload = _normalise_payload(record.model_data, project)
        generation = payload.get("generation") if isinstance(payload.get("generation"), dict) else {}
        if generation.get("generationId") != generation_id:
            return
        generation["status"] = "failed"
        generation["error"] = message
        generation["completedAt"] = _generation_timestamp()
        stages = generation.get("stages") if isinstance(generation.get("stages"), dict) else {}
        for stage, value in list(stages.items()):
            if value == "running":
                stages[stage] = "failed"
        generation["stages"] = stages
        payload["generation"] = generation
        record.model_data = payload
        record.last_generation_error = message
        await self.db.flush()
        await self.db.commit()

    async def mark_running_generation_failed(self, *, project_id: uuid.UUID, message: str) -> None:
        """Release a durable running marker after an unexpected request failure.

        The normal generation path records its own terminal state.  This fallback is used by
        the HTTP route for exceptions raised after the marker was committed (for example a
        persistence or response construction error), so a failed request cannot leave the
        Generate button locked forever on the next page load.
        """

        try:
            # A database exception leaves the request transaction unusable until it is rolled
            # back.  Start the recovery lookup from a clean transaction.
            await self.db.rollback()
            project = await self._project(project_id)
            record = await self._record(project.id, for_update=True)
            if record is None:
                return
            payload = _normalise_payload(record.model_data, project)
            generation = payload.get("generation") if isinstance(payload.get("generation"), dict) else {}
            if generation.get("status") != "running":
                return
            generation["status"] = "failed"
            generation["error"] = message[:500]
            generation["completedAt"] = _generation_timestamp()
            stages = generation.get("stages") if isinstance(generation.get("stages"), dict) else {}
            for stage, value in list(stages.items()):
                if value in {"running", "pending"}:
                    stages[stage] = "failed"
            generation["stages"] = stages
            payload["generation"] = generation
            record.model_data = payload
            record.last_generation_error = generation["error"]
            await self.db.flush()
            await self.db.commit()
        except Exception:
            # Never mask the original request failure with cleanup failure.
            await self.db.rollback()

    async def _update_generation_batch_progress(
        self,
        *,
        project: Project,
        generation_id: str,
        batch_count: int,
        completed_batch_count: int,
    ) -> None:
        """Publish module-batch progress without replacing the source-backed model."""

        record = await self._record(project.id, for_update=True)
        if record is None:
            return
        payload = _normalise_payload(record.model_data, project)
        generation = payload.get("generation") if isinstance(payload.get("generation"), dict) else {}
        if generation.get("generationId") != generation_id:
            return
        generation["batchCount"] = batch_count
        generation["completedBatchCount"] = completed_batch_count
        generation["status"] = "running"
        payload["generation"] = generation
        record.model_data = payload
        await self.db.flush()
        await self.db.commit()

    async def _persist_generation_progress(
        self, *, project: Project, generation_id: str, payload: dict[str, Any]
    ) -> None:
        record = await self._record(project.id, for_update=True)
        if record is None:
            return
        stored = _normalise_payload(record.model_data, project)
        stored_generation = stored.get("generation") if isinstance(stored.get("generation"), dict) else {}
        if stored_generation.get("generationId") != generation_id:
            return
        progress_payload = copy.deepcopy(payload)
        generation = progress_payload.get("generation") if isinstance(progress_payload.get("generation"), dict) else {}
        generation["generationId"] = generation_id
        generation["status"] = "running"
        generation.setdefault("startedAt", stored_generation.get("startedAt"))
        progress_payload["generation"] = generation
        record.model_data = progress_payload
        record.last_generation_error = None
        await self.db.flush()
        await self.db.commit()

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
            request_timeout=settings.use_case_generation_batch_timeout_seconds,
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
        # The system boundary represents the product being built.  Document registry titles
        # such as "Product Requirements Document" are source labels, not the product name.
        system_name = project.name
        if model.system.name != system_name:
            model.system = UseCaseSystem(
                id=model.system.id,
                name=system_name,
                description=model.system.description,
                source_refs=model.system.source_refs,
            )
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
                    "sourceRefs": list(item.source_refs),
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
            "diagramLayout": None,
            "plantUml": None,
        }
        return _normalise_payload(payload, project)

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
        # One relationship per pair of use cases, in either direction: include (mandatory) and
        # extend (optional) on the same pair contradict each other. The provider's first wins.
        seen_pairs: set[frozenset[str]] = set()
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
            pair = frozenset((draft.source_id, draft.target_id))
            if pair in seen_pairs:
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
            seen_pairs.add(pair)

        # ``include`` denotes reusable behavior shared by at least two base
        # use cases.  Drop singleton candidates before validation/rendering so
        # an over-eager provider cannot leave a review warning in the saved
        # model.
        include_target_counts: dict[str, int] = {}
        for relation in accepted:
            if relation["type"] == "include":
                target = relation["targetId"]
                include_target_counts[target] = include_target_counts.get(target, 0) + 1
        # Ids are assigned only now, one at a time, so each sees the ones before it --
        # _next_relationship_id deduplicates against payload["relationships"].
        payload["relationships"] = []
        for relation in accepted:
            if relation["type"] == "include" and include_target_counts.get(relation["targetId"], 0) < 2:
                continue
            relation_id = self._next_relationship_id(payload, relation["sourceId"], relation["targetId"])
            payload["relationships"].append({"id": relation_id, **relation})

    @staticmethod
    def _draft_row(use_case_id: str, module_id: str, draft: Any, source: RequirementsSourceSnapshot) -> dict[str, Any]:
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
                # One actor per use case -- see UseCaseGroupDetailDraft, which has no field for a
                # second (supporting) actor to come from.
                "secondary_actor_ids": [],
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
        payload["diagramLayout"] = None
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


def _generation_timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _http_exception_message(error: HTTPException) -> str:
    detail = error.detail
    if isinstance(detail, dict):
        return str(detail.get("message") or detail.get("detail") or detail)
    return str(detail)


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
            # Raw evidence ids (not the human-readable sourceTrace strings above) for the module
            # extraction phase's own module<->evidence mapping. generate_module_candidates uses
            # this to bound its prompt to what Phase 1 actually attributed to this module,
            # instead of sending the whole BRD/PRD to every module's request.
            "sourceRefs": item.get("sourceRefs") or item.get("source_refs") or [],
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
        "diagramLayout": raw.get("diagramLayout") if isinstance(raw.get("diagramLayout"), dict) else None,
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
            "diagramLayout": output["diagramLayout"],
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
        logger.warning("LLM response was not text-like (got %s); cannot parse as JSON", type(raw).__name__)
        raise ValueError("LLM did not return JSON")
    cleaned = content.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I | re.S).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            # Nothing here logs the raw text the provider sent back on a parse failure, so a
            # failure like this is otherwise undiagnosable from outside a debugger. Truncated
            # rather than full: this can carry a large chunk of BRD/PRD-derived content.
            logger.warning("Could not find a JSON object in LLM response: %s", content[:4000])
            raise ValueError("Could not parse JSON from LLM response") from None
        try:
            parsed = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            logger.warning("LLM response JSON fragment did not parse (%s): %s", exc, content[:4000])
            raise ValueError("Could not parse JSON from LLM response") from None
    if not isinstance(parsed, dict):
        logger.warning("LLM response JSON was not an object: %s", content[:4000])
        raise ValueError("LLM response must be a JSON object")
    return parsed


def _match_codes_from_refs(source: RequirementsSourceSnapshot, refs: list[str]) -> set[str]:
    """Resolve module extraction's own source_refs into the requirement/capability codes
    _module_generation_context matches against source lines (e.g. "FR-TM01", "C3")."""

    evidence = source.evidence_by_id()
    codes: set[str] = set()
    for ref in refs:
        item = evidence.get(ref)
        if item is None:
            continue
        code = item.entity_id or (item.excerpt.strip() if len(item.excerpt) <= 40 else None)
        if code:
            codes.add(code)
    return codes


def _module_generation_context(
    source: RequirementsSourceSnapshot,
    module: UseCaseModule,
    rows: list[UseCaseEntry],
    *,
    extra_match_codes: set[str] | None = None,
) -> tuple[str, str]:
    """Build a bounded, source-backed prompt for one module batch.

    The complete source remains available to the deterministic parser and traceability checks.
    The provider only needs the module's requirement rows, the stakeholder roster, and the nearby
    BRD rules for detail enrichment.  Keeping this excerpt bounded is what makes batching useful
    for large projects.

    `extra_match_codes` widens matching beyond `rows`' own related-requirement ids -- the split
    generation flow has no pre-existing rows for a module (that is what this call is generating),
    so it resolves the module's own source_refs (module extraction's own module<->evidence
    mapping) into requirement/capability codes and passes them here instead.
    """

    requirement_ids = {
        link.id
        for row in rows
        for link in row.related_requirements
        if link.id
    }
    if extra_match_codes:
        requirement_ids |= extra_match_codes
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


def _use_case_detail_context(source: RequirementsSourceSnapshot, refs: set[str]) -> tuple[str, str]:
    """Bounded excerpt for writing a few use cases' detail: the evidence lines they cite plus
    the requirement/rule rows sharing those codes -- not the whole BRD/PRD. What is not in here
    is what the prompt tells the provider to leave empty rather than invent."""

    evidence = source.evidence_by_id()
    codes = _match_codes_from_refs(source, sorted(refs))
    blocks: list[str] = []
    cited = [f"[{ref}] {evidence[ref].excerpt}" for ref in sorted(refs) if ref in evidence]
    if cited:
        blocks.append("--- cited evidence ---\n" + "\n".join(cited))
    for component in source.components:
        if not codes or component.artifact_type not in {
            "functional_requirement",
            "non_functional_requirement",
            "business_rules",
            "scope_capabilities",
            "use_case",
        }:
            continue
        lines = [
            line
            for line in component.body.splitlines()
            if any(code.casefold() in line.casefold() for code in codes)
        ][:40]
        if lines:
            blocks.append(f"--- {component.document_type}.{component.artifact_type} ---\n" + "\n".join(lines))
    context = "\n\n".join(blocks)
    if len(context) > 16000:
        context = context[:15950].rstrip() + "\n[context bounded by backend]"

    evidence_lines = ["evidence_id | kind | locator | excerpt", "---|---|---|---"]
    for item in source.evidence:
        if item.evidence_id not in refs and (item.entity_id or "") not in codes:
            continue
        excerpt = " ".join(item.excerpt.split()).replace("|", "\\|")[:180]
        evidence_lines.append(f"{item.evidence_id} | {item.kind} | {item.locator} | {excerpt}")
        if len("\n".join(evidence_lines)) > 8000:
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
        # One actor per use case now -- the draft schema no longer has a secondary_actor_ids
        # field for the provider to fill in (see UseCaseGroupDetailDraft), so this clears
        # whatever the deterministic floor row it is enriching had, rather than leaving a stale
        # one the AI pass never got a chance to confirm or drop.
        data["secondary_actor_ids"] = []
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
