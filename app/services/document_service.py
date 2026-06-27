from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.documents.registry import (
    all_container_types,
    all_item_types,
    children_of,
    container_for,
    get_config,
    item_configs,
    output_contract,
)
from app.models.artifact import (
    Artifact,
    ArtifactStatus,
    ArtifactType,
    ArtifactVersion,
    ChangeSource,
    VersionStatus,
)
from app.schemas.document import (
    DocumentItemView,
    DocumentItemWriteRequest,
    DocumentTypesResponse,
    DocumentTypeView,
    DocumentView,
)
from app.services.artifact_service import ArtifactService


class DocumentService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.artifacts = ArtifactService(db)

    def types(self) -> DocumentTypesResponse:
        return DocumentTypesResponse(
            containers=[self._type_view(item) for item in all_container_types()],
            items=[self._type_view(item) for item in all_item_types()],
        )

    async def get_current_item_body(
        self,
        *,
        artifact_id: uuid.UUID,
        project_id: uuid.UUID | None = None,
    ) -> str:
        artifact = await self.get_document_item_artifact(
            artifact_id=artifact_id,
            project_id=project_id,
        )
        if artifact.current_version_id is None:
            return ""
        version = await self.db.get(ArtifactVersion, artifact.current_version_id)
        return version.body if version is not None else ""

    async def get_document_item_artifact(
        self,
        *,
        artifact_id: uuid.UUID,
        project_id: uuid.UUID | None = None,
        for_update: bool = False,
    ) -> Artifact:
        query = select(Artifact).where(
            Artifact.id == artifact_id,
            Artifact.parent_id.is_not(None),
        )
        if project_id is not None:
            query = query.where(Artifact.project_id == project_id)
        if for_update:
            query = query.with_for_update()
        artifact = (await self.db.execute(query)).scalar_one_or_none()
        if artifact is None:
            raise ValueError("Document item does not exist")
        return artifact

    async def create_item_version(
        self,
        *,
        artifact_id: uuid.UUID,
        project_id: uuid.UUID,
        title: str,
        body: str,
        created_by_id: uuid.UUID | None,
        change_source: ChangeSource,
        agent_run_id: uuid.UUID | None = None,
        tool_call_id: uuid.UUID | None = None,
        metadata: dict[str, Any] | None = None,
        mark_accepted: bool = False,
    ) -> tuple[Artifact, ArtifactVersion]:
        if tool_call_id is not None:
            existing_version = (
                await self.db.execute(select(ArtifactVersion).where(ArtifactVersion.tool_call_id == tool_call_id))
            ).scalar_one_or_none()
            if existing_version is not None:
                artifact = await self.get_document_item_artifact(
                    artifact_id=existing_version.artifact_id,
                    project_id=project_id,
                    for_update=True,
                )
                artifact.current_version_id = existing_version.id
                artifact.title = existing_version.title or artifact.title
                if mark_accepted:
                    artifact.status = ArtifactStatus.ACCEPTED
                    await self._recompute_parent_acceptance(artifact)
                await self.db.flush()
                return artifact, existing_version

        artifact = await self.get_document_item_artifact(
            artifact_id=artifact_id,
            project_id=project_id,
            for_update=True,
        )
        current_version = await self._current_version(artifact)
        next_version_number = current_version.version_number + 1 if current_version is not None else 1
        version = ArtifactVersion(
            artifact_id=artifact.id,
            version_number=next_version_number,
            title=title or artifact.title,
            body=body,
            status=VersionStatus.DRAFT,
            change_source=change_source,
            parent_version_id=current_version.id if current_version is not None else None,
            agent_run_id=agent_run_id,
            tool_call_id=tool_call_id,
            created_by_id=created_by_id,
            extra_metadata={**(metadata or {}), "focused_artifact_id": str(artifact.id)},
        )
        self.db.add(version)
        await self.db.flush()

        artifact.current_version_id = version.id
        artifact.title = title or artifact.title
        if mark_accepted:
            artifact.status = ArtifactStatus.ACCEPTED
            await self._recompute_parent_acceptance(artifact)
        await self.db.flush()
        return artifact, version

    async def _recompute_parent_acceptance(self, artifact: Artifact) -> None:
        if artifact.parent_id is None:
            return
        parent = await self.db.get(Artifact, artifact.parent_id, with_for_update=True)
        if parent is None:
            return
        expected_children = children_of(parent.type.value)
        if not expected_children:
            return
        versioned_rows = (
            await self.db.execute(
                select(Artifact.type).where(
                    Artifact.project_id == artifact.project_id,
                    Artifact.parent_id == parent.id,
                    Artifact.type.in_([ArtifactType(item) for item in expected_children]),
                    Artifact.current_version_id.is_not(None),
                )
            )
        ).scalars()
        versioned_types = {value.value for value in versioned_rows}
        parent.status = (
            ArtifactStatus.ACCEPTED if set(expected_children).issubset(versioned_types) else ArtifactStatus.DRAFT
        )

    async def document_coverage(
        self,
        *,
        project_id: uuid.UUID,
        artifact_type: str,
        focused_artifact_id: uuid.UUID | None,
    ) -> dict[str, Any]:
        container_type = container_for(artifact_type)
        container_id: uuid.UUID | None = None

        if focused_artifact_id is not None:
            focused = await self.db.get(Artifact, focused_artifact_id)
            if focused is not None and focused.project_id == project_id:
                if focused.parent_id is not None:
                    parent = await self.db.get(Artifact, focused.parent_id)
                    if parent is not None:
                        container_type = parent.type.value
                        container_id = parent.id
                elif focused.type.value in all_container_types():
                    container_type = focused.type.value
                    container_id = focused.id
        elif artifact_type in all_container_types():
            container_type = artifact_type

        if container_type is None:
            return {
                "section_coverage": None,
                "coverage_complete": None,
                "section_coverage_stall_count": 0,
            }

        registry_items = children_of(container_type)
        if not registry_items:
            return {
                "section_coverage": {},
                "coverage_complete": False,
                "section_coverage_stall_count": 0,
            }

        if container_id is None:
            container_id = (
                await self.db.execute(
                    select(Artifact.id).where(
                        Artifact.project_id == project_id,
                        Artifact.type == ArtifactType(container_type),
                        Artifact.parent_id.is_(None),
                    )
                )
            ).scalar_one_or_none()

        accepted_types: set[str] = set()
        if container_id is not None:
            accepted_rows = (
                await self.db.execute(
                    select(Artifact.type).where(
                        Artifact.project_id == project_id,
                        Artifact.parent_id == container_id,
                        Artifact.type.in_([ArtifactType(item) for item in registry_items]),
                        Artifact.status == ArtifactStatus.ACCEPTED,
                    )
                )
            ).scalars()
            accepted_types = {value.value for value in accepted_rows}

        coverage = {item_type: ("filled" if item_type in accepted_types else "missing") for item_type in registry_items}
        accepted_count = len(accepted_types)
        return {
            "section_coverage": coverage,
            "coverage_complete": accepted_count == len(registry_items),
            "section_coverage_stall_count": 0,
        }

    async def get_document(self, *, project_id: uuid.UUID, document_type: str) -> DocumentView:
        config = self._require_container(document_type)
        container = await self._load_container(project_id, document_type)
        children_by_type: dict[str, Artifact] = {}
        if container is not None:
            children_by_type = {child.type.value: child for child in container.children}
        items = [
            await self._item_view(item.artifact_type, children_by_type.get(item.artifact_type))
            for item in item_configs(document_type)
        ]
        return DocumentView(
            document_type=ArtifactType(document_type),
            label=config.label,
            description=config.description,
            artifact_id=container.id if container else None,
            project_id=project_id,
            status=container.status if container else None,
            title=container.title if container else None,
            current_version_id=container.current_version_id if container else None,
            items=items,
        )

    async def create_document(
        self,
        *,
        project_id: uuid.UUID,
        document_type: str,
        created_by_id: uuid.UUID,
    ) -> DocumentView:
        config = self._require_container(document_type)
        container = await self._load_container(project_id, document_type)
        if container is None:
            container = Artifact(
                project_id=project_id,
                type=ArtifactType(document_type),
                status=ArtifactStatus.DRAFT,
                title=config.label,
                extra_metadata={},
                created_by_id=created_by_id,
            )
            self.db.add(container)
            await self.db.flush()
        return await self.get_document(project_id=project_id, document_type=document_type)

    async def get_item(
        self,
        *,
        project_id: uuid.UUID,
        document_type: str,
        item_type: str,
    ) -> DocumentItemView:
        self._require_item(document_type, item_type)
        container = await self._load_container(project_id, document_type)
        if container is None:
            raise ValueError("Document does not exist")
        artifact = next((child for child in container.children if child.type.value == item_type), None)
        if artifact is None:
            raise ValueError("Document item does not exist")
        return await self._item_view(item_type, artifact)

    async def upsert_item(
        self,
        *,
        project_id: uuid.UUID,
        document_type: str,
        item_type: str,
        body: DocumentItemWriteRequest,
        created_by_id: uuid.UUID,
    ) -> DocumentItemView:
        self._require_item(document_type, item_type)
        container = await self._load_container(project_id, document_type)
        if container is None:
            raise ValueError("Document does not exist")
        artifact = next((child for child in container.children if child.type.value == item_type), None)
        item_config = get_config(item_type)
        if artifact is None:
            artifact = Artifact(
                project_id=project_id,
                parent_id=container.id,
                type=ArtifactType(item_type),
                status=body.status,
                priority=body.priority,
                code=body.code,
                title=body.title or item_config.label,
                confidence=body.confidence,
                extra_metadata=body.metadata,
                created_by_id=created_by_id,
            )
            self.db.add(artifact)
            await self.db.flush()
            version_number = 1
            parent_version_id = None
        else:
            current = await self._current_version(artifact)
            version_number = current.version_number + 1 if current else 1
            parent_version_id = current.id if current else None
            artifact.status = body.status
            artifact.priority = body.priority
            artifact.code = body.code
            artifact.title = body.title or artifact.title
            artifact.confidence = body.confidence
            artifact.extra_metadata = body.metadata

        version = ArtifactVersion(
            artifact_id=artifact.id,
            version_number=version_number,
            title=body.title or artifact.title,
            body=body.body,
            status=VersionStatus.DRAFT,
            change_source=body.change_source,
            change_summary=body.change_summary,
            parent_version_id=parent_version_id,
            created_by_id=created_by_id,
            extra_metadata=body.metadata,
        )
        self.db.add(version)
        await self.db.flush()
        artifact.current_version_id = version.id
        await self._recompute_parent_acceptance(artifact)
        await self.db.flush()
        return await self._item_view(item_type, artifact)

    async def _load_container(self, project_id: uuid.UUID, document_type: str) -> Artifact | None:
        return (
            await self.db.execute(
                select(Artifact)
                .where(
                    Artifact.project_id == project_id,
                    Artifact.type == ArtifactType(document_type),
                    Artifact.parent_id.is_(None),
                )
                .options(selectinload(Artifact.children))
                .limit(1)
            )
        ).scalar_one_or_none()

    async def _item_view(self, item_type: str, artifact: Artifact | None) -> DocumentItemView:
        config = get_config(item_type)
        if artifact is None:
            return DocumentItemView(
                artifact_type=ArtifactType(item_type),
                label=config.label,
                description=config.description,
            )
        versions = (
            (
                await self.db.execute(
                    select(ArtifactVersion)
                    .where(ArtifactVersion.artifact_id == artifact.id)
                    .order_by(ArtifactVersion.version_number)
                )
            )
            .scalars()
            .all()
        )
        version_views = [
            await self.artifacts.version_to_response(version, artifact_type=item_type) for version in versions
        ]
        current_version = next(
            (version for version in version_views if version.id == artifact.current_version_id),
            None,
        )
        return DocumentItemView(
            artifact_type=artifact.type,
            label=config.label,
            description=config.description,
            artifact_id=artifact.id,
            parent_id=artifact.parent_id,
            status=artifact.status,
            priority=artifact.priority,
            code=artifact.code,
            title=artifact.title,
            confidence=artifact.confidence,
            metadata=artifact.extra_metadata or {},
            current_version_id=artifact.current_version_id,
            current_version=current_version,
            versions=version_views,
            created_at=artifact.created_at,
        )

    async def _current_version(self, artifact: Artifact) -> ArtifactVersion | None:
        if artifact.current_version_id is None:
            return None
        return await self.db.get(ArtifactVersion, artifact.current_version_id)

    def _require_container(self, document_type: str):
        config = get_config(document_type)
        if not config.is_container:
            raise ValueError("Invalid document type")
        return config

    def _require_item(self, document_type: str, item_type: str) -> None:
        self._require_container(document_type)
        if item_type not in children_of(document_type):
            raise ValueError("Item type does not belong to document")

    def _type_view(self, artifact_type: str) -> DocumentTypeView:
        config = get_config(artifact_type)
        contract = None if config.is_container else output_contract(artifact_type).to_dict()
        return DocumentTypeView(
            artifact_type=config.artifact_type,
            label=config.label,
            description=config.description,
            children=list(config.children),
            is_container=config.is_container,
            output_contract=contract,
        )
