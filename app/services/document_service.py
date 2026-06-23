from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.documents.registry import (
    all_container_types,
    all_item_types,
    children_of,
    get_config,
    item_configs,
)
from app.models.artifact import (
    Artifact,
    ArtifactStatus,
    ArtifactType,
    ArtifactVersion,
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
            raise ValueError("Document chưa tồn tại")
        artifact = next((child for child in container.children if child.type.value == item_type), None)
        if artifact is None:
            raise ValueError("Document item chưa tồn tại")
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
            raise ValueError("Document chưa tồn tại")
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
            await self.db.execute(
                select(ArtifactVersion)
                .where(ArtifactVersion.artifact_id == artifact.id)
                .order_by(ArtifactVersion.version_number)
            )
        ).scalars().all()
        version_views = [
            await self.artifacts.version_to_response(version, artifact_type=item_type)
            for version in versions
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
            raise ValueError("Document type không hợp lệ")
        return config

    def _require_item(self, document_type: str, item_type: str) -> None:
        self._require_container(document_type)
        if item_type not in children_of(document_type):
            raise ValueError("Item type không thuộc document")

    def _type_view(self, artifact_type: str) -> DocumentTypeView:
        config = get_config(artifact_type)
        return DocumentTypeView(
            artifact_type=config.artifact_type,
            label=config.label,
            description=config.description,
            children=list(config.children),
            is_container=config.is_container,
        )
