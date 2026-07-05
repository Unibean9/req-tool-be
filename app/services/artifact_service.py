from __future__ import annotations

import logging
import uuid
from hashlib import sha256
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.documents.registry import container_for
from app.graphs.lifecycle_resolver import (
    ArtifactLifecycleSnapshot,
    BasedOnPredecessorSnapshot,
    LifecycleVerdict,
    StructuralPredecessorSnapshot,
    resolve_lifecycle,
)
from app.graphs.policy import ancestor_types
from app.models.artifact import (
    Artifact,
    ArtifactEvidence,
    ArtifactLink,
    ArtifactReview,
    ArtifactStatus,
    ArtifactVersion,
    RelationType,
    ReviewStatus,
    SourceDocument,
    VersionStatus,
)
from app.models.organization import OrgMember
from app.models.project import Project
from app.schemas.artifact import (
    ArtifactCreateRequest,
    ArtifactEvidenceCreateRequest,
    ArtifactEvidenceResponse,
    ArtifactGraphResponse,
    ArtifactLinkCreateRequest,
    ArtifactLinkResponse,
    ArtifactNode,
    ArtifactResponse,
    ArtifactReviewRequest,
    ArtifactReviewResponse,
    ArtifactUpdateRequest,
    ArtifactVersionResponse,
    GraphWarning,
    SourceDocumentCreateRequest,
    SourceDocumentResponse,
)

logger = logging.getLogger(__name__)

ALLOWED_STATUS_TRANSITIONS: dict[ArtifactStatus, set[ArtifactStatus]] = {
    ArtifactStatus.DRAFT: {ArtifactStatus.NEEDS_CLARIFICATION},
    ArtifactStatus.NEEDS_CLARIFICATION: {ArtifactStatus.DRAFT, ArtifactStatus.ACCEPTED},
    ArtifactStatus.ACCEPTED: set(),
    ArtifactStatus.REJECTED: {ArtifactStatus.DRAFT},
    ArtifactStatus.ARCHIVED: set(),
}


class InvalidArtifactStatusTransition(ValueError):
    pass


class ArtifactInUseError(Exception):
    def __init__(self, artifact_ids: list[uuid.UUID]):
        self.artifact_ids = artifact_ids
        super().__init__("Artifact has live downstream dependents")


def _enum_value(value: Any) -> str | None:
    if value is None:
        return None
    enum_value = getattr(value, "value", None)
    return str(enum_value if enum_value is not None else value)


def _artifact_type(value: Any) -> str:
    return str(_enum_value(value) or "")


def _current_metadata(artifact: Artifact | None) -> dict[str, Any]:
    if artifact is None or artifact.current_version is None:
        return {}
    metadata = artifact.current_version.extra_metadata or {}
    return metadata if isinstance(metadata, dict) else {}


def _based_on_from_artifact(artifact: Artifact | None) -> dict[str, str]:
    based_on = _current_metadata(artifact).get("based_on")
    if not isinstance(based_on, dict):
        return {}
    return {str(key): str(value) for key, value in based_on.items() if key and value}


def _log_terminal_review_noop(artifact: Artifact, version: ArtifactVersion, review_status: ReviewStatus) -> None:
    """A review outcome that could not drive the artifact's status (terminal or non-current version)."""
    logger.info(
        "review_status_noop artifact_id=%s version_id=%s artifact_status=%s version_status=%s review=%s",
        artifact.id,
        version.id,
        _enum_value(artifact.status),
        _enum_value(version.status),
        _enum_value(review_status),
    )


def _lifecycle_report_from_verdict(verdict: LifecycleVerdict) -> dict[str, Any]:
    return {
        "state": verdict.state.value,
        "reason": verdict.reason,
        "allowed_actions": sorted(action.value for action in verdict.allowed_actions),
        "blockers": list(verdict.blockers),
    }


class ArtifactService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(
        self,
        *,
        project_id: uuid.UUID,
        body: ArtifactCreateRequest,
        created_by_id: uuid.UUID,
    ) -> ArtifactResponse:
        await self._require_project_member(project_id, created_by_id)
        if body.parent_id is not None:
            parent = await self.db.get(Artifact, body.parent_id)
            if parent is None or parent.project_id != project_id:
                raise ValueError("Parent artifact does not belong to this project")
            expected_container = container_for(body.type.value)
            if parent.parent_id is not None or expected_container != parent.type.value:
                raise ValueError("Parent artifact does not match the document registry")
        artifact = Artifact(
            project_id=project_id,
            parent_id=body.parent_id,
            type=body.type,
            status=body.status,
            priority=body.priority,
            code=body.code,
            title=body.title,
            confidence=body.confidence,
            extra_metadata=body.metadata,
            created_by_id=created_by_id,
        )
        self.db.add(artifact)
        await self.db.flush()

        version = ArtifactVersion(
            artifact_id=artifact.id,
            version_number=1,
            title=body.title,
            body=body.body,
            status=VersionStatus.DRAFT,
            change_source=body.change_source,
            change_summary=body.change_summary,
            created_by_id=created_by_id,
            source_document_id=body.source_document_id,
            extra_metadata=body.metadata,
        )
        self.db.add(version)
        await self.db.flush()
        artifact.current_version_id = version.id
        await self.db.flush()
        return await self.to_response(artifact)

    async def list(
        self,
        *,
        project_id: uuid.UUID,
        user_id: uuid.UUID,
        artifact_type=None,
        status=None,
        step_key=None,
        phase=None,
        priority=None,
        current_version_status=None,
    ) -> list[ArtifactResponse]:
        await self._require_project_member(project_id, user_id)
        query = select(Artifact).where(Artifact.project_id == project_id).order_by(Artifact.created_at, Artifact.id)
        if artifact_type is not None:
            query = query.where(Artifact.type == artifact_type)
        if status is not None:
            query = query.where(Artifact.status == status)
        if priority is not None:
            query = query.where(Artifact.priority == priority)
        if step_key is not None or phase is not None:
            from app.models.artifact import WorkflowStep

            query = query.join(WorkflowStep, Artifact.step_id == WorkflowStep.id)
            if step_key is not None:
                query = query.where(WorkflowStep.step_key == step_key)
            if phase is not None:
                query = query.where(WorkflowStep.phase == phase)
        if current_version_status is not None:
            query = query.join(ArtifactVersion, Artifact.current_version_id == ArtifactVersion.id).where(
                ArtifactVersion.status == current_version_status
            )
        artifacts = (await self.db.execute(query)).scalars().all()
        lifecycle_reports = await self.lifecycle_reports_by_id(project_id=project_id)
        return [
            await self.to_response(artifact, lifecycle_report=lifecycle_reports.get(artifact.id))
            for artifact in artifacts
        ]

    async def get(
        self,
        *,
        project_id: uuid.UUID,
        artifact_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> ArtifactResponse:
        await self._require_project_member(project_id, user_id)
        artifact = await self._get_project_artifact(project_id, artifact_id)
        return await self.to_response(artifact)

    async def update(
        self,
        *,
        project_id: uuid.UUID,
        artifact_id: uuid.UUID,
        body: ArtifactUpdateRequest,
        updated_by_id: uuid.UUID,
    ) -> ArtifactResponse:
        await self._require_project_member(project_id, updated_by_id)
        artifact = await self._get_project_artifact(project_id, artifact_id)
        if artifact.status == ArtifactStatus.ARCHIVED:
            raise InvalidArtifactStatusTransition("Cannot edit an archived artifact")
        current = await self._get_current_version(artifact)

        next_title = body.title if body.title is not None else current.title
        next_body = body.body if body.body is not None else current.body
        next_metadata = body.metadata if body.metadata is not None else artifact.extra_metadata
        if body.status is not None:
            self._validate_status_transition(artifact.status, body.status)

        # Known limitation: concurrent updates need SELECT FOR UPDATE on PostgreSQL to avoid duplicate version_number.
        version = ArtifactVersion(
            artifact_id=artifact.id,
            version_number=current.version_number + 1,
            title=next_title,
            body=next_body,
            status=VersionStatus.DRAFT,
            change_source=body.change_source,
            change_summary=body.change_summary,
            parent_version_id=current.id,
            created_by_id=updated_by_id,
            source_document_id=body.source_document_id,
            extra_metadata=next_metadata,
        )
        self.db.add(version)
        await self.db.flush()

        artifact.title = next_title
        artifact.current_version_id = version.id
        if body.status is not None:
            artifact.status = body.status
        if body.priority is not None:
            artifact.priority = body.priority
        if body.code is not None:
            artifact.code = body.code
        if body.confidence is not None:
            artifact.confidence = body.confidence
        if body.metadata is not None:
            artifact.extra_metadata = body.metadata
        await self.db.flush()
        return await self.to_response(artifact)

    async def restore(
        self,
        *,
        project_id: uuid.UUID,
        artifact_id: uuid.UUID,
        version_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> ArtifactResponse:
        await self._require_project_member(project_id, user_id)
        artifact = await self._get_project_artifact(project_id, artifact_id)
        if artifact.status == ArtifactStatus.ARCHIVED:
            raise InvalidArtifactStatusTransition("Cannot restore a version of an archived artifact")
        version = await self.db.get(ArtifactVersion, version_id)
        if version is None or version.artifact_id != artifact.id:
            raise ValueError("Artifact version not found")
        artifact.current_version_id = version.id
        artifact.title = version.title
        await self.db.flush()
        return await self.to_response(artifact)

    async def delete(
        self,
        *,
        project_id: uuid.UUID,
        artifact_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> None:
        await self.archive_artifact(
            project_id=project_id,
            artifact_id=artifact_id,
            user_id=user_id,
            reason="Deleted through the REST artifact endpoint",
            source="human_delete",
        )

    async def archive_artifact(
        self,
        *,
        project_id: uuid.UUID,
        artifact_id: uuid.UUID,
        user_id: uuid.UUID,
        reason: str,
        superseded_by_id: uuid.UUID | None = None,
        source: str = "retirement",
    ) -> Artifact:
        await self._require_project_member(project_id, user_id)
        artifact = await self._get_project_artifact(project_id, artifact_id)
        if superseded_by_id is not None:
            if superseded_by_id == artifact_id:
                raise ValueError("Artifact cannot supersede itself")
            superseding = await self._get_project_artifact(project_id, superseded_by_id)
            if superseding.status == ArtifactStatus.ARCHIVED:
                raise ValueError("Superseding artifact is archived")
        dependent_ids = await self.downstream_usage_ids(project_id=project_id, artifact_id=artifact_id)
        if dependent_ids:
            raise ArtifactInUseError(dependent_ids)

        metadata = dict(artifact.extra_metadata or {})
        retirement = {
            "source": source,
            "reason": str(reason or "").strip(),
            "retired_by_id": str(user_id),
            "superseded_by": str(superseded_by_id) if superseded_by_id else None,
        }
        metadata["retirement"] = retirement
        if superseded_by_id is not None:
            metadata["superseded_by"] = str(superseded_by_id)
        artifact.extra_metadata = metadata
        artifact.status = ArtifactStatus.ARCHIVED
        if artifact.current_version_id is not None:
            current = await self.db.get(ArtifactVersion, artifact.current_version_id)
            if current is not None:
                current.status = VersionStatus.ARCHIVED
        await self.db.flush()
        return artifact

    async def add_evidence(
        self,
        *,
        project_id: uuid.UUID,
        artifact_id: uuid.UUID,
        body: ArtifactEvidenceCreateRequest,
        created_by_id: uuid.UUID,
    ) -> ArtifactEvidenceResponse:
        await self._require_project_member(project_id, created_by_id)
        artifact = await self.db.get(Artifact, artifact_id)
        if artifact is None:
            raise ValueError("Artifact not found")
        if artifact.project_id != project_id:
            raise PermissionError("Artifact does not belong to this project")
        if body.artifact_version_id is not None:
            version = await self.db.get(ArtifactVersion, body.artifact_version_id)
            if version is None or version.artifact_id != artifact.id:
                raise ValueError("Evidence version does not belong to this artifact")
        if body.source_document_id is not None:
            document = await self.db.get(SourceDocument, body.source_document_id)
            if document is None or document.project_id != project_id:
                raise ValueError("Source document does not belong to this project")
        evidence = ArtifactEvidence(
            artifact_id=artifact.id,
            artifact_version_id=body.artifact_version_id,
            source_document_id=body.source_document_id,
            source_type=body.source_type,
            locator=body.locator,
            excerpt=body.excerpt,
            confidence=body.confidence,
            extra_metadata=body.metadata,
        )
        self.db.add(evidence)
        await self.db.flush()
        return self.evidence_to_response(evidence)

    async def list_evidence(
        self,
        *,
        project_id: uuid.UUID,
        artifact_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> list[ArtifactEvidenceResponse]:
        await self._require_project_member(project_id, user_id)
        artifact = await self._get_project_artifact(project_id, artifact_id)
        result = await self.db.execute(
            select(ArtifactEvidence)
            .where(ArtifactEvidence.artifact_id == artifact.id)
            .order_by(ArtifactEvidence.created_at)
        )
        return [self.evidence_to_response(evidence) for evidence in result.scalars().all()]

    async def downstream_usage_ids(self, *, project_id: uuid.UUID, artifact_id: uuid.UUID) -> list[uuid.UUID]:
        linked_dependents = (
            await self.db.execute(
                select(ArtifactLink.source_artifact_id)
                .join(Artifact, Artifact.id == ArtifactLink.source_artifact_id)
                .where(
                    ArtifactLink.project_id == project_id,
                    ArtifactLink.target_artifact_id == artifact_id,
                    Artifact.status != ArtifactStatus.ARCHIVED,
                )
            )
        ).scalars()
        child_dependents = (
            await self.db.execute(
                select(Artifact.id).where(
                    Artifact.project_id == project_id,
                    Artifact.parent_id == artifact_id,
                    Artifact.status != ArtifactStatus.ARCHIVED,
                )
            )
        ).scalars()
        based_on_dependents: list[uuid.UUID] = []
        rows = await self._project_artifacts_with_current_versions(project_id)
        target = str(artifact_id)
        for row in rows:
            if row.id == artifact_id or row.status == ArtifactStatus.ARCHIVED:
                continue
            if target in _based_on_from_artifact(row):
                based_on_dependents.append(row.id)
        dependent_ids = {item for item in linked_dependents} | {item for item in child_dependents}
        dependent_ids.update(based_on_dependents)
        return sorted(dependent_ids, key=str)

    async def create_link(
        self,
        *,
        project_id: uuid.UUID,
        source_artifact_id: uuid.UUID,
        target_artifact_id: uuid.UUID,
        relation_type: RelationType,
        created_by_id: uuid.UUID | None = None,
    ) -> ArtifactLink:
        if source_artifact_id == target_artifact_id:
            raise ValueError("Artifact cannot link to itself")
        await self._require_project_member(project_id, created_by_id)

        result = await self.db.execute(
            select(Artifact).where(Artifact.id.in_([source_artifact_id, target_artifact_id]))
        )
        artifacts = {artifact.id: artifact for artifact in result.scalars()}
        source = artifacts.get(source_artifact_id)
        target = artifacts.get(target_artifact_id)
        if source is None or target is None:
            raise ValueError("Artifact to link was not found")
        if source.project_id != project_id or target.project_id != project_id:
            raise ValueError("Artifact link must stay within the same project")
        duplicate = await self.db.execute(
            select(ArtifactLink.id).where(
                ArtifactLink.project_id == project_id,
                or_(
                    (
                        (ArtifactLink.source_artifact_id == source_artifact_id)
                        & (ArtifactLink.target_artifact_id == target_artifact_id)
                    ),
                    (
                        (ArtifactLink.source_artifact_id == target_artifact_id)
                        & (ArtifactLink.target_artifact_id == source_artifact_id)
                    ),
                ),
            )
        )
        if duplicate.scalar_one_or_none() is not None:
            raise ValueError("Artifact link already exists")
        if await self._has_link_path(project_id, source_artifact_id, target_artifact_id):
            raise ValueError("Artifact link creates a cycle")

        link = ArtifactLink(
            project_id=project_id,
            source_artifact_id=source_artifact_id,
            target_artifact_id=target_artifact_id,
            relation_type=relation_type,
            created_by_id=created_by_id,
        )
        self.db.add(link)
        await self.db.flush()
        return link

    async def _has_link_path(
        self,
        project_id: uuid.UUID,
        source_artifact_id: uuid.UUID,
        target_artifact_id: uuid.UUID,
    ) -> bool:
        """True if a path target -> ... -> source already exists; adding source -> target would cycle.

        Best-effort: read-then-insert with no lock, so two complementary concurrent inserts can both
        pass. The agent creates links sequentially within a turn, so the real-world race risk is low.
        """
        rows = (
            await self.db.execute(
                select(ArtifactLink.source_artifact_id, ArtifactLink.target_artifact_id).where(
                    ArtifactLink.project_id == project_id
                )
            )
        ).all()
        adjacency: dict[uuid.UUID, list[uuid.UUID]] = {}
        for source_id, target_id in rows:
            adjacency.setdefault(source_id, []).append(target_id)
        visited: set[uuid.UUID] = set()
        queue = [target_artifact_id]
        while queue:
            current = queue.pop(0)
            if current == source_artifact_id:
                return True
            if current in visited:
                continue
            visited.add(current)
            queue.extend(adjacency.get(current, []))
        return False

    async def lifecycle_reports_by_id(self, *, project_id: uuid.UUID) -> dict[uuid.UUID, dict[str, Any]]:
        artifacts = await self._project_artifacts_with_current_versions(project_id)
        artifacts_by_type: dict[str, list[Artifact]] = {}
        artifacts_by_id = {artifact.id: artifact for artifact in artifacts}
        for artifact in artifacts:
            artifacts_by_type.setdefault(_artifact_type(artifact.type), []).append(artifact)

        reports: dict[uuid.UUID, dict[str, Any]] = {}
        for artifact in artifacts:
            artifact_type = _artifact_type(artifact.type)
            based_on = _based_on_from_artifact(artifact)
            snapshot = ArtifactLifecycleSnapshot(
                artifact_type=artifact_type,
                artifact_id=str(artifact.id),
                status=_enum_value(artifact.status),
                current_version_id=str(artifact.current_version_id) if artifact.current_version_id else None,
                based_on=based_on,
            )
            verdict = resolve_lifecycle(
                artifact_type,
                snapshot,
                required_predecessors=self._structural_predecessors(artifact_type, artifacts_by_type),
                based_on_predecessors=self._based_on_predecessors(based_on, artifacts_by_id),
            )
            reports[artifact.id] = _lifecycle_report_from_verdict(verdict)
        return reports

    async def _project_artifacts_with_current_versions(self, project_id: uuid.UUID) -> list[Artifact]:
        return list(
            (
                await self.db.execute(
                    select(Artifact)
                    .where(Artifact.project_id == project_id)
                    .options(selectinload(Artifact.current_version))
                    .order_by(Artifact.created_at, Artifact.id)
                )
            )
            .scalars()
            .all()
        )

    def _structural_predecessors(
        self,
        artifact_type: str,
        artifacts_by_type: dict[str, list[Artifact]],
    ) -> list[StructuralPredecessorSnapshot]:
        snapshots: list[StructuralPredecessorSnapshot] = []
        for predecessor_type in ancestor_types(artifact_type):
            accepted = next(
                (
                    item
                    for item in artifacts_by_type.get(predecessor_type, [])
                    if item.status == ArtifactStatus.ACCEPTED and item.current_version_id is not None
                ),
                None,
            )
            if accepted is None:
                snapshots.append(StructuralPredecessorSnapshot(artifact_type=predecessor_type, found=False))
                continue
            snapshots.append(
                StructuralPredecessorSnapshot(
                    artifact_type=predecessor_type,
                    artifact_id=str(accepted.id),
                    status=_enum_value(accepted.status),
                    current_version_id=str(accepted.current_version_id) if accepted.current_version_id else None,
                    found=True,
                )
            )
        return snapshots

    def _based_on_predecessors(
        self,
        based_on: dict[str, str],
        artifacts_by_id: dict[uuid.UUID, Artifact],
    ) -> dict[str, BasedOnPredecessorSnapshot]:
        predecessors: dict[str, BasedOnPredecessorSnapshot] = {}
        for raw_id in sorted(based_on):
            try:
                artifact_id = uuid.UUID(str(raw_id))
            except (TypeError, ValueError):
                predecessors[str(raw_id)] = BasedOnPredecessorSnapshot(artifact_id=str(raw_id), found=False)
                continue
            predecessor = artifacts_by_id.get(artifact_id)
            if predecessor is None:
                predecessors[str(raw_id)] = BasedOnPredecessorSnapshot(artifact_id=str(raw_id), found=False)
                continue
            predecessors[str(raw_id)] = BasedOnPredecessorSnapshot(
                artifact_id=str(predecessor.id),
                artifact_type=_artifact_type(predecessor.type),
                status=_enum_value(predecessor.status),
                current_version_id=str(predecessor.current_version_id) if predecessor.current_version_id else None,
                found=True,
                retired=predecessor.status == ArtifactStatus.ARCHIVED,
            )
        return predecessors

    def link_to_response(self, link: ArtifactLink) -> ArtifactLinkResponse:
        return ArtifactLinkResponse(
            id=link.id,
            project_id=link.project_id,
            source_artifact_id=link.source_artifact_id,
            target_artifact_id=link.target_artifact_id,
            relation_type=link.relation_type,
            created_by_id=link.created_by_id,
            metadata=link.extra_metadata or {},
            created_at=link.created_at,
        )

    def evidence_to_response(self, evidence: ArtifactEvidence) -> ArtifactEvidenceResponse:
        return ArtifactEvidenceResponse(
            id=evidence.id,
            artifact_id=evidence.artifact_id,
            artifact_version_id=evidence.artifact_version_id,
            source_document_id=evidence.source_document_id,
            source_type=evidence.source_type,
            locator=evidence.locator,
            excerpt=evidence.excerpt,
            confidence=evidence.confidence,
            metadata=evidence.extra_metadata or {},
            created_at=evidence.created_at,
        )

    async def to_response(
        self,
        artifact: Artifact,
        lifecycle_report: dict[str, Any] | None = None,
    ) -> ArtifactResponse:
        version = None
        if artifact.current_version_id is not None:
            current = await self.db.get(ArtifactVersion, artifact.current_version_id)
            if current is not None:
                version = await self.version_to_response(current, artifact_type=artifact.type.value)
        if lifecycle_report is None:
            lifecycle_report = (await self.lifecycle_reports_by_id(project_id=artifact.project_id)).get(artifact.id)
        return ArtifactResponse(
            id=artifact.id,
            project_id=artifact.project_id,
            parent_id=artifact.parent_id,
            current_version_id=artifact.current_version_id,
            type=artifact.type,
            status=artifact.status,
            priority=artifact.priority,
            code=artifact.code,
            title=artifact.title,
            confidence=artifact.confidence,
            created_by_id=artifact.created_by_id,
            created_at=artifact.created_at,
            metadata=artifact.extra_metadata or {},
            current_version=version,
            lifecycle_state=(lifecycle_report or {}).get("state"),
            lifecycle_reason=(lifecycle_report or {}).get("reason"),
        )

    async def version_to_response(
        self, version: ArtifactVersion, artifact_type: str | None = None
    ) -> ArtifactVersionResponse:
        if artifact_type is None:
            artifact_type = (
                await self.db.execute(select(Artifact.type).where(Artifact.id == version.artifact_id))
            ).scalar_one_or_none()
            artifact_type = artifact_type.value if artifact_type is not None else None
        review_result = await self.db.execute(
            select(ArtifactReview.review_status)
            .where(ArtifactReview.artifact_version_id == version.id)
            .order_by(ArtifactReview.created_at.desc(), ArtifactReview.id.desc())
            .limit(1)
        )
        return ArtifactVersionResponse(
            id=version.id,
            artifact_id=version.artifact_id,
            version_number=version.version_number,
            title=version.title,
            body=version.body,
            status=version.status,
            parent_version_id=version.parent_version_id,
            change_source=version.change_source,
            change_summary=version.change_summary,
            review_status=review_result.scalar_one_or_none(),
            created_by_id=version.created_by_id,
            created_at=version.created_at,
            metadata=version.extra_metadata or {},
        )

    async def _get_project_artifact(self, project_id: uuid.UUID, artifact_id: uuid.UUID) -> Artifact:
        result = await self.db.execute(
            select(Artifact).where(Artifact.id == artifact_id, Artifact.project_id == project_id)
        )
        artifact = result.scalar_one_or_none()
        if artifact is None:
            raise ValueError("Artifact not found")
        return artifact

    async def _get_current_version(self, artifact: Artifact) -> ArtifactVersion:
        if artifact.current_version_id is None:
            raise ValueError("Artifact has no current version")
        current = await self.db.get(ArtifactVersion, artifact.current_version_id)
        if current is None:
            raise ValueError("Current version not found")
        return current

    async def _require_project_member(self, project_id: uuid.UUID, user_id: uuid.UUID | None) -> None:
        if user_id is None:
            raise PermissionError("User does not have access to the project")
        result = await self.db.execute(select(Project.org_id).where(Project.id == project_id))
        org_id = result.scalar_one_or_none()
        if org_id is None:
            raise ValueError("Project not found")
        member = await self.db.execute(
            select(OrgMember.id).where(OrgMember.org_id == org_id, OrgMember.user_id == user_id)
        )
        if member.scalar_one_or_none() is None:
            raise PermissionError("User does not have access to the project")

    def _validate_status_transition(self, current: ArtifactStatus, target: ArtifactStatus) -> None:
        if current == target:
            return
        if target not in ALLOWED_STATUS_TRANSITIONS[current]:
            raise InvalidArtifactStatusTransition(
                f"Cannot transition artifact status from {current.value} sang {target.value}"
            )


class ArtifactVersionService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.artifacts = ArtifactService(db)

    def _apply_review_status_side_effects(
        self,
        *,
        artifact: Artifact,
        version: ArtifactVersion,
        review_status: ReviewStatus,
    ) -> None:
        # A review is a distinct status-transition authority from update(): approving a DRAFT jumps it
        # straight to ACCEPTED, which the update() edit matrix intentionally forbids. ARCHIVED/ACCEPTED
        # stay terminal here — a later review row is kept for audit but must not drive status. That
        # no-op is deliberate (see test_review_approve_is_terminal_and_later_reject_is_audit_only); it
        # is logged rather than silent so reviewers can see the outcome had no status effect.
        version_terminal = version.status in {VersionStatus.ACCEPTED, VersionStatus.ARCHIVED}
        artifact_terminal = artifact.status in {ArtifactStatus.ACCEPTED, ArtifactStatus.ARCHIVED}
        if review_status == ReviewStatus.APPROVED:
            if version.status != VersionStatus.ARCHIVED:
                version.status = VersionStatus.ACCEPTED
            if artifact.status != ArtifactStatus.ARCHIVED:
                artifact.status = ArtifactStatus.ACCEPTED
                artifact.current_version_id = version.id
                artifact.title = version.title
            else:
                _log_terminal_review_noop(artifact, version, review_status)
            return
        if version_terminal:
            _log_terminal_review_noop(artifact, version, review_status)
            return
        version.status = VersionStatus.REJECTED
        if artifact.current_version_id != version.id or artifact_terminal:
            _log_terminal_review_noop(artifact, version, review_status)
            return
        if review_status == ReviewStatus.REJECTED:
            artifact.status = ArtifactStatus.REJECTED
        elif review_status == ReviewStatus.CHANGES_REQUESTED:
            artifact.status = ArtifactStatus.NEEDS_CLARIFICATION

    async def review(
        self,
        *,
        project_id: uuid.UUID,
        artifact_id: uuid.UUID,
        version_id: uuid.UUID,
        body: ArtifactReviewRequest,
        reviewed_by_id: uuid.UUID,
    ) -> ArtifactReviewResponse:
        await self.artifacts._require_project_member(project_id, reviewed_by_id)
        artifact = await self.artifacts._get_project_artifact(project_id, artifact_id)
        version = await self.db.get(ArtifactVersion, version_id)
        if version is None or version.artifact_id != artifact.id:
            raise ValueError("Artifact version not found")
        review = ArtifactReview(
            artifact_id=artifact.id,
            artifact_version_id=version.id,
            reviewed_by_id=reviewed_by_id,
            review_status=body.review_status,
            comment=body.comment,
        )
        self.db.add(review)
        self._apply_review_status_side_effects(
            artifact=artifact,
            version=version,
            review_status=body.review_status,
        )
        await self.db.flush()
        await self.db.refresh(review)
        return ArtifactReviewResponse(
            id=review.id,
            artifact_id=review.artifact_id,
            artifact_version_id=review.artifact_version_id,
            reviewed_by_id=review.reviewed_by_id,
            review_status=review.review_status,
            comment=review.comment,
            created_at=review.created_at,
        )

    async def restore(
        self,
        *,
        project_id: uuid.UUID,
        artifact_id: uuid.UUID,
        version_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> ArtifactResponse:
        return await self.artifacts.restore(
            project_id=project_id,
            artifact_id=artifact_id,
            version_id=version_id,
            user_id=user_id,
        )


class ArtifactLinkService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.artifacts = ArtifactService(db)

    async def create(
        self,
        *,
        project_id: uuid.UUID,
        body: ArtifactLinkCreateRequest,
        created_by_id: uuid.UUID,
    ) -> ArtifactLinkResponse:
        link = await self.artifacts.create_link(
            project_id=project_id,
            source_artifact_id=body.source_artifact_id,
            target_artifact_id=body.target_artifact_id,
            relation_type=body.relation_type,
            created_by_id=created_by_id,
        )
        link.extra_metadata = body.metadata
        await self.db.flush()
        return self.artifacts.link_to_response(link)

    async def delete(self, *, project_id: uuid.UUID, link_id: uuid.UUID, user_id: uuid.UUID) -> None:
        await self.artifacts._require_project_member(project_id, user_id)
        link = await self.db.get(ArtifactLink, link_id)
        if link is None or link.project_id != project_id:
            raise ValueError("Artifact not found link")
        await self.db.delete(link)
        await self.db.flush()

    async def graph(self, *, project_id: uuid.UUID, user_id: uuid.UUID) -> ArtifactGraphResponse:
        await self.artifacts._require_project_member(project_id, user_id)
        artifact_rows = (
            (
                await self.db.execute(
                    select(Artifact).where(Artifact.project_id == project_id).order_by(Artifact.created_at, Artifact.id)
                )
            )
            .scalars()
            .all()
        )
        link_rows = (
            (
                await self.db.execute(
                    select(ArtifactLink).where(ArtifactLink.project_id == project_id).order_by(ArtifactLink.created_at)
                )
            )
            .scalars()
            .all()
        )

        lifecycle_reports = await self.artifacts.lifecycle_reports_by_id(project_id=project_id)
        nodes = [
            ArtifactNode(
                id=artifact.id,
                type=artifact.type,
                status=artifact.status,
                title=artifact.title,
                current_version_id=artifact.current_version_id,
                current_version=await self.artifacts.version_to_response(
                    await self.artifacts._get_current_version(artifact)
                )
                if artifact.current_version_id
                else None,
                lifecycle_state=(lifecycle_reports.get(artifact.id) or {}).get("state"),
                lifecycle_reason=(lifecycle_reports.get(artifact.id) or {}).get("reason"),
            )
            for artifact in artifact_rows
        ]
        return ArtifactGraphResponse(
            nodes=nodes,
            links=[self.artifacts.link_to_response(link) for link in link_rows],
            warnings=self._build_warnings(artifact_rows, link_rows),
        )

    def _build_warnings(self, artifacts: list[Artifact], links: list[ArtifactLink]) -> list[GraphWarning]:
        warnings: list[GraphWarning] = []
        linked_ids = {link.source_artifact_id for link in links} | {link.target_artifact_id for link in links}
        artifacts_by_id = {artifact.id: artifact for artifact in artifacts}

        for artifact in artifacts:
            if artifact.id not in linked_ids:
                warnings.append(GraphWarning(type="orphan_artifact", artifact_id=artifact.id))
            if artifact.status.value == "needs_clarification":
                warnings.append(GraphWarning(type="needs_clarification", artifact_id=artifact.id))
            if artifact.type.value in {"functional_requirement", "story"} and not self._has_upstream_trace(
                artifact, links, artifacts_by_id
            ):
                warnings.append(GraphWarning(type="missing_upstream_trace", artifact_id=artifact.id))

        for link in links:
            if link.relation_type.value == "conflicts_with":
                warnings.append(GraphWarning(type="conflicting_artifacts", artifact_id=link.source_artifact_id))
                warnings.append(GraphWarning(type="conflicting_artifacts", artifact_id=link.target_artifact_id))
        return warnings

    def _has_upstream_trace(
        self,
        artifact: Artifact,
        links: list[ArtifactLink],
        artifacts_by_id: dict[uuid.UUID, Artifact],
    ) -> bool:
        upstream_relations = {"satisfies", "derives_from"}
        upstream_types = {"brd"}
        for link in links:
            if link.source_artifact_id != artifact.id or link.relation_type.value not in upstream_relations:
                continue
            target = artifacts_by_id.get(link.target_artifact_id)
            if target is not None and target.type.value in upstream_types:
                return True
        return False


class SourceDocumentInUseError(Exception):
    def __init__(self, artifact_ids: list[uuid.UUID]):
        self.artifact_ids = artifact_ids
        super().__init__("Source document is being used as evidence")


class SourceDocumentService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.artifacts = ArtifactService(db)

    async def upload(
        self,
        *,
        project_id: uuid.UUID,
        body: SourceDocumentCreateRequest,
        uploaded_by_id: uuid.UUID,
    ) -> SourceDocumentResponse:
        await self.artifacts._require_project_member(project_id, uploaded_by_id)
        content = body.content_text or ""
        document = SourceDocument(
            project_id=project_id,
            uploaded_by_id=uploaded_by_id,
            title=body.title,
            source_type=body.source_type,
            locator=body.locator,
            content_text=body.content_text,
            content_hash=sha256(content.encode("utf-8")).hexdigest() if body.content_text is not None else None,
            mime_type=body.mime_type,
            size_bytes=len(content.encode("utf-8")) if body.content_text is not None else None,
            extra_metadata=body.metadata,
        )
        self.db.add(document)
        await self.db.flush()
        return self.to_response(document)

    async def delete(
        self,
        *,
        project_id: uuid.UUID,
        source_document_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> None:
        await self.artifacts._require_project_member(project_id, user_id)
        document = await self.db.get(SourceDocument, source_document_id)
        if document is None or document.project_id != project_id:
            raise ValueError("Source document not found")
        evidence = await self.db.execute(
            select(ArtifactEvidence.artifact_id).where(ArtifactEvidence.source_document_id == source_document_id)
        )
        artifact_ids = list(evidence.scalars().all())
        if artifact_ids:
            raise SourceDocumentInUseError(artifact_ids)
        await self.db.delete(document)
        await self.db.flush()

    def to_response(self, document: SourceDocument) -> SourceDocumentResponse:
        return SourceDocumentResponse(
            id=document.id,
            project_id=document.project_id,
            uploaded_by_id=document.uploaded_by_id,
            title=document.title,
            source_type=document.source_type,
            locator=document.locator,
            content_text=document.content_text,
            content_hash=document.content_hash,
            mime_type=document.mime_type,
            size_bytes=document.size_bytes,
            metadata=document.extra_metadata or {},
            created_at=document.created_at,
        )
