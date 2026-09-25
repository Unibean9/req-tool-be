"""Load and index the stored BRD/PRD components that feed use-case generation.

There is intentionally no filesystem read here.  The async loader uses ``DocumentService`` and
the same document registry that the existing BRD/PRD UI uses.  The pure adapter is also useful for
tests and for a future API that already has ``DocumentView`` payloads in memory.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.artifact import ArtifactVersion
from app.schemas.document import DocumentItemView, DocumentView
from app.services.document_service import DocumentService
from app.use_cases.models import (
    RequirementsSourceSnapshot,
    SourceEvidence,
    StoredComponentSnapshot,
    StoredDocumentSnapshot,
)

_ENTITY_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?P<id>"
    # BRD rules use a domain letter (BR-R1, BR-D3, BR-L4) as well as the
    # older numeric form.  Keep both forms source-backed so relationship
    # citations can resolve to the exact stored rule row.
    r"BR-(?:[A-Z]{1,4}\d{1,3}|\d{1,3})|BRule-\d{1,3}|BC-\d{1,3}|"
    r"FR-[A-Za-z0-9-]*\d{1,3}|NFR-\d{1,3}|"
    r"PG-\d{1,3}|BO-\d{1,3}|KPI-\d{1,3})"
    r"(?:\s*[:—-]\s*|\s+)(?P<name>[^|\n]+)?",
    re.IGNORECASE,
)
_HEADING_RE = re.compile(r"^(?P<marks>#{1,6})\s+(?P<title>.+?)\s*$")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)+\|?\s*$")
_FEATURE_HEADING_RE = re.compile(
    r"^Feature\s+Module\s+(?P<code>[A-Z]{1,3})\s*[—-]\s*(?P<name>.+)$", re.IGNORECASE
)
_ROLE_KEY_RE = re.compile(
    r"^\s*(?:[-*]\s*)?\*{0,2}(?P<key>primary_user|secondary_stakeholders|decision_maker|operator|user_segment|"
    r"target_user|target_users|actor|actors|role|roles)\*{0,2}\s*:\s*(?P<value>.+?)\s*$",
    re.IGNORECASE,
)
_ROLE_HEADING_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*\.\s*)?(?P<name>[A-Z][A-Za-z0-9 /&()]+?)(?:\s+[—-].*)?$"
)
_BULLET_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)(?P<text>.+?)\s*$")

_OUT_OF_SCOPE_MARKERS = ("out of scope", "non-goal", "non goal", "excluded", "not in scope")
_ACTOR_CONTEXT_MARKERS = ("stakeholder", "target user", "target users", "role", "persona", "actor")
_INTERNAL_ACTOR_TOKENS = (
    "database",
    "backend",
    "frontend",
    "server",
    "api",
    "ai model",
    "ai agent",
    "engine",
    "module",
)


def _as_document(value: DocumentView | Mapping[str, Any]) -> DocumentView:
    if isinstance(value, DocumentView):
        return value
    return DocumentView.model_validate(value)


def _component_snapshot(document_type: str, item: DocumentItemView) -> StoredComponentSnapshot:
    version = item.current_version
    return StoredComponentSnapshot(
        document_type=document_type,  # type: ignore[arg-type]
        artifact_type=item.artifact_type.value,
        label=item.label,
        description=item.description,
        artifact_id=str(item.artifact_id) if item.artifact_id else None,
        parent_id=str(item.parent_id) if item.parent_id else None,
        status=item.status.value if item.status is not None else None,
        priority=item.priority.value if item.priority is not None else None,
        code=item.code,
        title=item.title,
        confidence=float(item.confidence) if item.confidence is not None else None,
        current_version_id=str(item.current_version_id) if item.current_version_id else None,
        version_number=version.version_number if version is not None else None,
        body=version.body if version is not None else "",
        metadata=item.metadata if isinstance(item.metadata, dict) else {},
    )


def _document_snapshot(
    document: DocumentView | Mapping[str, Any],
    *,
    container_body: str | None = None,
) -> StoredDocumentSnapshot:
    view = _as_document(document)
    document_type = view.document_type.value
    if document_type not in {"brd", "prd"}:
        raise ValueError(f"Use-case source must be a BRD or PRD document, got {document_type!r}")
    return StoredDocumentSnapshot(
        document_type=document_type,  # type: ignore[arg-type]
        label=view.label,
        description=view.description,
        artifact_id=str(view.artifact_id) if view.artifact_id else None,
        project_id=str(view.project_id),
        status=view.status.value if view.status is not None else None,
        title=view.title,
        current_version_id=str(view.current_version_id) if view.current_version_id else None,
        container_body=container_body or "",
        components=[_component_snapshot(document_type, item) for item in view.items],
    )


def source_snapshot_from_documents(
    brd: DocumentView | Mapping[str, Any],
    prd: DocumentView | Mapping[str, Any],
    *,
    container_bodies: Mapping[str, str] | None = None,
) -> RequirementsSourceSnapshot:
    """Build a complete source snapshot from stored document views.

    ``brd`` and ``prd`` are expected to be the current project documents, not repository fixtures.
    Every registry item is retained, including an unversioned/missing slot, so the agent can state
    that evidence is missing instead of silently filling it with an invented capability.
    """

    body_map = container_bodies or {}
    brd_snapshot = _document_snapshot(brd, container_body=body_map.get("brd"))
    prd_snapshot = _document_snapshot(prd, container_body=body_map.get("prd"))
    if brd_snapshot.project_id != prd_snapshot.project_id:
        raise ValueError("BRD and PRD must belong to the same project")

    components = [*brd_snapshot.components, *prd_snapshot.components]
    evidence = _extract_evidence(brd_snapshot, prd_snapshot)
    digest_input = "\n\n".join(
        (
            brd_snapshot.model_dump_json(),
            prd_snapshot.model_dump_json(),
            "\n".join(item.model_dump_json() for item in evidence),
        )
    )
    source_hash = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()
    return RequirementsSourceSnapshot(
        project_id=brd_snapshot.project_id,
        brd=brd_snapshot,
        prd=prd_snapshot,
        components=components,
        evidence=evidence,
        source_hash=source_hash,
    )


async def load_project_requirements_source(
    db: AsyncSession,
    *,
    project_id,
) -> RequirementsSourceSnapshot:
    """Read the current BRD/PRD containers and all registry children for one project.

    This is deliberately a core service function rather than a router.  A future API can call it,
    while the agent harness and validator remain usable without HTTP or persistence side effects.
    """

    service = DocumentService(db)
    brd = await service.get_document(project_id=project_id, document_type="brd")
    prd = await service.get_document(project_id=project_id, document_type="prd")
    container_bodies: dict[str, str] = {}
    for document_type, document in (("brd", brd), ("prd", prd)):
        if document.artifact_id is None or document.current_version_id is None:
            continue
        body = await db.scalar(
            select(ArtifactVersion.body).where(
                ArtifactVersion.id == document.current_version_id,
                ArtifactVersion.artifact_id == document.artifact_id,
            )
        )
        if body:
            container_bodies[document_type] = body
    return source_snapshot_from_documents(brd, prd, container_bodies=container_bodies)


def _all_source_units(document: StoredDocumentSnapshot) -> list[tuple[str, str, str]]:
    units: list[tuple[str, str, str]] = []
    units.append((document.document_type, document.document_type, document.container_body))
    for component in document.components:
        units.append((document.document_type, component.artifact_type, component.body))
    return units


def _extract_evidence(
    brd: StoredDocumentSnapshot,
    prd: StoredDocumentSnapshot,
) -> list[SourceEvidence]:
    evidence: list[SourceEvidence] = []
    seen: set[str] = set()

    def add(item: SourceEvidence) -> None:
        if item.evidence_id in seen:
            return
        seen.add(item.evidence_id)
        evidence.append(item)

    for document in (brd, prd):
        for document_type, artifact_type, body in _all_source_units(document):
            component = next(
                (
                    item
                    for item in document.components
                    if item.artifact_type == artifact_type
                ),
                None,
            )
            artifact_id = component.artifact_id if component else document.artifact_id
            version_id = component.current_version_id if component else document.current_version_id
            component_id = f"component:{document_type}:{artifact_type}"
            add(
                SourceEvidence(
                    evidence_id=component_id,
                    document_type=document_type,  # type: ignore[arg-type]
                    artifact_type=artifact_type,
                    artifact_id=artifact_id,
                    version_id=version_id,
                    kind="component",
                    locator=f"{document_type}.{artifact_type}",
                    excerpt=_excerpt(body or "(missing body)"),
                )
            )
            _extract_headings(
                add,
                document_type=document_type,
                artifact_type=artifact_type,
                artifact_id=artifact_id,
                version_id=version_id,
                body=body,
            )
            _extract_entities(
                add,
                document_type=document_type,
                artifact_type=artifact_type,
                artifact_id=artifact_id,
                version_id=version_id,
                body=body,
            )
            _extract_scope_and_roles(
                add,
                document_type=document_type,
                artifact_type=artifact_type,
                artifact_id=artifact_id,
                version_id=version_id,
                body=body,
            )
    return evidence


def _extract_headings(
    add,
    *,
    document_type: str,
    artifact_type: str,
    artifact_id: str | None,
    version_id: str | None,
    body: str,
) -> None:
    for line_number, line in enumerate(body.splitlines(), start=1):
        match = _HEADING_RE.match(line.strip())
        if not match:
            continue
        title = match.group("title").strip()
        if not title:
            continue
        context = title.lower()
        kind: str = "heading"
        if any(marker in context for marker in _OUT_OF_SCOPE_MARKERS):
            kind = "out_of_scope"
        elif any(marker in context for marker in ("scope", "capabilit", "business process")):
            kind = "scope"
        add(
            SourceEvidence(
                evidence_id=f"heading:{document_type}:{artifact_type}:{line_number}",
                document_type=document_type,  # type: ignore[arg-type]
                artifact_type=artifact_type,
                artifact_id=artifact_id,
                version_id=version_id,
                kind=kind,  # type: ignore[arg-type]
                locator=f"{document_type}.{artifact_type}:L{line_number}",
                excerpt=title,
                entity_name=title,
            )
        )


def _extract_entities(
    add,
    *,
    document_type: str,
    artifact_type: str,
    artifact_id: str | None,
    version_id: str | None,
    body: str,
) -> None:
    for line_number, line in enumerate(body.splitlines(), start=1):
        matches = list(_ENTITY_RE.finditer(line))
        if not matches:
            heading = _HEADING_RE.match(line.strip())
            if not heading:
                continue
            feature = _FEATURE_HEADING_RE.match(heading.group("title").strip())
            if feature:
                subsystem_id = f"FM-{feature.group('code').upper()}"
                subsystem_name = _clean_entity_name(feature.group("name"))
                add(
                    SourceEvidence(
                        evidence_id=f"subsystem:{document_type}:{artifact_type}:{subsystem_id}:{line_number}",
                        document_type=document_type,  # type: ignore[arg-type]
                        artifact_type=artifact_type,
                        artifact_id=artifact_id,
                        version_id=version_id,
                        kind="subsystem",
                        locator=f"{document_type}.{artifact_type}:L{line_number}",
                        excerpt=_excerpt(heading.group("title")),
                        entity_id=subsystem_id,
                        entity_name=subsystem_name,
                    )
                )
            continue
        for match in matches:
            entity_id = match.group("id").upper()
            raw_name = _clean_entity_name(match.group("name") or "")
            if not raw_name:
                raw_name = _clean_entity_name(line.replace(match.group("id"), "", 1))
            kind = _entity_kind(entity_id, artifact_type)
            evidence_id = f"entity:{document_type}:{artifact_type}:{entity_id}:{line_number}"
            add(
                SourceEvidence(
                    evidence_id=evidence_id,
                    document_type=document_type,  # type: ignore[arg-type]
                    artifact_type=artifact_type,
                    artifact_id=artifact_id,
                    version_id=version_id,
                    kind=kind,
                    locator=f"{document_type}.{artifact_type}:L{line_number}",
                    excerpt=_excerpt(line),
                    entity_id=entity_id,
                    entity_name=raw_name or None,
                )
            )
            if entity_id.startswith("BC-"):
                add(
                    SourceEvidence(
                        evidence_id=f"subsystem:{document_type}:{artifact_type}:{entity_id}:{line_number}",
                        document_type=document_type,  # type: ignore[arg-type]
                        artifact_type=artifact_type,
                        artifact_id=artifact_id,
                        version_id=version_id,
                        kind="subsystem",
                        locator=f"{document_type}.{artifact_type}:L{line_number}",
                        excerpt=_excerpt(line),
                        entity_id=entity_id,
                        entity_name=raw_name or None,
                    )
                )


def _extract_scope_and_roles(
    add,
    *,
    document_type: str,
    artifact_type: str,
    artifact_id: str | None,
    version_id: str | None,
    body: str,
) -> None:
    heading_context: list[str] = []
    for line_number, line in enumerate(body.splitlines(), start=1):
        heading = _HEADING_RE.match(line.strip())
        if heading:
            depth = len(heading.group("marks"))
            heading_context = heading_context[: depth - 1]
            heading_context.append(heading.group("title").strip())
            continue
        normalized_context = " ".join(heading_context).lower()
        stripped = line.strip()
        if not stripped:
            continue
        role_match = _ROLE_KEY_RE.match(stripped)
        if role_match and (artifact_type in {"stakeholder_register", "use_case"} or "role" in normalized_context):
            for role in _split_names(role_match.group("value")):
                if _looks_like_actor(role):
                    add(
                        SourceEvidence(
                            evidence_id=f"actor:{document_type}:{artifact_type}:{line_number}:{_slug(role)}",
                            document_type=document_type,  # type: ignore[arg-type]
                            artifact_type=artifact_type,
                            artifact_id=artifact_id,
                            version_id=version_id,
                            kind="actor",
                            locator=f"{document_type}.{artifact_type}:L{line_number}",
                            excerpt=_excerpt(line),
                            entity_name=role,
                        )
                    )
        if any(marker in normalized_context for marker in _ACTOR_CONTEXT_MARKERS):
            heading_match = _ROLE_HEADING_RE.match(stripped.lstrip("# "))
            if heading_match and len(stripped) < 130 and not stripped.startswith(("Objective", "User Stories")):
                candidate = _clean_entity_name(heading_match.group("name"))
                if _looks_like_actor(candidate):
                    add(
                        SourceEvidence(
                            evidence_id=f"actor:{document_type}:{artifact_type}:{line_number}:{_slug(candidate)}",
                            document_type=document_type,  # type: ignore[arg-type]
                            artifact_type=artifact_type,
                            artifact_id=artifact_id,
                            version_id=version_id,
                            kind="actor",
                            locator=f"{document_type}.{artifact_type}:L{line_number}",
                            excerpt=_excerpt(line),
                            entity_name=candidate,
                        )
                    )
        bullet = _BULLET_RE.match(stripped)
        if bullet:
            text = bullet.group("text")
            if any(marker in normalized_context for marker in _OUT_OF_SCOPE_MARKERS):
                add(
                    SourceEvidence(
                        evidence_id=f"out-of-scope:{document_type}:{artifact_type}:{line_number}",
                        document_type=document_type,  # type: ignore[arg-type]
                        artifact_type=artifact_type,
                        artifact_id=artifact_id,
                        version_id=version_id,
                        kind="out_of_scope",
                        locator=f"{document_type}.{artifact_type}:L{line_number}",
                        excerpt=_excerpt(text),
                        entity_name=_clean_entity_name(text),
                    )
                )
            if any(marker in normalized_context for marker in ("traceability", "brd → prd", "brd to prd")):
                add(
                    SourceEvidence(
                        evidence_id=f"traceability:{document_type}:{artifact_type}:{line_number}",
                        document_type=document_type,  # type: ignore[arg-type]
                        artifact_type=artifact_type,
                        artifact_id=artifact_id,
                        version_id=version_id,
                        kind="traceability",
                        locator=f"{document_type}.{artifact_type}:L{line_number}",
                        excerpt=_excerpt(text),
                    )
                )

    for line_number, (headers, values) in enumerate(_markdown_table_rows(body), start=1):
        header_text = " ".join(headers).lower()
        row_text = " | ".join(values)
        if "priority" in header_text and any(prefix in row_text for prefix in ("BR-", "FR-", "BC-")):
            add(
                SourceEvidence(
                    evidence_id=f"priority:{document_type}:{artifact_type}:{line_number}:{_slug(row_text)}",
                    document_type=document_type,  # type: ignore[arg-type]
                    artifact_type=artifact_type,
                    artifact_id=artifact_id,
                    version_id=version_id,
                    kind="scope",
                    locator=f"{document_type}.{artifact_type}:table:{line_number}",
                    excerpt=_excerpt(row_text),
                )
            )


def _markdown_table_rows(body: str) -> list[tuple[list[str], list[str]]]:
    lines = body.splitlines()
    rows: list[tuple[list[str], list[str]]] = []
    for index, line in enumerate(lines[:-1]):
        if "|" not in line or "|" not in lines[index + 1]:
            continue
        headers = _split_table_line(line)
        if not headers or not _TABLE_SEPARATOR_RE.match(lines[index + 1]):
            continue
        row_index = index + 2
        while row_index < len(lines) and "|" in lines[row_index] and lines[row_index].strip():
            values = _split_table_line(lines[row_index])
            if values:
                rows.append((headers, values))
            row_index += 1
    return rows


def _split_table_line(line: str) -> list[str]:
    text = line.strip().strip("|")
    return [part.strip() for part in text.split("|")]


def _entity_kind(entity_id: str, artifact_type: str) -> str:
    if re.match(r"^BR-[A-Z]", entity_id):
        return "business_rule"
    if entity_id.startswith("BR-"):
        return "business_requirement"
    if entity_id.startswith("BRULE-"):
        return "business_rule"
    if entity_id.startswith("BC-"):
        return "business_capability"
    if entity_id.startswith("FR-"):
        return "functional_requirement"
    if entity_id.startswith("NFR-"):
        return "non_functional_requirement"
    if artifact_type == "business_rules":
        return "business_rule"
    return "workflow"


def canonical_evidence_refs(
    values: Any,
    source: RequirementsSourceSnapshot | None,
    *,
    fallback_internal: bool = False,
) -> list[str]:
    """Resolve API/LLM evidence labels back to stable source evidence IDs.

    The UI renders evidence as ``PRD · prd.functional_requirement:L11`` while
    the model stores IDs such as ``entity:prd:functional_requirement:FR-TM01:11``.
    Providers may append an explanation to that label, or cite a requirement
    code directly.  Validation must never treat either rendered form as a new
    evidence ID, so resolve the code/locator and discard anything unknown.
    """

    if isinstance(values, str):
        values = [values]
    refs = [str(value).strip() for value in values or [] if str(value).strip()]
    if not source:
        return refs or (["internal"] if fallback_internal else [])

    evidence = source.evidence_by_id()
    by_entity: dict[str, list[str]] = {}
    by_locator: dict[str, list[str]] = {}
    for item in source.evidence:
        if item.entity_id:
            by_entity.setdefault(item.entity_id.casefold(), []).append(item.evidence_id)
        by_locator.setdefault(item.locator.casefold(), []).append(item.evidence_id)

    def choose(candidates: list[str]) -> str | None:
        if not candidates:
            return None
        # Prefer an entity row over a heading/component when a locator has
        # several evidence records on the same source line.
        priority = {
            "functional_requirement": 0,
            "non_functional_requirement": 0,
            "business_rule": 0,
            "business_requirement": 1,
            "business_capability": 1,
            "subsystem": 2,
            "heading": 3,
            "component": 4,
        }
        return min(candidates, key=lambda ref: priority.get(evidence[ref].kind, 5))

    output: list[str] = []
    seen: set[str] = set()
    for raw in refs:
        if raw in evidence or raw.startswith("sha256:") or raw == "internal":
            resolved = raw
        else:
            normalized = re.sub(r"\s+", " ", raw).strip()
            resolved = None
            # A provider often includes a requirement/rule code in the
            # explanation.  Prefer that precise entity over a line locator.
            codes = re.findall(r"\b(?:NFR|FR|BRule|BR|BC)-[A-Z0-9]+\b", normalized, flags=re.I)
            for code in codes:
                resolved = choose(by_entity.get(code.casefold(), []))
                if resolved:
                    break
            if resolved is None:
                # Match the canonical display label, including labels with an
                # appended explanation after an em dash.
                for locator in sorted(by_locator, key=len, reverse=True):
                    candidates = by_locator[locator]
                    if locator in normalized.casefold():
                        resolved = choose(candidates)
                        if resolved:
                            break
            if resolved is None:
                # ``brd.business_rules BR-R1`` does not use the UI separator;
                # the code pass above normally resolves it, but a plain
                # component citation is still useful when no code exists.
                for locator, candidates in by_locator.items():
                    if locator in normalized.casefold().replace(" · ", "."):
                        resolved = choose(candidates)
                        if resolved:
                            break
        if resolved and resolved not in seen:
            output.append(resolved)
            seen.add(resolved)
    if not output and fallback_internal:
        return ["internal"]
    return output


def _clean_entity_name(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("**", "").replace("`", "").strip(" :-—|"))


def _split_names(value: str) -> list[str]:
    cleaned = _clean_entity_name(value)
    cleaned = re.sub(r"\s+and\s+", ",", cleaned, flags=re.IGNORECASE)
    return [item.strip(" .") for item in re.split(r",|/", cleaned) if item.strip(" .")]


def _looks_like_actor(value: str) -> bool:
    candidate = _clean_entity_name(value)
    if not candidate or len(candidate) > 100:
        return False
    lowered = candidate.lower()
    if any(token in lowered for token in _INTERNAL_ACTOR_TOKENS):
        return False
    return not lowered.startswith(("the ", "a ", "an "))


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:60] or "item"


def _excerpt(value: str, limit: int = 1200) -> str:
    compact = value.strip()
    return compact if len(compact) <= limit else compact[: limit - 1].rstrip() + "…"
