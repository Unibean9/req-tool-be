"""Complete a use-case model from the stored BRD/PRD component snapshot.

The extractor supplies a deterministic, source-backed floor so a provider cannot silently omit a
functional-requirement family.  AI wording/detail is merged only onto rows whose IDs or names map
to that floor.  Modules are groups, never synthetic use cases.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass, field

from app.use_cases.models import (
    RequirementsSourceSnapshot,
    UseCaseActor,
    UseCaseEntry,
    UseCaseModel,
    UseCaseModule,
    UseCaseRelation,
    UseCaseRequirementLink,
    UseCaseSystem,
)
from app.use_cases.source_loader import _markdown_table_rows, canonical_evidence_refs

_BC_HEADING_RE = re.compile(r"^(?:#{2,6}\s*)?(?P<id>BC[- _]?\d+)\s*[:—-]\s*(?P<name>.+?)\s*$", re.I)
_FEATURE_HEADING_RE = re.compile(
    r"^(?:#{2,6}\s*)?Feature\s+Module\s+(?P<id>[A-Z]{1,3})\s*[—-]\s*(?P<name>.+?)\s*$", re.I
)
_CAPABILITY_SECTION_RE = re.compile(r"^business\s+capabilit(?:y|ies)\s*$", re.I)
_NUMBERED_HEADING_RE = re.compile(r"^(?:\d+[.)]\s*)?(?P<name>[A-Za-z][^:|]{1,155})$")
_FR_CODE_RE = re.compile(r"FR-(?P<family>[A-Z0-9][A-Z0-9-]*)-(?P<number>\d+)", re.I)
_FR_COMPACT_CODE_RE = re.compile(r"\bFR-(?P<family>[A-Z][A-Z0-9]*?)(?P<number>\d+)\b", re.I)
_SIMPLE_FR_CODE_RE = re.compile(r"\bFR-(?P<number>\d+)\b", re.I)
_FR_HEADING_RE = re.compile(r"^#{2,6}\s*(?P<code>FR-[A-Z0-9-]+)\s*[:—-]\s*(?P<title>.+)$", re.I)
_ROLE_KEY_RE = re.compile(r"(?:user[ _-]*segment|target[ _-]*users?|actors?|roles?)\s*:\s*(.+)$", re.I)
_BULLET_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)(?P<text>.+?)\s*$")
_GENERIC_ACTOR_ROLE = "Stakeholder"


@dataclass
class _Capability:
    id: str
    name: str
    goal: str = ""
    user_segment: list[str] = field(default_factory=list)
    source_refs: list[str] = field(default_factory=list)


@dataclass
class _RequirementFamily:
    bc_id: str
    family: str
    title: str
    requirement_codes: list[str] = field(default_factory=list)
    source_refs: list[str] = field(default_factory=list)
    priorities: list[str] = field(default_factory=list)
    description: str = ""
    requirement_titles: dict[str, str] = field(default_factory=dict)
    inferred: bool = False


def complete_use_case_table(source: RequirementsSourceSnapshot, candidate: UseCaseModel | None = None) -> UseCaseModel:
    """Return the complete source-backed canonical model.

    The stored component registry is the source of truth.  A candidate can improve wording and
    detail, but it cannot remove source-backed use cases, invent actors/modules, or inject a
    relation with an endpoint that is not in the completed table.
    """

    canonical = _build_source_model(source)
    if candidate is not None:
        _merge_candidate_details(canonical, candidate, source)
    _ensure_relationship_ids(canonical)
    return canonical


def _build_source_model(source: RequirementsSourceSnapshot) -> UseCaseModel:
    capabilities = _parse_capabilities(source)
    families = _parse_requirement_families(source)
    system_name = source.prd.title or source.brd.title or "Requirements System"
    if not capabilities:
        # A capability may be implicit in the FR table. Keep one source-backed module rather than
        # fabricating a module for an empty source.
        if families:
            refs = _fallback_component_refs(source)
            capabilities = [_Capability(id="BC-01", name="Requirements", goal="", source_refs=refs)]
    capability_by_id = {item.id.upper(): item for item in capabilities}
    modules: list[UseCaseModule] = []
    module_by_bc: dict[str, UseCaseModule] = {}
    for index, capability in enumerate(capabilities, start=1):
        bc_id = capability.id.upper()
        module = UseCaseModule(
            id=f"SUB-BC-{index:02d}",
            name=capability.name,
            goal=capability.goal or None,
            source_refs=capability.source_refs or _fallback_component_refs(source),
        )
        modules.append(module)
        module_by_bc[bc_id] = module

    actor_names: OrderedDict[str, str] = OrderedDict()
    actor_refs: dict[str, list[str]] = {}
    for capability in capabilities:
        for role in capability.user_segment:
            # Capability/user-segment text is source content, not a validated actor label.  Clean
            # inferred annotations before it reaches the Pydantic actor contract.
            role = _clean_actor_role(role)
            if not role:
                continue
            actor_id = _actor_id(role)
            if actor_id:
                actor_names.setdefault(actor_id, role)
                actor_refs.setdefault(
                    actor_id, _unique_refs(capability.source_refs or _fallback_component_refs(source))
                )
    if not actor_names:
        for role in _roles_from_stakeholders(source):
            role = _clean_actor_role(role)
            if not role:
                continue
            actor_id = _actor_id(role)
            actor_names.setdefault(actor_id, role)
            actor_refs.setdefault(actor_id, _fallback_component_refs(source))
    if not actor_names and families:
        # This is a conservative role label only when the source contains requirements but no
        # stakeholder role. It is marked inferred and will be visible for review.
        actor_names[_actor_id(_GENERIC_ACTOR_ROLE)] = _GENERIC_ACTOR_ROLE
        actor_refs[_actor_id(_GENERIC_ACTOR_ROLE)] = _fallback_component_refs(source)

    actors = [
        UseCaseActor(id=actor_id, name=name, kind="human", source_refs=actor_refs.get(actor_id, []))
        for actor_id, name in actor_names.items()
    ]
    actors_by_id = {actor.id: actor for actor in actors}

    use_cases: list[UseCaseEntry] = []
    for family in families:
        bc_id = family.bc_id.upper()
        module = module_by_bc.get(bc_id)
        if module is None:
            # Some sources omit the BC code in the capability section. Attach to the first
            # source-backed module only when the requirement itself has that same source evidence.
            module = modules[0] if modules else None
        if module is None:
            continue
        capability = capability_by_id.get(bc_id) or capabilities[0]
        roles = list(capability.user_segment) or list(actor_names.values())[:1] or [_GENERIC_ACTOR_ROLE]
        actor_ids = [_actor_id(role) for role in roles if _actor_id(role) in actors_by_id]
        if not actor_ids:
            continue
        primary_id, secondary_ids = actor_ids[0], actor_ids[1:]
        refs = _unique_refs(
            [*family.source_refs, *capability.source_refs, _component_ref(source, "prd", "functional_requirement")]
        )
        requirement_links = [
            UseCaseRequirementLink(
                id=code,
                type="functional",
                title=family.requirement_titles.get(code) or family.title,
                source_refs=_evidence_refs(source, code, kind="functional_requirement"),
            )
            for code in family.requirement_codes
        ]
        use_cases.append(
            UseCaseEntry(
                id=_use_case_id(bc_id, family.family),
                name=family.title or _family_title(family),
                module_id=module.id,
                primary_actor_id=primary_id,
                secondary_actor_ids=[item for item in secondary_ids if item != primary_id],
                description=(family.description or family.title or "").strip() or f"{family.title}.",
                preconditions=[],
                priority=_family_priority(family.priorities),
                evidence="inferred" if family.inferred else ("explicit" if family.source_refs else "inferred"),
                related_requirements=requirement_links,
                source_refs=refs,
            )
        )

    model = UseCaseModel(
        system=UseCaseSystem(id="SYSTEM", name=system_name, source_refs=_fallback_component_refs(source)),
        modules=modules,
        actors=actors,
        use_cases=use_cases,
        relationships=[],
    )
    _ensure_relationship_ids(model)
    return model


def _merge_candidate_details(target: UseCaseModel, candidate: UseCaseModel, source: RequirementsSourceSnapshot) -> None:
    """Merge only source-mapped AI details and relationships."""

    target_module_by_id = {item.id: item for item in target.modules}
    target_module_by_name = {_name_key(item.name): item for item in target.modules}
    module_map: dict[str, str] = {}
    for draft_module in candidate.modules:
        draft_module.source_refs = _candidate_source_refs(draft_module.source_refs, source)
        target_module = target_module_by_id.get(draft_module.id) or target_module_by_name.get(
            _name_key(draft_module.name)
        )
        if target_module is None and _valid_candidate_refs(draft_module.source_refs, source):
            target.modules.append(draft_module)
            target_module_by_id[draft_module.id] = draft_module
            target_module_by_name[_name_key(draft_module.name)] = draft_module
            target_module = draft_module
        if target_module is not None:
            module_map[draft_module.id] = target_module.id

    target_actor_by_id = {item.id: item for item in target.actors}
    target_actor_by_name = {_name_key(item.name): item for item in target.actors}
    actor_map: dict[str, str] = {}
    for draft_actor in candidate.actors:
        draft_actor.source_refs = _candidate_source_refs(draft_actor.source_refs, source)
        target_actor = target_actor_by_id.get(draft_actor.id) or target_actor_by_name.get(_name_key(draft_actor.name))
        if target_actor is None and _valid_candidate_refs(draft_actor.source_refs, source):
            target.actors.append(draft_actor)
            target_actor_by_id[draft_actor.id] = draft_actor
            target_actor_by_name[_name_key(draft_actor.name)] = draft_actor
            target_actor = draft_actor
        if target_actor is not None:
            actor_map[draft_actor.id] = target_actor.id

    target_by_id = {item.id: item for item in target.use_cases}
    target_by_name = {_name_key(item.name): item for item in target.use_cases}
    for draft in candidate.use_cases:
        item = target_by_id.get(draft.id) or target_by_name.get(_name_key(draft.name))
        if item is None:
            continue
        mapped_module = module_map.get(draft.module_id)
        if mapped_module:
            item.module_id = mapped_module
        if draft.description and draft.description != item.description:
            item.description = draft.description
        for detail_field in (
            "trigger",
            "preconditions",
            "main_flow",
            "alternative_flows",
            "exception_flows",
            "postconditions_success",
            "postconditions_failure",
            "business_rules",
            "related_requirements",
            "note",
        ):
            value = getattr(draft, detail_field)
            if value and detail_field != "related_requirements":
                setattr(item, detail_field, value)
        if draft.priority != "recommended" or item.priority == "recommended":
            item.priority = draft.priority
        if draft.evidence != "inferred" or item.evidence == "inferred":
            item.evidence = draft.evidence
        if draft.related_requirements:
            known_links = {link.id: link for link in item.related_requirements}
            for link in draft.related_requirements:
                link.source_refs = _candidate_source_refs(link.source_refs, source)
                known_links[link.id] = link
            item.related_requirements = list(known_links.values())
        item.secondary_actor_ids = _unique_refs(
            actor_map.get(actor_id, actor_id)
            for actor_id in draft.secondary_actor_ids
            if actor_map.get(actor_id, actor_id) in target_actor_by_id
        )
        mapped_primary = actor_map.get(draft.primary_actor_id, draft.primary_actor_id)
        if mapped_primary in target_actor_by_id:
            item.primary_actor_id = mapped_primary
        item.source_refs = _unique_refs([*item.source_refs, *_candidate_source_refs(draft.source_refs, source)])

    # LLM relationships are mapped by current use-case IDs first, then by names. Associations are
    # generated by actor fields; the semantic relation list is include/extend/generalization only.
    candidate_by_id = {item.id: item for item in candidate.use_cases}
    for relation in candidate.relationships:
        relation.source_refs = _candidate_source_refs(relation.source_refs, source)
        if not _valid_candidate_refs(relation.source_refs, source):
            continue
        source_draft = candidate_by_id.get(relation.source_id)
        target_draft = candidate_by_id.get(relation.target_id)
        source_item = target_by_id.get(relation.source_id) or (
            target_by_name.get(_name_key(source_draft.name)) if source_draft else None
        )
        target_item = target_by_id.get(relation.target_id) or (
            target_by_name.get(_name_key(target_draft.name)) if target_draft else None
        )
        if source_item is None or target_item is None or source_item.id == target_item.id:
            continue
        if relation.review_state == "rejected":
            continue
        if relation.kind == "extend" and not relation.condition:
            continue
        if any(
            item.kind == relation.kind and item.source_id == source_item.id and item.target_id == target_item.id
            for item in target.relationships
        ):
            continue
        target.relationships.append(
            UseCaseRelation(
                id=_relation_id(
                    relation.kind, source_item.id, target_item.id, {item.id for item in target.relationships}
                ),
                kind=relation.kind,
                source_id=source_item.id,
                target_id=target_item.id,
                condition=relation.condition,
                reason=relation.reason,
                confidence=relation.confidence,
                review_state=relation.review_state,
                source_refs=relation.source_refs,
            )
        )

    include_target_counts: dict[str, int] = {}
    for relation in target.relationships:
        if relation.kind == "include":
            include_target_counts[relation.target_id] = include_target_counts.get(relation.target_id, 0) + 1
    target.relationships = [
        relation
        for relation in target.relationships
        if relation.kind != "include" or include_target_counts.get(relation.target_id, 0) >= 2
    ]


def _valid_candidate_refs(refs: Iterable[str], source: RequirementsSourceSnapshot) -> bool:
    candidate_refs = _candidate_source_refs(refs, source)
    if not candidate_refs:
        return False
    evidence = source.evidence_by_id()
    # A component-level citation only proves that the document exists. It cannot support adding
    # a new actor/module or a semantic relationship that the deterministic source floor did not
    # already contain.
    return any(ref in evidence and evidence[ref].kind != "component" for ref in candidate_refs)


def _candidate_source_refs(refs: Iterable[str], source: RequirementsSourceSnapshot) -> list[str]:
    return _unique_refs(canonical_evidence_refs(refs, source))


def _parse_capabilities(source: RequirementsSourceSnapshot) -> list[_Capability]:
    component = _find_component(source, "prd", "use_case") or _find_component(source, "prd", "functional_requirement")
    if component is None:
        return []

    # The normal PRD shape is a set of markdown tables headed by ``ID | Capability | Goal``.
    # Parse those rows before the legacy heading parser.  The latter intentionally supports the
    # older feature-module documents, but it cannot distinguish a domain heading from a capability
    # heading in the current table-based PRD.
    table_capabilities = _parse_capability_tables(source, component.body)
    if table_capabilities:
        return table_capabilities

    lines = component.body.splitlines()
    capabilities: list[_Capability] = []
    current: _Capability | None = None
    capability_section_level: int | None = None
    for raw_line in lines:
        line = raw_line.strip()
        heading = re.match(r"^(#{1,6})\s*(.+?)\s*$", line)
        if heading:
            heading_level = len(heading.group(1))
            heading_title = heading.group(2).strip()
            if _CAPABILITY_SECTION_RE.match(heading_title):
                capability_section_level = heading_level
                current = None
                continue
            if capability_section_level is not None and heading_level <= capability_section_level:
                capability_section_level = None
        match = _BC_HEADING_RE.match(line)
        feature_match = _FEATURE_HEADING_RE.match(line)
        if match:
            if current:
                capabilities.append(current)
            bc_id = re.sub(r"[ _]+", "-", match.group("id").upper())
            current = _Capability(
                id=bc_id,
                name=_clean_text(match.group("name")),
                source_refs=_evidence_refs(source, bc_id, kind="business_capability")
                or _component_ref_list(source, "prd", "use_case"),
            )
            continue
        if feature_match:
            if current:
                capabilities.append(current)
            feature_id = f"FM-{feature_match.group('id').upper()}"
            current = _Capability(
                id=feature_id,
                name=_clean_text(feature_match.group("name")),
                source_refs=_evidence_refs(source, feature_id, kind="subsystem")
                or _component_ref_list(source, "prd", "use_case"),
            )
            continue
        if heading and capability_section_level is not None and heading_level > capability_section_level:
            numbered = _NUMBERED_HEADING_RE.match(heading_title)
            if numbered and numbered.group("name").strip().lower() not in {"goal", "scope", "objective"}:
                if current:
                    capabilities.append(current)
                current = _Capability(
                    id=f"BC-{len(capabilities) + 1:02d}",
                    name=_clean_text(numbered.group("name")),
                    source_refs=_component_ref_list(source, "prd", "use_case"),
                )
                continue
        # A markdown table may use a numbered capability id/name instead of BC-xx headings.
        if current is None and line.startswith("|") and "|" in line[1:]:
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if (
                len(cells) >= 2
                and re.fullmatch(r"(?:C|CAP|BC)?[- ]?\d+", cells[0], re.I)
                and cells[1]
                and not set(cells[1]) <= {"-", ":"}
            ):
                current = _Capability(
                    id=f"BC-{len(capabilities) + 1:02d}",
                    name=_clean_text(cells[1]),
                    source_refs=_component_ref_list(source, "prd", "use_case"),
                )
                continue
        if current is None:
            continue
        bullet = _BULLET_RE.match(line)
        text = bullet.group("text") if bullet else line
        key_match = _ROLE_KEY_RE.search(text)
        if key_match:
            current.user_segment.extend(_split_roles(key_match.group(1)))
        if re.search(r"(?:^|[*_])goal(?:[*_]|\s*:)", text, re.I):
            current.goal = _clean_text(re.sub(r".*?goal(?:[*_]|\s*:)", "", text, flags=re.I))
        elif re.search(r"(?:^|[*_])scope(?:[*_]|\s*:)", text, re.I) and not current.goal:
            current.goal = _clean_text(re.sub(r".*?scope(?:[*_]|\s*:)", "", text, flags=re.I))
    if current:
        capabilities.append(current)
    for capability in capabilities:
        capability.user_segment = _unique_strings(capability.user_segment)
    return capabilities


def _parse_requirement_families(source: RequirementsSourceSnapshot) -> list[_RequirementFamily]:
    component = _find_component(source, "prd", "functional_requirement")
    if component is None:
        return []

    table_families = _parse_requirement_tables(source, component.body)
    if table_families:
        return table_families

    grouped: OrderedDict[tuple[str, str], _RequirementFamily] = OrderedDict()
    priority_index: int | None = None
    feature_family: _RequirementFamily | None = None
    feature_families: OrderedDict[str, _RequirementFamily] = OrderedDict()
    for line in component.body.splitlines():
        stripped = line.strip()
        feature_match = _FEATURE_HEADING_RE.match(stripped)
        if feature_match:
            priority_index = None
            feature_id = f"FM-{feature_match.group('id').upper()}"
            feature_family = _RequirementFamily(
                bc_id=feature_id,
                family=feature_id,
                title=_clean_text(feature_match.group("name")),
                source_refs=_evidence_refs(source, feature_id, kind="subsystem")
                or _component_ref_list(source, "prd", "functional_requirement"),
            )
            feature_families[feature_id] = feature_family
            continue
        if feature_family and stripped and not stripped.startswith(("|", "<!--", "#")):
            text = _clean_text(stripped)
            if text.casefold() not in {"objective", "goal", "scope", "description"}:
                feature_family.description = feature_family.description or text
        heading_match = _FR_HEADING_RE.match(stripped)
        if heading_match:
            cells = [heading_match.group("code").upper(), _clean_text(heading_match.group("title"))]
            joined = " — ".join(cells)
            priority_value = ""
            code_match = _FR_CODE_RE.search(cells[0])
        else:
            if not stripped.startswith("|") or re.match(r"^\s*\|?\s*:?-{2,}", line):
                continue
            cells = [part.strip() for part in stripped.strip("|").split("|")]
            if not cells:
                continue
            lowered_cells = [cell.casefold() for cell in cells]
            if any("priority" in cell for cell in lowered_cells) and not _FR_CODE_RE.search(" | ".join(cells)):
                priority_index = next(index for index, cell in enumerate(lowered_cells) if "priority" in cell)
                continue
            joined = " | ".join(cells)
            code_match = _FR_CODE_RE.search(joined)
            priority_value = (
                cells[priority_index]
                if priority_index is not None and priority_index < len(cells)
                else (cells[-1] if len(cells) > 1 else "")
            )
        if code_match is None:
            simple_match = _SIMPLE_FR_CODE_RE.search(joined)
            if simple_match is None:
                continue
            number = simple_match.group("number")
            code = f"FR-{number}".upper()
            family = code
            title = _clean_text(cells[1] if len(cells) > 1 else code)
        else:
            original_family = code_match.group("family").upper()
            family = original_family
            number = code_match.group("number")
            code = f"FR-{original_family}-{number}".upper()
            title = _requirement_title(cells, code)
        bc_match = re.search(r"BC[- ]?\d+", joined, re.I)
        fallback_bc = feature_family.bc_id if feature_family else "BC-01"
        bc_id = re.sub(r"[ _]+", "-", (bc_match.group(0) if bc_match else fallback_bc).upper())
        key = (bc_id, family)
        item = grouped.get(key)
        if item is None:
            item = _RequirementFamily(bc_id=bc_id, family=family, title=title)
            grouped[key] = item
        item.requirement_codes.append(code)
        if feature_family is not None:
            feature_family.requirement_codes.append(code)
        item.source_refs.extend(_evidence_refs(source, code, kind="functional_requirement"))
        if not item.source_refs:
            item.source_refs.extend(_component_ref_list(source, "prd", "functional_requirement"))
        if len(cells) > 2 and not item.description:
            item.description = _clean_text(cells[2])
        item.requirement_titles.setdefault(code, title)
        item.priorities.append(priority_value)
    # A raw component can describe a feature module only through its objective and omit an FR
    # table. Preserve that source-backed behavior as one inferred *goal* use case with an action
    # name; never expose the module title itself as a use case row.
    for feature in feature_families.values():
        if feature.requirement_codes:
            continue
        feature.family = "OBJECTIVE"
        feature.title = _feature_use_case_title(feature.title)
        feature.inferred = True
        grouped.setdefault((feature.bc_id, feature.family), feature)
    return list(grouped.values())


def _parse_capability_tables(source: RequirementsSourceSnapshot, body: str) -> list[_Capability]:
    """Read capability rows from the stored PRD component.

    A component may contain a summary table and actor/objective matrices after the main table.
    Only rows with both an ID column and a Capability column are accepted, so matrix rows cannot
    become fake modules.
    """

    capabilities: list[_Capability] = []
    seen: set[str] = set()
    for headers, values in _markdown_table_rows(body):
        normalized = [_clean_text(header).casefold() for header in headers]
        id_index = _first_column(normalized, {"id", "code", "capability id"})
        capability_index = _first_header_containing(normalized, ("capability", "business capability"))
        if id_index is None or capability_index is None or id_index >= len(values) or capability_index >= len(values):
            continue
        raw_id = _clean_text(values[id_index]).upper()
        id_match = re.fullmatch(r"(?:C|CAP|BC)[-_ ]?(\d+)", raw_id, flags=re.I)
        if id_match is None:
            continue
        capability_id = f"C{int(id_match.group(1))}"
        if capability_id in seen:
            continue
        name = _clean_text(values[capability_index])
        if not name:
            continue
        goal_index = _first_header_containing(normalized, ("goal", "objective", "purpose"))
        segment_index = _first_header_containing(normalized, ("user segment", "actor", "user role", "users"))
        goal = _clean_text(values[goal_index]) if goal_index is not None and goal_index < len(values) else ""
        segment = (
            _split_roles(values[segment_index])
            if segment_index is not None and segment_index < len(values)
            else []
        )
        capabilities.append(
            _Capability(
                id=capability_id,
                name=name,
                goal=goal,
                user_segment=segment,
                source_refs=_evidence_refs(source, capability_id, kind="business_capability")
                or _component_ref_list(source, "prd", "use_case"),
            )
        )
        seen.add(capability_id)
    return capabilities


def _parse_requirement_tables(source: RequirementsSourceSnapshot, body: str) -> list[_RequirementFamily]:
    """Read functional-requirement tables, including compact IDs such as ``FR-TM01``."""

    capabilities = _parse_capabilities(source)
    grouped: OrderedDict[tuple[str, str], _RequirementFamily] = OrderedDict()
    for headers, values in _markdown_table_rows(body):
        normalized = [_clean_text(header).casefold() for header in headers]
        id_index = _first_column(normalized, {"id", "code", "requirement id", "requirement code"})
        title_index = _first_header_containing(normalized, ("requirement", "title", "name", "use case"))
        if id_index is None or title_index is None or id_index >= len(values) or title_index >= len(values):
            continue
        code_match = _match_functional_requirement(values[id_index])
        if code_match is None:
            code_match = _match_functional_requirement(" | ".join(values))
        if code_match is None:
            continue
        code, family = code_match
        title = _clean_text(values[title_index]) or code
        description_index = _first_header_containing(normalized, ("behavior", "description", "goal", "objective"))
        priority_index = _first_header_containing(normalized, ("priority", "moscow"))
        description = (
            _clean_text(values[description_index])
            if description_index is not None and description_index < len(values)
            else title
        )
        priority = values[priority_index] if priority_index is not None and priority_index < len(values) else ""
        capability_id = _capability_for_requirement(family, title, capabilities)
        # Each functional requirement is an atomic source-backed behavior.  Keep one row per FR
        # so later relationship resolution can correctly express include/extend (for example,
        # create-task -> validate-task) instead of hiding several behaviors in one family row.
        key = (capability_id, code)
        item = grouped.get(key)
        if item is None:
            item = _RequirementFamily(
                bc_id=capability_id,
                family=code,
                title=title,
                description=description,
            )
            grouped[key] = item
        item.requirement_codes.append(code)
        item.requirement_titles.setdefault(code, title)
        item.priorities.append(priority)
        item.source_refs.extend(_evidence_refs(source, code, kind="functional_requirement"))
        if not item.source_refs:
            item.source_refs.extend(_component_ref_list(source, "prd", "functional_requirement"))
        # Keep the first behavior as the summary, but prefer a more useful title if a malformed
        # row supplied an empty behavior cell.
        if not item.description or item.description == item.title:
            item.description = description or title
    return list(grouped.values())


def _match_functional_requirement(value: str) -> tuple[str, str] | None:
    text = _clean_text(value)
    match = _FR_CODE_RE.search(text)
    if match:
        code = match.group(0).upper()
        return code, match.group("family").upper()
    match = _FR_COMPACT_CODE_RE.search(text)
    if match:
        code = match.group(0).upper()
        return code, match.group("family").upper()
    simple = _SIMPLE_FR_CODE_RE.search(text)
    if simple:
        code = simple.group(0).upper()
        return code, code.removeprefix("FR-")
    return None


def _capability_for_requirement(family: str, title: str, capabilities: list[_Capability]) -> str:
    """Map a requirement family to the closest stored capability name.

    The mapping is lexical and source-backed: it only chooses among capability rows parsed from
    the same PRD.  If a source uses opaque family codes, the first capability is a conservative
    grouping fallback and the AI may enrich details without inventing a module.
    """

    preferred_ids = {
        "tm": "C1",
        "tp": "C3",
        "st": "C5",
        "aa": "C2",
        "mb": "C4",
        "nt": "C6",
        "rp": "C7",
        "pl": "C11",
    }
    preferred_id = preferred_ids.get(family.casefold())
    if preferred_id and any(item.id.casefold() == preferred_id.casefold() for item in capabilities):
        return preferred_id

    haystack = f"{family} {title}".casefold()
    hints = {
        "tm": ("tạo task", "task"),
        "tp": ("template", "quy trình lặp"),
        "st": ("trạng thái", "theo dõi"),
        "aa": ("auto", "phân công", "assignment"),
        "mb": ("thành viên", "member", "onboarding"),
        "nt": ("thông báo", "nhắc", "notification"),
        "rp": ("dashboard", "audit", "báo cáo", "report"),
        "pl": ("tích hợp", "platform", "mobile", "pwa"),
    }
    family_hints = hints.get(family.casefold(), ())
    if family_hints:
        for capability in capabilities:
            capability_text = capability.name.casefold()
            if any(token in haystack and token in capability_text for token in family_hints):
                return capability.id
    return capabilities[0].id if capabilities else "BC-01"


def _first_column(headers: list[str], names: set[str]) -> int | None:
    for index, header in enumerate(headers):
        if header in names:
            return index
    return None


def _first_header_containing(headers: list[str], tokens: tuple[str, ...]) -> int | None:
    for index, header in enumerate(headers):
        if any(token in header for token in tokens):
            return index
    return None


def _roles_from_stakeholders(source: RequirementsSourceSnapshot) -> list[str]:
    """Extract actor roles from stakeholder roster tables only.

    Stakeholder components commonly contain several markdown tables: the actual roster, a
    responsibility matrix, and assumptions/traceability tables.  Treating the first cell of
    every table row as an actor (the old behavior) promoted headers such as ``#``/``SR1`` and
    long ``agent-inferred`` annotations into actor names.  Those values can violate the 120
    character actor contract and make the whole generate endpoint return HTTP 500 before the
    provider call is even attempted.
    """

    component = _find_component(source, "brd", "stakeholder_register")
    if component is None:
        return []
    roles: list[str] = []
    for headers, values in _markdown_table_rows(component.body):
        if not headers or not values:
            continue
        header = _clean_text(headers[0]).casefold()
        if header not in {
            "vai trò",
            "vai tro",
            "role",
            "actor",
            "stakeholder",
            "stakeholders",
            "persona",
        }:
            continue
        role = _clean_actor_role(values[0])
        if role:
            roles.append(role)
    return _unique_strings(roles)


def _ensure_relationship_ids(model: UseCaseModel) -> None:
    valid_ids = {item.id for item in model.use_cases} | {item.id for item in model.actors}
    for item in model.use_cases:
        item.relationship_ids = []
    for relation in model.relationships:
        if relation.source_id not in valid_ids or relation.target_id not in valid_ids:
            continue
        if relation.source_id in {item.id for item in model.use_cases}:
            model.use_cases[[item.id for item in model.use_cases].index(relation.source_id)].relationship_ids.append(
                relation.id
            )
        if relation.target_id in {item.id for item in model.use_cases}:
            model.use_cases[[item.id for item in model.use_cases].index(relation.target_id)].relationship_ids.append(
                relation.id
            )
    for item in model.use_cases:
        item.relationship_ids = _unique_refs(item.relationship_ids)


def _find_component(source: RequirementsSourceSnapshot, document_type: str, artifact_type: str):
    for component in source.components:
        if component.document_type == document_type and component.artifact_type == artifact_type:
            return component
    return None


def _component_ref(source: RequirementsSourceSnapshot, document_type: str, artifact_type: str) -> str | None:
    refs = _component_ref_list(source, document_type, artifact_type)
    return refs[0] if refs else None


def _component_ref_list(source: RequirementsSourceSnapshot, document_type: str, artifact_type: str) -> list[str]:
    return [
        item.evidence_id
        for item in source.evidence
        if item.document_type == document_type and item.artifact_type == artifact_type and item.kind == "component"
    ]


def _evidence_refs(source: RequirementsSourceSnapshot, entity: str, *, kind: str) -> list[str]:
    normalized = entity.upper()
    return [
        item.evidence_id
        for item in source.evidence
        if item.entity_id and item.entity_id.upper() == normalized and item.kind == kind
    ]


def _fallback_component_refs(source: RequirementsSourceSnapshot) -> list[str]:
    refs = [item.evidence_id for item in source.evidence if item.kind == "component"]
    return refs[:2] or [f"sha256:{source.source_hash}"]


def _actor_id(name: str) -> str:
    slug = re.sub(r"[^A-Z0-9]+", "-", name.upper()).strip("-")
    return f"ACT-{slug[:70]}" if slug else "ACT-STAKEHOLDER"


def _use_case_id(bc_id: str, family: str) -> str:
    bc = re.sub(r"[^A-Z0-9]+", "-", bc_id.upper()).strip("-") or "BC-01"
    code = re.sub(r"[^A-Z0-9]+", "-", family.upper()).strip("-") or "FEATURE"
    return f"UC-{bc}-{code}"


def _relation_id(kind: str, source_id: str, target_id: str, existing: set[str]) -> str:
    base = f"REL-{kind.upper()}-{source_id}-{target_id}"
    candidate = base
    n = 2
    while candidate in existing:
        candidate = f"{base}-{n}"
        n += 1
    return candidate


def _family_title(family: _RequirementFamily) -> str:
    return family.title or family.family.replace("-", " ").title()


def _requirement_title(cells: list[str], code: str) -> str:
    """Extract a human requirement title without treating a hyphen in ``FR-A-01`` as a separator."""

    candidate = " | ".join(cells)
    em_dash = re.search(r"[—–]\s*([^|\n]+)", candidate)
    if em_dash:
        title = _clean_text(em_dash.group(1))
        if title:
            return title
    if len(cells) > 1:
        title = _clean_text(cells[1])
        if title and title.casefold() != code.casefold():
            return title
    return code


def _feature_use_case_title(module_name: str) -> str:
    """Turn a feature-module noun phrase into a distinct, business-goal use-case name."""

    name = _clean_text(module_name)
    lowered = name.casefold()
    if "authentication" in lowered and "access" in lowered:
        return "Authenticate User Access"
    if "upload" in lowered and "validat" in lowered:
        return "Upload and Validate Dataset"
    if "profil" in lowered and "data card" in lowered:
        return "Profile Dataset"
    if "question" in lowered and "hypothes" in lowered:
        return "Manage Research Questions and Hypotheses"
    if "experiment" in lowered and "plan" in lowered:
        return "Plan Experiment"
    if "finding" in lowered or "report" in lowered:
        return "Generate Research Findings and Reports"
    return f"Manage {name}" if name else "Manage Requirements"


def _family_priority(priorities: Iterable[str]) -> str:
    text = " ".join(priorities).lower()
    if any(token in text for token in ("p0", "p1", "must", "required", "critical")):
        return "required"
    if any(token in text for token in ("p2", "should", "recommended", "high")):
        return "recommended"
    # Missing priority is not evidence that a feature is optional. Keep the inferred row visible
    # as recommended until a BRD/PRD priority or an AI-backed source statement says otherwise.
    return "recommended"


def _roles_for_capability(capability: _Capability) -> list[str]:
    return capability.user_segment or [_GENERIC_ACTOR_ROLE]


def _split_roles(value: str) -> list[str]:
    return [
        role
        for part in re.split(r",|;|\band\b", value, flags=re.I)
        if (role := _clean_actor_role(part))
    ]


def _clean_actor_role(value: str) -> str:
    """Convert source prose into a bounded, human-readable actor label.

    The BRD editor deliberately leaves ``agent-inferred`` and ``needs_confirmation`` notes in
    the stored component body.  They are useful evidence, but they are not part of the role name
    shown in the use-case table.  Remove only that annotation, preserve the actual role text, and
    enforce the model's maximum as a final guard against malformed source rows.
    """

    text = _clean_text(value).strip(" -*")
    if not text:
        return ""
    # Drop a trailing markdown annotation such as ``*(agent-inferred: ... needs_confirmation)*``.
    text = re.sub(r"\s*\*?\(?\s*(?:agent[- ]inferred|suy luận)\b.*$", "", text, flags=re.I)
    text = re.sub(r"\s*\*?\(?\s*needs[_ -]?confirmation\s*\)?\s*$", "", text, flags=re.I)
    text = _clean_text(text).strip(" -*")
    if not text:
        return ""
    lowered = text.casefold()
    if lowered in {
        "vai trò",
        "vai tro",
        "role",
        "actor",
        "stakeholder",
        "stakeholders",
        "persona",
        "#",
    }:
        return ""
    # Ignore assumption/traceability identifiers that appear in later BRD tables.
    if re.fullmatch(r"(?:sr|br|brule|fr|nfr)?[-_ ]?\d+(?:[-_ ]\w+)?", text, flags=re.I):
        return ""
    return text[:120].rstrip(" -*")


def _clean_text(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", str(value))
    text = text.replace("**", "").replace("__", "").replace("`", "")
    return re.sub(r"\s+", " ", text).strip(" .")


def _name_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _unique_strings(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = _clean_text(value)
        key = cleaned.casefold()
        if cleaned and key not in seen:
            output.append(cleaned)
            seen.add(key)
    return output


def _unique_refs(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            output.append(value)
            seen.add(value)
    return output
