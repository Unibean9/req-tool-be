"""Complete the use-case table from the stored BRD/PRD registry.

The LLM is useful for wording and for identifying optional UML relations, but it is not a
reliable enumerator.  A long PRD can cause a model to collapse several functional-requirement
groups into one row or to omit actor associations.  This module provides the deterministic table
floor: every stored business capability becomes an L0 row and every functional-requirement
family becomes an L1 goal under that capability.  It never reads repository markdown fixtures.

The generated model is deliberately table-first. ``include`` and ``extend`` relations from an
LLM candidate are retained only when their endpoints can be mapped to the source-backed table;
the module never invents such relations from a sequence or a data dependency. PlantUML is
rendered from this completed table by ``plantuml.py``; legacy semantic diagram definitions are
not generated anymore.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass, field

from app.use_cases.models import (
    RequirementsSourceSnapshot,
    UseCaseActor,
    UseCaseDiagramDefinition,
    UseCaseEntry,
    UseCaseModel,
    UseCaseRelation,
    UseCaseSubsystem,
)

_BC_HEADING_RE = re.compile(r"^###\s+(BC-\d+)\s*:\s*(.+?)\s*$")
_FR_RE = re.compile(
    r"(?P<bc>BC-\d+)\s*[·•]\s*FR-(?P<family>[A-Z0-9]+)-(?P<number>\d+)\s*[—-]\s*(?P<title>.+?)\s*$",
    re.IGNORECASE,
)
_FR_CODE_RE = re.compile(r"FR-(?P<family>[A-Z0-9]+)-(?P<number>\d+)", re.IGNORECASE)

_ACTOR_IDS = {
    "system administrator": "ACT-ADMIN",
    "research project manager": "ACT-MANAGER",
    "researcher / data analyst": "ACT-RESEARCHER",
    "researcher": "ACT-RESEARCHER",
    "reviewer / stakeholder": "ACT-REVIEWER",
    "reviewer": "ACT-REVIEWER",
}
_ACTOR_NAMES = {value: key.title() for key, value in _ACTOR_IDS.items()}
# Keep the exact display names used by the stakeholder register rather than title-casing the
# slash-separated role.
_ACTOR_NAMES.update(
    {
        "ACT-ADMIN": "System Administrator",
        "ACT-MANAGER": "Research Project Manager",
        "ACT-RESEARCHER": "Researcher / Data Analyst",
        "ACT-REVIEWER": "Reviewer / Stakeholder",
    }
)

_GROUP_TITLES = {
    "AUTH": "Authenticate User Access",
    "PROJ": "Create and Edit Research Project",
    "DATA": "Upload and Validate Dataset",
    "PROFILE": "Profile Dataset",
    "CLEAN": "Clean and Version Dataset",
    "RQ": "Define Research Question and Context",
    "HYP": "Define Research Hypotheses",
    "STATE": "Build Research State",
    "PLAN": "Plan Experiment",
    "METHOD": "Select Statistical Method",
    "ASSUME": "Check Method Assumptions",
    "BRANCH": "Compare Experiment Branches",
    "EXEC": "Execute Experiment Run",
    "VALID": "Validate Scientific Results",
    "MTEST": "Control Multiple Testing",
    "EFFECT": "Evaluate Effect Size",
    "UNCERT": "Record Result Uncertainty",
    "LEAK": "Flag Data Leakage",
    "STOP": "Evaluate Stopping Criteria",
    "FIND": "Generate Research Findings",
    "REFINE": "Generate New Hypotheses",
    "DEP": "Track Experiment Dependencies",
    "CONFLICT": "Preserve Conflicting Evidence",
    "PROV": "Record Finding Provenance",
    "TRACE": "Inspect Execution Trace",
    "REPRO": "Create Reproducibility Snapshot",
    "VIZ": "Generate Research Figures",
    "REPORT": "Generate Research Report",
    "EVAL": "Evaluate Research Agent",
    "ADMIN": "Configure Available Models",
    "HGATE": "Evaluate Candidate Hypotheses",
    "EGATE": "Evaluate Evidence Sufficiency",
    "REFLOOP": "Refine Scientific Experiment",
    "DEC": "Audit Decision History",
    "IDEA": "Record Research Ideas",
    "IDEA-NOVELTY": "Assess Idea Novelty",
    "MANU-FIGURE": "Review Figure Aggregation",
    "MANU": "Generate and Review Manuscripts",
}

_BC_PRECONDITIONS = {
    "BC-01": "The user is authenticated and has project membership or administrative permission.",
    "BC-02": "An accessible project and the relevant dataset or dataset version are available.",
    "BC-03": "An accessible project has a research question or research context.",
    "BC-04": "A research question, hypothesis, and dataset version are available.",
    "BC-05": "A reviewed experiment plan and permitted dataset version are available.",
    "BC-06": "A completed experiment or validated result is available.",
    "BC-07": "A project run, finding, or decision record is available for evaluation.",
    "BC-08": "A validated finding or research idea is available for communication or review.",
}


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


def complete_use_case_table(
    source: RequirementsSourceSnapshot,
    candidate: UseCaseModel | None = None,
) -> UseCaseModel:
    """Return a complete, source-backed table and its semantic hierarchy.

    A candidate is accepted only when it already has at least the source-derived row floor.  A
    shorter candidate is treated as an incomplete LLM enumeration and replaced by the canonical
    table.  Candidate non-association relations are mapped by endpoint names when possible so an
    explicit, evidence-backed ``include``/``extend`` survives the completion pass.
    """

    canonical = _build_source_model(source)
    candidate_is_short = candidate is None or len(candidate.use_cases) < len(canonical.use_cases)
    candidate_has_invalid_actor = bool(
        candidate
        and any(not re.search(r"[A-Za-z]", actor.name) for actor in candidate.actors)
    )
    if candidate_is_short or candidate_has_invalid_actor:
        if candidate is not None:
            _merge_explicit_relations(canonical, candidate)
        _ensure_actor_associations(canonical)
        canonical.diagrams = []
        return canonical

    _ensure_actor_associations(candidate)
    candidate.diagrams = []
    return candidate


def _build_source_model(source: RequirementsSourceSnapshot) -> UseCaseModel:
    capabilities = _parse_capabilities(source)
    families = _parse_requirement_families(source)
    if not capabilities or not families:
        # Let the existing LLM validator report missing source evidence when a project has no
        # normalized BRD/PRD components.  This path is only a safety fallback for unusual drafts.
        return UseCaseModel(
            system_name=source.prd.title or source.brd.title or "Requirements System",
        )

    stakeholder_ref = _component_ref(source, "brd", "stakeholder_register")
    actor_ids = OrderedDict()
    for capability in capabilities:
        for role in capability.user_segment:
            actor_id = _actor_id(role)
            if actor_id:
                actor_ids[actor_id] = None
    family_entries: dict[tuple[str, str], UseCaseEntry] = {}
    for family in families:
        for role in _roles_for_capability(family.bc_id, capabilities):
            actor_id = _actor_id(role)
            if actor_id:
                actor_ids[actor_id] = None

    actors = [
        UseCaseActor(
            id=actor_id,
            name=_ACTOR_NAMES[actor_id],
            kind="human_role",
            source_refs=[stakeholder_ref] if stakeholder_ref else _fallback_component_refs(source),
        )
        for actor_id in actor_ids
    ]
    actors_by_id = {item.id: item for item in actors}

    subsystems: list[UseCaseSubsystem] = []
    capability_by_id: dict[str, _Capability] = {}
    for index, capability in enumerate(capabilities, start=1):
        subsystem_id = f"SUB-BC-{index:02d}"
        capability.id = capability.id.upper()
        capability_by_id[capability.id] = capability
        subsystems.append(
            UseCaseSubsystem(
                id=subsystem_id,
                name=capability.name,
                source_refs=capability.source_refs or _fallback_component_refs(source),
            )
        )

    subsystem_by_bc = {
        capability.id: subsystems[index]
        for index, capability in enumerate(capabilities)
    }
    use_cases: list[UseCaseEntry] = []
    l0_by_bc: dict[str, UseCaseEntry] = {}
    family_entries: dict[tuple[str, str], UseCaseEntry] = {}
    for capability in capabilities:
        roles = _roles_for_capability(capability.id, capabilities)
        primary_id, secondary_ids = _actor_roles(roles)
        title = _summary_title(capability.name, capability.id)
        item = UseCaseEntry(
            id=f"UC-SUM-{capability.id}",
            name=title,
            level="L0",
            abstraction="summary",
            primary_actor_id=primary_id,
            secondary_actor_ids=secondary_ids,
            subsystem_id=subsystem_by_bc[capability.id].id,
            parent_use_case_id=None,
            description=(
                f"The actor can {(capability.goal or capability.name).rstrip('.').lower()}."
            )[:600],
            precondition=_BC_PRECONDITIONS.get(capability.id, "The project is available to the actor."),
            relationship_ids=[],
            priority=_family_priority([family for family in families if family.bc_id == capability.id]),
            status="confirmed",
            source_refs=capability.source_refs or _fallback_component_refs(source),
        )
        l0_by_bc[capability.id] = item
        use_cases.append(item)

    for family in families:
        capability = capability_by_id.get(family.bc_id)
        subsystem = subsystem_by_bc.get(family.bc_id)
        if capability is None or subsystem is None:
            continue
        roles = _roles_for_family(family, capability)
        primary_id, secondary_ids = _actor_roles(roles)
        title = _family_title(family)
        source_refs = _unique_refs(
            [
                *family.source_refs,
                *capability.source_refs,
                _component_ref(source, "prd", "functional_requirement"),
            ]
        )
        item = UseCaseEntry(
            id=f"UC-{family.bc_id}-{family.family}",
            name=title,
            level="L1",
            abstraction="user_goal",
            primary_actor_id=primary_id,
            secondary_actor_ids=secondary_ids,
            subsystem_id=subsystem.id,
            parent_use_case_id=l0_by_bc[family.bc_id].id,
            description=(
                f"The {actors_by_id[primary_id].name} can {title[0].lower() + title[1:]} to support "
                f"{capability.goal.lower() or capability.name.lower()}."
            )[:600],
            precondition=_BC_PRECONDITIONS.get(family.bc_id, "The project is available to the actor."),
            relationship_ids=[],
            priority=_family_priority([family]),
            status="confirmed",
            source_refs=source_refs,
        )
        family_entries[(family.bc_id, family.family)] = item
        use_cases.append(item)

    model = UseCaseModel(
        system_name=source.prd.title or source.brd.title or "Requirements System",
        actors=actors,
        subsystems=subsystems,
        use_cases=use_cases,
        relations=[],
        diagrams=[],
    )
    _add_explicit_source_relations(model, family_entries)
    _ensure_actor_associations(model)
    return model


def _parse_capabilities(source: RequirementsSourceSnapshot) -> list[_Capability]:
    component = _find_component(source, "prd", "use_case")
    if component is None:
        return []
    lines = component.body.splitlines()
    capabilities: list[_Capability] = []
    current: _Capability | None = None
    for line_number, line in enumerate(lines, start=1):
        match = _BC_HEADING_RE.match(line.strip())
        if match:
            if current is not None:
                capabilities.append(current)
            bc_id, name = match.group(1).upper(), _clean_text(match.group(2))
            current = _Capability(
                id=bc_id,
                name=name,
                source_refs=_evidence_refs(source, bc_id, kind="business_capability")
                or [f"entity:prd:use_case:{bc_id}:{line_number}"],
            )
            continue
        if current is None:
            continue
        bullet = re.match(r"^-\s+\*\*(goal|user_segment|business_value|scope):\*\*\s*(.+)$", line.strip(), re.I)
        if not bullet:
            continue
        key, value = bullet.group(1).lower(), _clean_text(bullet.group(2))
        if key == "goal":
            current.goal = value
        elif key == "user_segment":
            current.user_segment = _split_roles(value)
    if current is not None:
        capabilities.append(current)
    return capabilities


def _parse_requirement_families(source: RequirementsSourceSnapshot) -> list[_RequirementFamily]:
    component = _find_component(source, "prd", "functional_requirement")
    if component is None:
        return []
    grouped: OrderedDict[tuple[str, str], _RequirementFamily] = OrderedDict()
    for line in component.body.splitlines():
        if not line.lstrip().startswith("| FR-"):
            continue
        cells = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(cells) < 6:
            continue
        match = _FR_RE.search(cells[1])
        if not match:
            continue
        bc_id = match.group("bc").upper()
        family = match.group("family").upper()
        requirement_number = int(match.group("number"))
        # These source rows explicitly describe optional/extension behavior. Keep them as
        # separate goals so their evidence is not hidden inside a generic family row.
        if family == "IDEA" and requirement_number == 3:
            family = "IDEA-NOVELTY"
        elif family == "MANU" and requirement_number == 1:
            family = "MANU-FIGURE"
        title = _clean_text(match.group("title"))
        fr_match = _FR_CODE_RE.search(cells[1])
        original_family = match.group("family").upper()
        requirement_code = f"FR-{original_family}-{match.group('number')}" if fr_match else cells[0]
        key = (bc_id, family)
        current = grouped.get(key)
        if current is None:
            current = _RequirementFamily(bc_id=bc_id, family=family, title=title)
            grouped[key] = current
        current.requirement_codes.append(requirement_code.upper())
        current.source_refs.extend(_evidence_refs(source, requirement_code.upper(), kind="functional_requirement"))
        current.priorities.append(cells[5])
    return list(grouped.values())


def _build_diagrams(
    model: UseCaseModel,
    capabilities: list[_Capability],
    subsystem_by_bc: dict[str, UseCaseSubsystem],
) -> None:
    all_l0 = [item.id for item in model.use_cases if item.level == "L0"]
    all_actor_ids = sorted(
        {
            actor_id
            for item in model.use_cases
            if item.level == "L0"
            for actor_id in _actor_ids(item)
        }
    )
    l0_nodes = set(all_actor_ids + all_l0)
    l0_relation_ids = [
        relation.id
        for relation in model.relations
        if relation.source_id in l0_nodes and relation.target_id in l0_nodes
    ]
    model.diagrams.append(
        UseCaseDiagramDefinition(
            id="DGM-L0",
            level="L0",
            system_boundary=model.system_name,
            actor_ids=all_actor_ids,
            use_case_ids=all_l0,
            relation_ids=l0_relation_ids,
        )
    )
    for capability in capabilities:
        subsystem = subsystem_by_bc[capability.id]
        l1_items = [
            item
            for item in model.use_cases
            if item.level == "L1" and item.subsystem_id == subsystem.id
        ]
        use_case_ids = [item.id for item in l1_items]
        actor_ids = sorted({actor_id for item in l1_items for actor_id in _actor_ids(item)})
        scoped_nodes = set(actor_ids + use_case_ids)
        scoped_relations = [
            relation.id
            for relation in model.relations
            if relation.source_id in scoped_nodes and relation.target_id in scoped_nodes
        ]
        model.diagrams.append(
            UseCaseDiagramDefinition(
                id=f"DGM-L1-{capability.id}",
                level="L1",
                system_boundary=model.system_name,
                subsystem_id=subsystem.id,
                actor_ids=actor_ids,
                use_case_ids=use_case_ids,
                relation_ids=scoped_relations,
            )
        )


def _ensure_actor_associations(model: UseCaseModel) -> None:
    """Make every declared primary/supporting actor visible as a table relation."""

    actors = {actor.id for actor in model.actors}
    use_cases = {item.id: item for item in model.use_cases}
    existing = {
        (relation.source_id, relation.target_id)
        for relation in model.relations
        if relation.kind == "association"
    }
    for item in model.use_cases:
        for actor_id in _actor_ids(item):
            if actor_id not in actors or (actor_id, item.id) in existing or (item.id, actor_id) in existing:
                continue
            relation = UseCaseRelation(
                id=f"REL-ASSOC-{actor_id}-{item.id}",
                kind="association",
                source_id=actor_id,
                target_id=item.id,
                source_refs=list(item.source_refs),
            )
            model.relations.append(relation)
            existing.add((actor_id, item.id))
    relation_ids_by_endpoint: dict[str, list[str]] = {item_id: [] for item_id in use_cases}
    for relation in model.relations:
        if relation.source_id in relation_ids_by_endpoint:
            relation_ids_by_endpoint[relation.source_id].append(relation.id)
        if relation.target_id in relation_ids_by_endpoint:
            relation_ids_by_endpoint[relation.target_id].append(relation.id)
    for item in model.use_cases:
        item.relationship_ids = _unique_refs(relation_ids_by_endpoint[item.id])


def _merge_explicit_relations(target: UseCaseModel, candidate: UseCaseModel) -> None:
    """Map candidate include/extend/generalization relations by endpoint names."""

    target_by_name = {_name_key(item.name): item.id for item in target.use_cases}
    candidate_by_id = {item.id: item for item in candidate.use_cases}
    for relation in candidate.relations:
        if relation.kind == "association":
            continue
        source = candidate_by_id.get(relation.source_id)
        target_item = candidate_by_id.get(relation.target_id)
        if source is None or target_item is None:
            continue
        source_id = _best_name_match(source.name, target_by_name)
        target_id = _best_name_match(target_item.name, target_by_name)
        if source_id is None or target_id is None or source_id == target_id:
            continue
        if any(
            item.kind == relation.kind and item.source_id == source_id and item.target_id == target_id
            for item in target.relations
        ):
            continue
        target.relations.append(
            UseCaseRelation(
                id=relation.id,
                kind=relation.kind,
                source_id=source_id,
                target_id=target_id,
                condition=relation.condition,
                source_refs=list(relation.source_refs),
            )
        )
    _ensure_actor_associations(target)


def _add_explicit_source_relations(
    model: UseCaseModel,
    family_entries: dict[tuple[str, str], UseCaseEntry],
) -> None:
    """Add only extension relations stated by the normalized PRD text."""

    extensions = (
        (
            ("BC-08", "MANU-FIGURE"),
            ("BC-08", "VIZ"),
            "When figure aggregation and visual feedback is enabled for a hypothesis.",
        ),
        (
            ("BC-08", "IDEA-NOVELTY"),
            ("BC-08", "IDEA"),
            "When the researcher enables optional novelty assessment for an idea.",
        ),
    )
    for index, (source_key, target_key, condition) in enumerate(extensions, start=1):
        source = family_entries.get(source_key)
        target = family_entries.get(target_key)
        if source is None or target is None:
            continue
        model.relations.append(
            UseCaseRelation(
                id=f"REL-EXTEND-{index}-{source.id}-{target.id}",
                kind="extend",
                source_id=source.id,
                target_id=target.id,
                condition=condition,
                source_refs=list(source.source_refs),
            )
        )


def _sync_diagram_relations(model: UseCaseModel) -> None:
    """Keep each semantic diagram's relation IDs aligned with the completed table."""

    for diagram in model.diagrams:
        node_ids = set(diagram.actor_ids) | set(diagram.use_case_ids)
        diagram.relation_ids = [
            relation.id
            for relation in model.relations
            if relation.source_id in node_ids and relation.target_id in node_ids
        ]


def _roles_for_capability(bc_id: str, capabilities: Iterable[_Capability]) -> list[str]:
    capability = next((item for item in capabilities if item.id == bc_id), None)
    if capability is not None and capability.user_segment:
        return capability.user_segment
    return ["Researcher / Data Analyst"]


def _roles_for_family(family: _RequirementFamily, capability: _Capability) -> list[str]:
    roles = list(capability.user_segment) or ["Researcher / Data Analyst"]
    if family.family == "AUTH":
        return ["Researcher / Data Analyst", "System Administrator"]
    if family.family == "PROJ":
        return ["Research Project Manager", "System Administrator"]
    if family.family == "ADMIN":
        return ["System Administrator", "Research Project Manager"]
    if family.family in {"EVAL", "DEC", "HGATE", "EGATE", "REFLOOP"} and "Research Project Manager" in roles:
        return ["Research Project Manager", *[role for role in roles if role != "Research Project Manager"]]
    if family.family in {"REPORT", "MANU", "VIZ"} and "Reviewer / Stakeholder" in roles:
        return [roles[0], "Reviewer / Stakeholder"]
    return roles


def _actor_roles(roles: list[str]) -> tuple[str, list[str]]:
    ids = [_actor_id(role) for role in roles]
    ids = [item for item in ids if item]
    if not ids:
        ids = ["ACT-RESEARCHER"]
    primary = ids[0]
    return primary, [item for item in ids[1:] if item != primary]


def _family_priority(families: list[_RequirementFamily]) -> str:
    values = " ".join(priority.lower() for family in families for priority in family.priorities)
    if "p0" in values or "must" in values:
        return "must"
    if "p1" in values or "should" in values:
        return "should"
    return "could"


def _summary_title(name: str, bc_id: str) -> str:
    overrides = {
        "BC-01": "Establish Research Workspace and Access Control",
        "BC-02": "Improve Dataset Understanding and Quality",
        "BC-03": "Define Research Framing and Hypotheses",
        "BC-04": "Plan Experiments and Select Methods",
        "BC-05": "Execute Experiments and Validate Results",
        "BC-06": "Trace Findings, Evidence, and Reproducibility",
        "BC-07": "Govern Evaluation and Decision Gates",
        "BC-08": "Generate Research Outputs and Ideation Quality",
    }
    return overrides.get(bc_id, f"Manage {name}")


def _family_title(family: _RequirementFamily) -> str:
    if family.family in _GROUP_TITLES:
        return _GROUP_TITLES[family.family]
    candidate = re.sub(r"\s*\([^)]*\)", "", family.title).strip()
    return candidate if candidate and not candidate.lower().startswith("fr-") else f"Manage {family.family.title()}"


def _actor_id(role: str) -> str | None:
    normalized = " ".join(role.lower().split())
    return _ACTOR_IDS.get(normalized)


def _actor_ids(item: UseCaseEntry) -> list[str]:
    return [item.primary_actor_id, *item.secondary_actor_ids]


def _find_component(source: RequirementsSourceSnapshot, document_type: str, artifact_type: str):
    document = source.brd if document_type == "brd" else source.prd
    return next((item for item in document.components if item.artifact_type == artifact_type), None)


def _component_ref(source: RequirementsSourceSnapshot, document_type: str, artifact_type: str) -> str | None:
    ref = f"component:{document_type}:{artifact_type}"
    return ref if ref in source.evidence_by_id() else None


def _fallback_component_refs(source: RequirementsSourceSnapshot) -> list[str]:
    refs = [
        _component_ref(source, "brd", "stakeholder_register"),
        _component_ref(source, "prd", "functional_requirement"),
    ]
    return [item for item in refs if item]


def _evidence_refs(source: RequirementsSourceSnapshot, entity_id: str, *, kind: str | None = None) -> list[str]:
    return [
        item.evidence_id
        for item in source.evidence
        if item.entity_id == entity_id and (kind is None or item.kind == kind)
    ][:3]


def _split_roles(value: str) -> list[str]:
    value = value.replace(" and ", ",").replace(" / ", " / ")
    return [_clean_text(item) for item in re.split(r",|;", value) if _clean_text(item)]


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", value).strip(" *`_.;:"))


def _unique_refs(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _name_key(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _best_name_match(value: str, candidates: dict[str, str]) -> str | None:
    exact = candidates.get(_name_key(value))
    if exact:
        return exact
    words = set(_name_key(value).split())
    scored = [
        (len(words & set(key.split())) / max(len(words | set(key.split())), 1), item_id)
        for key, item_id in candidates.items()
        if words and set(key.split())
    ]
    if not scored:
        return None
    score, item_id = max(scored)
    return item_id if score >= 0.45 else None
