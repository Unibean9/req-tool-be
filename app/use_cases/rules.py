"""Deterministic use-case rules from the supplied SRS guide and reference article.

The LLM may propose names and relations, but this module is the authority that decides whether a
model is traceable and renderable.  Unsupported or out-of-scope proposals are reported as issues;
the API can persist the reviewable result without exposing it as an eligible SRS diagram.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

from app.use_cases.models import (
    RequirementsSourceSnapshot,
    SourceEvidence,
    UseCaseDiagramDefinition,
    UseCaseEntry,
    UseCaseModel,
    UseCaseRelation,
    UseCaseValidationReport,
    ValidationIssue,
)

_VERB_PREFIXES = {
    "access",
    "analyze",
    "approve",
    "assign",
    "build",
    "capture",
    "check",
    "compare",
    "configure",
    "control",
    "create",
    "define",
    "delete",
    "evaluate",
    "execute",
    "generate",
    "handle",
    "inspect",
    "invite",
    "manage",
    "measure",
    "monitor",
    "open",
    "pay",
    "plan",
    "preserve",
    "publish",
    "record",
    "refine",
    "register",
    "reject",
    "remove",
    "request",
    "review",
    "run",
    "save",
    "select",
    "send",
    "submit",
    "track",
    "upload",
    "validate",
    "version",
    "view",
    "receive",
    "start",
    "stop",
    "use",
}
_TECHNICAL_OR_UI_PREFIXES = {
    "call",
    "click",
    "display",
    "enter",
    "load",
    "open",
    "save",
    "send",
    "store",
}
_SIGNIFICANT_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]{3,}")
_GENERIC_ACTORS = {"user", "customer", "person", "someone"}
_INTERNAL_ACTOR_TERMS = (
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
_TECHNICAL_TEXT_TERMS = (
    "database",
    "backend",
    "frontend",
    "server",
    "api",
    "ai model",
    "ai agent",
    "click",
    "button",
    "screen",
    "endpoint",
)


def validate_use_case_model(
    model: UseCaseModel,
    source: RequirementsSourceSnapshot,
    *,
    require_confirmed: bool = False,
) -> UseCaseValidationReport:
    """Validate traceability, naming, UML direction, hierarchy, and diagram size."""

    issues: list[ValidationIssue] = []
    evidence_by_id = source.evidence_by_id()
    actors_by_id = {actor.id: actor for actor in model.actors}
    subsystems_by_id = {subsystem.id: subsystem for subsystem in model.subsystems}
    use_cases_by_id = {item.id: item for item in model.use_cases}
    relations_by_id = {relation.id: relation for relation in model.relations}

    _duplicate_ids(issues, "actor", [actor.id for actor in model.actors])
    _duplicate_ids(issues, "subsystem", [item.id for item in model.subsystems])
    _duplicate_ids(issues, "use_case", [item.id for item in model.use_cases])
    _duplicate_ids(issues, "relation", [item.id for item in model.relations])
    _duplicate_ids(issues, "diagram", [item.id for item in model.diagrams])

    for actor in model.actors:
        _validate_refs(issues, actor.source_refs, evidence_by_id, f"actor[{actor.id}]")
        _validate_actor_name(issues, actor.name, actor.id)
        if not _supported_name(actor.name, actor.source_refs, source):
            issues.append(
                _issue(
                    "error",
                    "ACTOR_NOT_SUPPORTED",
                    f"Actor {actor.name!r} is not supported by its cited stored BRD/PRD evidence.",
                    f"actor[{actor.id}]",
                )
            )

    for subsystem in model.subsystems:
        _validate_refs(issues, subsystem.source_refs, evidence_by_id, f"subsystem[{subsystem.id}]")
        if not _supported_name(subsystem.name, subsystem.source_refs, source):
            issues.append(
                _issue(
                    "error",
                    "SUBSYSTEM_NOT_SUPPORTED",
                    f"Subsystem {subsystem.name!r} is not supported by its cited stored BRD/PRD evidence.",
                    f"subsystem[{subsystem.id}]",
                )
            )

    for item in model.use_cases:
        _validate_use_case(
            issues,
            item,
            actors_by_id=actors_by_id,
            subsystems_by_id=subsystems_by_id,
            relations_by_id=relations_by_id,
            evidence_by_id=evidence_by_id,
            source=source,
        )

    _validate_use_case_hierarchy(issues, model, use_cases_by_id)

    for relation in model.relations:
        _validate_relation(
            issues,
            relation,
            actors_by_id=actors_by_id,
            use_cases_by_id=use_cases_by_id,
            evidence_by_id=evidence_by_id,
        )

    include_targets: dict[str, set[str]] = {}
    for relation in model.relations:
        if relation.kind == "include":
            include_targets.setdefault(relation.target_id, set()).add(relation.source_id)
    for relation in model.relations:
        if relation.kind != "include" or len(include_targets.get(relation.target_id, set())) >= 2:
            continue
        issues.append(
            _issue(
                "error",
                "INCLUDE_NOT_SHARED",
                f"Included use case {relation.target_id} must be reused by at least two base use cases.",
                f"relation[{relation.id}]",
            )
        )

    for item in model.use_cases:
        for relation_id in item.relationship_ids:
            if relation_id not in relations_by_id:
                issues.append(
                    _issue(
                        "error",
                        "UNKNOWN_RELATIONSHIP",
                        f"Use case {item.id} references unknown relationship {relation_id}.",
                        f"use_case[{item.id}].relationship_ids",
                    )
                )

    _validate_actor_and_goal_coverage(issues, model, actors_by_id, use_cases_by_id)
    _validate_diagrams(issues, model, actors_by_id, use_cases_by_id, relations_by_id)

    confirmed_ids = [item.id for item in model.use_cases if item.status == "confirmed"]
    error_free = not any(issue.severity == "error" for issue in issues)
    eligible_diagram_ids = [
        diagram.id
        for diagram in model.diagrams
        if _diagram_has_no_errors(diagram, issues)
        and all(use_cases_by_id.get(item_id, None) is not None for item_id in diagram.use_case_ids)
        and all(use_cases_by_id[item_id].status == "confirmed" for item_id in diagram.use_case_ids)
    ]
    all_confirmed = bool(model.use_cases) and len(confirmed_ids) == len(model.use_cases)
    if model.use_cases and not all_confirmed:
        issues.append(
            _issue(
                "warning",
                "HUMAN_CONFIRMATION_REQUIRED",
                "Inferred or suggested use cases are not eligible for the SRS diagrams until a human confirms them.",
                "use_cases",
            )
        )
    if require_confirmed and not all_confirmed:
        issues.append(
            _issue(
                "error",
                "UNCONFIRMED_MODEL",
                "This operation requires every use case to have status=confirmed.",
                "use_cases",
            )
        )

    return UseCaseValidationReport(
        issues=issues,
        eligible_for_srs=error_free and all_confirmed and bool(model.diagrams),
        eligible_diagram_ids=eligible_diagram_ids,
        confirmed_use_case_ids=confirmed_ids,
    )


def _validate_use_case(
    issues: list[ValidationIssue],
    item: UseCaseEntry,
    *,
    actors_by_id: dict[str, object],
    subsystems_by_id: dict[str, object],
    relations_by_id: dict[str, UseCaseRelation],
    evidence_by_id: dict[str, SourceEvidence],
    source: RequirementsSourceSnapshot,
) -> None:
    path = f"use_case[{item.id}]"
    _validate_refs(issues, item.source_refs, evidence_by_id, path)
    if item.primary_actor_id not in actors_by_id:
        issues.append(_issue("error", "UNKNOWN_ACTOR", f"Primary actor {item.primary_actor_id} does not exist.", path))
    for actor_id in item.secondary_actor_ids:
        if actor_id not in actors_by_id:
            issues.append(_issue("error", "UNKNOWN_ACTOR", f"Secondary actor {actor_id} does not exist.", path))
    if item.subsystem_id not in subsystems_by_id:
        issues.append(_issue("error", "UNKNOWN_SUBSYSTEM", f"Subsystem {item.subsystem_id} does not exist.", path))
    _validate_name(issues, item)
    _validate_business_language(issues, item)
    if not _supported_name(item.name, item.source_refs, source):
        severity = "error" if item.status == "confirmed" else "warning"
        issues.append(
            _issue(
                severity,
                "USE_CASE_NOT_SUPPORTED",
                f"Use case {item.id} ({item.name!r}) is not supported by the cited stored BRD/PRD evidence.",
                path,
            )
        )
    out_of_scope = [
        evidence.excerpt
        for evidence in source.evidence
        if evidence.kind == "out_of_scope"
    ]
    if _matches_out_of_scope(item.name, out_of_scope):
        issues.append(
            _issue(
                "error",
                "OUT_OF_SCOPE_USE_CASE",
                f"Use case {item.id} overlaps an explicit BRD/PRD out-of-scope item.",
                path,
            )
        )
    if item.level == "L0" and item.abstraction != "summary":
        issues.append(_issue("error", "L0_MUST_BE_SUMMARY", "L0 use cases must be summary abstractions.", path))
    if item.level == "L1" and item.abstraction != "user_goal":
        issues.append(_issue("error", "L1_MUST_BE_USER_GOAL", "L1 use cases must be user goals.", path))
    has_shared_include = any(
        relation.kind == "include"
        and (relation.source_id == item.id or relation.target_id == item.id)
        for relation in relations_by_id.values()
    )
    if item.level == "L2" and item.abstraction == "subfunction" and not has_shared_include:
        issues.append(
            _issue(
                "error",
                "L2_SUBFUNCTION_NOT_SHARED",
                "An L2 subfunction must participate in an explicit shared relationship such as include.",
                path,
            )
        )
    if item.level == "L2" and item.abstraction == "subfunction" and item.priority == "must":
        issues.append(
            _issue(
                "warning",
                "SUBFUNCTION_PRIORITY",
                "A subfunction is normally kept inside a user goal; review whether it needs its own row.",
                path,
            )
        )
    if item.name.lower().startswith("manage "):
        crud_words = ("create", "update", "deactivate", "delete", "view", "read")
        if item.abstraction != "summary" or sum(word in item.description.lower() for word in crud_words) < 2:
            issues.append(
                _issue(
                    "error",
                    "MANAGE_NOT_ABSTRACT_CRUD",
                    "Manage X is allowed only as a summary of at least two explicitly listed CRUD-style operations.",
                    path,
                )
            )


def _validate_use_case_hierarchy(
    issues: list[ValidationIssue],
    model: UseCaseModel,
    use_cases_by_id: dict[str, UseCaseEntry],
) -> None:
    level_order = {"L0": 0, "L1": 1, "L2": 2}
    for item in model.use_cases:
        path = f"use_case[{item.id}].parent_use_case_id"
        if item.parent_use_case_id is None:
            if item.level == "L2" and item.abstraction != "subfunction":
                issues.append(
                    _issue(
                        "warning",
                        "L2_PARENT_REVIEW",
                        "A detailed L2 user goal should normally identify its L1 parent.",
                        path,
                    )
                )
            continue
        if item.parent_use_case_id == item.id:
            issues.append(_issue("error", "USE_CASE_SELF_PARENT", "A use case cannot parent itself.", path))
            continue
        parent = use_cases_by_id.get(item.parent_use_case_id)
        if parent is None:
            issues.append(
                _issue(
                    "error",
                    "UNKNOWN_PARENT_USE_CASE",
                    f"Unknown parent use case {item.parent_use_case_id}.",
                    path,
                )
            )
            continue
        if level_order[parent.level] >= level_order[item.level]:
            issues.append(
                _issue(
                    "error",
                    "INVALID_USE_CASE_LEVEL_PARENT",
                    f"Parent {parent.id} must be at a higher abstraction level than {item.id}.",
                    path,
                )
            )
        if parent.subsystem_id != item.subsystem_id:
            issues.append(
                _issue(
                    "warning",
                    "CROSS_SUBSYSTEM_PARENT",
                    "Review a parent relationship that crosses subsystem boundaries.",
                    path,
                )
            )


def _validate_name(issues: list[ValidationIssue], item: UseCaseEntry) -> None:
    words = _SIGNIFICANT_WORD_RE.findall(item.name.lower())
    path = f"use_case[{item.id}].name"
    if len(words) < 2:
        issues.append(_issue("error", "USE_CASE_NAME_TOO_SHORT", "Use-case names must be Verb + Object.", path))
        return
    first = words[0]
    if first in _TECHNICAL_OR_UI_PREFIXES:
        issues.append(
            _issue(
                "error",
                "TECHNICAL_OR_UI_USE_CASE",
                f"{item.name!r} starts with a UI/implementation step; use the actor's business goal.",
                path,
            )
        )
    elif first not in _VERB_PREFIXES:
        issues.append(
            _issue(
                "warning",
                "NAME_MAY_NOT_BE_VERB_OBJECT",
                f"{item.name!r} could not be verified as an active Verb + Object name.",
                path,
            )
        )
    if item.name.lower().startswith("login ") or item.name.lower() == "login":
        issues.append(
            _issue(
                "warning",
                "LOGIN_PRECONDITION",
                "Login/authentication should normally be a standalone goal or a precondition, not a repeated include.",
                path,
            )
        )


def _validate_business_language(issues: list[ValidationIssue], item: UseCaseEntry) -> None:
    text = f"{item.description} {item.precondition}".lower()
    technical_terms = [
        term
        for term in _TECHNICAL_TEXT_TERMS
        if re.search(rf"\b{re.escape(term)}\b", text)
    ]
    if technical_terms:
        issues.append(
            _issue(
                "error",
                "TECHNICAL_USE_CASE_TEXT",
                "Use-case descriptions and preconditions must describe a business goal, not UI or implementation steps "
                f"({', '.join(technical_terms)}).",
                f"use_case[{item.id}]",
            )
        )


def _validate_actor_name(issues: list[ValidationIssue], name: str, actor_id: str) -> None:
    normalized = " ".join(name.lower().split())
    path = f"actor[{actor_id}].name"
    if normalized in _GENERIC_ACTORS:
        issues.append(
            _issue(
                "error",
                "GENERIC_ACTOR",
                "Use a concrete role or external system name instead of a generic User/Customer actor.",
                path,
            )
        )
    if any(re.search(rf"\b{re.escape(token)}\b", normalized) for token in _INTERNAL_ACTOR_TERMS):
        issues.append(
            _issue(
                "error",
                "INTERNAL_COMPONENT_ACTOR",
                "Database, backend, server, module, and the platform's own AI are not external actors.",
                path,
            )
        )


def _validate_relation(
    issues: list[ValidationIssue],
    relation: UseCaseRelation,
    *,
    actors_by_id: dict[str, object],
    use_cases_by_id: dict[str, UseCaseEntry],
    evidence_by_id: dict[str, SourceEvidence],
) -> None:
    path = f"relation[{relation.id}]"
    _validate_refs(issues, relation.source_refs, evidence_by_id, path, required=relation.kind != "association")
    if relation.source_id == relation.target_id:
        issues.append(_issue("error", "SELF_RELATION", "A relation cannot point to itself.", path))
        return
    source_is_actor = relation.source_id in actors_by_id
    target_is_actor = relation.target_id in actors_by_id
    source_is_use_case = relation.source_id in use_cases_by_id
    target_is_use_case = relation.target_id in use_cases_by_id
    if relation.kind == "association":
        if not ((source_is_actor and target_is_use_case) or (source_is_use_case and target_is_actor)):
            issues.append(
                _issue(
                    "error",
                    "ASSOCIATION_ENDPOINTS",
                    "Association may connect only one actor and one use case.",
                    path,
                )
            )
        return
    if relation.kind in {"include", "extend"} and not (source_is_use_case and target_is_use_case):
        issues.append(
            _issue("error", "RELATION_ENDPOINTS", f"{relation.kind} may connect only two use cases.", path)
        )
        return
    if relation.kind == "include":
        target = use_cases_by_id[relation.target_id]
        if target.abstraction != "subfunction":
            issues.append(
                _issue(
                    "warning",
                    "INCLUDE_TARGET_NOT_SUBFUNCTION",
                    "An include target is normally a reusable L2 subfunction rather than a user-goal summary.",
                    path,
                )
            )
    if relation.kind == "extend":
        source = use_cases_by_id[relation.source_id]
        target = use_cases_by_id[relation.target_id]
        if source.abstraction == "summary" or target.abstraction == "subfunction":
            issues.append(
                _issue(
                    "warning",
                    "EXTEND_ABSTRACTION_REVIEW",
                    "Review extend direction: an optional user-goal extension should extend a complete base goal.",
                    path,
                )
            )
    if relation.kind == "generalization" and not (
        (source_is_actor and target_is_actor) or (source_is_use_case and target_is_use_case)
    ):
        issues.append(
            _issue(
                "error",
                "GENERALIZATION_ENDPOINTS",
                "Generalization must connect two actors or two use cases.",
                path,
            )
        )
    if relation.kind == "extend" and not relation.condition:
        issues.append(
            _issue(
                "error",
                "EXTEND_CONDITION_REQUIRED",
                "Every extend relation must state its business condition/extension point.",
                path,
            )
        )
    if relation.kind == "include":
        target_name = use_cases_by_id.get(relation.target_id)
        if target_name and "login" in target_name.name.lower():
            issues.append(
                _issue(
                    "error",
                    "LOGIN_MUST_NOT_BE_INCLUDED",
                    "Authentication is a precondition or standalone use case, not a repeated include target.",
                    path,
                )
            )


def _validate_actor_and_goal_coverage(
    issues: list[ValidationIssue],
    model: UseCaseModel,
    actors_by_id: dict[str, object],
    use_cases_by_id: dict[str, UseCaseEntry],
) -> None:
    associated_actor_ids: set[str] = set()
    direct_actors_by_use_case: dict[str, set[str]] = {use_case_id: set() for use_case_id in use_cases_by_id}
    for relation in model.relations:
        if relation.kind != "association":
            continue
        if relation.source_id in actors_by_id and relation.target_id in use_cases_by_id:
            associated_actor_ids.add(relation.source_id)
            direct_actors_by_use_case[relation.target_id].add(relation.source_id)
        elif relation.target_id in actors_by_id and relation.source_id in use_cases_by_id:
            associated_actor_ids.add(relation.target_id)
            direct_actors_by_use_case[relation.source_id].add(relation.target_id)

    # An included or extended goal may inherit the actor context of its base goal.  The same
    # propagation applies to use-case generalization (child -> parent).  Keep the fixed-point
    # calculation small and deterministic because the model is bounded by diagram-size rules.
    effective_actors_by_use_case = {
        use_case_id: set(actor_ids) for use_case_id, actor_ids in direct_actors_by_use_case.items()
    }
    for _ in range(len(use_cases_by_id) + 1):
        changed = False
        for relation in model.relations:
            if relation.kind in {"include", "extend"}:
                source_id, target_id = relation.source_id, relation.target_id
                if relation.kind == "extend":
                    source_id, target_id = target_id, source_id
            elif relation.kind == "generalization" and relation.source_id in use_cases_by_id:
                source_id, target_id = relation.target_id, relation.source_id
            else:
                continue
            inherited = effective_actors_by_use_case.get(source_id, set())
            target_actors = effective_actors_by_use_case.get(target_id)
            if target_actors is None:
                continue
            before = len(target_actors)
            target_actors.update(inherited)
            changed = changed or len(target_actors) != before
        if not changed:
            break

    # Actor generalization lets a specialized actor inherit the parent's associations.  Count
    # both endpoints as covered once either side has a usable association.
    actor_generalizations = [
        relation
        for relation in model.relations
        if relation.kind == "generalization"
        and relation.source_id in actors_by_id
        and relation.target_id in actors_by_id
    ]
    for _ in range(len(actors_by_id) + 1):
        changed = False
        for relation in actor_generalizations:
            endpoints = {relation.source_id, relation.target_id}
            if not endpoints & associated_actor_ids:
                continue
            before = len(associated_actor_ids)
            associated_actor_ids.update(endpoints)
            changed = changed or len(associated_actor_ids) != before
        if not changed:
            break

    for actor_id in actors_by_id:
        if actor_id not in associated_actor_ids:
            issues.append(
                _issue(
                    "error",
                    "ACTOR_WITHOUT_ASSOCIATION",
                    f"Actor {actor_id} has no association.",
                    f"actor[{actor_id}]",
                )
            )
    for use_case_id in use_cases_by_id:
        effective_actor_ids = effective_actors_by_use_case[use_case_id]
        if not effective_actor_ids:
            issues.append(
                _issue(
                    "error",
                    "USE_CASE_WITHOUT_ACTOR",
                    f"Use case {use_case_id} has no direct or inherited actor association.",
                    f"use_case[{use_case_id}]",
                )
            )
            continue
        item = use_cases_by_id[use_case_id]
        if item.primary_actor_id not in effective_actor_ids:
            issues.append(
                _issue(
                    "error",
                    "PRIMARY_ACTOR_ASSOCIATION",
                    f"Use case {use_case_id} must have a direct or inherited association with its primary actor "
                    f"{item.primary_actor_id}.",
                    f"use_case[{use_case_id}]",
                )
            )
        for actor_id in item.secondary_actor_ids:
            if actor_id not in effective_actor_ids:
                issues.append(
                    _issue(
                        "error",
                        "SECONDARY_ACTOR_ASSOCIATION",
                        f"Use case {use_case_id} must have a direct or inherited association with secondary actor "
                        f"{actor_id}.",
                        f"use_case[{use_case_id}]",
                    )
                )


def _validate_diagrams(
    issues: list[ValidationIssue],
    model: UseCaseModel,
    actors_by_id: dict[str, object],
    use_cases_by_id: dict[str, UseCaseEntry],
    relations_by_id: dict[str, UseCaseRelation],
) -> None:
    level_counts = Counter(diagram.level for diagram in model.diagrams)
    if level_counts["L0"] != 1:
        issues.append(
            _issue(
                "error",
                "ONE_L0_REQUIRED",
                "The model must contain exactly one L0 overview diagram.",
                "diagrams",
            )
        )
    if not model.diagrams:
        issues.append(
            _issue(
                "error",
                "NO_DIAGRAMS",
                "At least one semantic diagram definition is required.",
                "diagrams",
            )
        )

    l1_by_subsystem = Counter(
        diagram.subsystem_id
        for diagram in model.diagrams
        if diagram.level == "L1" and diagram.subsystem_id is not None
    )
    for subsystem_id, count in l1_by_subsystem.items():
        if count != 1:
            issues.append(
                _issue(
                    "error",
                    "ONE_L1_PER_SUBSYSTEM",
                    f"Subsystem {subsystem_id} must have exactly one L1 diagram; found {count}.",
                    "diagrams",
                )
            )
    represented_subsystems = {subsystem.id for subsystem in model.subsystems}
    for subsystem_id in represented_subsystems:
        if l1_by_subsystem[subsystem_id] != 1:
            issues.append(
                _issue(
                    "error",
                    "MISSING_L1_PER_SUBSYSTEM",
                    f"Subsystem {subsystem_id} has L1 use cases but no single L1 diagram.",
                    "diagrams",
                )
            )

    for diagram in model.diagrams:
        path = f"diagram[{diagram.id}]"
        if diagram.level == "L0" and diagram.subsystem_id is not None:
            issues.append(
                _issue(
                    "error",
                    "L0_SUBSYSTEM_SCOPE",
                    "L0 is the system overview and cannot be scoped to one subsystem.",
                    path,
                )
            )
        if diagram.level == "L1" and diagram.subsystem_id is None:
            issues.append(
                _issue(
                    "error",
                    "L1_SUBSYSTEM_REQUIRED",
                    "Every L1 diagram must identify one subsystem.",
                    path,
                )
            )
        unknown_actor_ids = [item for item in diagram.actor_ids if item not in actors_by_id]
        unknown_use_case_ids = [item for item in diagram.use_case_ids if item not in use_cases_by_id]
        unknown_relation_ids = [item for item in diagram.relation_ids if item not in relations_by_id]
        if unknown_actor_ids:
            issues.append(_issue("error", "DIAGRAM_UNKNOWN_ACTOR", f"Unknown actors: {unknown_actor_ids}.", path))
        if unknown_use_case_ids:
            issues.append(
                _issue(
                    "error",
                    "DIAGRAM_UNKNOWN_USE_CASE",
                    f"Unknown use cases: {unknown_use_case_ids}.",
                    path,
                )
            )
        if unknown_relation_ids:
            issues.append(
                _issue(
                    "error",
                    "DIAGRAM_UNKNOWN_RELATION",
                    f"Unknown relations: {unknown_relation_ids}.",
                    path,
                )
            )
        diagram_node_ids = set(diagram.actor_ids) | set(diagram.use_case_ids)
        for relation_id in diagram.relation_ids:
            relation = relations_by_id.get(relation_id)
            if relation is None:
                continue
            if relation.source_id not in diagram_node_ids or relation.target_id not in diagram_node_ids:
                issues.append(
                    _issue(
                        "error",
                        "DIAGRAM_RELATION_CROSS_SCOPE",
                        f"Relation {relation_id} has an endpoint outside diagram {diagram.id}.",
                        path,
                    )
                )
        if len(diagram.use_case_ids) > 12:
            issues.append(
                _issue(
                    "warning",
                    "DIAGRAM_TOO_MANY_USE_CASES",
                    "Keep a diagram at roughly 5–12 use cases.",
                    path,
                )
            )
        if len(diagram.actor_ids) > 6:
            issues.append(
                _issue(
                    "warning",
                    "DIAGRAM_TOO_MANY_ACTORS",
                    "Keep a diagram at no more than six actors.",
                    path,
                )
            )
        relation_count = sum(
            1
            for relation_id in diagram.relation_ids
            if relation_id in relations_by_id and relations_by_id[relation_id].kind in {"include", "extend"}
        )
        if relation_count > 5:
            issues.append(
                _issue(
                    "warning",
                    "DIAGRAM_TOO_MANY_RELATIONS",
                    "Keep include/extend relations to roughly five or fewer.",
                    path,
                )
            )
        for use_case_id in diagram.use_case_ids:
            item = use_cases_by_id.get(use_case_id)
            if item is None:
                continue
            if diagram.level != item.level:
                issues.append(
                    _issue(
                        "error",
                        "DIAGRAM_LEVEL_MISMATCH",
                        f"Diagram {diagram.id} contains {use_case_id} at level {item.level}, not {diagram.level}.",
                        path,
                    )
                )
            if diagram.level == "L1" and diagram.subsystem_id != item.subsystem_id:
                issues.append(
                    _issue(
                        "error",
                        "DIAGRAM_SUBSYSTEM_MISMATCH",
                        f"L1 diagram {diagram.id} contains a use case from another subsystem.",
                        path,
                    )
                )


def _validate_refs(
    issues: list[ValidationIssue],
    refs: Iterable[str],
    evidence_by_id: dict[str, SourceEvidence],
    path: str,
    *,
    required: bool = True,
) -> None:
    refs_list = list(refs)
    if required and not refs_list:
        issues.append(
            _issue(
                "error",
                "MISSING_SOURCE_REF",
                "Every generated element must cite stored BRD/PRD evidence.",
                path,
            )
        )
    for ref in refs_list:
        if ref not in evidence_by_id:
            issues.append(_issue("error", "UNKNOWN_SOURCE_REF", f"Unknown stored-evidence reference {ref!r}.", path))


def _supported_name(name: str, refs: Iterable[str], source: RequirementsSourceSnapshot) -> bool:
    evidence_by_id = source.evidence_by_id()
    referenced = [evidence_by_id[ref] for ref in refs if ref in evidence_by_id]
    if not referenced:
        return False
    needle = _significant_words(name)
    if not needle:
        return False
    source_texts: list[str] = []
    for evidence in referenced:
        source_texts.append(evidence.excerpt.lower())
        for document in (source.brd, source.prd):
            if document.document_type == evidence.document_type and document.document_type == evidence.artifact_type:
                source_texts.append(document.container_body.lower())
        for component in source.components:
            if component.artifact_type == evidence.artifact_type and component.document_type == evidence.document_type:
                source_texts.append(component.body.lower())
    haystack = " ".join(source_texts)
    return all(word in haystack for word in needle)


def _significant_words(value: str) -> set[str]:
    return {word.lower() for word in _SIGNIFICANT_WORD_RE.findall(value) if word.lower() not in {"manage", "system"}}


def _matches_out_of_scope(name: str, excerpts: Iterable[str]) -> bool:
    words = _significant_words(name)
    if not words:
        return False
    for excerpt in excerpts:
        lowered = excerpt.lower()
        if len(words) >= 2 and all(word in lowered for word in words):
            return True
    return False


def _duplicate_ids(issues: list[ValidationIssue], kind: str, ids: list[str]) -> None:
    counts = Counter(ids)
    for item_id, count in counts.items():
        if count > 1:
            issues.append(_issue("error", "DUPLICATE_ID", f"Duplicate {kind} id {item_id}.", f"{kind}s"))


def _diagram_has_no_errors(diagram: UseCaseDiagramDefinition, issues: list[ValidationIssue]) -> bool:
    prefix = f"diagram[{diagram.id}]"
    related_paths = {
        *(f"actor[{actor_id}]" for actor_id in diagram.actor_ids),
        *(f"use_case[{use_case_id}]" for use_case_id in diagram.use_case_ids),
        *(f"relation[{relation_id}]" for relation_id in diagram.relation_ids),
    }
    return not any(
        issue.severity == "error"
        and (
            issue.path == prefix
            or (issue.path or "").startswith(f"{prefix}.")
            or issue.path in related_paths
            or any((issue.path or "").startswith(f"{path}.") for path in related_paths)
        )
        for issue in issues
    )


def _issue(severity: str, code: str, message: str, path: str | None = None) -> ValidationIssue:
    return ValidationIssue(severity=severity, code=code, message=message, path=path)  # type: ignore[arg-type]
