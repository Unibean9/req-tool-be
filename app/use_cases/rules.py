"""Deterministic semantic validation for the canonical use-case model.

The LLM supplies meaning and evidence. This module checks the contract before PlantUML rendering:
IDs, enum values, endpoint types, relationship direction/conditions, actor coverage, and source
traceability. It does not impose L0/L1/L2 hierarchy and it never requires a human confirmation step.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

from app.use_cases.models import (
    RequirementsSourceSnapshot,
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
    "assess",
    "audit",
    "build",
    "capture",
    "check",
    "clean",
    "compare",
    "configure",
    "control",
    "create",
    "define",
    "delete",
    "establish",
    "evaluate",
    "execute",
    "authenticate",
    "flag",
    "generate",
    "govern",
    "handle",
    "inspect",
    "invite",
    "improve",
    "manage",
    "measure",
    "monitor",
    "open",
    "plan",
    "preserve",
    "profile",
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
    "trace",
    "upload",
    "validate",
    "version",
    "view",
    "receive",
    "start",
    "stop",
    "use",
    "retry",
    "search",
    "export",
    "import",
}
_VIETNAMESE_VERB_PREFIXES = {
    "áp dụng",
    "cập nhật",
    "cài đặt",
    "cảnh báo",
    "gợi ý",
    "ngăn chặn",
    "nhắc nhở",
    "phân quyền",
    "sửa",
    "tái phân công",
    "tạo",
    "thêm",
    "thông báo",
    "tích hợp",
    "truy cập",
    "xem",
    "xóa",
}
_TECHNICAL_TEXT_TERMS = (
    "database",
    "backend",
    "frontend",
    "server",
    "api",
    "endpoint",
    "click",
    "button",
    "screen",
    "table",
    "json",
)
_GENERIC_ACTORS = {"user", "customer", "person", "someone", "stakeholder"}
_INTERNAL_ACTOR_TERMS = ("database", "backend", "frontend", "server", "api", "engine", "module")


def validate_use_case_model(model: UseCaseModel, source: RequirementsSourceSnapshot) -> UseCaseValidationReport:
    issues: list[ValidationIssue] = []
    evidence = source.evidence_by_id()
    actors = {item.id: item for item in model.actors}
    modules = {item.id: item for item in model.modules}
    use_cases = {item.id: item for item in model.use_cases}
    relations = {item.id: item for item in model.relationships}

    _duplicate_ids(issues, "actor", list(actors))
    _duplicate_ids(issues, "module", list(modules))
    _duplicate_ids(issues, "use_case", list(use_cases))
    _duplicate_ids(issues, "relationship", list(relations))

    for actor in model.actors:
        _validate_refs(issues, actor.source_refs, evidence, f"actor[{actor.id}]")
        if actor.name.strip().lower() in _GENERIC_ACTORS:
            issues.append(
                _issue(
                    "warning",
                    "GENERIC_ACTOR",
                    "Use a concrete role or external system name when the source provides one.",
                    f"actor[{actor.id}]",
                )
            )
        if any(re.search(rf"\b{re.escape(term)}\b", actor.name.lower()) for term in _INTERNAL_ACTOR_TERMS):
            issues.append(
                _issue(
                    "error",
                    "INTERNAL_COMPONENT_ACTOR",
                    "Internal implementation components cannot be actors.",
                    f"actor[{actor.id}]",
                )
            )

    for module in model.modules:
        _validate_refs(issues, module.source_refs, evidence, f"module[{module.id}]")
        if not module.name.strip():
            issues.append(_issue("error", "EMPTY_MODULE_NAME", "Module name is required.", f"module[{module.id}]"))

    for item in model.use_cases:
        _validate_use_case(issues, item, actors=actors, modules=modules, evidence=evidence, source=source)

    seen_relations: set[tuple[str, str, str]] = set()
    for relation in model.relationships:
        _validate_relation(issues, relation, actors=actors, use_cases=use_cases, evidence=evidence)
        key = (relation.kind, relation.source_id, relation.target_id)
        if key in seen_relations:
            issues.append(
                _issue(
                    "error",
                    "DUPLICATE_RELATIONSHIP",
                    "Duplicate relationships are not rendered.",
                    f"relationship[{relation.id}]",
                )
            )
        seen_relations.add(key)

    include_sources: Counter[str] = Counter()
    include_graph: dict[str, set[str]] = {}
    for relation in model.relationships:
        if relation.kind == "include":
            include_sources[relation.target_id] += 1
            include_graph.setdefault(relation.source_id, set()).add(relation.target_id)
    for relation in model.relationships:
        if relation.kind == "include" and include_sources[relation.target_id] < 2:
            issues.append(
                _issue(
                    "warning",
                    "INCLUDE_NOT_SHARED",
                    "Use include only for reusable behavior shared by at least two base use cases.",
                    f"relationship[{relation.id}]",
                )
            )
    if _has_cycle(include_graph):
        issues.append(_issue("error", "INCLUDE_CYCLE", "Include relationships cannot form a cycle.", "relationships"))

    _validate_actor_coverage(issues, model, actors)
    if not model.use_cases:
        issues.append(
            _issue(
                "error",
                "NO_USE_CASES",
                "No source-backed use cases were found in the stored BRD/PRD components.",
                "use_cases",
            )
        )
    has_errors = any(issue.severity == "error" for issue in issues)
    return UseCaseValidationReport(
        issues=issues,
        eligible_for_srs=not has_errors,
        eligible_diagram_ids=["SYSTEM"] if model.use_cases and not has_errors else [],
        confirmed_use_case_ids=[],
    )


def _validate_use_case(issues, item: UseCaseEntry, *, actors, modules, evidence, source) -> None:
    path = f"use_case[{item.id}]"
    _validate_refs(issues, item.source_refs, evidence, path)
    if item.module_id not in modules:
        issues.append(_issue("error", "UNKNOWN_MODULE", f"Module {item.module_id} does not exist.", path))
    if item.primary_actor_id not in actors:
        issues.append(_issue("error", "UNKNOWN_ACTOR", f"Primary actor {item.primary_actor_id} does not exist.", path))
    for actor_id in item.secondary_actor_ids:
        if actor_id not in actors:
            issues.append(_issue("error", "UNKNOWN_ACTOR", f"Supporting actor {actor_id} does not exist.", path))
    if item.primary_actor_id in item.secondary_actor_ids:
        issues.append(_issue("error", "DUPLICATE_ACTOR_ROLE", "Primary actor cannot also be a supporting actor.", path))
    if not _active_verb_name(item.name) and not _source_backed_title(item, evidence):
        issues.append(
            _issue(
                "warning",
                "USE_CASE_NAME_STYLE",
                "Use Case names should use an active verb plus a business object.",
                path,
            )
        )
    text = " ".join([item.description, *item.preconditions, *item.business_rules]).lower()
    technical = [term for term in _TECHNICAL_TEXT_TERMS if re.search(rf"\b{re.escape(term)}\b", text)]
    if technical:
        issues.append(
            _issue(
                "error",
                "TECHNICAL_USE_CASE_TEXT",
                f"Detail contains implementation/UI terms: {', '.join(technical)}.",
                path,
            )
        )
    _validate_refs(issues, item.source_refs, evidence, path)
    for link in item.related_requirements:
        _validate_refs(issues, link.source_refs, evidence, f"{path}.related_requirements[{link.id}]", required=False)
    if item.evidence == "explicit" and not _supported_name(item.name, item.source_refs, source):
        issues.append(
            _issue(
                "warning",
                "USE_CASE_EVIDENCE_MISMATCH",
                "Explicit evidence should contain the use-case wording or requirement code.",
                path,
            )
        )
    if item.trigger and len(item.trigger) > 300:
        issues.append(_issue("error", "TRIGGER_TOO_LONG", "Trigger is too long.", path))
    flow_steps = list(item.main_flow)
    for flow in [*item.alternative_flows, *item.exception_flows]:
        flow_steps.extend(flow.steps)
    for step in flow_steps:
        if step.participant_type == "actor" and step.participant_id not in actors:
            issues.append(
                _issue("error", "FLOW_UNKNOWN_ACTOR", f"Flow references unknown actor {step.participant_id}.", path)
            )
        if step.participant_type == "external_system" and step.participant_id not in actors:
            issues.append(
                _issue(
                    "error",
                    "FLOW_UNKNOWN_EXTERNAL_SYSTEM",
                    f"Flow references unknown external system {step.participant_id}.",
                    path,
                )
            )
        if step.participant_type == "system" and step.participant_id not in {None, "SYSTEM"}:
            issues.append(
                _issue(
                    "warning",
                    "FLOW_SYSTEM_ID",
                    "System flow steps should use SYSTEM or omit participant_id.",
                    path,
                )
            )


def _validate_relation(issues, relation: UseCaseRelation, *, actors, use_cases, evidence) -> None:
    path = f"relationship[{relation.id}]"
    source_actor = relation.source_id in actors
    target_actor = relation.target_id in actors
    source_uc = relation.source_id in use_cases
    target_uc = relation.target_id in use_cases
    if relation.source_id == relation.target_id:
        issues.append(_issue("error", "SELF_RELATIONSHIP", "A relationship cannot point to itself.", path))
        return
    if relation.kind == "association":
        if not ((source_actor and target_uc) or (source_uc and target_actor)):
            issues.append(
                _issue("error", "ASSOCIATION_ENDPOINTS", "Association must connect one actor and one use case.", path)
            )
        return
    _validate_refs(issues, relation.source_refs, evidence, path)
    if relation.kind in {"include", "extend"} and not (source_uc and target_uc):
        issues.append(_issue("error", "RELATION_ENDPOINTS", f"{relation.kind} must connect two use cases.", path))
    elif relation.kind == "generalization" and not ((source_uc and target_uc) or (source_actor and target_actor)):
        issues.append(
            _issue(
                "error", "GENERALIZATION_ENDPOINTS", "Generalization must connect two actors or two use cases.", path
            )
        )
    if relation.kind == "extend" and not relation.condition:
        issues.append(
            _issue("error", "EXTEND_CONDITION_REQUIRED", "Extend needs a business condition or extension point.", path)
        )
    if relation.review_state == "rejected":
        issues.append(
            _issue(
                "warning",
                "RELATION_REJECTED",
                "Rejected relationship is retained for audit but excluded by the renderer.",
                path,
            )
        )


def _validate_actor_coverage(issues, model: UseCaseModel, actors) -> None:
    used = set()
    for item in model.use_cases:
        used.add(item.primary_actor_id)
        used.update(item.secondary_actor_ids)
    for actor_id in actors:
        if actor_id not in used:
            issues.append(
                _issue(
                    "warning",
                    "ACTOR_WITHOUT_USE_CASE",
                    f"Actor {actor_id} is not associated with a use case.",
                    f"actor[{actor_id}]",
                )
            )
    for item in model.use_cases:
        if item.primary_actor_id not in actors:
            continue
        # Actor IDs are the canonical association source; no separate association row is needed.
        if not item.primary_actor_id:
            issues.append(
                _issue(
                    "error", "USE_CASE_WITHOUT_ACTOR", "Every use case needs a primary actor.", f"use_case[{item.id}]"
                )
            )


def _validate_refs(issues, refs: Iterable[str], evidence: dict, path: str, *, required: bool = True) -> None:
    values = list(refs)
    if required and not values:
        issues.append(
            _issue("warning", "MISSING_SOURCE_REF", "Generated elements should cite stored BRD/PRD evidence.", path)
        )
    for ref in values:
        if ref not in evidence and ref not in {"internal"} and not ref.startswith("sha256:"):
            issues.append(_issue("error", "UNKNOWN_SOURCE_REF", f"Unknown evidence reference {ref!r}.", path))


def _supported_name(name: str, refs: Iterable[str], source: RequirementsSourceSnapshot) -> bool:
    evidence = source.evidence_by_id()
    if any(ref.startswith("entity:") for ref in refs):
        return True
    words = _significant_words(name)
    if not words:
        return False
    haystack = " ".join(evidence[ref].excerpt.lower() for ref in refs if ref in evidence)
    return all(word in haystack for word in words)


def _active_verb_name(value: str) -> bool:
    words = re.findall(r"[^\W\d_][\w-]*", value.lower(), flags=re.UNICODE)
    if not words or len(words) < 2:
        return False
    first_two = " ".join(words[:2])
    return (
        words[0] in _VERB_PREFIXES
        or words[0].endswith("ing")
        or words[0] in _VIETNAMESE_VERB_PREFIXES
        or first_two in _VIETNAMESE_VERB_PREFIXES
    )


def _source_backed_title(item: UseCaseEntry, evidence: dict) -> bool:
    """Accept a localized title when it is copied from a stored requirement row."""

    if item.evidence != "explicit":
        return False
    return any(ref in evidence and evidence[ref].kind in {
        "functional_requirement",
        "non_functional_requirement",
        "business_requirement",
        "business_rule",
    } for ref in item.source_refs)


def _significant_words(value: str) -> set[str]:
    return {
        word.lower()
        for word in re.findall(r"[A-Za-z][A-Za-z0-9]{3,}", value)
        if word.lower() not in {"manage", "system"}
    }


def _has_cycle(graph: dict[str, set[str]]) -> bool:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        if any(visit(child) for child in graph.get(node, ())):
            return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(visit(node) for node in graph)


def _duplicate_ids(issues, kind: str, ids: list[str]) -> None:
    for item_id, count in Counter(ids).items():
        if count > 1:
            issues.append(_issue("error", "DUPLICATE_ID", f"Duplicate {kind} id {item_id}.", f"{kind}s"))


def _issue(severity: str, code: str, message: str, path: str | None = None) -> ValidationIssue:
    return ValidationIssue(severity=severity, code=code, message=message, path=path)
