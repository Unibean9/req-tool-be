"""Selective predecessor context: load only the sections/columns an artifact actually needs.

Loading every accepted sibling/ancestor in full made later artifacts slow (input grows with each
one) and was still incomplete -- each body was cut at a fixed character count, which dropped whole
rows from a long table. Instead, CONTEXT_PICKS names, per artifact type, which `##` sections of
which predecessor to preload and, for tables, which columns -- the entry itself decides which
predecessors are loaded, so it can draw on a type outside the formal predecessor chain (e.g. a use
case's user segments come from the stakeholder register). Selection is deterministic: `##`
headings are the fixed English headings every artifact's output contract requires, and a table's
first column (its identifier: C1, BR-R1, O1, ...) is always kept so references stay traceable.
Anything not preloaded is still one `read_artifact(id, sections=[...])` call away.

Artifact types without an entry here keep the previous full-body preload.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class SectionPick:
    heading: str
    # Column keys (see COLUMN_KEYWORDS) to keep in this section's tables; empty keeps it verbatim.
    columns: tuple[str, ...] = ()


def _pick(heading: str, *columns: str) -> SectionPick:
    return SectionPick(heading, tuple(columns))


# artifact being drafted -> {predecessor artifact type -> sections to preload}
CONTEXT_PICKS: dict[str, dict[str, tuple[SectionPick, ...]]] = {
    "problem_statement": {},
    "vision_objectives": {
        "problem_statement": (
            _pick("Problem Statement"),
            _pick("Affected Users"),
            _pick("Impact"),
            _pick("Root Cause"),
        ),
    },
    "stakeholder_register": {
        "problem_statement": (_pick("Affected Users"),),
        "vision_objectives": (_pick("Vision"), _pick("Objectives", "objective", "value")),
    },
    "scope_capabilities": {
        "problem_statement": (_pick("Root Cause"),),
        "vision_objectives": (
            _pick("Objectives", "objective", "metric", "target", "timeframe"),
            _pick("Out of Scope"),
        ),
        "stakeholder_register": (_pick("Stakeholders", "role", "needs"),),
    },
    "business_rules": {
        "scope_capabilities": (_pick("Capabilities", "capability", "priority"), _pick("Out of Scope")),
        "stakeholder_register": (_pick("Stakeholders", "role", "responsibility", "decision"),),
    },
    "constraints_assumptions": {
        "problem_statement": (_pick("Assumptions"),),
        "vision_objectives": (_pick("Objectives", "objective", "target", "timeframe"), _pick("Assumptions")),
        "stakeholder_register": (_pick("Stakeholders", "role", "decision"), _pick("Assumptions")),
        "scope_capabilities": (
            _pick("Capabilities", "capability", "priority", "dependency"),
            _pick("Out of Scope"),
            _pick("Assumptions"),
        ),
        "business_rules": (_pick("Business Rules", "rule", "condition", "outcome"),),
    },
    # PRD
    "use_case": {
        "scope_capabilities": (
            _pick("Capabilities", "capability", "priority", "rationale", "dependency"),
            _pick("Out of Scope"),
        ),
        "vision_objectives": (_pick("Objectives", "objective", "value"),),
        "stakeholder_register": (_pick("Stakeholders", "role", "needs"),),
    },
    "functional_requirement": {
        "use_case": (_pick("Business Capabilities", "capability", "objective", "segment", "priority"),),
        "business_rules": (_pick("Business Rules", "rule", "condition", "outcome"),),
    },
    "non_functional_requirement": {
        "constraints_assumptions": (
            _pick("Constraints", "constraint", "type", "impact"),
            _pick("Risks", "risk", "impact", "severity"),
        ),
        "vision_objectives": (_pick("Success Metrics"),),
        "use_case": (_pick("Business Capabilities", "capability", "priority"),),
    },
    # Event Storming (flow chain: use case -> command -> event -> policy / aggregate)
    "actor_command": {
        "use_case": (_pick("Business Capabilities", "capability", "objective", "segment"),),
        "stakeholder_register": (_pick("Stakeholders", "role", "responsibility"),),
        "business_rules": (_pick("Business Rules", "rule", "condition"),),
    },
    "domain_event": {
        "actor_command": (_pick("Actors and Commands"),),
        "business_rules": (_pick("Business Rules", "rule", "condition", "outcome"),),
    },
    "policy": {
        "domain_event": (_pick("Domain Events"),),
        "business_rules": (_pick("Business Rules", "rule", "condition", "outcome"),),
    },
    "aggregate": {
        "domain_event": (_pick("Domain Events"),),
        "actor_command": (_pick("Actors and Commands", "command", "event"),),
        "business_rules": (_pick("Business Rules", "rule", "condition"),),
    },
    # ADD
    "tech_stack": {
        "constraints_assumptions": (_pick("Constraints", "constraint", "type", "impact"),),
        "non_functional_requirement": (
            _pick("Non-Functional Requirements", "quality", "requirement", "measurement"),
        ),
    },
    "domain_entity": {
        "functional_requirement": (_pick("Functional Requirements", "requirement", "io"),),
        "aggregate": (_pick("Aggregates", "responsibility", "invariant"),),
        "use_case": (_pick("Business Capabilities", "capability"),),
    },
    "component": {
        "domain_entity": (_pick("Domain Entity"), _pick("Responsibilities"), _pick("Relationships")),
        "functional_requirement": (_pick("Functional Requirements", "requirement"),),
        "aggregate": (_pick("Aggregates", "responsibility", "command", "event"),),
    },
    "interface": {
        "component": (_pick("Component"), _pick("Interfaces"), _pick("Dependencies")),
        "domain_entity": (_pick("Domain Entity"), _pick("Attributes")),
    },
    "tech_decision": {
        "component": (_pick("Component"), _pick("Responsibilities"), _pick("Constraints")),
        "non_functional_requirement": (
            _pick("Non-Functional Requirements", "quality", "requirement", "measurement"),
        ),
        "tech_stack": (_pick("Tech Stack", "category", "choice", "rationale"),),
        "constraints_assumptions": (_pick("Constraints", "constraint", "impact"),),
        "aggregate": (_pick("Aggregates", "responsibility"),),
    },
}

# Table headers are written in the project's language (often Vietnamese), so a column is matched by
# any of these substrings rather than by the contract's English column name alone.
COLUMN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "role": ("role", "vai trò", "stakeholder"),
    "responsibility": ("responsibilit", "trách nhiệm"),
    "decision": ("decision", "quyết định"),
    "needs": ("need", "concern", "nhu cầu", "mối quan tâm"),
    "objective": ("objective", "goal", "mục tiêu"),
    "value": ("value", "giá trị"),
    "metric": ("metric", "kpi", "chỉ số"),
    "target": ("target", "chỉ tiêu"),
    "timeframe": ("timeframe", "thời hạn", "thời gian", "deadline"),
    "capability": ("capabilit", "năng lực", "chức năng"),
    "priority": ("priority", "ưu tiên", "moscow"),
    "dependency": ("dependenc", "phụ thuộc"),
    "rule": ("rule", "quy tắc"),
    "condition": ("condition", "trigger", "điều kiện"),
    "outcome": ("outcome", "result", "kết quả"),
    "rationale": ("rationale", "reason", "lý do"),
    "segment": ("segment", "user", "actor", "người dùng", "đối tượng"),
    "constraint": ("constraint", "ràng buộc"),
    "type": ("type", "loại", "category"),
    "impact": ("impact", "tác động", "ảnh hưởng"),
    "risk": ("risk", "rủi ro"),
    "severity": ("severity", "level", "mức độ"),
    "requirement": ("requirement", "yêu cầu"),
    "io": ("input", "output", "đầu vào", "đầu ra"),
    "quality": ("quality", "chất lượng", "thuộc tính"),
    "measurement": ("measurement", "đo lường", "kiểm chứng"),
    "command": ("command", "lệnh"),
    "event": ("event", "sự kiện"),
    "invariant": ("invariant", "bất biến"),
    "category": ("category", "hạng mục", "loại"),
    "choice": ("choice", "lựa chọn"),
}

# Generous safety bound only; with column selection an excerpt is normally far below it.
EXCERPT_MAX_CHARS = 12000

_SECTION_RE = re.compile(r"^##(?!#)\s*(.+?)\s*$")
_ID_RE = re.compile(r"^[A-Z]{1,5}(?:-[A-Z]{1,5})?-?\d{1,3}$")


def _normalize_heading(value: str) -> str:
    return re.sub(r"[\s\-_]+", " ", value).strip().casefold()


def split_sections(body: str) -> list[tuple[str, str]]:
    """`##` sections as (heading, full text including the heading line). `###` stays inside."""

    sections: list[tuple[str, list[str]]] = []
    for line in body.splitlines():
        match = _SECTION_RE.match(line)
        if match:
            sections.append((match.group(1), [line]))
        elif sections:
            sections[-1][1].append(line)
    return [(heading, "\n".join(lines).rstrip()) for heading, lines in sections]


def section_headings(body: str) -> list[str]:
    return [heading for heading, _text in split_sections(body)]


def _matches(heading: str, wanted: str) -> bool:
    return _normalize_heading(heading).startswith(_normalize_heading(wanted))


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _is_separator(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and "-" in stripped and set(stripped) <= set("|-: ")


def _filter_table(lines: list[str], columns: tuple[str, ...]) -> list[str]:
    header = _cells(lines[0])
    keywords = [keyword for key in columns for keyword in COLUMN_KEYWORDS.get(key, (key,))]
    keep = [0] + [
        index
        for index, cell in enumerate(header)
        if index > 0 and any(keyword in cell.casefold() for keyword in keywords)
    ]
    if len(keep) == 1:
        # None of the wanted columns exist in this table: keep it whole rather than guess.
        return lines
    result = []
    for line in lines:
        if _is_separator(line):
            result.append("|" + "|".join("---" for _ in keep) + "|")
            continue
        cells = _cells(line)
        result.append("| " + " | ".join(cells[index] if index < len(cells) else "" for index in keep) + " |")
    return result


_FIELD_LINE_RE = re.compile(r"^\s*[-*]\s+\*\*(.+?):?\*\*:?")


def _filter_entry_fields(lines: list[str], keywords: list[str]) -> list[str]:
    """Entry-style items ("### EVT-01: ..." then "- **trigger:** ...") keep only the wanted field
    lines; headings and other text stay. If no field line matches, the entries are kept whole."""

    def wanted(line: str) -> bool | None:
        match = _FIELD_LINE_RE.match(line)
        if match is None:
            return None
        return any(keyword in match.group(1).casefold() for keyword in keywords)

    verdicts = [wanted(line) for line in lines]
    if not any(verdicts):
        return lines
    return [line for line, verdict in zip(lines, verdicts, strict=True) if verdict is not False]


def _filter_section(text: str, columns: tuple[str, ...]) -> str:
    if not columns:
        return text
    keywords = [keyword for key in columns for keyword in COLUMN_KEYWORDS.get(key, (key,))]
    lines = _filter_entry_fields(text.splitlines(), keywords)
    output: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.strip().startswith("|") and index + 1 < len(lines) and _is_separator(lines[index + 1]):
            end = index + 2
            while end < len(lines) and lines[end].strip().startswith("|"):
                end += 1
            output.extend(_filter_table(lines[index:end], columns))
            index = end
            continue
        output.append(line)
        index += 1
    return "\n".join(output)


@dataclass(frozen=True)
class Excerpt:
    text: str
    included: list[str]
    omitted: list[str]
    truncated: bool


def excerpt_body(body: str, picks: Iterable[SectionPick]) -> Excerpt:
    """The picked sections of `body`, tables narrowed to the picked columns. If none of the picked
    headings exist (a body not written to its contract), the whole body is used instead."""

    sections = split_sections(body)
    picks = list(picks)
    chosen: list[str] = []
    included: list[str] = []
    for heading, text in sections:
        pick = next((item for item in picks if _matches(heading, item.heading)), None)
        if pick is None:
            continue
        chosen.append(_filter_section(text, pick.columns))
        included.append(heading)
    omitted = [heading for heading, _text in sections if heading not in included]
    text = "\n\n".join(chosen) if chosen else body
    if not chosen:
        included, omitted = ["(whole artifact)"], []
    truncated = len(text) > EXCERPT_MAX_CHARS
    return Excerpt(text=text[:EXCERPT_MAX_CHARS], included=included, omitted=omitted, truncated=truncated)


def select_sections(body: str, wanted: Iterable[str]) -> str | None:
    """Only the `##` sections whose heading starts with one of `wanted`; None if none match."""

    wanted = [item for item in wanted if str(item).strip()]
    chosen = [text for heading, text in split_sections(body) if any(_matches(heading, item) for item in wanted)]
    return "\n\n".join(chosen) if chosen else None


def table_ids(body: str) -> set[str]:
    """Identifiers in the first column of the body's tables (C1, BR-R1, CON-1, ...)."""

    ids: set[str] = set()
    for line in body.splitlines():
        if not line.strip().startswith("|") or _is_separator(line):
            continue
        first = re.sub(r"[*_`]", "", _cells(line)[0])
        if _ID_RE.match(first):
            ids.add(first)
    return ids


def _id_prefix(identifier: str) -> str:
    return re.sub(r"\d+$", "", identifier)


def unknown_reference_warnings(draft_body: str, predecessor_bodies: dict[str, str], limit: int = 10) -> list[str]:
    """IDs the draft cites in a predecessor's ID scheme (C7, BR-R9, ...) that the predecessor does
    not define. The draft's own ID schemes (the ids it defines itself) are not checked."""

    own_prefixes = {_id_prefix(identifier) for identifier in table_ids(draft_body)}
    known_by_prefix: dict[str, set[str]] = {}
    source_by_prefix: dict[str, str] = {}
    for artifact_type, body in predecessor_bodies.items():
        for identifier in table_ids(body):
            prefix = _id_prefix(identifier)
            if prefix in own_prefixes:
                continue
            known_by_prefix.setdefault(prefix, set()).add(identifier)
            source_by_prefix.setdefault(prefix, artifact_type)
    warnings: list[str] = []
    for prefix in sorted(known_by_prefix, key=len, reverse=True):
        pattern = re.compile(rf"(?<![A-Za-z0-9-]){re.escape(prefix)}\d{{1,3}}(?![A-Za-z0-9])")
        for cited in sorted(set(pattern.findall(draft_body))):
            if cited not in known_by_prefix[prefix]:
                warnings.append(
                    f"unknown_reference: {cited} is not defined in {source_by_prefix[prefix]}"
                )
    return warnings[:limit]
