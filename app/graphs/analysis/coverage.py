"""Deterministic coverage: a downstream artifact must address every item of the upstream section it
is derived from (every in-scope capability gets business rules and functional requirements, every
constraint is reflected in the non-functional requirements).

"Addressed" means the draft cites the item's ID at least once -- in a row derived from it, or with a
stated reason it does not apply. Items prioritised "Won't" are exempt. Nothing here calls an LLM, so
it can gate a proposal without adding latency.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from app.graphs.analysis.context_excerpts import COLUMN_KEYWORDS, select_sections

# artifact being drafted -> (upstream artifact type, upstream `##` section it must cover)
COVERAGE_RULES: dict[str, tuple[str, str]] = {
    "use_case": ("scope_capabilities", "Capabilities"),
    "business_rules": ("scope_capabilities", "Capabilities"),
    "functional_requirement": ("use_case", "Business Capabilities"),
    "non_functional_requirement": ("constraints_assumptions", "Constraints"),
}

# An ID at the start of a table's first cell: C1, BC1, CON-1, BR-R1, C-LEG-01, FR-AUTH-01, NFR-PE01.
_LEADING_ID_RE = re.compile(r"^([A-Z]{1,5}(?:-[A-Z]{1,6})*-?\d{1,3})(?![A-Za-z0-9])")
_WONT_RE = re.compile(r"\bwon['’]?t\b|\bwont\b", re.IGNORECASE)
_EXEMPT_HEADING_RE = re.compile(r"\bwon['’]?t\b|\bwont\b|out[\s-]*of[\s-]*scope|ngoài phạm vi", re.IGNORECASE)
_LABEL_MAX_CHARS = 80


@dataclass(frozen=True)
class CoverageItem:
    id: str
    label: str


def _plain(text: str) -> str:
    return re.sub(r"[*_`]", "", text).strip()


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _is_separator(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and "-" in stripped and set(stripped) <= set("|-: ")


def _family(identifier: str) -> str:
    return re.match(r"[A-Z]+", identifier).group(0)


def _priority_column(header: list[str]) -> int | None:
    keywords = COLUMN_KEYWORDS["priority"]
    return next(
        (index for index, cell in enumerate(header) if any(keyword in cell.casefold() for keyword in keywords)),
        None,
    )


def coverage_items(source_body: str, section: str) -> list[CoverageItem]:
    """The in-scope items of `section`: rows whose first cell starts with an ID, restricted to the
    section's main ID family (a stray assumptions table inside the section is not a capability)."""

    text = select_sections(source_body or "", [section])
    if not text:
        return []
    items: list[CoverageItem] = []
    seen: set[str] = set()
    exempt_heading = False
    priority_index: int | None = None
    lines = text.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#"):
            exempt_heading = bool(_EXEMPT_HEADING_RE.search(stripped))
            continue
        if not stripped.startswith("|") or _is_separator(line):
            continue
        cells = _cells(line)
        if index + 1 < len(lines) and _is_separator(lines[index + 1]):
            priority_index = _priority_column(cells)
            continue
        match = _LEADING_ID_RE.match(_plain(cells[0]))
        if match is None:
            continue
        identifier = match.group(1)
        wont = priority_index is not None and priority_index < len(cells) and _WONT_RE.search(cells[priority_index])
        if exempt_heading or wont or identifier in seen:
            continue
        seen.add(identifier)
        label = _plain(cells[0])[len(identifier) :].strip(" -—:") or (_plain(cells[1]) if len(cells) > 1 else "")
        items.append(CoverageItem(identifier, label[:_LABEL_MAX_CHARS]))
    if not items:
        return []
    main_family = Counter(_family(item.id) for item in items).most_common(1)[0][0]
    return [item for item in items if _family(item.id) == main_family]


def _table_blocks(lines: list[str]) -> list[tuple[int, int]]:
    """(start, end) line ranges of each Markdown table: a header row, its separator, then rows."""
    blocks: list[tuple[int, int]] = []
    index = 0
    while index < len(lines):
        if lines[index].strip().startswith("|") and index + 1 < len(lines) and _is_separator(lines[index + 1]):
            end = index + 2
            while end < len(lines) and lines[end].strip().startswith("|"):
                end += 1
            blocks.append((index, end))
            index = end
            continue
        index += 1
    return blocks


def item_rows(source_body: str, section: str, ids: list[str]) -> str:
    """The full rows (every column, with their table header) of the given items in `section`."""

    text = select_sections(source_body or "", [section]) or ""
    wanted = set(ids)
    lines = text.splitlines()
    output: list[str] = []
    for start, end in _table_blocks(lines):
        rows = [
            line
            for line in lines[start + 2 : end]
            if (match := _LEADING_ID_RE.match(_plain(_cells(line)[0]))) and match.group(1) in wanted
        ]
        if rows:
            output.extend([lines[start], lines[start + 1], *rows, ""])
    return "\n".join(output).strip()


_SCOPE_KEYWORDS = ("scope", "phạm vi")
_SUMMARY_MAX_CHARS = 240


def item_summaries(source_body: str, section: str) -> dict[str, str]:
    """One line per item of `section`: its name and, when the table has one, its scope column --
    enough for a writer of one item to tell what belongs to another."""

    text = select_sections(source_body or "", [section]) or ""
    lines = text.splitlines()
    summaries: dict[str, str] = {}
    for start, end in _table_blocks(lines):
        header = [cell.casefold() for cell in _cells(lines[start])]
        scope_index = next(
            (index for index, cell in enumerate(header) if any(keyword in cell for keyword in _SCOPE_KEYWORDS)), None
        )
        for line in lines[start + 2 : end]:
            cells = [_plain(cell) for cell in _cells(line)]
            match = _LEADING_ID_RE.match(cells[0])
            if match is None or match.group(1) in summaries:
                continue
            identifier = match.group(1)
            name = cells[0][len(identifier) :].strip(" -—:") or (cells[1] if len(cells) > 1 else "")
            scope = cells[scope_index] if scope_index is not None and scope_index < len(cells) else ""
            summary = f"{name} -- scope: {scope}" if scope and scope != name else name
            summaries[identifier] = summary[:_SUMMARY_MAX_CHARS]
    return summaries


def cited_ids(text: str) -> set[str]:
    """IDs cited anywhere in `text` (C1, BR-R1, C-LEG-01, ...)."""
    return set(re.findall(r"(?<![A-Za-z0-9-])[A-Z]{1,5}(?:-[A-Z]{1,6})*-?\d{1,3}(?![A-Za-z0-9])", text or ""))


def rows_citing(text: str, related_ids: set[str]) -> str:
    """Narrow `text` to what concerns `related_ids`: in each table, keep only the rows citing one of
    them. A text that never cites an ID of the same schemes (it is not organised by these items, e.g.
    business rules written without capability links) is returned whole -- it cannot be narrowed
    without guessing, and dropping it could lose a rule that applies."""

    families = {_family(identifier) for identifier in related_ids}
    if not any(_family(identifier) in families for identifier in cited_ids(text)):
        return text
    patterns = [_citation_pattern(identifier) for identifier in related_ids]
    lines = text.splitlines()
    blocks = _table_blocks(lines)
    keep = [True] * len(lines)
    for start, end in blocks:
        for index in range(start + 2, end):
            keep[index] = any(pattern.search(lines[index]) for pattern in patterns)
        if not any(keep[start + 2 : end]):
            keep[start:end] = [False] * (end - start)
    return "\n".join(line for line, kept in zip(lines, keep, strict=True) if kept)


def _citation_pattern(identifier: str) -> re.Pattern[str]:
    # C-LEG-01 also matches CLEG01 / C-LEG-1; BC1 also matches BC-01. Not preceded by a letter, digit
    # or hyphen, so C1 is not found inside BC1, BR-C1 or C10.
    parts = re.findall(r"[A-Z]+|\d+", identifier)
    body = "-?".join(re.escape(part) if part.isalpha() else f"0*{int(part)}" for part in parts)
    return re.compile(rf"(?<![A-Za-z0-9-]){body}(?![0-9A-Za-z])")


def missing_coverage(draft_body: str, items: list[CoverageItem]) -> list[CoverageItem]:
    return [item for item in items if not _citation_pattern(item.id).search(draft_body or "")]


def coverage_instruction(artifact_type: str, items: list[CoverageItem]) -> str | None:
    """The rule stated up front, so the model covers everything on its first write."""

    rule = COVERAGE_RULES.get(artifact_type)
    if rule is None or not items:
        return None
    source, section = rule
    listed = ", ".join(item.id for item in items)
    return (
        f"Coverage rule: this draft must address every item of {source} > {section}: {listed}. "
        "Cite each ID in the rows derived from it. If an item genuinely does not apply here, still list "
        "its ID with the reason. A draft that leaves any of these IDs out is rejected."
    )


def coverage_gap_message(artifact_type: str, missing: list[CoverageItem]) -> str:
    source, section = COVERAGE_RULES[artifact_type]
    listed = "; ".join(f"{item.id} ({item.label})" if item.label else item.id for item in missing)
    return (
        f"The draft does not address these {source} > {section} items: {listed}. "
        "Write the rows for each (cite its ID), or state per ID why it does not apply. "
        "Write the content out in full; do not replace it with a reference."
    )
