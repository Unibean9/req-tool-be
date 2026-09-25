"""Seed the AI Research Experimentation Platform BRD/PRD demo workspace.

Idempotent for the dedicated organization/project slugs below. Source Markdown is
kept verbatim in ``source_documents``; document-section artifacts are normalized to
the active BE registry contracts so the FE can render them and the approval/readiness
workflow can consume the same shape as UI-created sections.

Usage (from req-tool-be):
    uv run python scripts/seed_research_platform_demo.py
    uv run python scripts/seed_research_platform_demo.py --github-login ThinhTP204
    SEED_GITHUB_LOGIN=ThinhTP204 task seed:demo

The no-selector form is automatic only when exactly one real GitHub OAuth user exists in the
local database.  When several people have logged in locally, select the intended owner explicitly
so the seed cannot be attached to the wrong account.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import re
import sys
from pathlib import Path

from sqlalchemy import func, select

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.core.utils import slugify
from app.database import async_session_factory
from app.documents.registry import get_config, output_contract
from app.models.artifact import (
    Artifact,
    ArtifactLink,
    ArtifactReview,
    ArtifactStatus,
    ArtifactType,
    ArtifactVersion,
    ChangeSource,
    RelationType,
    ReviewStatus,
    SourceDocument,
    SourceType,
    VersionStatus,
)
from app.models.organization import Organization, OrgMember
from app.models.project import Project
from app.models.user import User
from app.schemas.artifact import ArtifactReviewRequest
from app.services.artifact_service import ArtifactVersionService

ROOT = Path(__file__).resolve().parents[1]
BRD_PATH = ROOT / "docs/AI-Research-Experimentation-Platform-BRD-v1.8-REQ-TOOL.md"
PRD_PATH = ROOT / "docs/AI-Research-Experimentation-Platform-PRD-v1.7-REQ-TOOL.md"

SEED_KEY = "ai-research-experimentation-platform-demo-v1"
ORG_NAME = "AI Research Experimentation Platform"
ORG_SLUG = "ai-research-experimentation"
PROJECT_NAME = "AI Research Experimentation Platform"
PROJECT_SLUG = "ai-research-experimentation-platform"
SEED_REVIEW_COMMENT = "Approved during demo seed import after BRD/PRD section mapping."
SEED_OWNER_EMAIL_ENV = "SEED_OWNER_EMAIL"
SEED_GITHUB_LOGIN_ENV = "SEED_GITHUB_LOGIN"


def split_h2_sections(markdown: str) -> tuple[str, dict[str, str]]:
    matches = list(re.finditer(r"(?m)^##\s+(.+?)\s*$", markdown))
    if not matches:
        raise ValueError("Expected at least one level-two Markdown heading")

    preamble = markdown[: matches[0].start()].strip()
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        title = match.group(1).strip()
        if title in sections:
            raise ValueError(f"Duplicate level-two heading: {title}")
        sections[title] = markdown[match.start() : end].strip()
    return preamble, sections


def strip_leading_h2(markdown: str) -> str:
    """Remove the source item's level-two title before nesting it in a contract section."""
    return re.sub(r"^##\s+.+?(?:\r?\n|$)", "", markdown.strip(), count=1).strip()


def extract_heading_body(markdown: str, title: str, *, level: int = 3) -> str:
    """Return a Markdown heading's body until the next heading at the same or higher level."""
    marker = "#" * level
    match = re.search(rf"(?m)^{re.escape(marker)}\s+{re.escape(title)}\s*$", markdown)
    if match is None:
        return ""
    next_heading = re.search(rf"(?m)^#{{1,{level}}}\s+", markdown[match.end() :])
    end = match.end() + next_heading.start() if next_heading else len(markdown)
    return markdown[match.end() : end].strip()


def extract_heading_bodies(markdown: str, *, level: int = 3) -> dict[str, str]:
    """Extract all headings at ``level`` while retaining lower-level source headings."""
    marker = "#" * level
    matches = list(re.finditer(rf"(?m)^{re.escape(marker)}\s+(.+?)\s*$", markdown))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        sections[match.group(1).strip()] = markdown[match.end() : end].strip()
    return sections


def contract_body(*sections: tuple[str, str]) -> str:
    """Render item content with the exact level-two headings required by the registry contract."""
    rendered: list[str] = []
    for title, content in sections:
        rendered.append(f"## {title}\n\n{content.strip()}".rstrip())
    return "\n\n".join(rendered).strip() + "\n"


def validate_contract_sections(sections: dict[str, tuple[str, list[str]]]) -> None:
    """Fail fast when a source edit no longer satisfies the active BE document registry."""
    for item_type, (body, _source_sections) in sections.items():
        required = output_contract(item_type).required_headings
        present = {line.strip() for line in body.splitlines() if line.startswith("## ")}
        missing = [heading for heading in required if heading not in present]
        if missing:
            raise ValueError(
                f"Normalized {item_type} section does not satisfy the registry contract: "
                f"missing={missing}"
            )


def strip_source_comments(markdown: str) -> str:
    """Drop authoring-only source annotations from the content shown in the FE."""
    return re.sub(r"<!--.*?-->", "", str(markdown or ""), flags=re.DOTALL).strip()


def clean_markdown_cell(value: str, *, limit: int | None = None) -> str:
    """Turn source prose/lists into a safe compact Markdown table-cell value."""
    text = strip_source_comments(value)
    text = re.sub(r"```(?:[A-Za-z0-9_+-]+)?", "", text)
    text = text.replace("```", "")
    parts: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line == "---":
            continue
        line = re.sub(r"^#{1,6}\s+", "", line)
        line = re.sub(r"^(?:[-*]|\d+[.)])\s+", "", line)
        line = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
        line = re.sub(r"`([^`]+)`", r"\1", line)
        parts.append(line)
    result = "<br>".join(parts).strip()
    if limit is not None and len(result) > limit:
        return result[: limit - 1].rstrip() + "…"
    return result


def clean_markdown_text(value: str, *, limit: int | None = None) -> str:
    """Flatten source prose for a paragraph or a single bullet."""
    return clean_markdown_cell(value, limit=limit).replace("<br>", " ")


def markdown_table(columns: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    """Render the canonical table shape consumed by the document contracts."""
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rendered_rows = []
    for row in rows:
        cells = [str(value or "").replace("|", "\\|").replace("\n", "<br>").strip() for value in row]
        if len(cells) != len(columns):
            raise ValueError(f"Expected {len(columns)} cells, got {len(cells)}")
        rendered_rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, separator, *rendered_rows])


def split_markdown_table_row(line: str) -> list[str]:
    """Split a GFM row while preserving escaped pipe characters in a cell."""
    content = line.strip().strip("|")
    return [cell.replace("\\|", "|").strip() for cell in re.split(r"(?<!\\)\|", content)]


def parse_markdown_table(markdown: str) -> list[dict[str, str]]:
    """Parse simple GFM tables from the source docs without adding a table dependency."""
    lines = [line.strip() for line in str(markdown or "").splitlines() if line.strip().startswith("|")]
    if len(lines) < 2:
        return []
    headers = split_markdown_table_row(lines[0])
    rows: list[dict[str, str]] = []
    for line in lines[1:]:
        cells = split_markdown_table_row(line)
        if not cells or all(re.fullmatch(r":?-{2,}:?", cell.replace(" ", "")) for cell in cells):
            continue
        cells += [""] * (len(headers) - len(cells))
        rows.append(dict(zip(headers, cells[: len(headers)], strict=False)))
    return rows


def extract_heading_entries(markdown: str, *, level: int) -> list[tuple[str, str]]:
    """Return ordered heading/body pairs, stopping each body at the next same/higher heading."""
    marker = "#" * level
    matches = list(re.finditer(rf"(?m)^{re.escape(marker)}\s+(.+?)\s*$", str(markdown or "")))
    entries: list[tuple[str, str]] = []
    for _index, match in enumerate(matches):
        next_heading = re.search(rf"(?m)^#{{1,{level}}}\s+", markdown[match.end() :])
        end = match.end() + next_heading.start() if next_heading else len(markdown)
        entries.append((match.group(1).strip(), markdown[match.end() : end].strip()))
    return entries


def extract_h4_entries(markdown: str) -> list[tuple[str, str]]:
    return extract_heading_entries(markdown, level=4)


def extract_h5_entries(markdown: str) -> list[tuple[str, str]]:
    return extract_heading_entries(markdown, level=5)


def source_id_and_title(title: str, prefix: str) -> tuple[str, str]:
    match = re.match(rf"({re.escape(prefix)}[A-Za-z0-9-]*)\s*(?:—\s*)?(.*)$", title.strip())
    if match is None:
        return title.strip(), title.strip()
    name = match.group(2).lstrip(": ").strip()
    return match.group(1).strip(), name or match.group(1).strip()


def source_bullets(markdown: str, *, limit: int | None = None) -> list[str]:
    """Extract source list/paragraph lines as clean, readable bullets."""
    result: list[str] = []
    for line in strip_source_comments(markdown).splitlines():
        stripped = line.strip()
        if not stripped or stripped == "---" or stripped.startswith("```"):
            continue
        if stripped.startswith("#"):
            continue
        if stripped.startswith("|"):
            continue
        cleaned = clean_markdown_text(stripped, limit=limit)
        if cleaned:
            result.append(cleaned)
    return result


def module_capability(module_key: str) -> str:
    """Map PRD feature modules to the existing PRD business-capability IDs."""
    groups = {
        "BC-01": {"A", "B"},
        "BC-02": {"C", "D", "E"},
        "BC-03": {"F", "G", "AC"},
        "BC-04": {"H", "I", "J"},
        "BC-05": {"K", "L", "M", "N", "O", "P", "U"},
        "BC-06": {"Q", "R", "S", "T", "X", "Y", "Z"},
        "BC-07": {"AA", "AB", "AD", "AE", "AF", "AG"},
        "BC-08": {"V", "W", "AH"},
    }
    for capability, modules in groups.items():
        if module_key in modules:
            return capability
    return "BC-01"


def acceptance_signal(acceptance: str, source_id: str) -> str:
    criteria = source_bullets(acceptance, limit=260)
    if not criteria:
        return (
            f"Given the module prerequisites are met, when {source_id} runs, "
            "then the behavior is observable and traceable."
        )
    first = "<br>".join(criteria[:2])
    return f"Given the module prerequisites are met, when {source_id} runs, then {first}"


def input_output_signal(requirement: str, source_id: str) -> str:
    values: list[str] = []
    for line in str(requirement or "").splitlines():
        if re.match(r"(?i)^\s*(inputs?|outputs?|fields?|requires?|produces?|result):", line.strip()):
            values.append(clean_markdown_text(line, limit=180))
    if values:
        return "<br>".join(values[:3])
    return f"Input: project context and referenced data<br>Output: persisted, traceable result for {source_id}"


def normalize_brd_sections(
    markdown: str,
    mapped: dict[str, tuple[str, list[str]]],
) -> dict[str, tuple[str, list[str]]]:
    """Convert the BRD sample into the six compact, FE-facing document item shapes."""
    _preamble, source = split_h2_sections(markdown)
    executive_summary = source["Executive Summary"]
    problem = source["Problem Statement"]
    vision = source["Vision and Objectives"]
    stakeholder_source = source["Stakeholder Register"]
    scope_source = source["Scope and Capabilities"]
    rules_source = source["Business Rules"]
    constraints_source = source["Constraints, Assumptions, and Risks"]

    positioning = extract_heading_body(executive_summary, "High-Level Product Positioning")
    if not positioning:
        positioning = extract_heading_body(executive_summary, "Executive Summary")
    objective_entries = extract_h4_entries(extract_heading_body(vision, "Business Objectives"))
    objective_bullets = "\n".join(
        f"- **{title}:** {clean_markdown_text(body, limit=420)}" for title, body in objective_entries
    )

    metric_rows: list[tuple[str, ...]] = []
    metric_table = parse_markdown_table(extract_heading_body(vision, "Success Metrics"))
    for row in metric_table:
        metric_id = row.get("ID", "KPI")
        metric = row.get("Metric", "")
        target = row.get("Target đề xuất", row.get("Target", ""))
        timeframe = (
            "Benchmark evaluation"
            if "benchmark" in metric.lower() or "evaluation" in metric.lower()
            else "Per applicable workflow"
        )
        if "finding" in metric.lower() or "evidence" in metric.lower():
            timeframe = "Before official finding"
        elif "experiment" in metric.lower() or "reproducibility" in metric.lower() or "trace" in metric.lower():
            timeframe = "Each official experiment"
        elif "destructive" in metric.lower() or "approval" in metric.lower():
            timeframe = "Every risky change"
        metric_rows.append(
            (
                f"{metric_id}: {row.get('Objective', '').strip()}",
                f"Supports {row.get('Objective', '').strip().lower()}",
                metric,
                target,
                timeframe,
            )
        )
    success_metrics = markdown_table(
        ("goal", "user/business value", "metric", "target", "timeframe"), metric_rows
    )

    problem_entries = extract_h4_entries(extract_heading_body(problem, "Business Problem Statement"))
    problem_lines = "\n".join(
        f"- **{title}:** {clean_markdown_text(body, limit=520)}" for title, body in problem_entries
    )
    stakeholder_entries = extract_h4_entries(extract_heading_body(stakeholder_source, "Stakeholders"))
    affected_users = "\n".join(
        f"- **{re.sub(r'^\d+(?:\.\d+)*\.\s*', '', title)}:** "
        f"{clean_markdown_text(extract_heading_body(body, 'Nhu cầu', level=5), limit=380)}"
        for title, body in stakeholder_entries
    )
    pain_entries = extract_h4_entries(extract_heading_body(problem, "Current State — As-Is"))
    impact_lines = "\n".join(
        f"- **{title}:** {clean_markdown_text(body, limit=360)}" for title, body in pain_entries
    )
    root_cause_lines = "\n".join(
        [
            "- Workflow is fragmented across profiling, cleaning, analysis, reporting, and experiment tracking.",
            "- Method selection and assumption checking depend on specialist knowledge and are easy to skip.",
            "- Hypotheses, findings, provenance, dependencies, and conflicting evidence are difficult "
            "to maintain across iterations.",
            "- Repeated tests, post-hoc hypotheses, leakage, and unstructured LLM decisions create "
            "scientific-validity risk.",
        ]
    )

    scope_body = extract_heading_body(scope_source, "Scope")
    scope_entries = extract_h4_entries(scope_body)
    scope_groups: list[str] = []
    for title, body in scope_entries:
        if title.startswith("9.4."):
            continue
        subgroups = extract_heading_entries(body, level=5)
        if subgroups:
            subgroup_lines = "\n".join(
                f"- **{sub_title}:** {clean_markdown_text(sub_body, limit=520)}"
                for sub_title, sub_body in subgroups
            )
            scope_groups.append(f"### {title}\n\n{subgroup_lines}")
        else:
            scope_groups.append(f"### {title}\n\n{clean_markdown_text(body, limit=700)}")
    scope_lines = "\n\n".join(scope_groups)
    out_of_scope_entry = next((body for title, body in scope_entries if title.startswith("9.4.")), "")
    out_of_scope = "\n".join(f"- {item}" for item in source_bullets(out_of_scope_entry, limit=260))

    priority_rows = parse_markdown_table(
        extract_heading_body(scope_source, "Business Requirement Prioritization — MoSCoW")
    )
    priority_by_id = {row.get("BR", ""): row for row in priority_rows}
    capability_groups = (
        ("Workspace & access control", range(1, 4), "User identity and project membership"),
        ("Dataset understanding and quality", range(4, 13), "Workspace & access control"),
        ("Research framing", range(13, 17), "Dataset understanding and quality"),
        ("Planning and method selection", range(17, 21), "Research framing"),
        ("Execution and scientific validation", range(21, 26), "Planning and method selection"),
        ("Findings, provenance, and reporting", range(26, 43), "Execution and scientific validation"),
        ("Evaluation and metrics", range(43, 47), "Findings, provenance, and reporting"),
        ("Scientific validity and reproducibility", range(47, 56), "Findings, provenance, and reporting"),
        ("Research state, decision gates, and ideation", (*range(56, 66), 73, 74, 75, 78), "Research framing"),
    )
    capability_rows: list[tuple[str, ...]] = []
    for label, numbers, dependency in capability_groups:
        rows = [priority_by_id.get(f"BR-{number:02d}") for number in numbers]
        rows = [row for row in rows if row]
        priorities = list(dict.fromkeys(row.get("Priority", "") for row in rows if row.get("Priority")))
        priority = " / ".join(priorities) if priorities else "Must"
        rationale = "; ".join(dict.fromkeys(row.get("Rationale", "") for row in rows if row.get("Rationale")))
        references = f"BR-{numbers[0]:02d}–BR-{numbers[-1]:02d}"
        capability_rows.append(
            (f"{label} ({references})", priority, clean_markdown_text(rationale, limit=320), dependency)
        )
    capabilities = markdown_table(("capability", "priority", "rationale", "dependency"), capability_rows)

    rule_entries = extract_h4_entries(extract_heading_body(rules_source, "Business Rules"))
    rule_rows: list[tuple[str, ...]] = []
    for title, body in rule_entries:
        rule_id, rule_name = source_id_and_title(title, "BRule-")
        condition = clean_markdown_text(rule_name, limit=220)
        outcome = clean_markdown_text(body, limit=420)
        exception = "Explicit exception/override is not stated in the source."
        exception_match = re.search(r"(?i)(?:unless|except|ngoại lệ|nếu)[:\s]+(.+)", body)
        if exception_match:
            exception = clean_markdown_text(exception_match.group(1), limit=220)
        rule_rows.append(
            (
                rule_id,
                condition,
                f"When the {condition.lower()} condition is evaluated",
                outcome,
                "Project research workflow",
                exception,
            )
        )
    business_rules = markdown_table(
        ("rule id", "condition", "trigger", "outcome", "scope", "exception"), rule_rows
    )

    constraints = extract_heading_body(constraints_source, "Constraints")
    assumptions = extract_heading_body(constraints_source, "Assumptions")
    assumptions_rows = [
        (
            clean_markdown_text(item, limit=320),
            "Affects planning, interpretation, or governance",
            "BRD §18 Assumptions",
            "Confirm during project setup and before the affected workflow runs",
        )
        for item in source_bullets(assumptions, limit=320)
    ]
    nfr_direction = extract_heading_body(
        extract_heading_body(constraints_source, "High-Level Non-Functional Requirements"),
        "NFR Acceptance Direction",
        level=4,
    )
    validation_plan = "\n".join(f"- {item}" for item in source_bullets(nfr_direction, limit=360))
    risk_table = parse_markdown_table(extract_heading_body(constraints_source, "Risks & Mitigation"))
    risk_rows = [
        (
            row.get("Risk", ""),
            "High" if row.get("Impact", "").lower() == "high" else "Medium",
            row.get("Impact", ""),
            row.get("Mitigation", ""),
            "Open",
        )
        for row in risk_table
    ]
    mitigation_rows = [
        (row.get("Risk", ""), row.get("Mitigation", ""), "Project owner / review at each iteration")
        for row in risk_table
    ]

    return {
        "vision_objectives": (
            contract_body(
                ("Vision", clean_markdown_text(positioning, limit=900)),
                ("Objectives", objective_bullets),
                ("Success Metrics", success_metrics),
            ),
            mapped["vision_objectives"][1],
        ),
        "problem_statement": (
            contract_body(
                ("Problem Statement", problem_lines),
                ("Affected Users", affected_users),
                ("Impact", impact_lines),
                ("Root Cause / Contributing Factors", root_cause_lines),
            ),
            mapped["problem_statement"][1],
        ),
        "stakeholder_register": (
            contract_body(
                (
                    "Stakeholders",
                    markdown_table(
                        ("role", "responsibility", "decision authority", "needs/concerns", "involvement"),
                        [
                            (
                                re.sub(r"^\d+(?:\.\d+)*\.\s*", "", title),
                                clean_markdown_text(extract_heading_body(body, "Mục tiêu", level=5), limit=300),
                                {
                                    "System Administrator": "System and operational controls",
                                    "Research Project Manager": "Project scope and membership",
                                    "Researcher / Data Analyst": "Research decisions and execution",
                                    "Reviewer / Stakeholder": "Review and feedback",
                                }.get(re.sub(r"^\d+(?:\.\d+)*\.\s*", "", title), "Review"),
                                clean_markdown_cell(extract_heading_body(body, "Nhu cầu", level=5), limit=360),
                                {
                                    "System Administrator": "System administration",
                                    "Research Project Manager": "Project lifecycle",
                                    "Researcher / Data Analyst": "Research lifecycle",
                                    "Reviewer / Stakeholder": "Review / approval",
                                }.get(re.sub(r"^\d+(?:\.\d+)*\.\s*", "", title), "Project review"),
                            )
                            for title, body in stakeholder_entries
                        ],
                    ),
                )
            ),
            mapped["stakeholder_register"][1],
        ),
        "scope_capabilities": (
            contract_body(("Scope", scope_lines), ("Capabilities", capabilities), ("Out of Scope", out_of_scope)),
            mapped["scope_capabilities"][1],
        ),
        "business_rules": (
            contract_body(("Business Rules", business_rules)),
            mapped["business_rules"][1],
        ),
        "constraints_assumptions": (
            contract_body(
                ("Constraints", "\n".join(f"- {item}" for item in source_bullets(constraints, limit=360))),
                (
                    "Assumptions",
                    markdown_table(
                        ("constraint/assumption", "impact", "owner/source", "validation"), assumptions_rows
                    ),
                ),
                ("Validation Plan", validation_plan),
                ("Risks", markdown_table(("risk", "likelihood", "impact", "mitigation", "status"), risk_rows)),
                (
                    "Mitigation Plan",
                    markdown_table(("risk", "mitigation", "owner/status"), mitigation_rows),
                ),
            ),
            mapped["constraints_assumptions"][1],
        ),
    }


def normalize_prd_sections(
    markdown: str,
    mapped: dict[str, tuple[str, list[str]]],
) -> dict[str, tuple[str, list[str]]]:
    """Convert PRD samples into canonical capability and requirement tables/entries."""
    _preamble, source = split_h2_sections(markdown)

    capability_source = extract_heading_body(source["Business Capabilities"], "Normalized capability map")
    capability_entries = extract_h4_entries(capability_source)
    capability_blocks: list[str] = []
    for title, body in capability_entries:
        capability_id, capability_name = source_id_and_title(title, "BC-")
        fields: dict[str, str] = {}
        for field in ("goal", "user_segment", "business_value", "scope"):
            match = re.search(rf"(?m)^-\s+\*\*{re.escape(field)}:\*\*\s*(.+?)\s*$", body)
            fields[field] = clean_markdown_text(match.group(1), limit=420) if match else "See source capability map"
        capability_blocks.append(
            "\n".join(
                [
                    f"### {capability_id}: {capability_name}",
                    f"- **goal:** {fields['goal']}",
                    f"- **user_segment:** {fields['user_segment']}",
                    f"- **business_value:** {fields['business_value']}",
                    f"- **scope:** {fields['scope']}",
                ]
            )
        )
    capabilities = "\n\n".join(capability_blocks)

    functional_source = strip_leading_h2(source["Functional Requirements"])
    module_sections = extract_heading_bodies(functional_source, level=3)
    functional_rows: list[tuple[str, ...]] = []
    for module_title, module_body in module_sections.items():
        if not module_title.startswith("Feature Module "):
            continue
        module_match = re.match(r"Feature Module ([A-Z]+)", module_title)
        module_key = module_match.group(1) if module_match else "A"
        objective = clean_markdown_text(extract_heading_body(module_body, "Objective", level=4), limit=260)
        requirement_source = extract_heading_body(module_body, "Functional Requirements", level=4)
        requirement_entries = extract_h5_entries(requirement_source)
        acceptance = extract_heading_body(module_body, "Acceptance Criteria", level=4)
        priority_match = re.search(r"(?m)^\*\*Priority:\*\*\s*(.+?)\s*$", module_body)
        priority = clean_markdown_text(priority_match.group(1), limit=180) if priority_match else "Must"
        capability = module_capability(module_key)
        if not requirement_entries:
            requirement_entries = [(f"FR-{module_key}-01", objective or module_title)]
        for requirement_title, requirement_body in requirement_entries:
            source_id, requirement_name = source_id_and_title(requirement_title, "FR-")
            behavior = clean_markdown_text(requirement_body, limit=520) or objective
            referenced_requirements = [
                reference
                for reference in re.findall(r"\bFR-[A-Z0-9-]+\b", requirement_body)
                if reference != source_id
            ]
            dependencies = "<br>".join(dict.fromkeys(referenced_requirements)) or "None stated in source"
            functional_rows.append(
                (
                    "",  # FR id is assigned by the canonical registry render order.
                    f"{capability} · {source_id} — {requirement_name}",
                    f"{capability}: {behavior}",
                    input_output_signal(requirement_body, source_id),
                    acceptance_signal(acceptance, source_id),
                    priority,
                    dependencies,
                )
            )
    functional_table = markdown_table(
        (
            "id",
            "requirement",
            "behavior",
            "inputs/outputs",
            "acceptance signal",
            "priority",
            "dependencies",
        ),
        [
            (f"FR-{index:02d}", *row[1:])
            for index, row in enumerate(functional_rows, start=1)
        ],
    )

    nfr_source = extract_heading_body(source["Non-Functional Requirements"], "Non-Functional Requirements")
    nfr_entries = extract_h4_entries(nfr_source)
    nfr_rows: list[tuple[str, ...]] = []
    for nfr_title, nfr_body in nfr_entries:
        source_id, quality_attribute = source_id_and_title(nfr_title, "NFR-")
        requirement = clean_markdown_text(nfr_body, limit=620)
        measurements = [
            clean_markdown_text(line, limit=260)
            for line in nfr_body.splitlines()
            if re.search(r"(?:≤|≥|p95|100%|second|keyboard|timestamp|snapshot|immutable|retry)", line, re.I)
        ]
        measurement = "<br>".join(measurements[:3]) or "Verify through acceptance review and quality checks"
        scope_tradeoff = "Project-wide"
        if quality_attribute.lower() in {"performance", "scalability"}:
            scope_tradeoff = "Interactive MVP paths; long-running work may be asynchronous"
        elif quality_attribute.lower() in {"security", "privacy", "decision safety"}:
            scope_tradeoff = "All projects; fail safely when the requirement is not met"
        elif quality_attribute.lower() in {"maintainability", "provider portability"}:
            scope_tradeoff = "MVP architecture; preserve modular provider/tool boundaries"
        nfr_rows.append((source_id, f"{quality_attribute} ({source_id})", requirement, measurement, scope_tradeoff))
    nfr_table = markdown_table(
        ("id", "quality attribute", "requirement", "measurement", "scope/tradeoff"),
        [
            (f"NFR-{index:02d}", *row[1:])
            for index, row in enumerate(nfr_rows, start=1)
        ],
    )

    return {
        "use_case": (contract_body(("Business Capabilities", capabilities)), mapped["use_case"][1]),
        "functional_requirement": (
            contract_body(("Functional Requirements", functional_table)),
            mapped["functional_requirement"][1],
        ),
        "non_functional_requirement": (
            contract_body(("Non-Functional Requirements", nfr_table)),
            mapped["non_functional_requirement"][1],
        ),
    }


def map_document_sections(
    markdown: str,
    section_map: dict[str, tuple[str, ...]],
    *,
    preamble_item: str,
) -> dict[str, tuple[str, list[str]]]:
    preamble, sections = split_h2_sections(markdown)
    used_headings = {heading for headings in section_map.values() for heading in headings}
    missing = used_headings - sections.keys()
    unassigned = sections.keys() - used_headings
    if missing or unassigned:
        raise ValueError(
            "Document section mapping mismatch; "
            f"missing={sorted(missing)}, unassigned={sorted(unassigned)}"
        )

    mapped: dict[str, tuple[str, list[str]]] = {}
    for item_type, headings in section_map.items():
        parts: list[str] = []
        sources: list[str] = []
        if item_type == preamble_item and preamble:
            parts.append(preamble)
            sources.append("document preamble")
        for heading in headings:
            parts.append(sections[heading])
            sources.append(heading)
        mapped[item_type] = ("\n\n".join(parts).strip() + "\n", sources)
    return mapped


def extract_h3_section(markdown: str, title: str) -> str:
    match = re.search(rf"(?m)^###\s+{re.escape(title)}\s*$", markdown)
    if match is None:
        return ""
    next_heading = re.search(r"(?m)^###\s+", markdown[match.end() :])
    end = match.end() + next_heading.start() if next_heading else len(markdown)
    return markdown[match.end() : end].strip()


async def get_or_create_source_document(
    *,
    session,
    project: Project,
    user: User,
    path: Path,
    version: str,
) -> SourceDocument:
    content = path.read_bytes().decode("utf-8")
    locator = f"repo:{path.relative_to(ROOT).as_posix()}"
    seed_metadata = {
        "seed_key": SEED_KEY,
        "document_version": version,
        "repository_path": path.relative_to(ROOT).as_posix(),
    }
    source = (
        await session.execute(
            select(SourceDocument).where(SourceDocument.project_id == project.id, SourceDocument.locator == locator)
        )
    ).scalar_one_or_none()

    if source is None:
        source = SourceDocument(
            project_id=project.id,
            uploaded_by_id=user.id,
            title=path.stem.replace("-", " "),
            source_type=SourceType.MARKDOWN_UPLOAD,
            locator=locator,
            content_text=content,
            content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            mime_type="text/markdown",
            size_bytes=len(content.encode("utf-8")),
            extra_metadata=seed_metadata,
        )
        session.add(source)
        await session.flush()
        return source

    metadata = source.extra_metadata if isinstance(source.extra_metadata, dict) else {}
    if metadata.get("seed_key") != SEED_KEY:
        raise RuntimeError(f"Refusing to replace non-seed source document at {locator}")
    source.uploaded_by_id = user.id
    source.title = path.stem.replace("-", " ")
    source.source_type = SourceType.MARKDOWN_UPLOAD
    source.content_text = content
    source.content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    source.mime_type = "text/markdown"
    source.size_bytes = len(content.encode("utf-8"))
    source.extra_metadata = seed_metadata
    await session.flush()
    return source


async def get_or_create_container(
    *,
    session,
    project: Project,
    user: User,
    artifact_type: str,
) -> Artifact:
    container = (
        await session.execute(
            select(Artifact).where(
                Artifact.project_id == project.id,
                Artifact.type == ArtifactType(artifact_type),
                Artifact.parent_id.is_(None),
            )
        )
    ).scalar_one_or_none()
    if container is not None:
        metadata = container.extra_metadata if isinstance(container.extra_metadata, dict) else {}
        if metadata.get("seed_key") != SEED_KEY:
            raise RuntimeError(f"Refusing to modify an existing non-seed {artifact_type} document")
        return container

    config = get_config(artifact_type)
    container = Artifact(
        project_id=project.id,
        type=ArtifactType(artifact_type),
        status=ArtifactStatus.DRAFT,
        title=config.label,
        extra_metadata={"seed_key": SEED_KEY, "seed_role": "document_container"},
        created_by_id=user.id,
    )
    session.add(container)
    await session.flush()
    return container


async def get_or_create_item(
    *,
    session,
    project: Project,
    user: User,
    container: Artifact,
    item_type: str,
    body: str,
    source: SourceDocument,
    source_sections: list[str],
) -> Artifact:
    item = (
        await session.execute(
            select(Artifact).where(
                Artifact.project_id == project.id,
                Artifact.parent_id == container.id,
                Artifact.type == ArtifactType(item_type),
            )
        )
    ).scalar_one_or_none()
    config = get_config(item_type)
    item_metadata = {
        "seed_key": SEED_KEY,
        "source_document_id": str(source.id),
        "source_sections": source_sections,
        "output_contract": {
            "required_headings": list(output_contract(item_type).required_headings),
            "table_columns": list(output_contract(item_type).table_columns),
            "id_prefix": output_contract(item_type).id_prefix,
            "render_style": output_contract(item_type).render_style,
        },
        "normalization": "registry-canonical-v2",
    }

    if item is None:
        item = Artifact(
            project_id=project.id,
            parent_id=container.id,
            type=ArtifactType(item_type),
            status=ArtifactStatus.DRAFT,
            title=config.label,
            extra_metadata=item_metadata,
            created_by_id=user.id,
        )
        session.add(item)
        await session.flush()
    else:
        metadata = item.extra_metadata if isinstance(item.extra_metadata, dict) else {}
        if metadata.get("seed_key") != SEED_KEY:
            raise RuntimeError(f"Refusing to modify a non-seed {item_type} section")

    current_version = (
        await session.get(ArtifactVersion, item.current_version_id) if item.current_version_id is not None else None
    )
    if current_version is not None:
        version_metadata = current_version.extra_metadata if isinstance(current_version.extra_metadata, dict) else {}
        if version_metadata.get("seed_key") != SEED_KEY:
            raise RuntimeError(f"Refusing to replace a manually edited {item_type} version")
        if current_version.body == body and current_version.source_document_id == source.id:
            return item

    version = ArtifactVersion(
        artifact_id=item.id,
        version_number=(current_version.version_number + 1) if current_version is not None else 1,
        title=config.label,
        body=body,
        status=VersionStatus.DRAFT,
        change_source=ChangeSource.IMPORT,
        change_summary=f"Imported from {source.title}",
        parent_version_id=current_version.id if current_version is not None else None,
        created_by_id=user.id,
        source_document_id=source.id,
        extra_metadata={**item_metadata, "seed_key": SEED_KEY},
    )
    session.add(version)
    await session.flush()
    item.current_version_id = version.id
    item.status = ArtifactStatus.DRAFT
    item.title = config.label
    item.extra_metadata = item_metadata
    await session.flush()
    return item


async def approve_seed_item(*, session, project: Project, user: User, item: Artifact) -> None:
    """Complete a seeded section through the same approval path as the UI."""
    if item.current_version_id is None:
        raise RuntimeError(f"Seeded section {item.type.value} has no current version")

    existing = (
        await session.execute(
            select(ArtifactReview).where(
                ArtifactReview.artifact_id == item.id,
                ArtifactReview.artifact_version_id == item.current_version_id,
                ArtifactReview.review_status == ReviewStatus.APPROVED,
                ArtifactReview.comment == SEED_REVIEW_COMMENT,
            )
        )
    ).scalar_one_or_none()
    if existing is not None:
        return

    await ArtifactVersionService(session).review(
        project_id=project.id,
        artifact_id=item.id,
        version_id=item.current_version_id,
        body=ArtifactReviewRequest(
            review_status=ReviewStatus.APPROVED,
            comment=SEED_REVIEW_COMMENT,
        ),
        reviewed_by_id=user.id,
    )


async def ensure_link(
    *,
    session,
    project: Project,
    user: User,
    source: Artifact,
    target: Artifact,
) -> None:
    existing = (
        await session.execute(
            select(ArtifactLink).where(
                ArtifactLink.source_artifact_id == source.id,
                ArtifactLink.target_artifact_id == target.id,
                ArtifactLink.relation_type == RelationType.DERIVES_FROM,
            )
        )
    ).scalar_one_or_none()
    if existing is None:
        session.add(
            ArtifactLink(
                project_id=project.id,
                source_artifact_id=source.id,
                target_artifact_id=target.id,
                relation_type=RelationType.DERIVES_FROM,
                created_by_id=user.id,
                extra_metadata={"seed_key": SEED_KEY},
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed the BRD/PRD demo workspace for a local GitHub-authenticated user."
    )
    parser.add_argument(
        "--github-login",
        help=(
            "GitHub login to own the seeded organization. If omitted, the task uses the only "
            "real GitHub-authenticated local user."
        ),
    )
    parser.add_argument(
        "--owner-email",
        help="Verified email of the local user to own the seeded organization.",
    )
    return parser.parse_args()


async def resolve_seed_user(
    *,
    session,
    owner_email: str | None = None,
    github_login: str | None = None,
) -> User:
    """Resolve the seed owner without hard-coding a particular GitHub account.

    A CLI task has no browser cookie, so it cannot know which browser tab is currently logged in.
    Explicit ``--github-login``/``--owner-email`` selectors are therefore supported.  With no
    selector, the safe convenience path only works when exactly one local user has completed
    GitHub OAuth (identified by a stored encrypted GitHub access token); synthetic users from
    ``seed_dev_users.py`` are not considered.
    """

    owner_email = (owner_email or os.getenv(SEED_OWNER_EMAIL_ENV) or "").strip() or None
    github_login = (github_login or os.getenv(SEED_GITHUB_LOGIN_ENV) or "").strip() or None
    if owner_email and github_login:
        raise RuntimeError("Choose one seed owner selector: --github-login or --owner-email")

    if owner_email:
        result = await session.execute(
            select(User).where(
                func.lower(User.email) == owner_email.lower(),
                User.is_active.is_(True),
            )
        )
        user = result.scalar_one_or_none()
        if user is None:
            raise RuntimeError(f"No active local user found for email {owner_email}")
        return user

    if github_login:
        result = await session.execute(
            select(User).where(
                func.lower(User.github_login) == github_login.lower(),
                User.is_active.is_(True),
            )
        )
        user = result.scalar_one_or_none()
        if user is None:
            raise RuntimeError(f"No active local user found for GitHub login @{github_login}")
        return user

    result = await session.execute(
        select(User)
        .where(
            User.is_active.is_(True),
            User.github_id.is_not(None),
            User.github_login.is_not(None),
            User.github_access_token.is_not(None),
        )
        .order_by(User.updated_at.desc(), User.created_at.desc())
    )
    candidates = list(result.scalars().all())
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise RuntimeError(
            "No GitHub-authenticated local user found. Sign in with GitHub once, then rerun "
            "the seed, or pass --github-login <login>."
        )

    candidate_list = ", ".join(
        f"@{item.github_login} <{item.email}>" for item in candidates
    )
    raise RuntimeError(
        "More than one GitHub-authenticated local user was found: "
        f"{candidate_list}. Choose one with --github-login <login> or --owner-email <email>."
    )


def owner_slug_suffix(user: User) -> str:
    """Return a stable readable suffix for a user's per-owner organization slug."""

    basis = user.github_login or user.email.split("@", 1)[0] or str(user.id)[:8]
    return slugify(basis, fallback=f"user-{str(user.id)[:8]}")


async def unique_organization_slug(*, session, user: User) -> str:
    """Keep the legacy slug for the first owner and isolate later owners by GitHub login."""

    async def exists(slug: str) -> bool:
        return (
            await session.execute(select(Organization.id).where(Organization.slug == slug))
        ).scalar_one_or_none() is not None

    if not await exists(ORG_SLUG):
        return ORG_SLUG

    suffix = owner_slug_suffix(user)
    base = slugify(f"{ORG_SLUG}-{suffix}", fallback=f"{ORG_SLUG}-{str(user.id)[:8]}")
    candidate = base
    counter = 2
    while await exists(candidate):
        candidate = slugify(f"{base}-{counter}", fallback=f"{ORG_SLUG}-{str(user.id)[:8]}-{counter}")
        counter += 1
    return candidate


async def get_or_create_seed_organization(*, session, user: User) -> Organization:
    """Reuse this user's demo organization, or create an owner-isolated one."""

    organization = (
        await session.execute(
            select(Organization)
            .where(Organization.owner_id == user.id, Organization.name == ORG_NAME)
            .order_by(Organization.created_at)
            .limit(1)
        )
    ).scalar_one_or_none()
    if organization is None:
        organization = Organization(
            name=ORG_NAME,
            slug=await unique_organization_slug(session=session, user=user),
            owner_id=user.id,
        )
        session.add(organization)
        await session.flush()

    membership = (
        await session.execute(
            select(OrgMember).where(
                OrgMember.org_id == organization.id,
                OrgMember.user_id == user.id,
            )
        )
    ).scalar_one_or_none()
    if membership is None:
        session.add(OrgMember(org_id=organization.id, user_id=user.id, role="owner"))
    elif membership.role != "owner":
        membership.role = "owner"
    return organization


async def get_or_create_seed_project(*, session, organization: Organization, brd_markdown: str) -> Project:
    """Reuse the user's project by name while preserving the canonical project slug."""

    project = (
        await session.execute(
            select(Project)
            .where(Project.org_id == organization.id, Project.name == PROJECT_NAME)
            .order_by(Project.created_at)
            .limit(1)
        )
    ).scalar_one_or_none()
    if project is not None:
        return project

    conflicting_project = (
        await session.execute(
            select(Project).where(
                Project.org_id == organization.id,
                Project.slug == PROJECT_SLUG,
            )
        )
    ).scalar_one_or_none()
    if conflicting_project is not None:
        raise RuntimeError(
            f"Project slug {PROJECT_SLUG} already exists with a different name in organization "
            f"{organization.slug}"
        )

    executive_summary = extract_h3_section(brd_markdown, "Executive Summary")
    description = executive_summary.split("\n\n", 1)[0] if executive_summary else None
    project = Project(
        org_id=organization.id,
        name=PROJECT_NAME,
        slug=PROJECT_SLUG,
        description=description,
        executive_summary=executive_summary or None,
    )
    session.add(project)
    await session.flush()
    return project


async def seed(*, owner_email: str | None = None, github_login: str | None = None) -> None:
    if settings.app_env != "development":
        raise RuntimeError("This demo seed is restricted to APP_ENV=development")
    if not BRD_PATH.is_file() or not PRD_PATH.is_file():
        raise FileNotFoundError("Expected the BRD and PRD source files under req-tool-be/docs")

    brd_markdown = BRD_PATH.read_bytes().decode("utf-8")
    prd_markdown = PRD_PATH.read_bytes().decode("utf-8")
    brd_sections = map_document_sections(
        brd_markdown,
        {
            "vision_objectives": (
                "AI Research Experimentation Platform",
                "Executive Summary",
                "Vision and Objectives",
            ),
            "problem_statement": ("Problem Statement",),
            "stakeholder_register": ("Stakeholder Register",),
            "scope_capabilities": ("Scope and Capabilities",),
            "business_rules": ("Business Rules",),
            "constraints_assumptions": ("Constraints, Assumptions, and Risks", "Research Basis"),
        },
        preamble_item="vision_objectives",
    )
    prd_sections = map_document_sections(
        prd_markdown,
        {
            "use_case": ("AI Research Experimentation Platform", "Business Capabilities"),
            "functional_requirement": ("Functional Requirements",),
            "non_functional_requirement": ("Non-Functional Requirements",),
        },
        preamble_item="use_case",
    )
    brd_sections = normalize_brd_sections(brd_markdown, brd_sections)
    prd_sections = normalize_prd_sections(prd_markdown, prd_sections)
    validate_contract_sections({**brd_sections, **prd_sections})

    async with async_session_factory() as session:
        async with session.begin():
            user = await resolve_seed_user(
                session=session,
                owner_email=owner_email,
                github_login=github_login,
            )
            organization = await get_or_create_seed_organization(session=session, user=user)
            project = await get_or_create_seed_project(
                session=session,
                organization=organization,
                brd_markdown=brd_markdown,
            )

            brd_source = await get_or_create_source_document(
                session=session,
                project=project,
                user=user,
                path=BRD_PATH,
                version="1.8",
            )
            prd_source = await get_or_create_source_document(
                session=session,
                project=project,
                user=user,
                path=PRD_PATH,
                version="1.7",
            )

            brd_container = await get_or_create_container(
                session=session,
                project=project,
                user=user,
                artifact_type="brd",
            )
            prd_container = await get_or_create_container(
                session=session,
                project=project,
                user=user,
                artifact_type="prd",
            )

            brd_items: dict[str, Artifact] = {}
            for item_type, (body, source_sections) in brd_sections.items():
                brd_items[item_type] = await get_or_create_item(
                    session=session,
                    project=project,
                    user=user,
                    container=brd_container,
                    item_type=item_type,
                    body=body,
                    source=brd_source,
                    source_sections=source_sections,
                )

            prd_items: dict[str, Artifact] = {}
            for item_type, (body, source_sections) in prd_sections.items():
                prd_items[item_type] = await get_or_create_item(
                    session=session,
                    project=project,
                    user=user,
                    container=prd_container,
                    item_type=item_type,
                    body=body,
                    source=prd_source,
                    source_sections=source_sections,
                )

            for item in [*brd_items.values(), *prd_items.values()]:
                await approve_seed_item(session=session, project=project, user=user, item=item)

            # A document container is complete when every registry child has an
            # imported version that passed the approval path above.
            brd_container.status = ArtifactStatus.ACCEPTED
            prd_container.status = ArtifactStatus.ACCEPTED

            await ensure_link(
                session=session,
                project=project,
                user=user,
                source=prd_container,
                target=brd_container,
            )
            await ensure_link(
                session=session,
                project=project,
                user=user,
                source=prd_items["use_case"],
                target=brd_items["scope_capabilities"],
            )
            await ensure_link(
                session=session,
                project=project,
                user=user,
                source=prd_items["functional_requirement"],
                target=brd_container,
            )
            await ensure_link(
                session=session,
                project=project,
                user=user,
                source=prd_items["non_functional_requirement"],
                target=brd_items["constraints_assumptions"],
            )

        print(f"[seed] owner: {user.full_name} <{user.email}> (@{user.github_login or 'no-github-login'})")
        print(f"[seed] organization: {organization.name} ({organization.slug})")
        print(f"[seed] project: {project.name} ({project.slug})")
        print(f"[seed] source documents: {brd_source.title}, {prd_source.title}")
        print(f"[seed] BRD sections: {len(brd_sections)}; PRD sections: {len(prd_sections)}; trace links: 4")
        print("[seed] full Markdown content stored as source documents and imported document-section versions")


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(seed(owner_email=args.owner_email, github_login=args.github_login))
