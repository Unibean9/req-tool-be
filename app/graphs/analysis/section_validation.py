"""Incremental, section-scoped structural validation (Phase 4).

Runs when a section is WRITTEN (a decision-node create/update), not only when the full draft is
proposed — so a defect surfaces within one turn of its cause. Pure Python, NO LLM, NO DB. Composes
two existing rule sources at section scope (read-only reuse, no logic duplicated):

- the Gate-4 structural placeholder rule (an unfilled required cell) from artifact_synthesis, and
- the section-applicable subset of `validate_proposal` (weasel words, business-rule completeness).

A finding is a JSON-safe dict so it can live in WorkflowState and survive a checkpoint round trip.
"""

from typing import Any

from app.documents.registry import INCOMPLETE_CELL_PLACEHOLDER
from app.graphs.validators import validate_proposal

VIOLATION = "violation"
WARNING = "warning"


def _finding(section: str, severity: str, message: str) -> dict[str, Any]:
    return {"section": section, "severity": severity, "message": message}


def _is_business_rule_section(section_key: str) -> bool:
    """A business-rules section carries the condition+outcome completeness rule (validators §3)."""
    return "business rule" in section_key.lower()


def validate_section(artifact_type: str, section_key: str, content: str) -> list[dict[str, Any]]:
    """Structural findings for ONE section's content. Empty list == the section passes.

    Never blocks a write — the caller records the result and lets it drive the next turn's prompt.
    """
    text = str(content or "")
    findings: list[dict[str, Any]] = []

    # Gate-4 structural rule, scoped to one section: an unfilled required cell.
    if INCOMPLETE_CELL_PLACEHOLDER in text:
        findings.append(
            _finding(section_key, VIOLATION, "has unfilled required cells — fill every column or mark for confirmation")
        )

    # validate_proposal reuse. Map to "business_rule" for a rules section so the condition/outcome
    # completeness check fires; any other section keeps the weasel-word check only. title/body are
    # both present, so the required-field violation never triggers here.
    proposal_type = "business_rule" if _is_business_rule_section(section_key) else artifact_type
    result = validate_proposal(proposal_type, {"title": section_key, "body": text})
    findings.extend(_finding(section_key, VIOLATION, v) for v in result.violations)
    findings.extend(_finding(section_key, WARNING, w) for w in result.warnings)
    return findings


def _has_violation(findings: list[dict[str, Any]] | None) -> bool:
    return any(f.get("severity") == VIOLATION for f in (findings or []))


def validated_coverage(
    section_coverage: dict[str, str] | None, section_findings: dict[str, list[dict[str, Any]]] | None
) -> dict[str, str]:
    """Coverage with the numerator made honest: a section with a `violation` finding is downgraded to
    "missing" so "covered" means "covered with structurally acceptable content".

    Pure function of its inputs. A section without a matching finding key is left untouched, so a
    checkpoint that predates section_findings (or whose node sections don't align with the coverage
    keys) scores exactly as before.
    """
    coverage = dict(section_coverage or {})
    findings = section_findings or {}
    for key, value in list(coverage.items()):
        if value != "missing" and _has_violation(findings.get(key)):
            coverage[key] = "missing"
    return coverage
