"""Deterministic validator for proposal artifacts — pure Python.

NO LLM calls, NO DB access. Detects:
- Missing required fields (`title`, `body`) -> violation (hard block).
- Weasel words (Vietnamese + English) -> warning (non-blocking).
- Open question and assumption quality signals.

Limitation: heuristics only match whole words (word boundary) to reduce
false positives; deeper analysis is added later by the LLM critic.
"""

import re
import unicodedata
from dataclasses import dataclass, field

REQUIRED_FIELDS = ("title", "body")

WEASEL_WORDS = (
    # Vietnamese
    "nhanh",
    "easy to use",
    "optimized",
    "friendly",
    "flexible",
    "effective",
    "robust",
    # English
    "fast",
    "easy to use",
    "easy",
    "optimal",
    "friendly",
    "flexible",
    "efficient",
    "powerful",
    "seamless",
)

# Condition / outcome signals for a business rule (spec §9.4). Vietnamese has no rigid if/then
# syntax, so the keyword lists are broad; false negatives are acceptable, false positives are not.
_RULE_CONDITION = ("if", "when", "in case", "condition", "trigger", "neu", "khi", "truong hop", "dieu kien")
_RULE_OUTCOME = ("then", "will", "result", "must", "thi", "se", "ket qua", "phai")

# BMAD workflow enums (addendum §17). Kept local to avoid importing the graph layer into validators.
_WORKFLOW_MODES = {"brainstorm", "brief", "prd", "readiness_check", "architecture_readiness"}
_PLANNING_TRACKS = {"quick", "standard", "enterprise"}


@dataclass
class ValidationResult:
    passed: bool
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _has_word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE | re.UNICODE) is not None


def _ascii_fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower()


def validate_proposal(artifact_type: str, proposal: dict) -> ValidationResult:
    violations: list[str] = []
    warnings: list[str] = []

    # 1. Required fields — missing or empty -> violation
    for fld in REQUIRED_FIELDS:
        value = proposal.get(fld)
        if value is None or (isinstance(value, str) and not value.strip()):
            violations.append(f"Missing required field: {fld}")

    title = proposal.get("title") or ""
    body = proposal.get("body") or ""
    text = f"{title} {body}"

    # 2. Weasel words — at most one warning per word
    for word in WEASEL_WORDS:
        if _has_word(text, word):
            warnings.append(f"Weasel word (weasel word): '{word}'")

    lowered = text.lower()
    folded = _ascii_fold(text)

    # 3. Business rule must carry a condition AND an outcome — a rule missing either is
    # structurally meaningless, so this is a hard violation (spec §9.4).
    if artifact_type == "business_rule":
        if not any(token in folded for token in _RULE_CONDITION):
            violations.append("Business rule is missing condition (trigger condition)")
        if not any(token in folded for token in _RULE_OUTCOME):
            violations.append("Business rule is missing outcome (result when condition is satisfied)")

    # 4. Open question must declare a tracking status — quality signal, not a hard block.
    if artifact_type == "open_question" and not _has_status(proposal, lowered):
        warnings.append("Open question missing status (unresolved/answered/deferred)")

    # 5. Each captured assumption should name a confidence and an owner — quality signals.
    for assumption in proposal.get("assumptions") or []:
        if not (assumption.get("confidence") or "").strip():
            warnings.append("Assumption missing confidence")
        if not (assumption.get("owner") or "").strip():
            warnings.append("Assumption missing owner")

    # 6. BMAD workflow validators (addendum §17) — block invalid transitions.
    if artifact_type == "workflow_state":
        if proposal.get("workflow_mode") not in _WORKFLOW_MODES:
            violations.append("Invalid workflow_mode")
        if proposal.get("planning_track") not in _PLANNING_TRACKS:
            violations.append("Invalid planning_track")
    if artifact_type == "workflow_recommendation":
        recommended = proposal.get("recommended_next_workflow")
        if recommended == "epic_story_readiness":
            violations.append("epic_story_readiness is outside BMAD MVP scope")
        if recommended == "architecture_readiness" and (proposal.get("unresolved_critical_risks") or []):
            violations.append("unresolved_critical_risks blocks transition to implementation phase")

    return ValidationResult(passed=len(violations) == 0, violations=violations, warnings=warnings)


def _has_status(proposal: dict, lowered_text: str) -> bool:
    """Whether the open question carries a tracking status, via a status field or a status word."""
    if (proposal.get("status") or "").strip():
        return True
    return any(token in lowered_text for token in ("status", "unresolved", "answered", "deferred"))
