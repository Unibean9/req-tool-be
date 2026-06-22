"""Deterministic validator for proposal artifacts — pure Python.

NO LLM calls, NO DB access. Detects:
- Missing required fields (`title`, `body`) -> violation (hard block).
- Weasel words (Vietnamese + English) -> warning (non-blocking).
- INVEST for story/epic, SMART for goal -> simple heuristic warnings.

Limitation: heuristics only match whole words (word boundary) to reduce
false positives; deeper analysis is added later by the LLM critic.
"""

import re
from dataclasses import dataclass, field

REQUIRED_FIELDS = ("title", "body")

WEASEL_WORDS = (
    # Vietnamese
    "nhanh",
    "dễ dùng",
    "tối ưu",
    "thân thiện",
    "linh hoạt",
    "hiệu quả",
    "mạnh mẽ",
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

# "Testable" signals for INVEST (story/epic)
_INVEST_TESTABLE = ("given", "when", "then", "khi", "thì", "acceptance", "tiêu chí")

# "Time-bound" signals for SMART (goal); the measurable part uses digits/`%` separately
_SMART_TIMEBOUND = ("trong vòng", "trước ngày", "deadline", "by ")

# Condition / outcome signals for a business rule (spec §9.4). Vietnamese has no rigid if/then
# syntax, so the keyword lists are broad; false negatives are acceptable, false positives are not.
_RULE_CONDITION = ("nếu", "khi", "trong trường hợp", "điều kiện", "trigger", "if", "when")
_RULE_OUTCOME = ("thì", "sẽ", "kết quả", "phải", "then", "will", "must")


@dataclass
class ValidationResult:
    passed: bool
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _has_word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE | re.UNICODE) is not None


def validate_proposal(artifact_type: str, proposal: dict) -> ValidationResult:
    violations: list[str] = []
    warnings: list[str] = []

    # 1. Required fields — missing or empty -> violation
    for fld in REQUIRED_FIELDS:
        value = proposal.get(fld)
        if value is None or (isinstance(value, str) and not value.strip()):
            violations.append(f"Thiếu trường bắt buộc: {fld}")

    title = proposal.get("title") or ""
    body = proposal.get("body") or ""
    text = f"{title} {body}"

    # 2. Weasel words — at most one warning per word
    for word in WEASEL_WORDS:
        if _has_word(text, word):
            warnings.append(f"Từ mơ hồ (weasel word): '{word}'")

    lowered = text.lower()

    # 3. INVEST — applies to story/epic only
    if artifact_type in ("story", "epic"):
        if not any(token in lowered for token in _INVEST_TESTABLE):
            warnings.append("Story thiếu tiêu chí kiểm thử (INVEST: acceptance criteria / given-when-then)")

    # 4. SMART — applies to goal only
    if artifact_type == "goal":
        has_metric = bool(re.search(r"\d", text)) or "%" in text
        has_timebound = any(token in lowered for token in _SMART_TIMEBOUND)
        if not (has_metric and has_timebound):
            warnings.append("Goal thiếu yếu tố đo lường/thời hạn (SMART)")

    # 5. Business rule must carry a condition AND an outcome — a rule missing either is
    # structurally meaningless, so this is a hard violation (spec §9.4).
    if artifact_type == "business_rule":
        if not any(token in lowered for token in _RULE_CONDITION):
            violations.append("Business rule thiếu condition (điều kiện kích hoạt)")
        if not any(token in lowered for token in _RULE_OUTCOME):
            violations.append("Business rule thiếu outcome (kết quả khi điều kiện thỏa)")

    # 6. Open question must declare a tracking status — quality signal, not a hard block.
    if artifact_type == "open_question" and not _has_status(proposal, lowered):
        warnings.append("Open question thiếu status (unresolved/answered/deferred)")

    # 7. Each captured assumption should name a confidence and an owner — quality signals.
    for assumption in proposal.get("assumptions") or []:
        if not (assumption.get("confidence") or "").strip():
            warnings.append("Assumption thiếu confidence")
        if not (assumption.get("owner") or "").strip():
            warnings.append("Assumption thiếu owner")

    return ValidationResult(passed=len(violations) == 0, violations=violations, warnings=warnings)


def _has_status(proposal: dict, lowered_text: str) -> bool:
    """Whether the open question carries a tracking status, via a status field or a status word."""
    if (proposal.get("status") or "").strip():
        return True
    return any(token in lowered_text for token in ("status", "unresolved", "answered", "deferred"))
