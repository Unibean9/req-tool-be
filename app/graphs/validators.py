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

    return ValidationResult(passed=len(violations) == 0, violations=violations, warnings=warnings)
