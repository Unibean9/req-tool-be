"""Section taxonomy for the requirements harness.

Pure Python with no LLM or DB dependency. Replaces the 9-key BRD slot model
with a 7-section assessment taxonomy (spec §3). Sections are scored by status
(``missing`` | ``partial`` | ``filled`` | ``needs_review``) instead of a flat
checklist of sub-slots — reflecting the "progress over interrogation" philosophy.

Coexists with ``slot_schema.py`` during the Phase 1–2 migration window; the slot
model is removed at Checkpoint A.
"""

from typing import Any

# Allowed per-section assessment states. ``needs_review`` marks content the agent
# captured but flagged as unverified — distinct from ``partial`` (incomplete).
SECTION_STATUSES: tuple[str, ...] = ("missing", "partial", "filled", "needs_review")

# section_key -> {"sub_dimensions": {sub_key: description}, "threshold": float}.
# sub_dimensions are assessment angles the LLM self-evaluates, not slots to fill.
# vision_objectives carries 7 sub-dimensions to preserve the depth of the former
# goal + intent slots; business_rules is new and gets detailed sub-dimensions to
# compensate for the lack of training signal.
SECTION_SPECS: dict[str, dict[str, Any]] = {
    "vision_objectives": {
        "sub_dimensions": {
            "business_goal": "Mục tiêu kinh doanh cụ thể cần đạt.",
            "user_goal": "Kết quả người dùng muốn đạt được.",
            "metric": "Chỉ số đo lường thành công.",
            "target": "Ngưỡng hoặc kết quả mục tiêu cần đạt.",
            "timeframe": "Mốc thời gian hoặc hạn hoàn thành.",
            "intent": "Lý do và động cơ sáng kiến được thực hiện.",
            "success_definition": "Trạng thái lý tưởng khi sáng kiến thành công.",
        },
        "threshold": 0.8,
    },
    "problem_statement": {
        "sub_dimensions": {
            "who": "Đối tượng bị ảnh hưởng trực tiếp bởi vấn đề.",
            "obstacle": "Trở ngại cụ thể mà đối tượng đang gặp.",
            "root_cause": "Nguyên nhân gốc rễ sau khi đào sâu vấn đề.",
            "frequency": "Tần suất hoặc quy mô lặp lại của vấn đề.",
            "impact": "Ảnh hưởng lên người dùng, quy trình, hoặc kinh doanh.",
        },
        "threshold": 0.8,
    },
    "stakeholder_register": {
        "sub_dimensions": {
            "primary_user": "Người dùng cuối trực tiếp hoặc persona chính.",
            "secondary_stakeholders": "Các bên liên quan gián tiếp và cách họ bị ảnh hưởng.",
            "decision_maker": "Người có quyền phê duyệt hoặc ra quyết định.",
            "operator": "Người triển khai, vận hành, hoặc hỗ trợ sau ra mắt.",
        },
        "threshold": 0.75,
    },
    "scope_capabilities": {
        "sub_dimensions": {
            "in_scope": "Năng lực và hạng mục thuộc phạm vi lần này.",
            "out_of_scope": "Phần được loại trừ rõ ràng khỏi phạm vi.",
            "capability": "Năng lực sản phẩm hoặc hệ thống cần có.",
            "priority": "Mức ưu tiên như Must, Should, Could.",
        },
        "threshold": 0.75,
    },
    "business_rules": {
        "sub_dimensions": {
            "condition": "Điều kiện kích hoạt quy tắc nghiệp vụ.",
            "outcome": "Kết quả hoặc hành động khi điều kiện thỏa.",
            "trigger": "Sự kiện hoặc thời điểm quy tắc được áp dụng.",
            "scope": "Phạm vi áp dụng của quy tắc (đối tượng, ngữ cảnh).",
        },
        "threshold": 0.75,
    },
    "constraints_assumptions": {
        "sub_dimensions": {
            "constraint": "Ràng buộc cứng (thời gian, ngân sách, kỹ thuật, pháp lý).",
            "assumption": "Giả định đang được dựa vào để ra quyết định.",
            "validation": "Cách kiểm chứng giả định trước khi build.",
            "dependency": "Phụ thuộc bên ngoài, vendor, hoặc team khác.",
        },
        "threshold": 0.75,
    },
    "risks_issues": {
        "sub_dimensions": {
            "risk": "Sự kiện hoặc điều kiện bất lợi có thể xảy ra.",
            "likelihood": "Xác suất xảy ra của rủi ro.",
            "mitigation": "Chiến lược giảm thiểu hoặc xử lý rủi ro.",
            "status": "Trạng thái theo dõi của rủi ro hoặc vấn đề mở.",
        },
        "threshold": 0.75,
    },
}

# section_key -> human-readable description. Single source of truth for prompt
# directives and judge prompts (parallels SLOT_DESCRIPTIONS).
SECTION_DESCRIPTIONS: dict[str, str] = {
    "vision_objectives": "Tầm nhìn và mục tiêu: vì sao làm, thành công trông như thế nào, đo bằng gì.",
    "problem_statement": "Phát biểu vấn đề: ai gặp trở ngại gì, nguyên nhân gốc, tần suất và tác động.",
    "stakeholder_register": "Danh sách bên liên quan: người dùng chính, bên liên quan phụ, người quyết định, vận hành.",
    "scope_capabilities": "Phạm vi và năng lực: cái gì trong scope, cái gì ngoài scope, năng lực cần có và ưu tiên.",
    "business_rules": "Quy tắc nghiệp vụ: điều kiện, kết quả, trigger và phạm vi áp dụng.",
    "constraints_assumptions": "Ràng buộc và giả định: giới hạn cứng, giả định đang dựa vào và cách kiểm chứng.",
    "risks_issues": "Rủi ro và vấn đề: sự kiện bất lợi, xác suất, giảm thiểu và trạng thái theo dõi.",
}

# Reference mapping from the legacy BRD slot model to the new taxonomy. Not used
# at runtime — documents the Phase 2 migration reasoning, including which goal/intent
# sub-slots are absorbed into vision_objectives sub-dimensions.
LEGACY_BRD_TO_SECTION: dict[str, str] = {
    "intent": "vision_objectives",
    "goal": "vision_objectives",
    "problem": "problem_statement",
    "stakeholder": "stakeholder_register",
    "capability": "scope_capabilities",
    "constraint": "constraints_assumptions",
    "assumption": "constraints_assumptions",
    "risk": "risks_issues",
    "open_question": "risks_issues",
}

# Artifact types whose elicitation is assessed against the 7-section taxonomy. Mirrors the legacy
# BRD-slot gating: discovery artifacts report section_assessment; derived types (story, epic,
# functional_requirement) fail open so the coverage gate never blocks them.
SECTION_TRACKED_ARTIFACT_TYPES: frozenset[str] = frozenset(LEGACY_BRD_TO_SECTION)

# Consecutive non-improving coverage turns before the elicitation gate relaxes.
COVERAGE_STALL_LIMIT = 2


def compute_section_coverage(section_assessment: dict[str, Any]) -> dict[str, Any]:
    """Score 7-section coverage from a flat or granular assessment.

    Each section value may be a flat status string ("missing"|"partial"|"filled"|
    "needs_review") or a dict of sub-dimension statuses; the dict form is reduced
    to a section status by sub-dimension fill ratio against the section threshold.
    """
    normalized: dict[str, str] = {
        section: _resolve_section_status(section, (section_assessment or {}).get(section))
        for section in SECTION_SPECS
    }
    score = sum(_section_score(normalized[section]) for section in SECTION_SPECS)
    ratio = score / len(SECTION_SPECS)
    coverage_complete = all(normalized[section] == "filled" for section in SECTION_SPECS)

    return {
        "section_coverage": normalized,
        "coverage_ratio": ratio,
        "coverage_complete": coverage_complete,
    }


def _resolve_section_status(section: str, value: Any) -> str:
    if isinstance(value, dict):
        return _status_from_sub_dimensions(section, value)
    if value in SECTION_STATUSES:
        return value
    return "missing"


def _status_from_sub_dimensions(section: str, assessment: dict[str, Any]) -> str:
    sub_dimensions = SECTION_SPECS[section]["sub_dimensions"]
    sub_score = sum(_status_score(assessment.get(sub)) for sub in sub_dimensions)
    ratio = sub_score / len(sub_dimensions)
    if ratio >= SECTION_SPECS[section]["threshold"]:
        return "filled"
    if ratio > 0:
        return "partial"
    return "missing"


def _section_score(status: str) -> float:
    return _status_score(status)


def _status_score(status: Any) -> float:
    if status == "filled":
        return 1.0
    if status in {"partial", "needs_review"}:
        return 0.5
    return 0.0
