from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DocumentTypeConfig:
    artifact_type: str
    label: str
    children: tuple[str, ...]
    is_container: bool
    description: str
    sub_dimensions: tuple[tuple[str, str], ...] = ()
    threshold: float = 1.0


def _item(
    artifact_type: str,
    label: str,
    description: str,
    *,
    sub_dimensions: dict[str, str] | None = None,
    threshold: float = 1.0,
) -> DocumentTypeConfig:
    return DocumentTypeConfig(
        artifact_type=artifact_type,
        label=label,
        children=(),
        is_container=False,
        description=description,
        sub_dimensions=tuple((sub_dimensions or {}).items()),
        threshold=threshold,
    )


_CONFIGS: tuple[DocumentTypeConfig, ...] = (
    DocumentTypeConfig(
        artifact_type="brd",
        label="Business Requirements Document",
        children=(
            "vision_objectives",
            "problem_statement",
            "stakeholder_register",
            "scope_capabilities",
            "business_rules",
            "constraints_assumptions",
            "risks_issues",
        ),
        is_container=True,
        description="Business context, scope, rules, constraints, and risks.",
    ),
    DocumentTypeConfig(
        artifact_type="prd",
        label="Product Requirements Document",
        children=(
            "functional_requirement",
            "use_case",
            "non_functional_requirement",
            "acceptance_criteria",
        ),
        is_container=True,
        description="Product behavior, quality attributes, use cases, and acceptance criteria.",
    ),
    DocumentTypeConfig(
        artifact_type="sad",
        label="Software Architecture Document",
        children=("domain_entity", "component", "interface", "tech_decision"),
        is_container=True,
        description="Architecture domains, components, interfaces, and technical decisions.",
    ),
    _item(
        "vision_objectives",
        "Vision and Objectives",
        "Tầm nhìn và mục tiêu: vì sao làm, thành công trông như thế nào, đo bằng gì.",
        sub_dimensions={
            "business_goal": "Mục tiêu kinh doanh cụ thể cần đạt.",
            "user_goal": "Kết quả người dùng muốn đạt được.",
            "metric": "Chỉ số đo lường thành công.",
            "target": "Ngưỡng hoặc kết quả mục tiêu cần đạt.",
            "timeframe": "Mốc thời gian hoặc hạn hoàn thành.",
            "intent": "Lý do và động cơ sáng kiến được thực hiện.",
            "success_definition": "Trạng thái lý tưởng khi sáng kiến thành công.",
        },
        threshold=0.8,
    ),
    _item(
        "problem_statement",
        "Problem Statement",
        "Phát biểu vấn đề: ai gặp trở ngại gì, nguyên nhân gốc, tần suất và tác động.",
        sub_dimensions={
            "who": "Đối tượng bị ảnh hưởng trực tiếp bởi vấn đề.",
            "obstacle": "Trở ngại cụ thể mà đối tượng đang gặp.",
            "root_cause": "Nguyên nhân gốc rễ sau khi đào sâu vấn đề.",
            "frequency": "Tần suất hoặc quy mô lặp lại của vấn đề.",
            "impact": "Ảnh hưởng lên người dùng, quy trình, hoặc kinh doanh.",
        },
        threshold=0.8,
    ),
    _item(
        "stakeholder_register",
        "Stakeholder Register",
        "Danh sách bên liên quan: người dùng chính, bên liên quan phụ, người quyết định, vận hành.",
        sub_dimensions={
            "primary_user": "Người dùng cuối trực tiếp hoặc persona chính.",
            "secondary_stakeholders": "Các bên liên quan gián tiếp và cách họ bị ảnh hưởng.",
            "decision_maker": "Người có quyền phê duyệt hoặc ra quyết định.",
            "operator": "Người triển khai, vận hành, hoặc hỗ trợ sau ra mắt.",
        },
        threshold=0.75,
    ),
    _item(
        "scope_capabilities",
        "Scope and Capabilities",
        "Phạm vi và năng lực: cái gì trong scope, cái gì ngoài scope, năng lực cần có và ưu tiên.",
        sub_dimensions={
            "in_scope": "Năng lực và hạng mục thuộc phạm vi lần này.",
            "out_of_scope": "Phần được loại trừ rõ ràng khỏi phạm vi.",
            "capability": "Năng lực sản phẩm hoặc hệ thống cần có.",
            "priority": "Mức ưu tiên như Must, Should, Could.",
        },
        threshold=0.75,
    ),
    _item(
        "business_rules",
        "Business Rules",
        "Quy tắc nghiệp vụ: điều kiện, kết quả, trigger và phạm vi áp dụng.",
        sub_dimensions={
            "condition": "Điều kiện kích hoạt quy tắc nghiệp vụ.",
            "outcome": "Kết quả hoặc hành động khi điều kiện thỏa.",
            "trigger": "Sự kiện hoặc thời điểm quy tắc được áp dụng.",
            "scope": "Phạm vi áp dụng của quy tắc.",
        },
        threshold=0.75,
    ),
    _item(
        "constraints_assumptions",
        "Constraints and Assumptions",
        "Ràng buộc và giả định: giới hạn cứng, giả định đang dựa vào và cách kiểm chứng.",
        sub_dimensions={
            "constraint": "Ràng buộc cứng về thời gian, ngân sách, kỹ thuật, hoặc pháp lý.",
            "assumption": "Giả định đang được dựa vào để ra quyết định.",
            "validation": "Cách kiểm chứng giả định trước khi build.",
            "dependency": "Phụ thuộc bên ngoài, vendor, hoặc team khác.",
        },
        threshold=0.75,
    ),
    _item(
        "risks_issues",
        "Risks and Issues",
        "Rủi ro và vấn đề: sự kiện bất lợi, xác suất, giảm thiểu và trạng thái theo dõi.",
        sub_dimensions={
            "risk": "Sự kiện hoặc điều kiện bất lợi có thể xảy ra.",
            "likelihood": "Xác suất xảy ra của rủi ro.",
            "mitigation": "Chiến lược giảm thiểu hoặc xử lý rủi ro.",
            "status": "Trạng thái theo dõi của rủi ro hoặc vấn đề mở.",
        },
        threshold=0.75,
    ),
    _item("functional_requirement", "Functional Requirement", "A testable product behavior."),
    _item("use_case", "Use Case", "An actor-goal interaction flow."),
    _item("non_functional_requirement", "Non-Functional Requirement", "A measurable quality constraint."),
    _item("acceptance_criteria", "Acceptance Criteria", "Conditions that prove a requirement is met."),
    _item("domain_entity", "Domain Entity", "A core domain concept and its responsibilities."),
    _item("component", "Component", "A deployable or logical architecture component."),
    _item("interface", "Interface", "A contract between architecture components."),
    _item("tech_decision", "Technical Decision", "A technical choice, rationale, and consequences."),
)

_BY_TYPE = {config.artifact_type: config for config in _CONFIGS}
_CONTAINER_BY_ITEM = {
    child: config.artifact_type
    for config in _CONFIGS
    if config.is_container
    for child in config.children
}


def get_config(artifact_type: str) -> DocumentTypeConfig:
    try:
        return _BY_TYPE[artifact_type]
    except KeyError as exc:
        raise ValueError(f"Document artifact type không hỗ trợ: {artifact_type}") from exc


def container_for(item_type: str) -> str | None:
    return _CONTAINER_BY_ITEM.get(item_type)


def children_of(container_type: str) -> tuple[str, ...]:
    config = get_config(container_type)
    if not config.is_container:
        raise ValueError(f"{container_type} không phải document container")
    return config.children


def all_container_types() -> tuple[str, ...]:
    return tuple(config.artifact_type for config in _CONFIGS if config.is_container)


def all_item_types() -> tuple[str, ...]:
    return tuple(config.artifact_type for config in _CONFIGS if not config.is_container)


def item_configs(container_type: str) -> tuple[DocumentTypeConfig, ...]:
    return tuple(get_config(item_type) for item_type in children_of(container_type))


def item_description(item_type: str) -> str:
    return get_config(item_type).description


def item_label(item_type: str) -> str:
    return get_config(item_type).label


def status_score(status: Any) -> float:
    if status == "filled":
        return 1.0
    if status in {"partial", "needs_review"}:
        return 0.5
    return 0.0
