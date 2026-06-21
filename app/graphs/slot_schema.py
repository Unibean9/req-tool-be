"""Slot schema for the BRD question loop.

Pure Python with no LLM or DB dependency. Slots are internal information
dimensions used to measure coverage before artifact creation; users only see
natural follow-up questions.
"""

# slot_key -> short description so prompt and coverage share the same meaning.
SLOT_DESCRIPTIONS: dict[str, str] = {
    "why_now": "Lý do sáng kiến cần được thực hiện ở thời điểm hiện tại.",
    "sponsor": "Người hoặc nhóm đặt hàng, tài trợ, hoặc chịu trách nhiệm kết quả.",
    "expected_outcome": "Kết quả mong đợi trong khung thời gian gần hoặc trung hạn.",
    "success_state": "Trạng thái lý tưởng khi sáng kiến thành công.",
    "cost_of_inaction": "Tác động nếu không làm gì hoặc trì hoãn.",
    "who": "Đối tượng bị ảnh hưởng trực tiếp bởi vấn đề hoặc nhu cầu.",
    "obstacle": "Trở ngại cụ thể mà đối tượng đang gặp.",
    "journey_step": "Bước trong hành trình hoặc quy trình nơi vấn đề xảy ra.",
    "root_cause": "Nguyên nhân gốc rễ sau khi đào sâu vấn đề.",
    "frequency": "Tần suất hoặc quy mô lặp lại của vấn đề.",
    "impact": "Ảnh hưởng định lượng hoặc định tính lên người dùng, quy trình, hoặc kinh doanh.",
    "workaround": "Cách xử lý tạm hiện tại và điểm chưa đủ của cách đó.",
    "business_goal": "Mục tiêu kinh doanh cụ thể cần đạt.",
    "user_goal": "Kết quả người dùng muốn đạt được.",
    "metric": "Chỉ số đo lường thành công.",
    "baseline": "Giá trị hiện tại của chỉ số hoặc trạng thái xuất phát.",
    "target": "Ngưỡng hoặc kết quả mục tiêu cần đạt.",
    "timeframe": "Mốc thời gian hoặc hạn hoàn thành.",
    "alignment": "Liên kết với OKR, chiến lược, hoặc mục tiêu cấp trên.",
    "out_of_scope": "Phần không thuộc phạm vi mục tiêu lần này.",
    "primary_user": "Người dùng cuối trực tiếp hoặc persona chính.",
    "secondary_stakeholders": "Các bên liên quan gián tiếp và cách họ bị ảnh hưởng.",
    "decision_maker": "Người có quyền phê duyệt, phủ quyết, hoặc ra quyết định.",
    "operator": "Người triển khai, vận hành, hoặc hỗ trợ sau khi ra mắt.",
    "concern": "Lo ngại, động cơ phản đối, hoặc rủi ro từ stakeholder.",
    "capability": "Năng lực sản phẩm hoặc hệ thống cần có.",
    "description": "Mô tả rõ ý nghĩa của năng lực hoặc hạng mục.",
    "priority": "Mức ưu tiên như Must, Should, Could.",
    "availability": "Trạng thái đã có sẵn, cần xây, cần mua, hoặc cần thuê ngoài.",
    "differentiator": "Năng lực tạo khác biệt cốt lõi cần tự xây.",
    "deferrable": "Năng lực có thể lùi sang giai đoạn sau mà không chặn MVP.",
    "time": "Ràng buộc thời gian, deadline, hoặc milestone cứng.",
    "budget": "Ràng buộc ngân sách hoặc phân bổ chi phí.",
    "technical": "Ràng buộc kỹ thuật, công nghệ bắt buộc, hoặc hệ thống cần tích hợp.",
    "people": "Ràng buộc nhân lực, năng lực đội ngũ, hoặc kỹ năng thiếu.",
    "compliance": "Ràng buộc pháp lý, tiêu chuẩn, bảo mật, hoặc chính sách.",
    "user_behavior": "Giả định về hành vi, nhu cầu, hoặc kỹ năng của người dùng.",
    "market": "Giả định về thị trường, cạnh tranh, quy mô, hoặc tăng trưởng.",
    "technical_feasibility": "Giả định về khả năng tích hợp, hiệu năng, hoặc vận hành kỹ thuật.",
    "riskiest": "Giả định có hậu quả lớn nhất nếu sai.",
    "validation": "Cách kiểm chứng giả định trước khi build hoặc ra quyết định.",
    "risk": "Sự kiện hoặc điều kiện bất lợi có thể xảy ra.",
    "likelihood": "Xác suất xảy ra của rủi ro.",
    "mitigation": "Chiến lược giảm thiểu, né tránh, chuyển giao, hoặc chấp nhận rủi ro.",
    "ownership": "Phạm vi kiểm soát hoặc chủ sở hữu theo dõi rủi ro.",
    "dependency": "Phụ thuộc bên ngoài, vendor, team khác, hoặc third-party.",
    "question": "Câu hỏi còn mở cần trả lời.",
    "domain": "Nhóm câu hỏi thuộc người dùng, kỹ thuật, kinh doanh, scope, hoặc team.",
    "decision_needed": "Quyết định cần được chốt dựa trên câu hỏi.",
    "research_needed": "Thông tin hoặc hoạt động research cần có để trả lời.",
    "blocker": "Mức độ câu hỏi có thể chặn Sprint 1 hoặc chặn team.",
}

# brd_key -> {"required": [...], "optional": [...]}
BRD_SLOTS: dict[str, dict[str, list[str]]] = {
    "intent": {
        "required": ["why_now", "sponsor", "expected_outcome", "success_state"],
        "optional": ["cost_of_inaction"],
    },
    "problem": {
        "required": ["who", "obstacle", "root_cause", "frequency", "impact"],
        "optional": ["journey_step", "workaround"],
    },
    "goal": {
        "required": ["business_goal", "user_goal", "metric", "target", "timeframe"],
        "optional": ["baseline", "alignment", "out_of_scope"],
    },
    "stakeholder": {
        "required": ["primary_user", "secondary_stakeholders", "decision_maker", "operator"],
        "optional": ["concern"],
    },
    "capability": {
        "required": ["capability", "description", "priority", "availability"],
        "optional": ["differentiator", "deferrable"],
    },
    "constraint": {
        "required": ["time", "budget", "technical", "people", "compliance"],
        "optional": [],
    },
    "assumption": {
        "required": ["user_behavior", "market", "technical_feasibility", "riskiest", "validation"],
        "optional": [],
    },
    "risk": {
        "required": ["risk", "likelihood", "impact", "mitigation"],
        "optional": ["ownership", "dependency"],
    },
    "open_question": {
        "required": ["question", "domain", "decision_needed", "research_needed", "blocker"],
        "optional": [],
    },
}

# Consecutive non-improving coverage turns before the gate relaxes and the hint
# switches from "ask the same slot" to "move on / propose" — bounds the elicitation
# loop well under max_agent_turns so a model that under-reports slots cannot nag forever.
COVERAGE_STALL_LIMIT = 2

# brd_key -> coverage threshold for required slots.
COVERAGE_THRESHOLD: dict[str, float] = {
    "intent": 0.8,
    "problem": 0.8,
    "goal": 0.8,
    "stakeholder": 0.8,
    "capability": 0.8,
    "constraint": 0.8,
    "assumption": 0.8,
    "risk": 0.8,
    "open_question": 0.8,
}


def compute_coverage(artifact_type: str, slot_assessment: dict[str, str]) -> dict[str, object]:
    slot_spec = BRD_SLOTS.get(artifact_type)
    if slot_spec is None:
        return {
            "slot_coverage": dict(slot_assessment or {}),
            "coverage_ratio": 1.0,
            "coverage_complete": True,
        }

    required_slots = slot_spec["required"]
    normalized = _normalize_slot_assessment(slot_assessment, required_slots)
    score = sum(_slot_score(normalized[slot]) for slot in required_slots)
    ratio = score / len(required_slots)
    threshold = COVERAGE_THRESHOLD[artifact_type]

    return {
        "slot_coverage": normalized,
        "coverage_ratio": ratio,
        "coverage_complete": ratio >= threshold,
    }


def _normalize_slot_assessment(slot_assessment: dict[str, str], required_slots: list[str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for slot, value in (slot_assessment or {}).items():
        normalized[slot] = value if value in {"filled", "partial", "empty"} else "empty"
    for slot in required_slots:
        normalized.setdefault(slot, "empty")
    return normalized


def _slot_score(status: str) -> float:
    if status == "filled":
        return 1.0
    if status == "partial":
        return 0.5
    return 0.0
