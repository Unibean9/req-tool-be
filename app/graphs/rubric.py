"""Rubric for scoring requirements-artifact quality.

Pure Python — NO LLM or DB dependency. Defines the criteria from
ISO/IEC/IEEE 29148 (6 quality characteristics) plus INVEST (user story) and
SMART (goal). Each criterion holds `name`, `description` and `guidance`
(instructions for the judge when scoring 0.0–1.0).
"""

# key → {"name", "description", "guidance"}
RUBRIC_CRITERIA: dict[str, dict[str, str]] = {
    # --- 6 ISO/IEC/IEEE 29148 quality characteristics ---
    "unambiguous": {
        "name": "Không mơ hồ (Unambiguous)",
        "description": "Phát biểu chỉ có một cách hiểu; không dùng từ mơ hồ (nhanh, dễ dùng, tối ưu, thân thiện).",
        "guidance": "Điểm thấp nếu chứa weasel words hoặc diễn giải đa nghĩa; cao nếu đo lường và cụ thể.",
    },
    "verifiable": {
        "name": "Kiểm chứng được (Verifiable)",
        "description": "Có thể kiểm tra/đo lường để xác nhận artifact đã được thoả mãn.",
        "guidance": "Điểm cao nếu có tiêu chí đo lường, ngưỡng, hoặc cách kiểm thử rõ ràng.",
    },
    "complete": {
        "name": "Đầy đủ (Complete)",
        "description": "Bao phủ đủ thông tin cần thiết, không để khoảng trống quan trọng.",
        "guidance": "Điểm thấp nếu thiếu actor, điều kiện, hoặc kết quả mong đợi.",
    },
    "consistent": {
        "name": "Nhất quán (Consistent)",
        "description": "Không mâu thuẫn nội tại và không xung đột với các artifact khác.",
        "guidance": "Điểm thấp nếu phát biểu tự mâu thuẫn hoặc trùng lặp gây nhiễu.",
    },
    "traceable": {
        "name": "Truy vết được (Traceable)",
        "description": "Có thể liên kết ngược về nguồn (intent/problem/goal) và xuôi tới artifact con.",
        "guidance": "Điểm cao nếu nêu rõ lý do/nguồn gốc và liên hệ tới mục tiêu cấp trên.",
    },
    "feasible": {
        "name": "Khả thi (Feasible)",
        "description": "Thực hiện được trong ràng buộc kỹ thuật, thời gian và nguồn lực hợp lý.",
        "guidance": "Điểm thấp nếu yêu cầu phi thực tế hoặc vượt ràng buộc đã biết.",
    },
    # --- INVEST (user story) + SMART (goal) — applied per artifact_type ---
    "invest": {
        "name": "INVEST (user story)",
        "description": "Independent, Negotiable, Valuable, Estimable, Small, Testable.",
        "guidance": "Chỉ chấm khi artifact là story/epic; trả null nếu không áp dụng.",
    },
    "smart": {
        "name": "SMART (goal)",
        "description": "Specific, Measurable, Achievable, Relevant, Time-bound.",
        "guidance": "Chỉ chấm khi artifact là goal; trả null nếu không áp dụng.",
    },
}


def render_criteria_block() -> str:
    """Render the rubric as a text block to embed in the judge prompt."""
    lines = []
    for key, spec in RUBRIC_CRITERIA.items():
        lines.append(f"- {key} ({spec['name']}): {spec['description']} {spec['guidance']}")
    return "\n".join(lines)
