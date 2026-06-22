"""Production judge logic for the in-loop run_critique tool (spec §6.6, §9.3).

Imports only app.graphs.rubric — never nodes/agent_tools (no cycle) and never tests/ (production
must not depend on test infrastructure). tests/eval may import from here, not the reverse.

run_critique targets a draft body with one critique `mode` and returns a compact report
{mode, score, findings, suggestions}. When no LLM client is configured the judge degrades to a
well-defined empty report instead of raising, so the tool-loop never crashes on a missing key.
"""

from typing import Any

from app.graphs.rubric import render_criteria_block

# Critique angles the analyst may request (spec §6.6).
CRITIQUE_MODES: tuple[str, ...] = (
    "clarity",
    "completeness",
    "consistency",
    "feasibility",
    "testability",
    "traceability",
    "six_hats",
    "swot",
    "risk_review",
)
_DEFAULT_MODE = "completeness"

# Short focus per mode, embedded in the judge prompt so the score reflects the requested angle.
_MODE_FOCUS: dict[str, str] = {
    "clarity": "Độ rõ ràng và không mơ hồ của từng phát biểu.",
    "completeness": "Độ đầy đủ: actor, điều kiện, kết quả mong đợi có thiếu không.",
    "consistency": "Tính nhất quán nội tại và với các phần khác.",
    "feasibility": "Tính khả thi trong ràng buộc kỹ thuật, thời gian, nguồn lực.",
    "testability": "Khả năng kiểm chứng: tiêu chí đo lường, cách kiểm thử.",
    "traceability": "Khả năng truy vết ngược nguồn và xuôi tới artifact con.",
    "six_hats": "Soi qua sáu chiếc mũ tư duy (dữ kiện, cảm xúc, rủi ro, lợi ích, sáng tạo, điều phối).",
    "swot": "Điểm mạnh, điểm yếu, cơ hội, thách thức.",
    "risk_review": "Rủi ro tiềm ẩn, xác suất và mức tác động.",
}

JUDGE_SCHEMA: dict[str, Any] = {
    "name": "critique_result",
    "schema": {
        "type": "object",
        "properties": {
            "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "findings": {"type": "array", "items": {"type": "string"}},
            "suggestions": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["score", "findings", "suggestions"],
    },
}

_JUDGE_SYSTEM = (
    "Bạn là chuyên gia kỹ nghệ yêu cầu. Phản biện artifact theo góc độ được chỉ định, "
    "chấm điểm 0.0–1.0, liệt kê findings (điểm yếu cụ thể) và suggestions (cách cải thiện) bằng tiếng Việt."
)


def _normalize_mode(mode: str) -> str:
    return mode if mode in CRITIQUE_MODES else _DEFAULT_MODE


def _build_judge_prompt(body: str, mode: str) -> str:
    return (
        f"Phản biện artifact dưới đây, tập trung vào góc độ '{mode}': {_MODE_FOCUS.get(mode, '')}\n\n"
        f"RUBRIC THAM CHIẾU:\n{render_criteria_block()}\n\n"
        f"ARTIFACT:\n{body or '(trống)'}"
    )


async def _invoke_judge(body: str, mode: str, llm_client: Any = None) -> dict[str, Any]:
    """Critique `body` along `mode`. Degrades to an empty report when no LLM client is configured."""
    mode = _normalize_mode(mode)
    if llm_client is None:
        return {"mode": mode, "score": 0.0, "findings": [], "suggestions": ["no_llm_client"]}

    result, _usage = await llm_client.generate(
        messages=[{"role": "user", "content": _build_judge_prompt(body, mode)}],
        system=_JUDGE_SYSTEM,
        max_tokens=1024,
        response_format=JUDGE_SCHEMA,
    )
    if not isinstance(result, dict):
        return {"mode": mode, "score": 0.0, "findings": [], "suggestions": []}
    return {
        "mode": mode,
        "score": float(result.get("score", 0.0) or 0.0),
        "findings": list(result.get("findings", []) or []),
        "suggestions": list(result.get("suggestions", []) or []),
    }
