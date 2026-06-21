"""LLM-as-judge: score requirements artifacts against the rubric, returning structured scores.

Reuses the `response_format` mechanism of `app.services.llm_clients` (like
`ANALYSIS_SCHEMA`). The judge uses a fixed strong model (configured in
`app.config`), decoupled from the agent session's provider.
"""

from typing import Any

from tests.eval.rubric import render_criteria_block

# 6 tiêu chí 29148 luôn bắt buộc; invest/smart có thể null khi không áp dụng
_REQUIRED_SCORE_KEYS = ("unambiguous", "verifiable", "complete", "consistent", "traceable", "feasible")
_OPTIONAL_SCORE_KEYS = ("invest", "smart")

_SCORE_FIELD = {"type": ["number", "null"], "minimum": 0.0, "maximum": 1.0}

JUDGE_SCHEMA: dict[str, Any] = {
    "name": "judge_result",
    "schema": {
        "type": "object",
        "properties": {
            "scores": {
                "type": "object",
                "properties": {
                    **{key: {"type": "number", "minimum": 0.0, "maximum": 1.0} for key in _REQUIRED_SCORE_KEYS},
                    **{key: _SCORE_FIELD for key in _OPTIONAL_SCORE_KEYS},
                },
                "required": list(_REQUIRED_SCORE_KEYS),
            },
            "overall": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "rationale": {"type": "string"},
        },
        "required": ["scores", "overall", "rationale"],
    },
}

_JUDGE_SYSTEM = (
    "Bạn là chuyên gia kỹ nghệ yêu cầu (requirements engineering). "
    "Chấm artifact theo từng tiêu chí trong rubric, mỗi tiêu chí 0.0–1.0. "
    "Trả null cho invest/smart nếu tiêu chí không áp dụng cho loại artifact này. "
    "overall là điểm tổng hợp 0.0–1.0; rationale giải thích ngắn gọn bằng tiếng Việt."
)


def _build_prompt(artifact_text: str) -> str:
    return (
        "Chấm artifact requirements sau theo rubric.\n\n"
        f"RUBRIC:\n{render_criteria_block()}\n\n"
        f"ARTIFACT:\n{artifact_text}"
    )


def _validate(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise ValueError("Judge phải trả JSON object")
    scores = result.get("scores")
    if not isinstance(scores, dict):
        raise ValueError("Judge thiếu trường 'scores'")
    for key in _REQUIRED_SCORE_KEYS:
        if key not in scores:
            raise ValueError(f"Judge thiếu điểm tiêu chí bắt buộc: {key}")
    if not isinstance(result.get("overall"), (int, float)):
        raise ValueError("Judge thiếu 'overall' dạng số")
    if not isinstance(result.get("rationale"), str):
        raise ValueError("Judge thiếu 'rationale' dạng chuỗi")
    result["overall"] = float(result["overall"])
    return result


async def judge_artifact(artifact_text: str, rubric_criteria: dict, llm_client) -> dict[str, Any]:
    """Score one artifact, returning {scores, overall, rationale}. Raises ValueError on a bad shape."""
    result, _usage = await llm_client.generate(
        messages=[{"role": "user", "content": _build_prompt(artifact_text)}],
        system=_JUDGE_SYSTEM,
        max_tokens=1024,
        response_format=JUDGE_SCHEMA,
    )
    return _validate(result)


async def judge_multiple(artifact_text: str, rubric_criteria: dict, llm_client, n: int = 3) -> list[dict[str, Any]]:
    """Run the judge n times on the same input to measure inter-run variance (O4)."""
    return [await judge_artifact(artifact_text, rubric_criteria, llm_client) for _ in range(n)]


# A judge for the conversation as a whole, independent of the per-artifact rubric judge above.
# It measures multi-angle behaviour (S1/S2): how varied the agent's modes were and how many
# turns it proactively switched. Kept separate so it never freezes the checkpoint interface (N1).
CONVERSATION_JUDGE_SCHEMA: dict[str, Any] = {
    "name": "conversation_judge_result",
    "schema": {
        "type": "object",
        "properties": {
            "mode_variety": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "proactive_count": {"type": "integer", "minimum": 0},
            "rationale": {"type": "string"},
        },
        "required": ["mode_variety", "proactive_count", "rationale"],
    },
}

_CONVERSATION_JUDGE_SYSTEM = (
    "Bạn đánh giá một hội thoại giữa trợ lý phân tích yêu cầu và người dùng theo góc độ đa chế độ. "
    "mode_variety (0.0–1.0): mức độ trợ lý sử dụng nhiều chế độ khác nhau (hỏi/đánh giá/khám phá) "
    "thay vì chỉ hỏi. proactive_count: số lượt trợ lý CHỦ ĐỘNG chuyển khỏi chế độ hỏi-đáp thuần. "
    "rationale: giải thích ngắn gọn bằng tiếng Việt."
)


def _validate_conversation(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise ValueError("Conversation judge phải trả JSON object")
    if not isinstance(result.get("mode_variety"), (int, float)):
        raise ValueError("Conversation judge thiếu 'mode_variety' dạng số")
    if not isinstance(result.get("proactive_count"), int) or isinstance(result.get("proactive_count"), bool):
        raise ValueError("Conversation judge thiếu 'proactive_count' dạng số nguyên")
    if not isinstance(result.get("rationale"), str):
        raise ValueError("Conversation judge thiếu 'rationale' dạng chuỗi")
    result["mode_variety"] = float(result["mode_variety"])
    return result


async def judge_conversation(conversation_text: str, llm_client) -> dict[str, Any]:
    """Score one conversation for multi-angle behaviour. Raises ValueError on a bad shape."""
    result, _usage = await llm_client.generate(
        messages=[{"role": "user", "content": f"Đánh giá hội thoại sau:\n\n{conversation_text}"}],
        system=_CONVERSATION_JUDGE_SYSTEM,
        max_tokens=512,
        response_format=CONVERSATION_JUDGE_SCHEMA,
    )
    return _validate_conversation(result)
