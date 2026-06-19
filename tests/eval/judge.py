"""LLM-as-judge: chấm artifact requirements theo rubric, trả điểm có cấu trúc.

Tái dùng cơ chế `response_format` của `app.services.llm_clients` (giống
`ANALYSIS_SCHEMA`). Judge dùng model mạnh cố định (cấu hình ở `app.config`),
tách khỏi provider của session agent.
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
    """Chấm một artifact, trả dict {scores, overall, rationale}. Raise ValueError nếu sai shape."""
    result, _usage = await llm_client.generate(
        messages=[{"role": "user", "content": _build_prompt(artifact_text)}],
        system=_JUDGE_SYSTEM,
        max_tokens=1024,
        response_format=JUDGE_SCHEMA,
    )
    return _validate(result)


async def judge_multiple(artifact_text: str, rubric_criteria: dict, llm_client, n: int = 3) -> list[dict[str, Any]]:
    """Chạy judge n lần trên cùng input để đo inter-run variance (O4)."""
    return [await judge_artifact(artifact_text, rubric_criteria, llm_client) for _ in range(n)]
