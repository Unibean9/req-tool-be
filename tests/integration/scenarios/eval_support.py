"""Shared eval helpers for scenario tests.

Default mode uses a mock judge (deterministic, no API key). The integration
variant reuses the real LLM judge from tests/eval when JUDGE_API_KEY is set.
"""

from typing import Any
from unittest.mock import AsyncMock

from tests.eval import rubric
from tests.eval.judge import judge_artifact


def mock_judge(overall: float = 0.82) -> AsyncMock:
    """An LLM stub shaped like the real judge response (see JUDGE_SCHEMA)."""
    client = AsyncMock()
    client.generate = AsyncMock(
        return_value=(
            {
                "scores": {
                    "unambiguous": 0.85,
                    "verifiable": 0.8,
                    "complete": 0.85,
                    "consistent": 0.85,
                    "traceable": 0.75,
                    "feasible": 0.8,
                    "invest": None,
                    "smart": None,
                },
                "overall": overall,
                "rationale": "Artifact is clear, co pham vi va tac dong cu the.",
            },
            None,
        )
    )
    return client


async def score_artifacts(artifacts: list[dict[str, Any]], judge_client) -> list[dict[str, Any]]:
    """Run the judge over a list of {artifact_type, title, body} artifacts."""
    scored = []
    for art in artifacts:
        text = f"{art['title']}\n\n{art['body']}".strip()
        score = await judge_artifact(text, rubric.RUBRIC_CRITERIA, judge_client)
        scored.append({**art, "score": score})
    return scored
