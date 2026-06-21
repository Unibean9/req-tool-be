from unittest.mock import AsyncMock

import pytest

from tests.eval import rubric
from tests.eval.judge import judge_artifact, judge_multiple


def _valid_result(overall: float = 0.8) -> dict:
    return {
        "scores": {
            "unambiguous": 0.8,
            "verifiable": 0.7,
            "complete": 0.9,
            "consistent": 0.85,
            "traceable": 0.6,
            "feasible": 0.75,
            "invest": None,
            "smart": 0.8,
        },
        "overall": overall,
        "rationale": "Artifact rõ ràng, đo lường được.",
    }


def _mock_client(result):
    client = AsyncMock()
    client.generate = AsyncMock(return_value=(result, {"input": 5, "output": 10, "total": 15}))
    return client


@pytest.mark.asyncio
async def test_judge_artifact_returns_structured_scores():
    client = _mock_client(_valid_result())

    result = await judge_artifact("Mục tiêu: tăng doanh thu 10%.", rubric.RUBRIC_CRITERIA, client)

    assert "scores" in result
    assert "overall" in result
    assert "rationale" in result
    assert isinstance(result["overall"], float)
    assert 0.0 <= result["overall"] <= 1.0


@pytest.mark.asyncio
async def test_judge_artifact_rejects_invalid_shape():
    client = _mock_client({"foo": "bar"})

    with pytest.raises(ValueError):
        await judge_artifact("Bất kỳ", rubric.RUBRIC_CRITERIA, client)


@pytest.mark.asyncio
async def test_judge_multiple_runs_n_times():
    client = AsyncMock()
    client.generate = AsyncMock(
        side_effect=[
            (_valid_result(0.80), None),
            (_valid_result(0.82), None),
            (_valid_result(0.78), None),
        ]
    )

    results = await judge_multiple("Mục tiêu X.", rubric.RUBRIC_CRITERIA, client, n=3)

    assert len(results) == 3
    for item in results:
        assert "overall" in item
