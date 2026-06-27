from unittest.mock import AsyncMock

import pytest

from tests.eval import rubric
from tests.eval.judge import judge_artifact, judge_conversation, judge_multiple


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
        "rationale": "Artifact is clear, measurable.",
    }


def _mock_client(result):
    client = AsyncMock()
    client.generate = AsyncMock(return_value=(result, {"input": 5, "output": 10, "total": 15}))
    return client


@pytest.mark.asyncio
async def test_judge_artifact_returns_structured_scores():
    client = _mock_client(_valid_result())

    result = await judge_artifact("Goal: increase revenue by 10%.", rubric.RUBRIC_CRITERIA, client)

    assert "scores" in result
    assert "overall" in result
    assert "rationale" in result
    assert isinstance(result["overall"], float)
    assert 0.0 <= result["overall"] <= 1.0


@pytest.mark.asyncio
async def test_judge_artifact_rejects_invalid_shape():
    client = _mock_client({"foo": "bar"})

    with pytest.raises(ValueError):
        await judge_artifact("Bat ky", rubric.RUBRIC_CRITERIA, client)


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

    results = await judge_multiple("Goal X.", rubric.RUBRIC_CRITERIA, client, n=3)

    assert len(results) == 3
    for item in results:
        assert "overall" in item


# ---------------------------------------------------------------------------
# Multi-angle: conversation judge (independent of the artifact judge)
# ---------------------------------------------------------------------------

def _valid_conversation_result(proactive_count: int = 2) -> dict:
    return {
        "mode_variety": 0.66,
        "proactive_count": proactive_count,
        "rationale": "Agent chu dong chuyen sang critique roi explore.",
    }


@pytest.mark.asyncio
async def test_judge_conversation_returns_structured_shape():
    client = AsyncMock()
    client.generate = AsyncMock(return_value=(_valid_conversation_result(), {"total": 12}))

    result = await judge_conversation("user: ...\nagent: ...", client)

    assert isinstance(result["proactive_count"], int)
    assert 0.0 <= result["mode_variety"] <= 1.0
    assert isinstance(result["rationale"], str)


@pytest.mark.asyncio
async def test_judge_conversation_rejects_invalid_shape():
    client = AsyncMock()
    client.generate = AsyncMock(return_value=({"foo": "bar"}, None))

    with pytest.raises(ValueError):
        await judge_conversation("bat ky", client)
