"""Eval baseline harness.

Runs the whole golden set through the judge, printing a per-criterion
baseline table and inter-run variance (O3 + O4). Does NOT assert hard
thresholds at this stage — the goal is to print the baseline, not gate quality.

- Mock test (default): no real LLM call, verifies the harness runs with the
  right shape.
- Integration test (marker `integration`): uses a real LLM judge, only runs
  when the shared LLM_API_KEY environment variable is set.
"""

import json
import statistics
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from tests.eval import rubric
from tests.eval.judge import judge_artifact, judge_multiple

_FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_fixtures() -> list[dict]:
    # Exclude *_weak.json — low-quality fixtures used by the quality-gate eval, not the baseline.
    paths = [p for p in sorted(_FIXTURES_DIR.glob("*.json")) if not p.stem.endswith("_weak")]
    return [json.loads(p.read_text(encoding="utf-8")) for p in paths]


async def run_eval_baseline(llm_client, n: int = 3) -> list[dict]:
    """Score each fixture: baseline once + variance over n runs. Returns the result list and prints a table."""
    rows: list[dict] = []
    print("\n=== EVAL BASELINE ===")
    for fixture in _load_fixtures():
        content = fixture["content"]
        baseline = await judge_artifact(content, rubric.RUBRIC_CRITERIA, llm_client)
        runs = await judge_multiple(content, rubric.RUBRIC_CRITERIA, llm_client, n=n)
        overalls = [r["overall"] for r in runs]
        stdev = statistics.pstdev(overalls) if len(overalls) > 1 else 0.0

        rows.append({"artifact_type": fixture["artifact_type"], "baseline": baseline, "overall_stdev": stdev})

        scores = baseline["scores"]
        score_str = " ".join(f"{k}={scores.get(k)}" for k in scores)
        print(f"[{fixture['artifact_type']}] overall={baseline['overall']:.3f} stdev={stdev:.3f} | {score_str}")
    return rows


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
        "rationale": "ok",
    }


def test_fixtures_count_is_three():
    assert len(_load_fixtures()) == 3


@pytest.mark.asyncio
async def test_baseline_runs_over_all_fixtures_with_mock(capsys):
    client = AsyncMock()
    client.generate = AsyncMock(return_value=(_valid_result(), None))

    rows = await run_eval_baseline(client, n=3)

    assert len(rows) == 3
    for row in rows:
        assert "baseline" in row
        assert "overall_stdev" in row
    assert "EVAL BASELINE" in capsys.readouterr().out


@pytest.mark.integration
@pytest.mark.asyncio
async def test_baseline_with_real_judge(capsys):
    from tests.eval.config import judge_settings

    if not judge_settings.judge_api_key:
        pytest.skip("LLM_API_KEY is required in .env.test to run the real judge")

    from app.models.llm_provider import ProviderType
    from app.services.llm_clients import LLMClientFactory

    client = LLMClientFactory.create(
        provider_type=ProviderType(judge_settings.judge_provider),
        api_key=judge_settings.judge_api_key,
        model=judge_settings.judge_model,
        region=judge_settings.judge_region,
        secret_key=judge_settings.judge_secret_key or None,
    )

    rows = await run_eval_baseline(client, n=3)

    assert len(rows) == 3
    assert "EVAL BASELINE" in capsys.readouterr().out
