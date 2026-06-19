"""Eval baseline harness.

Chạy toàn bộ golden set qua judge, in bảng điểm baseline từng tiêu chí và
inter-run variance (O3 + O4). KHÔNG assert ngưỡng cứng ở giai đoạn này —
mục tiêu là in baseline, không gate chất lượng.

- Test mock (mặc định): không gọi LLM thật, kiểm tra harness chạy đúng shape.
- Test integration (marker `integration`): dùng judge LLM thật, chỉ chạy khi
  có biến môi trường JUDGE_API_KEY.
"""

import json
import os
import statistics
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from tests.eval import rubric
from tests.eval.judge import judge_artifact, judge_multiple

_FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_fixtures() -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(_FIXTURES_DIR.glob("*.json"))]


async def run_eval_baseline(llm_client, n: int = 3) -> list[dict]:
    """Chấm từng fixture: baseline 1 lần + variance qua n lần. Trả list kết quả và in bảng."""
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
        "rationale": "ổn",
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
    if not os.getenv("JUDGE_API_KEY"):
        pytest.skip("Cần JUDGE_API_KEY để chạy judge thật")

    from app.config import settings
    from app.models.llm_provider import ProviderType
    from app.services.llm_clients import LLMClientFactory

    client = LLMClientFactory.create(
        provider_type=ProviderType(settings.judge_provider),
        api_key=os.environ["JUDGE_API_KEY"],
        model=settings.judge_model,
    )

    rows = await run_eval_baseline(client, n=3)

    assert len(rows) == 3
    assert "EVAL BASELINE" in capsys.readouterr().out
