"""Record baseline latency/token numbers for the canonical journeys.

Run explicitly with `pytest -m benchmark tests/benchmark/test_baseline_benchmark.py -s -q`.
The final benchmark re-runs the same journey fixture and writes the before/after
delta.
"""

from pathlib import Path

import pytest

from tests.benchmark.fixture import FIXTURE_JOURNEY_COUNT, aggregate_runs, run_fixture

pytestmark = [pytest.mark.benchmark, pytest.mark.evidence]

RUNS = 3
EVIDENCE_PATH = Path(__file__).resolve().parents[2] / "plans" / "260701-optimize-latency-token" / "evidence" / "benchmark-baseline.md"


def _render_report(all_runs: list[list[dict]], agg: dict[str, dict[str, float]]) -> str:
    lines = [
        "# Baseline Benchmark",
        "",
        "Recorded before any latency/token fix lands. Same canonical journey fixture and methodology as "
        "`evidence/benchmark-final.md` - see `tests/benchmark/fixture.py`.",
        "",
        "## Mode",
        "",
        "Scripted mock LLM through the real HTTP/graph scenario harness. Latency is in-process "
        "harness and application overhead, not real external LLM latency. Token counts are a "
        "deterministic payload-size proxy, not provider-billed usage.",
        "",
        f"## Fixture ({FIXTURE_JOURNEY_COUNT} canonical journeys)",
        "",
        "`tests/benchmark/fixture.py:run_fixture` runs the shared canonical journeys through "
        "`ScenarioDriver`, so integration, eval, and benchmark lanes exercise the same behavior.",
        "",
        f"## Results ({RUNS} runs, values summed across all {FIXTURE_JOURNEY_COUNT} journeys per run)",
        "",
        "| Metric | Min | Median | Max |",
        "| --- | --- | --- | --- |",
    ]
    for name, stats in agg.items():
        lines.append(f"| {name} | {stats['min']:.0f} | {stats['median']:.0f} | {stats['max']:.0f} |")
    lines += [
        "",
        "## Raw per-run, per-journey data",
        "",
        "```json",
    ]
    import json

    lines.append(json.dumps(all_runs, indent=2))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


@pytest.mark.asyncio
async def test_record_baseline_benchmark(client, scenario_env, scenario_project):
    all_runs = []
    for _ in range(RUNS):
        metrics = await run_fixture(client, scenario_env, scenario_project)
        assert len(metrics) == FIXTURE_JOURNEY_COUNT
        all_runs.append(metrics)

    agg = aggregate_runs(all_runs)
    report = _render_report(all_runs, agg)
    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text(report, encoding="utf-8")

    # Sanity: every journey recorded non-negative latency and a positive token estimate.
    for run in all_runs:
        for journey in run:
            assert journey["final_status"] == "completed"
            assert journey["total_latency_ms"] >= 0
            assert journey["total_tokens"] > 0
