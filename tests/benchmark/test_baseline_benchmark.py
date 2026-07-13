"""Record baseline latency/token numbers for the canonical journeys.

Run explicitly with `pytest -m benchmark tests/benchmark/test_baseline_benchmark.py -s -q`.
"""

import pytest

from tests.benchmark.fixture import (
    FIXTURE_JOURNEY_COUNT,
    aggregate_runs,
    evidence_dir,
    raw_runs_block,
    record_runs,
)

pytestmark = [pytest.mark.benchmark, pytest.mark.evidence]

RUNS = 3
EVIDENCE_PATH = evidence_dir() / "benchmark-baseline.md"


def _render_report(all_runs: list[list[dict]], agg: dict[str, dict[str, float]]) -> str:
    lines = [
        "# Baseline Benchmark",
        "",
        "Sole ongoing benchmark lane for the canonical journeys - see `tests/benchmark/fixture.py`.",
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
    lines += ["", *raw_runs_block(all_runs, "Raw per-run, per-journey data")]
    return "\n".join(lines)


@pytest.mark.asyncio
async def test_record_baseline_benchmark(client, scenario_env, scenario_project):
    all_runs = await record_runs(client, scenario_env, scenario_project, RUNS)
    report = _render_report(all_runs, aggregate_runs(all_runs))
    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text(report, encoding="utf-8")
