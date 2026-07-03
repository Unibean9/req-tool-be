"""Re-run the canonical journey benchmark and write the before/after delta.

Run explicitly with `pytest -m benchmark tests/benchmark/test_final_benchmark.py -s -q`.
"""

import pytest

from tests.benchmark.fixture import (
    FIXTURE_JOURNEY_COUNT,
    METRIC_NAMES,
    aggregate_runs,
    evidence_dir,
    load_recorded_runs,
    raw_runs_block,
    record_runs,
)

pytestmark = [pytest.mark.benchmark, pytest.mark.evidence]

RUNS = 3
EVIDENCE_DIR = evidence_dir()
BASELINE_PATH = EVIDENCE_DIR / "benchmark-baseline.md"
FINAL_PATH = EVIDENCE_DIR / "benchmark-final.md"


def _delta(baseline: dict[str, dict[str, float]], final: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        name: {
            "median_delta": final[name]["median"] - baseline[name]["median"],
            "median_delta_pct": (
                (final[name]["median"] - baseline[name]["median"]) / baseline[name]["median"] * 100
                if baseline[name]["median"]
                else 0.0
            ),
        }
        for name in METRIC_NAMES
    }


def _render_report(
    baseline_agg: dict[str, dict[str, float]],
    final_agg: dict[str, dict[str, float]],
    delta: dict[str, dict[str, float]],
    all_runs: list[list[dict]],
) -> str:
    lines = [
        "# Final Benchmark Comparison",
        "",
        "Same canonical journey fixture and methodology as `evidence/benchmark-baseline.md`, re-run after the behavior "
        "quality changes.",
        "",
        "## Mode",
        "",
        "Identical to the baseline: scripted mock LLM through the real HTTP/graph scenario harness. "
        "Latency is in-process overhead only. Token counts are a deterministic payload-size proxy, "
        "not provider-billed usage.",
        "",
        f"## Results ({RUNS} runs, values summed across all {FIXTURE_JOURNEY_COUNT} journeys per run)",
        "",
        "| Metric | Baseline Median | Final Median | Median Delta | Median Delta % |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name in METRIC_NAMES:
        b = baseline_agg[name]["median"]
        f = final_agg[name]["median"]
        d = delta[name]["median_delta"]
        dp = delta[name]["median_delta_pct"]
        lines.append(f"| {name} | {b:.0f} | {f:.0f} | {d:+.0f} | {dp:+.1f}% |")

    lines += [
        "",
        "## Baseline vs Final, full min/median/max",
        "",
        "| Metric | Baseline Min | Baseline Median | Baseline Max | Final Min | Final Median | Final Max |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name in METRIC_NAMES:
        b = baseline_agg[name]
        f = final_agg[name]
        lines.append(
            f"| {name} | {b['min']:.0f} | {b['median']:.0f} | {b['max']:.0f} | "
            f"{f['min']:.0f} | {f['median']:.0f} | {f['max']:.0f} |"
        )

    lines += [
        "",
        "## Attribution (approximate, not isolated per-phase measurement)",
        "",
        "The delta above reflects the combined end-to-end effect across the canonical journey set. "
        "It is not a clean per-phase attribution report.",
        "",
        "## No Numeric Target",
        "",
        "No pass/fail threshold was set for this plan (see plan.md Not Doing) - this report is "
        "evidence-gathering, reporting the measured delta honestly (including any regression), not a "
        "gate.",
        "",
        *raw_runs_block(all_runs, "Raw per-run, per-journey data (final)"),
    ]
    return "\n".join(lines)


@pytest.mark.asyncio
async def test_record_final_benchmark(client, scenario_env, scenario_project):
    all_runs = await record_runs(client, scenario_env, scenario_project, RUNS)
    final_agg = aggregate_runs(all_runs)
    baseline_agg = aggregate_runs(load_recorded_runs(BASELINE_PATH))
    delta = _delta(baseline_agg, final_agg)
    report = _render_report(baseline_agg, final_agg, delta, all_runs)
    FINAL_PATH.write_text(report, encoding="utf-8")
