"""Re-run the identical fixture from test_baseline_benchmark.py and write the before/after delta.

Not part of the default `pytest tests/unit tests/integration -q` run (this directory is
outside both) — invoke explicitly: `pytest tests/benchmark/test_final_benchmark.py -s -q`.
"""

import json
import re
import statistics
from pathlib import Path

import pytest

from tests.benchmark.fixture import FIXTURE_TURN_COUNT, run_fixture
from tests.helpers import create_org, create_project, make_auth_headers

RUNS = 3
EVIDENCE_DIR = Path(__file__).resolve().parents[2] / "plans" / "260701-optimize-latency-token" / "evidence"
BASELINE_PATH = EVIDENCE_DIR / "benchmark-baseline.md"
FINAL_PATH = EVIDENCE_DIR / "benchmark-final.md"

METRIC_NAMES = [
    "total_latency_ms",
    "total_tokens",
    "analyze_latency_ms",
    "triage_latency_ms",
    "critique_latency_ms",
]


def _aggregate(all_runs: list[list[dict]]) -> dict[str, dict[str, float]]:
    """min/median/max per metric, aggregated across runs, summed across the fixture's turns."""
    totals = {name: [sum(t[name] for t in run) for run in all_runs] for name in METRIC_NAMES}
    return {
        name: {"min": min(values), "median": statistics.median(values), "max": max(values)}
        for name, values in totals.items()
    }


def _load_baseline_aggregate() -> dict[str, dict[str, float]]:
    """Re-derive the baseline aggregate from its recorded raw per-run, per-turn JSON, so the
    comparison is computed the same way on both sides rather than re-parsing a rendered table."""
    text = BASELINE_PATH.read_text(encoding="utf-8")
    match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
    if not match:
        raise AssertionError(f"could not find raw JSON block in {BASELINE_PATH}")
    baseline_runs = json.loads(match.group(1))
    return _aggregate(baseline_runs)


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
        "Same fixture and methodology as `evidence/benchmark-baseline.md`, re-run after the behavior "
        "quality changes.",
        "",
        "## Mode",
        "",
        "Identical to the baseline: mocked LLM client (`tests/benchmark/fixture.py:BenchmarkLLM`), no real "
        "API key in this environment. Latency is in-process overhead only, not real LLM API round-trip "
        "latency. Token counts are a deterministic size proxy (`len(payload) // 4`), not real "
        "provider-billed tokens — this proxy cannot show P1's (prompt caching) actual billing effect, "
        "since caching changes what is billed, not the payload size sent.",
        "",
        f"## Results ({RUNS} runs, values summed across all {FIXTURE_TURN_COUNT} turns per run)",
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
        "The delta above reflects the combined end-to-end effect of every implemented phase, not a "
        "clean per-phase breakdown (per this phase's own documented Attribution risk). Plausible "
        "contributors, without overclaiming precision:",
        "- `total_tokens`: P3 (bounded conversation history) and P4 (diff-based draft/ancestor sending) "
        "are the phases most likely to move this metric downward in this mocked-payload-size proxy — "
        "both reduce what is actually sent in `messages`/tool-read payloads. P1 (prompt caching) does "
        "not change payload size (it changes what is billed), so it is invisible to this proxy.",
        "- `analyze_latency_ms` / `total_latency_ms`: P6 (Bedrock/Mistral client reuse) and P9 "
        "(tool-loop early-exit) are the phases most likely to move this metric, but this fixture's "
        "mocked `BenchmarkLLM` does not exercise Bedrock/Mistral client construction or a real repeated "
        "tool-loop, so any observed change here is dominated by in-process overhead noise, not those "
        "phases' actual production effect.",
        "- P8 (cheap-model routing) and P7 (cached view rendering) are not observable in this fixture: "
        "P8 only changes which client is called (irrelevant to a mocked client), and P7's cache benefit "
        "requires a warm cross-turn cache hit that this 3-fresh-runs methodology does not specifically "
        "isolate.",
        "",
        "## No Numeric Target",
        "",
        "No pass/fail threshold was set for this plan (see plan.md Not Doing) — this report is "
        "evidence-gathering, reporting the measured delta honestly (including any regression), not a "
        "gate.",
        "",
        "## Raw per-run, per-turn data (final)",
        "",
        "```json",
    ]
    lines.append(json.dumps(all_runs, indent=2))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


@pytest.mark.asyncio
async def test_record_final_benchmark(client, db_session):
    headers = await make_auth_headers(client)
    org = await create_org(client, headers)
    project = await create_project(client, headers, org["id"])
    import uuid

    project_id = uuid.UUID(project["id"])

    all_runs = []
    for _ in range(RUNS):
        metrics = await run_fixture(db_session, project_id)
        assert len(metrics) == FIXTURE_TURN_COUNT
        all_runs.append(metrics)

    final_agg = _aggregate(all_runs)
    baseline_agg = _load_baseline_aggregate()
    delta = _delta(baseline_agg, final_agg)
    report = _render_report(baseline_agg, final_agg, delta, all_runs)
    FINAL_PATH.write_text(report, encoding="utf-8")

    for run in all_runs:
        for turn in run:
            assert turn["total_latency_ms"] >= 0
            assert turn["total_tokens"] > 0
