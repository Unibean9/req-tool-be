"""Record baseline latency/token numbers before any latency/token fix lands.

Not part of the default `pytest tests/unit tests/integration -q` run (this directory is
outside both) — invoke explicitly: `pytest tests/benchmark/test_baseline_benchmark.py -s -q`.
The final benchmark re-runs the identical fixture (tests/benchmark/fixture.py) via
test_final_benchmark.py and writes evidence/benchmark-final.md for the before/after delta.
"""

import statistics
from pathlib import Path

import pytest

from tests.benchmark.fixture import FIXTURE_TURN_COUNT, run_fixture
from tests.helpers import create_org, create_project, make_auth_headers

RUNS = 3
EVIDENCE_PATH = Path(__file__).resolve().parents[2] / "plans" / "260701-optimize-latency-token" / "evidence" / "benchmark-baseline.md"


def _aggregate(all_runs: list[list[dict]]) -> dict[str, dict[str, float]]:
    """min/median/max per metric, aggregated across runs, summed across the fixture's turns."""
    totals = {
        "total_latency_ms": [sum(t["total_latency_ms"] for t in run) for run in all_runs],
        "total_tokens": [sum(t["total_tokens"] for t in run) for run in all_runs],
        "analyze_latency_ms": [sum(t["analyze_latency_ms"] for t in run) for run in all_runs],
        "triage_latency_ms": [sum(t["triage_latency_ms"] for t in run) for run in all_runs],
        "critique_latency_ms": [sum(t["critique_latency_ms"] for t in run) for run in all_runs],
    }
    return {
        name: {"min": min(values), "median": statistics.median(values), "max": max(values)}
        for name, values in totals.items()
    }


def _render_report(all_runs: list[list[dict]], agg: dict[str, dict[str, float]]) -> str:
    lines = [
        "# Baseline Benchmark",
        "",
        "Recorded before any latency/token fix lands. Same fixture and methodology as "
        "`evidence/benchmark-final.md` — see `tests/benchmark/fixture.py`.",
        "",
        "## Mode",
        "",
        "Mocked LLM client (`tests/benchmark/fixture.py:BenchmarkLLM`) — no real API key in this "
        "environment. Consequences:",
        "- Latency here is **in-process overhead only** (DB session open/close inside `analyze_node`, "
        "the triage/analyze/critique call plumbing, in-memory prompt assembly). It does NOT capture "
        "real LLM API round-trip latency.",
        "- Token counts are a **deterministic size proxy** (`len(payload) // 4` on the exact "
        "messages/system/tools sent to `generate()`), not real provider-billed tokens. This proxy "
        "cannot show P1's (prompt caching) actual billing effect, since caching changes what is "
        "billed, not the payload size sent. It CAN show P3's history-capping effect (fewer/smaller "
        "history messages in the payload) and P2's same-turn critique cache-boundary effect if that "
        "phase changes what is embedded in the critique prompt.",
        "",
        f"## Fixture ({FIXTURE_TURN_COUNT} turns)",
        "",
        "`tests/benchmark/fixture.py:run_fixture` — triage_node + analyze_node called directly per "
        "turn (real DB session_factory, real AgentRun rows), with one same-turn `_invoke_judge` call "
        "at turn 5 (mirrors run_critique firing in the same turn as analyze) and a growing draft body "
        "across write_draft turns. Direct node calls were used instead of the full graph + ToolNode "
        "+ interrupt wiring — that wiring is orthogonal to what P1/P2/P3/P5/P6 change and adds "
        "complexity disproportionate to a --fast benchmark harness.",
        "",
        f"## Results ({RUNS} runs, values summed across all {FIXTURE_TURN_COUNT} turns per run)",
        "",
        "| Metric | Min | Median | Max |",
        "| --- | --- | --- | --- |",
    ]
    for name, stats in agg.items():
        lines.append(f"| {name} | {stats['min']:.0f} | {stats['median']:.0f} | {stats['max']:.0f} |")
    lines += [
        "",
        "## Raw per-run, per-turn data",
        "",
        "```json",
    ]
    import json

    lines.append(json.dumps(all_runs, indent=2))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


@pytest.mark.asyncio
async def test_record_baseline_benchmark(client, db_session):
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

    agg = _aggregate(all_runs)
    report = _render_report(all_runs, agg)
    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text(report, encoding="utf-8")

    # Sanity: every turn recorded a non-negative latency and a positive token estimate.
    for run in all_runs:
        for turn in run:
            assert turn["total_latency_ms"] >= 0
            assert turn["total_tokens"] > 0
