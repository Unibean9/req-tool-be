"""Reusable journey benchmark fixture.

The benchmark lane measures the same canonical end-to-end journeys used by
integration and eval. It records deterministic in-process latency plus a simple
payload-size token proxy from the scripted LLM calls. This keeps benchmark
coverage aligned with product behavior without maintaining a benchmark-only
conversation script.
"""

import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

from tests.integration.scenarios.driver import ScenarioDriver
from tests.integration.scenarios.library import CANONICAL_SCENARIOS

CHARS_PER_TOKEN = 4

_DEFAULT_EVIDENCE_DIR = Path(__file__).resolve().parent / "evidence"


def evidence_dir() -> Path:
    """Directory where benchmark evidence is written and read.

    Defaults to a benchmark-owned ``evidence/`` folder so the lane is not tied to
    any one plan. Override with ``BENCHMARK_EVIDENCE_DIR`` to point elsewhere.
    """
    override = os.environ.get("BENCHMARK_EVIDENCE_DIR")
    return Path(override) if override else _DEFAULT_EVIDENCE_DIR
FIXTURE_JOURNEY_COUNT = len(CANONICAL_SCENARIOS)
METRIC_NAMES = [
    "total_latency_ms",
    "total_tokens",
    "llm_calls",
    "tool_call_count",
    "artifact_count",
]


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def _call_token_proxy(call: dict[str, Any]) -> int:
    payload = " ".join(
        [
            str(call.get("route", "")),
            str(call.get("system", "")),
            str(call.get("last_message", "")),
            str(call.get("tool_names", "")),
            str(call.get("result", "")),
        ]
    )
    return _estimate_tokens(payload)


async def run_fixture(client, scenario_env, scenario_project) -> list[dict[str, Any]]:
    """Run every canonical journey once and return per-journey metrics."""
    headers, project = scenario_project
    project_id = uuid.UUID(project["id"])
    metrics: list[dict[str, Any]] = []

    for factory in CANONICAL_SCENARIOS:
        scenario = factory()
        driver = ScenarioDriver(client, scenario_env, headers, project_id, scenario)

        started = time.monotonic()
        recorder = await driver.run()
        total_latency_ms = int((time.monotonic() - started) * 1000)

        artifacts = await driver.executed_artifacts()
        calls = list(getattr(scenario.llm, "calls", []))
        tool_call_count = sum(
            len(step["snapshot"].get("tool_calls") or [])
            for step in recorder.steps
        )

        metrics.append(
            {
                "journey": scenario.name,
                "final_status": recorder.summary.get("final_status"),
                "total_latency_ms": total_latency_ms,
                "total_tokens": sum(_call_token_proxy(call) for call in calls),
                "llm_calls": len(calls),
                "tool_call_count": tool_call_count,
                "artifact_count": len(artifacts),
                "step_count": len(recorder.steps),
            }
        )

    return metrics


async def record_runs(client, scenario_env, scenario_project, runs: int) -> list[list[dict[str, Any]]]:
    """Run the canonical journey fixture ``runs`` times with per-journey sanity checks.

    Shared by the baseline and final benchmark tests so a new recording lane never
    re-implements the run loop or the sanity invariants.
    """
    all_runs: list[list[dict[str, Any]]] = []
    for _ in range(runs):
        metrics = await run_fixture(client, scenario_env, scenario_project)
        assert len(metrics) == FIXTURE_JOURNEY_COUNT
        for journey in metrics:
            assert journey["final_status"] == "completed"
            assert journey["total_latency_ms"] >= 0
            assert journey["total_tokens"] > 0
        all_runs.append(metrics)
    return all_runs


_RAW_JSON_RE = re.compile(r"```json\n(.*?)\n```", re.DOTALL)


def raw_runs_block(all_runs: list[list[dict[str, Any]]], heading: str) -> list[str]:
    """Render the raw per-run JSON block that :func:`load_recorded_runs` reads back."""
    return [f"## {heading}", "", "```json", json.dumps(all_runs, indent=2), "```", ""]


def load_recorded_runs(path: Path) -> list[list[dict[str, Any]]]:
    """Re-read the raw per-run data embedded in a previously recorded report."""
    match = _RAW_JSON_RE.search(path.read_text(encoding="utf-8"))
    if not match:
        raise AssertionError(f"could not find raw JSON block in {path}")
    return json.loads(match.group(1))


def aggregate_runs(all_runs: list[list[dict[str, Any]]]) -> dict[str, dict[str, float]]:
    """Aggregate numeric benchmark metrics across all canonical journeys."""
    import statistics

    totals = {
        name: [sum(float(row[name]) for row in run) for run in all_runs]
        for name in METRIC_NAMES
    }
    return {
        name: {"min": min(values), "median": statistics.median(values), "max": max(values)}
        for name, values in totals.items()
    }
