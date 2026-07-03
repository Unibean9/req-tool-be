"""Reusable journey benchmark fixture.

The benchmark lane measures the same canonical end-to-end journeys used by
integration and eval. It records deterministic in-process latency plus a simple
payload-size token proxy from the scripted LLM calls. This keeps benchmark
coverage aligned with product behavior without maintaining a benchmark-only
conversation script.
"""

import time
import uuid
from typing import Any

from tests.integration.scenarios.driver import ScenarioDriver
from tests.integration.scenarios.library import CANONICAL_SCENARIOS

CHARS_PER_TOKEN = 4
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
