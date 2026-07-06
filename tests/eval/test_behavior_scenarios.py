"""Behavior eval suite: 4 scripted scenarios + metric extraction.

Stub mode (default, CI): the analyst brain is a ScriptedLLM — green tests guard harness
mechanics (driver, metric extraction, report shape), NOT agent behavior. All before/after
behavior claims come from live-mode runs.

Live mode: set BEHAVIOR_EVAL_MODE=live with credentials in `.env.test` (see tests/eval/config.py).
The same scenario actions run against the real model; metric reports + transcripts are written to
plans/260702-agent-behavior-quality/evidence/baseline/ for the recorded baseline.
"""

import json
import os
import uuid
from pathlib import Path

import pytest

from app.eval.behavior_metrics import behavior_report, extract_behavior_metrics
from tests.eval.behavior_scenarios import BEHAVIOR_SCENARIOS
from tests.integration.scenarios.driver import ScenarioDriver

pytestmark = [pytest.mark.eval, pytest.mark.evidence]

_MODE = os.environ.get("BEHAVIOR_EVAL_MODE", "stub")

# Live runs get extra generic turns beyond the scripted actions: the real model may need more
# exchanges to reach a draft proposal, and turns_to_first_draft should measure that, not be
# truncated by the script length. No-op approve_all steps are harmless.
_LIVE_PADDING = [
    {"type": "send", "content": "Ban da du thong tin roi, hay de xuat draft cho toi xem."},
    {"type": "approve_all"},
    {"type": "send", "content": "Tiep tuc hoan thien va de xuat draft."},
    {"type": "approve_all"},
]
_BASELINE_DIR = Path(__file__).parents[2] / "plans" / "260702-agent-behavior-quality" / "evidence" / "baseline"


def _live_client():
    from app.models.llm_provider import ProviderType
    from app.services.llm_clients import LLMClientFactory
    from tests.eval.config import judge_settings

    if not judge_settings.llm_api_key:
        pytest.skip("BEHAVIOR_EVAL_MODE=live requires llm_api_key in .env.test")
    return LLMClientFactory.create(
        provider_type=ProviderType(judge_settings.llm_provider_type),
        api_key=judge_settings.llm_api_key,
        model=judge_settings.llm_model_name,
        region=judge_settings.llm_region,
        secret_key=judge_settings.llm_secret_key or None,
    )


def _report_dir(tmp_path: Path) -> Path:
    if _MODE == "live":
        # Sweep/final-eval runs redirect output away from the committed baseline via
        # BEHAVIOR_EVAL_OUTDIR (per-cell dirs); unset keeps the recorded baseline destination.
        override = os.environ.get("BEHAVIOR_EVAL_OUTDIR")
        out = Path(override) if override else _BASELINE_DIR
        out.mkdir(parents=True, exist_ok=True)
        return out

    out = tmp_path / "behavior-eval"
    out.mkdir(parents=True, exist_ok=True)
    return out


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario_factory", BEHAVIOR_SCENARIOS, ids=lambda f: f.__name__)
async def test_behavior_scenario(scenario_factory, client, scenario_env, scenario_project, tmp_path):
    headers, project = scenario_project
    scenario = scenario_factory()
    model_id = None
    if _MODE == "live":
        scenario.llm = _live_client()
        model_id = getattr(getattr(scenario.llm, "config", None), "model", None)
        scenario.actions = [*scenario.actions, *_LIVE_PADDING]

    driver = ScenarioDriver(client, scenario_env, headers, uuid.UUID(project["id"]), scenario)
    recorder = await driver.run()
    artifacts = await driver.executed_artifacts()

    transcript = recorder.to_dict()
    report = behavior_report(transcript, artifacts, mode=_MODE, model=model_id)

    out_dir = _report_dir(tmp_path)
    if _MODE != "live" or not os.environ.get("BEHAVIOR_EVAL_METRICS_ONLY"):
        recorder.write(out_dir)
    # Live baseline runs each scenario several times (nondeterminism); tag files per run.
    run_tag = os.environ.get("BEHAVIOR_EVAL_RUN", "")
    suffix = f".run{run_tag}" if run_tag else ""
    # Sweep cells only need metric JSON; final eval archives transcripts. METRICS_ONLY
    # keeps the swept evidence dir small (transcripts are ~50-100KB each).
    if _MODE == "live" and not os.environ.get("BEHAVIOR_EVAL_METRICS_ONLY"):
        (out_dir / f"{scenario.name}{suffix}.transcript.json").write_text(
            json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    (out_dir / f"{scenario.name}{suffix}.metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if _MODE != "stub":
        return  # live runs record, they do not assert scripted expectations

    assert recorder.summary["final_status"] == scenario.expect["final_status"]
    assert len(artifacts) >= scenario.expect["min_artifacts"]
    metrics = report["metrics"]
    assert metrics["turns_to_first_draft"] is not None
    assert metrics["out_of_phase_tool_calls"] == 0
    rubric = report["deterministic_rubric"]
    assert rubric["violations_total"] == 0
    assert rubric["heading_completeness_mean"] == 1.0


@pytest.mark.asyncio
async def test_ambiguous_scenario_records_failed_first_critique(client, scenario_env, scenario_project):
    if _MODE != "stub":
        pytest.skip("scripted-expectation test is stub-mode only")
    from tests.eval.behavior_scenarios import reject_critique_redraft

    headers, project = scenario_project
    scenario = reject_critique_redraft()
    driver = ScenarioDriver(client, scenario_env, headers, uuid.UUID(project["id"]), scenario)
    recorder = await driver.run()

    metrics = extract_behavior_metrics(recorder.to_dict())
    assert metrics["first_critique_pass"] is False
    assert metrics["critique_rounds_used"] >= 1
    assert metrics["questions_asked"] >= 3


# ---------------------------------------------------------------------------
# Metric extraction is pure over a transcript object — no scenario run needed.
# ---------------------------------------------------------------------------


def _fake_transcript() -> dict:
    return {
        "scenario": "fake",
        "summary": {"final_status": "completed"},
        "steps": [
            {
                "step": 1,
                "action": {"type": "send", "content": "hi"},
                "snapshot": {
                    "session": {"status": "waiting_for_human", "interrupt_type": "ask_human"},
                    "messages": [{"payload": {"kind": "question"}}],
                    "tool_calls": [],
                    "state": {"diagnosis_signal": {"risk_level": "low", "escalation": "not_needed"}},
                },
            },
            {
                "step": 2,
                "action": {"type": "send", "content": "answer"},
                "snapshot": {
                    "session": {"status": "waiting_for_human", "interrupt_type": "propose_artifacts"},
                    "messages": [{"payload": {"kind": "question"}}],
                    "tool_calls": [{"tool_name": "write_draft:abc", "status": "proposed"}],
                    "state": {
                        "quality_report": {"quality_gate_result": "fail"},
                        "critique_rounds": 1,
                        "section_coverage": {"vision_objectives": "filled", "problem_statement": "missing"},
                        "diagnosis_signal": {"risk_level": "high", "escalation": "escalated"},
                    },
                },
            },
        ],
        "eval": [],
    }


def test_metric_extraction_is_pure_over_transcript():
    metrics = extract_behavior_metrics(_fake_transcript())
    assert metrics["turns_to_first_draft"] == 2
    assert metrics["questions_asked"] == 1
    assert metrics["questions_per_covered_section"] == 1.0
    assert metrics["first_critique_pass"] is False
    assert metrics["critique_rounds_used"] == 1
    assert metrics["out_of_phase_tool_calls"] == 0
    assert metrics["diagnosis_trail"] == [
        {"risk_level": "low", "escalation": "not_needed"},
        {"risk_level": "high", "escalation": "escalated"},
    ]


def test_deterministic_rubric_scores_headings_and_weasels():
    from app.eval.behavior_metrics import deterministic_rubric

    rubric = deterministic_rubric(
        [
            {
                "artifact_type": "vision_objectives",
                "title": "Vision",
                "body": "## Vision\nA fast tool.\n\n## Objectives\n- x\n\n## Success Metrics\n- y",
            }
        ]
    )
    row = rubric["artifacts"][0]
    assert row["violations"] == 0
    assert row["warnings"] >= 1  # "fast" is a weasel word
    assert row["heading_completeness"] == 1.0
