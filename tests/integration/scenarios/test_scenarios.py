"""Agent behavior scenarios at the API layer.

Each scenario runs a conversation through the real HTTP API, asserts the
user-facing contract, records a JSON transcript, and evaluates the artifact.
By default it uses a mock judge so golden transcripts stay deterministic; the
real judge is enabled only through an explicit environment variable.

"""

import os
import uuid

import pytest

from tests.integration.scenarios.driver import ScenarioDriver
from tests.integration.scenarios.eval_support import mock_judge, score_artifacts
from tests.integration.scenarios.library import ALL_SCENARIOS

_REAL_JUDGE_ENV = "SCENARIO_USE_REAL_JUDGE"


def _judge_client():
    if os.getenv(_REAL_JUDGE_ENV) != "1":
        return mock_judge()

    from tests.eval.config import JudgeSettings
    judge_settings = JudgeSettings()
    if not judge_settings.judge_api_key:
        pytest.skip(f"{_REAL_JUDGE_ENV}=1 nhưng thiếu JUDGE_API_KEY")
    from app.models.llm_provider import ProviderType
    from app.services.llm_clients import LLMClientFactory
    return LLMClientFactory.create(
        provider_type=ProviderType(judge_settings.judge_provider),
        api_key=judge_settings.judge_api_key,
        model=judge_settings.judge_model,
        region=judge_settings.judge_region,
        secret_key=judge_settings.judge_secret_key or None,
    )

pytestmark = pytest.mark.asyncio

# Tool-loop surfaces proposals as proposed tool calls, not chat messages; the only agent chat
# messages are ask_user questions and respond assessments (greetings are now plain ask_user turns).
_AGENT_PAYLOAD_KINDS = {"question", "assessment"}


@pytest.mark.parametrize("factory", ALL_SCENARIOS, ids=lambda f: f().name)
async def test_behavior_scenario(factory, client, scenario_env, scenario_project):
    headers, project = scenario_project
    scenario = factory()
    project_id = uuid.UUID(project["id"])
    driver = ScenarioDriver(client, scenario_env, headers, project_id, scenario)

    recorder = await driver.run()
    # Persist the transcript up-front so it survives even when an assertion fails.
    recorder.write()

    # --- API contract assertions ---
    assert recorder.summary["final_status"] == scenario.expect["final_status"], (
        f"{scenario.name}: expected {scenario.expect['final_status']}, got {recorder.summary['final_status']}"
    )

    artifacts = await driver.executed_artifacts()
    assert len(artifacts) >= scenario.expect["min_artifacts"], (
        f"{scenario.name}: expected >= {scenario.expect['min_artifacts']} artifacts, got {len(artifacts)}"
    )

    # Every agent chat message carries a typed payload envelope.
    final_snapshot = recorder.steps[-1]["snapshot"]
    agent_msgs = [m for m in final_snapshot["messages"] if m["role"] == "agent"]
    for m in agent_msgs:
        payload = m.get("payload") or {}
        assert payload.get("kind") in _AGENT_PAYLOAD_KINDS, (
            f"{scenario.name}: unexpected agent payload kind {payload.get('kind')!r}"
        )
    # A scenario that produces artifacts must surface a proposed draft as a tool call.
    if scenario.expect["min_artifacts"] > 0:
        assert final_snapshot["tool_calls"], f"{scenario.name}: expected a proposed write_draft tool call"

    # --- Eval: score produced artifacts and record into the transcript ---
    scored = await score_artifacts(artifacts, _judge_client())
    for s in scored:
        recorder.record_eval(
            artifact_type=s["artifact_type"], title=s["title"], body=s["body"], score=s["score"]
        )
    overalls = [s["score"]["overall"] for s in scored]
    recorder.set_summary(
        artifacts_produced=len(artifacts),
        mean_overall=(sum(overalls) / len(overalls)) if overalls else None,
    )

    path = recorder.write()
    assert path.exists() and path.stat().st_size > 0
