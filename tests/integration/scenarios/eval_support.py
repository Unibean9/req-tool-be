"""Shared eval helpers for scenario tests.

Default mode uses a mock judge (deterministic, no API key). The integration
variant reuses the real LLM judge from tests/eval when LLM_API_KEY is set.
"""

import os
from typing import Any
from unittest.mock import AsyncMock

import pytest

from tests.eval import rubric
from tests.eval.judge import judge_artifact

_REAL_JUDGE_ENV = "SCENARIO_USE_REAL_JUDGE"


def judge_client():
    """The judge LLM client for scenario scoring: a deterministic mock by default,
    the real judge only when SCENARIO_USE_REAL_JUDGE=1 and credentials are present.
    """
    if os.getenv(_REAL_JUDGE_ENV) != "1":
        return mock_judge()

    from tests.eval.config import JudgeSettings
    judge_settings = JudgeSettings()
    if not judge_settings.judge_api_key:
        pytest.skip(f"{_REAL_JUDGE_ENV}=1 nhung missing LLM_API_KEY")
    from app.models.llm_provider import ProviderType
    from app.services.llm_clients import LLMClientFactory
    return LLMClientFactory.create(
        provider_type=ProviderType(judge_settings.judge_provider),
        api_key=judge_settings.judge_api_key,
        model=judge_settings.judge_model,
        region=judge_settings.judge_region,
        secret_key=judge_settings.judge_secret_key or None,
    )


def mock_judge(overall: float = 0.82) -> AsyncMock:
    """An LLM stub shaped like the real judge response (see JUDGE_SCHEMA)."""
    client = AsyncMock()
    client.generate = AsyncMock(
        return_value=(
            {
                "scores": {
                    "unambiguous": 0.85,
                    "verifiable": 0.8,
                    "complete": 0.85,
                    "consistent": 0.85,
                    "traceable": 0.75,
                    "feasible": 0.8,
                    "invest": None,
                    "smart": None,
                },
                "overall": overall,
                "rationale": "Artifact is clear, co pham vi va tac dong cu the.",
            },
            None,
        )
    )
    return client


async def score_artifacts(artifacts: list[dict[str, Any]], judge_client) -> list[dict[str, Any]]:
    """Run the judge over a list of {artifact_type, title, body} artifacts."""
    scored = []
    for art in artifacts:
        text = f"{art['title']}\n\n{art['body']}".strip()
        score = await judge_artifact(text, rubric.RUBRIC_CRITERIA, judge_client)
        scored.append({**art, "score": score})
    return scored
