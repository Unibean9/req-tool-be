import asyncio
import json
from unittest.mock import AsyncMock

from scripts.run_live_artifact_http_flow import create_provider_config
from tests.integration.scenarios.diff_transcripts import diff_pair
from tests.integration.scenarios.eval_support import judge_client


class _ProviderClient:
    def __init__(self):
        self.posts = []

    def post(self, path, payload=None, **kwargs):
        self.posts.append((path, payload, kwargs))
        if path.endswith("/health-check"):
            return _Result({"data": {"config": {"id": "cfg-1", "provider_type": "custom"}}})
        return _Result({"data": {"id": "cfg-1"}})


class _Result:
    def __init__(self, body):
        self.data = body


def test_live_artifact_flow_env_supports_custom_provider_base_url(tmp_path, monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER_TYPE", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL_NAME", raising=False)
    env_path = tmp_path / ".env.test"
    env_path.write_text(
        "\n".join(
            [
                "LLM_PROVIDER_TYPE=custom",
                "LLM_API_KEY=test-key",
                "LLM_BASE_URL=https://custom.example/v1",
                "LLM_MODEL_NAME=custom-model",
            ]
        ),
        encoding="utf-8",
    )
    client = _ProviderClient()

    provider = create_provider_config(client, env_path)

    assert provider["provider_type"] == "custom"
    create_path, payload, kwargs = client.posts[0]
    assert create_path.endswith("/llm-provider-configs")
    assert kwargs["timeout"] == 180
    assert payload["provider_type"] == "custom"
    assert payload["api_key"] == "test-key"
    assert payload["base_url"] == "https://custom.example/v1"
    assert payload["model_name"] == "custom-model"


def test_behavior_scenario_judge_defaults_to_mock(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "real-key-present")
    monkeypatch.delenv("SCENARIO_USE_REAL_JUDGE", raising=False)

    client = judge_client()

    assert isinstance(client.generate, AsyncMock)
    result, _ = asyncio.run(client.generate())
    assert result["overall"] == 0.82


def test_transcript_diff_ignores_runtime_noise_and_eval_score(tmp_path):
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    baseline.write_text(
        json.dumps(
            {
                "scenario": "demo",
                "summary": {"session_id": "11111111-1111-1111-1111-111111111111", "mean_overall": 0.7},
                "steps": [
                    {
                        "snapshot": {
                            "messages": [
                                {
                                    "id": "22222222-2222-2222-2222-222222222222",
                                    "created_at": "2026-06-25T10:00:00",
                                    "content": "Giu nguyen content",
                                }
                            ],
                            "tool_calls": [
                                {
                                    "tool_name": "write_draft:33333333-3333-3333-3333-333333333333",
                                    "input_snapshot": {"title": "Draft"},
                                }
                            ],
                        }
                    }
                ],
                "eval": [{"score": {"overall": 0.7, "rationale": "judge that"}}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    current.write_text(
        json.dumps(
            {
                "scenario": "demo",
                "summary": {"session_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "mean_overall": 0.82},
                "steps": [
                    {
                        "snapshot": {
                            "messages": [
                                {
                                    "id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                                    "created_at": "2026-06-26T10:00:00",
                                    "content": "Giu nguyen content",
                                }
                            ],
                            "tool_calls": [
                                {
                                    "tool_name": "write_draft:cccccccc-cccc-cccc-cccc-cccccccccccc",
                                    "input_snapshot": {"title": "Draft"},
                                }
                            ],
                        }
                    }
                ],
                "eval": [{"score": {"overall": 0.82, "rationale": "mock judge"}}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    assert diff_pair(baseline, current) == []


def test_transcript_diff_catches_behavior_content_drift(tmp_path):
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    baseline.write_text(json.dumps({"steps": [{"snapshot": {"messages": [{"content": "A"}]}}]}), encoding="utf-8")
    current.write_text(json.dumps({"steps": [{"snapshot": {"messages": [{"content": "B"}]}}]}), encoding="utf-8")

    assert diff_pair(baseline, current)
