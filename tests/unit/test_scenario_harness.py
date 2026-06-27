import asyncio
import json
from unittest.mock import AsyncMock

from tests.integration.scenarios.diff_transcripts import diff_pair
from tests.integration.scenarios.test_scenarios import _judge_client


def test_behavior_scenario_judge_defaults_to_mock(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "real-key-present")
    monkeypatch.delenv("SCENARIO_USE_REAL_JUDGE", raising=False)

    client = _judge_client()

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
