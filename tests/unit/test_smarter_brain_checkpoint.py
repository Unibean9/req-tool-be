import json
import os
import subprocess
import sys
import uuid

import pytest

from app.eval.smarter_brain_checkpoint import (
    CheckpointThresholds,
    collect_token_sessions,
    evaluate_checkpoint,
    main,
)
from app.models.agent import AgentRun, AgentSession


def test_checkpoint_passes_when_token_ratio_and_eval_delta_meet_thresholds():
    result = evaluate_checkpoint(
        baseline_sessions=[
            {"session_id": "b1", "token_usage": [{"total": 100}]},
            {"session_id": "b2", "token_usage": [{"input": 60, "output": 40}]},
        ],
        candidate_sessions=[
            {"session_id": "c1", "token_usage": [{"total": 180}]},
            {"session_id": "c2", "token_usage": [{"input": 90, "output": 95}]},
        ],
        baseline_eval_rows=[{"overall": 0.70}, {"overall": 0.72}],
        candidate_eval_rows=[{"overall": 0.77}, {"overall": 0.78}],
    )

    assert result.token_ratio <= 2.0
    assert result.eval_delta >= 0.05
    assert result.passed is True


def test_checkpoint_fails_when_token_ratio_exceeds_limit():
    result = evaluate_checkpoint(
        baseline_sessions=[{"session_id": "b1", "token_usage": [{"total": 100}]}],
        candidate_sessions=[{"session_id": "c1", "token_usage": [{"total": 250}]}],
        baseline_eval_rows=[{"overall": 0.70}],
        candidate_eval_rows=[{"overall": 0.80}],
    )

    assert result.token_passed is False
    assert result.eval_passed is True
    assert result.passed is False


def test_checkpoint_fails_when_eval_delta_is_too_small():
    result = evaluate_checkpoint(
        baseline_sessions=[{"session_id": "b1", "token_usage": [{"total": 100}]}],
        candidate_sessions=[{"session_id": "c1", "token_usage": [{"total": 150}]}],
        baseline_eval_rows=[{"overall": 0.70}],
        candidate_eval_rows=[{"overall": 0.74}],
        thresholds=CheckpointThresholds(min_eval_delta=0.05),
    )

    assert result.token_passed is True
    assert result.eval_passed is False
    assert result.passed is False


def test_checkpoint_rejects_missing_token_usage():
    with pytest.raises(ValueError, match="Thieu session token_usage"):
        evaluate_checkpoint(
            baseline_sessions=[],
            candidate_sessions=[{"session_id": "c1", "token_usage": [{"total": 100}]}],
            baseline_eval_rows=[{"overall": 0.70}],
            candidate_eval_rows=[{"overall": 0.80}],
        )


@pytest.mark.asyncio
async def test_collect_token_sessions_reads_agent_run_totals(db_session):
    session = AgentSession(
        project_id=uuid.uuid4(),
        artifact_type="goal",
        workflow_area="analysis",
        graph_checkpoint={},
    )
    db_session.add(session)
    await db_session.flush()
    db_session.add_all(
        [
            AgentRun(session_id=session.id, analysis_result={}, token_usage={"total": 100}),
            AgentRun(session_id=session.id, analysis_result={}, token_usage={"input": 40, "output": 70}),
            AgentRun(session_id=session.id, analysis_result={}, token_usage=None),
        ]
    )
    await db_session.flush()

    sessions = await collect_token_sessions(db_session, [session.id])

    assert sessions == [{"session_id": str(session.id), "token_usage": [{"total": 100}, {"input": 40, "output": 70}]}]


def test_checkpoint_averages_token_usage_per_session_not_per_run():
    result = evaluate_checkpoint(
        baseline_sessions=[
            {"session_id": "b1", "token_usage": [{"total": 100}]},
            {"session_id": "b2", "token_usage": [{"total": 100}]},
        ],
        candidate_sessions=[
            {"session_id": "c1", "token_usage": [{"total": 100}]},
            {"session_id": "c2", "token_usage": [{"total": 100}, {"total": 300}]},
        ],
        baseline_eval_rows=[{"overall": 0.70}],
        candidate_eval_rows=[{"overall": 0.80}],
    )

    assert result.candidate_token_avg == 250
    assert result.token_ratio == 2.5
    assert result.passed is False


def test_checkpoint_counts_all_llm_calls_reported_for_a_session():
    result = evaluate_checkpoint(
        baseline_sessions=[{"session_id": "b1", "token_usage": [{"source": "analyze", "total": 100}]}],
        candidate_sessions=[
            {
                "session_id": "c1",
                "token_usage": [
                    {"source": "analyze", "total": 100},
                    {"source": "summarize", "total": 50},
                    {"source": "quality_gate", "input": 20, "output": 30},
                ],
            }
        ],
        baseline_eval_rows=[{"overall": 0.70}],
        candidate_eval_rows=[{"overall": 0.80}],
    )

    assert result.candidate_token_avg == 200
    assert result.token_ratio == 2.0
    assert result.passed is True


def test_checkpoint_cli_returns_nonzero_when_gate_fails(tmp_path, capsys):
    report = {
        "baseline": {"sessions": [{"session_id": "b1", "token_usage": [{"total": 100}]}], "eval": [{"overall": 0.70}]},
        "candidate": {"sessions": [{"session_id": "c1", "token_usage": [{"total": 250}]}], "eval": [{"overall": 0.80}]},
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    exit_code = main([str(path)])

    assert exit_code == 1
    assert '"passed": false' in capsys.readouterr().out


def test_checkpoint_cli_returns_2_for_invalid_input_without_encoding_error(tmp_path, capsys):
    path = tmp_path / "empty.json"
    path.write_text("{}", encoding="utf-8")

    exit_code = main([str(path)])

    out = capsys.readouterr().out
    assert exit_code == 2
    assert '"passed": false' in out
    assert "\\u" in out


def test_checkpoint_cli_help_survives_cp1252_encoding():
    env = {**os.environ, "PYTHONIOENCODING": "cp1252"}
    result = subprocess.run(
        [sys.executable, "scripts/checkpoint_smarter_brain.py", "--help"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0
    assert "checkpoint" in result.stdout
