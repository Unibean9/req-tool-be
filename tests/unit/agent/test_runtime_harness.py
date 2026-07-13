import json
import os
import subprocess
import sys

from app.eval import runtime_harness


def test_runtime_harness_report_contains_all_gates_and_thresholds(tmp_path):
    json_report = tmp_path / "runtime-harness.json"
    markdown_report = tmp_path / "runtime-harness.md"

    exit_code = runtime_harness.main(
        [
            "--json-report",
            str(json_report),
            "--markdown-report",
            str(markdown_report),
        ]
    )

    data = json.loads(json_report.read_text(encoding="utf-8"))
    gates = {row["gate"]: row for row in data["gates"]}

    assert exit_code == 0
    assert set(gates) == {
        "harness_tool_gate_eval",
        "artifact_readiness_eval",
        "stale_state_eval",
        "feedback_revision_eval",
        "structured_output_eval",
        "runtime_recovery_eval",
        "turn_budget_eval",
    }
    assert gates["feedback_revision_eval"]["threshold"] == 0.8
    assert gates["runtime_recovery_eval"]["threshold"] == 0.9
    assert all(row["passed"] for row in gates.values())
    assert "runtime_recovery_eval" in markdown_report.read_text(encoding="utf-8")


def test_runtime_harness_cli_returns_nonzero_for_critical_failure(tmp_path):
    fixture = tmp_path / "forced-fail.json"
    fixture.write_text(
        json.dumps(
            {
                "gates": [
                    {
                        "gate": "structured_output_eval",
                        "passed": False,
                        "score": 0.0,
                        "threshold": 1.0,
                        "critical": True,
                        "reason": "invalid schema van dispatch",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    json_report = tmp_path / "runtime-harness.json"

    exit_code = runtime_harness.main(["--input", str(fixture), "--json-report", str(json_report)])

    data = json.loads(json_report.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert data["passed"] is False
    assert data["gates"][0]["reason"] == "invalid schema van dispatch"


def test_runtime_harness_cli_stdout_survives_cp1252_encoding():
    env = {**os.environ, "PYTHONIOENCODING": "cp1252"}
    result = subprocess.run(
        [sys.executable, "-m", "app.eval.runtime_harness"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0
    assert "stale_state_eval" in result.stdout
    assert '"passed": true' in result.stdout
