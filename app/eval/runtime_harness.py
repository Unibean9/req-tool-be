from __future__ import annotations

import argparse
import json
import sys
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.graphs.state import build_initial_workflow_state
from app.schemas.artifact_synthesis import ArtifactSynthesisMetadata, evaluate_candidate_readiness


@dataclass(frozen=True)
class RuntimeGateResult:
    gate: str
    passed: bool
    score: float
    threshold: float
    critical: bool
    reason: str


def run_runtime_harness_eval() -> dict[str, Any]:
    gates = [
        _harness_tool_gate_eval(),
        _artifact_readiness_eval(),
        _stale_state_eval(),
        _feedback_revision_eval(),
        _structured_output_eval(),
        _runtime_recovery_eval(),
        _turn_budget_eval(),
    ]
    return _report_from_gates(gates)


def _harness_tool_gate_eval() -> RuntimeGateResult:
    # C1 moved finalize safety from menu coercion into _finalize_impl itself; assert the invariant
    # there: a finalize while the quality gate is unmet returns a recoverable error, not a dispatch.
    import asyncio

    from app.graphs.agent_tools import _finalize_impl

    state = build_initial_workflow_state(
        artifact_type="vision_objectives",
        workflow_area="analysis",
        step_key=None,
    )
    command = asyncio.run(_finalize_impl("Xong", state, {"configurable": {}}, "call_finalize"))
    errors = (getattr(command, "update", None) or {}).get("tool_errors") or []
    passed = bool(errors) and errors[0].get("code") == "finalize_gate_blocked"
    return RuntimeGateResult(
        gate="harness_tool_gate_eval",
        passed=passed,
        score=1.0 if passed else 0.0,
        threshold=1.0,
        critical=True,
        reason="unsafe finalize is blocked by tool-error" if passed else "unsafe finalize can still dispatch",
    )


def _artifact_readiness_eval() -> RuntimeGateResult:
    metadata = ArtifactSynthesisMetadata(
        artifact_type="vision_objectives",
        focused_artifact_id=uuid.uuid4(),
        base_version_id=None,
        evidence_refs=[],
        inference_level="medium",
        confirmed_assumptions=[],
        pending_assumptions=["Target retention needs confirmation"],
    )
    readiness = evaluate_candidate_readiness(
        artifact_type="vision_objectives",
        body="## Vision\nIncrease retention.",
        synthesis_metadata=metadata,
    )
    passed = not readiness.can_persist
    return RuntimeGateResult(
        gate="artifact_readiness_eval",
        passed=passed,
        score=1.0 if passed else 0.0,
        threshold=1.0,
        critical=True,
        reason="candidate with missing contract is blocked"
        if passed
        else "candidate with missing contract can still persist",
    )


def _stale_state_eval() -> RuntimeGateResult:
    from app.services.agent_service import _stale_base_version_detail

    base_version_id = uuid.uuid4()
    current_version_id = uuid.uuid4()
    detail = _stale_base_version_detail(
        snapshot={
            "base_version_id": str(base_version_id),
            "synthesis_metadata": {"base_version_id": str(base_version_id)},
        },
        requested_base_version_id=None,
        current_version_id=current_version_id,
    )
    passed = detail is not None and detail["base_version_id"] == str(base_version_id)
    return RuntimeGateResult(
        gate="stale_state_eval",
        passed=passed,
        score=1.0 if passed else 0.0,
        threshold=1.0,
        critical=True,
        reason="service stale guard returns structured 409 detail"
        if passed
        else "service stale guard does not detect stale state",
    )


def _feedback_revision_eval() -> RuntimeGateResult:
    from app.graphs.nodes import _build_tool_selection_prompt

    state = build_initial_workflow_state(
        artifact_type="vision_objectives",
        workflow_area="analysis",
        step_key=None,
    )
    state["quality_report"] = {
        "mode": "critique",
        "score": 0.4,
        "findings": ["Metric is not measurable yet"],
        "suggestions": ["Add baseline and target"],
        "blocking_issues": ["Metric is not measurable yet"],
        "non_blocking_warnings": [],
        "revision_plan": ["Add baseline and target"],
        "quality_gate_result": "fail",
        "recommended_next_action": "revise",
    }
    prompt = _build_tool_selection_prompt(state, [])
    passed = all(
        fragment in prompt for fragment in ("Metric is not measurable yet", "Add baseline and target", "revise")
    )
    return RuntimeGateResult(
        gate="feedback_revision_eval",
        passed=passed,
        score=1.0 if passed else 0.0,
        threshold=0.8,
        critical=False,
        reason="feedback failure appears in observation"
        if passed
        else "feedback failure does not enter the next prompt",
    )


def _structured_output_eval() -> RuntimeGateResult:
    from app.services.llm_clients import _parse_generate_text

    response_format = {
        "name": "tool_selection",
        "schema": {
            "type": "object",
            "properties": {
                "tool": {"type": "string", "enum": ["ask_user", "write_draft"]},
                "message": {"type": "string"},
            },
            "required": ["tool", "message"],
            "additionalProperties": False,
        },
    }
    invalid_payloads = [
        {"message": "missing tool"},
        {"tool": "finalize", "message": "wrong enum"},
        {"tool": "ask_user", "message": "ok", "extra": "extra"},
    ]
    rejected = 0
    for payload in invalid_payloads:
        try:
            _parse_generate_text(json.dumps(payload, ensure_ascii=False), response_format)
        except ValueError:
            rejected += 1
    passed = rejected == len(invalid_payloads)
    return RuntimeGateResult(
        gate="structured_output_eval",
        passed=passed,
        score=rejected / len(invalid_payloads),
        threshold=1.0,
        critical=True,
        reason="invalid structured output is rejected" if passed else "invalid structured output still passes through",
    )


def _runtime_recovery_eval() -> RuntimeGateResult:
    from app.graphs.agent_tools import RecoverableToolError, _recoverable_tool_update

    command = _recoverable_tool_update(
        RecoverableToolError(
            code="missing_focused_artifact",
            message="Missing focused_artifact_id",
            user_fixable=True,
        ),
        "call_1",
    )
    error = command.update["tool_errors"][0]
    passed = error["classification"] == "recoverable" and command.update["messages"]
    return RuntimeGateResult(
        gate="runtime_recovery_eval",
        passed=passed,
        score=1.0 if passed else 0.0,
        threshold=0.9,
        critical=False,
        reason="recoverable tool error returns to observation"
        if passed
        else "recoverable tool error still fails the turn",
    )


def _turn_budget_eval() -> RuntimeGateResult:
    # Measured from a recorded behavior-scenario transcript (see tests/eval/test_behavior_scenarios.py;
    # the fixture is the committed brd-happy-path stub run). Refresh the fixture when scenario flows
    # change.
    from app.eval.behavior_metrics import turns_to_first_draft

    transcript_path = Path(__file__).parent / "fixtures" / "behavior_reference_transcript.json"
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    turn_count = turns_to_first_draft(transcript)
    budget = 8
    passed = turn_count is not None and turn_count <= budget
    return RuntimeGateResult(
        gate="turn_budget_eval",
        passed=passed,
        score=1.0 if passed else 0.0,
        threshold=1.0,
        critical=False,
        reason=f"turns_to_first_draft={turn_count} (scenario {transcript.get('scenario')}), budget={budget}",
    )


def _report_from_gates(gates: list[RuntimeGateResult | dict[str, Any]]) -> dict[str, Any]:
    rows = [asdict(gate) if isinstance(gate, RuntimeGateResult) else dict(gate) for gate in gates]
    normalized: list[dict[str, Any]] = []
    for row in rows:
        score = float(row.get("score") or 0.0)
        threshold = float(row.get("threshold") if row.get("threshold") is not None else 1.0)
        passed = bool(row.get("passed")) and score >= threshold
        normalized.append({**row, "score": score, "threshold": threshold, "passed": passed})
    overall_passed = all(row["passed"] for row in normalized)
    return {
        "passed": overall_passed,
        "gates": normalized,
    }


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Runtime Harness Eval",
        "",
        f"- passed: {str(report['passed']).lower()}",
        "",
        "| Gate | Score | Threshold | Passed | Reason |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for row in report["gates"]:
        lines.append(
            f"| {row['gate']} | {row['score']:.2f} | {row['threshold']:.2f} | "
            f"{str(row['passed']).lower()} | {row.get('reason', '')} |"
        )
    return "\n".join(lines) + "\n"


def _load_input(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return _report_from_gates(data.get("gates") or [])


def main(argv: list[str] | None = None) -> int:
    _configure_cli_streams()
    parser = argparse.ArgumentParser(description="Run deterministic runtime harness eval.")
    parser.add_argument("--input", type=Path, help="Precomputed gate fixture JSON.")
    parser.add_argument("--json-report", type=Path, help="Path to write the JSON report.")
    parser.add_argument("--markdown-report", type=Path, help="Path to write the Markdown report.")
    args = parser.parse_args(argv)

    report = _load_input(args.input) if args.input else run_runtime_harness_eval()
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.json_report:
        args.json_report.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    if args.markdown_report:
        args.markdown_report.write_text(_markdown_report(report), encoding="utf-8")
    return 0 if report["passed"] else 1


def _configure_cli_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding=getattr(stream, "encoding", None) or "utf-8", errors="backslashreplace")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
