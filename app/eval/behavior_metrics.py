"""Pure metric extraction over a behavior-scenario transcript.

Input is the transcript dict produced by tests.integration.scenarios.recorder.TranscriptRecorder
(`{"scenario", "summary", "steps": [{"action", "snapshot"}], "eval"}`). Each snapshot may carry a
"state" dict with checkpoint fields (turn_count, critique_rounds, quality_report,
diagnosis_signal, section_coverage) recorded by the scenario driver.

Every function here is pure over that object — no DB, no LLM, no graph — so metrics are
unit-testable without running a scenario and identical between stub and live modes.

`deterministic_rubric` reuses `validate_proposal` and the artifact output contracts READ-ONLY to
score final artifact bodies; it does not gate anything.
"""

from typing import Any

from app.documents.registry import all_item_types, output_contract
from app.graphs.validators import validate_proposal

# Coverage states that count a section as "covered" for the question-efficiency ratio.
_COVERED_STATES = {"filled", "partial", "needs_review"}


def _steps(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    return list(transcript.get("steps") or [])


def _final_snapshot(transcript: dict[str, Any]) -> dict[str, Any]:
    steps = _steps(transcript)
    return steps[-1].get("snapshot") or {} if steps else {}


def _state_of(step: dict[str, Any]) -> dict[str, Any]:
    return (step.get("snapshot") or {}).get("state") or {}


def turns_to_first_draft(transcript: dict[str, Any]) -> int | None:
    """User-visible turns (send actions) up to and including the first write_draft proposal.

    None when the scenario never reached a draft proposal.
    """
    sends = 0
    for step in _steps(transcript):
        if (step.get("action") or {}).get("type") == "send":
            sends += 1
        tool_calls = (step.get("snapshot") or {}).get("tool_calls") or []
        if any(str(tc.get("tool_name") or "").startswith("write_draft") for tc in tool_calls):
            return sends
    return None


def questions_asked(transcript: dict[str, Any]) -> int:
    """Count of agent question messages over the whole session (messages are cumulative)."""
    messages = _final_snapshot(transcript).get("messages") or []
    return sum(1 for m in messages if (m.get("payload") or {}).get("kind") == "question")


def covered_sections(transcript: dict[str, Any]) -> int | None:
    """Sections whose final coverage state counts as covered; None when no coverage data exists."""
    coverage = (_final_snapshot(transcript).get("state") or {}).get("section_coverage")
    if not coverage:
        return None
    return sum(1 for value in coverage.values() if value in _COVERED_STATES)


def critique_outcome(transcript: dict[str, Any]) -> dict[str, Any]:
    """First-critique verdict + rounds used, from quality_report snapshots across steps."""
    first_pass: bool | None = None
    for step in _steps(transcript):
        report = _state_of(step).get("quality_report")
        if report and report.get("quality_gate_result") in ("pass", "fail"):
            first_pass = report["quality_gate_result"] == "pass"
            break
    final_state = _final_snapshot(transcript).get("state") or {}
    return {
        "first_critique_pass": first_pass,
        "critique_rounds_used": final_state.get("critique_rounds") or 0,
    }


def diagnosis_trail(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    """Sequence of distinct diagnosis signals observed (consecutive duplicates collapsed)."""
    trail: list[dict[str, Any]] = []
    for step in _steps(transcript):
        signal = _state_of(step).get("diagnosis_signal")
        if not signal:
            continue
        compact = {"risk_level": signal.get("risk_level"), "escalation": signal.get("escalation")}
        if not trail or trail[-1] != compact:
            trail.append(compact)
    return trail


def out_of_phase_tool_calls(transcript: dict[str, Any]) -> int:
    """Tool calls rejected for being outside the session phase.

    Scaffolded counter: reads the state field of the same name; 0 until the phase state machine
    populates it.
    """
    final_state = _final_snapshot(transcript).get("state") or {}
    return final_state.get("out_of_phase_tool_calls") or 0


def deterministic_rubric(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    """Score final artifact bodies with validate_proposal + required-heading completeness.

    `artifacts` is a list of {"artifact_type", "title", "body"} (the executed tool-call
    snapshots). Read-only use of the deterministic validator — nothing is gated here.
    """
    rows: list[dict[str, Any]] = []
    for art in artifacts:
        artifact_type = str(art.get("artifact_type") or "")
        result = validate_proposal(artifact_type, {"title": art.get("title"), "body": art.get("body")})
        row: dict[str, Any] = {
            "artifact_type": artifact_type,
            "violations": len(result.violations),
            "warnings": len(result.warnings),
        }
        try:
            required = output_contract(artifact_type).required_headings if artifact_type in all_item_types() else ()
        except ValueError:
            required = ()
        if required:
            body = str(art.get("body") or "")
            present = sum(1 for heading in required if heading in body)
            row["heading_completeness"] = present / len(required)
        else:
            row["heading_completeness"] = None
        rows.append(row)

    completeness_values = [r["heading_completeness"] for r in rows if r["heading_completeness"] is not None]
    return {
        "artifacts": rows,
        "violations_total": sum(r["violations"] for r in rows),
        "warnings_total": sum(r["warnings"] for r in rows),
        "heading_completeness_mean": (
            sum(completeness_values) / len(completeness_values) if completeness_values else None
        ),
    }


def extract_behavior_metrics(transcript: dict[str, Any]) -> dict[str, Any]:
    """All transcript-level behavior metrics as one flat dict."""
    questions = questions_asked(transcript)
    covered = covered_sections(transcript)
    return {
        "turns_to_first_draft": turns_to_first_draft(transcript),
        "questions_asked": questions,
        "questions_per_covered_section": (questions / covered) if covered else None,
        **critique_outcome(transcript),
        "out_of_phase_tool_calls": out_of_phase_tool_calls(transcript),
        "diagnosis_trail": diagnosis_trail(transcript),
    }


def behavior_report(
    transcript: dict[str, Any],
    artifacts: list[dict[str, Any]],
    *,
    mode: str,
    model: str | None = None,
) -> dict[str, Any]:
    """One comparable JSON report for a completed scenario run.

    `mode` is "stub" or "live". Stub-mode reports only guard harness mechanics; all
    before/after behavior claims must come from live-mode runs.
    """
    return {
        "scenario": transcript.get("scenario"),
        "mode": mode,
        "model": model,
        "final_status": (transcript.get("summary") or {}).get("final_status"),
        "metrics": extract_behavior_metrics(transcript),
        "deterministic_rubric": deterministic_rubric(artifacts),
    }
