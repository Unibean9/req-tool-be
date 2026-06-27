"""Smarter Brain quality checkpoint.

This module does not assume the candidate agent is better. It receives baseline
and candidate measurements, then evaluates two hard gates:

- candidate average token usage <= 2x baseline
- candidate average eval score improves by >= 0.05 over baseline
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent import AgentRun


@dataclass(frozen=True)
class CheckpointThresholds:
    max_token_ratio: float = 2.0
    min_eval_delta: float = 0.05


@dataclass(frozen=True)
class CheckpointResult:
    baseline_token_avg: float
    candidate_token_avg: float
    token_ratio: float
    token_passed: bool
    baseline_eval_avg: float
    candidate_eval_avg: float
    eval_delta: float
    eval_passed: bool
    passed: bool


def token_total(usage: dict[str, Any]) -> int:
    if not isinstance(usage, dict):
        raise ValueError("token_usage must be an object")
    if usage.get("total") is not None:
        total = int(usage["total"])
    elif usage.get("input") is not None or usage.get("output") is not None:
        total = int(usage.get("input") or 0) + int(usage.get("output") or 0)
    else:
        raise ValueError("token_usage is missing total or input/output")
    if total <= 0:
        raise ValueError("token_usage total must be > 0")
    return total


def session_token_total(session: dict[str, Any]) -> int:
    usages = session.get("token_usage") or session.get("runs") or []
    if not usages:
        raise ValueError("session checkpoint is missing token_usage")
    return sum(token_total(usage) for usage in usages)


def average_session_token_total(sessions: Sequence[dict[str, Any]]) -> float:
    if not sessions:
        raise ValueError("Missing session token_usage for checkpoint calculation")
    totals = [session_token_total(session) for session in sessions]
    return sum(totals) / len(totals)


def average_overall(eval_rows: Sequence[dict[str, Any]]) -> float:
    if not eval_rows:
        raise ValueError("Missing eval rows for checkpoint calculation")
    values: list[float] = []
    for row in eval_rows:
        if "overall" not in row:
            raise ValueError("eval row is missing overall")
        values.append(float(row["overall"]))
    return sum(values) / len(values)


def evaluate_checkpoint(
    *,
    baseline_sessions: Sequence[dict[str, Any]],
    candidate_sessions: Sequence[dict[str, Any]],
    baseline_eval_rows: Sequence[dict[str, Any]],
    candidate_eval_rows: Sequence[dict[str, Any]],
    thresholds: CheckpointThresholds | None = None,
) -> CheckpointResult:
    thresholds = thresholds or CheckpointThresholds()

    baseline_token_avg = average_session_token_total(baseline_sessions)
    candidate_token_avg = average_session_token_total(candidate_sessions)
    token_ratio = candidate_token_avg / baseline_token_avg
    token_passed = token_ratio <= thresholds.max_token_ratio

    baseline_eval_avg = average_overall(baseline_eval_rows)
    candidate_eval_avg = average_overall(candidate_eval_rows)
    eval_delta = candidate_eval_avg - baseline_eval_avg
    eval_passed = eval_delta >= thresholds.min_eval_delta

    return CheckpointResult(
        baseline_token_avg=baseline_token_avg,
        candidate_token_avg=candidate_token_avg,
        token_ratio=token_ratio,
        token_passed=token_passed,
        baseline_eval_avg=baseline_eval_avg,
        candidate_eval_avg=candidate_eval_avg,
        eval_delta=eval_delta,
        eval_passed=eval_passed,
        passed=token_passed and eval_passed,
    )


async def collect_token_sessions(db: AsyncSession, session_ids: Sequence[uuid.UUID]) -> list[dict[str, Any]]:
    if not session_ids:
        raise ValueError("Missing session_ids for querying AgentRun.token_usage")
    rows = (
        await db.execute(
            select(AgentRun.session_id, AgentRun.token_usage)
            .where(AgentRun.session_id.in_(session_ids))
            .where(AgentRun.token_usage.is_not(None))
        )
    ).all()
    by_session: dict[str, list[dict[str, Any]]] = {str(session_id): [] for session_id in session_ids}
    for session_id, usage in rows:
        if usage is not None:
            by_session[str(session_id)].append(dict(usage))
    return [{"session_id": session_id, "token_usage": usages} for session_id, usages in by_session.items() if usages]


def _load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _thresholds_from_report(report: dict[str, Any]) -> CheckpointThresholds:
    raw = report.get("thresholds") or {}
    return CheckpointThresholds(
        max_token_ratio=float(raw.get("max_token_ratio", CheckpointThresholds.max_token_ratio)),
        min_eval_delta=float(raw.get("min_eval_delta", CheckpointThresholds.min_eval_delta)),
    )


def evaluate_report(report: dict[str, Any]) -> CheckpointResult:
    baseline = report.get("baseline") or {}
    candidate = report.get("candidate") or {}
    return evaluate_checkpoint(
        baseline_sessions=_sessions_from_report_side(baseline),
        candidate_sessions=_sessions_from_report_side(candidate),
        baseline_eval_rows=baseline.get("eval") or [],
        candidate_eval_rows=candidate.get("eval") or [],
        thresholds=_thresholds_from_report(report),
    )


def _sessions_from_report_side(side: dict[str, Any]) -> list[dict[str, Any]]:
    sessions = side.get("sessions")
    if sessions is not None:
        return list(sessions)
    legacy_usage = side.get("token_usage") or []
    return [{"session_id": f"legacy-{index}", "token_usage": [usage]} for index, usage in enumerate(legacy_usage, 1)]


def _configure_cli_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding=getattr(stream, "encoding", None) or "utf-8", errors="backslashreplace")


def main(argv: Sequence[str] | None = None) -> int:
    _configure_cli_streams()
    parser = argparse.ArgumentParser(description="Measure Smarter Brain checkpoint from a JSON report.")
    parser.add_argument("report", type=Path, help="Path to JSON with baseline/candidate token_usage and eval rows.")
    args = parser.parse_args(argv)

    try:
        result = evaluate_report(_load_report(args.report))
    except Exception as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, ensure_ascii=True, indent=2))
        return 2

    print(json.dumps(asdict(result), ensure_ascii=True, indent=2))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
