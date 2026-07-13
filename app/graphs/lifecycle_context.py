"""Lifecycle report rendering and tool-gate helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

STALE_CURATION_ACTIONS = frozenset({"ADD", "UPDATE", "SUPERSEDE", "RETIRE", "NOOP"})
_LIFECYCLE_WRITE_TOOL = "write_draft"


def focused_lifecycle_report(state: dict[str, Any]) -> dict[str, Any] | None:
    reports = [item for item in state.get("lifecycle_reports") or [] if isinstance(item, dict)]
    focused_artifact_id = str(state.get("focused_artifact_id") or "")
    if focused_artifact_id:
        for report in reports:
            if str(report.get("artifact_id") or "") == focused_artifact_id:
                return report
    artifact_type = str(state.get("artifact_type") or "")
    for report in reports:
        if str(report.get("artifact_type") or "") == artifact_type:
            return report
    return None


def has_stale_curation(args: dict[str, Any]) -> bool:
    action = str(args.get("curation_action") or "").strip().upper()
    justification = str(args.get("curation_justification") or "").strip()
    return action in STALE_CURATION_ACTIONS and bool(justification)


def lifecycle_tool_block_reason(
    state: dict[str, Any],
    tool_name: str,
    args: dict[str, Any] | None = None,
) -> str | None:
    if tool_name != _LIFECYCLE_WRITE_TOOL:
        return None
    report = focused_lifecycle_report(state)
    if not report:
        return None
    lifecycle_state = str(report.get("state") or "").lower()
    if lifecycle_state == "current":
        return "current_artifact_reproposal_blocked"
    if lifecycle_state == "blocked":
        return "blocked_predecessor_amend_blocked"
    if lifecycle_state == "orphan":
        return "orphan_artifact_relink_or_retire_required"
    if lifecycle_state == "stale" and not has_stale_curation(args or {}):
        return "stale_artifact_requires_curation_action"
    return None




def lifecycle_blocked_tool_names(state: dict[str, Any], requested: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    blocked: list[dict[str, str]] = []
    report = focused_lifecycle_report(state) or {}
    for item in requested:
        name = str(item.get("name") or "")
        reason = lifecycle_tool_block_reason(state, name, dict(item.get("args") or {}))
        if not reason:
            continue
        blocked.append({"name": name, "reason": reason, "state": str(report.get("state") or "")})
    return blocked


def render_situation_report(reports: Sequence[dict[str, Any]]) -> str:
    if not reports:
        return ""
    lines = []
    for report in reports:
        actions = report.get("allowed_actions") or []
        action_text = ", ".join(str(item) for item in actions) if actions else "none"
        artifact_id = report.get("artifact_id") or "missing"
        lines.append(
            "- "
            f"[{report.get('artifact_type')}] "
            f"id={artifact_id} "
            f"state={str(report.get('state') or '').upper()} "
            f"actions={action_text} "
            f"reason={report.get('reason')}"
        )
    return "\n\nSITUATION REPORT:\n" + "\n".join(lines)


def render_artifact_history(history: Sequence[dict[str, Any]]) -> str:
    if not history:
        return ""
    lines = []
    for item in history:
        lines.append(
            "- "
            f"[{item.get('artifact_type')}] "
            f"v{item.get('version_number')} "
            f"source={item.get('change_source')} "
            f"version_id={item.get('version_id')}"
        )
    return "\n\nRECENT ARTIFACT CHANGES:\n" + "\n".join(lines)
