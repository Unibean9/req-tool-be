"""Eval: Intent Phase Gate (D6).

Proves each behavior from the intent-phase-gate plan:
confirm_intent schema registration, interrupt kind, user_confirmed update.
confirm_intent is solo-enforced while note tools may ride along.
full gate → confirm → draft menu open (schema-level proof).
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch


@dataclass(frozen=True)
class GateResult:
    gate: str
    passed: bool
    score: float
    threshold: float
    critical: bool
    reason: str


# ---------------------------------------------------------------------------
# confirm_intent tool — schema registration, arg keys, interrupt kind
# ---------------------------------------------------------------------------


def _confirm_intent_in_schema_enum() -> GateResult:
    # Native tool calling: confirm_intent is registered iff it is bound to the intent-phase menu.
    from app.graphs.agent_tools import get_available_tools

    names = [t.name for t in get_available_tools({"user_confirmed": None, "messages": []})]
    passed = "confirm_intent" in names
    return GateResult(
        gate="confirm_intent bound to the intent-phase tool menu",
        passed=passed,
        score=1.0 if passed else 0.0,
        threshold=1.0,
        critical=True,
        reason="confirm_intent in intent menu" if passed else f"menu: {names}",
    )


def _confirm_intent_arg_keys() -> GateResult:
    from app.graphs.agent_tools import confirm_intent
    from app.graphs.nodes import _TOOL_REQUIRED_ARGS, _build_tool_schemas

    params = _build_tool_schemas([confirm_intent])[0]["parameters"]
    arg_ok = list(params.get("properties", {}).keys()) == ["summary"]
    req_ok = _TOOL_REQUIRED_ARGS.get("confirm_intent") == ["summary"]
    passed = arg_ok and req_ok
    reason = (
        f"params={list(params.get('properties', {}).keys())}, "
        f"_TOOL_REQUIRED_ARGS={_TOOL_REQUIRED_ARGS.get('confirm_intent')}"
    )
    return GateResult(
        gate="confirm_intent native arg = ['summary']; summary is required",
        passed=passed,
        score=1.0 if passed else 0.0,
        threshold=1.0,
        critical=True,
        reason=reason,
    )


def _confirm_intent_is_interrupt_bearing() -> GateResult:
    from app.graphs.nodes import _INTERRUPT_BEARING_TOOLS

    passed = "confirm_intent" in _INTERRUPT_BEARING_TOOLS
    return GateResult(
        gate="confirm_intent is in _INTERRUPT_BEARING_TOOLS",
        passed=passed,
        score=1.0 if passed else 0.0,
        threshold=1.0,
        critical=True,
        reason="confirm_intent interrupt-bearing" if passed else f"set: {_INTERRUPT_BEARING_TOOLS}",
    )


def _confirm_intent_sets_user_confirmed_stream_response() -> GateResult:
    """_confirm_intent_impl must set user_confirmed=True and use STREAM_RESPONSE interrupt."""

    async def run() -> tuple[bool, str]:
        from app.graphs.agent_tools import _confirm_intent_impl

        call_kwargs: dict[str, Any] = {}

        async def _fake_save(
            _state,
            _config,
            _content,
            *,
            run_id,
            kind="question",
            mode=None,
            interrupt_kind="ask_human",
        ):
            _ = (run_id, mode)
            call_kwargs["interrupt_kind"] = interrupt_kind
            call_kwargs["kind"] = kind
            return "ok"

        state = {"messages": [], "user_confirmed": None}
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        with patch("app.graphs.interrupts._save_and_interrupt_ask", new=AsyncMock(side_effect=_fake_save)):
            command = await _confirm_intent_impl("Build Y for audience A", state, config, "tc-001")

        ok_confirmed = command.update.get("user_confirmed") is True
        ok_kind = call_kwargs.get("interrupt_kind") == "stream_response"
        ok_assessment = call_kwargs.get("kind") == "assessment"
        if ok_confirmed and ok_kind and ok_assessment:
            return True, "user_confirmed=True, interrupt_kind=stream_response, kind=assessment"
        return False, (
            f"user_confirmed={command.update.get('user_confirmed')!r}, "
            f"interrupt_kind={call_kwargs.get('interrupt_kind')!r}, "
            f"kind={call_kwargs.get('kind')!r}"
        )

    passed, reason = asyncio.run(run())
    return GateResult(
        gate="confirm_intent sets user_confirmed=True with STREAM_RESPONSE/assessment interrupt",
        passed=passed,
        score=1.0 if passed else 0.0,
        threshold=1.0,
        critical=True,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Intent phase gate — get_available_tools hides artifact tools before confirm_intent
# ---------------------------------------------------------------------------


def _names(state: dict) -> set[str]:
    from app.graphs.agent_tools import get_available_tools

    return {t.name for t in get_available_tools(state)}


def _gate_hides_artifact_tools_before_confirm() -> GateResult:
    names = _names({"messages": [], "user_confirmed": None})
    hidden = {"write_draft", "finalize", "run_critique"}
    missing = hidden - names
    passed = missing == hidden
    return GateResult(
        gate="write_draft/finalize/run_critique hidden when user_confirmed is None",
        passed=passed,
        score=1.0 if passed else 0.0,
        threshold=1.0,
        critical=True,
        reason="hidden tools verified" if passed else f"unexpected tools still available: {hidden & names}",
    )


def _gate_offers_confirm_intent() -> GateResult:
    names = _names({"messages": [], "user_confirmed": None})
    passed = "confirm_intent" in names
    return GateResult(
        gate="confirm_intent available in intent phase (user_confirmed=None)",
        passed=passed,
        score=1.0 if passed else 0.0,
        threshold=1.0,
        critical=True,
        reason="confirm_intent in menu" if passed else f"available tools: {names}",
    )


def _gate_unlocks_after_confirm() -> GateResult:
    names_after = _names({"messages": [], "user_confirmed": True})
    passed = "write_draft" in names_after and "confirm_intent" not in names_after
    return GateResult(
        gate="after user_confirmed=True — write_draft offered, confirm_intent retired",
        passed=passed,
        score=1.0 if passed else 0.0,
        threshold=1.0,
        critical=True,
        reason=(f"write_draft={'write_draft' in names_after}, confirm_intent={'confirm_intent' in names_after}"),
    )


def _gate_confirm_intent_one_shot() -> GateResult:
    """confirm_intent must disappear from menu once user_confirmed=True (one-shot gate)."""
    names_before = _names({"messages": [], "user_confirmed": None})
    names_after = _names({"messages": [], "user_confirmed": True})
    passed = "confirm_intent" in names_before and "confirm_intent" not in names_after
    return GateResult(
        gate="confirm_intent one-shot — disappears from menu after user_confirmed=True",
        passed=passed,
        score=1.0 if passed else 0.0,
        threshold=1.0,
        critical=True,
        reason=f"before={{'confirm_intent' in names_before}}, after={'confirm_intent' in names_after}",
    )


# ---------------------------------------------------------------------------
# Solo enforcement — confirm_intent is interrupt-bearing; side-effect-free notes may ride along
# ---------------------------------------------------------------------------


def _solo_confirm_keeps_companion_note() -> GateResult:
    from app.graphs.nodes import _gate_selected_tools

    state = {"messages": [], "user_confirmed": None}
    gated = _gate_selected_tools(
        state,
        [
            {"name": "explore_note", "args": {"content": "x"}},
            {"name": "confirm_intent", "args": {"summary": "Build Y for A"}},
        ],
    )
    passed = [g["name"] for g in gated] == ["explore_note", "confirm_intent"]
    return GateResult(
        gate="confirm_intent in composite keeps side-effect-free explore_note",
        passed=passed,
        score=1.0 if passed else 0.0,
        threshold=1.0,
        critical=True,
        reason=f"gated result: {[g['name'] for g in gated]}",
    )


def _solo_empty_summary_passes_for_tool_feedback() -> GateResult:
    from app.graphs.nodes import _gate_selected_tools

    state = {"messages": [], "user_confirmed": None}
    gated = _gate_selected_tools(state, [{"name": "confirm_intent", "args": {"summary": ""}}])
    passed = len(gated) == 1 and gated[0]["name"] == "confirm_intent"
    return GateResult(
        gate="empty summary on confirm_intent passes through for ToolMessage error",
        passed=passed,
        score=1.0 if passed else 0.0,
        threshold=1.0,
        critical=False,
        reason=f"dispatched to: {gated[0]['name'] if gated else 'none'}",
    )


# ---------------------------------------------------------------------------
# End-to-end gate flow — write_draft self-reject path before confirm, allowed after frame
# ---------------------------------------------------------------------------


def _flow_blocks_write_draft_before_confirm() -> GateResult:
    from app.graphs.nodes import _gate_selected_tools

    state = {"messages": [], "user_confirmed": None}
    gated = _gate_selected_tools(
        state,
        [{"name": "write_draft", "args": {"title": "Draft", "body": "Content"}}],
    )
    passed = len(gated) == 1 and gated[0]["name"] == "write_draft"
    return GateResult(
        gate="write_draft before confirm dispatches to tool-level error feedback",
        passed=passed,
        score=1.0 if passed else 0.0,
        threshold=1.0,
        critical=True,
        reason=f"write_draft dispatched to: {gated[0]['name'] if gated else 'none'}",
    )


def _flow_allows_write_draft_after_confirm() -> GateResult:
    names_after = _names({"messages": [], "user_confirmed": True})
    passed = "write_draft" in names_after
    return GateResult(
        gate="write_draft available after confirm",
        passed=passed,
        score=1.0 if passed else 0.0,
        threshold=1.0,
        critical=True,
        reason=f"write_draft={'write_draft' in names_after}",
    )


# ---------------------------------------------------------------------------
# Aggregation & markdown
# ---------------------------------------------------------------------------


def run_intent_gate_eval() -> dict[str, Any]:
    gates = [
        # confirm_intent tool
        _confirm_intent_in_schema_enum(),
        _confirm_intent_arg_keys(),
        _confirm_intent_is_interrupt_bearing(),
        _confirm_intent_sets_user_confirmed_stream_response(),
        # intent phase gate
        _gate_hides_artifact_tools_before_confirm(),
        _gate_offers_confirm_intent(),
        _gate_unlocks_after_confirm(),
        _gate_confirm_intent_one_shot(),
        # solo enforcement
        _solo_confirm_keeps_companion_note(),
        _solo_empty_summary_passes_for_tool_feedback(),
        # end-to-end flow
        _flow_blocks_write_draft_before_confirm(),
        _flow_allows_write_draft_after_confirm(),
    ]
    rows = [asdict(g) for g in gates]
    overall = all(g.passed for g in gates)
    return {"passed": overall, "gates": rows}


def _markdown_report(report: dict[str, Any]) -> str:
    icon = "✅" if report["passed"] else "❌"
    lines = [
        "# Eval: Intent Phase Gate",
        "",
        f"**Status:** {icon} {'PASSED' if report['passed'] else 'FAILED'}",
        "",
        "## Scenarios",
        "",
        "| # | Scenario | Score | Passed | Reason |",
        "| --- | --- | ---: | :---: | --- |",
    ]
    for i, row in enumerate(report["gates"], 1):
        icon_row = "✅" if row["passed"] else "❌"
        lines.append(f"| {i} | {row['gate']} | {row['score']:.2f} | {icon_row} | {row['reason']} |")
    total = len(report["gates"])
    passed_count = sum(1 for r in report["gates"] if r["passed"])
    lines += [
        "",
        f"**{passed_count}/{total} scenarios passed.**",
        "",
        "## Plan coverage",
        "",
        "| Phase | Scenarios |",
        "| --- | --- |",
        "| confirm_intent tool | schema name enum, arg/required keys, interrupt-bearing registration, "
        "user_confirmed+STREAM_RESPONSE |",
        "| Intent phase gate | artifact tools hidden, confirm_intent offered, post-confirm write_draft, "
        "one-shot enforcement |",
        "| Solo enforcement | interrupt-bearing keeps side-effect-free note companion, empty summary self-rejects "
        "by ToolMessage |",
        "| End-to-end flow | write_draft self-reject path before confirm, allowed after confirm |",
    ]
    return "\n".join(lines) + "\n"


def main(output_path: Path | None = None) -> int:
    report = run_intent_gate_eval()
    md = _markdown_report(report)
    if output_path:
        output_path.write_text(md, encoding="utf-8")
        print(f"Report written to {output_path}")
    else:
        print(md)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    import sys

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    raise SystemExit(main(out))
