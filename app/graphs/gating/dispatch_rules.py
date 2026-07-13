"""Dispatch-time gating rules.

`SoloInvariantBatchRule` is registered in the gating `Registry` (via
`menu_rules.ensure_dispatch_rules_registered()`) and reached generically through
`gating.check_batch(...)` from `analysis.tool_gating._gate_selected_tools`.

`LifecycleWriteDraftBlockRule` and `ColdStartDraftBlockRule` are deliberately NOT
registered anywhere. `agent_tools._write_draft_impl` instantiates and calls
`.evaluate(...)` on them directly, at the exact point its two inline checks used to
sit. Routing them through the generic `gating.check(..., Mode.DISPATCH)` entrypoint
instead would re-evaluate `PhaseLifecycleMenuRule(mode=Mode.DISPATCH)` a second time
for the same `write_draft` call already dispatched past `_gate_selected_tools`,
double-firing its `log_gate_decision("lifecycle_tool_gate", ...)` call — see
`implementation-notes.md`'s 2026-07-09 Phase 4 entry for the full rationale.

This module resolves cross-module facts (`analysis.tool_gating`'s
`_INTERRUPT_BEARING_TOOLS`/`_SIDE_EFFECT_FREE_NOTE_TOOLS`/`_log_tool_error`,
`agent_tools`'s `_cold_start_draft_blocked`) via module-reference imports
(`from app.graphs import agent_tools`, `from app.graphs.analysis import
tool_gating`) with attribute lookups deferred to inside `evaluate()` bodies, the
same technique `gating/menu_rules.py` uses for `agent_tools` — both `tool_gating`
and `menu_rules` import this module at load time, so eager attribute access here
would trip a circular-import failure before either finishes loading.
"""

from __future__ import annotations

from typing import Any

from app.graphs import agent_tools
from app.graphs.analysis import tool_gating
from app.graphs.gate_logging import log_gate_decision
from app.graphs.gating.verdict import Verdict
from app.graphs.lifecycle_context import lifecycle_tool_block_reason

# Priority order when the model proposes multiple interrupt-bearing tools in the same batch —
# the tool that comes first in this tuple is kept, the remaining interrupt-bearing tools are
# dropped even if the model listed them first in tool_calls. Tools not in the tuple (note/
# read-only/...) keep their relative position.
_SOLO_INVARIANT_PRIORITY: tuple[str, ...] = (
    "ask_user",
    "confirm_intent",
    "write_draft",
    "finalize",
    "create_artifact_link",
    "propose_retirement",
    "respond",
)


def _sort_by_solo_invariant_priority(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def priority(indexed_item: tuple[int, dict[str, Any]]) -> tuple[int, int]:
        index, item = indexed_item
        name = item["name"]
        if name in _SOLO_INVARIANT_PRIORITY:
            rank = _SOLO_INVARIANT_PRIORITY.index(name)
        else:
            rank = len(_SOLO_INVARIANT_PRIORITY)
        return (rank, index)

    return [item for _, item in sorted(enumerate(tool_calls), key=priority)]


class SoloInvariantBatchRule:
    """At most one interrupt-bearing tool per turn; side-effect-free notes may ride
    along under it. Reproduces the pre-Phase-4 solo-invariant block from
    `analysis.tool_gating._gate_selected_tools` verbatim, including both
    `_log_tool_error` call sites (code/message strings unchanged). When multiple
    interrupt-bearing tools appear in the same batch, the one kept follows the
    explicit priority table (`_SOLO_INVARIANT_PRIORITY`) instead of "whichever
    tool the model listed first"."""

    name = "solo_invariant"

    def evaluate(self, tool_calls: list[dict[str, Any]], _state: Any) -> list[dict[str, Any]]:
        interrupt_bearing = tool_gating._INTERRUPT_BEARING_TOOLS
        side_effect_free = tool_gating._SIDE_EFFECT_FREE_NOTE_TOOLS
        interrupt_count = sum(1 for item in tool_calls if item["name"] in interrupt_bearing)
        if not interrupt_count:
            return tool_calls
        if interrupt_count > 1:
            tool_calls = _sort_by_solo_invariant_priority(tool_calls)
        kept: list[dict[str, Any]] = []
        seen_interrupt = False
        for item in tool_calls:
            name = item["name"]
            if name in interrupt_bearing:
                if not seen_interrupt:
                    kept.append(item)
                    seen_interrupt = True
                else:
                    tool_gating._log_tool_error(
                        "dropped_interrupt_tool",
                        name,
                        "dropped: an interrupt-bearing tool was already selected this turn",
                    )
            elif name in side_effect_free:
                kept.append(item)
            else:
                tool_gating._log_tool_error(
                    "dropped_with_interrupt_tool",
                    name,
                    "dropped: paired with an interrupt-bearing tool",
                )
        return kept


class LifecycleWriteDraftBlockRule:
    """`write_draft`'s lifecycle-block hard gate. NOT registered — called directly
    by `agent_tools._write_draft_impl` (see module docstring for why)."""

    name = "lifecycle_write_draft_block"
    side_effecting = True

    def evaluate(self, tool_call: dict[str, Any], state: Any) -> Verdict:
        args = tool_call.get("args") or {}
        reason = lifecycle_tool_block_reason(state, "write_draft", args)
        if not reason:
            return Verdict.allow()
        log_gate_decision("lifecycle_tool_impl", "blocked", reason=reason)
        return Verdict.deny(reason)


class ColdStartDraftBlockRule:
    """`write_draft`'s cold-start-block hard gate. NOT registered — called directly
    by `agent_tools._write_draft_impl` (see module docstring for why). No logging on
    deny, matching the pre-Phase-4 inline check, which never called
    `log_gate_decision` for this path."""

    name = "cold_start_draft_block"
    side_effecting = True

    def evaluate(self, _tool_call: dict[str, Any], state: Any) -> Verdict:
        if agent_tools._cold_start_draft_blocked(state):
            return Verdict.deny("cold_start_requires_elicitation")
        return Verdict.allow()
