"""Per-call gating rules for `get_available_tools`'s menu (Mode.MENU), plus a
phase+lifecycle rule shared with dispatch-time gating (Mode.DISPATCH).

Rules that need facts derived from `WorkflowState` (draft body, session phase,
finalize gate) reach `app.graphs.agent_tools` through a module reference
(`from app.graphs import agent_tools`) rather than importing names out of it,
and only look up attributes on it inside `evaluate()` bodies — `agent_tools`
imports this module at module load time, so those attributes do not exist yet
while this module is being imported; resolving them lazily at call time (after
both modules have finished loading) avoids the cycle without the function-local
`import`/`from ... import` statements `test_analysis_decomposition.py` forbids
for cycle modules.
"""

from __future__ import annotations

from typing import Any

from app.config import settings
from app.documents.registry import status_score
from app.graphs import agent_tools
from app.graphs.gate_logging import log_gate_decision
from app.graphs.gating import dispatch_rules
from app.graphs.gating.engine import is_batch_registered, is_registered, register_batch_rule, register_rule
from app.graphs.gating.rules import Mode
from app.graphs.gating.verdict import Verdict
from app.graphs.lifecycle_context import focused_lifecycle_report, lifecycle_tool_block_reason
from app.graphs.session_phase import phase_allows


def _has_draft_and_critique_rounds(state: Any) -> tuple[bool, int]:
    has_draft = bool(agent_tools._cached_draft_body(state).strip())
    critique_rounds = state.get("critique_rounds") or 0
    return has_draft, critique_rounds


class FinalizeMenuRule:
    """`finalize`: candidate iff a draft exists, at least one critique round has
    run, AND the finalize quality gate is open. `_finalize_gate_open` (the sole
    source of `log_gate_decision("finalize", ...)`) is only called when the
    first two conditions hold, matching the original inline logic exactly.
    """

    name = "finalize_gate"
    side_effecting = True

    def evaluate(self, tool_call: Any, state: Any) -> Verdict:
        if tool_call.get("name") != "finalize":
            return Verdict.allow()
        has_draft, critique_rounds = _has_draft_and_critique_rounds(state)
        if has_draft and critique_rounds > 0 and agent_tools._finalize_gate_open(state):
            return Verdict.allow()
        return Verdict.deny("finalize_gate_closed")


class RunCritiqueMenuRule:
    """`run_critique`: candidate iff a draft exists and either the critique-rounds
    cap has not been reached, or (at/past the cap) the draft has been edited since
    the last critique — a single grace round to re-score the edited draft. Uses
    `agent_tools._draft_hash_stale` so this condition can never drift from the
    one `_run_critique_impl`'s cap guard enforces at call time."""

    name = "run_critique_menu"
    side_effecting = False

    def evaluate(self, tool_call: Any, state: Any) -> Verdict:
        if tool_call.get("name") != "run_critique":
            return Verdict.allow()
        has_draft, critique_rounds = _has_draft_and_critique_rounds(state)
        if not has_draft:
            return Verdict.deny("run_critique_unavailable")
        if critique_rounds < settings.max_critique_rounds:
            return Verdict.allow()
        if agent_tools._draft_hash_stale(state):
            return Verdict.allow()
        return Verdict.deny("run_critique_unavailable")


class RecommendNextWorkflowMenuRule:
    """`recommend_next_workflow`: candidate once there is a draft, or once >= 2
    sections have any coverage signal."""

    name = "recommend_next_workflow_menu"
    side_effecting = False

    def evaluate(self, tool_call: Any, state: Any) -> Verdict:
        if tool_call.get("name") != "recommend_next_workflow":
            return Verdict.allow()
        has_draft, _critique_rounds = _has_draft_and_critique_rounds(state)
        coverage = state.get("section_coverage") or {}
        sections_with_signal = sum(1 for v in coverage.values() if status_score(v) > 0.0)
        if has_draft or sections_with_signal >= 2:
            return Verdict.allow()
        return Verdict.deny("recommend_next_workflow_unavailable")


class RunReadinessCheckMenuRule:
    """`run_readiness_check`: candidate iff a draft exists and a critique round
    has run."""

    name = "run_readiness_check_menu"
    side_effecting = False

    def evaluate(self, tool_call: Any, state: Any) -> Verdict:
        if tool_call.get("name") != "run_readiness_check":
            return Verdict.allow()
        has_draft, critique_rounds = _has_draft_and_critique_rounds(state)
        if has_draft and critique_rounds > 0:
            return Verdict.allow()
        return Verdict.deny("run_readiness_check_unavailable")


class PhaseLifecycleMenuRule:
    """Combined session-phase + artifact-lifecycle gate, applied to every tool
    name after all tool-specific candidacy checks above.

    Mode-aware: the same class serves menu-time gating (this phase) and
    dispatch-time gating (wired in a later phase) since the two modes apply
    the lifecycle check with different args and logging semantics — see the
    per-branch docstrings below.
    """

    name = "phase_and_lifecycle"
    side_effecting = True

    def __init__(self, mode: Mode) -> None:
        self.mode = mode

    def evaluate(self, tool_call: Any, state: Any) -> Verdict:
        name = tool_call.get("name")
        # Menu-time passes the already-computed phase in tool_call["phase"] so this rule (invoked
        # once per candidate tool) does not re-derive it — and re-trigger `_finalize_gate_open`'s
        # logging — up to 22 times per `get_available_tools` call; original code computed phase
        # exactly once per call. Isolated/dispatch callers that omit "phase" fall back to computing it.
        phase = tool_call["phase"] if "phase" in tool_call else agent_tools.current_session_phase(state)
        if not phase_allows(phase, name):
            # Silent in both modes: menu-time never logged this; dispatch-time
            # logs it via `_log_tool_error` at the `_gate_selected_tools` call
            # site, not from this rule.
            return Verdict.deny("phase_excludes_tool")

        if self.mode is Mode.MENU:
            reason = lifecycle_tool_block_reason(state, name, {})
            if reason is None or reason == "stale_artifact_requires_curation_action":
                return Verdict.allow()
            report = focused_lifecycle_report(state) or {}
            log_gate_decision(
                "lifecycle_tool_menu",
                "blocked",
                reason=reason,
                extra={
                    "tool": name,
                    "artifact_type": report.get("artifact_type"),
                    "artifact_id": report.get("artifact_id"),
                    "lifecycle_state": report.get("state"),
                },
            )
            return Verdict.deny(reason)

        # Mode.DISPATCH: every truthy reason blocks (no stale-curation exception),
        # using the tool call's real args instead of `{}`.
        args = tool_call.get("args") or {}
        reason = lifecycle_tool_block_reason(state, name, args)
        if reason is None:
            return Verdict.allow()
        report = focused_lifecycle_report(state) or {}
        log_gate_decision(
            "lifecycle_tool_gate",
            "blocked",
            reason=reason,
            extra={"tool": name, "lifecycle_state": report.get("state") or ""},
        )
        return Verdict.deny(reason)


def ensure_menu_rules_registered() -> None:
    """Idempotently register the Mode.MENU rules.

    Safe to call on every `get_available_tools` invocation: production code
    shares the gating engine's process-global registry with tests that call
    `gating.reset()` for isolation (see `test_gating_engine.py`), so
    registration must self-heal if something clears it mid test-session
    rather than assuming import-time registration is permanent.
    """
    if is_registered(PhaseLifecycleMenuRule.name, Mode.MENU):
        return
    register_rule(FinalizeMenuRule(), (Mode.MENU,))
    register_rule(RunCritiqueMenuRule(), (Mode.MENU,))
    register_rule(RecommendNextWorkflowMenuRule(), (Mode.MENU,))
    register_rule(RunReadinessCheckMenuRule(), (Mode.MENU,))
    register_rule(PhaseLifecycleMenuRule(mode=Mode.MENU), (Mode.MENU,))


def ensure_dispatch_rules_registered() -> None:
    """Idempotently register the Mode.DISPATCH rules used by dispatch-selection
    gating (`analysis.tool_gating._gate_selected_tools`).

    Same self-healing motivation as `ensure_menu_rules_registered` above. A
    distinct `PhaseLifecycleMenuRule` instance is registered for `Mode.DISPATCH`
    (never the `Mode.MENU` instance registered above) — see that class's
    per-branch docstring for how the two modes differ.
    """
    if not is_registered(PhaseLifecycleMenuRule.name, Mode.DISPATCH):
        register_rule(PhaseLifecycleMenuRule(mode=Mode.DISPATCH), (Mode.DISPATCH,))
    if not is_batch_registered(dispatch_rules.SoloInvariantBatchRule.name):
        register_batch_rule(dispatch_rules.SoloInvariantBatchRule())
