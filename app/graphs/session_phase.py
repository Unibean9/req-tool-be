"""Session phase state machine — the explicit "where are we in the workflow" signal.

Pure module: no state reads, no DB, no imports from nodes/agent_tools. Callers compute
PhaseSignals from WorkflowState (agent_tools._phase_signals) and orchestrator_node is the ONLY
writer of state["session_phase"]; every other reader derives on the fly for legacy checkpoints.
"""

from dataclasses import dataclass

INTENT = "intent"
ELICIT = "elicit"
DRAFT = "draft"
REVIEW = "review"
FINALIZE = "finalize"

PHASES = (INTENT, ELICIT, DRAFT, REVIEW, FINALIZE)

# Directed legal edges (self-loops implicit). Regressions toward intent/elicit/draft are legal
# only for the focus-reset path — a new focused artifact clears critique state, so the derived
# target can move backwards; anything else backwards indicates state corruption.
_LEGAL_EDGES: frozenset[tuple[str, str]] = frozenset(
    {
        (INTENT, ELICIT),
        (INTENT, DRAFT),  # confirm + evidence can land within one composite turn
        (ELICIT, DRAFT),
        (DRAFT, REVIEW),
        (DRAFT, FINALIZE),  # a first-pass critique that immediately opens finalize skips the REVIEW dwell
        (REVIEW, DRAFT),
        (REVIEW, FINALIZE),
        (FINALIZE, REVIEW),
        # focus-reset regressions (critique state cleared, or a session re-targeted mid-flight)
        (DRAFT, ELICIT),
        (REVIEW, ELICIT),
        (FINALIZE, ELICIT),
        (REVIEW, INTENT),
        (DRAFT, INTENT),
        (ELICIT, INTENT),
        (FINALIZE, INTENT),
        (FINALIZE, DRAFT),
    }
)

# Tools removed from the menu per phase; everything else stays offered (permissive start — the
# baseline's failure modes are drafting pre-intent and re-eliciting mid-review, which these block).
PHASE_EXCLUDED_TOOLS: dict[str, frozenset[str]] = {
    INTENT: frozenset({"write_draft", "run_critique", "run_readiness_check", "finalize"}),
    ELICIT: frozenset({"confirm_intent", "run_critique", "run_readiness_check", "finalize"}),
    DRAFT: frozenset({"confirm_intent", "finalize"}),
    REVIEW: frozenset({"confirm_intent", "elicit", "web_search"}),
    FINALIZE: frozenset({"confirm_intent", "elicit", "web_search"}),
}


class IllegalPhaseTransition(RuntimeError):
    def __init__(self, current: str, target: str):
        self.current = current
        self.target = target
        super().__init__(f"Illegal session phase transition: {current} -> {target}")


@dataclass(frozen=True)
class PhaseSignals:
    """State-derived facts the machine transitions on (computed by agent_tools._phase_signals)."""

    user_confirmed: bool
    has_draft: bool
    has_evidence: bool
    critique_started: bool
    finalize_open: bool


def derive_phase(signals: PhaseSignals) -> str:
    """Target phase for the given signals — also the legacy-checkpoint derivation."""
    if not signals.user_confirmed:
        return INTENT
    if signals.has_draft and signals.critique_started:
        return FINALIZE if signals.finalize_open else REVIEW
    if signals.has_draft or signals.has_evidence:
        return DRAFT
    return ELICIT


def transition(current: str | None, signals: PhaseSignals) -> str:
    """Compute the next phase and validate edge legality; the ONLY path that may move the phase.

    current=None (fresh session or legacy checkpoint) adopts the derived phase directly.
    """
    target = derive_phase(signals)
    if current is None or current == target:
        return target
    if (current, target) not in _LEGAL_EDGES:
        raise IllegalPhaseTransition(current, target)
    return target


def phase_allows(phase: str | None, tool_name: str) -> bool:
    """Whether the per-phase menu offers this tool. Unknown/unset phase blocks nothing (legacy)."""
    if not phase:
        return True
    return tool_name not in PHASE_EXCLUDED_TOOLS.get(phase, frozenset())
