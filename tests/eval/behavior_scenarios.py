"""Behavior-eval scenario registry.

Eval uses the same canonical journeys as the integration and benchmark lanes.
Keep this module as a compatibility import surface for older eval callers, but
do not define separate behavior scripts here.
"""

from tests.integration.scenarios.library import (
    CANONICAL_SCENARIOS,
    clarify_draft_approve,
    context_artifact_from_predecessors,
    reject_critique_redraft,
)

BEHAVIOR_SCENARIOS = CANONICAL_SCENARIOS

# Backward-compatible names for older tests or evidence scripts.
brd_happy_path = clarify_draft_approve
brd_ambiguous = reject_critique_redraft
prd_from_brd = context_artifact_from_predecessors
