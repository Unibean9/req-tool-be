"""Shared workflow-state fixtures for the menu-gating behavior-preservation matrix.

`test_menu_gating_matrix.py` asserts these states against the legacy inline-condition tool menu;
`test_capability_resolver_golden.py` asserts the same states against `CapabilityResolver` and
diffs the two verdicts. Both files must exercise identical state shapes for the comparison to mean
anything, so the states live here once instead of being hand-copied in each file (a hand-copy can
silently drift when one file is updated and the other is not).

Not a test module itself (no `test_` prefix): pytest does not collect it.
"""

import hashlib

from app.graphs.agent_tools import CRITIQUE_ROUNDS_MAX
from app.schemas.artifact_synthesis import ArtifactReadinessState

NO_DRAFT_NO_PHASE: dict = {}

HAS_DRAFT_CRITIQUE_ZERO: dict = {"user_confirmed": True, "draft_body": "A draft", "critique_rounds": 0}

HAS_DRAFT_CRITIQUE_GT_ZERO_GATE_CLOSED: dict = {
    "user_confirmed": True,
    "draft_body": "A draft",
    "critique_rounds": 1,
}


def _draft_hash(draft: str) -> str:
    return hashlib.md5(draft.encode()).hexdigest()[:8]


def has_draft_critique_gt_zero_gate_open() -> dict:
    draft = "A draft"
    return {
        "user_confirmed": True,
        "draft_body": draft,
        "critique_rounds": 1,
        "quality_report": {"quality_gate_result": "pass", "blocking_issues": []},
        "candidate_readiness": {"state": ArtifactReadinessState.SUFFICIENT, "score": 1.0, "gaps": []},
        "last_critiqued_draft_hash": _draft_hash(draft),
    }


def critique_rounds_at_max() -> dict:
    draft = "A draft"
    return {
        "user_confirmed": True,
        "draft_body": draft,
        "critique_rounds": CRITIQUE_ROUNDS_MAX,
        "last_critiqued_draft_hash": _draft_hash(draft),
    }


COVERAGE_SIGNAL_WITHOUT_DRAFT: dict = {
    "user_confirmed": True,
    "section_coverage": {"a": "filled", "b": "partial"},
}


def phase_excludes_tool(session_phase) -> dict:
    return {"session_phase": session_phase, "draft_body": "A draft", "critique_rounds": 1}
