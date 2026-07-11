"""Menu matching matrix: `get_available_tools` behavior, branch by branch, computed by hand
from the documented per-tool conditions (see phase-03-brief.md). Every branch below asserts the
exact tool-name set the OLD inline-condition code returned for the same state, proving the
Policy-layer rewrite is behavior-preserving. Expected sets also account for the session-phase
menu (session_phase.py's PHASE_EXCLUDED_TOOLS), derived from each state's signals exactly as
`_phase_signals`/`derive_phase` compute it.
"""

from app.graphs.agent_tools import get_available_tools
from app.graphs.decision_graph import create_node
from app.graphs.session_phase import INTENT
from app.schemas.artifact_synthesis import ArtifactReadinessState

# The 14 tools that are unconditional at the tool-specific-condition layer (still subject to the
# phase+lifecycle rule).
_UNCONDITIONAL = {
    "ask_user",
    "respond",
    "write_draft",
    "critique_note",
    "explore_note",
    "confirm_intent",
    "read_artifact",
    "read_source_documents",
    "read_artifact_graph",
    "create_artifact_link",
    "propose_retirement",
    "run_impact_analysis",
    "elicit",
    "web_search",
}


def _names(state):
    return {t.name for t in get_available_tools(state)}


def test_no_draft_no_phase():
    """Fresh state, no session_phase set: derives to INTENT (unconfirmed) -> excludes
    write_draft/run_critique/run_readiness_check/finalize; no draft/coverage/decision-graph
    conditions hold."""
    state = {}
    assert _names(state) == _UNCONDITIONAL - {"write_draft", "run_critique", "run_readiness_check", "finalize"}


def test_has_draft_critique_zero():
    """Draft exists, critique_rounds == 0: run_critique available (has_draft and 0 < max);
    recommend_next_workflow available (has_draft); finalize/run_readiness_check need
    critique_rounds > 0, so absent. critique_started is False (rounds == 0, no quality_report) ->
    phase derives to DRAFT (has_draft, not yet critiqued) -> excludes confirm_intent/finalize."""
    state = {"user_confirmed": True, "draft_body": "A draft", "critique_rounds": 0}
    expected = (_UNCONDITIONAL - {"confirm_intent"}) | {"run_critique", "recommend_next_workflow"}
    assert _names(state) == expected


def test_has_draft_critique_gt_zero_gate_closed():
    """critique_not_passed: draft + critique_rounds > 0, but no passing quality_report ->
    _finalize_gate_open returns False -> finalize absent; run_readiness_check present (only needs
    has_draft and critique_rounds > 0, independent of the finalize gate); recommend_next_workflow
    present (has_draft). Phase derives to REVIEW (has_draft and critique_started, finalize
    closed) -> excludes confirm_intent/elicit/web_search."""
    state = {"user_confirmed": True, "draft_body": "A draft", "critique_rounds": 1}
    expected = (_UNCONDITIONAL - {"confirm_intent", "elicit", "web_search"}) | {
        "run_critique",
        "run_readiness_check",
        "recommend_next_workflow",
    }
    assert _names(state) == expected


def test_has_draft_critique_gt_zero_gate_open():
    """The finalize-open branch: a passing quality_report + sufficient readiness + fresh hash opens
    the gate -> finalize present. Phase derives to FINALIZE (also excludes
    confirm_intent/elicit/web_search, same set as REVIEW)."""
    import hashlib

    draft = "A draft"
    state = {
        "user_confirmed": True,
        "draft_body": draft,
        "critique_rounds": 1,
        "quality_report": {"quality_gate_result": "pass", "blocking_issues": []},
        "candidate_readiness": {"state": ArtifactReadinessState.SUFFICIENT, "score": 1.0, "gaps": []},
        "last_critiqued_draft_hash": hashlib.md5(draft.encode()).hexdigest()[:8],
    }
    expected = (_UNCONDITIONAL - {"confirm_intent", "elicit", "web_search"}) | {
        "run_critique",
        "run_readiness_check",
        "recommend_next_workflow",
        "finalize",
    }
    assert _names(state) == expected


def test_critique_rounds_at_max():
    """run_critique requires critique_rounds < CRITIQUE_ROUNDS_MAX, unless the draft hash differs
    from the last-critiqued hash (grace round). Here the hash matches (no edit since the last
    critique), so it stays gated off at the cap (even though run_readiness_check, which only needs
    > 0, stays). No quality_report -> finalize gate closed -> phase REVIEW (excludes
    confirm_intent/elicit/web_search)."""
    import hashlib

    from app.graphs.agent_tools import CRITIQUE_ROUNDS_MAX

    draft = "A draft"
    state = {
        "user_confirmed": True,
        "draft_body": draft,
        "critique_rounds": CRITIQUE_ROUNDS_MAX,
        "last_critiqued_draft_hash": hashlib.md5(draft.encode()).hexdigest()[:8],
    }
    expected = (_UNCONDITIONAL - {"confirm_intent", "elicit", "web_search"}) | {
        "run_readiness_check",
        "recommend_next_workflow",
    }
    assert _names(state) == expected


def test_coverage_signal_without_draft():
    """recommend_next_workflow's OR branch: no draft, but >= 2 sections have coverage signal ->
    still available. No draft means run_critique/run_readiness_check/finalize stay absent, and no
    decision_nodes/session_elicit_count means has_evidence is also False, so phase derives to
    ELICIT (confirmed, no draft/evidence) -> excludes confirm_intent (plus already-absent
    run_critique/run_readiness_check/finalize)."""
    state = {
        "user_confirmed": True,
        "section_coverage": {"a": "filled", "b": "partial"},
    }
    from app.documents.registry import status_score

    coverage = state["section_coverage"]
    assert sum(1 for v in coverage.values() if status_score(v) > 0.0) >= 2
    expected = (_UNCONDITIONAL - {"confirm_intent"}) | {"recommend_next_workflow"}
    assert _names(state) == expected


def test_decision_graph_disabled(monkeypatch):
    monkeypatch.setattr("app.graphs.agent_tools.settings.decision_graph_enabled", False)
    state = {"user_confirmed": True, "decision_nodes": {"N1": {"kind": "objective", "status": "confirmed"}}}
    names = _names(state)
    assert not ({"create_decision_node", "update_decision_node", "supersede_decision_node", "dismiss_question"} & names)


def test_decision_graph_enabled_no_nodes(monkeypatch):
    monkeypatch.setattr("app.graphs.agent_tools.settings.decision_graph_enabled", True)
    state = {"user_confirmed": True, "decision_nodes": {}}
    names = _names(state)
    assert "create_decision_node" in names
    assert "update_decision_node" not in names
    assert "supersede_decision_node" not in names
    assert "dismiss_question" not in names


def test_decision_graph_enabled_with_nodes(monkeypatch):
    monkeypatch.setattr("app.graphs.agent_tools.settings.decision_graph_enabled", True)
    node = create_node(kind="objective", statement="A goal", origin={"source": "test"}, status="confirmed")
    state = {"user_confirmed": True, "decision_nodes": {"N1": node}}
    names = _names(state)
    assert {"create_decision_node", "update_decision_node", "supersede_decision_node"} <= names
    assert "dismiss_question" not in names


def test_decision_graph_dismiss_question(monkeypatch):
    """A parked open_question node present -> dismiss_question joins the menu."""
    monkeypatch.setattr("app.graphs.agent_tools.settings.decision_graph_enabled", True)
    node = create_node(kind="open_question", statement="Confirm scope", origin={"source": "test"}, status="parked")
    state = {"user_confirmed": True, "decision_nodes": {"N1": node}}
    names = _names(state)
    assert "dismiss_question" in names
    assert {"create_decision_node", "update_decision_node", "supersede_decision_node"} <= names


def test_phase_excludes_tool():
    """A phase from PHASE_EXCLUDED_TOOLS hides at least one tool: INTENT (user unconfirmed) hides
    write_draft/run_critique/run_readiness_check/finalize even when a draft technically exists in
    state (session_phase overrides derivation)."""
    state = {"session_phase": INTENT, "draft_body": "A draft", "critique_rounds": 1}
    names = _names(state)
    assert "write_draft" not in names
    assert "run_critique" not in names
    assert "run_readiness_check" not in names
    assert "finalize" not in names


def test_lifecycle_blocked_tool():
    """A lifecycle report in a truthy, non-curation-exception state ("current") hides write_draft
    silently (menu-time logs via lifecycle_tool_menu, but does not surface a tool_error) while
    leaving every other tool untouched. No draft/evidence in state -> phase derives to ELICIT ->
    also excludes confirm_intent (plus already-absent run_critique/run_readiness_check/finalize)."""
    state = {
        "user_confirmed": True,
        "focused_artifact_id": "artifact-1",
        "artifact_type": "vision_objectives",
        "lifecycle_reports": [
            {
                "artifact_type": "vision_objectives",
                "artifact_id": "artifact-1",
                "state": "current",
                "reason": "current reason",
                "allowed_actions": [],
            }
        ],
    }
    names = _names(state)
    assert "write_draft" not in names
    assert names == (_UNCONDITIONAL - {"write_draft", "confirm_intent"})


def test_lifecycle_stale_curation_exception():
    """The documented asymmetry: at menu-time, a "stale" lifecycle report with reason
    "stale_artifact_requires_curation_action" is the ONE truthy reason that still keeps the tool
    (write_draft is the tool that resolves the stale-curation state) — no other truthy reason gets
    this exception. Same ELICIT-phase exclusion as the "current" branch above."""
    state = {
        "user_confirmed": True,
        "focused_artifact_id": "artifact-1",
        "artifact_type": "vision_objectives",
        "lifecycle_reports": [
            {
                "artifact_type": "vision_objectives",
                "artifact_id": "artifact-1",
                "state": "stale",
                "reason": "stale reason",
                "allowed_actions": ["reconcile"],
            }
        ],
    }
    names = _names(state)
    assert "write_draft" in names
    assert names == (_UNCONDITIONAL - {"confirm_intent"})
