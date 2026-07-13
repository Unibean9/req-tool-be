"""Phase 4 Test D — `gate_model_selection`'s `next_feedback` content equivalence.

Constructs a single turn where a tool is dropped for each of the three reasons
(out-of-phase, lifecycle-blocked, solo-dropped) and asserts `next_feedback` matches
the pre-Phase-4 documented logic exactly (see `gate_model_selection`'s own
feedback-construction code, unchanged this phase, and `phase-04-brief.md` section A).
"""

from langchain_core.messages import AIMessage

from app.graphs.analysis.tool_gating import gate_model_selection
from app.graphs.session_phase import DRAFT
from app.graphs.state import build_initial_workflow_state


def _state():
    state = build_initial_workflow_state(artifact_type="vision_objectives", workflow_area="analysis", step_key=None)
    state.update(
        {
            "user_confirmed": True,
            "session_phase": DRAFT,
            "focused_artifact_id": "artifact-vision",
            "lifecycle_reports": [
                {
                    "artifact_type": "vision_objectives",
                    "artifact_id": "artifact-vision",
                    "state": "current",
                    "reason": "current reason",
                    "allowed_actions": [],
                    "blockers": [],
                }
            ],
        }
    )
    return state


def test_feedback_summary_covers_all_three_drop_reasons_in_one_turn():
    state = _state()
    message = AIMessage(
        content="",
        tool_calls=[
            # out-of-phase in DRAFT: finalize requires REVIEW+.
            {"id": "c1", "name": "finalize", "args": {"summary": "done"}},
            # lifecycle-blocked: "current" artifact blocks write_draft regardless of args.
            {"id": "c2", "name": "write_draft", "args": {"title": "Vision", "body": "## Vision\nx"}},
            # solo-dropped: paired with the interrupt-bearing ask_user, not a note tool.
            {"id": "c3", "name": "ask_user", "args": {"message": "?"}},
            {"id": "c4", "name": "read_artifact", "args": {"id": "00000000-0000-0000-0000-000000000001"}},
        ],
    )

    _model, gated, dropped, feedback, out_of_phase = gate_model_selection(state, message)

    assert out_of_phase == ["finalize"]
    assert feedback["out_of_phase_tools"] == {"phase": DRAFT, "dropped": ["finalize"]}
    assert feedback["lifecycle_blocked_tools"] == [
        {"name": "write_draft", "reason": "current_artifact_reproposal_blocked", "state": "current"}
    ]
    assert feedback["dropped_tools"] == ["read_artifact"]
    assert [g["name"] for g in gated] == ["ask_user"]
    assert set(dropped) == {"finalize", "write_draft", "read_artifact"}
