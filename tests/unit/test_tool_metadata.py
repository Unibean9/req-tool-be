from app.graphs.tool_metadata import interrupt_bearing_tools, policy_table, side_effect_free_note_tools


def test_policy_table_matches_pre_refactor_values():
    assert policy_table() == {
        "read_artifacts": "allow",
        "read_artifact_graph": "allow",
        "read_workflow_steps": "allow",
        "read_source_documents": "allow",
        "read_project_context": "allow",
        "init_workflow_run": "require_approval",
        "create_artifact": "require_approval",
        "update_artifact": "require_approval",
        "create_artifact_link": "require_approval",
        "propose_retirement": "require_approval",
        "delete_artifact_link": "require_approval",
        "create_artifact_review": "require_approval",
        "finalize": "require_critique",
    }


def test_interrupt_and_note_sets_match_pre_refactor_values():
    assert interrupt_bearing_tools() == frozenset(
        {
            "ask_user",
            "respond",
            "write_draft",
            "create_artifact_link",
            "propose_retirement",
            "finalize",
            "confirm_intent",
        }
    )
    assert side_effect_free_note_tools() == frozenset({"critique_note", "explore_note"})
