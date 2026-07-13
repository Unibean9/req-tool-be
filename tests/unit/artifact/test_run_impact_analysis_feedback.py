import json

import pytest

from app.graphs.agent_tools import artifact_links


@pytest.mark.asyncio
async def test_impact_stale_warning_stays_on_its_tool_message(monkeypatch):
    async def load_links(_config):
        return []

    result = {
        "affected_node_ids": ["requirement-1"],
        "stale_artifact_ids": ["prd-1"],
        "decision_nodes": {"requirement-1": {"status": "needs_confirmation"}},
    }
    monkeypatch.setattr(artifact_links, "_load_artifact_links", load_links)
    monkeypatch.setattr(artifact_links, "impact", lambda *_args, **_kwargs: result)

    command = await artifact_links._run_impact_analysis_impl(
        "Changed scope",
        {"feedback_summary": {"dropped_tools": ["write_draft"]}},
        {},
        "impact-call-1",
    )

    assert "feedback_summary" not in command.update
    message = command.update["messages"][0]
    assert message.tool_call_id == "impact-call-1"
    assert json.loads(message.content) == {
        "affected_node_ids": ["requirement-1"],
        "stale_artifact_ids": ["prd-1"],
        "stale_warning": "1 node need reconfirmation due to change: requirement-1",
    }


def test_feedback_prompt_ignores_legacy_stale_warning():
    from app.graphs.analysis.prompt_assembly import _build_feedback_control_block

    assert "stale_warning" not in _build_feedback_control_block(
        {"feedback_summary": {"stale_warning": "legacy state value"}}
    )
