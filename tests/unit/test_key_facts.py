"""D3 — Key Facts State tests.

Covers: note_parser tag parsing, prompt injection, summarize_node non-compression guard.
"""

import pytest

from app.graphs.nodes import _build_key_facts_block, _build_tool_selection_prompt
from app.graphs.note_parser import extract_structured_objects
from tests.integration.test_graph_nodes import _state

# ---------------------------------------------------------------------------
# note_parser
# ---------------------------------------------------------------------------

def test_note_parser_parses_key_fact_tag():
    content = "KEY_FACT: App will support 10k DAU | source: user | turn: 2"
    result = extract_structured_objects(content)
    assert len(result["key_facts"]) == 1
    fact = result["key_facts"][0]
    assert fact["statement"] == "App will support 10k DAU"
    assert fact["source"] == "user"
    assert fact["turn"] == "2"


def test_note_parser_key_fact_optional_fields_default_to_empty():
    content = "KEY_FACT: DAU target la 10k"
    result = extract_structured_objects(content)
    assert len(result["key_facts"]) == 1
    fact = result["key_facts"][0]
    assert fact["statement"] == "DAU target la 10k"
    assert fact["source"] == ""
    assert fact["turn"] == ""


def test_note_parser_key_fact_and_assumption_in_same_content():
    content = (
        "KEY_FACT: DAU la 10k | source: user | turn: 1\n"
        "ASSUMPTION: Mobile users | source: pm | confidence: high | impact: high | owner: pm | status: open"
    )
    result = extract_structured_objects(content)
    assert len(result["key_facts"]) == 1
    assert len(result["assumptions"]) == 1


def test_note_parser_unrecognized_lines_ignored():
    content = "This is a regular note without a tag\nKEY_FACT: Fact that | source: dev | turn: 3"
    result = extract_structured_objects(content)
    assert len(result["key_facts"]) == 1


def test_note_parser_empty_content_returns_empty_buckets():
    result = extract_structured_objects("")
    assert result["key_facts"] == []
    assert result["assumptions"] == []
    assert result["risks"] == []
    assert result["open_questions"] == []


# ---------------------------------------------------------------------------
# prompt injection (_build_key_facts_block)
# ---------------------------------------------------------------------------

def test_key_facts_block_empty_when_no_facts():
    assert _build_key_facts_block(_state()) == ""


def test_key_facts_block_contains_statement_when_facts_present():
    state = _state()
    state["key_facts"] = [{"statement": "DAU la 10k", "source": "user", "turn": "1"}]
    block = _build_key_facts_block(state)
    assert "DAU la 10k" in block
    assert "user" in block


def test_build_tool_selection_prompt_contains_key_facts_when_non_empty():
    state = _state()
    state["key_facts"] = [{"statement": "Budget toi da 500tr", "source": "cfo", "turn": "2"}]
    prompt = _build_tool_selection_prompt(state, [])
    assert "Budget toi da 500tr" in prompt


def test_build_tool_selection_prompt_no_key_facts_section_when_empty():
    prompt = _build_tool_selection_prompt(_state(), [])
    assert "KEY FACTS" not in prompt


# ---------------------------------------------------------------------------
# summarize_node does not compress key_facts
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_summarize_node_does_not_clear_key_facts():
    """summarize_node must return key_facts unchanged (no field in its return dict)."""
    from unittest.mock import AsyncMock, patch

    from app.graphs.nodes import summarize_node

    state = _state()
    state["key_facts"] = [{"statement": "DAU target 10k", "source": "user", "turn": "1"}]

    # Route must go to 'summarize' path — patch route_before_analyze to return 'summarize'
    # and mock the llm_client to avoid real API call.
    with patch("app.graphs.nodes.route_before_analyze", return_value="summarize"):
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value=({"summary": "summary"}, {}))
        config = {"configurable": {"llm_client": mock_llm}}

        result = await summarize_node(state, config)

    # summarize_node only updates conversation_summary — it must NOT touch key_facts.
    assert "key_facts" not in result
