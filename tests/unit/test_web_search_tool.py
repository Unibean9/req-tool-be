"""web_search tool: registration, structured results, graceful unavailability.

The plain `web_search(query, *, client=None)` is the importable callable; the registry exposes a
`@tool` whose `.name == "web_search"`. With no provider configured (settings default) and no client
injected, web_search must degrade to {"results": [], "error": "search_unavailable"} — never raise.
"""

from app.graphs.agent_tools import get_all_analyzer_tools, web_search


def test_web_search_tool_registered():
    tool = next((t for t in get_all_analyzer_tools() if t.name == "web_search"), None)
    assert tool is not None
    props = tool.args_schema.model_json_schema()["properties"]
    assert "query" in props


def test_web_search_returns_structured_results():
    def fake_client(query: str) -> list[dict]:
        return [
            {"title": "POS A", "snippet": "coffee shop software", "url": "https://a.example"},
            {"title": "POS B", "snippet": "sales management", "url": "https://b.example"},
            {"title": "POS C", "snippet": "customer points", "url": "https://c.example"},
        ]

    out = web_search("coffee shop management software", client=fake_client)

    assert isinstance(out["results"], list)
    assert len(out["results"]) >= 1
    for item in out["results"]:
        assert {"title", "snippet", "url"} <= set(item)


def test_web_search_handles_unavailable_client():
    # No provider configured (settings default) and no client injected -> graceful empty result.
    out = web_search("bat ky truy van nao")

    assert out == {"results": [], "error": "search_unavailable"}
