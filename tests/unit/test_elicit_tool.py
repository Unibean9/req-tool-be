"""elicit tool: deterministic technique scaffolds and external knowledge integration.

elicit returns the technique FRAME (deterministic) for the agent to reason over and record as
nodes; comparable_products pulls real data via web_search and falls back to model knowledge when
search is unavailable. The plain `elicit(...)` is importable; the registry exposes a `@tool` named
"elicit".
"""

import pytest

from app.graphs.agent_tools import elicit, get_all_analyzer_tools


def test_elicit_5_whys_returns_structured_chain():
    out = elicit(technique="5_whys", seed="hụt nguyên liệu")

    assert out["technique"] == "5_whys"
    assert isinstance(out["chain"], list)
    assert len(out["chain"]) >= 3
    assert isinstance(out["root_cause"], str) and out["root_cause"].strip()


def test_elicit_reverse_returns_failure_modes():
    out = elicit(technique="reverse", seed="app định lượng thất bại vì?")

    assert isinstance(out["failure_modes"], list)
    assert len(out["failure_modes"]) >= 2
    for item in out["failure_modes"]:
        assert isinstance(item["mode"], str)
        assert isinstance(item["mitigation_hint"], str)


def test_elicit_moscow_returns_categorized_items():
    out = elicit(technique="moscow", seed="scope v1 cho định lượng+trừ kho")

    for key in ("must", "should", "could", "wont"):
        assert key in out
    assert out["must"]
    assert out["wont"]


def test_elicit_comparable_products_uses_web_search():
    calls = []

    def fake_client(query: str) -> list[dict]:
        calls.append(query)
        return [
            {"title": "iPOS", "snippet": "quản lý quán", "url": "https://1.example"},
            {"title": "KiotViet", "snippet": "bán hàng F&B", "url": "https://2.example"},
            {"title": "Sapo", "snippet": "POS cà phê", "url": "https://3.example"},
        ]

    out = elicit(technique="comparable_products", seed="app quản lý quán cà phê", search_client=fake_client)

    assert calls, "web_search client should have been invoked"
    assert "quán cà phê" in calls[0]
    assert isinstance(out["products"], list)
    assert len(out["products"]) >= 3
    for item in out["products"]:
        assert {"name", "model", "relevance"} <= set(item)
    assert out["source"] == "web_search"


def test_elicit_comparable_products_falls_back_to_model_knowledge():
    def failing_client(query: str) -> list[dict]:
        raise RuntimeError("network down")

    out = elicit(technique="comparable_products", seed="app quản lý quán cà phê", search_client=failing_client)

    assert out["products"]
    assert out["source"] == "model_knowledge"


def test_elicit_unknown_technique_raises_value_error():
    with pytest.raises(ValueError, match="unknown_xyz"):
        elicit(technique="unknown_xyz", seed="...")


def test_elicit_tool_in_registry():
    tool = next((t for t in get_all_analyzer_tools() if t.name == "elicit"), None)
    assert tool is not None
    schema = tool.args_schema.model_json_schema()
    props = schema["properties"]
    assert "technique" in props
    assert "seed" in props
    assert set(schema.get("required", [])) >= {"technique", "seed"}
    assert "enum" in props["technique"]
