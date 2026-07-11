"""Elicitation surface — BMAD technique scaffolds + external knowledge (web search).

Self-contained: depends only on stdlib, langchain/langgraph, settings, and the shared
tool-error helpers. No import back into the coordinator.
"""

import json
import re
from typing import Annotated, Literal

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from app.config import settings
from app.graphs.agent_tools._shared import RecoverableToolError, _recoverable_tool_update

ELICIT_TECHNIQUES = (
    "5_whys",
    "reverse",
    "moscow",
    "first_principles",
    "comparable_products",
    "pre_mortem",
    "tree_of_thought",
    "socratic_questioning",
    "challenge_assumptions",
    "event_storming",
)


def _duckduckgo_search(query: str) -> list[dict]:
    """Keyless DuckDuckGo HTML scrape. Best-effort; web_search wraps this in graceful fallback."""

    import httpx

    resp = httpx.get(
        "https://html.duckduckgo.com/html/",
        params={"q": query},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=10.0,
    )
    resp.raise_for_status()
    results = []
    for match in re.finditer(r'result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', resp.text):
        title = re.sub(r"<[^>]+>", "", match.group(2)).strip()
        results.append({"title": title, "snippet": title, "url": match.group(1)})
    return results[:8]


def _default_search_client():
    """Resolve the configured search client, or None when search is disabled (CI default)."""
    if settings.search_provider == "duckduckgo":
        return _duckduckgo_search
    return None


def web_search(query: str, *, client=None) -> dict:
    """Run a web search, degrading gracefully when no provider is available.

    Returns {"results": [...], "source": "web_search"} on success, or
    {"results": [], "error": "search_unavailable"} when no client is configured or the call fails —
    never raises, so elicit() can fall back to model knowledge. Each result is {title, snippet, url}.
    """
    search = client or _default_search_client()
    if search is None:
        return {"results": [], "error": "search_unavailable"}
    try:
        raw = search(query)
    except Exception:
        return {"results": [], "error": "search_unavailable"}
    return {"results": list(raw), "source": "web_search"}


def _elicit_5_whys(seed: str) -> dict:
    chain = [{"depth": 1, "prompt": f"Why '{seed}' happen?"}] + [
        {"depth": d, "prompt": "Why does the upper-level cause exist?"} for d in range(2, 6)
    ]
    return {
        "technique": "5_whys",
        "seed": seed,
        "chain": chain,
        "root_cause": "Follow the why-chain to the final layer, identify the root cause, then record it as a node.",
    }


def _elicit_reverse(seed: str) -> dict:
    failure_modes = [
        {
            "mode": f"Fastest way to make '{seed}' fail completely",
            "mitigation_hint": "Invert into a required success condition.",
        },
        {
            "mode": "Implicit assumption breaks in reality",
            "mitigation_hint": "List assumptions and attach each to a validation.",
        },
        {
            "mode": "External dependency is not ready in time",
            "mitigation_hint": "Identify a fallback or reduce dependency.",
        },
    ]
    return {"technique": "reverse", "seed": seed, "failure_modes": failure_modes}


def _elicit_moscow(seed: str) -> dict:
    return {
        "technique": "moscow",
        "seed": seed,
        "must": [f"(Required for v1) core item for: {seed}"],
        "should": [],
        "could": [],
        "wont": [f"(Excluded from v1) deferred item for: {seed}"],
    }


def _elicit_first_principles(seed: str) -> dict:
    return {
        "technique": "first_principles",
        "seed": seed,
        "fundamentals": [
            f"Undeniable first-principle fact about '{seed}'",
            "Real physical/economic constraint (not a design convention)",
        ],
        "rebuilt_approach": "Rebuild a minimal solution from first principles without design assumptions.",
    }


def _elicit_comparable_products(seed: str, search_client) -> dict:
    res = web_search(seed, client=search_client)
    results = res.get("results") or []
    if res.get("error") or not results:
        return {
            "technique": "comparable_products",
            "seed": seed,
            "products": [
                {
                    "name": f"(Comparable product for {seed})",
                    "model": "Reference model to validate",
                    "relevance": "Fill when real data is available.",
                }
            ],
            "source": "model_knowledge",
            "evidence_source": "model_knowledge",
        }
    products = [
        {"name": r.get("title", ""), "model": r.get("snippet", ""), "relevance": f"Related to: {seed}"} for r in results
    ]
    return {
        "technique": "comparable_products",
        "seed": seed,
        "products": products,
        "source": "web_search",
        "evidence_source": "web",
    }


def _elicit_pre_mortem(seed: str) -> dict:
    return {
        "technique": "pre_mortem",
        "seed": seed,
        "premise": f"Imagine '{seed}' has already failed six months from now.",
        "failure_causes": [
            {"cause": "Most likely cause of failure", "prevention_hint": "Turn into a guarded precondition."},
            {"cause": "Second most likely cause of failure", "prevention_hint": "Turn into a guarded precondition."},
            {"cause": "Overlooked/silent cause of failure", "prevention_hint": "Turn into a monitored assumption."},
        ],
    }


def _elicit_tree_of_thought(seed: str) -> dict:
    return {
        "technique": "tree_of_thought",
        "seed": seed,
        "branches": [
            {"path": "Conservative approach", "outcome_hint": "Lower risk, slower/narrower payoff."},
            {"path": "Aggressive approach", "outcome_hint": "Higher risk, faster/broader payoff."},
            {"path": "Hybrid approach", "outcome_hint": "Combine strengths, watch for added complexity."},
        ],
        "evaluation_hint": "Score each branch against feasibility and value, then pick or merge.",
    }


def _elicit_socratic_questioning(seed: str) -> dict:
    return {
        "technique": "socratic_questioning",
        "seed": seed,
        "questions": [
            {"probe": f"What does '{seed}' actually mean in this context?", "targets": "clarification"},
            {"probe": "What evidence supports this being true?", "targets": "assumption"},
            {"probe": "What would change if this were false?", "targets": "implication"},
        ],
    }


def _elicit_challenge_assumptions(seed: str) -> dict:
    return {
        "technique": "challenge_assumptions",
        "seed": seed,
        "assumptions_to_challenge": [
            {
                "assumption": f"Implicit assumption behind '{seed}'",
                "counter_argument": "State the strongest case against it.",
                "revised_statement": "Rewrite the requirement so it holds even if the assumption is false.",
            },
        ],
    }


def elicit(technique: str, seed: str, *, search_client=None) -> dict:
    """Apply a BMAD elicitation technique to a seed, returning a structured frame.

    Reasoning techniques (5_whys/reverse/moscow/first_principles) return a deterministic frame for
    the agent to fill; comparable_products pulls real external knowledge via web_search and falls
    back to model knowledge when search is unavailable.
    """
    if technique not in ELICIT_TECHNIQUES:
        raise ValueError(f"unknown elicit technique {technique!r}; expected one of {ELICIT_TECHNIQUES}")
    if technique == "5_whys":
        return _elicit_5_whys(seed)
    if technique == "reverse":
        return _elicit_reverse(seed)
    if technique == "moscow":
        return _elicit_moscow(seed)
    if technique == "first_principles":
        return _elicit_first_principles(seed)
    if technique == "comparable_products":
        return _elicit_comparable_products(seed, search_client)
    if technique == "pre_mortem":
        return _elicit_pre_mortem(seed)
    if technique == "tree_of_thought":
        return _elicit_tree_of_thought(seed)
    if technique == "socratic_questioning":
        return _elicit_socratic_questioning(seed)
    return _elicit_challenge_assumptions(seed)


@tool("web_search")
async def web_search_tool(
    query: Annotated[str, "External knowledge search query."],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Search the web for external knowledge (comparable products, industry standards). Returns structured results.

    When no provider is configured or the call fails, returns an empty result with an error field — never
    interrupts the tool loop.
    """
    result = web_search(query)
    return Command(
        update={"messages": [ToolMessage(content=json.dumps(result, ensure_ascii=False), tool_call_id=tool_call_id)]}
    )


@tool("elicit")
async def elicit_tool(
    technique: Annotated[
        Literal[
            "5_whys",
            "reverse",
            "moscow",
            "first_principles",
            "comparable_products",
            "pre_mortem",
            "tree_of_thought",
            "socratic_questioning",
            "challenge_assumptions",
        ],
        "BMAD elicitation technique applied to the seed.",
    ],
    seed: Annotated[str, "Seed/topic to apply the technique to."],
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Apply a BMAD elicitation technique to a seed and return a structured frame to reason over and record as nodes.

    comparable_products fetches real external knowledge via web_search (falls back to model knowledge).
    Each successful call increments session_elicit_count so policy knows the cold start has been explored.
    """
    try:
        result = elicit(technique, seed)
    except ValueError as exc:
        return _recoverable_tool_update(
            RecoverableToolError(code="elicit_unknown_technique", message=str(exc), user_fixable=True),
            tool_call_id,
        )
    # Emit a DELTA (+1), not an absolute count: the channel uses an additive reducer so two elicits in
    # one turn accumulate correctly. Returning state+1 from both (the same pre-turn snapshot) would
    # either collide (no reducer) or double-count (absolute + add). _ = state kept for signature parity.
    _ = state
    return Command(
        update={
            "session_elicit_count": 1,
            "messages": [ToolMessage(content=json.dumps(result, ensure_ascii=False), tool_call_id=tool_call_id)],
        }
    )
