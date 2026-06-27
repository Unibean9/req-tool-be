"""Production judge logic for the in-loop run_critique tool (spec §6.6, §9.3).

Imports only app.graphs.rubric — never nodes/agent_tools (no cycle) and never tests/ (production
must not depend on test infrastructure). tests/eval may import from here, not the reverse.

run_critique targets a draft body with one critique `mode` and returns a compact report
{mode, score, findings, suggestions}. When no LLM client is configured the judge degrades to a
well-defined empty report instead of raising, so the tool-loop never crashes on a missing key.
"""

from typing import Any

from app.graphs.rubric import render_criteria_block

# Critique angles the analyst may request (spec §6.6).
CRITIQUE_MODES: tuple[str, ...] = (
    "clarity",
    "completeness",
    "consistency",
    "feasibility",
    "testability",
    "traceability",
    "six_hats",
    "swot",
    "risk_review",
)
_DEFAULT_MODE = "completeness"

# Short focus per mode, embedded in the judge prompt so the score reflects the requested angle.
_MODE_FOCUS: dict[str, str] = {
    "clarity": "Clarity and lack of ambiguity in each statement.",
    "completeness": "Completeness: whether actor, condition, and expected result are missing.",
    "consistency": "Internal consistency and consistency with other sections.",
    "feasibility": "Feasibility within technical, time, and resource constraints.",
    "testability": "Testability: measurement criteria and test method.",
    "traceability": "Traceability backward to sources and forward to child artifacts.",
    "six_hats": "Review through six thinking hats (facts, emotions, risks, benefits, creativity, orchestration).",
    "swot": "Strengths, weaknesses, opportunities, threats.",
    "risk_review": "Latent risks, likelihood, and impact level.",
}

JUDGE_SCHEMA: dict[str, Any] = {
    "name": "critique_result",
    "schema": {
        "type": "object",
        "properties": {
            "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "findings": {"type": "array", "items": {"type": "string"}},
            "suggestions": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["score", "findings", "suggestions"],
    },
}

_JUDGE_SYSTEM = (
    "You are a requirements engineering expert. Critique the artifact from the specified angle, "
    "score 0.0-1.0, and list findings (specific weaknesses) and suggestions (improvements) in the user's locale."
)


def _normalize_mode(mode: str) -> str:
    return mode if mode in CRITIQUE_MODES else _DEFAULT_MODE


def _build_judge_prompt(body: str, mode: str) -> str:
    return (
        f"Critique the artifact below, focusing on angle '{mode}': {_MODE_FOCUS.get(mode, '')}\n\n"
        f"REFERENCE RUBRIC:\n{render_criteria_block()}\n\n"
        f"ARTIFACT:\n{body or '(empty)'}"
    )


async def _invoke_judge(body: str, mode: str, llm_client: Any = None) -> dict[str, Any]:
    """Critique `body` along `mode`. Degrades to an empty report when no LLM client is configured."""
    mode = _normalize_mode(mode)
    if llm_client is None:
        return {"mode": mode, "score": 0.0, "findings": [], "suggestions": ["no_llm_client"]}

    try:
        result, _usage = await llm_client.generate(
            messages=[{"role": "user", "content": _build_judge_prompt(body, mode)}],
            system=_JUDGE_SYSTEM,
            # Verbose locale critiques (e.g. Vietnamese) run ~2x tokens; an undersized budget
            # truncates the JSON mid-string and makes it unparseable.
            max_tokens=4096,
            response_format=JUDGE_SCHEMA,
        )
    except ValueError:
        # Per this module's contract the judge must never crash the tool-loop. A parse failure
        # (truncation, prose-wrapping the provider's instruction-only schema could not prevent)
        # degrades to an empty report so the turn proceeds instead of failing.
        return {"mode": mode, "score": 0.0, "findings": [], "suggestions": ["judge_unparseable"]}
    if not isinstance(result, dict):
        return {"mode": mode, "score": 0.0, "findings": [], "suggestions": []}
    return {
        "mode": mode,
        "score": float(result.get("score", 0.0) or 0.0),
        "findings": list(result.get("findings", []) or []),
        "suggestions": list(result.get("suggestions", []) or []),
    }
