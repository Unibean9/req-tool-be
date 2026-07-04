"""Rubric for scoring requirements-artifact quality.

Pure Python — NO LLM or DB dependency. Defines the criteria from
ISO/IEC/IEEE 29148 (6 quality characteristics) plus INVEST (user story) and
SMART (goal). Each criterion holds `name`, `description` and `guidance`
(instructions for the judge when scoring 0.0–1.0).
"""

# key → {"name", "description", "guidance"}
RUBRIC_CRITERIA: dict[str, dict[str, str]] = {
    # --- 6 ISO/IEC/IEEE 29148 quality characteristics ---
    "unambiguous": {
        "name": "Unambiguous",
        "description": (
            "Statement has only one interpretation and avoids weasel words "
            "(fast, easy to use, optimized, friendly)."
        ),
        "guidance": (
            "Low score if it contains weasel words or ambiguous phrasing; "
            "high score if measurable and specific."
        ),
    },
    "verifiable": {
        "name": "Verifiable",
        "description": "Can be tested/measured to confirm the artifact is satisfied.",
        "guidance": "High score if it has measurement criteria, thresholds, or clear test method.",
    },
    "complete": {
        "name": "Complete",
        "description": "Covers necessary information without important gaps.",
        "guidance": "Low score if actor, condition, or expected result is missing.",
    },
    "consistent": {
        "name": "Consistent",
        "description": "No internal contradictions and no conflict with other artifacts.",
        "guidance": "Low score if statements contradict themselves or duplicate noisily.",
    },
    "traceable": {
        "name": "Traceable",
        "description": "Can link backward to sources (intent/problem/goal) and forward to child artifacts.",
        "guidance": "High score if it states rationale/source and connects to parent goals.",
    },
    "feasible": {
        "name": "Feasible",
        "description": "Achievable within reasonable technical, time, and resource constraints.",
        "guidance": "Low score if unrealistic or beyond known constraints.",
    },
    # --- INVEST (user story) + SMART (goal) — applied per artifact_type ---
    "invest": {
        "name": "INVEST (user story)",
        "description": "Independent, Negotiable, Valuable, Estimable, Small, Testable.",
        "guidance": "Score only for story/epic artifacts; return null if not applicable.",
    },
    "smart": {
        "name": "SMART (goal)",
        "description": "Specific, Measurable, Achievable, Relevant, Time-bound.",
        "guidance": "Score only for goal artifacts; return null if not applicable.",
    },
    # --- 3 business-specific dimensions (spec §12), aligned to the 7-section taxonomy ---
    "business_alignment": {
        "name": "Business alignment",
        "description": "Whether the artifact supports stated business vision and objectives.",
        "guidance": "High score if clearly tied to vision_objectives; low if off-goal.",
    },
    "risk_awareness": {
        "name": "Risk awareness",
        "description": "Whether related risks and issues have been surfaced.",
        "guidance": "High score if risks are in constraints_assumptions and reflect collected structured risks.",
    },
    "scope_control": {
        "name": "Scope control",
        "description": "Whether in-scope/out-of-scope boundaries are clear.",
        "guidance": "High score if scope_capabilities states both in-scope and out-of-scope; low if vague.",
    },
}


def render_criteria_block() -> str:
    """Render the rubric as a text block to embed in the judge prompt."""
    lines = []
    for key, spec in RUBRIC_CRITERIA.items():
        lines.append(f"- {key} ({spec['name']}): {spec['description']} {spec['guidance']}")
    return "\n".join(lines)
