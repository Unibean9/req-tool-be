"""Instruction layer — role-scoped decision frames for the analyst.

An *instruction* steers the model's policy (which mode/tool to pick, when to finalize),
not the per-turn payload. The harness owns schema and state (TOOL_SELECTION_SCHEMA plus
the prompt assembled from runtime context in analyze_node); these files own judgment. They
live inside the app so they ship and version with the code, and are loaded once at startup.
"""
from pathlib import Path

# artifact_type → role key
ARTIFACT_ROLE_MAP: dict[str, str] = {
    # Phase 1 — Business Analyst
    "intent":       "business_analyst",
    "problem":      "business_analyst",
    "stakeholder":  "business_analyst",
    # Phase 2 — Product Manager
    "goal":         "product_manager",
    "feature":      "product_manager",
    "user_story":   "product_manager",
    "requirement":  "product_manager",
}

# workflow_area fallback
_WORKFLOW_AREA_MAP: dict[str, str] = {
    "product_analysis": "business_analyst",
    "requirements":     "product_manager",
}

# role key → filename inside this package
_ROLE_PROMPT_FILE: dict[str, str] = {
    "business_analyst": "business-analyst.md",
    "product_manager":  "product-manager.md",
}

_CACHE: dict[str, str] = {}


def load_instructions(base_path: Path | None = None) -> None:
    """Load and cache instruction files. Called once at startup.

    Defaults to this package's own directory; an override is accepted for tests.
    """
    _CACHE.clear()
    base = base_path or Path(__file__).parent
    for role, filename in _ROLE_PROMPT_FILE.items():
        path = base / filename
        if path.exists():
            _CACHE[role] = path.read_text(encoding="utf-8")


def get_instruction(
    artifact_type: str,
    workflow_area: str,
    agent_role: str | None,
) -> str | None:
    """
    Return cached instruction content for the resolved role, or None if not found.
    Priority: explicit agent_role > artifact_type map > workflow_area fallback.
    """
    role = (
        agent_role
        or ARTIFACT_ROLE_MAP.get(artifact_type)
        or _WORKFLOW_AREA_MAP.get(workflow_area)
    )
    return _CACHE.get(role) if role else None


def loaded_roles() -> list[str]:
    """Return list of role keys successfully loaded. Used for startup logging."""
    return list(_CACHE.keys())
