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

# role key → filename inside prompts dir
_ROLE_PROMPT_FILE: dict[str, str] = {
    "business_analyst": "business-analyst.md",
    "product_manager":  "product-manager.md",
}

_CACHE: dict[str, str] = {}


def load_personas(base_path: Path | None) -> None:
    """Load and cache prompt files from base_path. Called once at startup. Safe with None."""
    _CACHE.clear()
    if base_path is None:
        return
    for role, filename in _ROLE_PROMPT_FILE.items():
        path = base_path / filename
        if path.exists():
            _CACHE[role] = path.read_text(encoding="utf-8")


def get_persona(
    artifact_type: str,
    workflow_area: str,
    agent_role: str | None,
) -> str | None:
    """
    Return cached prompt content for the resolved role, or None if not found.
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
