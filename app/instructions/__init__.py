"""Instruction layer — a policy + decision contract assembled from shared layers + a role overlay.

An *instruction* steers the model's policy (which mode/tool to pick, when to finalize), not the
per-turn payload. The harness owns schema and state; these files own judgment.

The contract is split into 10 responsibility layers (spec §5, §6, §13; addendum §9). Nine are shared
(role-agnostic); only Layer 2 is a per-role overlay. ``get_instruction`` assembles a single string —
the call site in ``analyze_node`` is unchanged:

    [01 system] + [layer 2 = role overlay] + [03 taxonomy .. 10 output]

Layer 3 (taxonomy) is rendered from ``section_schema`` so the section list never drifts from the
engine. Files live inside the app so they ship and version with the code, loaded once at startup.
"""
from pathlib import Path

from app.graphs.section_schema import SECTION_DESCRIPTIONS

# Default role when nothing else resolves — guarantees get_instruction never returns None, so the
# system prompt always carries the policy contract regardless of artifact_type / workflow_area.
_DEFAULT_ROLE = "business_analyst"

# artifact_type → role key. Every ArtifactType maps explicitly: discovery/analysis artifacts to the
# Business Analyst, delivery/requirement artifacts to the Product Manager.
ARTIFACT_ROLE_MAP: dict[str, str] = {
    # Business Analyst — discovery and problem framing
    "requirements":  "business_analyst",
    "domain_entity": "business_analyst",
    # Product Manager — prioritized, testable requirements and delivery breakdown
    "functional_requirement":     "product_manager",
    "non_functional_requirement": "product_manager",
    "use_case":                   "product_manager",
    "epic":                       "product_manager",
    "story":                      "product_manager",
    "acceptance_criteria":        "product_manager",
}

# workflow_area fallback
_WORKFLOW_AREA_MAP: dict[str, str] = {
    "product_analysis": "business_analyst",
    "requirements":     "product_manager",
}

# role key → overlay filename inside roles/
_ROLE_OVERLAY_FILE: dict[str, str] = {
    "business_analyst": "business-analyst.md",
    "product_manager":  "product-manager.md",
}

# Shared layers in assembly order. Layer 1 leads; the role overlay is inserted after it; layers
# 03–10 follow. (Layer 2 is the role overlay, not a file here.)
_LAYER_01 = "01-system-contract.md"
_SHARED_LAYERS_AFTER_ROLE = (
    "03-taxonomy-contract.md",
    "04-bmad-method.md",
    "05-decision-policy.md",
    "06-question-policy.md",
    "07-tool-policy.md",
    "08-critique-policy.md",
    "09-governance-policy.md",
    "10-output-contract.md",
)

_layer_cache: dict[str, str] = {}
_overlay_cache: dict[str, str] = {}
_assembled_cache: dict[str, str] = {}


def _render_taxonomy_sections() -> str:
    """The 7-section list, sourced from section_schema so it never drifts from the engine."""
    return "\n".join(f"- {section}: {desc}" for section, desc in SECTION_DESCRIPTIONS.items())


def load_instructions(base_path: Path | None = None) -> None:
    """Load and cache the shared layers and role overlays. Called once at startup."""
    _layer_cache.clear()
    _overlay_cache.clear()
    _assembled_cache.clear()
    base = base_path or Path(__file__).parent

    for filename in (_LAYER_01, *_SHARED_LAYERS_AFTER_ROLE):
        path = base / "layers" / filename
        if path.exists():
            text = path.read_text(encoding="utf-8").strip()
            # Layer 3 carries the section list rendered from section_schema (single source of truth).
            if filename == "03-taxonomy-contract.md":
                text = f"{text}\n{_render_taxonomy_sections()}"
            _layer_cache[filename] = text

    for role, filename in _ROLE_OVERLAY_FILE.items():
        path = base / "roles" / filename
        if path.exists():
            _overlay_cache[role] = path.read_text(encoding="utf-8").strip()


def role_overlay(role: str) -> str | None:
    """The Layer 2 overlay text for a role (used by callers and tests)."""
    return _overlay_cache.get(role)


def _assemble(role: str) -> str | None:
    overlay = _overlay_cache.get(role)
    if overlay is None:
        return None
    parts = [_layer_cache[_LAYER_01], overlay, *(_layer_cache[f] for f in _SHARED_LAYERS_AFTER_ROLE)]
    return "\n\n".join(parts)


def get_instruction(
    artifact_type: str,
    workflow_area: str,
    agent_role: str | None,
) -> str | None:
    """Return the assembled instruction for the resolved role.

    Priority: explicit agent_role > artifact_type map > workflow_area fallback > default role. Never
    returns None for a known role: the analyst always receives the policy contract, even for an
    artifact_type or workflow_area that is not explicitly mapped.
    """
    role = (
        agent_role
        or ARTIFACT_ROLE_MAP.get(artifact_type)
        or _WORKFLOW_AREA_MAP.get(workflow_area)
        or _DEFAULT_ROLE
    )
    if role not in _assembled_cache:
        # An explicit agent_role with no overlay file falls back to the default role rather than
        # leaving the analyst with no contract.
        assembled = _assemble(role) or _assemble(_DEFAULT_ROLE)
        if assembled is None:
            return None
        _assembled_cache[role] = assembled
    return _assembled_cache[role]


def loaded_roles() -> list[str]:
    """Return list of role keys successfully loaded. Used for startup logging."""
    return list(_overlay_cache.keys())
