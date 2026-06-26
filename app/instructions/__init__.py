"""Instruction layer — a policy + decision contract assembled from shared layers + a role overlay.

An *instruction* steers the model's policy (which mode/tool to pick, when to finalize), not the
per-turn payload. The harness owns schema and state; these files own judgment.

The contract is split into 10 responsibility layers (spec §5, §6, §13; addendum §9). Nine are shared
(role-agnostic); only Layer 2 is a per-role overlay. ``get_instruction`` assembles a single string —
the call site in ``analyze_node`` is unchanged:

    [01 system] + [layer 2 = role overlay] + [03 taxonomy .. 10 output]

Layer 3 (taxonomy) is rendered from the document registry so the item list never drifts from the
engine. Files live inside the app so they ship and version with the code, loaded once at startup.
"""
from pathlib import Path

# Default role when nothing else resolves — guarantees get_instruction never returns None, so the
# system prompt always carries the policy contract regardless of artifact_type / workflow_area.
_DEFAULT_ROLE = "business_analyst"

# artifact_type → role key. Every ArtifactType maps explicitly: discovery/analysis artifacts to the
# Business Analyst, delivery/requirement artifacts to the Product Manager.
ARTIFACT_ROLE_MAP: dict[str, str] = {
    # Business Analyst — discovery and problem framing
    "brd": "business_analyst",
    "vision_objectives": "business_analyst",
    "problem_statement": "business_analyst",
    "stakeholder_register": "business_analyst",
    "scope_capabilities": "business_analyst",
    "business_rules": "business_analyst",
    "constraints_assumptions": "business_analyst",
    "risks_issues": "business_analyst",
    # Product Manager — prioritized, testable requirements and delivery breakdown
    "prd":                        "product_manager",
    "functional_requirement":     "product_manager",
    "non_functional_requirement": "product_manager",
    "use_case":                   "product_manager",
    "acceptance_criteria":        "product_manager",
    # Architecture items currently reuse the product-manager overlay until a dedicated role lands.
    "sad":            "product_manager",
    "domain_entity":  "product_manager",
    "component":      "product_manager",
    "interface":      "product_manager",
    "tech_decision":  "product_manager",
    "epic":           "product_manager",
    "story":          "product_manager",
}

# workflow_area fallback
_WORKFLOW_AREA_MAP: dict[str, str] = {
    "product_analysis": "business_analyst",
    "prd":              "product_manager",
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
# Cache key: (role, has_draft). has_draft=None means context was not passed (full instruction set).
_assembled_cache: dict[tuple[str, bool | None], str] = {}


def load_instructions(base_path: Path | None = None) -> None:
    """Load and cache the shared layers and role overlays. Called once at startup."""
    _layer_cache.clear()
    _overlay_cache.clear()
    _assembled_cache.clear()  # type: ignore[attr-defined]
    base = base_path or Path(__file__).parent

    for filename in (_LAYER_01, *_SHARED_LAYERS_AFTER_ROLE):
        path = base / "layers" / filename
        if path.exists():
            _layer_cache[filename] = path.read_text(encoding="utf-8").strip()

    for role, filename in _ROLE_OVERLAY_FILE.items():
        path = base / "roles" / filename
        if path.exists():
            _overlay_cache[role] = path.read_text(encoding="utf-8").strip()


def role_overlay(role: str) -> str | None:
    """The Layer 2 overlay text for a role (used by callers and tests)."""
    return _overlay_cache.get(role)


_DRAFT_SKIP_LAYERS: frozenset[str] = frozenset({
    "08-critique-policy.md",
    "09-governance-policy.md",
    "10-output-contract.md",
})


def _assemble(role: str, context: dict | None = None) -> str | None:
    overlay = _overlay_cache.get(role)
    if overlay is None:
        return None
    has_draft: bool | None = context.get("has_draft") if context is not None else None
    skip = _DRAFT_SKIP_LAYERS if has_draft is False else frozenset()
    layers_after_role = [
        _layer_cache[f] for f in _SHARED_LAYERS_AFTER_ROLE
        if f not in skip and f in _layer_cache
    ]
    parts = [_layer_cache[_LAYER_01], overlay, *layers_after_role]
    return "\n\n".join(parts)


def get_instruction(
    artifact_type: str,
    workflow_area: str,
    agent_role: str | None,
    context: dict | None = None,
) -> str | None:
    """Return the assembled instruction for the resolved role.

    Priority: explicit agent_role > artifact_type map > workflow_area fallback > default role. Never
    returns None for a known role: the analyst always receives the policy contract, even for an
    artifact_type or workflow_area that is not explicitly mapped.

    context: optional dict; when context={"has_draft": False}, layers 08/09/10 are skipped
    (no critique/governance/output-contract policy needed before a draft exists). context=None
    or has_draft=True keeps all layers. Cache key is (role, has_draft) to prevent collision.
    """
    role = (
        agent_role
        or ARTIFACT_ROLE_MAP.get(artifact_type)
        or _WORKFLOW_AREA_MAP.get(workflow_area)
        or _DEFAULT_ROLE
    )
    has_draft: bool | None = context.get("has_draft") if context is not None else None
    cache_key = (role, has_draft)
    if cache_key not in _assembled_cache:
        # An explicit agent_role with no overlay file falls back to the default role rather than
        # leaving the analyst with no contract. Both calls must receive context for consistent filtering.
        assembled = _assemble(role, context) or _assemble(_DEFAULT_ROLE, context)
        if assembled is None:
            return None
        _assembled_cache[cache_key] = assembled
    return _assembled_cache[cache_key]


def loaded_roles() -> list[str]:
    """Return list of role keys successfully loaded. Used for startup logging."""
    return list(_overlay_cache.keys())
