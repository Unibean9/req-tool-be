"""Declarative manifest for capabilities migrated to `CapabilityResolver`.

Metadata only — no logic. The resolver owns policy; a manifest entry only declares that a
capability id is eligible for the shadow/enforce pilot and its effect class, so a new capability
never has to be re-derived by hand from menu/dispatch/handler code once it is migrated. Do not add
evaluation logic here — keep the manifest a pure data declaration, never an executable god-object.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CapabilityManifestEntry:
    capability_id: str
    effect_class: str  # "read_only" | "mutating"
    presentation: str  # tool name as exposed to the model (identical to capability_id today)


# Read-only pilot: no artifact mutation, no HITL interrupt. The candidate set is run_critique,
# run_readiness_check, and recommend_next_workflow, verified against app/graphs/tool_metadata.py
# (none interrupt, none mutate the artifact/document graph; the two that write are best-effort
# AgentToolCall audit inserts that never gate the response).
READ_ONLY_PILOT_MANIFEST: tuple[CapabilityManifestEntry, ...] = (
    CapabilityManifestEntry("run_critique", "read_only", "run_critique"),
    CapabilityManifestEntry("run_readiness_check", "read_only", "run_readiness_check"),
    CapabilityManifestEntry("recommend_next_workflow", "read_only", "recommend_next_workflow"),
)

READ_ONLY_PILOT_CAPABILITIES: frozenset[str] = frozenset(entry.capability_id for entry in READ_ONLY_PILOT_MANIFEST)

# Declared for `CapabilityDecision.effect_class` classification only — never enforced by the
# resolver. write_draft/finalize stay on the legacy path 100% of the time, in every resolver mode,
# until their fenced command boundaries land. create_artifact_link/propose_retirement are
# the repository-side approval-proposal tools (interrupting + writes_db in tool_metadata.py).
MUTATING_CAPABILITIES: frozenset[str] = frozenset(
    {"write_draft", "finalize", "create_artifact_link", "propose_retirement"}
)
