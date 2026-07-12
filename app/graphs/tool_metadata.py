from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolMetadata:
    name: str
    readonly: bool
    interrupts: bool
    writes_db: bool
    policy: str | None = None
    can_run_with_interrupt: bool = False


TOOL_METADATA: dict[str, ToolMetadata] = {
    # Governed/read-only repository tools.
    "read_artifacts": ToolMetadata("read_artifacts", readonly=True, interrupts=False, writes_db=False, policy="allow"),
    "read_artifact_graph": ToolMetadata(
        "read_artifact_graph", readonly=True, interrupts=False, writes_db=False, policy="allow"
    ),
    "read_workflow_steps": ToolMetadata(
        "read_workflow_steps", readonly=True, interrupts=False, writes_db=False, policy="allow"
    ),
    "read_source_documents": ToolMetadata(
        "read_source_documents", readonly=True, interrupts=False, writes_db=False, policy="allow"
    ),
    "read_project_context": ToolMetadata(
        "read_project_context", readonly=True, interrupts=False, writes_db=False, policy="allow"
    ),
    # Governed write/checkpoint tools.
    "init_workflow_run": ToolMetadata(
        "init_workflow_run", readonly=False, interrupts=False, writes_db=True, policy="require_approval"
    ),
    "create_artifact": ToolMetadata(
        "create_artifact", readonly=False, interrupts=False, writes_db=True, policy="require_approval"
    ),
    "update_artifact": ToolMetadata(
        "update_artifact", readonly=False, interrupts=False, writes_db=True, policy="require_approval"
    ),
    "create_artifact_link": ToolMetadata(
        "create_artifact_link", readonly=False, interrupts=True, writes_db=True, policy="require_approval"
    ),
    "propose_retirement": ToolMetadata(
        "propose_retirement", readonly=False, interrupts=True, writes_db=True, policy="require_approval"
    ),
    "delete_artifact_link": ToolMetadata(
        "delete_artifact_link", readonly=False, interrupts=False, writes_db=True, policy="require_approval"
    ),
    "create_artifact_review": ToolMetadata(
        "create_artifact_review", readonly=False, interrupts=False, writes_db=True, policy="require_approval"
    ),
    "finalize": ToolMetadata(
        "finalize", readonly=False, interrupts=True, writes_db=True, policy="require_critique"
    ),
    # Native analyzer loop tools.
    "ask_user": ToolMetadata("ask_user", readonly=False, interrupts=True, writes_db=True),
    "respond": ToolMetadata("respond", readonly=False, interrupts=True, writes_db=True),
    "write_draft": ToolMetadata("write_draft", readonly=False, interrupts=True, writes_db=True),
    "confirm_intent": ToolMetadata("confirm_intent", readonly=False, interrupts=True, writes_db=True),
    "critique_note": ToolMetadata(
        "critique_note", readonly=False, interrupts=False, writes_db=False, can_run_with_interrupt=True
    ),
    "explore_note": ToolMetadata(
        "explore_note", readonly=False, interrupts=False, writes_db=False, can_run_with_interrupt=True
    ),
    "read_artifact": ToolMetadata("read_artifact", readonly=True, interrupts=False, writes_db=False),
    "run_critique": ToolMetadata("run_critique", readonly=False, interrupts=False, writes_db=False),
    "recommend_next_workflow": ToolMetadata(
        "recommend_next_workflow", readonly=False, interrupts=False, writes_db=True
    ),
    "run_readiness_check": ToolMetadata(
        "run_readiness_check", readonly=False, interrupts=False, writes_db=True
    ),
    "elicit": ToolMetadata("elicit", readonly=False, interrupts=False, writes_db=False),
    "web_search": ToolMetadata("web_search", readonly=True, interrupts=False, writes_db=False),
    "run_impact_analysis": ToolMetadata("run_impact_analysis", readonly=False, interrupts=False, writes_db=False),
}


def policy_table() -> dict[str, str]:
    return {name: metadata.policy for name, metadata in TOOL_METADATA.items() if metadata.policy is not None}


def interrupt_bearing_tools() -> frozenset[str]:
    return frozenset(name for name, metadata in TOOL_METADATA.items() if metadata.interrupts)


def side_effect_free_note_tools() -> frozenset[str]:
    return frozenset(name for name, metadata in TOOL_METADATA.items() if metadata.can_run_with_interrupt)
