from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any, TypeVar


class ApprovalRequired(Exception):
    def __init__(self, tool_name: str, args_snapshot: dict[str, Any]):
        self.tool_name = tool_name
        self.args_snapshot = args_snapshot
        super().__init__(f"Tool cần approval trước khi chạy: {tool_name}")


class GovernanceDenied(Exception):
    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        super().__init__(f"Tool bị policy từ chối: {tool_name}")


POLICY = {
    "read_artifacts": "allow",
    "read_artifact_graph": "allow",
    "read_workflow_steps": "allow",
    "read_source_documents": "allow",
    "read_project_context": "allow",
    "init_workflow_run": "require_approval",
    "create_artifact": "require_approval",
    "update_artifact": "require_approval",
    "create_artifact_link": "require_approval",
    "delete_artifact_link": "require_approval",
    "create_artifact_review": "require_approval",
    # finalize requires at least one run_critique round before it is offered (spec §15.1). This is
    # a documentation signal; the actual gate lives in agent_tools.get_available_tools.
    "finalize": "require_critique",
    # BMAD governance gates (addendum §18): three transitions need human approval; forcing the full
    # lifecycle is denied outright.
    "finalize_prd": "require_human_approval",
    "lock_scope": "require_human_approval",
    "accept_implementation_readiness": "require_human_approval",
    "force_full_bmad_lifecycle": "deny",
}


ARTIFACT_PREDECESSORS = {
    "brd": [],
    "vision_objectives": [],
    "problem_statement": ["vision_objectives"],
    "stakeholder_register": ["problem_statement"],
    "scope_capabilities": ["problem_statement"],
    "business_rules": ["scope_capabilities"],
    "constraints_assumptions": ["scope_capabilities"],
    "risks_issues": ["constraints_assumptions"],
    "prd": ["brd"],
    "domain_entity": ["brd"],
    "functional_requirement": ["brd"],
    "non_functional_requirement": ["brd"],
    "use_case": ["functional_requirement"],
    "acceptance_criteria": ["functional_requirement"],
    "sad": ["prd"],
    "component": ["domain_entity"],
    "interface": ["component"],
    "tech_decision": ["component"],
    "epic": ["functional_requirement"],
    "story": ["epic"],
}


def ancestor_types(artifact_type: str) -> list[str]:
    """All transitive predecessors of an artifact type (deduped, nearest-first).

    Walks ARTIFACT_PREDECESSORS recursively so a derived type sees its full
    upstream provenance regardless of how each entry is declared — some list the
    whole chain, others only the direct parent. Dedup keeps the result (and any
    prompt built from it) token-light: each ancestor type appears at most once.
    """
    seen: list[str] = []
    visited: set[str] = set()
    queue = list(ARTIFACT_PREDECESSORS.get(artifact_type, []))
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        seen.append(current)
        queue.extend(ARTIFACT_PREDECESSORS.get(current, []))
    return seen


T = TypeVar("T")


def governed[T](fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
    @wraps(fn)
    async def wrapper(*args: Any, context: dict[str, Any] | None = None, **kwargs: Any) -> T:
        tool_name = fn.__name__
        rule = POLICY.get(tool_name, "deny")
        if rule == "deny":
            raise GovernanceDenied(tool_name)

        context = context or {}
        if rule in ("require_approval", "require_human_approval"):
            if tool_name == "init_workflow_run" and context.get("workflow_area") != "orchestrator":
                raise GovernanceDenied(tool_name)
            if tool_name == "create_artifact":
                allowed = context.get("allowed_types", [])
                if kwargs.get("artifact_type") not in allowed:
                    raise GovernanceDenied(tool_name)
            raise ApprovalRequired(tool_name, dict(kwargs))

        return await fn(*args, **kwargs)

    return wrapper


# BMAD governance gates (addendum §18). These are checkpoints, not loop tools: calling one runs the
# @governed wrapper, which raises ApprovalRequired (human-approval gates) or GovernanceDenied (force
# full lifecycle) per the POLICY rule above — the body is only reached if a rule ever allows it.

@governed
async def finalize_prd(**kwargs: Any) -> None:  # noqa: ARG001
    return None


@governed
async def lock_scope(**kwargs: Any) -> None:  # noqa: ARG001
    return None


@governed
async def accept_implementation_readiness(**kwargs: Any) -> None:  # noqa: ARG001
    return None


@governed
async def force_full_bmad_lifecycle(**kwargs: Any) -> None:  # noqa: ARG001
    return None
