from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any, TypeVar


class ApprovalRequired(Exception):
    def __init__(self, tool_name: str, args_snapshot: dict[str, Any]):
        self.tool_name = tool_name
        self.args_snapshot = args_snapshot
        super().__init__(f"Tool requires approval before running: {tool_name}")


class GovernanceDenied(Exception):
    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        super().__init__(f"Tool was rejected by policy: {tool_name}")


# DOCUMENTATION-ONLY for tool-loop tools: the enforcement authority for what the loop may run is
# the session-phase menu (app/graphs/session_phase.py) applied inside agent_tools.get_available_tools
# and the post-LLM gate (analysis/tool_gating.py). Keys here either back a genuine @governed
# approval checkpoint (init_workflow_run/create_artifact/... stubs, finalize_prd, lock_scope) or
# document enum-era tool names the loop no longer calls.
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
    # BMAD governance gates (addendum §18): phase completion and scope lock need human approval.
    "finalize_prd": "require_human_approval",
    "lock_scope": "require_human_approval",
}


# The intentional, acyclic artifact chain. Advisory only: shapes
# ancestor_types() prompt-context loading and the finalize predecessor check —
# never a hard gate. Event Storming sits between PRD and SAD: SAD depends on
# event_storming instead of prd directly, and the ES item types form their own
# flow chain (use_case -> actor_command -> domain_event -> policy/aggregate).
# interface and tech_decision are intentional pipeline leaves (nothing consumes
# them); stakeholder_register and tech_stack are interim leaves until domain
# work rewires them; policy and aggregate are intentional ES leaves.
ARTIFACT_PREDECESSORS = {
    "brd": [],
    "problem_statement": [],
    "vision_objectives": ["problem_statement"],
    "stakeholder_register": ["problem_statement"],
    "scope_capabilities": ["vision_objectives"],
    "business_rules": ["scope_capabilities"],
    "constraints_assumptions": ["scope_capabilities"],
    "prd": ["brd"],
    "use_case": ["scope_capabilities"],
    "functional_requirement": ["use_case", "business_rules"],
    "non_functional_requirement": ["constraints_assumptions"],
    "event_storming": ["prd"],
    "sad": ["event_storming"],
    "tech_stack": ["constraints_assumptions", "non_functional_requirement"],
    "domain_entity": ["functional_requirement"],
    "component": ["domain_entity", "functional_requirement"],
    "interface": ["component"],
    "tech_decision": ["component", "non_functional_requirement"],
    "actor_command": ["use_case"],
    "domain_event": ["actor_command"],
    "policy": ["domain_event"],
    "aggregate": ["domain_event", "actor_command"],
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
# @governed wrapper, which raises ApprovalRequired per the POLICY rule above.


@governed
async def finalize_prd(**kwargs: Any) -> None:  # noqa: ARG001
    return None


@governed
async def lock_scope(**kwargs: Any) -> None:  # noqa: ARG001
    return None
