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
}


ARTIFACT_PREDECESSORS = {
    "research_output": [],
    "intent": [],
    "problem": ["intent"],
    "goal": ["intent", "problem"],
    "stakeholder": ["intent", "problem"],
    "capability": ["intent", "problem", "goal"],
    "domain_entity": ["capability"],
    "business_rule": ["capability"],
    "constraint": ["intent", "problem"],
    "assumption": ["intent", "problem"],
    "risk": ["intent", "problem"],
    "open_question": ["intent", "problem"],
    "functional_requirement": ["capability"],
    "non_functional_requirement": ["capability"],
    "use_case": ["functional_requirement"],
    "epic": ["functional_requirement"],
    "story": ["epic"],
    "acceptance_criteria": ["story"],
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
        if rule == "require_approval":
            if tool_name == "init_workflow_run" and context.get("workflow_area") != "orchestrator":
                raise GovernanceDenied(tool_name)
            if tool_name == "create_artifact":
                allowed = context.get("allowed_types", [])
                if kwargs.get("artifact_type") not in allowed:
                    raise GovernanceDenied(tool_name)
            raise ApprovalRequired(tool_name, dict(kwargs))

        return await fn(*args, **kwargs)

    return wrapper
