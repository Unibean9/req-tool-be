"""Gating engine: evaluates registered rules against tool calls.

Two entrypoints:
- `check`: per-call gate, evaluated once per tool call.
- `check_batch`: batch/selection rule, evaluated once per turn over the
  whole list of tool calls.
"""

from __future__ import annotations

from typing import Any

from app.graphs.gating.registry import Registry
from app.graphs.gating.rules import BatchRule, Mode, Rule
from app.graphs.gating.verdict import Verdict

_registry = Registry()


def register_rule(rule: Rule, modes: tuple[Mode, ...]) -> None:
    _registry.register_rule(rule, modes)


def register_batch_rule(rule: BatchRule) -> None:
    _registry.register_batch_rule(rule)


def reset() -> None:
    """Clear all registered rules. Intended for test isolation."""
    _registry.clear()


def is_registered(name: str, mode: Mode) -> bool:
    """Whether a rule with this name is registered for this mode.

    Lets callers that share this process-global registry with tests (which may
    call `reset()` for isolation) re-register idempotently instead of assuming
    import-time registration survives the whole test session.
    """
    return _registry.has_rule(name, mode)


def is_batch_registered(name: str) -> bool:
    """Whether a batch rule with this name is already registered.

    Same self-healing motivation as `is_registered` above, for batch rules
    (which aren't keyed by `Mode`).
    """
    return _registry.has_batch_rule(name)


def check(tool_call: Any, state: Any, mode: Mode) -> Verdict:
    for rule in _registry.rules_for(mode):
        verdict = rule.evaluate(tool_call, state)
        if not verdict.is_allow:
            return verdict
    return Verdict.allow()


def check_batch(tool_calls: list[Any], state: Any) -> list[Any]:
    selection = tool_calls
    for rule in _registry.batch_rules():
        selection = rule.evaluate(selection, state)
    return selection
