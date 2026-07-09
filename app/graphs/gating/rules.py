"""Rule protocols and mode vocabulary for the gating engine."""

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol

from app.graphs.gating.verdict import Verdict


class Mode(str, Enum):
    """When a per-call rule is evaluated."""

    MENU = "menu"
    DISPATCH = "dispatch"


class Rule(Protocol):
    """A per-call gate rule.

    Evaluated once per tool call, in the mode(s) it is explicitly
    registered for. Rules that perform observable side effects must set
    `side_effecting = True` so the engine never invokes them more times
    than the explicit registration implies.
    """

    name: str
    side_effecting: bool

    def evaluate(self, tool_call: Any, state: Any) -> Verdict:
        ...


class BatchRule(Protocol):
    """A batch/selection rule evaluated once per turn over the whole
    list of tool calls.
    """

    name: str

    def evaluate(self, tool_calls: list[Any], state: Any) -> list[Any]:
        ...
