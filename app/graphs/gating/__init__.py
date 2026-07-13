"""Policy gating engine: per-call `check` and per-turn `check_batch`.

Standalone package, independent of app.graphs.policy.
"""

from app.graphs.gating.engine import (
    check,
    check_batch,
    is_batch_registered,
    is_registered,
    register_batch_rule,
    register_rule,
    reset,
)
from app.graphs.gating.rules import BatchRule, Mode, Rule
from app.graphs.gating.verdict import Verdict, VerdictKind

__all__ = [
    "Verdict",
    "VerdictKind",
    "Mode",
    "Rule",
    "BatchRule",
    "check",
    "check_batch",
    "register_rule",
    "register_batch_rule",
    "reset",
    "is_registered",
    "is_batch_registered",
]
