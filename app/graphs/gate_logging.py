"""Structured, greppable logging for every gate decision in the graph.

One line per decision, `gate=<name> verdict=<v> ...` key=value pairs. Pass-like verdicts log at DEBUG
to keep default log volume flat; failures / degradations log at INFO. Leaf module — imports only
stdlib `logging` so it is safe to import from nodes, agent_tools, critique, and services.
"""

import logging

logger = logging.getLogger(__name__)

# Verdicts that represent a healthy / non-degraded outcome. Everything else logs at INFO.
_BENIGN_VERDICTS = frozenset({"pass", "open", "ready", "low", "accepted", "allowed", "sufficient"})


def log_gate_decision(
    gate: str,
    verdict: str,
    *,
    score: float | None = None,
    reason: str | None = None,
    session_id: str | None = None,
    extra: dict | None = None,
) -> None:
    level = logging.DEBUG if verdict in _BENIGN_VERDICTS else logging.INFO
    parts = [f"gate={gate}", f"verdict={verdict}"]
    if score is not None:
        parts.append(f"score={score:.2f}")
    if session_id:
        parts.append(f"session_id={session_id}")
    if reason:
        parts.append(f"reason={reason!r}")
    for key, value in (extra or {}).items():
        parts.append(f"{key}={value!r}")
    logger.log(level, " ".join(parts))
