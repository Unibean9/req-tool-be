"""Verdict type for the gating engine.

A Verdict represents the outcome of a gate check: allow, deny (with a
reason), or needs_human (optionally with a reason).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class VerdictKind(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    NEEDS_HUMAN = "needs_human"


@dataclass(frozen=True)
class Verdict:
    kind: VerdictKind
    reason: str | None = None

    @staticmethod
    def allow() -> "Verdict":
        return Verdict(kind=VerdictKind.ALLOW)

    @staticmethod
    def deny(reason: str) -> "Verdict":
        return Verdict(kind=VerdictKind.DENY, reason=reason)

    @staticmethod
    def needs_human(reason: str | None = None) -> "Verdict":
        return Verdict(kind=VerdictKind.NEEDS_HUMAN, reason=reason)

    @property
    def is_allow(self) -> bool:
        return self.kind is VerdictKind.ALLOW
