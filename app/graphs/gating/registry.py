"""Registries for per-call and batch gating rules.

Rules are held in explicit registration order. Per-call rules are tagged
with the mode(s) they apply to; a rule applies to both modes only if it
is explicitly registered for both.
"""

from __future__ import annotations

from app.graphs.gating.rules import BatchRule, Mode, Rule


class Registry:
    def __init__(self) -> None:
        self._rules: list[tuple[Rule, tuple[Mode, ...]]] = []
        self._batch_rules: list[BatchRule] = []

    def register_rule(self, rule: Rule, modes: tuple[Mode, ...]) -> None:
        self._rules.append((rule, modes))

    def register_batch_rule(self, rule: BatchRule) -> None:
        self._batch_rules.append(rule)

    def rules_for(self, mode: Mode) -> list[Rule]:
        return [rule for rule, modes in self._rules if mode in modes]

    def has_rule(self, name: str, mode: Mode) -> bool:
        return any(rule.name == name and mode in modes for rule, modes in self._rules)

    def has_batch_rule(self, name: str) -> bool:
        return any(rule.name == name for rule in self._batch_rules)

    def batch_rules(self) -> list[BatchRule]:
        return list(self._batch_rules)

    def clear(self) -> None:
        self._rules.clear()
        self._batch_rules.clear()
