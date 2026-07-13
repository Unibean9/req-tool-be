"""Unit tests for the standalone policy gating engine.

Entirely synthetic: uses plain dicts for tool_call/state and small fake
Rule/BatchRule implementations. Does not import real agent state/tools.
"""

import pytest

from app.graphs.gating import Mode, Verdict, VerdictKind, check, check_batch, reset
from app.graphs.gating.engine import register_batch_rule, register_rule


@pytest.fixture(autouse=True)
def _clean_registry():
    reset()
    yield
    reset()


class FakeRule:
    def __init__(self, name, verdict, side_effecting=False, on_call=None):
        self.name = name
        self.side_effecting = side_effecting
        self._verdict = verdict
        self._on_call = on_call
        self.call_count = 0

    def evaluate(self, tool_call, state):
        self.call_count += 1
        if self._on_call:
            self._on_call()
        return self._verdict


class RaisingRule:
    name = "raising"
    side_effecting = False

    def evaluate(self, tool_call, state):
        raise AssertionError("this rule must not be evaluated")


class FakeBatchRule:
    def __init__(self, name, fn):
        self.name = name
        self._fn = fn

    def evaluate(self, tool_calls, state):
        return self._fn(tool_calls, state)


def test_empty_registry_allows():
    verdict = check({"name": "some_tool"}, {}, Mode.MENU)
    assert verdict.kind is VerdictKind.ALLOW


def test_earlier_deny_short_circuits_later_rule():
    denying = FakeRule("denier", Verdict.deny("nope"))
    register_rule(denying, (Mode.MENU,))
    register_rule(RaisingRule(), (Mode.MENU,))

    verdict = check({"name": "t"}, {}, Mode.MENU)

    assert verdict.kind is VerdictKind.DENY
    assert verdict.reason == "nope"


def test_needs_human_passes_through_unchanged():
    register_rule(FakeRule("needs_human_rule", Verdict.needs_human("ask a person")), (Mode.MENU,))

    verdict = check({"name": "t"}, {}, Mode.MENU)

    assert verdict.kind is VerdictKind.NEEDS_HUMAN
    assert verdict.reason == "ask a person"


def test_mode_scoping_filters_rules():
    dispatch_only = FakeRule("dispatch_only", Verdict.deny("should not fire in menu"))
    register_rule(dispatch_only, (Mode.DISPATCH,))

    verdict = check({"name": "t"}, {}, Mode.MENU)

    assert verdict.kind is VerdictKind.ALLOW
    assert dispatch_only.call_count == 0


def test_side_effecting_rule_evaluated_exactly_once():
    side_effecting = FakeRule("logger", Verdict.allow(), side_effecting=True)
    register_rule(side_effecting, (Mode.MENU,))

    check({"name": "t"}, {}, Mode.MENU)

    assert side_effecting.call_count == 1


def test_check_batch_no_rules_returns_input_unchanged():
    tool_calls = [{"name": "a"}, {"name": "b"}]

    result = check_batch(tool_calls, {})

    assert result == tool_calls


def test_check_batch_filters_selection():
    tool_calls = [{"name": "a"}, {"name": "b"}]
    drop_b = FakeBatchRule("drop_b", lambda calls, _state: [c for c in calls if c["name"] != "b"])
    register_batch_rule(drop_b)

    result = check_batch(tool_calls, {})

    assert result == [{"name": "a"}]


def test_check_batch_chains_rules_in_order():
    tool_calls = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
    drop_first = FakeBatchRule("drop_first", lambda calls, _state: calls[1:])
    reverse = FakeBatchRule("reverse", lambda calls, _state: list(reversed(calls)))
    register_batch_rule(drop_first)
    register_batch_rule(reverse)

    result = check_batch(tool_calls, {})

    assert result == [{"name": "c"}, {"name": "b"}]
