# Phase 1 Evidence: Diagnosis foundation

Execution path: inline (main controller) — single-module-family change with clear acceptance
criteria; controller already held full context from plan.md/context.md read during preflight.

## Changes

- `app/graphs/state.py`: added `thinking_mode: str | None` and `diagnosis_signal: dict[str, Any] | None`
  to `WorkflowState`; added both with `None` defaults to `build_initial_workflow_state`.
- `app/config.py`: added `enable_adaptive_diagnosis: bool = True`.
- `app/graphs/nodes.py`: added `_THINKING_MODES`, `_LOW_COVERAGE_RATIO`, `_COVERAGE_SCORES`, and
  `_diagnose_section()` (pure, no LLM call) near `_should_run_completeness_sweep`; wired into
  `orchestrator_node` behind the `enable_adaptive_diagnosis` flag, merged into the returned
  `update` dict without touching the existing parked-question/completeness-sweep return values.
- `tests/unit/test_orchestrator.py`: added 4 tests (diagnosis present every turn, high vs. low
  risk classification, single-weak-signal non-escalation, kill-switch no-op).

## Verification

`python -m pytest tests/unit/test_orchestrator.py -q` -> 7 passed (3 pre-existing + 4 new),
0 modified assertions in the pre-existing 3 tests.

## Success Criteria check

- [x] Every orchestrator call returns `thinking_mode` + `diagnosis_signal`.
- [x] Low-coverage + failed-quality-gate classified higher risk than well-covered/passing, via test.
- [x] 3 pre-existing orchestrator tests pass unmodified.
- [x] No LLM client invoked anywhere in this phase's code path.
- [x] Risk threshold is a named conjunction (low_coverage AND (quality_gate_failed OR sparse_draft));
      single weak signal test proves no escalation.
- [x] `enable_adaptive_diagnosis = False` makes the diagnosis step a no-op, verified by test.

Status: DONE
