# Phase 4 Evidence: Judge escalation + stuck-loop/critique-cap escalation behavior

Execution path: inline.

## Changes

- `app/config.py`: added `max_diagnosis_judge_calls: int = 1`, sibling to `max_critique_rounds`,
  under the existing `enable_adaptive_diagnosis` comment block.
- `app/graphs/agent_tools.py`: added `DIAGNOSIS_JUDGE_CALLS_MAX = settings.max_diagnosis_judge_calls`,
  sibling to `CRITIQUE_ROUNDS_MAX`, as its own independent counter (not shared with the
  critique-round budget).
- `app/graphs/state.py`: added `diagnosis_judge_calls_used: int` plain state field (session-lifetime
  counter) plus its `0` default in `build_initial_workflow_state()`.
- `app/graphs/nodes.py`:
  - `_diagnosis_llm_client(config)`: tolerant LLM-client extraction for `orchestrator_node` (unlike
    `analyze_node`'s strict access, `orchestrator_node` may run with `config=None`/`{}` per the
    existing test suite).
  - `_apply_judge_escalation(diagnosis, state, config)`: low-risk -> `"not_needed"` (judge never
    invoked); high-risk + budget exhausted -> `"escalation_skipped_budget"` (distinct signal, no
    invocation, counter untouched); high-risk + budget available -> one `_invoke_judge(..., mode=
    "risk_review", ...)` call, counter incremented, `"escalated"`. Uses **local imports** for
    `DIAGNOSIS_JUDGE_CALLS_MAX` (from `agent_tools`) and `_invoke_judge` (from `critique`) inside
    the function body to avoid the circular import (`agent_tools.py` imports `nodes` at module
    level, confirmed via `grep`).
  - `orchestrator_node`: wires `_apply_judge_escalation` into the existing `enable_adaptive_diagnosis`
    branch; `diagnosis_signal` gains `escalation` (and `judge_result` when a call was actually made);
    `diagnosis_judge_calls_used` is written every enabled turn. Kill-switch branch is untouched --
    when disabled, `thinking_mode`/`diagnosis_signal` stay `None` and `diagnosis_judge_calls_used`
    is simply not included in the update (field keeps its prior value, same as before this phase).
  - `_is_near_stuck(recent_tool_calls)`: reuses `_has_repeated_tool_calls`'s tail-identity check at
    `_REPEATED_TOOL_CALL_EXIT_THRESHOLD - 1` (2, not 3) -- purely advisory, never itself exits a turn.
  - `_build_stuck_escalation_block(state)`: returns `""` when `enable_adaptive_diagnosis` is `False`
    or not near-stuck; otherwise a "LOOP WARNING" instruction telling the model to change technique.
  - `analyze_node`: appends `_build_stuck_escalation_block(effective_state)` to `system_prompt`
    right after the existing thinking-mode block. `route_node` (`nodes.py:842-...`) is **not
    modified** -- its threshold, return values, and hard-stop logic are read-only per the plan's
    red-team correction.
- `tests/unit/test_orchestrator.py`: added `test_orchestrator_low_risk_never_triggers_judge`,
  `test_orchestrator_high_risk_escalates_judge_and_spends_budget`,
  `test_orchestrator_high_risk_skips_judge_when_budget_exhausted`.
- `tests/unit/test_thinking_mode_block.py`: added `test_stuck_escalation_block_fires_one_repeat_before_hard_stop`,
  `test_stuck_escalation_block_empty_when_not_near_stuck`,
  `test_stuck_escalation_block_disabled_by_kill_switch`.

## Verification

`python -m pytest tests/unit/test_orchestrator.py tests/unit/test_elicit_tool.py tests/unit/test_thinking_mode_block.py tests/integration/test_graph_nodes.py -q`
-> 102 passed, including the pre-existing `test_route_node_exits_early_on_repeated_identical_tool_calls`
and `test_route_node_does_not_exit_on_varying_or_below_threshold_repeats` unmodified (proving
`route_node`'s hard stop is unaffected by this phase).

Full repo test suite (`python -m pytest -q`) run for final regression confirmation: 17 failed /
844 passed / 2 skipped on first pass. Diffed against a `git stash` baseline (pre-this-session) run
of the same failing tests: 16 of 17 were already failing before any of this plan's 4 phases (Bedrock
credential/model-ID issues, a live web_search network dependency, and unrelated pre-existing
integration gaps in documents/export/readiness/checkpoint/artifact-link/bmad-validator/dropped-tool
tests). The 17th, `tests/unit/test_graph_foundation.py::test_workflow_state_structure_and_add_messages_reducer`,
was a genuine regression from Phase 1/4's `WorkflowState` additions (`thinking_mode`,
`diagnosis_signal`, `diagnosis_judge_calls_used` were never added to this test's strict
`set(WorkflowState.__annotations__)` assertion) -- fixed by adding the 3 field names to the
expected set. Re-ran the full suite (excluding the Bedrock-only `tests/eval/` directory): 15
failed / 843 passed / 2 skipped, identical to the confirmed-pre-existing set, zero new failures.

## Manual sample-run check (step 9)

Ran `orchestrator_node` directly against two representative states:

- **Low-risk BRD scope section** (`section_coverage` all `filled`, no failed quality gate, drafted
  body): `thinking_mode="synthesizing"`, `diagnosis_signal={"risk_level": "low", "signals": [],
  "escalation": "not_needed"}` -- no judge call, `_build_thinking_mode_block` returns `""` (fast
  path, byte-identical prompt).
- **High-risk SAD tech_decision section** (`section_coverage` `missing`, failed quality gate, empty
  draft): `thinking_mode="risk_probing"`, `diagnosis_signal={"risk_level": "high", "signals":
  ["low_coverage", "quality_gate_failed", "sparse_draft"], "escalation": "escalated",
  "judge_result": {...}}` -- one judge call made (degraded gracefully to `no_llm_client` since no
  client was wired in this standalone check), `_build_thinking_mode_block` returns the
  `THINKING MODE: risk_probing` guidance naming `5_whys, reverse, pre_mortem`.

Confirms the diagnosis, escalation, and prompt-suffix mechanisms compose correctly end to end.

## Success Criteria check

- [x] A section flagged uncertain by the heuristic triggers exactly one judge call, refining (not
      replacing) the heuristic's classification.
- [x] A low-risk section never triggers a judge call (test +manual check).
- [x] Judge-call budget enforced per session, independent from the critique-round budget;
      budget-exhausted turn writes distinct `"escalation_skipped_budget"` signal.
- [x] `route_node` itself unmodified; near-stuck escalation instruction injected by `analyze_node`
      one repeat short of the hard-stop threshold (existing route_node regression tests pass
      unmodified; new stuck-escalation-block tests cover the injection condition).
- [x] `enable_adaptive_diagnosis` also gates judge escalation and near-stuck escalation instruction.
- [x] Full test suite passes with no regressions (see Verification).
- [x] Manual low-risk vs. high-risk sample-run comparison performed and recorded above.

Status: DONE
