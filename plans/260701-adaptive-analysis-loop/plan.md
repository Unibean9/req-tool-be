# Plan: Adaptive diagnosis -> thinking-mode -> technique loop

Status: Complete
Date: 2026-07-02
Mode: Hard
Source: plans/260701-adaptive-analysis-loop/brainstorm.md (refined by a context-scout pass)

## Design Contract

### Objective

Add a proactive, cost-gated, multi-axis diagnosis step to `orchestrator_node` that classifies the
current section's risk/ambiguity/novelty (heuristic-first, LLM-judge-escalated only when the
heuristic flags concern), writes a system-selected `thinking_mode` + technique shortlist into
`WorkflowState`, and have `analyze_node`'s prompt-building path consume it to bias tone/depth and
narrow the tool menu. Expand `elicit_tool`'s registry with BMAD-style technique lenses following the
existing dispatch pattern. Applies uniformly across BRD/PRD/SAD today (and any future stage, e.g.
Event Storming, for free).

### User / Operator Value

Users get sharper, context-appropriate elicitation (deep-dive on risky/vague sections, fast pass on
straightforward ones) without managing a technique menu. Operators get bounded LLM cost via
heuristic gating -- this is not "run 3 judge calls every turn."

### Success Metrics

- Every `analyze_node` turn is preceded by a diagnosis result (risk/ambiguity/novelty
  classification, or explicit "skipped -- low risk") visible in state/logs.
- High-risk/ambiguous sections measurably receive deeper technique application (e.g.
  pre-mortem/challenge-assumptions shows up in the transcript) vs. low-risk sections passing
  through with the current fast flow -- spot-checkable via a sample BRD scope section (fast) vs.
  SAD tech_decision section (deep) comparison.
- No regression in the existing stuck-loop/critique-cap safety valves -- escalation replaces "stop
  silently" with "try a different technique, then stop," never removes the bound.

### Acceptance Criteria

- [x] A heuristic diagnosis function exists, is pure/cheap (no LLM call), computes a
      risk/ambiguity signal from `section_coverage` state plus prior critique/gate flags, and runs
      on every `orchestrator_node` invocation.
- [x] LLM judge escalation (reusing `_invoke_judge`) fires only when the heuristic crosses a
      defined threshold, and is capped via a `CRITIQUE_ROUNDS_MAX`-style sibling constant
      (`DIAGNOSIS_JUDGE_CALLS_MAX`, sourced from a new `max_diagnosis_judge_calls` config field).
- [x] `thinking_mode` and `diagnosis_signal` exist as new top-level `WorkflowState` fields (plain,
      non-reducer), written by `orchestrator_node`, and are read by `analyze_node`'s prompt
      assembly path via a suffix block appended after `get_instruction()` (same mechanism as
      `_build_artifact_contract_block`) -- no change to the cached, role-keyed instruction
      assembly itself.
- [x] `elicit_tool`'s registry is expanded with 4 new techniques (`pre_mortem`, `tree_of_thought`,
      `socratic_questioning`, `challenge_assumptions`) following the exact existing 4-step pattern
      (tuple entry, `Literal` entry, `_elicit_X` function, elif branch) -- no dispatch
      restructuring, no argument-shape change.
- [x] The stuck-loop detector (`route_node`, `nodes.py:777-788`) is **left unmodified** -- it is a
      LangGraph conditional-edge function that returns only a routing label and cannot write state,
      so the "escalate before hard stop" behavior lives in `analyze_node`'s prompt-building step
      instead (reading the same `recent_tool_calls` fingerprints one repeat short of `route_node`'s
      threshold), never in `route_node` itself. Covered by a regression test proving `route_node`'s
      hard stop still fires after the same number of repeats as today.
- [x] The diagnosis judge-call budget is scoped per WorkflowState session (persists across
      artifact-type transitions within a session, resets only on a new session), tracked via a new
      `diagnosis_judge_calls_used` state field, and a budget-exhausted turn writes a distinct
      `diagnosis_signal` value (`"escalation_skipped_budget"`) rather than being indistinguishable
      from a low-risk assessment.
- [x] An `enable_adaptive_diagnosis` config flag exists; setting it to `False` restores byte-identical
      pre-plan behavior across all four phases (no diagnosis, no prompt suffix, no judge calls, no
      escalation instruction) without a code revert.
- [x] `orchestrator_node`'s existing responsibilities (resurfacing parked questions,
      `_should_run_completeness_sweep`/`is_brd_stable` BRD->PRD gating) keep passing their existing
      tests unmodified.

### Not Doing

- No user-facing elicitation menu -- selection fully automatic.
- No graph restructuring into separate diagnose/elicit/critique nodes in v1.
- No full multi-agent debate as a default path -- reserved as one high-cost technique for
  high-risk sections only, not a standing architecture change.
- No Event Storming work -- this is the prerequisite; ES will consume this later.
- No generalizing `_should_run_completeness_sweep`/`is_brd_stable` to N stages in this pass.
- No unconditional per-turn LLM judge calls for diagnosis -- heuristic gating is mandatory, not
  optional.

### Constraints

- No new LangGraph nodes/edges -- stays inside the existing `orchestrator_node` -> `analyze_node`
  seam (`app/graphs/graph.py:51`).
- New `WorkflowState` fields must be plain (non-reducer) -- do not add `Annotated` reducers unless
  a genuine accumulation need is found (none identified here: both fields are replace-on-write,
  same as `feedback_summary`).
- `elicit_tool` changes must follow the existing tuple+Literal+function+elif pattern exactly -- no
  schema-shape change to the tool's argument contract (`technique: Literal[...]`, `seed: str`).
- Diagnosis LLM escalation must be bounded by an explicit round/call cap, following the
  `CRITIQUE_ROUNDS_MAX` precedent (`app/config.py:62`, `app/graphs/agent_tools.py:1517`).
- Must not regress `_should_run_completeness_sweep`/`is_brd_stable` BRD->PRD gating -- those stay
  as-is (out of scope, per Not Doing).
- `route_node` (`nodes.py:777-788`) is a conditional-edge function and cannot write state -- any
  "escalate before hard stop" behavior must live in a node that already returns state updates
  (`analyze_node`), never in `route_node` itself. (Added after red-team review found the original
  Phase 4 design asserted a write-point that doesn't exist.)
- Phase 2's prompt-builder guidance block must only name techniques that exist in `elicit_tool`'s
  `Literal` at the time Phase 2 ships (the original 5) -- forward-referencing Phase 3's new
  technique names before Phase 3 lands would let the model attempt an invalid tool call.
- `get_instruction()`'s assembled string is cached by `(role, has_draft)` (`app/instructions/__init__.py:80,143-149`)
  -- `thinking_mode` content must NOT be baked into that cache key or that cached string; it must be
  appended per-turn the same way `_build_artifact_contract_block` is appended today
  (`app/graphs/nodes.py:594`).

### Assumptions

Must be true:
- State hand-off `orchestrator_node` -> `analyze_node` is sequential and safe for new fields
  (confirmed by scout: `app/graphs/graph.py:51` is a hardwired sequential edge; `state.py:164-229`
  shows only 3 reducer fields, everything else is plain replace-on-write and visible same-turn).
- `elicit_tool` extension is additive, no restructuring needed (confirmed by scout:
  `agent_tools.py:1524,1682-1691,1646-1663` -- tuple, `Literal`, and elif chain are all
  independently extensible).

Should be true:
- A cheap heuristic signal (coverage delta / prior critique flags / keyword density) is expressive
  enough to gate LLM escalation without missing real high-risk sections -- validated via the
  sample-run success metric, not provable statically; tracked as a risk to watch (see Risk
  Register), not a blocker.
- The stuck-loop detector can be safely extended to "escalate technique, then stop" without
  weakening its role as a hard safety valve elsewhere -- needs a passing regression test (Phase 4),
  not just code review.

Might be true:
- Direction B (dedicated graph nodes) becomes worth it later -- explicitly deferred.

### Open Questions

None left materially blocking. The two items flagged by scout are resolved as design decisions:
- Where to write `thinking_mode`/`diagnosis_signal`: top-level `WorkflowState` fields (not nested
  in `feedback_summary`) for discoverability and independent typing.
- How the prompt builder consumes `thinking_mode`: a new suffix block appended to `system_prompt`
  in `analyze_node`, mirroring `_build_artifact_contract_block` exactly (see context.md for the
  full rationale).

### Verification Strategy

- Build: no build step beyond existing Python import/lint checks (`ruff`/`mypy` if configured in
  CI); no new dependencies.
- Test: extend `tests/unit/test_orchestrator.py` (heuristic diagnosis + gating logic),
  `tests/unit/test_elicit_tool.py` (new techniques), `tests/unit/test_run_critique_tool.py` if
  judge-call reuse touches shared code, `tests/unit/test_instruction_contract.py` and/or new
  prompt-builder assertions for `thinking_mode` consumption, `tests/integration/test_graph_nodes.py`
  for an end-to-end orchestrator->analyze state-flow check.
- Review: standard code review; flag the cost/latency tradeoff explicitly since this affects every
  Q&A turn.
- Runtime/manual: one sample run comparing a low-risk BRD section (fast path, no escalation)
  against a high-risk SAD `tech_decision` section (escalated technique visible in transcript), per
  the brainstorm's success metric -- document the outcome in Session Notes since no automated
  content-quality assertion exists yet.

### Support Checks

| Check | Trigger | Evidence |
| --- | --- | --- |
| testing-strategy | Behavior change to a core loop (`orchestrator_node`/`analyze_node`) that runs every turn | New/extended tests per file listed in Verification Strategy; each phase lists its own test deltas |
| None (security-hardening) | No auth/PII/secrets touched -- internal graph-node behavior only | N/A |
| None (migration-safety) | New `WorkflowState` fields are plain TypedDict additions, no DB schema change, no migration | N/A |
| None (documentation-adrs) | No public API or contract change -- internal agent-loop behavior | N/A |

### Ship Criteria

- All acceptance criteria above are met and covered by passing tests.
- Full existing test suite has no regressions, especially `test_orchestrator.py`,
  `test_elicit_tool.py`, `test_run_critique_tool.py`, and stuck-loop-related tests in
  `test_graph_nodes.py`.
- Manual sample-run spot check (low-risk vs. high-risk section) performed and documented in
  Session Notes.

## Dependency Graph

Foundation:
- New `WorkflowState` fields (`thinking_mode: str | None`, `diagnosis_signal: dict[str, Any] | None`)
  in `app/graphs/state.py` -- no migration, pure Python TypedDict addition.
- Heuristic diagnosis scoring function (new, colocated with `orchestrator_node` in
  `app/graphs/nodes.py`) -- depends only on existing `section_coverage`/`decision_nodes`/
  `quality_report`/`critique_rounds` state shape.
- New config constants: `max_diagnosis_judge_calls` (`app/config.py`, sibling to
  `max_critique_rounds`) and derived `DIAGNOSIS_JUDGE_CALLS_MAX` (sibling to
  `CRITIQUE_ROUNDS_MAX` in `app/graphs/agent_tools.py`).

Features:
- Wire heuristic diagnosis (+ gated LLM judge escalation reusing `_invoke_judge`) into
  `orchestrator_node`, writing `thinking_mode`/`diagnosis_signal`.
- Update `analyze_node`'s prompt-building path to consume `thinking_mode` (tone/depth bias) via a
  new suffix block, and narrow technique suggestions surfaced to the model.
- Expand `elicit_tool` registry with 4 new BMAD-style techniques (tuple + Literal + function +
  elif, per existing pattern).
- Extend stuck-loop/critique-cap escalation behavior (from hard-stop to "try alternate technique,
  then stop") -- additive only, existing hard-stop threshold and behavior preserved.

Surface:
- None required (no UI/CLI/docs surface change -- this is internal agent-loop behavior). Any
  transcript/logging visibility for diagnosis results is folded into the Features phases above,
  not a separate Surface phase.

## Phases

- [x] Phase 1: Diagnosis foundation -- heuristic signal, state fields, orchestrator wiring
- [x] Phase 2: Prompt-builder consumption -- analyze_node reads thinking_mode
- [x] Phase 3: Technique registry expansion -- 4 new elicit_tool techniques
- [x] Phase 4: Judge escalation + stuck-loop/critique-cap escalation behavior

## Risk Register

| Risk | Impact | Mitigation | Owner |
| --- | --- | --- | --- |
| Heuristic signal quality: coverage-delta/keyword heuristic misses real high-risk sections (false negative), so those sections silently get the fast path | Medium | Phase 1 tests cover known high-risk shapes (empty section, prior critique fail, low coverage); success metric's manual sample run explicitly checks a SAD tech_decision section escalates; threshold is a single named constant so it can be retuned without a redesign | Implementer |
| Cost/latency creep if the escalation threshold is set too loosely, or `DIAGNOSIS_JUDGE_CALLS_MAX` is too high, causing judge calls on most turns | Medium | Threshold and call cap are both named, defaulted conservatively (favor skipping over escalating), and covered by a unit test asserting low-risk state never triggers a judge call | Implementer |
| Stuck-loop detector regression: repurposing it for "escalate then stop" accidentally weakens or removes the hard stop | High | Resolved by design, not just mitigation: `route_node` is left completely unmodified (it cannot write state -- red-team review caught the original design asserting otherwise); escalation lives entirely in `analyze_node`'s prompt content, verified by a regression test that `route_node`'s hard stop still fires after the same repeat count as today | Implementer |
| Judge-call budget exhaustion silently looks identical to a low-risk assessment, masking real diagnosis degradation mid-session | Medium | Phase 4 writes a distinct `diagnosis_signal` value (`"escalation_skipped_budget"`) when the budget is exhausted, with a dedicated test | Implementer |
| No operator-facing rollback if the heuristic misfires broadly in production (e.g. everything reads high-risk) | Medium | `enable_adaptive_diagnosis` config flag added in Phase 1 -- one config flip disables diagnosis end-to-end without a redeploy | Implementer |
| `get_instruction()` cache is keyed by `(role, has_draft)` -- an implementer might be tempted to add `thinking_mode` to that cache key or bake it into a cached string, which would explode cache cardinality and reintroduce staleness bugs | Low | Design Contract and context.md explicitly document the suffix-append mechanism (same as `_build_artifact_contract_block`); Phase 2 acceptance criteria call this out directly | Implementer |
| New elicit techniques increase the flat tool menu's cognitive load for the model if all 9 techniques are always offered | Low | Out of scope for this pass (menu narrowing via `thinking_mode` is Phase 2's job); if Phase 2's narrowing doesn't fully address it, flag as follow-up, not a blocker here | Implementer |

## Research Summary

External prior art (BMAD-METHOD advanced-elicitation pattern, Self-Refine, Reflexion, risk-driven
elicitation ordering) is documented in `brainstorm.md` sections 1 and 3-4 -- not duplicated here.
The scout pass (this plan's source input) confirmed all repo-side "Must be true" assumptions from
the brainstorm and resolved the two open design questions (state field placement, prompt-builder
mechanism); see `context.md` for the resolution rationale.

## Session Notes

**2026-07-02 -- Red-team review (plan-reviewer) adjudication:**
- Verdict was BLOCK on the original draft due to one CRITICAL finding: Phase 4's stuck-loop
  "escalate before hard stop" design asserted `route_node` could record an escalation signal,
  but `route_node` is a LangGraph conditional-edge function that only returns a routing label and
  cannot write state. ACCEPTED and fixed: escalation now lives in `analyze_node`'s prompt-building
  step instead; `route_node` is untouched. See Phase 4 steps 4-5 and the updated Constraints/Risk
  Register above.
- ACCEPTED: judge-call budget scoping was ambiguous ("per session" undefined) and a
  budget-exhausted turn was indistinguishable from a low-risk one. Fixed: budget is explicitly
  per-WorkflowState-session via a new `diagnosis_judge_calls_used` field, and exhaustion writes a
  distinct `diagnosis_signal` value. See Phase 4 step 6.
- ACCEPTED: Phase 2 (ships before Phase 3) risked referencing technique names not yet in
  `elicit_tool`'s `Literal`. Fixed: Phase 2 is now scoped to only the 5 pre-existing techniques,
  with an explicit Phase 3 follow-up step to widen the guidance-block vocabulary.
- NOTED items also applied (cheap, non-blocking): added `enable_adaptive_diagnosis` kill switch
  (Phase 1) as an operator rollback path; added an interim checkpoint after Phase 2; made the
  heuristic threshold a named conjunction of concrete signals instead of an unbound soft score
  (Phase 1); documented the cold-start "low risk by default" behavior as intentional.
- REJECTED (not load-bearing): brainstorm.md's stale `CRITIQUE_ROUNDS_MAX` default citation
  (says 5, actual code default is 2) -- no plan/phase file references the value itself, only the
  config field by name, so this doesn't propagate into any acceptance criterion.
- Full reviewer transcript available on request; not persisted as a separate file per this plan's
  scope (internal agent-loop change, no external audit requirement).

**2026-07-02 -- Cook (--auto) implementation, all 4 phases:**
- All phases executed inline (no dispatch trigger hit -- each phase touched 1-2 files, no
  contract/security/data/migration risk, acceptance criteria were clear and verification was a
  focused `pytest` run each time).
- Phase 1: heuristic diagnosis (`_diagnose_section`), `thinking_mode`/`diagnosis_signal` state
  fields, `enable_adaptive_diagnosis` kill switch, orchestrator wiring. 4 tests added.
- Phase 2: `_build_thinking_mode_block` suffix appended to `analyze_node`'s `system_prompt`,
  scoped to the 5 pre-existing `elicit_tool` techniques per the red-team-fixed ordering. 3 tests
  added.
- Phase 3: `elicit_tool` registry expanded from 5 to 9 techniques (`pre_mortem`,
  `tree_of_thought`, `socratic_questioning`, `challenge_assumptions`); Phase 2's tracked
  follow-up (widening `_THINKING_MODE_TECHNIQUE_HINTS`) applied immediately in the same run since
  no stale-TODO window existed. 4 elicit tests + 1 revised guard test.
- Phase 4: `max_diagnosis_judge_calls`/`DIAGNOSIS_JUDGE_CALLS_MAX` sibling to the critique-round
  cap; `_apply_judge_escalation` wired into the orchestrator's diagnosis step (low-risk ->
  `"not_needed"`, budget-exhausted -> distinct `"escalation_skipped_budget"`, otherwise one
  `_invoke_judge(..., "risk_review", ...)` call); `_is_near_stuck`/`_build_stuck_escalation_block`
  added to `analyze_node`'s prompt-building step, `route_node` itself left completely unmodified
  (confirmed by its pre-existing regression tests passing unchanged). 6 tests added. Manual
  low-risk-BRD vs. high-risk-SAD sample run performed and recorded in
  `evidence/phase-04-evidence.md`.
- Verification: `pytest tests/unit/test_orchestrator.py tests/unit/test_elicit_tool.py
  tests/unit/test_thinking_mode_block.py tests/integration/test_graph_nodes.py -q` -> 102 passed
  after Phase 4 (up from 96 after Phase 3); full repo suite run for final regression confirmation.
- No task-reviewer/code-reviewer dispatched per-phase (all phases stayed inline-eligible under
  the dispatch heuristic); a final code-reviewer pass on the whole diff is recommended before
  `/ck:ship` since the change crosses `state.py`/`config.py`/`nodes.py`/`agent_tools.py` module
  boundaries, per the ck-cook skill's Step 4 trigger for inline-only multi-module diffs.
- Full-suite regression check: first full run found 17 failures. Diffed each against a `git stash`
  baseline of this session's changes -- 16 were already failing before any of this plan's 4 phases
  (Bedrock credential/model-ID issues, a live-network `web_search` test, and pre-existing gaps in
  documents/export/readiness/checkpoint/artifact-link/bmad-validator/dropped-tool tests, none
  touching this plan's files). The 17th,
  `test_graph_foundation.py::test_workflow_state_structure_and_add_messages_reducer`, was a real
  regression: its strict `set(WorkflowState.__annotations__)` assertion was never updated for
  Phase 1's `thinking_mode`/`diagnosis_signal` or Phase 4's `diagnosis_judge_calls_used` fields.
  Fixed by adding the 3 names to the test's expected set. Re-ran the full suite (excluding the
  Bedrock-only `tests/eval/`): 15 failed / 843 passed / 2 skipped, matching the confirmed
  pre-existing set exactly -- zero regressions from this plan's implementation.
