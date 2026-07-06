# Phase 2 Evidence: Prompt-builder consumption

Execution path: inline.

## Changes

- `app/graphs/nodes.py`: added `_THINKING_MODE_TECHNIQUE_HINTS`, `_THINKING_MODE_RATIONALE`, and
  `_build_thinking_mode_block()` near `_build_artifact_contract_block`; appended its output to
  `system_prompt` in `analyze_node` right after the existing artifact-contract append. Only
  `challenging` and `risk_probing` (the two high-risk modes from Phase 1) produce a non-empty
  block; `structuring`/`synthesizing`/unset all return `""`, keeping the low-risk fast path's
  prompt unchanged. Techniques named are limited to the 5 pre-existing `elicit_tool` entries
  (`reverse`, `first_principles`, `5_whys`) -- no forward reference to Phase 3's additions.
- `tests/unit/test_thinking_mode_block.py` (new): high-risk block content, low-risk/unset no-op,
  and a guard test that none of Phase 3's future technique names leak into the block early.

## Verification

`python -m pytest tests/unit/test_thinking_mode_block.py tests/unit/test_instruction_contract.py -q`
-> 18 passed. `get_instruction()`'s cache keying (role, has_draft) is untouched -- confirmed by
inspection (`_build_thinking_mode_block` is a separate function appended after `get_instruction()`
returns, never passed into it).

## Success Criteria check

- [x] High-risk section's system prompt contains mode + 2 suggested techniques.
- [x] Low-risk section's system prompt unchanged (empty suffix).
- [x] `get_instruction()` cache keying verified unchanged by inspection + passing existing tests.
- [x] Existing prompt/instruction tests pass unmodified.

Status: DONE. Phase 3 follow-up (widen technique vocabulary once new techniques land) tracked
below in this plan's Session Notes.
