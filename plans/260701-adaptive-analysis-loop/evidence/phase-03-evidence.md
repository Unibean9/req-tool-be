# Phase 3 Evidence: Technique registry expansion

Execution path: inline.

## Changes

- `app/graphs/agent_tools.py`: extended `ELICIT_TECHNIQUES` with `pre_mortem`, `tree_of_thought`,
  `socratic_questioning`, `challenge_assumptions`; added one `_elicit_X` function per new technique
  (same seed-in/structured-frame-out shape as the existing 5); extended `elicit()`'s if/elif
  dispatch chain (converted the trailing `comparable_products` branch to an explicit `if` so the
  new branches read the same way); extended `elicit_tool`'s `Literal[...]` with the 4 new names.
  No argument-shape change (`technique: Literal[...]`, `seed: str` unchanged).
- Applied Phase 2's tracked follow-up immediately (both phases land in this same cook run, so no
  stale TODO window exists): widened `_THINKING_MODE_TECHNIQUE_HINTS` in `nodes.py` to include
  `challenge_assumptions` (challenging mode) and `pre_mortem` (risk_probing mode).
- `tests/unit/test_elicit_tool.py`: added one test per new technique.
- `tests/unit/test_thinking_mode_block.py`: updated the forward-reference guard test into a
  positive "every hinted technique exists in ELICIT_TECHNIQUES" check, since the techniques are no
  longer forward references.

## Verification

`python -m pytest tests/unit/test_elicit_tool.py tests/unit/test_thinking_mode_block.py -q` ->
14 passed (11 elicit + 3 thinking-mode, including the 5 pre-existing elicit tests unmodified).

## Success Criteria check

- [x] All 4 new techniques invokable, return structured frames (tests).
- [x] Existing 5 techniques unmodified, existing tests pass unedited.
- [x] Unknown technique still raises `ValueError` (existing test, unaffected by the dispatch change).
- [x] Tool argument contract unchanged -- `test_elicit_tool_in_registry` (technique/seed shape)
      passes unmodified.

Status: DONE
