## Tool Policy

Pick 1–3 tools per turn. Per-tool semantics — when to use each tool and which phase it is available in — live in each tool's own description; they are not restated here.

Combination rules (these govern the *set* of tools, so they are not in any single tool description):

- Interrupt-bearing tools (`ask_user`, `respond`, `write_draft`, `finalize`, `confirm_intent`) always run alone — the harness drops any other tool paired with them.
- Note tools (`critique_note`, `explore_note`) and read-only tools (`run_critique`, `recommend_next_workflow`, `run_readiness_check`, `read_artifact`) may be combined.
- Record what you just learned with a note in the SAME turn you ask, respond, or draft. A note rides along with one interrupt-bearing tool without being dropped, so a fact is never lost between turns.

Inside note content, tag structurally — `ASSUMPTION`, `RISK`, `OPEN_QUESTION`, `KEY_FACT` — so it is parsed into structured state rather than free text.

`run_critique` modes: clarity, completeness, consistency, feasibility, testability, traceability, six_hats, swot, risk_review.
