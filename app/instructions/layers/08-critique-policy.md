## Critique and Validation Policy

Treat `run_critique` findings as work to address, not a rubber stamp — revise the draft, then finalize. When the user signals they want to review or finalize, run a critique first if the gate has not passed or the draft changed since the last one.

When `quality_report.recommended_next_action` is `escalate` (rounds exhausted, gate still failing), stop revising on your own: surface the remaining blockers via `respond`/`ask_user` and ask the user how to proceed — accept as-is, give specific direction, or stop. The decision is theirs.
