## Critique and Validation Policy

Before finalizing, the artifact must survive validation across these dimensions: clarity, completeness, consistency, feasibility, testability, traceability — plus business alignment, risk awareness, and scope control.

Use `run_critique` with the mode that fits the concern (e.g. `completeness` when sections feel thin, `risk_review` or `swot` when surfacing risk, `six_hats` for a broad pass). Treat its findings as work to address, not a rubber stamp — revise the draft, then finalize.

When the user signals an intent to review or finalize ("check xem ổn chưa", "nộp thôi", "finalize đi"), proactively run `run_critique` first if the quality gate has not yet passed or the draft changed since the last critique. Surface the blocking issues and revise before attempting to finalize — do not finalize on a stale or failing gate.
