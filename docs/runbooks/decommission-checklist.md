# Decommission checklist (Phase 8 Step 6 placeholder)

Status: NOT STARTED — requires architecture owner sign-off.

This document is a named-owner/threshold slot for the Phase 8 quality bar
("No-Placeholder: every required CI/deploy job, alert, SLO and removal
criterion has named owner/threshold before enablement"). No legacy
representation is decommissioned by the Phase 8 CI-gate/correlation slice
implemented on 2026-07-16 — see the ADR addendum in
`docs/adr/0001-logical-turn-control-plane.md` for the scope decision that
deferred this work. Nothing on this page is claimed as satisfied.

Per representation being considered for removal (e.g. `AgentSession.status` as
a compatibility projection, `checkpoint_version == "v1"` reader, legacy inline
execution mode), the following must all pass before removal, in order:

1. **Active cohort scan** — Owner: TBD. Threshold: TBD (e.g. zero active
   sessions/turns on the legacy cohort for N days). Evidence: a query against
   the relevant cohort/flag column (e.g. `agent_sessions.checkpoint_version`,
   turn envelope `cohort` JSON) showing no live rows on the legacy path.
2. **Retention expiry** — Owner: TBD. Threshold: TBD (business-defined
   retention window for audit/compliance). Evidence: retention policy
   document plus a query proving all rows within the window have aged out.
3. **Compatibility/replay drill** — Owner: TBD. Evidence: a rehearsed replay
   of a legacy-cohort session/turn against current code, run in a
   staging-like environment, with a passing result recorded.
4. **Invariant evidence** — Owner: TBD. Evidence: the specific contract
   test(s) proving the new path fully replaces the legacy one (test file +
   run output), not just "unit tests pass."
5. **Rollback plan** — Owner: TBD. Evidence: a documented, tested rollback
   procedure to restore the legacy path if removal causes a regression.
6. **ADR sign-off** — Owner: TBD (architecture owner). Evidence: an ADR
   addendum recording the removal decision, the above five gates having
   passed, and the sign-off date/name.

## Explicit non-goal of this phase

This checklist is a placeholder slot only. No representation is removed, no
cohort scan has been run, and no rollback plan has been written as part of
the 2026-07-16 Phase 8 slice — decommission execution requires business
sign-off that cannot be decided by code and is out of scope for this session.
