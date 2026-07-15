# Observability spec — dashboards/alerts (Phase 8 Step 5 placeholder)

Status: NOT ENFORCED — pending observability stack selection and human sign-off.

This document is a named-owner/threshold slot for the Phase 8 quality bar
("No-Placeholder: every required CI/deploy job, alert, SLO and removal
criterion has named owner/threshold before enablement"). It is not claimed as
satisfied by the Phase 8 CI-gate/correlation slice implemented on 2026-07-16 —
see the ADR addendum in `docs/adr/0001-logical-turn-control-plane.md` for the
scope decision that deferred this work.

No dashboard/alerting stack (Grafana, Datadog, Prometheus, or equivalent) is
configured in this repository as of 2026-07-16. Every row below lists the SQL
query an alert would run against tables that already exist and are provable
today (via a plain `SELECT`), separately from what a real stack would still
need to add (routing, thresholds, on-call paging, historical retention).

| Alert | Owner | Threshold | Provable today (SQL against existing tables) | Needs a real stack |
| --- | --- | --- | --- | --- |
| Admission conflict | Unassigned | TBD | `SELECT * FROM agent_turn_triggers WHERE ...` idempotency-key collisions — see `uq_agent_turn_trigger_idempotency` | Alert routing/paging, rate-of-change threshold |
| Fence rejection | Unassigned | TBD | Count of `StaleTurnOwnershipError`/`StaleCheckpointAppendError` raises — not yet logged to a queryable table, only raised in-process | Structured error log sink + query, alert threshold |
| Queued/dead-letter age | Unassigned | TBD | `SELECT id, scheduled_at, status FROM agent_turn_jobs WHERE status IN ('queued','dead_letter') ORDER BY scheduled_at` | Age-based threshold alert, paging |
| Retry/reconciliation | Unassigned | TBD | `SELECT * FROM agent_turn_jobs WHERE attempt > 0` and reconciliation helper in `app/graphs/analysis/turn_reconciliation.py` | Aggregated rate/percentage threshold |
| Terminal mismatch | Unassigned | TBD | Join `agent_sessions.status` vs `agent_turn_outcomes.outcome_type` for the same `turn_id`/`session_id` to find disagreement | Automated diff job + alert |
| Checkpoint fork/replay error | Unassigned | TBD | `AgentCheckpointHistorySaver` raises `StaleCheckpointAppendError` on parent/fence mismatch — same gap as fence rejection above | Structured error log sink + query |
| Event authorization/cursor error | Unassigned | TBD | `SELECT * FROM agent_turn_events WHERE session_sequence <> expected` (cursor gap detection against `agent_sessions.event_cursor`) | Automated cursor-gap scan job + alert |
| Cost/provider attribution | Unassigned | TBD | `SELECT provider_config_id, model_name, token_usage FROM agent_runs` (now also joinable to `turn_id` — see Phase 8 correlation addendum) | Cost aggregation/rollup dashboard |

## What this phase's slice actually provides

The 2026-07-16 slice makes the *data* behind every row above queryable and
correlation-joinable (via `AgentRun.turn_id`, `AgentTurnEvent`, `TurnOutcome`,
`DraftCommandLedger`, all keyed off `AgentTurnEnvelope.correlation_id`) — see
`tests/unit/agent/test_turn_audit.py` for the proof. It does not stand up any
dashboard, alert rule, or paging integration. Selecting and wiring an actual
observability stack, naming real owners, and setting real thresholds remains
open work requiring a human decision, not a code change.
