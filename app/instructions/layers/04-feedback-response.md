## Feedback Response

The `FEEDBACK CONTROL:` block and diagnosis lines are signals, not narration. Act on signals present this turn; do not restate them. `[state]` comes from prompt state; `[tool]` comes from the matching tool result.

| Signal | Do (skip only if…) | Escalation |
| --- | --- | --- |
| `[state] resurfaced_questions` | Answer now, or re-park with a reason (skip if resolved). | Resolve before finalizing. |
| `[state] depth_signal` | Deepen the section (skip if complete). | Stop moving on; add substance. |
| `[state] sweep_gaps` | Elicit or draft gaps (skip if out of scope). | Prioritize over new work. |
| `[state] created_parked_questions` | Acknowledge and schedule; do not re-ask now. | — |
| `[tool] stale_warning` | Reconfirm the node before building on it. | Reconfirm before finalizing. |
| `[state] stale_base_version` | Re-read and rebase before drafting/finalizing. | The write cannot persist until rebased. |
| `[state] lifecycle_persist_rejection` | Re-read upstream artifacts; rebase. | Rejects until upstream versions match. |
| `[state] candidate_readiness_rejection` | Resolve readiness blockers. | Rejects until ready. |
| `[state] dropped tools` | Call again alone if still needed. | — |
| `[state] out-of-phase tools` | Pick from the offered menu. | Repeated rejection wastes the turn. |
| `[state] lifecycle-blocked tools` | Resolve the lifecycle blocker first. | Repeated rejection wastes the turn. |
| `[tool] deterministic draft warnings` | Address before relying on the draft. | Reconfirm before finalizing. |
| `diagnosis_risk` / `diagnosis_signals` / `diagnosis_judge_*` | Apply the suggested technique and address findings. | Resolve before drafting/finalizing. |
| repeated `tool_errors` | Change tool, args, or ask the user. | Do not repeat an identical call. |

Skipping a signal is a justified judgment, not a default. When signals conflict, resolve the one blocking finalize first.
