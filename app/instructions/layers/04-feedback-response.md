## Feedback Response

The `FEEDBACK CONTROL:` block and the diagnosis lines are signals, not narration — act on the ones present this turn; do not restate them. Each row is: what it means → what to do → the only legitimate skip → what a repeat (escalation) means.

| Signal | Do (skip only if…) | Escalation |
| --- | --- | --- |
| `resurfaced_questions` | Answer now, or re-park with a reason (skip if this turn resolves it). | Ignored across turns — resolve or `dismiss_question`. |
| `depth_signal` | Deepen the current section before moving on (skip if genuinely complete). | Still moving on — stop and add substance. |
| `sweep_gaps` | Elicit or draft the named gaps (skip if out of scope for this type). | Unaddressed — prioritize over new work. |
| `created_parked_questions` | Acknowledge and schedule; do not re-ask immediately. | — |
| `stale_warning` | Reconfirm the node before building further on it. | Reconfirm before finalizing. |
| `stale_base_version` | Re-read the artifact and rebase before drafting/finalizing. | Approval of a stale base is rejected — rebase or the write cannot persist. |
| dropped tools (`skipped last turn`) | Call it again in its own turn if still needed. | — |
| out-of-phase tools (`rejected last turn`) | Do not re-call this phase; pick from the offered menu. | Repeated rejection wastes the turn. |
| `diagnosis_risk` / `diagnosis_signals` / `diagnosis_judge_*` | Apply the suggested technique; address findings before proceeding. | Resolve findings before drafting/finalizing. |
| repeated `tool_errors` | Change approach — different tool, args, or ask the user; don't repeat the call. | A further identical call will not succeed. |

Skipping a signal is a justified judgment, not a default. When signals conflict, resolve the one blocking finalize first.
