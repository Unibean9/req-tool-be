# Final Benchmark Comparison

Same canonical journey fixture and methodology as `evidence/benchmark-baseline.md`, re-run after the behavior quality changes.

## Mode

Identical to the baseline: scripted mock LLM through the real HTTP/graph scenario harness. Latency is in-process overhead only. Token counts are a deterministic payload-size proxy, not provider-billed usage.

## Results (3 runs, values summed across all 3 journeys per run)

| Metric | Baseline Median | Final Median | Median Delta | Median Delta % |
| --- | --- | --- | --- | --- |
| total_latency_ms | 4406 | 4229 | -177 | -4.0% |
| total_tokens | 9856 | 9856 | +0 | +0.0% |
| llm_calls | 20 | 20 | +0 | +0.0% |
| tool_call_count | 10 | 10 | +0 | +0.0% |
| artifact_count | 3 | 3 | +0 | +0.0% |

## Baseline vs Final, full min/median/max

| Metric | Baseline Min | Baseline Median | Baseline Max | Final Min | Final Median | Final Max |
| --- | --- | --- | --- | --- | --- | --- |
| total_latency_ms | 4096 | 4406 | 7082 | 4095 | 4229 | 4240 |
| total_tokens | 9844 | 9856 | 9856 | 9844 | 9856 | 9856 |
| llm_calls | 20 | 20 | 20 | 20 | 20 | 20 |
| tool_call_count | 10 | 10 | 10 | 10 | 10 | 10 |
| artifact_count | 3 | 3 | 3 | 3 | 3 | 3 |

## Attribution (approximate, not isolated per-phase measurement)

The delta above reflects the combined end-to-end effect across the canonical journey set. It is not a clean per-phase attribution report.

## No Numeric Target

No pass/fail threshold was set for this plan (see plan.md Not Doing) - this report is evidence-gathering, reporting the measured delta honestly (including any regression), not a gate.

## Raw per-run, per-journey data (final)

```json
[
  [
    {
      "journey": "canonical-clarify-draft-approve",
      "final_status": "completed",
      "total_latency_ms": 1022,
      "total_tokens": 1431,
      "llm_calls": 4,
      "tool_call_count": 2,
      "artifact_count": 1,
      "step_count": 5
    },
    {
      "journey": "canonical-reject-critique-redraft",
      "final_status": "completed",
      "total_latency_ms": 2404,
      "total_tokens": 6851,
      "llm_calls": 13,
      "tool_call_count": 6,
      "artifact_count": 1,
      "step_count": 8
    },
    {
      "journey": "canonical-context-artifact-from-predecessors",
      "final_status": "completed",
      "total_latency_ms": 803,
      "total_tokens": 1562,
      "llm_calls": 3,
      "tool_call_count": 2,
      "artifact_count": 1,
      "step_count": 4
    }
  ],
  [
    {
      "journey": "canonical-clarify-draft-approve",
      "final_status": "completed",
      "total_latency_ms": 1041,
      "total_tokens": 1449,
      "llm_calls": 4,
      "tool_call_count": 2,
      "artifact_count": 1,
      "step_count": 5
    },
    {
      "journey": "canonical-reject-critique-redraft",
      "final_status": "completed",
      "total_latency_ms": 2365,
      "total_tokens": 6851,
      "llm_calls": 13,
      "tool_call_count": 6,
      "artifact_count": 1,
      "step_count": 8
    },
    {
      "journey": "canonical-context-artifact-from-predecessors",
      "final_status": "completed",
      "total_latency_ms": 834,
      "total_tokens": 1556,
      "llm_calls": 3,
      "tool_call_count": 2,
      "artifact_count": 1,
      "step_count": 4
    }
  ],
  [
    {
      "journey": "canonical-clarify-draft-approve",
      "final_status": "completed",
      "total_latency_ms": 1141,
      "total_tokens": 1449,
      "llm_calls": 4,
      "tool_call_count": 2,
      "artifact_count": 1,
      "step_count": 5
    },
    {
      "journey": "canonical-reject-critique-redraft",
      "final_status": "completed",
      "total_latency_ms": 2166,
      "total_tokens": 6851,
      "llm_calls": 13,
      "tool_call_count": 6,
      "artifact_count": 1,
      "step_count": 8
    },
    {
      "journey": "canonical-context-artifact-from-predecessors",
      "final_status": "completed",
      "total_latency_ms": 788,
      "total_tokens": 1556,
      "llm_calls": 3,
      "tool_call_count": 2,
      "artifact_count": 1,
      "step_count": 4
    }
  ]
]
```
