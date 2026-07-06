# Baseline Benchmark

Recorded before any latency/token fix lands. Same canonical journey fixture and methodology as `evidence/benchmark-final.md` - see `tests/benchmark/fixture.py`.

## Mode

Scripted mock LLM through the real HTTP/graph scenario harness. Latency is in-process harness and application overhead, not real external LLM latency. Token counts are a deterministic payload-size proxy, not provider-billed usage.

## Fixture (3 canonical journeys)

`tests/benchmark/fixture.py:run_fixture` runs the shared canonical journeys through `ScenarioDriver`, so integration, eval, and benchmark lanes exercise the same behavior.

## Results (3 runs, values summed across all 3 journeys per run)

| Metric | Min | Median | Max |
| --- | --- | --- | --- |
| total_latency_ms | 4096 | 4406 | 7082 |
| total_tokens | 9844 | 9856 | 9856 |
| llm_calls | 20 | 20 | 20 |
| tool_call_count | 10 | 10 | 10 |
| artifact_count | 3 | 3 | 3 |

## Raw per-run, per-journey data

```json
[
  [
    {
      "journey": "canonical-clarify-draft-approve",
      "final_status": "completed",
      "total_latency_ms": 1255,
      "total_tokens": 1431,
      "llm_calls": 4,
      "tool_call_count": 2,
      "artifact_count": 1,
      "step_count": 5
    },
    {
      "journey": "canonical-reject-critique-redraft",
      "final_status": "completed",
      "total_latency_ms": 2345,
      "total_tokens": 6851,
      "llm_calls": 13,
      "tool_call_count": 6,
      "artifact_count": 1,
      "step_count": 8
    },
    {
      "journey": "canonical-context-artifact-from-predecessors",
      "final_status": "completed",
      "total_latency_ms": 806,
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
      "total_latency_ms": 1070,
      "total_tokens": 1449,
      "llm_calls": 4,
      "tool_call_count": 2,
      "artifact_count": 1,
      "step_count": 5
    },
    {
      "journey": "canonical-reject-critique-redraft",
      "final_status": "completed",
      "total_latency_ms": 2223,
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
      "total_latency_ms": 1111,
      "total_tokens": 1449,
      "llm_calls": 4,
      "tool_call_count": 2,
      "artifact_count": 1,
      "step_count": 5
    },
    {
      "journey": "canonical-reject-critique-redraft",
      "final_status": "completed",
      "total_latency_ms": 2234,
      "total_tokens": 6851,
      "llm_calls": 13,
      "tool_call_count": 6,
      "artifact_count": 1,
      "step_count": 8
    },
    {
      "journey": "canonical-context-artifact-from-predecessors",
      "final_status": "completed",
      "total_latency_ms": 3737,
      "total_tokens": 1556,
      "llm_calls": 3,
      "tool_call_count": 2,
      "artifact_count": 1,
      "step_count": 4
    }
  ]
]
```
