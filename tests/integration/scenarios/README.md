# Agent Behavior Scenario Tests

This suite drives multi-turn conversations through the real HTTP API and
LangGraph wiring with a deterministic scripted LLM. High-level scenarios are
canaries, not an artifact-type matrix.

## Components

| File | Role |
| --- | --- |
| `library.py` | Canonical journey registry plus legacy helper factories kept for focused evidence tests. |
| `scripted_llm.py` | Deterministic mock LLM that routes by tool schemas and response formats. |
| `driver.py` | Runs user actions through HTTP, drains graph work, and captures snapshots. |
| `recorder.py` | Builds transcript objects and writes them to a caller-provided directory. |
| `eval_support.py` | Scores produced artifacts with a mock or real judge. |
| `conftest.py` | Provides the file-backed scenario DB, graph, checkpointer, and LLM patches. |

## Canonical Journeys

`CANONICAL_SCENARIOS` is the shared high-level source of truth for integration,
eval, benchmark, and live-smoke lanes:

- `canonical-clarify-draft-approve`: clarify one missing context point, draft,
  and approve.
- `canonical-reject-critique-redraft`: reject the first draft, run critique,
  clarify, and approve the revised draft.
- `canonical-context-artifact-from-predecessors`: derive a downstream artifact
  from accepted predecessor context.

Add a new canonical journey only when it protects a distinct end-to-end risk
that cannot be covered by a lower-level contract test.

## Output Policy

Tests should write transcripts to `tmp_path` by default. Explicit evidence or
live runs may pass a stable output directory intentionally.

## Run

```bash
pytest tests/integration/scenarios/test_scenarios.py -q
pytest -m eval tests/eval/test_behavior_scenarios.py -q
pytest -m benchmark tests/benchmark/test_baseline_benchmark.py -q
```
