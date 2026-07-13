# Testing

The default pytest lane is for routine regression work. It is intentionally
deterministic and excludes tests that measure behavior quality, call live
providers, update golden files, write transcripts, or generate benchmark
evidence.

## Default Regression

```bash
pytest
```

The default lane excludes these markers through `pyproject.toml`:

- `eval`
- `benchmark`
- `live`
- `golden`
- `evidence`

Default tests must not write to committed fixture, transcript, or plan evidence
paths. Use `tmp_path` for temporary output, or mark the test with the appropriate
explicit lane marker.

## Explicit Lanes

Run behavior evals intentionally:

```bash
pytest -m eval
```

Run prompt/schema golden checks intentionally:

```bash
pytest -m golden
```

Run benchmark evidence intentionally:

```bash
pytest -m benchmark
```

Run live external-provider checks only with the required credentials:

```bash
pytest -m live
```

Run cross-layer integration checks:

```bash
pytest -m integration
```

## Maintenance Rules

- Prefer low-level contract tests for pure prompt assembly, state builders,
  reducers, validators, and tool gating.
- Keep scenario tests as canaries for critical user-visible workflows, not as
  the only guard for every branch.
- Reuse `tests.integration.scenarios.library.CANONICAL_SCENARIOS` for
  integration, eval, benchmark, and live-smoke coverage. Do not create a
  separate high-level scenario list unless it protects a distinct risk.
- Do not add a broad golden snapshot unless the full serialized output is the
  reviewed product.
- Use `tests.factories` for workflow state, graph config, agent sessions, agent
  runs, projects, and focused document items.
- Do not import helpers from test modules such as
  `tests.integration.test_graph_nodes` or `tests.integration.test_tool_parity`.
- Put DB, HTTP, graph, checkpointer, or ToolNode composition tests under
  `tests/integration`, even if some assertions call implementation helpers
  directly.
- Legacy format checks should be named as compatibility tests and should not
  leak legacy fields into default factories.
