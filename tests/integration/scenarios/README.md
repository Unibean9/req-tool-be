# Agent behavior scenario tests (API-level)

This suite drives **multi-turn conversations through the real HTTP API** voi LangGraph that,
records **all raw messages, payloads, and tool calls** vao transcript JSON, va scores
generated artifact quality with a judge.

## Components

| File | Role |
|------|---------|
| `scripted_llm.py` | Deterministic mock LLM. Routes moi `generate()` theo `response_format`/`system` ve scripted responses: intent / analyze (brain) / summarize / critic / regenerate. Helper: `ask()`, `propose()`, `artifact()`. |
| `driver.py` | `ScenarioDriver` runs scenarios through HTTP (create session → send message → list messages/tool-calls → approve/reject), drain graph between steps, ghi snapshot. |
| `recorder.py` | Record transcript JSON per scenario into `transcripts/`. |
| `library.py` | Define behavior scenarios. |
| `eval_support.py` | Judge (mock default) scores artifact theo rubric 8 criteria. |
| `conftest.py` | Engine SQLite file rieng + checkpointer that + cac patch binding + fresh session/request. |
| `test_scenarios.py` | Run each scenario, assert API contract + payload envelope, scores eval, ghi transcript. |
| `test_documents.py` | Run full pipeline BA→PM theo thu tu predecessor, gom moi artifact thanh "tai lieu BRD/PRD" roi scores tokg hop. |

## Pham vi artifact type

`ALL_SCENARIOS` has one scenario happy-path cho moi type trong pipeline BA→PM:
`intent → problem → stakeholder → goal → functional_requirement →
non_functional_requirement → epic → story` (plus behavior scenarios intent:
multi-turn, reject...). `DOCUMENT_PIPELINE` runs this sequence in the same project
so predecessor constraints are satisfied. `capability` intentionally skipped → `functional/
non_functional_requirement` mang `missing_context=["capability"]` soft (non-blocking).
Type `goal` is checked by the branch SMART, `story/epic` boi nhanh INVEST cua validator
so scenario bodies intentionally include measurement/timeframe and acceptance criteria.

## Run

```bash
# Full suite (mock LLM + mock judge, deterministic, no API key required)
PYTHONIOENCODING=utf-8 python -m pytest tests/scenarios -q

# One specific scenario
python -m pytest "tests/scenarios/test_scenarios.py::test_behavior_scenario[multi-turn-qna]" -q
```

After running, transcript lives at `tests/scenarios/transcripts/<scenario-name>.json`
(gom tung buoc hanh dong + snapshot messages/tool_calls + diem eval). Cac file nay
is gitignored because it is generated on each run.

## Add a new scenario

Trong `library.py`, viet ham tra ve `Scenario(name, artifact_type, llm, actions, expect)`:

- `llm = ScriptedLLM(brain=[...])` — list of analyze turns (use `ask()` / `propose()`).
- `actions` — chuoi hanh dong: `{"type": "send", "content": "..."}`, `{"type": "approve_all"}`, `{"type": "reject_all"}`.
- `expect` — `{"final_status": ..., "min_artifacts": ...}`.

Add function to `ALL_SCENARIOS` to be parametrized automatically.

## Cham bang judge that

`eval_support.mock_judge()` la default. De scores bang LLM that, thay bang client
created from `tests/eval/config.py` (needs `LLM_API_KEY` in `.env.test`) — xem
`tests/eval/test_eval_baseline.py` cho mau marker `integration`.
