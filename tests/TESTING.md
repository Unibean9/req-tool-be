# Hướng dẫn test suite

Tài liệu này mô tả các tier (marker) test, lane mặc định của CI, cách chạy từng tier, và bản đồ cụm thư mục `tests/unit`.

## Lane mặc định (đúng bằng CI)

CI chạy đúng lệnh:

```bash
uv run pytest -q      # hoặc: python -m pytest -q
```

`pyproject.toml [tool.pytest.ini_options].addopts` đã cấu hình sẵn:

```
--import-mode=importlib -m "not eval and not benchmark and not live and not golden and not evidence"
```

Nghĩa là lane mặc định **loại trừ** 5 tier `eval / benchmark / live / golden / evidence` — chúng tốn kém hoặc cần credential, cố ý chỉ chạy khi gọi tường minh. `integration` **có** chạy trong lane mặc định.

Mốc tham chiếu (2026-07-13): ~1119 passed / 2 skipped / 17 deselected, ~3m15s.

## Bảng marker

| Marker | Ý nghĩa | Chạy trong lane mặc định? |
| --- | --- | --- |
| `integration` | Test xuyên lớp: API, DB, graph, hoặc service composition | Có |
| `eval` | Đo chất lượng hành vi (behavior quality) | Không — `-m eval` |
| `golden` | Snapshot / golden-contract | Không — `-m golden` |
| `benchmark` | Hiệu năng, latency, token | Không — `-m benchmark` |
| `live` | Gọi LLM/judge thật bên ngoài | Không — `-m live` + credential |
| `evidence` | Ghi transcript, metric, report artifact | Không — `-m evidence` |

## Lệnh chạy từng tier

```bash
python -m pytest -m eval           # đo chất lượng hành vi
python -m pytest -m golden         # golden/snapshot contract
python -m pytest -m benchmark      # latency/token
python -m pytest -m evidence       # sinh artifact bằng chứng
python -m pytest -m live           # cần credential LLM/judge thật
```

## Khi nào chạy tier nào

- **`golden` / `eval`**: chạy TRƯỚC khi sửa prompt / instruction / decision contract — đây là guard cho các success metric của agent (xem plan 260712). Prompt đổi mà golden/eval không chạy = mù regression chất lượng.
- **`benchmark`**: khi thay đổi có thể ảnh hưởng latency hoặc token budget.
- **`live`**: khi cần xác nhận hành vi với LLM/judge thật; cần credential provider (Anthropic/OpenAI/Google/Bedrock).
- **`evidence`**: khi cần sinh transcript/report để review thủ công.

> **Lưu ý mis-tier:** `tests/eval/test_eval_baseline.py::test_baseline_with_real_judge` gọi Bedrock thật nên **cần credential** dù chỉ mang marker `eval` (không có `live`). Khi chạy `-m eval` không có credential AWS, test này sẽ fail với `UnrecognizedClientException` — đây là vấn đề môi trường, không phải regression. Cân nhắc gắn thêm marker `live` cho nó trong tương lai.

## Bản đồ cụm `tests/unit`

`tests/unit` được gom thành 5 cụm theo domain để dễ tìm/bảo trì (di chuyển bằng `git mv`, collection không đổi):

| Cụm | Nội dung |
| --- | --- |
| `agent/` | agent_service, agent_models/events/router/checkpointer, graph_*, orchestrator, session_phase, lifecycle_*, core_loop, structured_state, state_fields, analysis_decomposition, context_loader, scenario_harness, workflow_mode, foundation, grace_critique |
| `gates/` | menu_gating_*, gating_engine, gate_logging, gate_stack_minimal, dispatch_* , finalize_*_gate, write_draft_*_gate, deterministic_gate_wiring, illegal_phase_metric, triage_heuristic_skip, dropped_tool_feedback |
| `tools/` | run_critique/confirm_intent/elicit/ask_user_batched/web_search tool, note_tool_merge, explore_tools_gating (note-tool), tool_error_recovery/i18n, payload_caps |
| `llm/` | llm_clients/providers/usage, phase_prompt_profiles, instruction_contract, prompt_prefix_stability, thinking_mode_block, turn_audit_token_calibration, judge_smoke/config, jsonschema_parity_and_budget, key_facts, feedback_summary_contract, rubric |
| `artifact/` | artifact_*, decision_node_*/decision_graph_views, section_*, render_*, registry_merges, event_storming_registry, impact_traversal, coverage_stall, executive_summary, bmad_*, ux_reliability_fixes, run_impact_analysis_feedback |

Các file lẻ (`test_validators.py`, `test_project_delete_cascade.py`) và `conftest.py` giữ ở gốc `tests/unit/`.

Shared helper import bằng đường dẫn tuyệt đối `tests.*` (`tests.conftest`, `tests.factories`, `tests.helpers`, `tests.golden_decision_helpers`) nên không phụ thuộc vị trí file test. `tests/unit/conftest.py` cascade tự động xuống mọi cụm.

> **Cảnh báo khi viết test đọc cây source:** đừng suy ra đường dẫn `app/` bằng `Path(__file__).parents[N]` — nó vỡ (hoặc tệ hơn, pass giả) khi file test đổi cụm. Anchor theo package: `Path(importlib.util.find_spec("app").submodule_search_locations[0])`.
