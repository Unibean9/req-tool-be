# Bộ test kịch bản hành vi Agent (API-level)

Bộ test này drive **hội thoại nhiều lượt qua HTTP API thật** với LangGraph thật,
ghi lại **toàn bộ raw message + payload + tool-call** vào transcript JSON, và chấm
chất lượng artifact tạo ra bằng judge.

## Thành phần

| File | Vai trò |
|------|---------|
| `scripted_llm.py` | LLM giả lập deterministic. Định tuyến mỗi `generate()` theo `response_format`/`system` về phản hồi đã kịch bản: intent / analyze (brain) / summarize / critic / regenerate. Helper: `ask()`, `propose()`, `artifact()`. |
| `driver.py` | `ScenarioDriver` chạy kịch bản qua HTTP (tạo session → gửi message → list messages/tool-calls → approve/reject), drain graph giữa các bước, ghi snapshot. |
| `recorder.py` | Ghi transcript JSON theo từng kịch bản vào `transcripts/`. |
| `library.py` | Định nghĩa kịch bản hành vi. |
| `eval_support.py` | Judge (mock mặc định) chấm artifact theo rubric 8 tiêu chí. |
| `conftest.py` | Engine SQLite file riêng + checkpointer thật + các patch binding + fresh session/request. |
| `test_scenarios.py` | Chạy từng kịch bản, assert hợp đồng API + payload envelope, chấm eval, ghi transcript. |
| `test_documents.py` | Chạy full pipeline BA→PM theo thứ tự predecessor, gom mọi artifact thành "tài liệu BRD/PRD" rồi chấm tổng hợp. |

## Phạm vi artifact type

`ALL_SCENARIOS` có 1 kịch bản happy-path cho mỗi type trong pipeline BA→PM:
`intent → problem → stakeholder → goal → functional_requirement →
non_functional_requirement → epic → story` (cộng các kịch bản hành vi intent:
multi-turn, reject…). `DOCUMENT_PIPELINE` chạy đúng chuỗi này trong cùng project
để predecessor được thoả. `capability` được bỏ qua có chủ đích → `functional/
non_functional_requirement` mang `missing_context=["capability"]` mềm (không chặn).
Type `goal` được kiểm bởi nhánh SMART, `story/epic` bởi nhánh INVEST của validator
nên body kịch bản cố tình có yếu tố đo lường/thời hạn và tiêu chí chấp nhận.

## Chạy

```bash
# Toàn bộ suite (mock LLM + mock judge, deterministic, không cần API key)
PYTHONIOENCODING=utf-8 python -m pytest tests/scenarios -q

# Một kịch bản cụ thể
python -m pytest "tests/scenarios/test_scenarios.py::test_behavior_scenario[multi-turn-qna]" -q
```

Sau khi chạy, transcript nằm ở `tests/scenarios/transcripts/<tên-kịch-bản>.json`
(gồm từng bước hành động + snapshot messages/tool_calls + điểm eval). Các file này
được gitignore vì là output sinh ra mỗi lần chạy.

## Thêm kịch bản mới

Trong `library.py`, viết hàm trả về `Scenario(name, artifact_type, llm, actions, expect)`:

- `llm = ScriptedLLM(brain=[...])` — danh sách lượt analyze (dùng `ask()` / `propose()`).
- `actions` — chuỗi hành động: `{"type": "send", "content": "..."}`, `{"type": "approve_all"}`, `{"type": "reject_all"}`.
- `expect` — `{"final_status": ..., "min_artifacts": ...}`.

Thêm hàm vào `ALL_SCENARIOS` để được parametrize tự động.

## Chấm bằng judge thật

`eval_support.mock_judge()` là mặc định. Để chấm bằng LLM thật, thay bằng client
tạo từ `tests/eval/config.py` (cần `JUDGE_API_KEY` trong `.env.test`) — xem
`tests/eval/test_eval_baseline.py` cho mẫu marker `integration`.
