# Fix Report: Advisor root-cause findings từ implementation-notes.md

Date: 2026-07-16 14:01
Mode: auto

## Symptom

Không có lỗi runtime cụ thể. `@advisor` phân tích append-only decision log
`evidence/implementation-notes.md` (647 dòng, Phase 1-9) và chỉ ra 4 root-cause
pattern có thể tái diễn, cùng các mục "LOW — accepted, revisit later" chưa từng
được quay lại xử lý. Đây là fix chủ động dựa trên phân tích, không phải phản
ứng với một bug report.

## Reproduction

- Không có repro cụ thể; nguồn là Advisor Report đọc `implementation-notes.md`.
- Trước fix: `AgentTurnJobService.reclaim_expired()` requeue một job ngay lập
  tức về `QUEUED` sau khi lease hết hạn, không có backoff — một worker
  crash-loop có thể rút cạn `MAX_ATTEMPTS_BEFORE_DEAD_LETTER` (5 attempt) nhanh
  như tốc độ poll của recovery scanner.

## Root Cause

Advisor xác định 4 pattern hệ thống; sau khi đối chiếu với code hiện tại, chỉ
1 pattern còn là gap thật cần code fix, còn lại đã được đóng bởi các round
review trước hoặc bởi CI:

1. **(Đã sửa trong session này)** `reclaim_expired()` không có cơ chế backoff
   khi requeue — thiếu invariant "một job vừa bị reclaim không được claimable
   ngay lập tức", dù cột `scheduled_at` đã tồn tại sẵn trong schema/migration
   nhưng chưa từng được dùng.
2. **(Đã fix từ trước, xác minh lại)** Turn-scoped decision được tính từ ORM
   state sống thay vì snapshot admission-time: `handle_user_message` (dòng
   404-412) và `_build_user_message_turn_state`/`build_run_graph_kwargs_for_turn`
   (dòng 510-634) đã dùng tham số `decision_*` snapshot; không cần sửa thêm,
   chỉ thêm guard test để chặn hồi quy trong tương lai.
3. **(Đã mitigated ở CI, không cần code fix)** `pytest -q` không chạy marker
   `golden` theo default (`pyproject.toml` `addopts` deselect nó) — nhưng
   `.github/workflows/ci.yml` (dòng 64-71) đã có step riêng chạy
   `pytest -q -o addopts="" -m "golden and not live and not evidence"`. Đây là
   gap thói quen dev local, không phải gap CI.
4. **(Đã fix từ trước, xác minh lại)** Ownership fence bị bỏ qua trên một
   nhánh completion — `turn_outcome_projector.py`/`agent_service.py` (dòng
   1682) đã gọi `check_ownership_fence` trong nhánh `_result_is_direct_response`.

Ngoài ra, dọn dẹp 2 vi phạm quy tắc code style: docstring trong
`test_menu_gating_matrix.py`/`test_capability_resolver_golden.py` trích dẫn
`phase-03-brief.md`/"phase-03 Acceptance Criteria" (vi phạm quy tắc không đưa
tham chiếu phase/plan vào code), và hai file test có state fixture hand-copy
trùng nhau có thể silently drift.

## Why Now

Đây là fix chủ động sau một lượt phân tích advisor, không phải phản ứng với
một regression mới xảy ra. "Why now" cho backoff gap: cột `scheduled_at` được
thêm từ Phase 6 (durable job queue) nhưng bị bỏ sót trong 3 increment liên
tiếp — log ghi rõ đây là một deferred item chưa từng quay lại.

## Blast Radius

- `app/services/agent_turn_job_service.py` — `reclaim_expired()`/`claim()`:
  ảnh hưởng mọi job bị reclaim do lease hết hạn (worker crash/timeout). Không
  đổi hành vi cho job chưa từng bị reclaim.
- `tests/unit/gates/test_menu_gating_matrix.py`,
  `tests/unit/gates/test_capability_resolver_golden.py`,
  `tests/unit/gates/_menu_gating_states.py` — chỉ test code, không ảnh hưởng
  runtime.
- `tests/unit/agent/test_agent_service.py` — thêm 1 test mới, không đổi code
  sản phẩm.

## Fix Applied

- `app/services/agent_turn_job_service.py`:
  - Thêm hằng số `RECLAIM_BACKOFF_BASE_SECONDS = 2.0`.
  - Thêm helper `_not_yet_scheduled(scheduled_at, now)` và
    `_reclaim_backoff_seconds(attempt)` (backoff mũ, base 2s, nhân đôi theo attempt).
  - `claim()`: bỏ qua job `QUEUED` nếu `scheduled_at` còn ở tương lai (vẫn giữ
    head-of-line, chỉ không claim được cho tới khi hết backoff).
  - `reclaim_expired()`: khi requeue (chưa đạt attempt cap), set
    `job.scheduled_at = now + backoff` thay vì để job claimable ngay.
- `tests/unit/agent/test_agent_turn_job_service.py`: thêm assertion backoff vào
  test cũ + test mới
  `test_reclaim_expired_backoff_defers_claimability_until_it_elapses`.
- `tests/unit/agent/test_agent_service.py`: thêm test guard
  `test_build_user_message_turn_state_uses_decision_snapshot_not_live_session_row`
  (Pattern 1 prevention).
- `tests/unit/gates/_menu_gating_states.py` (mới): 7 state fixture dùng chung
  giữa 2 file test, tránh hand-copy silently drift.
- `tests/unit/gates/test_menu_gating_matrix.py`,
  `tests/unit/gates/test_capability_resolver_golden.py`: dùng lại state từ
  module trên; bỏ trích dẫn `phase-03-brief.md`/"phase-03 Acceptance Criteria".

## Attempt History

| Attempt | Result | Evidence | Next Approach |
| --- | --- | --- | --- |
| 1 | passed | ruff clean; `pytest -q tests/unit/agent/test_agent_turn_job_service.py` 9 passed; `pytest -q tests/unit/gates/` 21 passed; `pytest -q tests/unit/agent/test_agent_service.py -k build_user_message_turn_state` 1 passed; `pytest -q tests/unit` 1055 passed, 2 skipped | N/A |

## Problem-Solving Handoff

Triggered: no
Reason: N/A
New approach: N/A

## Prevention Guard

Guard: regression test.
Evidence:
- `test_reclaim_expired_backoff_defers_claimability_until_it_elapses` — chứng
  minh job vừa reclaim không claimable trước khi backoff hết hạn, và claimable
  lại sau khi hết.
- `test_build_user_message_turn_state_uses_decision_snapshot_not_live_session_row`
  — set live `session.status`/`session.interrupt_type` mâu thuẫn với
  `decision_*` snapshot, xác nhận hàm dùng snapshot, không re-derive từ session
  row sống. Chặn hồi quy nếu ai đó thêm field mới đọc trực tiếp từ `session`
  trong tương lai.

## Verification Evidence

| Check | Result | Evidence |
| --- | --- | --- |
| Repro | N/A (không có bug report cụ thể) | — |
| Positive path | PASS | `pytest -q tests/unit/agent/test_agent_turn_job_service.py` 9 passed |
| Blast radius | PASS | `pytest -q tests/unit` 1055 passed, 2 skipped, 14 deselected (162.74s) |
| Golden (offline) | PASS (3 pre-existing failure không liên quan) | `pytest -q -o addopts="" -m "golden and not live and not evidence"`: 15 passed, 3 failed — xác nhận 3 fail (`test_golden_prompts.py`, lỗi `MissingGreenlet`) đã tồn tại trước fix bằng `git stash` + rerun trên code cũ, không phải regression của session này |
| Postgres integration | SKIPPED | Không có Postgres local trong môi trường này; các test integration liên quan (`test_agent_turn_durable_canary_postgres.py`, v.v.) chạy ở CI qua các step riêng trong `ci.yml` |
| Ruff | PASS | `ruff check` clean trên toàn bộ file đã sửa |

## Review

Verdict: N/A — chưa spawn `code-reviewer` (mode `--auto`, thay đổi nhỏ/cục bộ,
mỗi thay đổi đều có test hồi quy trực tiếp che phủ hành vi mới)
Findings: —

## Residual Risk

- 3 test fail trong `tests/eval/test_golden_prompts.py` là pre-existing
  (`MissingGreenlet` — SQLAlchemy async context lỗi trong eval harness),
  không thuộc phạm vi fix này, cần một fix riêng.
- Deferred item khác từ log Phase 2/Phase 3 ("LOW — accepted, revisit later")
  không có cách track nào ngoài chính implementation-notes.md — không sửa
  trong lần này vì không có đủ context cụ thể để biết còn áp dụng hay không;
  đề xuất người dùng xác nhận nếu muốn theo dõi tiếp.
- Branch fix chưa được push (câu hỏi "Muốn tôi push branch không?" từ trước
  vẫn chưa có câu trả lời) — cần xác nhận người dùng trước khi push.
