# ADR: Logical turn là control plane bền vững của agent

Status: Accepted
Date: 2026-07-14

## Context

Runtime hiện tại là bounded-ReAct: graph điều phối vòng model/tool, còn session,
approval, checkpoint và background task cùng góp phần diễn giải trạng thái. Cách
này giữ được rào chắn nghiệp vụ nhưng chưa có một biên thực thi bền vững để phân
biệt input bên ngoài, attempt của graph, effect nghiệp vụ và terminal outcome.
Vì vậy retry, nhiều worker, crash giữa effect và checkpoint, hoặc approval trùng
có thể suy diễn khác nhau tùy lớp đang xử lý.

Phase 1 chỉ chốt hợp đồng migration và thêm cờ tắt mặc định. REST, SSE, graph,
checkpoint và route production hiện hữu chưa thay đổi trong phase này.

## Decision

### Cardinality và trigger

Một external trigger hợp lệ — user message, approval, cancel hoặc retry — mở một
logical turn. Graph resume và mỗi lần gọi model chỉ là execution attempt của turn
đó, không phải turn mới. `AgentRun`, metadata của message và provider tool-call
ID chỉ có giá trị audit hoặc transport; chúng không được dùng thay turn identity,
sequence, version hay logical command identity.

Phase 2 sẽ tạo additive aggregate `AgentTurn` trước khi runner nhận việc:

- `AgentTurnEnvelope` bất biến giữ turn ID, sequence, trigger gốc, cohort,
  principal/config snapshot và correlation.
- `TurnExecutionState` versioned giữ ownership, fence, attempt, transition và
  tham chiếu terminal. Mọi mutation dùng compare-and-swap trên turn ID,
  transition version và ownership generation.

Approval là typed trigger để resume execution. Nó không được trực tiếp đặt
session thành `COMPLETED`; approval trùng hoặc đến sai thứ tự phải được xử lý theo
identity/precedence của trigger.

Quyết định làm rõ ngày 2026-07-15: kể cả approval cuối cùng của một batch proposal
cũng chỉ tạo ordered resume trigger. Chỉ graph/terminal owner, sau khi resume,
được phát `TurnOutcome` terminal; approval command không được project terminal
outcome trực tiếp.

### State và terminal

`AgentSession.status` không còn là nơi nén conversation, execution, waiting,
approval, retry, business state và UI state. `TurnOutcome` là owner duy nhất của
terminal decision. Session status và payload SSE là projection tương thích từ
outcome trong các phase cutover; public payload không đổi cho đến khi có hợp đồng
API riêng.

Taxonomy outcome, capability decision, logical command/effect identity và
execution fence được định nghĩa có kiểu trước khi enforce. Trace chỉ ghi dữ liệu
đã allowlist/redact; không persist chain-of-thought, secret, prompt nhạy cảm hay
tool argument chưa redaction.

### Replay, effect và fencing

Nếu crash sau khi đã nhận model decision chuẩn hoá nhưng trước checkpoint, hệ
thống persist rồi reuse decision đó; không gọi lại LLM để suy đoán lại. Logical
command identity là business identity ổn định, không phải provider tool-call ID.
Mỗi mutation effect, checkpoint, outcome và outbox phải xác thực execution fence
ngay trong cùng transaction; executor lease cũ không được ghi sau khi ownership
đã chuyển.

Source of truth là transactional `AgentTurn` aggregate cùng ordered outbox, không
phải event log thuần. Event/history là projection hoặc audit có thứ tự, authorization
và cursor rõ ràng. Cross-store reconciliation phải phát hiện các cửa sổ crash giữa
domain effect, checkpoint và outbox thay vì diễn giải từ blob checkpoint mới nhất.

### Cohort, compatibility và rollout

Cohort/flow version và effective flags được snapshot lúc admission. Rollback cờ
chỉ áp dụng turn mới; turn đã admitted tiếp tục theo cohort đã persist. Migration
theo expand-contract: thêm contract và reader tương thích trước, shadow/observe,
enforce theo cohort, sau đó mới durable worker và decommission legacy khi có số
liệu chứng minh. Session cũ dùng compatibility adapter với default legacy, không
suy đoán cohort từ shape của checkpoint.

Các cờ Phase 1 là `AGENT_TURN_ADMISSION_ENABLED=false`,
`AGENT_POLICY_RESOLVER_MODE=legacy`, `AGENT_COMMAND_HANDLERS_ENABLED=false`,
`AGENT_EXECUTION_MODE=inline`, `AGENT_CHECKPOINT_HISTORY_ENABLED=false` và
`AGENT_TRACE_ENABLED=false`. Giá trị hợp lệ của policy resolver là `legacy`,
`shadow`, `enforce`; execution mode là `inline`, `durable`.

## Addendum 2026-07-15 — Quyết định vận hành Phase 6

Owner vận hành chốt các câu hỏi mở tại Phase 1 trước khi Phase 6 bắt đầu implement:

- **Worker topology**: worker chạy bằng một entrypoint/CLI riêng, không phải một web
  replica chạy thêm vòng lặp claim. Chỉ một nơi duy nhất (web hoặc job CI riêng, không
  phải mọi replica) chạy `alembic upgrade` khi deploy; worker entrypoint không tự chạy
  migration.
- **Lease/retry SLO**: lease mặc định 60s, heartbeat renew mỗi ~20s (thấp hơn 1/3 lease
  để chịu được một lần renew bị trễ/mất trước khi lease hết hạn); retry dùng exponential
  backoff base 2s, tối đa 5 lần; hết 5 lần chuyển dead-letter — dead-letter yêu cầu hành
  động operator tường minh (requeue/cancel/supersede), không tự động retry vô hạn. Các
  giá trị này cấu hình qua settings/env, không hard-code, để chỉnh theo SLO thực đo sau
  canary.
- **Canary scope Phase 6**: `agent_execution_mode=durable` chỉ được exercise trong
  test/CI (Postgres integration, hai-worker, fault-injection) ở phase này; không bật ở
  bất kỳ môi trường dev/staging/prod thật nào cho đến khi có dashboard/runbook vận hành
  riêng — việc đó thuộc phạm vi rollout sau Phase 6, không phải exit criterion của phase
  này.

## Addendum 2026-07-16 — Checkpoint v2 saver selection (Phase 7)

Resolves the Phase 7 "Might Be" assumption on `AsyncPostgresSaver`.

- **Decision**: Phase 7 checkpoint v2 stays a custom `BaseCheckpointSaver` subclass
  (extending the existing `AgentSessionCheckpointer` pattern in
  `app/graphs/checkpointer.py`) writing history rows through the same
  SQLAlchemy/asyncpg session used by turn admission, not the official
  `langgraph-checkpoint-postgres` `AsyncPostgresSaver`.
- **Why**: `AsyncPostgresSaver` (package `langgraph-checkpoint-postgres`, latest
  3.1.0 as of 2026-05-12) requires the `psycopg` (async) driver with its own
  connection/pool (`autocommit=True`, `row_factory=dict_row`) — a second Postgres
  driver next to the repo's pinned `asyncpg` + SQLAlchemy async stack. It manages
  its own commit boundary and has no notion of this codebase's turn ownership
  fence, so the Phase 7 requirement "append history only with expected parent +
  session sequence + active fence generation compare-and-set" (Constraints,
  Phase 7 Step 3) cannot be expressed as a single transaction against its API —
  it would need wrapping/subclassing anyway, at the cost of a duplicate driver
  and duplicate pool lifecycle. The existing custom saver already commits inside
  the same transaction as `TurnExecutionState`/`AgentTurnJob` writes, so extending
  its schema for parent-linked history plus a CAS predicate keeps one driver, one
  pool, and one transactional boundary.
  - Source: `pyproject.toml` (`asyncpg>=0.30.0`, `sqlalchemy[asyncio]>=2.0.40`,
    `langgraph==1.2.6`; no `langgraph-checkpoint-postgres` dependency); installed
    `langgraph-checkpoint==4.1.1` (`pip show`); PyPI project page and LangChain
    reference docs for `AsyncPostgresSaver` (`langgraph.checkpoint.postgres.aio`),
    fetched 2026-07-16. Confidence: high on driver/version facts, medium on
    whether a future LangGraph release changes this trade-off — re-check before
    reopening this decision.
- **Consequence**: v2 checkpoint history table and CAS logic are new code
  maintained in this repo rather than delegated to an upstream saver; Phase 7
  must implement parent/session-sequence/fence-generation CAS append itself
  (Step 3) instead of getting it from the library.

## Addendum 2026-07-16 — Phạm vi rút gọn và attribution `AgentRun.turn_id` (Phase 8)

Phase 8 gốc (`phase-08-quality-observability-decommission.md`) có 6 bước: mở rộng
CI gate, nightly live-eval workflow, redaction trace/audit, deploy gate topology,
dashboard/alert/runbook và decommission checklist. Quyết định của người vận hành
ngày 2026-07-16 giới hạn phiên làm việc này vào phần thực sự verify được bằng
code/CI:

- **Mở rộng CI gate bắt buộc**: cả 6 file `tests/integration/*_postgres.py` (trước
  đó chỉ 1/6 chạy trong CI) nay chạy trong job `agent-turn-postgres`, và
  `deploy.needs` vẫn phủ đủ mọi job — không cần đổi topology gate hiện có
  (`needs`-based), chỉ cần lấp khoảng trống độ phủ lane Postgres.
- **Trace/audit correlation**: thêm cột attribution `AgentRun.turn_id` (nullable,
  FK tới `agent_turn_envelopes.id`, additive migration) — **không phải turn
  identity**, chỉ để on-call join trigger → turn → attempt (`AgentRun`) →
  command (`DraftCommandLedger`) → outcome (`TurnOutcome`) → event
  (`AgentTurnEvent`) từ một `correlation_id`. `record_run_and_dispatch` nhận
  `turn_id` optional; giá trị lấy từ `cfg.get("turn_id")` đã có sẵn trong
  `analyze_node`, không cần plumbing mới.
- **Redaction regression test**: `_audit_arg_value`/`_audit_text_value` trong
  `turn_audit.py` đã redact các key trong `_AUDIT_TEXT_ARG_KEYS` từ trước; phase
  này thêm test end-to-end chứng minh giá trị secret không tồn tại verbatim
  trong `AgentRun.analysis_result` đã persist, không đổi logic redaction.

**Không làm trong phiên này** (ghi nhận rõ, không âm thầm bỏ qua): nightly/on-demand
live-provider workflow; dashboard/alert nối vào stack quan sát thật (chưa có stack
nào trong repo — không có Grafana/Datadog/Prometheus config); decommission
checklist thực thi (cần sign-off nghiệp vụ, không phải quyết định code). Hai file
placeholder (`docs/runbooks/observability-spec.md`,
`docs/runbooks/decommission-checklist.md`) được viết để có chỗ đặt tên
owner/threshold, đánh dấu rõ `NOT ENFORCED` / `NOT STARTED`, không claim là đã
hoàn thành.

## Alternatives

- Giữ `AgentSession.status` làm source of truth: ít thay đổi trước mắt nhưng tiếp
  tục trộn state axes và không giải quyết được crash/retry đa worker.
- Dùng `AgentRun` hoặc provider tool-call ID làm turn ID: thuận tiện theo model
  invocation nhưng không biểu diễn được user/approval/retry semantic.
- Gọi lại LLM khi thiếu checkpoint: đơn giản về storage nhưng có thể phân nhánh
  khác và sinh effect trùng.
- Chọn event log thuần làm source of truth: phù hợp khi đã có event-sourcing đầy
  đủ, nhưng hiện chưa có ordering, auth replay và projector đủ để thay aggregate.

## Consequences

- Các phase tiếp theo phải giữ immutable envelope, mutable execution state và
  fence xuyên cả inline/durable runner.
- Command handler, outcome projector và outbox làm tăng số boundary có transaction
  rõ ràng, đổi lại retry/recovery có thể kiểm chứng được.
- Cờ rollout cho phép dừng cutover cho turn mới mà không thay semantics của turn
  đang chạy; chúng chưa có tác dụng route ở Phase 1.
- Deterministic contract test là required CI từ phase này; live provider vẫn là
  lane tách riêng và không phải điều kiện của contract lane.
