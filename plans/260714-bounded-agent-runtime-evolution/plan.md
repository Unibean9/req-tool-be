# Plan: Tiến hoá runtime agent ReAct có kiểm soát

Status: Draft
Date: 2026-07-14
Mode: Deep
Source: Hai phân tích kiến trúc agentic và ReAct trong session hiện tại

## Design Contract

### Objective

Chuyển dần runtime agent hiện tại từ bounded-ReAct có nhiều nguồn quyết định và background task cục bộ thành control plane có **logical turn** bền vững, ownership/fencing rõ ràng, outcome có kiểu, command idempotent và job worker vận hành được; không big-bang rewrite graph hay public API.

### User / Operator Value

Người dùng vẫn nhận luồng agent và SSE quen thuộc, nhưng ít bị dừng/hoàn tất sai hơn. Vận hành biết chính xác input nào sinh ra execution/effect/outcome nào, retry không tạo effect trùng hoặc rẽ nhánh âm thầm; nhóm phát triển có thể thêm capability mà không sao chép luật qua prompt, menu, dispatch và tool handler.

### Success Metrics

- Mỗi quyết định capability được tạo bởi một `CapabilityResolver` thuần, có log differential trước khi được enforce.
- Mỗi trigger hợp lệ tạo đúng một `AgentTurn` aggregate trước execution: `AgentTurnEnvelope` bất biến mang identity/sequence/trigger/cohort/principal/config/correlation, và `TurnExecutionState` có version mang ownership/fence/attempt/transition/terminal reference; `AgentRun` chỉ là audit của một model invocation.
- `AgentService` nhận `TurnOutcome` có kiểu thay vì suy diễn hoàn tất từ `__interrupt__`, tool call cuối hoặc message cuối.
- Các hiệu ứng ghi draft/finalize có idempotency key và có thể chạy lại an toàn về hiệu ứng nghiệp vụ dù job là at-least-once.
- Không executor nào có thể ghi command effect, checkpoint transition hoặc terminal projection sau khi execution fence/lease generation của nó không còn hợp lệ.
- Turn đang chạy có thể được một worker khác reclaim sau lease hết hạn, không tạo bản ghi nghiệp vụ trùng.
- Session checkpoint v1 đang pause vẫn resume được trong giai đoạn chuyển đổi; session mới chỉ dùng v2 sau khi durable worker chứng minh ổn định.
- CI bắt buộc chạy bộ deterministic regression/golden; nightly tách lane live-provider; deploy phụ thuộc CI xanh.

### Acceptance Criteria

- [ ] Không còn một capability rule mới phải được sao chép vào prompt, menu, dispatch và tool handler để hoạt động.
- [ ] Shadow mode chứng minh parity trên golden matrix và telemetry thực tế trước khi resolver được enforce theo canary.
- [ ] Admission atomically ghi trigger + logical turn + session sequence/idempotency identity trước khi inline hoặc durable runner bắt đầu; hai trigger cùng session không thể advance cùng một thread.
- [ ] Command mutation xác thực business idempotency **và** execution fence trong cùng transaction; provider tool-call ID chỉ là correlation, không phải business identity.
- [ ] Một terminal protocol owner project `TurnOutcome`; `AgentSession.status` và SSE chỉ là compatibility projection, không tự suy diễn terminal nghĩa là gì.
- [ ] Không thay đổi payload REST/SSE hiện hữu cho tới khi adapter tương thích được kiểm chứng.
- [ ] Job có lease, heartbeat, retry policy, cancellation/approval-safe resume và audit đủ để khắc phục sự cố.
- [ ] Checkpoint, business effect và event/outbox có crash-reconciliation contract; plan không giả định transaction nguyên tử xuyên LangGraph saver và domain database.
- [ ] Trace/audit không ghi raw secret, prompt nhạy cảm hoặc tool argument chưa redaction.
- [ ] Mọi cutover có feature flag lưu theo turn/job, rollback và migration additive.

### Not Doing

- Không đổi bounded workflow thành swarm tự trị hoặc multi-agent orchestration.
- Không triển khai vector database, hybrid retrieval, memory dài hạn hay full prompt-injection/DLP program trong kế hoạch này.
- Không thay REST/SSE contract, không rewrite checkpoint đang tồn tại, không dual-write checkpoint v1/v2 từ ngày đầu.
- Không chuyển sang Temporal/external queue trong đợt này; thiết kế port để có thể quyết định sau.

### Constraints

- Giữ deterministic boundary cho phase, lifecycle, approval/HITL và completion; LLM chỉ chọn trong capability hợp lệ.
- Graph hiện tại `orchestrator -> analyze -> ToolNode` tiếp tục là kernel trong các phase đầu.
- Migration database theo expand–contract; code đọc v1 phải tồn tại cho active/paused session cũ.
- Worker không giữ database transaction trong lúc gọi LLM; credentials được resolve trong worker, không persist secret giải mã.
- `AgentTurn` là logical execution boundary gồm envelope bất biến và execution state có version; `AgentMessage`, `AgentRun`, checkpoint và provider tool-call không được dùng thay thế cho identity/ordering của turn.
- `AgentTurnEnvelope` không bị mutate sau admission; claim, heartbeat, reclaim, waiting và terminal transition chỉ mutate `TurnExecutionState` qua CAS/transition version.
- Inline và durable runner phải claim cùng session/turn ownership fence trong giai đoạn mixed mode; process-local lock không phải concurrency guarantee.
- Approval, completion và retry chỉ được thay đổi qua typed trigger/transition contract; không có path trực tiếp đặt terminal session status ngoài projector tương thích.
- Chỉ ghi business effect sau khi xác thực fencing generation trong transaction mutation; side effect external dùng idempotency/outbox riêng.
- `SKIP LOCKED` cần Postgres; test dialect khác phải có giới hạn hoặc integration Postgres thực.
- Feature flag quyết định execution/policy version phải được snapshot vào turn/job để retry không đổi hành vi giữa chừng.

### Implicit Standards

- Ưu tiên thay đổi nhỏ, đo được và rollback được hơn là tái kiến trúc một lần.
- Không persist chain-of-thought để “chứng minh ReAct”; chỉ lưu action, observation, decision summary và outcome cần cho audit.
- Bảo toàn authorization/project scope hiện có cho mọi job background và resume approval.
- Observability là projection sau decision/effect đã commit; pure policy evaluator không ghi log hoặc phụ thuộc log count.
- Mỗi abstraction mới phải có tiêu chí xoá representation legacy; parity không biến import alias/log-count thành public contract vĩnh viễn.

### Assumptions

Must be true:
- Postgres là datastore production và có thể chạy process worker độc lập với Uvicorn web workers.
- Existing session/message/artifact authorization có thể được tái dùng khi worker thực thi command.
- Có thể thêm migration additive cho `AgentTurn`/turn trigger/execution fence mà session/checkpoint v1 cũ vẫn chạy qua compatibility adapter.

Should be true:
- Có môi trường integration Postgres/Redis-like isolation (không cần queue mới) để kiểm tra race lease thực tế.
- Có thể thu thập một tập golden conversation đại diện trước canary.

Might be true:
- `AsyncPostgresSaver` phù hợp checkpoint v2 sau khi xác nhận version LangGraph và migration vận hành; quyết định này chỉ được chốt ở Phase 7.
- Tái sử dụng persisted model decision cho mutation resume có chi phí/churn thấp hơn việc cho phép model re-plan sau crash; Phase 1 ADR phải xác nhận hoặc thay bằng policy fail-safe.

### Open Questions

- Logical turn là một user/approval/system trigger, một graph invocation, hay chuỗi resume đến terminal business outcome? Architecture owner phải chốt tại Phase 1 trước Phase 2.
- Crash sau LLM response nhưng trước checkpoint phải reuse persisted model decision, gọi model lại, hay fail-safe chờ operator? Architecture/Product owner phải chốt tại Phase 1 trước command canary.
- `COMPLETED` là terminal của conversation, graph execution, approval batch hay artifact workflow; owner duy nhất của terminal protocol là ai? Product/Architecture owner phải chốt tại Phase 1 trước Phase 5.
- Approval là command terminal ngoài graph hay resume trigger của graph; event precedence với user message/retry/cancel là gì? Product/Architecture owner phải chốt transition table tại Phase 1, implement Phase 5–6.
- Source of truth là transactional `AgentTurn` aggregate + ordered outbox projection hay event log; retention/audit requirement nào bắt buộc? Architecture/SRE phải chốt ADR tại Phase 1 trước Phase 7.
- Worker deployment topology, migration ownership, retention job/event/trace, SLO/canary/rollback và API cancellation cần owner vận hành chốt trước Phase 6/7. **Đã chốt 2026-07-15** (ADR 0001 Addendum): worker entrypoint riêng, một nơi duy nhất chạy migration; lease 60s/heartbeat ~20s/backoff exp base 2s tối đa 5 lần/dead-letter sau đó; canary Phase 6 chỉ trong test/CI, chưa bật môi trường thật. API cancellation route và retention job/event/trace cụ thể vẫn để mở cho Phase 7 (chưa cần cho exit criteria Phase 6).

### Verification Strategy

- Build: `ruff check .`, import/compile checks phù hợp repo và Alembic head check.
- Test: unit policy/outcome/command; golden differential; integration Postgres cho admission, fence, lease/recovery/checkpoint; crash-point fault injection; existing regression suite.
- CI: Phase 1 đưa baseline deterministic/golden và Postgres contract lane vào required check; mỗi phase thêm invariant của mình vào lane trước canary. Nightly live giữ tách biệt.
- Review: bắt buộc adversarial review các state transition, idempotency, authorization và migration rollback.
- Runtime/manual: canary theo flag, dashboard parity/job lease/outcome, replay một turn lỗi và resume một checkpoint v1 pause.

### Support Checks

| Check | Trigger | Evidence |
| --- | --- | --- |
| testing strategy | Thay đổi state machine, retry và CI | Golden matrix, Postgres admission/fence/two-worker fault injection, required deterministic lane, nightly/live split |
| security hardening | Worker xử lý credential, artifact và trace | Server-side ActorContext trong turn, authorization recheck, secret redaction, wrong-tenant/replay probes |
| observability | Chạy agent bất đồng bộ và canary | Turn/attempt/command/fence correlation, outcome/job metrics, event cursor và alert thresholds |
| migration safety | Bổ sung turn/job/checkpoint schema | Additive migrations, explicit cohort reader, no destructive rollback, compatibility/reconciliation drill |
| documentation/ADR | Chọn ownership/control plane/checkpointer | ADR trong Phase 1 và decision records Phase 6–7 |
| source grounding | LangGraph persistence/HITL thay đổi theo version | Official LangGraph sources và repo pinned versions được kiểm tra ở từng phase |

### Ship Criteria

- Không có mismatch severity-high trong shadow/canary window đã chốt.
- Fault-injection suite cho reclaim, approval interrupt và idempotency xanh trên Postgres thực.
- Read compatibility v1 và rollback flag được diễn tập.
- CI deterministic bắt buộc xanh; nightly live lane không tạo regression chưa triage.

## Target Architecture

```text
request / approval / cancel trigger
          |
atomic admission: AgentTurnEnvelope + trigger + session_sequence + cohort snapshot
          |
TurnExecutionState có version + shared ownership fence (inline hoặc durable runner)
          |
WorkflowSnapshot --pure--> CapabilityResolver --> CapabilityDecision
          |                                      |
          |                           prompt/menu/dispatch projection
          v
bounded LangGraph ReAct kernel --> persisted decision --> typed command
          |                                      |                 |
          |                                      |        fence + idempotent effect
          +----------------- committed TurnOutcome <--------------+
                                  |
             compatibility session/SSE projection + ordered outbox/event
```

LLM vẫn quyết định lý giải và tool argument trong menu hợp lệ. Code là authority duy nhất cho phase/lifecycle/capability, approval, completion và side effect contract.

## Dependency Graph

Foundation:
- Phase 1 (logical turn/replay/terminal ADR, baseline và required CI) -> Phase 2 (AgentTurn admission + shared ownership fence).
- Phase 2 (turn identity, ActorContext và cohort snapshot) -> Phase 3 (shadow/read-only enforce resolver) -> Phase 4–5 (mutating command/HITL).
- Phase 1 (outcome taxonomy) -> Phase 5 (typed outcome projection).

Features:
- Phase 3 (single capability authority) -> Phase 4 (write command + effect fence) -> Phase 5 (finalize/artifact-link approval + terminal protocol).
- Phase 2 (admission fence) + Phase 5 (typed outcome/transition owner) -> Phase 6 (job worker).
- Phase 6 (ordered trigger/job transitions) -> Phase 7 (checkpoint v2 + ordered outbox/event replay).

Surface:
- Phase 1 baseline required CI, Phase 2–7 telemetry -> Phase 8 (quality/observability expansion, deploy gate, decommission).
- Public REST/SSE compatibility adapter spans Phases 5–8; no payload change is a prerequisite for every cutover.
- Phase 1–8 diff -> Phase 9 (comment/docstring hygiene, migration-guard test consolidation, dead-flag decision; housekeeping only, no behavior/contract change).

## Phases

- [x] Phase 1: [Logical turn contract, baseline và migration](phase-01-baseline-migration-contract.md) - Chốt ownership/replay/terminal semantics, golden baseline và CI gate.
- [x] Phase 2: [Logical turn admission và ownership fence](phase-02-logical-turn-admission.md) - Tạo `AgentTurn`, trigger bất biến, sequence và fence dùng chung cho inline/durable.
- [x] Phase 3: [Capability resolver shadow và read-only enforcement](phase-03-capability-resolver-shadow-enforcement.md) - Resolver thuần dựa trên turn snapshot; enforce theo canary cho read-only tools.
- [x] Phase 4: [Command boundary cho draft](phase-04-command-boundary-write-draft.md) - Tách `write_draft` thành logical command idempotent, fenced effect.
- [x] Phase 5: [Finalize, artifact-link HITL và TurnOutcome](phase-05-finalize-hitl-turn-outcome.md) - Hợp nhất approval ownership, terminal protocol và compatibility projection.
- [x] Phase 6: [Durable Postgres turn jobs](phase-06-durable-postgres-turn-jobs.md) - Thay background task cục bộ bằng job worker có lease/recovery.
- [x] Phase 7: [Checkpoint v2, history và event replay](phase-07-checkpoint-v2-history-events.md) - Đưa checkpoint persistent vào session mới, vẫn đọc v1.
- [~] Phase 8 (partial — CI gate + trace/audit correlation slice done, dashboard/live-eval/decommission còn lại): [Quality gates, observability và decommission](phase-08-quality-observability-decommission.md) - Enforce CI/telemetry và bỏ dần đường legacy có điều kiện.
- [x] Phase 9: [Dọn dẹp comment/docstring, test migration trùng lặp và slop code](phase-09-cleanup-migration-tests-comments-slop.md) - Consolidate guard alembic-revision, tiếng Anh hoá comment/docstring, bỏ trích số phase, chốt dứt điểm flag không có consumer.

## Rollout & Rollback

1. Phase 1 khóa ADR về logical turn, model replay, terminal/approval ownership và aggregate/outbox before schema behavior changes; required deterministic CI bắt đầu tại đây.
2. Phase 2 thêm `AgentTurn`/trigger/ownership fence theo expand–contract; mọi runner mới và legacy adapter cùng admission boundary trước canary.
3. Mỗi turn/job persist cohort (`flow_version`, policy/command/execution/checkpoint/outcome versions), ActorContext reference và correlation ID ngay khi admitted.
4. Canary theo project/tenant allowlist, sau đó tỷ lệ traffic nhỏ; dừng tăng nếu admission/fence violation, terminal mismatch, duplicate effect, job retry, latency hoặc error vượt ngưỡng đã chốt.
5. Rollback chỉ đổi allocation cho turn mới; turn đã snapshot version tiếp tục bằng compatibility reader/handler cùng version. Không rollback bằng xóa schema, rewrite state hoặc bỏ qua unresolved turn.
6. Chỉ remove legacy sau retention window, replay/resume/late-worker drill, active cohort scan và migration audit đạt yêu cầu Phase 8.

## Risk Register

| Risk | Impact | Mitigation | Owner |
| --- | --- | --- | --- |
| Resolver làm mất ngữ cảnh menu/dispatch | High | CapabilityDecision chứa evaluation context, golden differential trước enforce | Agent platform |
| Retry job tạo side effect trùng | High | Command idempotency key, unique constraint, effect ledger, two-worker test | Backend |
| Retry gọi LLM lại và rẽ logical action | High | Phase 1 replay ADR, persisted decision hoặc fail-safe policy, crash-point test | Agent platform |
| Stale/inline executor ghi sau ownership mất hiệu lực | High | Shared admission fence, generation check inside every mutation transaction, late-worker test | Backend |
| DB business state và checkpoint split-brain | High | Committed effect/outbox reconciliation contract, interrupt crash fixtures, operator diagnostic | Agent platform |
| Session status lẫn terminal/business/UI state | High | Explicit state-axis/terminal owner ADR, typed projector, no direct terminal writes | Architecture |
| Worker mất principal/secret context | High | Snapshot identity/reference config, resolve encrypted credential trong worker, authorization recheck | Security/Backend |
| Checkpoint v2 không resume state cũ | High | Không convert/dual-write sớm, v1 reader giữ lại, v1 paused resume test | Agent platform |
| Trace rò PII/secret hoặc tăng chi phí | High | Redaction allowlist, sampled payload, retention, access control | Security/SRE |
| Thay worker làm tăng latency | Medium | Queue/execute metrics, canary SLO, inline fallback only for new turns | SRE |
| Scope tràn sang retrieval/multi-agent | Medium | Follow-up trigger/ports rõ ràng, không thêm feature vào phases | Product/Architecture |

## Context & Evidence

Source inputs:
- User yêu cầu tạo plan dần dần từ hai phân tích kiến trúc/re-act trong session.

Scout evidence:
- Graph hiện hữu lặp `orchestrator -> analyze -> ToolNode -> orchestrator/analyze` tại `app/graphs/graph.py` và `app/graphs/nodes.py`; đây là bounded ReAct-style, không phải explicit persisted Thought/Action/Observation.
- Quyết định tool bị lặp giữa `app/graphs/agent_tools/__init__.py`, `app/graphs/analysis/tool_gating.py`, `app/graphs/gating/menu_rules.py`, `app/graphs/gating/dispatch_rules.py`, `app/graphs/session_phase.py` và handler tools.
- `write_draft`/`finalize` tập trung policy, persistence và HITL tại `app/graphs/agent_tools/draft_lifecycle.py`.
- `app/services/agent_service.py` khởi chạy `asyncio.create_task`; production có nhiều Uvicorn worker. `app/graphs/checkpointer.py` hiện chỉ lưu latest checkpoint blob.
- `app/services/agent_event_service.py` polling snapshot; `app/graphs/analysis/turn_audit.py` ghi `AgentRun` nhưng provider/model chưa được populate đầy đủ.
- CI và deploy hiện không bắt buộc toàn bộ deterministic eval/regression lane (`.github/workflows/ci.yml`, `pyproject.toml`).

Plan revision 2026-07-14:
- Blindspot analysis và second adversarial review cho thấy blockers cùng xuất phát từ thiếu logical turn boundary, state ownership, atomicity/replay semantics và cohort compatibility. `blindspot.md` là evidence chi tiết.
- Phase 2 được đổi thành AgentTurn admission/ownership foundation; resolver chỉ tiêu thụ snapshot sau foundation này. Phase 4–7 nhận thêm logical command identity, shared fence, approval transition, outbox/event và checkpoint reconciliation requirements.
- Required deterministic CI được kéo từ Phase 8 về Phase 1; Phase 8 chỉ hoàn thiện live/deploy/observability/decommission.

Research notes:
- LangGraph persistence/interrupt chính thức hỗ trợ durable execution và HITL nhưng side effect phải idempotent: [persistence](https://docs.langchain.com/oss/python/langgraph/persistence), [interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts).
- Tool contracts và loop agent nên tách policy/tool runtime: [LangChain tools](https://docs.langchain.com/oss/python/langchain/tools), [LangChain agents](https://docs.langchain.com/oss/python/langchain/agents).
- ReAct nhấn mạnh interleave reasoning/action/observation, không đòi hỏi lưu chain-of-thought: [ReAct paper](https://arxiv.org/abs/2210.03629).
- Tracing/HITL là capability vận hành độc lập với multi-agent: [OpenAI Agents tracing](https://openai.github.io/openai-agents-python/tracing/), [HITL](https://openai.github.io/openai-agents-python/human_in_the_loop/).
- Agent cần workflow đơn giản, tool rõ và eval thực nghiệm: [Anthropic effective agents](https://www.anthropic.com/engineering/building-effective-agents), [agent evals](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents).

Rejected options:
- Big-bang rewrite sang external orchestrator/Temporal: rủi ro compatibility, chưa có SLO/topology/retention decision.
- Chuyển ngay sang multi-agent: làm tăng bề mặt coordination trước khi single-agent control plane ổn định.
- Dual-write hoặc convert mọi checkpoint hiện hữu: rủi ro resume/HITL cao, khó rollback.

Plan revision 2026-07-15:
- User chọn approval luôn resume graph. ADR `0001` và Phase 5 được làm rõ: approval cuối không
  được project `COMPLETED` trực tiếp; graph terminal owner là producer duy nhất của outcome terminal.

## Session Notes

`cook` sẽ cập nhật trạng thái phase, evidence thực thi và các quyết định đã chốt sau khi implementation bắt đầu.

### 2026-07-14 23:58 - Phase 1

- Status: completed
- Evidence: `evidence/phase-01-evidence.md`
- Decisions: User chấp thuận sáu quyết định control-plane; ADR `0001` là record bền vững.
- Verification: `ruff check app tests`; deterministic agent-contract lane — 15 passed, 174 deselected; `pytest -q` — 1106 passed, 2 skipped, 32 deselected; `git diff --check`.
- Next: Phase 2 tạo `AgentTurn` admission và ownership fence theo ADR, không dùng queue baseline như bằng chứng distributed concurrency.

### 2026-07-15 02:20 - Phase 2

- Status: completed
- Evidence: `evidence/phase-02-evidence.md`
- Decisions: Sửa 2 HIGH (session-ownership race, actor fail-open) qua debugger cycle; code-reviewer phase-scoped độc lập xác nhận fix và tìm thêm 1 MEDIUM (turn_id lộ vào payload REST/SSE) — đã sửa bằng cách tra `AgentTurnTrigger` thay vì đọc payload; 2 LOW chấp nhận là residual cho Phase 5/6.
- Verification: `ruff check app tests alembic` PASS; unit agent lane 101 passed/13 deselected; full `pytest -q` — 1112 passed, 4 skipped, 32 deselected; Postgres 16 (Docker ephemeral) `alembic upgrade head` + `tests/integration/test_agent_turn_postgres.py` — 2 passed on real concurrent connections; `git diff --check`.
- Next: Phase 3 — capability resolver shadow mode và read-only enforcement, tiêu thụ turn snapshot từ Phase 2.

### 2026-07-15 - Phase 3

- Status: completed
- Evidence: `evidence/phase-03-evidence.md`
- Decisions: Resolver chỉ mở 3 capability đọc-thuần đúng phạm vi brief; code-reviewer phase-scoped tìm 3 LOW — sửa 2 (enforce substitution phải kiểm cả `policy_resolver_mode` riêng của cohort chứ không chỉ cờ global; dùng constant thay literal string cho lý do fail-closed), deferred 1 (golden-differential fixture trùng lặp, gộp vào phase turn-threading kế tiếp). Tester độc lập tìm và sửa thêm 1 gap AST purity test (bỏ sót `from pkg import sub as alias`). Tự-review dọn 2 comment code có trích dẫn số phase, viết lại không còn tham chiếu phase.
- Verification: `ruff check app/graphs/gating/ tests/unit/gates/` PASS; `pytest -q tests/unit/gates` — 145 passed, 1 deselected; `pytest -q tests/unit` — 952 passed, 2 skipped, 14 deselected.
- Next: Phase 4 — command boundary cho `write_draft` (logical command idempotent, fenced effect).

### 2026-07-15 - Phase 4

- Status: completed
- Evidence: `evidence/phase-04-evidence.md`
- Decisions: Tách `write_draft` thành logical command (`turn_id` + action + canonical intent hash + expected base version) + effect ledger fenced trong cùng transaction, đúng brief. Code-reviewer phase-scoped tìm 1 HIGH — race loser thua cùng `logical_command_id` chưa được xử lý `IntegrityError` ở production; sửa lần đầu (bọc `try/except` quanh `db.commit()` cuối hàm) sai vì autoflush từ query `select(AgentSession)` phía trước raise lỗi ngoài phạm vi `try` — phát hiện qua chính test hồi quy mới viết; sửa đúng bằng cách `flush()` tường minh và bắt lỗi ngay sau `record_effect()`, tái dùng outcome của winner khi thua race. 2 LOW: thêm log fence-stale/committed; 1 câu brief phóng đại persisted state được chấp nhận giữ nguyên (chỉ là văn bản).
- Verification: `ruff check` PASS; `pytest -q tests/unit/gates/test_write_draft_command_boundary.py` — 8 passed; 3 file legacy write_draft — 13 passed; `pytest -q tests/unit` — 960 passed, 2 skipped, 14 deselected; Postgres fault-injection tests không chạy trong môi trường này (thiếu `AGENT_TURN_POSTGRES_URL`), đã pass trước đó theo implementer/tester report.
- Next: Phase 5 — finalize, artifact-link HITL và TurnOutcome, hợp nhất approval ownership và terminal protocol.

### 2026-07-15 - Phase 5, Increment 3

- Status: in progress
- Evidence: `evidence/phase-05-increment-3-evidence.md`
- Decisions: Approval thua terminal race đóng transaction trước khi release lease; proof Postgres dùng hai approval turn khác nhau và chỉ cho phép một outcome terminal/session.
- Next: Hoàn tất transition contract typed và shadow projection còn thiếu trước khi đánh dấu Phase 5 complete.

### 2026-07-15 - Phase 5, Increment 4 (hoàn tất)

- Status: completed
- Evidence: `evidence/phase-05-increment-4-evidence.md`
- Decisions: Thêm `admit_cancel`/`admit_retry` (cùng khung fencing/idempotency/authorization của
  `admit_approval`); `admit_cancel` tự là terminal owner cho `CANCELLED` (ADR chỉ cấm approval tự
  project terminal, không cấm cancel), `admit_retry` chỉ mở lại turn trên checkpoint đã persist,
  không gọi model. Thêm `project_non_terminal_outcome` dùng chung helper ghi audit với
  `project_terminal_outcome`; route toàn bộ 4 điểm bypass còn lại (`interrupts.py`, `nodes.py`,
  `artifact_links.py`, `draft_lifecycle.py` ×2) qua projector, giữ nguyên byte-identical shape
  REST/SSE. Tester độc lập PASS (2 gap không blocker, theo dõi Phase 6); code-reviewer
  phase-scoped APPROVED, 2 LOW đã sửa (docstring trích SHA; import-cycle test allowlist). Follow-up
  của tôi phát hiện thêm 1 bug (`_record_outcome_if_enabled` truy cập `.id` trước early-return, vỡ
  với test double không có `.id`) và mở rộng routing sang 2 bypass site reviewer tìm thêm — đã sửa
  cả hai, xem `implementation-notes.md`.
- Verification: `ruff check` sạch trên toàn bộ file đụng tới (2 F401 tiền tồn tại không liên quan);
  `pytest -q tests/unit` — 997 passed, 2 skipped, 14 deselected; Postgres integration
  (`test_agent_turn_postgres.py`, `test_artifact_proposal_completion_postgres.py`,
  `test_draft_command_postgres.py`) — 9 passed; `git diff --check` chỉ còn cảnh báo CRLF.
- Residual đã ghi nhận, chuyển Phase 6: race cross-trigger-type chưa có test tường minh (an toàn
  theo code, chưa chứng minh bằng test); `_run_graph` (agent_service.py) chưa fence terminal write
  theo ownership generation, chỉ theo `session.status`; `admit_cancel` chưa bump
  `ownership_generation` của turn cũ (an toàn ở execution mode inline hiện tại).
- Next: Phase 6 — Durable Postgres turn jobs, siết chặt lease/fencing multi-worker cho các residual
  trên.

### 2026-07-15 - Phase 6, Increment 1 (hoàn tất)

- Status: completed
- Brief: `evidence/phase-06-brief-increment-1.md`
- Decisions: Thêm `AgentTurnJob`/`AgentTurnJobStatus` (migration `d38efc70b4f0`, additive, nối tiếp
  `c27dfb69a3e9`) và `AgentTurnJobService` (`enqueue`/`claim`/`renew_heartbeat`/`complete`/
  `reclaim_expired`) làm nền dữ liệu + CAS thuần cho job durable — chưa đấu nối worker loop thật/
  `_run_graph`/route REST/`docker-compose` (đúng phạm vi increment). Mọi claim/renew/complete/
  reclaim là một transaction Postgres khoá hàng thật (`FOR UPDATE`/`FOR UPDATE SKIP LOCKED`),
  không optimistic retry ở application layer.
- Code-reviewer phase-scoped round 1: BLOCK — 2 HIGH (head-of-line session ordering không được
  enforce dưới race thật do probe không khoá; self-heal trong `claim()` né được trần dead-letter
  vì chỉ `reclaim_expired()` tăng `attempt`) + 1 MEDIUM (`raise` trơ trong `enqueue()` mất exception
  gốc) + 1 LOW (chưa có backoff/`scheduled_at` khi requeue). Tôi tự sửa trực tiếp (không qua
  sub-agent, do tester round 1 chết giữa chừng vì session limit platform): thay
  `_session_head_blocked()` bằng subquery SQL `MIN(session_sequence)` theo session để candidate
  set của `claim()` chỉ còn đúng job head-of-line mỗi session; thêm tăng `attempt` + kiểm tra trần
  dead-letter ngay trong nhánh self-heal của `claim()`; sửa `except IntegrityError as exc: ...
  raise exc` giữ nguyên exception gốc. LOW được chấp nhận dời sang increment worker loop (lý do ghi
  ở `implementation-notes.md`). Thêm 2 test Postgres regression chứng minh cả hai HIGH đã đóng.
  Code-reviewer round 2 (độc lập, không nhìn lại review cũ) — APPROVED, xác nhận cả 4 finding đã
  giải quyết, không phát sinh lỗi mới từ bản sửa.
- Verification: `ruff check` sạch trên toàn bộ file đụng tới; `pytest -q tests/unit` — 1005
  passed/2 skipped/14 deselected, không regression; Postgres integration
  (`test_agent_turn_job_postgres.py`) — 4 passed (2 cũ + 2 regression HIGH mới) trên Postgres cục
  bộ đã ở head `d38efc70b4f0`; `pytest -q tests/integration` toàn bộ — 200 passed, 3 fail tiền tồn
  tại ở `test_tool_loop_scenarios.py` (xác nhận không liên quan phase này bằng cách chạy lại riêng
  file đó, không phụ thuộc `AGENT_TURN_POSTGRES_URL`, vẫn fail giống hệt).
- Next: Increment 2 — worker loop nối `_run_graph` execution, xác thực generation/fence trước mọi
  ghi command/checkpoint/outcome/outbox (Phase 6 Step 4).

### 2026-07-15 - Phase 6, Increment 2 (hoàn tất)

- Status: completed
- Brief: `evidence/phase-06-brief-increment-2.md`
- Decisions: Thêm fence check optional (`owner_id`/`expected_ownership_generation`, mặc định
  `None` — no-op giữ nguyên hành vi cũ) vào `project_terminal_outcome`/`project_non_terminal_outcome`
  qua `_check_ownership_fence` mới (`turn_outcome_projector.py`), raise `StaleTurnOwnershipError`
  trước mọi write nếu `TurnExecutionState.ownership_generation` lệch. `_run_graph` truyền cả hai
  tham số vào 5 call site `project_terminal_outcome` bên trong nó. File mới
  `app/services/agent_turn_worker.py`: `execute_claimed_job` chạy một job đã claim, heartbeat task
  độc lập renew lease, luôn cancel heartbeat + gọi `complete()` trong `finally` bất kể kết quả graph
  execution. Setting mới `agent_turn_job_heartbeat_interval_seconds=20.0`. Chưa dựng worker loop/CLI
  thật (đúng phạm vi, để Increment 4).
- Code-reviewer phase-scoped round 1: WARNING — 1 HIGH (implementer report mô tả sai fence là
  "inert trong production"; thực ra `_run_admitted_graph` đã mint owner_id/generation thật qua
  `claim_inline` nên fence đã sống ngay trên đường inline admitted, chỉ thiếu test phủ đường vui đó)
  + 1 MEDIUM (`execute_claimed_job`'s `finally` chỉ suppress `CancelledError` quanh
  `await heartbeat_task`, exception khác từ heartbeat loop có thể bỏ qua luôn `complete()`) + 2 LOW
  chấp nhận không sửa (test Postgres fence tuần tự không phải tranh chấp thật; heartbeat interval
  đúng 1/3 lease thay vì dưới 1/3 — cả hai là quyết định đã có lý do, không phải blocker). Tôi tự
  sửa trực tiếp: thêm test hồi quy
  `test_run_graph_admitted_turn_completes_under_matching_ownership_generation` (admit turn thật qua
  `AgentTurnService`, `claim_inline` lấy generation thật, xác nhận `_run_graph` hoàn tất COMPLETED
  dưới fence sống); đổi `finally` sang bắt riêng `CancelledError`/`Exception` để `complete()` luôn
  chạy tới. Code-reviewer round 2 (độc lập) — APPROVED, xác nhận cả hai fix đúng và đủ, không
  regression.
- Verification: `ruff check` sạch trên mọi file đụng tới (trừ 1 F401 tiền tồn tại không liên quan,
  xác nhận bằng `git stash`); `pytest -q tests/unit` — 1014 passed/2 skipped/14 deselected, không
  regression; Postgres integration (`test_agent_turn_job_postgres.py`) — 5 passed trên Postgres cục
  bộ đã ở head `d38efc70b4f0`.
- Next: Increment 3 — durable wake-up từ typed trigger + dead-letter + recovery scanner (Phase 6
  Step 5-6): re-validate authorization/base version/`WAIT_*`/head-of-line cho duplicate
  approval/message/cancel wake-up; nối `reclaim_expired()` chạy định kỳ thật (hiện có sẵn nhưng
  chưa được lên lịch); câu hỏi chính sách dead-letter có nên chặn thứ tự các turn sau cùng session
  hay không (mở, để lại quyết định cho increment này).

### 2026-07-15 - Phase 6, Increment 3 (hoàn tất)

- Status: completed
- Brief: `evidence/phase-06-brief-increment-3.md`
- Decisions: Xác nhận (không viết lại) rằng Step 5's "re-validate authorization/base
  version/`WAIT_*`/head-of-line cho duplicate wake-up" đã được thoả từ Phase 4/5 —
  `admit_user_message`/`admit_approval`/`admit_cancel`/`admit_retry` đều row-lock session/tool-call,
  re-check authorization + trạng thái, dedupe theo idempotency key/tool_call_id trong cùng
  transaction; `AgentTurnJobService.enqueue` (Increment 1) đã idempotent theo `turn_id`. Không có
  gap, không sửa code ở lớp admission. Phần thật của increment: thêm `_run_one_scan`/
  `_recovery_scanner_loop` mới trong `app/services/agent_turn_worker.py` (mirror `_heartbeat_loop`)
  gọi định kỳ `AgentTurnJobService.reclaim_expired()` (không sửa logic hàm này), log có cấu trúc
  (`job_id`, `turn_id`, `attempt`, `status`) mỗi lần requeue/dead-letter; setting mới
  `agent_turn_recovery_scan_interval_seconds=30.0`. Loop chưa được khởi động từ CLI/entrypoint thật
  (đúng phạm vi, để Increment 4). Quyết định chính sách dead-letter: giữ nguyên hành vi hiện tại của
  `claim()` (`non_terminal` không gồm `DEAD_LETTER`) làm chính thức — một turn dead-letter KHÔNG
  chặn head-of-line của turn kế tiếp cùng session, vì turn dead-letter đã hết attempt tự phục hồi và
  chặn cả session chờ can thiệp thủ công tạo single point of failure nặng hơn lợi ích thứ tự nghiêm
  ngặt; mỗi turn envelope vốn bất biến/độc lập theo thiết kế Phase 2. Chứng minh bằng test Postgres
  integration mới `test_postgres_dead_letter_does_not_block_next_turn_head_of_line`.
- Code-reviewer phase-scoped (độc lập) — APPROVED. Đọc sâu test dead-letter-ordering (không chỉ tin
  tên test) xác nhận đây là chứng minh thật, không phải tautology; xác nhận `claim()`/`enqueue()`/
  `renew_heartbeat()`/`complete()` không bị đụng tới bằng cách đọc toàn bộ file service. 1 finding
  LOW không chặn: test "nothing expired" của `_run_one_scan` chỉ chứng minh bảng rỗng trả `[]`, chưa
  seed một lease còn sống để chứng minh nó không bị reclaim nhầm — chấp nhận, không sửa (đã có test
  dương song song che phần requeue).
- Verification: `ruff check` sạch trên mọi file đụng tới; `pytest -q tests/unit` — 1016
  passed/2 skipped/14 deselected, không regression; Postgres integration
  (`test_agent_turn_job_postgres.py`) — 6 passed (gồm test dead-letter-ordering mới), 15 passed
  trên toàn bộ 4 file Postgres integration liên quan, không regression.
- Next: Increment 4 — worker entrypoint/deployment (CLI riêng, không phải web replica, theo quyết
  định topology của Step 1) + canary flag wiring (`agent_execution_mode=durable`, chỉ test/CI theo
  phạm vi canary đã quyết định ở Step 1) — Phase 6 Step 7, thay đổi `docker-compose`.

### 2026-07-15 - Phase 6, Increment 4 (hoàn tất — increment cuối cùng của phase)

- Status: completed
- Brief: `evidence/phase-06-brief-increment-4.md`; Implementer report:
  `evidence/phase-06-increment-4-evidence.md`
- Decisions: Thêm CLI entrypoint `app/worker_main.py` (poll `claim()` → `execute_claimed_job`,
  chạy song song `_recovery_scanner_loop`, shutdown graceful để job đang chạy hoàn tất, không tự
  chạy Alembic). Thêm `AgentService.build_run_graph_kwargs_for_turn(turn_id=...)` tái tạo kwargs
  của `_run_graph` thuần từ persisted state, chỉ hỗ trợ `USER_MESSAGE` (các trigger type khác
  raise `NotImplementedError` — chưa có nhánh dispatch nào enqueue chúng, rào chắn tường minh cho
  tương lai). Canary dispatch tại `handle_user_message`: đọc `execution_mode` từ
  `admitted.cohort` (snapshot tại thời điểm admit, không đọc `settings.agent_execution_mode` sống)
  — đúng compatibility contract; `"durable"` gọi `AgentTurnJobService.enqueue(...)` thay vì spawn
  task, `"inline"` (mặc định) giữ nguyên byte-identical. Không tạo service `docker-compose.dev.yml`
  giả cho worker — dev compose hiện không có service `web` nào để soi theo, bịa một service sẽ là
  cấu hình sai lệch không ai kiểm chứng được; ghi nhận là quyết định có chủ đích, không phải thiếu
  sót.
- Code-reviewer phase-scoped độc lập (round 1) — **BLOCK**: 1 HIGH — refactor tách logic
  `initial_state`/`resume_command` ra `_build_user_message_turn_state` (dùng chung cho đường
  inline và `build_run_graph_kwargs_for_turn`) vô tình đổi `is_first_message` từ đọc
  `decision_interrupt_type` (snapshot trước khi bị reset) sang đọc `session.interrupt_type` sống
  (đã bị nhánh non-admission gán `None` trước khi hàm được gọi) — phá vỡ resume
  `ASK_HUMAN`/`STREAM_RESPONSE` trên đường inline mặc định (`agent_turn_admission_enabled=False`):
  bỏ qua checkpoint, chạy `build_initial_workflow_state` với `resume_command=None` thay vì resume
  qua `_resume_command`. 1 MEDIUM — cùng root cause cũng ảnh hưởng
  `build_run_graph_kwargs_for_turn` (đường durable worker), nhưng "inert" vì chưa có trigger type
  nào khác `USER_MESSAGE` được dispatch durable. 1 LOW (chấp nhận) — `expected_transition_version=0`
  chưa được logic nào tiêu thụ.
- Fix trực tiếp (không dispatch lại implementer, finding nhỏ và rõ nguyên nhân): đổi
  `session.interrupt_type is None` thành `decision_interrupt_type is None` trong
  `_build_user_message_turn_state` (`app/services/agent_service.py`) — giải quyết cả HIGH lẫn
  MEDIUM cùng lúc vì cả hai đường gọi chung hàm này. Viết lại
  `test_handle_user_message_ask_human_resumes_graph` để patch `svc._run_graph` trực tiếp
  (`patch.object(svc, "_run_graph", new=AsyncMock())`) và assert trên
  `call_args.kwargs["resume_command"]`/`["initial_state"]`, thay vì kiểm tra `cr_frame.f_locals`
  của coroutine đã bị fixture `_no_background_tasks` đóng — cách cũ không dùng được. Xác nhận fix
  bắt đúng regression bằng cách revert tạm một dòng và chạy lại test — fail đúng như review mô tả;
  áp lại fix — pass. Chi tiết đầy đủ: `evidence/implementation-notes.md` (mục "fix theo
  code-reviewer BLOCK verdict").
- Re-review round 2: subagent code-reviewer độc lập bị API session limit chặn giữa chừng (không
  trả về verdict thật). Vì dispatch subagent khác lúc đó có nguy cơ gặp lại giới hạn tương tự, tôi
  tự đọc lại toàn bộ diff của increment (đọc lại `agent_service.py`, `agent_turn_service.py`,
  `worker_main.py`, `config.py`) và xác nhận: (1) fix đúng trên cả hai đường gọi
  `_build_user_message_turn_state` — đường inline dùng `decision_interrupt_type` tính tại
  `handle_user_message` trước reset, đường durable dùng `prior_interrupt_type` đọc lại từ
  `envelope.cohort` đã persist tại thời điểm admit, không đường nào còn đọc giá trị sống; (2) không
  có vị trí nào khác đọc live session state đáng lẽ phải dùng snapshot; (3) canary dispatch đọc
  đúng `admitted.cohort`, không đọc `settings.agent_execution_mode` sống; (4) `worker_main.py` chỉ
  cancel scanner task, luôn `await` poll task tới khi xong — đúng tuyên bố "không cắt ngang job
  đang chạy".
- Verification: `ruff check` sạch trên mọi file increment này đụng tới (1 lỗi F401 tiền tồn tại từ
  Phase 5, không liên quan). `pytest -q tests/unit` toàn bộ 1016 passed/2 skipped/14 deselected —
  không regression. Postgres cục bộ (`reqflow_db`, head `d38efc70b4f0`) chạy 5 file integration
  liên quan (`test_agent_turn_job_postgres.py`, `test_agent_turn_postgres.py`,
  `test_draft_command_postgres.py`, `test_artifact_proposal_completion_postgres.py`,
  `test_agent_turn_durable_canary_postgres.py`) — 16 passed, gồm canary end-to-end.
### 2026-07-15 - Phase 6 Final Review (whole-diff), fix round 2

- Status: completed
- Whole-diff final review độc lập (đúng bước Final Review của cook, không phải phase-scoped) trên
  toàn bộ diff Phase 6 (4 increment) — **BLOCK**: 1 HIGH mới, không trùng BLOCK round-1 đã fix ở
  Increment 4: `_build_user_message_turn_state`'s `is_direct_response_wait` đọc live
  `await self._latest_message_is_direct_response(session.id)` SAU KHI message USER của chính turn
  đã được persist (cả admission lẫn non-admission), nên luôn thấy message vừa insert thay vì message
  AGENT direct-response cần phát hiện — check luôn trả `False`, âm thầm bỏ qua checkpoint
  continuation cho direct-response trên ĐƯỜNG INLINE MẶC ĐỊNH, không chỉ canary/durable. Lọt qua 4
  increment vì test phủ đúng case này mang marker `@pytest.mark.golden`, bị deselect ở mọi lần chạy
  `pytest -q tests/unit` mặc định trong suốt phase. Reviewer xác nhận bằng `git stash` (pass trên
  HEAD, fail trên working tree) và `-m golden` (`1 failed, 8 passed`).
- Cũng tìm 1 MEDIUM: nhánh completion direct-response trong `_run_graph` gán
  `row.status`/`row.interrupt_type` trực tiếp, không qua ownership fence — khác 5 nhánh completion
  khác trong cùng hàm đều gọi `project_terminal_outcome(..., owner_id=..., expected_ownership_generation=...)`.
  1 LOW (chấp nhận, không sửa): import `TurnOutcomeType` không dùng trong
  `tests/unit/agent/test_agent_service.py`, đã biết từ Phase 5, ngoài phạm vi.
- Fix HIGH: thêm field `prior_latest_message_is_direct_response: bool` vào `AdmittedTurn`
  (`app/services/agent_turn_service.py`), capture bằng helper mới `_latest_message_is_direct_response`
  NGAY trước bất kỳ message insert nào (cả nhánh duplicate-replay), ghi vào `cohort` cho durable
  worker. `handle_user_message`/`build_run_graph_kwargs_for_turn` (`app/services/agent_service.py`)
  truyền field này làm tham số thay cho live query nội bộ cũ trong
  `_build_user_message_turn_state` — cùng pattern đã dùng cho `decision_status`/
  `decision_interrupt_type` ở round-1.
- Fix MEDIUM: thêm hàm public `check_ownership_fence` trong `turn_outcome_projector.py` (gọi thẳng
  `_check_ownership_fence` có sẵn, không kèm ghi `TurnOutcome` audit row vì không có
  `TurnOutcomeType` nào map đúng cặp `(WAITING_FOR_HUMAN, None)` mà nhánh này cần), gọi hàm này ngay
  trước khi gán trực tiếp trong nhánh direct-response của `_run_graph`. Chi tiết đầy đủ:
  `evidence/implementation-notes.md` (mục "Phase 6 whole-diff final review (BLOCK), fix round 2").
- Verification sau fix: `ruff check app/services/agent_service.py
  app/services/agent_turn_service.py app/graphs/analysis/turn_outcome_projector.py` sạch.
  `pytest -q -m golden tests/unit` 14 passed/1018 deselected (gồm đúng test đã lộ HIGH). `pytest -q
  tests/unit` toàn bộ 1016 passed/2 skipped/14 deselected — không regression. Postgres cục bộ
  (`reqflow_db`, head `d38efc70b4f0`) 4 file integration liên quan 15 passed +
  `test_agent_turn_durable_canary_postgres.py` 1 passed — không regression.
- Next: dispatch một round re-review độc lập cuối cùng để xác nhận cả hai fix trước khi Finalize

### 2026-07-16 - Phase 7 (hoàn tất — một unit duy nhất, không chia increment theo yêu cầu người dùng)

- Status: completed
- Evidence: `evidence/phase-07-brief.md`, `evidence/phase-07-implementer-report.md`,
  `evidence/phase-07-evidence.md`.
- Decisions: ADR 0001 addendum 2026-07-16 — checkpoint v2 dùng `AgentCheckpointHistorySaver` tự viết
  (tái dùng asyncpg/SQLAlchemy session hiện có), không dùng `AsyncPostgresSaver` chính thức của
  `langgraph-checkpoint-postgres` (driver `psycopg` riêng, không có khái niệm ownership fence của
  repo này). Migration additive `d02fa1bc91a3`: 2 cột mới trên `agent_sessions`
  (`checkpoint_version`, `event_cursor`), 2 bảng mới (`agent_checkpoints`, `agent_turn_events`).
- Implementer (dispatch `forge:implementer`) hoàn thiện phần lớn code đã có sẵn uncommitted từ một
  session trước bị ngắt: sửa 5 lỗi/gap thật (thiếu `_checkpoint_config` trên v2 saver; circular
  import; `getattr`-defensive cho fake session row; `checkpoint_version` chưa thread ở 6 call site
  thật trong `agent_service.py` — gap nghiêm trọng nhất; `_pending_interrupt_ids_v2` chưa populate).
- Code review phase-scoped (`forge:code-reviewer`) round 1: **BLOCK** — 1 HIGH (v2 resume qua
  `reject_tool_call`/in-loop feedback recovery thiếu turn context, có thể strand session âm thầm
  trong fire-and-forget task) + 2 MEDIUM (reconciliation coi non-terminal `TurnOutcome` là terminal;
  savepoint flush trong `emit_turn_event` có thể nuốt lỗi insert checkpoint) + 1 LOW (thiếu index
  `agent_checkpoints.turn_id`).
- Fix HIGH: KHÔNG mở rộng turn admission sang 2 path đó (cần trigger type mới + `ALTER TYPE`
  migration — scope của phase khác), thay vào đó thêm guard tường minh đầu `_run_graph`
  (`app/services/agent_service.py`): `checkpoint_version == "v2"` và `turn_id is None` → log lỗi rõ
  ràng và return sớm, thay vì để lỗi bị nuốt âm thầm. Residual còn lại (v2 session không resume được
  qua 2 path này) được ghi rõ trong `phase-07-evidence.md` là precondition bắt buộc phải đóng trước
  khi bật `agent_checkpoint_history_enabled=True` ở môi trường thật — hiện flag mặc định `False` nên
  không có blast radius.
- Fix MEDIUM #1: export `TERMINAL_OUTCOME_TYPES` từ `turn_outcome_projector.py`, filter
  `TurnOutcome.outcome_type.in_(TERMINAL_OUTCOME_TYPES)` trong `turn_reconciliation.py`.
- Fix MEDIUM #2: thêm `await db.flush()` sau `db.add(AgentCheckpoint(...))` và trước
  `emit_turn_event` trong `checkpointer.py:aput`.
- Fix LOW: thêm index `ix_agent_checkpoints_turn_id` vào migration `d02fa1bc91a3` (upgrade +
  downgrade, an toàn sửa trực tiếp vì chưa apply lên DB chia sẻ nào) và vào model
  `AgentCheckpoint.__table_args__` cho parity.
- Re-review sau fix: **APPROVED**.
- Verification: tự viết thêm 1 file integration test mới
  (`tests/integration/test_agent_checkpoint_history_postgres.py`) chứng minh CAS-append fenced dưới
  2 connection Postgres thật đua nhau (đúng 1 thắng, 1 nhận `StaleCheckpointAppendError`) và 1 case
  stale-ownership-generation (`StaleTurnOwnershipError`) — chạy trên Postgres 16 Docker ephemeral,
  migrate tới head `d02fa1bc91a3`. `ruff check` toàn bộ file thay đổi — sạch. `pytest -q tests/unit`
  toàn bộ — 1047 passed/2 skipped/14 deselected, không regression. 18 Postgres integration test
  (checkpoint v2 mới + 5 file cũ, `EXPECTED_ALEMBIC_REVISION` đã bump lên `d02fa1bc91a3` ở cả 5 file)
  — 18 passed, chạy lại từ database sạch sau khi thêm index. 3 test fail sẵn có trong
  `test_tool_loop_scenarios.py` xác nhận bằng `git stash` là tồn tại trên baseline trước Phase 7,
  không phải regression.
- Residual đã ghi nhận (không phải blocker để đóng phase, nhưng chặn canary thật): turn-admission
  coverage cho reject/in-loop-recovery path; retention/expiry và malformed/expired cursor cho event
  outbox (đúng như non-goal đã nêu trong brief).
- Next: Phase 8 (quality gates, observability, decommission) — canary thật của Phase 7 cần đóng
  residual turn-admission trước khi bật flag.

### 2026-07-16 - Phase 8 (partial — scope hẹp lại theo quyết định người dùng, chỉ làm phần code/CI verify được)

- Status: partial/in-progress. Phase 8 gốc có 6 bước rất rộng (CI gate, nightly live-eval workflow,
  trace/audit redaction, deploy gate topology, dashboard/alert/runbook, decommission checklist).
  Repo chưa có observability stack thật (không Grafana/Datadog/Prometheus), và decommission cần
  sign-off kiến trúc/business không thể tự quyết bằng code — hỏi người dùng qua
  `AskUserQuestion`, chọn: "CI gate + trace/audit hardening only".
- Evidence: `evidence/phase-08-brief.md`, `evidence/phase-08-implementer-report.md`,
  `evidence/phase-08-evidence.md`.
- Đã làm (verify được bằng code/test/CI):
  - Mở rộng `agent-turn-postgres` job trong `.github/workflows/ci.yml` từ chạy 1/6 lên 6/6 file
    `tests/integration/*_postgres.py` (5 file trước đó tồn tại trên đĩa nhưng không được wire vào
    CI nào — silent-skip risk đúng như Step 1 của phase gốc chỉ ra).
  - Thêm cột attribution-only `AgentRun.turn_id` (nullable FK tới `agent_turn_envelopes.id`) +
    index, qua migration mới `d2e5c8ecc7e0` (additive, chain từ `d02fa1bc91a3`). Không dùng làm
    turn identity ở đâu — reviewer xác nhận độc lập.
  - Thread `turn_id` qua `record_run_and_dispatch` (turn_audit.py) và call site trong
    `nodes.py::analyze_node`, dùng `cfg.get("turn_id")` đã có sẵn.
  - 3 test mới: correlation-join (correlation_id → AgentRun.turn_id → AgentTurnEvent/TurnOutcome/
    DraftCommandLedger), redaction regression (secret giả qua `record_run_and_dispatch`, assert
    không tồn tại verbatim trong `analysis_result` đã persist), CI-gate negative-proof (parse
    `ci.yml` thật, assert đủ 6 file Postgres được wire + `deploy.needs` là superset — đã chứng minh
    thật bằng cách break `ci.yml` 2 lần, xác nhận đỏ, rồi restore).
  - Docs: ADR addendum Phase 8, `docs/runbooks/observability-spec.md` (Status: NOT ENFORCED),
    `docs/runbooks/decommission-checklist.md` (Status: NOT STARTED) — placeholder có tên/threshold
    slot rõ ràng, không claim đã hoàn thành.
- Code review phase-scoped (`forge:code-reviewer`): **APPROVED ngay lần đầu** — 0
  CRITICAL/HIGH/MEDIUM, 2 LOW thông tin (migration không dùng CONCURRENTLY — khớp convention có
  sẵn của repo; CI-gate test dùng substring-match chứ không phân tích semantic step-graph — đã
  disclosed, vẫn bắt đúng failure mode phase yêu cầu). Không cần fix.
- Verification: `ruff check` sạch. `pytest -q` toàn bộ — 1239 passed/20 skipped/32 deselected
  (+6 đúng bằng 6 test mới, không regression; 3 fail sẵn có ở `test_tool_loop_scenarios.py` đã
  xác nhận pre-existing từ Phase 7). 18 Postgres integration test passed trên container Postgres
  16 tươi, alembic upgrade/downgrade -1 round-trip sạch, single head `d2e5c8ecc7e0`.
- Chưa làm (explicit non-goal, ghi rõ trong evidence, không phải gap bị bỏ sót):
  nightly/on-demand live-provider eval workflow; dashboard/alert wiring vào stack thật; thực thi
  decommission checklist (cần architecture owner sign-off).
- Next: chưa commit — cần finalize (dead-code check, finalizer dispatch, PUSH_PENDING). Phase 8
  vẫn "partial" trên plan.md cho tới khi phần dashboard/live-eval/decommission được làm trong một
  session riêng, có review riêng.
  Phase 6 (cập nhật checkbox `- [x] Phase 6`, `finalizer` commit + `PUSH_PENDING`).

### 2026-07-16 - Phase 9 (hoàn tất — phase housekeeping bổ sung theo yêu cầu người dùng)

- Status: completed
- Evidence: `evidence/phase-09-implementer-report.md`, `evidence/phase-09-evidence.md`.
- Scope: người dùng yêu cầu bổ sung một phase dọn dẹp cho 3 việc: (1) test migration đã xanh không
  cần thiết trong tương lai, (2) comment/docstring có chuẩn clean-code chưa, (3) slop code trong các
  phase trước. Scout (Explore agent) xác nhận cả 6 file Postgres integration test đều có giá trị hành
  vi thật (race/fence/idempotency), guard `EXPECTED_ALEMBIC_REVISION` chỉ là precondition trùng lặp
  cần consolidate chứ không phải test-migration-riêng cần xoá; không tìm thấy comment thừa nào giải
  thích lại điều code đã tự nói, chỉ có tiếng Việt + trích số phase cần sửa; chỉ một slop-code thật
  (`agent_trace_enabled` không consumer) và quyết định giữ nguyên vì ADR đã ghi nó là hook có chủ đích
  cho Phase 8 observability bị defer.
- Decisions: Consolidate guard schema-contract 6-file thành `assert_postgres_schema_contract()` dùng
  chung trong `tests/integration/conftest.py`; dịch toàn bộ comment/docstring tiếng Việt trong
  `agent_service.py`/`agent_turn_service.py`/`models/agent.py` sang tiếng Anh, giữ nguyên rationale;
  bỏ trích số phase trong `capability_manifest.py`/`workflow_snapshot.py` và docstring module 7 file
  test. Không xoá test nào, không đổi SQL/assertion, không đổi hành vi runtime.
- Code-reviewer phase-scoped (`forge:code-reviewer`) độc lập — **APPROVED**, 0 CRITICAL/HIGH/MEDIUM/
  LOW. Xác nhận cả 6 file gọi đúng helper với tham số đúng, translation giữ nguyên rationale, không
  logic nào bị đổi ẩn trong diff, `agent_trace_enabled` đúng như brief là confirm-only.
- Verification: `ruff check app tests` sạch; `pytest -q tests/unit` — 1053 passed/2 skipped/14
  deselected, không regression; 6 file Postgres integration test collect sạch (18 test, đúng số lượng
  trước refactor) và skip sạch khi thiếu `AGENT_TURN_POSTGRES_URL`; `git diff --check` sạch. Residual
  đã ghi nhận: không có Postgres thật trong môi trường này (Docker Desktop không sẵn sàng, không khởi
  động để tránh side-effect không cần thiết cho phase rủi ro thấp) — khuyến nghị chạy
  `pytest tests/integration -k postgres` trên Postgres thật trước khi merge lên nhánh chia sẻ.
- Next: Finalize (dead-code check, finalizer dispatch, PUSH_PENDING) cho Phase 9; Phase 8 vẫn giữ
  trạng thái "partial" độc lập, không liên quan tới phase này.

### 2026-07-16 - Advisor root-cause findings fix (attached to Phase 6-9 work-items)

- Status: completed
- Fix report: `fixes/fix-260716-1401-advisor-root-cause-findings.md`
- Highlights: exponential backoff cho `reclaim_expired()` để job vừa reclaim không bị claim ngay lập tức (backoff 2s base, nhân đôi theo attempt); guard test cho turn-state snapshot pattern (`decision_*` không re-derive từ live session row); consolidate test fixture `_menu_gating_states.py` giữa 2 file gate test; bỏ trích dẫn phase-03-brief.md khỏi docstring (tuân thủ quy tắc không đưa tham chiếu phase vào code).
- Verification: ruff clean, `pytest -q tests/unit` 1055 passed/2 skipped/14 deselected, `pytest -q -o addopts="" -m "golden and not live and not evidence"` 15 passed (3 pre-existing fail xác nhận không liên quan bằng `git stash`).
- No phase checkbox update — fix này là bảo trì dựa trên advisor phân tích implementation-notes.md, không phải một phase mới hoặc increment phase hiện tại.
