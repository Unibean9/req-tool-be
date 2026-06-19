"""E2E happy-case Phase 3: intent artifact → agent Q&A → BRD artifact → brd.md

Flow:
  1. Setup: đăng ký user, tạo org/project
  2. Seed: tạo intent artifact (prerequisite)
  3. Setup LLM provider
  4. Tạo agent session (artifact_type=goal)
  5. Gửi message khởi động với context đầy đủ
  6. Vòng lặp Q&A:
       - ask_human (câu hỏi thông thường) → trả lời với business context
       - ask_human (confirm "Bạn có muốn tạo?") → trả lời "Có, tạo ngay"
       - propose_artifacts → approve tất cả tool calls
  7. Poll đến khi completed
  8. Lấy artifacts vừa tạo → ghi ra brd.md

Biến môi trường (đọc từ .env.test):
    E2E_BASE_URL            URL server (mặc định http://127.0.0.1:8000)
    E2E_TIMEOUT             Timeout poll agent giây (mặc định 300)
    E2E_REQUEST_TIMEOUT     Timeout mỗi HTTP request giây (mặc định 60)
    E2E_LLM_PROVIDER_TYPE   openai | bedrock | ... (mặc định openai)
    E2E_LLM_API_KEY         API key (bắt buộc)
    E2E_LLM_SECRET_KEY      AWS secret key (chỉ Bedrock)
    E2E_LLM_REGION          AWS region (chỉ Bedrock)
    E2E_LLM_MODEL_NAME      Tên model (tuỳ chọn)
    E2E_BRD_OUTPUT          Đường dẫn file output (mặc định brd.md)

Ví dụ:
    python scripts/e2e_phase3_intent_to_brd.py
    python scripts/e2e_phase3_intent_to_brd.py --output docs/output/brd.md
    python scripts/e2e_phase3_intent_to_brd.py --health-check --verbose
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env.test")

API_PREFIX = "/api/v1"
_PASSWORD = "Secret123!"

# Từ khoá trong tin nhắn confirm_node để phân biệt với câu hỏi thông thường
_CONFIRM_KEYWORDS = ("bạn có muốn", "có muốn tôi tạo", "đủ thông tin để tạo")

# Context phong phú để agent có đủ thông tin tạo BRD ngay lần đầu
_INITIAL_MESSAGE = """\
Tôi muốn xây dựng BRD cho hệ thống quản lý yêu cầu phần mềm.

**Bối cảnh tổ chức:**
Doanh nghiệp phần mềm vừa, 80 nhân viên. BA team 3 người đang quản lý yêu cầu \
thủ công qua email và Google Docs. Developer team 10 người, sprint 2 tuần.

**Vấn đề cốt lõi:**
- Yêu cầu phân tán: email, Docs, Jira không đồng bộ
- Không truy vết được ai phê duyệt requirement nào, khi nào, tại sao
- Stakeholder mâu thuẫn làm sprint bị block trung bình 2 lần/tháng
- BA mới onboard mất 3 tuần mới nắm được context

**Mục tiêu đo lường (OKR):**
- Giảm thời gian phê duyệt requirement từ 5 ngày → 2 ngày (Q3 2026)
- 100% requirement có traceability link tới stakeholder interview
- Giảm sprint block do requirement mơ hồ từ 2 lần → 0.5 lần/tháng
- Onboarding BA mới dưới 5 ngày làm việc

**Scope MVP (Q3 2026):**
- Tạo/chỉnh sửa/phiên bản hoá requirement
- Phê duyệt requirement theo workflow: Draft → Review → Approved
- Traceability: requirement ↔ stakeholder interview ↔ artifact downstream
- Tích hợp: export sang Jira, import từ Confluence

**Out of scope:** Mobile app, AI auto-suggest, real-time collaboration

**Stakeholder:**
- Product Manager (owner)
- Business Analyst (primary user)
- Developer Lead (consumer)
- CTO (sponsor, approve budget)

**Timeline & Budget:** Q3 2026, 200M VND, team 3 BA + 2 dev backend
"""

# Trả lời mặc định khi agent hỏi thêm thông tin
_FOLLOWUP_REPLIES = [
    "Hệ thống cần hỗ trợ tối đa 50 concurrent users. SLA: uptime 99.5%, response time < 2s. "
    "Tech stack hiện tại: PostgreSQL, Python/FastAPI, React. Deployment trên AWS.",

    "Quy trình phê duyệt: BA tạo draft → PM review trong 24h → nếu approve thì chuyển Approved, "
    "nếu có comment thì quay về BA trong 4h làm việc. Tối đa 3 vòng review trước khi leo thang lên CTO.",

    "Tích hợp Jira: đồng bộ 1 chiều, mỗi Approved requirement tự tạo Epic trong Jira. "
    "Confluence: import existing docs dưới dạng requirement draft để BA review và điều chỉnh.",

    "Không có yêu cầu đặc biệt về bảo mật ngoài: single sign-on (SSO) với Google Workspace, "
    "audit log toàn bộ thao tác, role-based access (BA, PM, Developer, Admin).",
]


class E2EFailure(AssertionError):
    pass


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8")


def log(msg: str) -> None:
    print(msg, flush=True)


def log_verbose(msg: str, *, verbose: bool) -> None:
    if verbose:
        print(msg, flush=True)


@dataclass
class RunContext:
    client: httpx.AsyncClient
    headers: dict[str, str]
    org_id: str
    project_id: str
    verbose: bool
    provider_config_id: str | None = None


def _data(response: httpx.Response) -> Any:
    payload = response.json()
    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]
    return payload


def _expect_status(response: httpx.Response, expected: int | set[int], label: str) -> None:
    expected_set = expected if isinstance(expected, set) else {expected}
    if response.status_code not in expected_set:
        raise E2EFailure(
            f"{label}: status={response.status_code}, expected={sorted(expected_set)}, "
            f"body={response.text[:2000]}"
        )


async def _req(
    client: httpx.AsyncClient,
    method: str,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    json: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    expected: int | set[int] = 200,
) -> httpx.Response:
    resp = await client.request(method, f"{API_PREFIX}{path}", headers=headers, json=json, params=params)
    _expect_status(resp, expected, f"{method} {path}")
    return resp


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

async def bootstrap_context(client: httpx.AsyncClient, *, verbose: bool) -> RunContext:
    suffix = uuid.uuid4().hex[:8]
    email = f"e2e-phase3-{suffix}@example.com"
    log(f"[setup] Tạo user {email}")

    await _req(client, "POST", "/auth/register",
               json={"email": email, "password": _PASSWORD, "full_name": "E2E Phase3"},
               expected=201)
    token = _data(await _req(client, "POST", "/auth/login",
                              json={"email": email, "password": _PASSWORD}))["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    org = _data(await _req(client, "POST", "/orgs", headers=headers,
                            json={"name": f"Phase3 Org {suffix}"}, expected=201))
    project = _data(await _req(client, "POST", f"/orgs/{org['id']}/projects", headers=headers,
                                json={"name": f"Phase3 Project {suffix}",
                                      "description": "E2E phase3 intent → BRD"},
                                expected=201))
    log(f"[setup] project_id={project['id']}")
    return RunContext(client=client, headers=headers, org_id=org["id"],
                     project_id=project["id"], verbose=verbose)


async def seed_intent(ctx: RunContext) -> str:
    """Tạo intent artifact — prerequisite để agent biết context phase 3."""
    log("[seed] Tạo intent artifact")
    resp = await _req(ctx.client, "POST", f"/projects/{ctx.project_id}/artifacts",
                      headers=ctx.headers,
                      json={
                          "type": "intent",
                          "title": "Số hoá và tập trung hoá quy trình quản lý yêu cầu phần mềm",
                          "body": (
                              "Tổ chức cần một hệ thống duy nhất để ghi nhận, phiên bản hoá và "
                              "truy vết toàn bộ yêu cầu từ stakeholder tới delivery. "
                              "Mục tiêu: giảm thất thoát thông tin, tăng minh bạch quyết định, "
                              "rút ngắn vòng lặp phê duyệt."
                          ),
                          "metadata": {"source": "e2e-phase3"},
                      },
                      expected=201)
    artifact = _data(resp)
    log(f"  intent: {artifact['id']}")
    return artifact["id"]


async def setup_provider(ctx: RunContext, *, health_check: bool) -> str:
    api_key = os.getenv("E2E_LLM_API_KEY")
    if not api_key:
        raise E2EFailure("E2E_LLM_API_KEY chưa được set — script này cần LLM thật")

    provider_type = os.getenv("E2E_LLM_PROVIDER_TYPE", "openai")
    payload: dict[str, Any] = {
        "provider_type": provider_type,
        "api_key": api_key,
        "model_name": os.getenv("E2E_LLM_MODEL_NAME") or None,
        "region": os.getenv("E2E_LLM_REGION") or None,
    }
    secret_key = os.getenv("E2E_LLM_SECRET_KEY")
    if secret_key:
        payload["secret_key"] = secret_key
    payload = {k: v for k, v in payload.items() if v is not None}

    resp = await _req(ctx.client, "POST", "/users/me/llm-provider-configs",
                      headers=ctx.headers, json=payload, expected=201)
    config = _data(resp)

    if api_key in resp.text or (secret_key and secret_key in resp.text):
        raise E2EFailure("Provider response làm lộ secret key")

    log(f"[provider] {config['id'][:8]}… type={provider_type} model={config.get('model_name')}")

    if health_check:
        hc = _data(await _req(ctx.client, "POST",
                               f"/users/me/llm-provider-configs/{config['id']}/health-check",
                               headers=ctx.headers))
        log(f"  health-check ok: {hc.get('response_time_ms')}ms")

    return config["id"]


# ---------------------------------------------------------------------------
# Agent session helpers
# ---------------------------------------------------------------------------

async def start_session(ctx: RunContext) -> str:
    payload: dict[str, Any] = {
        "artifact_type": "goal",
        "workflow_area": "analysis",
        "agent_role": "business_analyst",
        "provider_config_id": ctx.provider_config_id,
    }
    resp = await _req(ctx.client, "POST", f"/projects/{ctx.project_id}/agent-sessions",
                      headers=ctx.headers, json=payload, expected=201)
    session = _data(resp)
    log(f"[session] id={session['session_id']} missing={session['missing_context']}")
    return session["session_id"]


async def poll_session(ctx: RunContext, session_id: str, *, timeout_s: float) -> dict[str, Any]:
    """Poll đến khi thoát khỏi active."""
    deadline = time.monotonic() + timeout_s
    last: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        resp = await _req(ctx.client, "GET",
                          f"/projects/{ctx.project_id}/agent-sessions/{session_id}",
                          headers=ctx.headers)
        last = _data(resp)
        log_verbose(f"  poll: status={last['status']} interrupt={last.get('interrupt_type')}", verbose=ctx.verbose)
        if last["status"] != "active":
            return last
        await asyncio.sleep(1.5)
    raise E2EFailure(f"Session kẹt sau {timeout_s}s: {last}")


async def send_message(ctx: RunContext, session_id: str, content: str) -> None:
    await _req(ctx.client, "POST",
               f"/projects/{ctx.project_id}/agent-sessions/{session_id}/messages",
               headers=ctx.headers, json={"content": content})


async def get_messages(ctx: RunContext, session_id: str) -> list[dict[str, Any]]:
    return _data(await _req(ctx.client, "GET",
                             f"/projects/{ctx.project_id}/agent-sessions/{session_id}/messages",
                             headers=ctx.headers))


async def get_tool_calls(ctx: RunContext, session_id: str) -> list[dict[str, Any]]:
    return _data(await _req(ctx.client, "GET",
                             f"/projects/{ctx.project_id}/agent-sessions/{session_id}/tool-calls",
                             headers=ctx.headers))


async def approve_tool_calls(ctx: RunContext, session_id: str) -> int:
    tool_calls = await get_tool_calls(ctx, session_id)
    proposed = [tc for tc in tool_calls if tc["status"] == "proposed"]
    if not proposed:
        raise E2EFailure("propose_artifacts interrupt nhưng không có tool call nào ở trạng thái proposed")
    log(f"  → approve {len(proposed)} tool call(s)")
    for tc in proposed:
        await _req(ctx.client, "POST",
                   f"/projects/{ctx.project_id}/agent-tool-calls/{tc['id']}/approve",
                   headers=ctx.headers)
    return len(proposed)


def _is_confirm_message(content: str) -> bool:
    """Phân biệt confirm_node ("Bạn có muốn tôi tạo?") với câu hỏi thông thường."""
    lower = content.lower()
    return any(kw in lower for kw in _CONFIRM_KEYWORDS)


# ---------------------------------------------------------------------------
# Main happy-case loop
# ---------------------------------------------------------------------------

async def run_happy_case(ctx: RunContext, session_id: str, *, timeout_s: float) -> None:
    """
    Điều hướng agent qua toàn bộ vòng lặp đến khi completed.

    Trạng thái có thể gặp:
      ask_human + câu hỏi thông thường → trả lời với context phong phú
      ask_human + confirm ("Bạn có muốn tạo?") → "Có, tạo ngay"
      propose_artifacts → approve tất cả
      completed / failed → kết thúc
    """
    followup_index = 0
    MAX_ROUNDS = 20

    for round_num in range(MAX_ROUNDS):
        session = await poll_session(ctx, session_id, timeout_s=timeout_s)
        status = session["status"]
        interrupt = session.get("interrupt_type")

        log(f"[round {round_num + 1}] status={status} interrupt={interrupt}")

        if status == "completed":
            log("  → session completed")
            return

        if status == "failed":
            messages = await get_messages(ctx, session_id)
            agent_msgs = [m["content"] for m in messages if m["role"] == "agent"]
            raise E2EFailure(f"Session failed: {agent_msgs[-1][:300] if agent_msgs else '(no message)'}")

        if status == "waiting_for_human" and interrupt == "ask_human":
            messages = await get_messages(ctx, session_id)
            agent_msgs = [m for m in messages if m["role"] == "agent"]
            if not agent_msgs:
                raise E2EFailure("interrupt=ask_human nhưng không có agent message")

            last_msg = agent_msgs[-1]["content"]
            log(f"  agent: {last_msg[:160]}{'…' if len(last_msg) > 160 else ''}")

            if _is_confirm_message(last_msg):
                log("  → confirm detected, trả lời: Có, tạo ngay")
                await send_message(ctx, session_id, "Có, tạo ngay.")
            else:
                reply = _FOLLOWUP_REPLIES[followup_index % len(_FOLLOWUP_REPLIES)]
                followup_index += 1
                log(f"  → trả lời followup #{followup_index}: {reply[:80]}…")
                await send_message(ctx, session_id, reply)
            continue

        if status == "waiting_for_human" and interrupt == "propose_artifacts":
            count = await approve_tool_calls(ctx, session_id)
            log(f"  → approved {count} proposal(s), poll tiếp")
            continue

        raise E2EFailure(f"Trạng thái không mong đợi: status={status} interrupt={interrupt}")

    raise E2EFailure(f"Vòng lặp quá {MAX_ROUNDS} lần, session chưa kết thúc")


# ---------------------------------------------------------------------------
# Collect & export
# ---------------------------------------------------------------------------

async def fetch_created_artifacts(ctx: RunContext, session_id: str) -> list[dict[str, Any]]:
    """Lấy artifacts được tạo ra bởi session này qua tool calls."""
    tool_calls = await get_tool_calls(ctx, session_id)
    artifact_ids = [
        tc["created_artifact_id"]
        for tc in tool_calls
        if tc.get("created_artifact_id") and tc["status"] in {"approved", "executed"}
    ]

    # Artifact API exposes a list endpoint; there is no GET by artifact id.
    all_arts = _data(await _req(ctx.client, "GET", f"/projects/{ctx.project_id}/artifacts",
                                headers=ctx.headers))
    if not artifact_ids:
        log("  (không tìm thấy artifact qua tool calls, fallback: list artifacts)")
        return [a for a in all_arts if a.get("type") == "goal"]

    by_id = {art["id"]: art for art in all_arts}
    return [by_id[aid] for aid in artifact_ids if aid in by_id]


def export_brd_md(artifacts: list[dict[str, Any]], *, output_path: Path) -> None:
    """Ghi artifacts ra file brd.md."""
    lines: list[str] = ["# BRD — Generated by AI Agent\n"]

    for i, art in enumerate(artifacts, 1):
        lines.append(f"## Artifact {i}: {art.get('title', '(no title)')}\n")
        lines.append(f"> type: `{art.get('type')}` | status: `{art.get('status')}`\n")
        lines.append("")
        body = _artifact_body(art) or "(empty body)"
        lines.append(body)
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    log(f"\n[output] Đã ghi {len(artifacts)} artifact(s) → {output_path}")


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

def assert_brd_quality(artifacts: list[dict[str, Any]]) -> None:
    """Kiểm tra artifact đầu ra đủ chất lượng tối thiểu."""
    if not artifacts:
        raise E2EFailure("Không có artifact nào được tạo ra — happy case phải tạo ít nhất 1")

    for art in artifacts:
        title = art.get("title", "")
        body = _artifact_body(art)

        if not title.strip():
            raise E2EFailure(f"Artifact {art.get('id')} không có title")
        if len(body.strip()) < 50:
            raise E2EFailure(
                f"Artifact '{title}' body quá ngắn ({len(body)} ký tự) — "
                "agent chưa tạo nội dung đủ"
            )

    log(f"  assertions ok: {len(artifacts)} artifact(s), title + body đủ")


def _artifact_body(artifact: dict[str, Any]) -> str:
    """Artifact API exposes content through current_version.body."""
    current_version = artifact.get("current_version") or {}
    return artifact.get("body") or current_version.get("body") or ""


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

async def run(args: argparse.Namespace) -> None:
    timeout = httpx.Timeout(args.request_timeout)
    async with httpx.AsyncClient(base_url=args.base_url.rstrip("/"), timeout=timeout) as client:
        health = await client.get("/health")
        _expect_status(health, 200, "GET /health")
        log(f"[health] {health.json()}\n")

        # 1. Setup
        ctx = await bootstrap_context(client, verbose=args.verbose)

        # 2. Seed intent
        await seed_intent(ctx)

        # 3. LLM provider
        ctx.provider_config_id = await setup_provider(ctx, health_check=args.health_check)

        # 4. Tạo session
        log("\n[phase3] Bắt đầu session agent: intent → goal (BRD)")
        session_id = await start_session(ctx)

        # 5. Kick off với context đầy đủ
        log(f"[kick-off] Gửi message ban đầu ({len(_INITIAL_MESSAGE)} chars)")
        await send_message(ctx, session_id, _INITIAL_MESSAGE)

        # 6. Chạy vòng lặp Q&A → confirm → propose → approve
        await run_happy_case(ctx, session_id, timeout_s=args.timeout)

        # 7. Fetch artifacts
        log("\n[collect] Lấy artifacts vừa được tạo")
        artifacts = await fetch_created_artifacts(ctx, session_id)
        log(f"  tìm thấy {len(artifacts)} artifact(s)")
        for art in artifacts:
            log(f"    [{art.get('type')}] {art.get('title')} ({len(_artifact_body(art))} chars)")

        # 8. Assert chất lượng
        assert_brd_quality(artifacts)

        # 9. Export ra brd.md
        export_brd_md(artifacts, output_path=Path(args.output))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E2E Phase 3 happy case: intent → BRD artifact → brd.md")
    parser.add_argument("--base-url", default=os.getenv("E2E_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--timeout", type=float, default=float(os.getenv("E2E_TIMEOUT", "300")),
                        help="Timeout poll agent giây (E2E_TIMEOUT, mặc định 300)")
    parser.add_argument("--request-timeout", type=float,
                        default=float(os.getenv("E2E_REQUEST_TIMEOUT", "60")))
    parser.add_argument("--output", default=os.getenv("E2E_BRD_OUTPUT", "brd.md"),
                        help="Đường dẫn file output (E2E_BRD_OUTPUT, mặc định brd.md)")
    parser.add_argument("--health-check", action="store_true",
                        help="Gọi health-check provider trước khi chạy")
    parser.add_argument("--verbose", action="store_true",
                        help="In log poll chi tiết")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    configure_stdio()
    args = parse_args(argv or sys.argv[1:])
    try:
        asyncio.run(run(args))
    except (httpx.HTTPError, E2EFailure) as exc:
        print(f"\nE2E FAILED: {exc}", file=sys.stderr)
        return 1
    print("\nE2E PASSED — brd.md đã được tạo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
