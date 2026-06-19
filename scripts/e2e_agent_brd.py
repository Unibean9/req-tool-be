"""E2E smoke test cho AI Agent - BRD flow.

Load biến từ .env với prefix E2E_:
    E2E_BASE_URL            URL server (mặc định http://127.0.0.1:8000)
    E2E_TIMEOUT             Timeout poll agent (giây, mặc định 60)
    E2E_LLM_PROVIDER_TYPE   openai | bedrock | ...
    E2E_LLM_API_KEY         API key / AWS access key
    E2E_LLM_SECRET_KEY      AWS secret key (chỉ Bedrock)
    E2E_LLM_REGION          AWS region (chỉ Bedrock)
    E2E_LLM_MODEL_NAME      Tên model (tuỳ chọn)
    E2E_ARTIFACT_TYPE       Artifact type dùng cho BRD session (mặc định brd)
    E2E_TOOL_ACTION         approve | reject (mặc định approve)

Ví dụ:
    python scripts/e2e_agent_brd.py
    python scripts/e2e_agent_brd.py --scenario ask-human
    python scripts/e2e_agent_brd.py --scenario all --health-check
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

# Đọc .env.test từ thư mục gốc project
load_dotenv(Path(__file__).resolve().parent.parent / ".env.test")

API_PREFIX = "/api/v1"
_PASSWORD = "Secret123!"
TERMINAL_STATUSES = {"completed", "failed"}
HUMAN_STATUSES = {"waiting_for_human"}
NON_ACTIVE_STATUSES = TERMINAL_STATUSES | HUMAN_STATUSES


class E2EFailure(AssertionError):
    pass


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8")


@dataclass
class RunContext:
    client: httpx.AsyncClient
    token: str
    headers: dict[str, str]
    org_id: str
    project_id: str
    provider_config_id: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(msg, flush=True)


def _data(response: httpx.Response) -> Any:
    payload = response.json()
    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]
    return payload


def _expect_status(response: httpx.Response, expected: int | set[int], label: str) -> None:
    expected_set = expected if isinstance(expected, set) else {expected}
    if response.status_code not in expected_set:
        raise E2EFailure(
            f"{label}: status={response.status_code}, expected={sorted(expected_set)}, body={response.text[:500]}"
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
    response = await client.request(method, f"{API_PREFIX}{path}", headers=headers, json=json, params=params)
    _expect_status(response, expected, f"{method} {path}")
    return response


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

async def bootstrap_context(client: httpx.AsyncClient, *, label: str) -> RunContext:
    suffix = uuid.uuid4().hex[:8]
    email = f"e2e-brd-{label}-{suffix}@example.com"

    user_resp = await _req(
        client, "POST", "/auth/register",
        json={"email": email, "password": _PASSWORD, "full_name": f"E2E {label}"},
        expected=201,
    )
    log(f"  user: {_data(user_resp)['email']}")

    token = _data(await _req(client, "POST", "/auth/login", json={"email": email, "password": _PASSWORD}))["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    org = _data(await _req(client, "POST", "/orgs", headers=headers,
                           json={"name": f"E2E BRD Org {suffix}"}, expected=201))
    project = _data(await _req(client, "POST", f"/orgs/{org['id']}/projects", headers=headers,
                               json={"name": f"E2E BRD Project {suffix}", "description": "BRD e2e test"},
                               expected=201))
    log(f"  project: {project['id']}")
    return RunContext(client=client, token=token, headers=headers, org_id=org["id"], project_id=project["id"])


async def _create_artifact(ctx: RunContext, *, artifact_type: str, title: str, body: str,
                           priority: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"type": artifact_type, "title": title, "body": body, "metadata": {"e2e": True}}
    if priority:
        payload["priority"] = priority
    resp = await _req(ctx.client, "POST", f"/projects/{ctx.project_id}/artifacts",
                      headers=ctx.headers, json=payload, expected=201)
    art = _data(resp)
    log(f"  artifact [{artifact_type}]: {art['id']}")
    return art


async def seed_brd_context(ctx: RunContext) -> None:
    """Tạo intent + problem + goal đủ để agent BRD có context phân tích."""
    log("[seed] Tạo BRD context (intent, problem, goal)")

    intent = await _create_artifact(ctx, artifact_type="intent", priority="must",
                                    title="Số hoá quy trình yêu cầu phần mềm",
                                    body="Tổ chức cần một hệ thống tập trung để ghi nhận, phiên bản hoá "
                                         "và truy vết các yêu cầu từ stakeholder tới delivery.")
    problem = await _create_artifact(ctx, artifact_type="problem", priority="must",
                                     title="Yêu cầu bị phân tán, không truy vết được",
                                     body="BA và PM đang dùng email, Google Docs và Jira riêng lẻ. "
                                          "Khi requirement thay đổi, không rõ ai quyết định và tại sao.")
    goal = await _create_artifact(ctx, artifact_type="goal", priority="should",
                                  title="Traceability đầu cuối từ stakeholder tới artifact",
                                  body="Mỗi requirement phải liên kết tới nguồn phỏng vấn, "
                                       "phiên bản phê duyệt và artifact downstream.")

    await _req(ctx.client, "POST", f"/projects/{ctx.project_id}/artifact-links",
               headers=ctx.headers,
               json={"source_artifact_id": intent["id"], "target_artifact_id": problem["id"],
                     "relation_type": "informs", "metadata": {}},
               expected=201)
    await _req(ctx.client, "POST", f"/projects/{ctx.project_id}/artifact-links",
               headers=ctx.headers,
               json={"source_artifact_id": problem["id"], "target_artifact_id": goal["id"],
                     "relation_type": "supports", "metadata": {}},
               expected=201)
    log(f"  seed xong: intent={intent['id'][:8]}… problem={problem['id'][:8]}… goal={goal['id'][:8]}…")


async def setup_provider(ctx: RunContext, *, health_check: bool) -> str | None:
    api_key = os.getenv("E2E_LLM_API_KEY")
    if not api_key:
        return None

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
    if not config["api_key_set"]:
        raise E2EFailure("api_key_set=False sau khi tạo provider config")

    log(f"[provider] {config['id'][:8]}… type={provider_type} model={config['model_name']}")

    if health_check:
        hc = _data(await _req(ctx.client, "POST",
                               f"/users/me/llm-provider-configs/{config['id']}/health-check",
                               headers=ctx.headers))
        if "response_time_ms" not in hc:
            raise E2EFailure("Health-check thiếu response_time_ms")
        log(f"  health-check: {hc['response_time_ms']}ms")

    return config["id"]


# ---------------------------------------------------------------------------
# Agent session helpers
# ---------------------------------------------------------------------------

async def start_brd_session(ctx: RunContext, artifact_type: str) -> str:
    payload: dict[str, Any] = {
        "artifact_type": artifact_type,
        "workflow_area": "analysis",
        "agent_role": "business_analyst",
    }
    if ctx.provider_config_id:
        payload["provider_config_id"] = ctx.provider_config_id

    resp = await _req(ctx.client, "POST", f"/projects/{ctx.project_id}/agent-sessions",
                      headers=ctx.headers, json=payload, expected=201)
    session = _data(resp)
    log(f"  session: {session['session_id']} missing_context={session['missing_context']}")
    return session["session_id"]


async def poll_session(ctx: RunContext, session_id: str, *, timeout_s: float,
                       desired: set[str] = NON_ACTIVE_STATUSES) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        resp = await _req(ctx.client, "GET",
                          f"/projects/{ctx.project_id}/agent-sessions/{session_id}",
                          headers=ctx.headers)
        last = _data(resp)
        log(f"    poll: status={last['status']} interrupt={last.get('interrupt_type')}")
        if last["status"] in desired:
            return last
        await asyncio.sleep(1.0)
    raise E2EFailure(f"Session kẹt sau {timeout_s}s: {last}")


async def get_agent_messages(ctx: RunContext, session_id: str) -> list[dict[str, Any]]:
    resp = await _req(ctx.client, "GET",
                      f"/projects/{ctx.project_id}/agent-sessions/{session_id}/messages",
                      headers=ctx.headers)
    return _data(resp)


async def get_tool_calls(ctx: RunContext, session_id: str) -> list[dict[str, Any]]:
    resp = await _req(ctx.client, "GET",
                      f"/projects/{ctx.project_id}/agent-sessions/{session_id}/tool-calls",
                      headers=ctx.headers)
    return _data(resp)


async def send_user_message(ctx: RunContext, session_id: str, content: str) -> dict[str, Any]:
    resp = await _req(ctx.client, "POST",
                      f"/projects/{ctx.project_id}/agent-sessions/{session_id}/messages",
                      headers=ctx.headers, json={"content": content})
    return _data(resp)


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

async def scenario_no_provider(client: httpx.AsyncClient, args: argparse.Namespace) -> None:
    """Agent không có LLM provider phải fail rõ ràng, không kẹt active."""
    log("\n[scenario: no-provider] Agent không có provider → phải failed")
    ctx = await bootstrap_context(client, label="no-provider")
    session_id = await start_brd_session(ctx, artifact_type=args.artifact_type)
    session = await poll_session(ctx, session_id, timeout_s=args.timeout, desired=TERMINAL_STATUSES)

    if session["status"] != "failed":
        raise E2EFailure(f"Kỳ vọng failed, nhận {session['status']}")

    messages = await get_agent_messages(ctx, session_id)
    agent_messages = [m for m in messages if m["role"] == "agent"]
    if not agent_messages:
        raise E2EFailure("Session failed nhưng không có agent message nào giải thích lý do")

    log(f"  ok: failed + {len(agent_messages)} agent message(s)")
    log(f"  message: {agent_messages[0]['content'][:120]}")


async def scenario_ask_human(client: httpx.AsyncClient, args: argparse.Namespace) -> None:
    """Agent BRD hỏi user → user trả lời → agent tiếp tục đến trạng thái cuối."""
    log("\n[scenario: ask-human] BRD ask_human loop")
    _require_provider_key()

    ctx = await bootstrap_context(client, label="ask-human")
    await seed_brd_context(ctx)
    ctx.provider_config_id = await setup_provider(ctx, health_check=args.health_check)
    if not ctx.provider_config_id:
        raise E2EFailure("Scenario ask-human cần E2E_LLM_API_KEY")

    session_id = await start_brd_session(ctx, artifact_type=args.artifact_type)
    session = await poll_session(ctx, session_id, timeout_s=args.timeout)

    # Nếu agent tự hoàn thành mà không hỏi, cũng OK
    if session["status"] == "completed":
        log("  ok: agent completed mà không cần hỏi")
        return

    if session["status"] == "failed":
        messages = await get_agent_messages(ctx, session_id)
        raise E2EFailure(f"Agent failed: {messages}")

    if session["interrupt_type"] != "ask_human":
        raise E2EFailure(f"Kỳ vọng ask_human interrupt, nhận {session['interrupt_type']}")

    # Kiểm tra agent message có nội dung (bug đã fix)
    messages = await get_agent_messages(ctx, session_id)
    agent_msgs = [m for m in messages if m["role"] == "agent"]
    if not agent_msgs:
        raise E2EFailure("ask_human nhưng không có agent message")
    if not agent_msgs[-1]["content"].strip():
        raise E2EFailure("ask_human agent message rỗng — bug chưa được fix")
    log(f"  agent hỏi: {agent_msgs[-1]['content'][:120]}")

    # User trả lời
    await send_user_message(
        ctx, session_id,
        "Ưu tiên tạo BRD với scope hẹp: chỉ module quản lý yêu cầu. "
        "Mục tiêu đo lường: giảm 50% thời gian phê duyệt requirement."
    )
    log("  user reply gửi xong, tiếp tục poll…")

    resumed = await poll_session(ctx, session_id, timeout_s=args.timeout)
    if resumed["status"] == "active":
        raise E2EFailure("Session vẫn active sau khi gửi user reply")

    log(f"  ok: ask-human loop hoàn tất → {resumed['status']} / {resumed.get('interrupt_type')}")


async def scenario_propose(client: httpx.AsyncClient, args: argparse.Namespace) -> None:
    """Agent BRD propose artifacts → user approve/reject → agent tiếp tục."""
    log(f"\n[scenario: propose] BRD propose_artifacts loop (action={args.tool_action})")
    _require_provider_key()

    ctx = await bootstrap_context(client, label="propose")
    await seed_brd_context(ctx)
    ctx.provider_config_id = await setup_provider(ctx, health_check=args.health_check)
    if not ctx.provider_config_id:
        raise E2EFailure("Scenario propose cần E2E_LLM_API_KEY")

    session_id = await start_brd_session(ctx, artifact_type=args.artifact_type)
    session = await poll_session(ctx, session_id, timeout_s=args.timeout)

    if session["status"] == "completed":
        log("  ok: agent completed mà không propose (context đủ)")
        return

    if session["status"] == "failed":
        messages = await get_agent_messages(ctx, session_id)
        raise E2EFailure(f"Agent failed: {messages}")

    if session["interrupt_type"] != "propose_artifacts":
        # Có thể agent hỏi trước — nếu ask_human thì reply và poll lại
        if session["interrupt_type"] == "ask_human":
            messages = await get_agent_messages(ctx, session_id)
            agent_msgs = [m for m in messages if m["role"] == "agent"]
            if agent_msgs and not agent_msgs[-1]["content"].strip():
                raise E2EFailure("ask_human agent message rỗng — bug chưa được fix")
            await send_user_message(ctx, session_id, "Hãy tạo BRD draft ngay, bao gồm scope và mục tiêu đo được.")
            session = await poll_session(ctx, session_id, timeout_s=args.timeout)
            if session["interrupt_type"] != "propose_artifacts":
                log(f"  ok: agent dừng ở {session['status']} / {session.get('interrupt_type')} sau ask-human reply")
                return
        else:
            raise E2EFailure(f"Interrupt type không mong đợi: {session['interrupt_type']}")

    tool_calls = await get_tool_calls(ctx, session_id)
    proposed = [tc for tc in tool_calls if tc["status"] == "proposed"]
    if not proposed:
        raise E2EFailure("propose_artifacts interrupt nhưng không có tool call nào ở trạng thái proposed")

    log(f"  {len(proposed)} tool call(s) proposed, thực hiện {args.tool_action}…")
    for tc in proposed:
        await _req(ctx.client, "POST",
                   f"/projects/{ctx.project_id}/agent-tool-calls/{tc['id']}/{args.tool_action}",
                   headers=ctx.headers)

    resumed = await poll_session(ctx, session_id, timeout_s=args.timeout)
    if resumed["status"] == "active":
        raise E2EFailure("Session vẫn active sau khi xử lý tool calls")

    log(f"  ok: {args.tool_action} {len(proposed)} tool call(s) → {resumed['status']} / {resumed.get('interrupt_type')}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def _require_provider_key() -> None:
    if not os.getenv("E2E_LLM_API_KEY"):
        raise E2EFailure("Scenario này cần E2E_LLM_API_KEY trong .env hoặc env")


async def run(args: argparse.Namespace) -> None:
    timeout = httpx.Timeout(args.request_timeout)
    async with httpx.AsyncClient(base_url=args.base_url.rstrip("/"), timeout=timeout) as client:
        health = await client.get("/health")
        _expect_status(health, 200, "GET /health")
        log(f"[health] {health.json()}")

        scenarios = args.scenario
        if "all" in scenarios:
            scenarios = ["no-provider", "ask-human", "propose"]

        for scenario in scenarios:
            if scenario == "no-provider":
                await scenario_no_provider(client, args)
            elif scenario == "ask-human":
                await scenario_ask_human(client, args)
            elif scenario == "propose":
                await scenario_propose(client, args)
            else:
                raise E2EFailure(f"Scenario không hỗ trợ: {scenario}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E2E smoke test AI Agent BRD flows.")
    parser.add_argument("--base-url", default=os.getenv("E2E_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument(
        "--scenario",
        action="append",
        choices=["all", "no-provider", "ask-human", "propose"],
        default=[],
        help="Có thể truyền nhiều lần. Mặc định: all.",
    )
    parser.add_argument("--timeout", type=float, default=float(os.getenv("E2E_TIMEOUT", "60")),
                        help="Timeout poll agent (giây).")
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--health-check", action="store_true",
                        help="Gọi health-check provider trước khi chạy agent.")
    parser.add_argument(
        "--artifact-type",
        default=os.getenv("E2E_ARTIFACT_TYPE", "goal"),
        help="Artifact type cho BRD session (E2E_ARTIFACT_TYPE, mặc định goal).",
    )
    parser.add_argument(
        "--tool-action",
        choices=["approve", "reject"],
        default=os.getenv("E2E_TOOL_ACTION", "approve"),
        help="Hành động với proposed tool calls (E2E_TOOL_ACTION, mặc định approve).",
    )
    args = parser.parse_args(argv)
    if not args.scenario:
        args.scenario = ["all"]
    return args


def main(argv: list[str] | None = None) -> int:
    configure_stdio()
    args = parse_args(argv or sys.argv[1:])
    try:
        asyncio.run(run(args))
    except (httpx.HTTPError, E2EFailure) as exc:
        print(f"\nE2E FAILED: {exc}", file=sys.stderr)
        return 1
    print("\nE2E PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
