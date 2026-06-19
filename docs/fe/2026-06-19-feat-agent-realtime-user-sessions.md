# FE Handoff: Agent Realtime User Sessions

> Branch: `feat/harness-req`
> Date: 2026-06-19

---

## 1) Endpoint map

- `GET /projects/{project_id}/agent-sessions/{session_id}/events` — stream trạng thái agent session bằng SSE
- `GET /projects/{project_id}/agent-sessions/{session_id}` — chỉ trả session do user hiện tại tạo
- `GET /projects/{project_id}/agent-sessions/{session_id}/messages` — chỉ trả messages của session do user hiện tại tạo
- `GET /projects/{project_id}/agent-sessions/{session_id}/tool-calls` — chỉ trả tool calls của session do user hiện tại tạo
- `POST /projects/{project_id}/agent-sessions/{session_id}/messages` — chỉ reply vào session do user hiện tại tạo
- `POST /projects/{project_id}/agent-tool-calls/{tool_call_id}/approve` — chỉ approve tool call thuộc session của user hiện tại
- `POST /projects/{project_id}/agent-tool-calls/{tool_call_id}/reject` — chỉ reject tool call thuộc session của user hiện tại
- `POST /projects/{project_id}/agent-tool-calls/{tool_call_id}/request-edit` — chỉ request edit tool call thuộc session của user hiện tại

---

## 2) Contracts

### GET /projects/{project_id}/agent-sessions/{session_id}/events

**Auth:** authenticated user, project member, session owner (`created_by_id == current_user.id`)

**Request headers:**

```http
Accept: text/event-stream
Authorization: Bearer <token>
```

**Response:** `text/event-stream`

Event đầu tiên luôn là `snapshot`:

```text
id: 2026-06-19T14:00:00+00:00:{session_id}:0
event: snapshot
data: {"type":"snapshot","session":{...},"messages":[...],"tool_calls":[...]}
```

**Snapshot `data`:**

```json
{
  "type": "snapshot",
  "session": {
    "id": "uuid",
    "project_id": "uuid",
    "created_by_id": "uuid",
    "artifact_type": "goal",
    "workflow_area": "analysis",
    "status": "waiting_for_human",
    "interrupt_type": "propose_artifacts",
    "missing_context": ["intent", "problem"],
    "updated_at": "2026-06-19T14:00:00Z"
  },
  "messages": [
    {
      "id": "uuid",
      "session_id": "uuid",
      "role": "agent",
      "content": "Tôi đã có đủ thông tin để tạo goal...",
      "created_at": "2026-06-19T14:00:00Z",
      "updated_at": "2026-06-19T14:00:00Z"
    }
  ],
  "tool_calls": [
    {
      "id": "uuid",
      "run_id": "uuid",
      "tool_name": "create_artifact",
      "input_snapshot": {
        "artifact_type": "goal",
        "title": "Giảm vòng làm rõ yêu cầu"
      },
      "status": "proposed",
      "created_artifact_id": null,
      "created_version_id": null,
      "resolved_at": null,
      "created_at": "2026-06-19T14:00:00Z",
      "updated_at": "2026-06-19T14:00:00Z"
    }
  ]
}
```

**Final event:** khi session đã `completed` hoặc `failed`, stream gửi thêm:

```text
event: stream_closed
data: {"type":"stream_closed","status":"completed"}
```

Sau đó backend đóng stream.

---

### Ownership thay đổi cho các endpoint cũ

Các endpoint session/message/tool-call cũ giữ nguyên path và response shape, nhưng đổi quyền truy cập:

```text
Project member là điều kiện cần.
Session owner mới là điều kiện đủ.
```

Backend filter bằng:

```text
AgentSession.created_by_id == current_user.id
```

Với tool call, backend resolve qua:

```text
AgentToolCall -> AgentRun -> AgentSession.created_by_id
```

FE không được giả định session là shared theo project nữa.

---

### POST /projects/{project_id}/agent-sessions

**Auth:** authenticated user, project member

Body giữ nguyên:

```json
{
  "artifact_type": "goal",
  "workflow_area": "analysis",
  "step_key": null,
  "agent_role": null,
  "provider_config_id": null
}
```

Response giữ nguyên:

```json
{
  "session_id": "uuid",
  "missing_context": ["intent", "problem"]
}
```

**Thay đổi quan trọng:** backend gán session mới cho user hiện tại:

```text
created_by_id = current_user.id
```

Unique active session giờ là per-user:

```text
(project_id, artifact_type, created_by_id)
```

Hai user trong cùng project có thể cùng tạo active session `artifact_type="goal"` mà không bị conflict.

---

## 3) Error codes

| HTTP | Tình huống | Message |
|------|------------|---------|
| 400 | Gửi message khi session không ở trạng thái chờ người dùng | `"Session không ở trạng thái chờ người dùng"` |
| 400 | Gửi message khi session đang chờ approve tool calls | `"Session đang chờ approval tool calls, không phải user message"` |
| 400 | Tool call không ở trạng thái `proposed` | `"Tool call không ở trạng thái proposed"` |
| 404 | Project không tồn tại hoặc user không phải project member | `"Không tìm thấy dự án"` |
| 404 | Session không tồn tại, không thuộc project, hoặc thuộc user khác | `"Agent session không tồn tại"` |
| 404 | Tool call không tồn tại, không thuộc project, hoặc thuộc session của user khác | `"Tool call không tồn tại"` |
| 409 | User hiện tại đã có active session cùng artifact type trong project | `{ "detail": "Active session already exists", "session_id": "uuid" }` |
| 503 | Agent graph chưa sẵn sàng | `"Agent service chưa sẵn sàng"` |

---

## 4) FE notes

- Ưu tiên dùng SSE `/events` thay cho polling 3 endpoint session/messages/tool-calls.
- Vẫn giữ fallback polling trong giai đoạn rollout nếu browser/client không hỗ trợ stream auth.
- `EventSource` browser native không set được `Authorization` header. Nếu FE dùng bearer token header, dùng `fetch` streaming hoặc thêm cơ chế stream token ngắn hạn ở phase sau.
- Event hiện tại là snapshot-based. FE có thể replace local session state bằng snapshot mới nhất thay vì merge phức tạp.
- Không hiển thị session của user khác trong cùng project. Nếu API trả 404 cho một session id từng thấy ở user khác, xử lý như không tồn tại.
- Khi nhận `stream_closed`, FE đóng client stream và giữ state cuối.
- Payload stream không có `graph_checkpoint`, raw prompt, API key hoặc provider secret.
