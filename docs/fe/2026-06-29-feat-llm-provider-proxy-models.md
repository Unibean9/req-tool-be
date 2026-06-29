# FE Handoff: LLM Provider Proxy Models

> Branch: `feat/harness-req`
> Date: 2026-06-29

## 1) Endpoint map

- `POST /users/me/llm-provider-configs` — tạo hoặc thay thế cấu hình BYOK LLM provider mặc định của user.
- `GET /users/me/llm-provider-configs` — lấy danh sách cấu hình LLM provider còn hiệu lực.
- `GET /users/me/llm-provider-configs/{config_id}` — lấy chi tiết một cấu hình LLM provider.
- `PATCH /users/me/llm-provider-configs/{config_id}` — cập nhật model/region của cấu hình.
- `DELETE /users/me/llm-provider-configs/{config_id}` — vô hiệu hóa cấu hình.
- `POST /users/me/llm-provider-configs/{config_id}/health-check` — kiểm tra API key, model và khả năng tool calling.
- Impact liên quan: `POST /projects/{project_id}/agent-sessions/{session_id}/messages` sẽ reject provider config chưa `active` trước khi ghi user message.

## 2) Contracts

### Provider types

`provider_type` hiện nhận các giá trị:

| Value | Default `model_name` khi bỏ trống |
|-------|-----------------------------------|
| `bedrock` | `amazon.nova-lite-v1:0` |
| `openai` | `gpt-5.4-mini` |
| `google` | `gemini-3.5-flash` |
| `anthropic` | `claude-haiku-4-5-20251001` |
| `deepseek` | `deepseek-v4-flash` |
| `mistral` | `mistral-large-latest` |
| `openrouter` | `~openai/gpt-latest` |

### Create provider config

`POST /users/me/llm-provider-configs`

**Auth:** user đã đăng nhập.

**Body:**

```json
{
  "provider_type": "deepseek",
  "api_key": "sk-...",
  "secret_key": null,
  "region": null,
  "model_name": null,
  "strong_model_name": null
}
```

Validation rules:
- `provider_type` optional, default `openai`; enum gồm `bedrock`, `openai`, `google`, `anthropic`, `deepseek`, `mistral`, `openrouter`.
- `api_key` required, non-empty.
- `secret_key` optional, non-empty nếu gửi; chủ yếu dùng cho provider cần secret riêng như Bedrock.
- `region` optional, non-empty, tối đa 64 ký tự.
- `model_name` optional, non-empty, tối đa 255 ký tự; nếu bỏ trống backend dùng default theo `provider_type`.
- `strong_model_name` optional, non-empty, tối đa 255 ký tự.
- Extra fields bị reject.

**Response `data`:**
- `id` — UUID.
- `user_id` — UUID.
- `provider_type` — enum provider type.
- `name` — string, hiện bằng `provider_type`.
- `base_url` — string hoặc null.
- `region` — string hoặc null.
- `model_name` — string hoặc null.
- `strong_model_name` — string hoặc null.
- `status` — `draft`, `active`, `error`, hoặc `disabled`.
- `is_default` — boolean.
- `last_checked_at` — ISO datetime hoặc null.
- `last_check_error` — string hoặc null.
- `created_at` — ISO datetime.
- `updated_at` — ISO datetime.
- `api_key_set` — boolean.
- `secret_key_set` — boolean.

### List provider configs

`GET /users/me/llm-provider-configs`

**Auth:** user đã đăng nhập.

**Query params:** không có.

**Response `data`:** array của `LLMProviderConfigRead`. Các config `disabled` không được trả về.

### Get provider config

`GET /users/me/llm-provider-configs/{config_id}`

**Auth:** user đã đăng nhập.

**Route params:**

| Param | Type | Note |
|-------|------|------|
| `config_id` | UUID | ID cấu hình thuộc user hiện tại |

**Response `data`:** `LLMProviderConfigRead`.

### Update provider config

`PATCH /users/me/llm-provider-configs/{config_id}`

**Auth:** user đã đăng nhập.

**Route params:**

| Param | Type | Note |
|-------|------|------|
| `config_id` | UUID | ID cấu hình thuộc user hiện tại |

**Body:**

```json
{
  "region": null,
  "model_name": "mistral-large-latest",
  "strong_model_name": null
}
```

Validation rules:
- Ít nhất một field phải được gửi.
- `region` optional, non-empty, tối đa 64 ký tự.
- `model_name` optional, non-empty, tối đa 255 ký tự; nếu gửi `null`, backend reset về default của provider hiện tại.
- `strong_model_name` optional, non-empty, tối đa 255 ký tự.
- Không thể đổi `provider_type`, `api_key`, `secret_key`, hoặc `is_default` qua endpoint này.

**Response `data`:** `LLMProviderConfigRead`. Sau update, backend reset `status` về `draft`, clear `last_checked_at` và `last_check_error`.

### Delete provider config

`DELETE /users/me/llm-provider-configs/{config_id}`

**Auth:** user đã đăng nhập.

**Route params:**

| Param | Type | Note |
|-------|------|------|
| `config_id` | UUID | ID cấu hình thuộc user hiện tại |

**Response:** `204 No Content`.

### Health check provider config

`POST /users/me/llm-provider-configs/{config_id}/health-check`

**Auth:** user đã đăng nhập.

**Route params:**

| Param | Type | Note |
|-------|------|------|
| `config_id` | UUID | ID cấu hình thuộc user hiện tại |

**Body:** không có.

**Response `data`:**
- `config` — `LLMProviderConfigRead` sau khi cập nhật trạng thái.
- `response_time_ms` — number.
- `provider_reply` — string hoặc null.
- `tool_calling_supported` — boolean hoặc null.

Behavior:
- Thành công chỉ khi backend ping được provider và probe native tool calling trả `true`.
- Nếu provider ping được nhưng model không hỗ trợ native tool calling, backend set config `status=error`, set `last_check_error="Model khong ho tro tool calling"` và trả `422`.
- Chỉ config `status=active` mới được dùng cho agent runtime.

### Agent message impact

`POST /projects/{project_id}/agent-sessions/{session_id}/messages`

**Thay đổi contract:** nếu session đang trỏ tới `provider_config_id` có status khác `active`, backend trả `422` với detail `LLM provider config must pass health check before use`.

FE impact:
- Không cho user chọn hoặc bắt đầu agent runtime với config `draft`, `error`, hoặc `disabled`.
- Sau `POST`/`PATCH` provider config, UI nên yêu cầu chạy health check lại trước khi tạo/gửi message cho agent session.
- Khi gặp lỗi `422` này, không append optimistic user message vì backend reject trước khi ghi message.

## 3) Error codes

| HTTP | code | message |
|------|------|---------|
| 401 | auth | User chưa đăng nhập hoặc token không hợp lệ. |
| 404 | not_found | `LLM provider config not found` |
| 422 | validation_error | Pydantic/FastAPI validation errors, gồm enum `provider_type` không hợp lệ hoặc extra fields. |
| 422 | provider_capability | `Model khong ho tro tool calling`; model ping được nhưng không đạt contract tool calling của agent. |
| 422 | provider_config_not_active | `LLM provider config must pass health check before use`; agent message dùng config chưa `active`. |
| 429 | cooldown | `Health check is in cooldown` |
| 503 | provider_unavailable | Provider ping lỗi; backend trả message đã sanitize và không leak secret. |

## 4) FE notes

- Thêm `deepseek`, `mistral`, `openrouter` vào dropdown/provider picker.
- Nếu user không nhập `model_name`, hiển thị default theo bảng provider types để tránh cảm giác field bị mất.
- `POST` hiện có hành vi gần giống replace/upsert cấu hình active của user: backend có thể cập nhật config đang có và disable config khác, dù status code vẫn là `201`.
- Sau `POST` hoặc `PATCH`, trạng thái thường là `draft`; user phải chạy health check thành công trước khi dùng config cho agent.
- `status=active` chỉ được set khi backend probe được native tool calling; FE nên chặn/disable chọn config `draft` hoặc `error` cho agent runtime.
- Agent send-message reject sớm với `422` nếu config chưa `active`; trạng thái session không bị chuyển sang processing và user message không được lưu.
- Health check có cooldown 30 giây; disable nút hoặc hiển thị countdown sau lỗi `429`.
- `deepseek`, `mistral`, `openrouter` dùng luồng chat completions/tool calling; FE không cần đổi request shape ngoài `provider_type` và `model_name`.
