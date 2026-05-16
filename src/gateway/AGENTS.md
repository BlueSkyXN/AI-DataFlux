# src/gateway navigation card

OpenAI-compatible FastAPI gateway for model dispatch, failover, rate limiting, session reuse, and admin health/model endpoints. Read this card before changing gateway routes, schemas, upstream request handling, streaming, dispatch, limiter, resolver, or session pooling.
Key files: `app.py`, `service.py`, `schemas.py`, `dispatcher.py`, `limiter.py`, `session.py`, `resolver.py`.

## Local invariants

- Preserve OpenAI Chat Completions compatibility for `/v1/chat/completions`, `/v1/models`, and request/response shapes unless the user explicitly asks for a breaking change.
- `ChatCompletionRequest` allows extra fields for forward compatibility. Do not drop unknown OpenAI-style fields unless they are intentionally unsupported and tested.
- Model name resolution maps configured `id`, `model`, and `name` aliases to internal IDs. Keep this compatible with existing configs.
- Dispatch must consider availability, model weight, rate limits, and exclusion of failed models during retry/failover.
- `chat_completion()` may retry another model when an upstream call fails, but should not loop indefinitely. Preserve the existing bounded retry behavior.
- Streaming responses must keep Server-Sent Events framing and `[DONE]` handling intact.
- `SessionPool` owns aiohttp connector/session reuse. Ensure startup/shutdown paths close async resources.

## Local rules

- Any public schema or endpoint change needs tests in `tests/test_gateway_service.py` or new focused tests.
- Keep proxy, `ssl_verify`, and optional IP-pool behavior wired through channel/model config.
- Do not let gateway code depend on `src/control/` or the React app.

## Validation

- `pytest tests/test_gateway_service.py -v`
- `python cli.py gateway --help`
- Starting a live gateway requires a valid config with models/channels and a free port.
