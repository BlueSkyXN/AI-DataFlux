# src/control navigation card

FastAPI backend for the local control panel. Read this before changing control APIs, auth, config read/write, process management, status polling, log WebSockets, or static serving.
Key files: `server.py`, `config_api.py`, `process_manager.py`, `runtime.py`.

## Why this is high-risk

- The control server can start and stop subprocesses and write YAML config files.
- It exposes local HTTP and WebSocket endpoints consumed by the React app.
- Path validation and token auth are the main guardrails.

## Local invariants

- `/api/*` and `/api/logs` must require the control token. Token sources are `DATAFLUX_CONTROL_TOKEN` or a generated token, compared with `secrets.compare_digest`.
- Browser WebSocket auth uses `Sec-WebSocket-Protocol` with `dataflux-token-b64.<base64url>`. Keep frontend and backend in sync.
- Config read/write is limited to `.yaml` and `.yml` inside `PROJECT_ROOT`; preserve realpath/commonpath containment checks.
- Config writes must remain backup + atomic replace operations.
- `ProcessManager` manages `gateway` and `process` states as `stopped`, `running`, or `exited`; stop operations must terminate child process trees where possible.
- Progress file handling must tolerate stale or missing files and avoid blocking the event loop during status polling.
- Static serving depends on `web/dist/`; root detection lives in `runtime.py`.

## Do not

- Do not broaden config file access to arbitrary file types without explicit user approval.
- Do not remove auth checks for convenience tests; tests should provide a token.
- Do not block the FastAPI event loop with long synchronous probes when an async/cache path exists.

## Validation

- `pytest tests/test_control.py -v`
- `DATAFLUX_CONTROL_TOKEN=test-token python cli.py gui --no-browser` for manual smoke testing.
- Rebuild frontend with `cd web && npm run build` if static asset behavior changed.
