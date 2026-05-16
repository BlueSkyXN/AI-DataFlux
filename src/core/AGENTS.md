# src/core navigation card

Core processing pipeline for AI-DataFlux. Read this before modifying `processor.py`, `scheduler.py`, `validator.py`, `token_estimator.py`, or `clients/`, `content/`, `retry/`, and `state/`.
Key files: `src/core/processor.py`, `src/core/scheduler.py`, `src/core/content/processor.py`, `src/core/retry/strategy.py`, `src/core/state/manager.py`, `src/core/clients/flux_client.py`.

## Local invariants

- `UniversalAIProcessor` is the coordinator. Keep data access in `src/data/`, API calls in `FluxAIClient`, parsing in `ContentProcessor`, retry decisions in `RetryStrategy`, and metadata in `TaskStateManager`.
- Preserve continuous task flow: maintain active concurrency, process completed tasks with `asyncio.wait(... FIRST_COMPLETED)`, and avoid reverting to "wait for whole batch then write" behavior.
- Error semantics are contractual: `API_ERROR` pauses and reloads data; `CONTENT_ERROR` retries without reload; `SYSTEM_ERROR` reloads without global pause.
- `TaskMetadata` must stay separate from business row data. Do not store retry counters or error history inside datasource records.
- Routing profiles may only override `prompt` and `validation`. Implicit routing fields are used for routing but excluded from prompt payloads.
- Progress JSON must be atomic, refreshed during long runs, and removed on normal completion.

## Local rules

- Add or update focused tests under `tests/core/` or the matching top-level `tests/test_*.py` file when changing component behavior.
- Keep `token_estimator.py` tolerant of missing `tiktoken` until estimation is actually requested.
- Do not let core modules import frontend/control concerns. The processing engine should remain usable from CLI and tests without the GUI.

## Validation

- `pytest tests/core/ tests/test_scheduler.py tests/test_validator.py tests/test_token_estimator.py -v`
- `pytest tests/test_integration.py -v` for end-to-end flow changes.
- Use root validation commands for broader changes.
