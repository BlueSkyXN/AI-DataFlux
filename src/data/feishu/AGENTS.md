# src/data/feishu navigation card

Native Feishu integration using aiohttp, without the official SDK. Read this card before modifying auth, request retry, Bitable/Sheet snapshots, pagination, chunking, or writeback.
Key files: `client.py`, `bitable.py`, `sheet.py`, `__init__.py`.

## Why this is high-risk

- Code here talks to external Feishu APIs and can update user spreadsheets or Bitable records.
- Auth tokens, app secrets, app tokens, table IDs, and spreadsheet tokens are sensitive even when local test credentials are used.
- Feishu rate limits and oversized range errors require careful retry/chunking behavior.

## Local invariants

- `FeishuClient` owns tenant token refresh, aiohttp session lifecycle, retry/backoff, QPS limiting, and concurrency limiting.
- A `ClientSession` is tied to its event loop. Preserve the existing event-loop switch detection and session rebuild behavior.
- 429, `Retry-After`, 5xx, network errors, token invalidation, Feishu business rate limits, and too-large errors must keep their distinct handling paths.
- Bitable uses integer `task_id` values mapped to string `record_id` values through stable snapshot maps.
- Sheet uses 0-based data row task IDs; actual spreadsheet row numbers account for the header row.
- Sheet writes are serialized by design. Do not parallelize writes for one document unless the API behavior is revalidated.

## Do not

- Do not add new logs that expose `app_secret`, tenant access tokens, bearer headers, or full credential payloads.
- Do not require real Feishu network calls in default tests.
- Do not replace split/chunk logic with recursion that can overflow or lose partial progress on large ranges.

## Validation

- `pytest tests/test_feishu_client_async.py tests/test_feishu_pool.py -v`
- For real API smoke tests, explicitly state that Feishu credentials and network access are required.
