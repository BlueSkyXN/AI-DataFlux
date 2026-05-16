# src/data navigation card

Datasource layer for file, database, and Feishu task pools. Read this before changing `base.py`, `factory.py`, `excel.py`, `mysql.py`, `postgresql.py`, `sqlite.py`, or nested datasource modules.
Key files: `src/data/base.py`, `src/data/factory.py`, `src/data/excel.py`, `src/data/mysql.py`, `src/data/postgresql.py`, `src/data/sqlite.py`.

## Local invariants

- Every datasource must satisfy `BaseTaskPool`: counts, ID boundaries, shard loading, batch pop, writeback, reload, sampling hooks, and `close()`.
- `columns_to_extract` names input fields. `columns_to_write` maps output aliases to actual datasource columns; AI result dictionaries use aliases.
- Unprocessed/processed semantics must stay consistent across datasources: input validity respects `require_all_input_fields`; processed rows have all output columns non-empty.
- `create_task_pool()` is the datasource factory. Adding a datasource requires implementation, guarded dependency detection if needed, factory wiring, docs/config updates, and tests.
- Optional DB and file dependencies must remain optional. Missing MySQL/PostgreSQL/Excel/performance libraries should fail with clear errors only when that datasource or feature is selected.
- File datasources should use `src/data/engines/` abstractions instead of bypassing them with direct pandas/polars code.
- Database writeback must preserve transactions, parameterized values, and identifier validation.

## Local rules

- If editing `src/data/engines/`, also read `src/data/engines/AGENTS.md`.
- If editing `src/data/feishu/`, also read `src/data/feishu/AGENTS.md`.
- Tests should use temp files, mocks, or isolated local databases, not real external MySQL/PostgreSQL/Feishu services.

## Validation

- `pytest tests/test_factory.py tests/test_csv_pool.py tests/test_sqlite_pool.py tests/test_postgresql_pool.py -v`
- `pytest tests/test_engines.py -v` when file/engine behavior changes.
- `pytest tests/test_feishu_client_async.py tests/test_feishu_pool.py -v` when Feishu behavior changes.
