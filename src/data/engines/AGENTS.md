# src/data/engines navigation card

DataFrame engine abstraction for Pandas and Polars. Read this card before changing `BaseEngine`, `PandasEngine`, `PolarsEngine`, engine selection, optional library detection, or vectorized filtering.
Key files: `base.py`, `__init__.py`, `pandas_engine.py`, `polars_engine.py`.

## Local invariants

- `BaseEngine` is the public contract. Higher layers depend on its method names and return semantics, not on pandas/polars-specific APIs.
- Pandas and Polars implementations should preserve equivalent behavior for row IDs, column names, empty values, string conversion, slicing, batch updates, and CSV/Excel I/O.
- `engine: auto` prefers Polars when usable and falls back to Pandas. `excel_reader: auto` prefers calamine/fastexcel, and `excel_writer: auto` prefers xlsxwriter.
- Optional library checks intentionally run in subprocesses to avoid crashing the main process on incompatible platforms. Do not replace this with direct top-level imports.
- Methods that mutate data should keep the established return convention. If an engine returns a new DataFrame, callers must receive and use it.

## Local rules

- Add parity tests when changing any `BaseEngine` method.
- Keep performance paths vectorized where the existing API already has vectorized operations.
- Do not introduce a new engine without updating `__init__.py`, availability detection, docs/config references, and tests.

## Validation

- `pytest tests/test_engines.py tests/test_integration.py -v`
- `python cli.py check` after dependency or availability-detection changes.
