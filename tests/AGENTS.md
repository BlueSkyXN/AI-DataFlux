# tests navigation card

Pytest suite for CLI, config, core components, datasources, gateway, control server, and integration behavior. Read this card before changing shared fixtures, markers, test layout, or broad test patterns.
Key files: `pytest.ini`, `tests/conftest.py`, `tests/README.md`, `tests/test_*.py`, `tests/core/`.

## Local invariants

- Test discovery follows `pytest.ini`: files `test_*.py`, classes `Test*`, functions `test_*`.
- Shared fixtures belong in `tests/conftest.py`; keep temporary files under pytest-provided temp paths.
- Use markers consistently: `integration` for integration tests and `slow` for expensive tests.
- Default tests should not require live AI APIs, Feishu, MySQL, PostgreSQL, or persistent local services. Mock external HTTP/API behavior unless an integration task explicitly asks otherwise.
- Optional dependency behavior is part of test coverage. Tests should verify fallback/skip behavior instead of assuming every optional library is installed.
- Component tests under `tests/core/` should stay focused on isolated logic.

## Local rules

- When fixing a bug, add or update the smallest regression test that fails without the fix.
- Keep test names behavior-oriented enough to diagnose failures from CI logs.
- Avoid asserting on brittle full log strings when stable fields or substrings are enough.

## Validation

- `pytest tests/ -v -m "not integration"`
- `pytest tests/ -v -m integration`
- `pytest tests/ --cov=src --cov-report=term-missing`
