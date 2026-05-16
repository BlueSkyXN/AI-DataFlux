# .github navigation card

GitHub Actions workflows and CI helper scripts. Read this card before modifying workflow triggers, lint/test matrices, packaging jobs, release behavior, dependency checks, or CI scripts.
Key files: `workflows/test.yml`, `workflows/build-pyinstaller.yml`, `workflows/build-nuitka.yml`, `scripts/check_deps.py`.

## Why this is high-risk

- Build workflows create release artifacts and GitHub Releases on `v*` tags.
- Test workflow covers multiple Python versions and OS/architecture combinations.
- Packaging jobs depend on frontend build output, Python dependencies, compilers, and platform-specific commands.

## Local invariants

- `test.yml` quality checks are `ruff`, `black --check`, permissive `mypy`, syntax compile, unit matrix, CLI tests, performance library checks, and integration tests.
- Build workflows run `cd web && npm ci && npm run build` before full GUI packaging.
- PyInstaller has full and CLI-only variants; Nuitka builds full GUI bundles.
- Python and Node versions in workflow `env` are part of the release surface. Keep docs and build assumptions aligned when changing them.
- Optional dependencies may be allowed to fail during install in unit-test matrix jobs, but critical runtime checks should remain explicit.

## Do not

- Do not add plain-text secrets, tokens, or credentials to workflow files.
- Do not remove branches, paths, platforms, or release conditions without calling out the compatibility impact.
- Do not trigger releases, upload artifacts, or push tags unless the user explicitly asks.

## Validation

- `python -m py_compile .github/scripts/check_deps.py`
- There is no local GitHub Actions runner configured by default. For workflow syntax, inspect YAML carefully or use external validation only when available.
