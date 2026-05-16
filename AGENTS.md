# AI-DataFlux repository agent instructions

## Purpose

AI-DataFlux is a Python 3.10+ batch AI processing engine with an OpenAI-compatible gateway, multiple datasource task pools, and a React/Vite local control panel.

## Codex startup behavior

- Codex is normally started from the repository root. This file is the startup router for the whole repository.
- Subdirectory `AGENTS.md` files are local navigation cards. They are not always loaded automatically when Codex starts at the root.
- Before editing files under any directory marked `Yes` in the `Local AGENTS.md` column, read that local file first with `cat <path>/AGENTS.md`.
- If more than one nested `AGENTS.md` applies, read them from shallow to deep before editing. For example, changes under `src/data/feishu/` require `src/data/AGENTS.md` and then `src/data/feishu/AGENTS.md`.
- If Codex is started from a subdirectory, local cards may be loaded by the path chain, but this root file remains the router for root-start workflows.

## Directory map

| Path | Responsibility | Local AGENTS.md | Read when |
|---|---|---:|---|
| `cli.py` | Unified CLI for `process`, `gateway`, `gui`, `token`, `check`, and `version`. | No | Root rules are enough; inspect this file before adding CLI commands. |
| `main.py` | Direct processing entry point that delegates to `src.core.processor`. | No | When changing process startup behavior. |
| `gateway.py` | Direct gateway entry point that delegates to `src.gateway.app`. | No | When changing standalone gateway startup. |
| `src/config/` | YAML loading, defaults, semantic validation, and logging setup. | No | When changing config schema; also check `web/`, `docs/`, and tests. |
| `src/core/` | Core processing pipeline: coordinator, scheduler, content parsing, retry, state, token estimation, and API client. | Yes | Before changing processing flow, routing, retry decisions, progress reporting, or token estimation. |
| `src/data/` | Datasource task pools, factory, file/DB/Feishu integrations, and DataFrame engine abstraction. | Yes | Before changing task pool contracts, datasource behavior, writeback semantics, or adding a datasource. |
| `src/data/engines/` | Pandas/Polars `BaseEngine` implementations and optional library detection. | Yes | Before changing DataFrame operations, engine selection, reader/writer fallback, or vectorized filtering. |
| `src/data/feishu/` | Native aiohttp Feishu client plus Bitable/Sheet task pools. | Yes | Before changing Feishu auth, rate limits, pagination, chunking, snapshots, or writeback. |
| `src/gateway/` | FastAPI OpenAI-compatible gateway, model dispatch, rate limiting, session pool, DNS resolver, schemas. | Yes | Before changing `/v1/*`, `/admin/*`, model routing, streaming, limiter, or upstream HTTP behavior. |
| `src/control/` | FastAPI control server for GUI, config file API, process management, auth, status, and log WebSockets. | Yes | Before changing GUI backend APIs, process lifecycle, config read/write, auth, static serving, or logs. |
| `src/models/` | Shared dataclasses and error enums. | No | Follow the closest caller card, usually `src/core/` or `src/gateway/`. |
| `src/utils/` | Console/output helpers. | No | Root rules are enough. |
| `web/` | React 19 + TypeScript + Vite control panel. | Yes | Before changing frontend API calls, config editor, dashboard, logs, i18n, or build settings. |
| `web/dist/` | Generated frontend build output served by `src/control/`. | No | Do not edit manually; regenerate with `cd web && npm run build`. |
| `tests/` | Pytest suite, fixtures, markers, unit/integration coverage. | Yes | Before changing shared fixtures, test conventions, markers, or broad test behavior. |
| `docs/` | Chinese architecture, config, datasource, GUI, routing, and build docs. | Yes | Before changing docs or when code changes affect documented config/API/CLI behavior. |
| `.github/` | GitHub Actions workflows and CI helper scripts. | Yes | Before changing CI, packaging, release, platform matrix, or dependency checks. |
| `legacy/` | Historical reference implementations. | Yes | Before using legacy code as migration input; do not edit casually. |
| `.config/rules/` | Local routing profile examples/configs, ignored by git. | No | Read only when a routing task references local profiles. |
| `.examples/`, `local/`, `.codex/`, `.claude/` | Local examples, reports, and agent/runtime material. | No | Treat as local context unless the user explicitly asks. |
| `config-example.yaml` | Public configuration template. | No | Update together with config schema changes. |
| `config.yaml` | Local ignored runtime configuration. | No | May contain test credentials; use only when user asks or when local validation needs it. |

## On-demand cat protocol

Before editing files under a directory that has a local `AGENTS.md`, run:

```bash
cat <path>/AGENTS.md
```

For nested cards, read them in order. Example:

```bash
cat src/data/AGENTS.md
cat src/data/feishu/AGENTS.md
```

Use the local card for directory-specific invariants. If a local card conflicts with this root file, the local card wins for that subtree.

## Commands

| Command | Purpose | Scope | Sandbox notes |
|---|---|---|---|
| `python -m pip install -r requirements-core.txt` | Install core runtime dependencies. | Python | Needs network unless packages are cached. |
| `python -m pip install -r requirements-optional.txt` | Install optional gateway/performance/token dependencies. | Python | Needs network; some optional wheels may be unavailable on specific platforms. |
| `python -m pip install -r requirements.txt` | Install full runtime dependency set. | Python | Needs network unless cached. |
| `python cli.py check` | Check available runtime and optional libraries. | Python | Requires installed dependencies; no external service expected. |
| `python cli.py version` | Print CLI version. | Python | Local. |
| `python cli.py process --config config-example.yaml --validate` | Validate example config without running jobs. | Python | Local; uses only local semantic validation. |
| `python cli.py process --config config.yaml --validate` | Validate local config. | Python | Uses ignored local config; may contain test credentials. |
| `python cli.py process --config config.yaml` | Run processing. | Python | Requires configured datasource and gateway/API availability. |
| `python cli.py gateway --port 8787` | Start OpenAI-compatible gateway. | Python | Long-running service; requires valid model/channel config for real traffic. |
| `python gateway.py --config config.yaml --port 8787` | Start gateway through standalone entry point. | Python | Long-running service; same external config constraints as gateway CLI. |
| `DATAFLUX_CONTROL_TOKEN=test-token python cli.py gui --no-browser` | Start local control panel with fixed auth token. | Python/control | Long-running service on local port; requires `web/dist/` for built frontend serving. |
| `python cli.py token --config config.yaml` | Estimate input/output token usage. | Python | Requires datasource access and `tiktoken` for full token estimation. |
| `pytest tests/ -v -m "not integration"` | Main non-integration pytest suite. | Tests | Requires test dependencies; should not require external DB/API. |
| `pytest tests/ -v -m integration` | Integration-marked tests. | Tests | CI describes current integration tests as local temp-file/data tests; still heavier than unit tests. |
| `pytest tests/ --cov=src --cov-report=term-missing` | Coverage report. | Tests | Requires `pytest-cov`; `.coveragerc` omits `src/gateway/*`. |
| `ruff check src/ tests/ cli.py main.py gateway.py` | Python lint check used by repository guidelines. | Python | Requires `ruff`. |
| `black --check src/ tests/ cli.py main.py gateway.py` | Python format check used by repository guidelines. | Python | Requires `black`; CI pins `black==25.11.0`. |
| `mypy src/ --ignore-missing-imports` | CI type check. | Python | CI allows this to fail with `|| true`; do not treat as the only gate. |
| `python -m py_compile cli.py main.py gateway.py` | Syntax check root entry points. | Python | Local. |
| `find src -name "*.py" -exec python -m py_compile {} \;` | Syntax check package modules. | Python | Local. |
| `cd web && npm ci` | Install frontend dependencies from `package-lock.json`. | Web | Needs network unless npm cache is warm. |
| `cd web && npm run build` | TypeScript build plus Vite production build. | Web | Requires installed npm deps; writes `web/dist/`. |
| `cd web && npm run lint` | ESLint check. | Web | Requires installed npm deps. |
| `cd web && npm run dev` | Start Vite dev server. | Web | Long-running local service. |
| `cd web && npm run preview` | Preview built frontend. | Web | Long-running local service; requires `web/dist/`. |

CI-specific packaging commands in `.github/workflows/build-*.yml` use PyInstaller/Nuitka, compilers, GitHub release actions, and sometimes platform setup steps. Do not treat those as default local validation.

## Global rules

- Default communication with the user is Chinese. Keep code, commands, file paths, API names, and config keys in English.
- Python targets 3.10+. Use 4-space indentation, `snake_case` for functions/modules, and `PascalCase` for classes.
- Follow existing module boundaries. Prefer extending `src/core/`, `src/data/`, `src/gateway/`, `src/control/`, and `web/` through their existing factories, components, and helper APIs.
- Preserve optional dependency behavior. Imports for optional libraries must stay guarded or lazy so missing `polars`, `fastexcel`, `xlsxwriter`, `aiohttp`, `psutil`, `mysql-connector-python`, `psycopg2`, or `tiktoken` degrades as designed.
- Config schema changes are cross-cutting. Check `src/config/settings.py`, `config-example.yaml`, `docs/CONFIG.md`, relevant `web/src/components/config/sections/*`, `web/src/types.ts`, and tests before claiming completion.
- CLI surface changes require `cli.py`, help tests in `tests/test_cli.py`, and docs/examples to stay aligned.
- Datasource behavior must preserve `BaseTaskPool` contracts: shard boundaries, `get_task_batch`, `reload_task_data`, `update_task_results`, `close`, and token estimation sampling hooks.
- DataFrame engine behavior must go through `BaseEngine` where possible; avoid direct pandas/polars calls in higher-level datasource code when an engine method already exists.
- Gateway request/response changes must preserve OpenAI Chat Completions compatibility unless the user explicitly asks to break it.
- Control server changes must preserve local-only serving defaults, token auth for `/api/*` and `/api/logs`, YAML-only config read/write, and project-root path containment.
- Frontend changes should reuse existing shared controls in `web/src/components/config/shared/` and keep TypeScript API types aligned with backend responses.
- Tests should be focused near the changed behavior. Prefer existing fixtures in `tests/conftest.py`; use mocks/temp files for external APIs and databases unless an integration task explicitly asks for real services.
- Generated or build output is not source. Regenerate `web/dist/` with the frontend build instead of hand-editing it.

## Do not

- Do not modify `AGENTS.override.md` or create one unless the user explicitly requests that strategy.
- Do not commit or print local secrets from `config.yaml`, `.env`, datasource credentials, model API keys, or Feishu tokens. It is acceptable to use local test credentials when the user asks for local validation.
- Do not hand-edit `web/dist/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, coverage artifacts, `node_modules/`, or packaging output.
- Do not replace the existing architecture with a new framework when an established local module already covers the need.
- Do not run publishing/release operations, upload artifacts, or create GitHub releases unless explicitly requested.
- Do not silently remove platform support from CI matrices or packaging workflows.
- Do not make tests depend on real external AI providers, Feishu, MySQL, PostgreSQL, or local ports unless the test is explicitly marked and documented for that dependency.

## Validation

For a typical Python-only change:

1. Run a targeted pytest command for the changed module.
2. Run `pytest tests/ -v -m "not integration"` when the blast radius is broader.
3. Run `ruff check src/ tests/ cli.py main.py gateway.py`.
4. Run `black --check src/ tests/ cli.py main.py gateway.py`.

For frontend changes:

1. Run `cd web && npm run lint`.
2. Run `cd web && npm run build`.
3. If behavior changed in the GUI, validate through the control server or browser workflow when practical.

For config/schema changes:

1. Run `python cli.py process --config config-example.yaml --validate`.
2. Run related config tests, usually `pytest tests/test_config.py -v`.
3. Check backend, frontend, docs, and example config for the same field names and defaults.

For gateway changes:

1. Run `pytest tests/test_gateway_service.py -v`.
2. Run `python cli.py gateway --help`.
3. Start the gateway only when a valid config and port are available; report if full upstream validation was skipped.

For control server changes:

1. Run `pytest tests/test_control.py -v`.
2. For manual smoke testing, use `DATAFLUX_CONTROL_TOKEN=test-token python cli.py gui --no-browser`.
3. If testing actual GUI pages, ensure `web/dist/` exists or rebuild it.

If a validation command cannot run because dependencies, network, credentials, ports, or external services are unavailable, say exactly what was skipped and why.

## Notes for future agents

- `CLAUDE.md` contains useful historical architecture notes, but `AGENTS.md` is the Codex project instruction surface. Keep future Codex-specific guidance here.
- `AGENTS.md` is ignored by this repository's `.gitignore`, so normal `git diff` may not show these instruction files unless they are force-added or compared manually.
- The project has Chinese docs and many Chinese comments. Keep new agent guidance in Chinese unless there is a strong local reason to use English.
