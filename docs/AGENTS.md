# docs navigation card

Chinese documentation for architecture, configuration, datasources, Feishu, GUI, routing, and build variants. Read this card before changing docs or when code changes alter documented commands, config fields, APIs, or behavior.
Key files: `ARCH.md`, `CONFIG.md`, `DATA_SOURCE.md`, `FEISHU.md`, `GUI.md`, `ROUTING.md`, `BUILD_VARIANTS.md`, `README.md`.

## Local invariants

- Documentation should be derived from current code and config, not guessed. Check source files before documenting behavior.
- Keep command examples aligned with actual CLI and `web/package.json` scripts.
- Config docs must stay aligned with `src/config/settings.py`, `config-example.yaml`, frontend config sections, and `tests/test_config.py`.
- API/control docs must stay aligned with `src/control/server.py`, `src/control/config_api.py`, `src/control/process_manager.py`, and `web/src/api.ts`.
- Gateway docs must preserve OpenAI compatibility details unless code intentionally changes them.
- Use placeholders for credentials and tokens. Do not paste real local `config.yaml` secrets into docs.
- Historical performance or coverage numbers should be labeled as examples unless they were freshly measured.

## Local rules

- Prefer Chinese prose to match the existing docs.
- Keep diagrams and file maps compact; update only the affected sections.
- If docs mention a command as validation, ensure it exists in the repository config or CLI.

## Validation

- No docs build command is configured.
- For docs-only changes, review rendered Markdown when practical.
- For docs tied to code behavior, run the relevant code/test validation from the root `AGENTS.md`.
