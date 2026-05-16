# legacy navigation card

Historical reference implementations for AI-DataFlux. Read this card before using files here as migration input or comparing old behavior to current `src/` modules.
Key files: `AI-DataFlux.py`, `Flux-Api.py`, `Flux_Data.py`.

## Why this is guarded

- `legacy/` is reference code, not the active runtime path.
- Copying behavior from here can reintroduce old architecture, duplicated logic, or outdated config semantics.

## Local rules

- Do not edit legacy files unless the user explicitly requests a legacy migration or archival cleanup.
- When porting behavior, implement it in the active modules under `src/` and add tests there.
- Treat legacy code as evidence to compare against current behavior, not as an import target.

## Validation

- Use the validation command for the active module that receives the migrated behavior.
- If only reading legacy code, no validation command is required.
