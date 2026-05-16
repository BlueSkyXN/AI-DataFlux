# web navigation card

React 19 + TypeScript + Vite control panel for AI-DataFlux. Read this card before changing frontend API calls, dashboard controls, config editor sections, log streaming, i18n, build config, or generated `dist` handling.
Key files: `src/api.ts`, `src/types.ts`, `src/App.tsx`, `src/pages/`, `src/components/config/`, `vite.config.ts`, `package.json`.

## Local invariants

- Package manager is npm, backed by `package-lock.json`. Use `npm ci` for clean installs.
- API calls use relative URLs so the app works when served by `src/control/server.py` from the same origin.
- Control token handling lives in `src/api.ts`: URL `#token=...` is persisted to `sessionStorage`, removed from the address bar, and sent as Bearer auth.
- Log WebSockets pass auth through `Sec-WebSocket-Protocol` using the `dataflux-token-b64.<base64url>` format. Keep this aligned with `src/control/server.py`.
- `src/types.ts` mirrors backend API responses. Update it with backend schema changes.
- Config editor sections must write YAML paths that backend validation and `config-example.yaml` understand.
- Reuse controls in `src/components/config/shared/` for form fields, toggles, number inputs, lists, cards, and selects.

## Do not

- Do not hardcode `localhost` or an absolute backend URL into API helpers.
- Do not hand-edit `web/dist/`; regenerate it with `npm run build`.
- Do not introduce new config keys only in the frontend. Backend validation, docs, examples, and tests must be updated too.

## Validation

- `cd web && npm run lint`
- `cd web && npm run build`
- For API contract changes, also run relevant backend tests, usually `pytest tests/test_control.py tests/test_config.py -v`.
