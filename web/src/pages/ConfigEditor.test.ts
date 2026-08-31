import { describe, expect, it } from 'vitest';

import { parseYaml, serializeYaml, updateConfigValue } from './configDocument';

describe('canonical v4 config editing', () => {
  it('preserves unrendered route, fallback, affinity, and writeback fields', () => {
    const original = {
      schema_version: 4,
      runtime: { log: { level: 'info' } },
      job: {
        gateway_url: 'http://127.0.0.1:8787',
        writeback: { reconciliation_max_attempts: 7 },
      },
      gateway: {
        routes: [{ id: 'route-a', aliases: ['kept-alias'] }],
        fallback_groups: { primary: ['route-a'] },
        affinity: { ttl_seconds: 7200, max_entries: 222 },
      },
    };

    const edited = updateConfigValue(original, ['runtime', 'log', 'level'], 'debug');
    const roundTrip = parseYaml(serializeYaml(edited));

    expect(roundTrip).toMatchObject({
      runtime: { log: { level: 'debug' } },
      job: { writeback: { reconciliation_max_attempts: 7 } },
      gateway: {
        routes: [{ aliases: ['kept-alias'] }],
        fallback_groups: { primary: ['route-a'] },
        affinity: { ttl_seconds: 7200, max_entries: 222 },
      },
    });
  });

  it('does not migrate legacy YAML on load', () => {
    const parsed = parseYaml('global:\n  flux_api_url: http://legacy.test\n');
    expect(parsed).toEqual({ global: { flux_api_url: 'http://legacy.test' } });
    expect(parsed).not.toHaveProperty('schema_version');
    expect(parsed).not.toHaveProperty('runtime');
  });
});
