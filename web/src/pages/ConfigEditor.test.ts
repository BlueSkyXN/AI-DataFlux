import { describe, expect, it } from 'vitest';

import { migrateConfigShape } from './configMigration';

describe('migrateConfigShape', () => {
  it('upgrades endpoints, model capabilities, and cooperative concurrency', () => {
    expect(
      migrateConfigShape({
        channels: { main: { base_url: 'https://example.test', api_path: '/chat' } },
        models: [{ id: 'm1', supports_json_schema: true, supports_advanced_params: true }],
        datasource: { concurrency: { batch_size: 12 } },
      }),
    ).toMatchObject({
      channels: { main: { endpoints: { chat_completions: '/chat' } } },
      models: [{ capabilities: ['chat_completions', 'stream', 'json_schema'] }],
      datasource: { concurrency: { batch_size: 12, max_in_flight: 12 } },
    });
  });
});
