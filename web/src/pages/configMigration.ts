/** Upgrade removed 3.1 fields to the canonical 3.2 config shape. */
export function migrateConfigShape(
  input: Record<string, unknown>,
): Record<string, unknown> {
  const next = structuredClone(input);
  const channels = next.channels;
  if (channels && typeof channels === 'object' && !Array.isArray(channels)) {
    for (const channel of Object.values(channels as Record<string, unknown>)) {
      if (!channel || typeof channel !== 'object' || Array.isArray(channel)) continue;
      const record = channel as Record<string, unknown>;
      const endpoints =
        record.endpoints && typeof record.endpoints === 'object' && !Array.isArray(record.endpoints)
          ? { ...(record.endpoints as Record<string, unknown>) }
          : {};
      if (!endpoints.chat_completions && typeof record.api_path === 'string') {
        endpoints.chat_completions = record.api_path;
      }
      record.endpoints = endpoints;
      delete record.api_path;
    }
  }
  if (Array.isArray(next.models)) {
    next.models = next.models.map((model) => {
      if (!model || typeof model !== 'object' || Array.isArray(model)) return model;
      const record = { ...(model as Record<string, unknown>) };
      if (!Array.isArray(record.capabilities) || record.capabilities.length === 0) {
        record.capabilities = [
          'chat_completions',
          'stream',
          ...(record.supports_json_schema ? ['json_schema'] : []),
        ];
      }
      delete record.supports_json_schema;
      delete record.supports_advanced_params;
      return record;
    });
  }
  const datasource = next.datasource;
  if (datasource && typeof datasource === 'object' && !Array.isArray(datasource)) {
    const concurrency = (datasource as Record<string, unknown>).concurrency;
    if (concurrency && typeof concurrency === 'object' && !Array.isArray(concurrency)) {
      const record = concurrency as Record<string, unknown>;
      if (record.max_in_flight === undefined && typeof record.batch_size === 'number') {
        record.max_in_flight = record.batch_size;
      }
    }
  }
  return next;
}
