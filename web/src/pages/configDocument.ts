import yaml from 'js-yaml';

export function parseYaml(content: string): Record<string, unknown> {
  const result = yaml.load(content);
  if (typeof result !== 'object' || result === null) {
    return {};
  }
  return result as Record<string, unknown>;
}

export function serializeYaml(data: Record<string, unknown>): string {
  return yaml.dump(data, {
    indent: 2,
    lineWidth: 120,
    noRefs: true,
    sortKeys: false,
    quotingType: '"',
    forceQuotes: false,
  });
}

export function updateConfigValue(
  data: Record<string, unknown>,
  path: string[],
  value: unknown,
): Record<string, unknown> {
  const next = structuredClone(data);
  let target = next;
  for (let i = 0; i < path.length - 1; i++) {
    if (target[path[i]] === undefined || target[path[i]] === null || typeof target[path[i]] !== 'object') {
      target[path[i]] = {};
    }
    target = target[path[i]] as Record<string, unknown>;
  }
  target[path[path.length - 1]] = value;
  return next;
}
