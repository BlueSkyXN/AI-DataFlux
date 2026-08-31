import { beforeEach, describe, expect, it, vi } from 'vitest';

import { fetchConfig, saveConfig, submitJob } from './api';

describe('Control API v1 client', () => {
  beforeEach(() => {
    window.history.replaceState({}, '', '/#token=test-control-token');
  });

  it('reads config from workspace coordinates and preserves the ETag', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(
        JSON.stringify({
          root_id: 'project',
          relative_path: 'configs/main.yaml',
          content: 'global: {}',
          revision: 4,
        }),
        { status: 200, headers: { ETag: '"rev-4"', 'Content-Type': 'application/json' } },
      ),
    );

    const result = await fetchConfig({ rootId: 'project', relativePath: 'configs/main.yaml' });

    expect(result.etag).toBe('"rev-4"');
    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toBe('/api/v1/config?root_id=project&relative_path=configs%2Fmain.yaml');
    expect(new Headers(init?.headers).get('Authorization')).toBe('Bearer test-control-token');
  });

  it('uses If-Match for config writes and workspace coordinates for job submission', async () => {
    const fetchMock = vi
      .spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            root_id: 'project',
            relative_path: 'config.yaml',
            content: 'global: {}',
            revision: 2,
          }),
          { status: 200, headers: { ETag: '"rev-2"', 'Content-Type': 'application/json' } },
        ),
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ job_id: 'job-1' }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }),
      );

    await saveConfig(
      { rootId: 'project', relativePath: 'config.yaml' },
      'global: {}',
      '"rev-1"',
    );
    await submitJob({ rootId: 'project', relativePath: 'config.yaml' });

    const saveRequest = fetchMock.mock.calls[0][1];
    expect(saveRequest?.method).toBe('PUT');
    expect(new Headers(saveRequest?.headers).get('If-Match')).toBe('"rev-1"');
    expect(JSON.parse(String(saveRequest?.body))).toEqual({
      root_id: 'project',
      relative_path: 'config.yaml',
      content: 'global: {}',
    });
    expect(fetchMock.mock.calls[1][0]).toBe('/api/v1/jobs');
    expect(JSON.parse(String(fetchMock.mock.calls[1][1]?.body))).toEqual({
      root_id: 'project',
      relative_path: 'config.yaml',
    });
  });
});
