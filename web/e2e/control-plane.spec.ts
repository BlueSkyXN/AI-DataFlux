import { expect, test } from '@playwright/test';
import type { Page, Route } from '@playwright/test';

type JobStatus =
  | 'queued'
  | 'running'
  | 'cancelling'
  | 'interrupted'
  | 'blocked'
  | 'completed'
  | 'completed_with_errors'
  | 'failed'
  | 'cancelled';

const jobId = '12345678-1234-4234-8234-123456789abc';

function job(status: JobStatus, revision = 1) {
  const terminal = ['completed', 'completed_with_errors', 'failed', 'cancelled'].includes(status);
  return {
    schema_version: 1,
    revision,
    job_id: jobId,
    mode: 'background',
    config_path: '/srv/project/config.yaml',
    config_sha256: 'a'.repeat(64),
    status,
    created_at: 1_700_000_000,
    updated_at: 1_700_000_001 + revision,
    started_at: status === 'queued' ? null : 1_700_000_000,
    finished_at: terminal ? 1_700_000_001 : null,
    counts: {
      discovered: status === 'queued' ? 0 : 10,
      pending: status === 'running' ? 3 : 0,
      in_flight: status === 'running' ? 2 : 0,
      ai_complete: status === 'completed' ? 10 : status === 'running' ? 5 : 0,
      persisted: status === 'completed' ? 10 : status === 'running' ? 3 : 0,
      failed: status === 'failed' ? 1 : 0,
      cancelled: status === 'cancelled' ? 1 : 0,
      retries: 0,
    },
    resource: {
      effective_max_in_flight: 4,
      pressure: false,
      control_status: 'normal',
    },
    last_error: status === 'failed' ? 'writeback failed' : null,
  };
}

const resource = {
  sampled_at: 1_700_000_000,
  status: 'normal',
  pressure: false,
  cpu_percent: 20,
  memory_percent: 30,
  available_memory_bytes: 2_000_000_000,
  reasons: [],
};

async function json(route: Route, body: unknown, status = 200, headers?: Record<string, string>) {
  await route.fulfill({
    status,
    contentType: 'application/json',
    headers,
    body: JSON.stringify(body),
  });
}

async function mockShell(page: Page) {
  await page.route('**/health', (route) =>
    json(route, { status: 'ok', version: '4.0.0-e2e' }),
  );
  await page.route('**/api/v1/workspace/roots', (route) =>
    json(route, { roots: [{ id: 'project', path: '/srv/project' }] }),
  );
  await page.route('**/api/v1/workspace/entries?*', (route) => {
    const url = new URL(route.request().url());
    const relativePath = url.searchParams.get('relative_path') ?? '';
    if (relativePath === 'configs') {
      return json(route, {
        root_id: 'project',
        relative_path: 'configs',
        entries: [
          {
            name: 'production.yaml',
            relative_path: 'configs/production.yaml',
            type: 'file',
          },
        ],
      });
    }
    return json(route, {
      root_id: 'project',
      relative_path: '.',
      entries: [
        { name: 'configs', relative_path: 'configs', type: 'directory' },
        { name: 'config.yaml', relative_path: 'config.yaml', type: 'file' },
      ],
    });
  });
}

async function mockEventRoutes(page: Page) {
  await page.route('**/api/v1/jobs/*/events/stream?*', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      body: `id: 1\ndata: ${JSON.stringify({
        seq: 1,
        ts: 1_700_000_001,
        type: 'status_changed',
        job_id: jobId,
        record_id: null,
        attempt: null,
        payload: { source: 'e2e' },
      })}\n\n`,
    }),
  );
  await page.route('**/api/v1/jobs/*/events?*', (route) =>
    json(route, { events: [], next_seq: 0 }),
  );
}

async function openApp(page: Page) {
  await page.goto('/#token=e2e-token');
  await expect(page.getByRole('banner').getByText('4.0.0-e2e')).toBeVisible();
}

test.beforeEach(async ({ page }) => {
  await mockShell(page);
});

test('submit transitions through running to completed', async ({ page }) => {
  let submitted = false;
  let listAfterSubmit = 0;
  await mockEventRoutes(page);
  await page.route('**/api/v1/jobs', async (route) => {
    if (route.request().method() === 'POST') {
      submitted = true;
      return json(route, job('running', 2), 201);
    }
    if (!submitted) return json(route, { jobs: [], resource });
    listAfterSubmit += 1;
    return json(route, {
      jobs: [listAfterSubmit === 1 ? job('running', 2) : job('completed', 3)],
      resource,
    });
  });

  await openApp(page);
  await page.getByRole('button', { name: 'Jobs' }).click();
  await page.getByRole('button', { name: 'Submit current config' }).click();
  await expect(page.getByText('running').first()).toBeVisible();
  await expect(page.getByText('completed').first()).toBeVisible({ timeout: 7_000 });
  await expect(page.getByText('10', { exact: true }).first()).toBeVisible();
});

test('cancel queued job reaches cancelled', async ({ page }) => {
  await mockEventRoutes(page);
  await page.route('**/api/v1/jobs/*/cancel', (route) => json(route, job('cancelled', 2)));
  await page.route('**/api/v1/jobs', (route) =>
    json(route, { jobs: [job('queued')], resource }),
  );

  await openApp(page);
  await page.getByRole('button', { name: 'Jobs' }).click();
  await page.getByRole('button', { name: 'Cancel' }).click();
  await expect(page.getByText('cancelled').first()).toBeVisible();
});

test('resume failed job returns it to queued', async ({ page }) => {
  await mockEventRoutes(page);
  await page.route('**/api/v1/jobs/*/resume', (route) => json(route, job('queued', 2)));
  await page.route('**/api/v1/jobs', (route) =>
    json(route, { jobs: [job('failed')], resource }),
  );

  await openApp(page);
  await page.getByRole('button', { name: 'Jobs' }).click();
  await expect(page.getByText('writeback failed')).toBeVisible();
  await page.getByRole('button', { name: 'Resume' }).click();
  await expect(page.getByText('queued').first()).toBeVisible();
});

test('config save sends If-Match and surfaces ETag conflict', async ({ page }) => {
  let ifMatch = '';
  await page.route('**/api/v1/jobs', (route) => json(route, { jobs: [], resource }));
  await page.route(/\/api\/v1\/config\?.*/, (route) =>
    json(
      route,
      {
        root_id: 'project',
        relative_path: 'config.yaml',
        content: 'global:\n  flux_api_url: http://gateway\n',
        revision: 4,
      },
      200,
      { ETag: '"rev-4"' },
    ),
  );
  await page.route('**/api/v1/config', async (route) => {
    ifMatch = route.request().headers()['if-match'] ?? '';
    return json(
      route,
      {
        error: {
          code: 'revision_conflict',
          message: 'Configuration changed on disk',
          details: null,
          request_id: 'req-conflict',
        },
      },
      409,
    );
  });

  await openApp(page);
  await page.getByRole('button', { name: 'Config' }).click();
  await expect(page.getByText('rev 4')).toBeVisible();
  await page.getByRole('button', { name: 'Raw YAML' }).click();
  await page.locator('textarea').fill('global:\n  flux_api_url: http://new-gateway\n');
  await page.getByRole('button', { name: 'Save' }).click();

  await expect(page.getByText(/Configuration changed on disk/)).toBeVisible();
  await expect.poll(() => ifMatch).toBe('"rev-4"');
});

test('Browse navigates server workspace entries and selects YAML', async ({ page }) => {
  await page.route('**/api/v1/jobs', (route) => json(route, { jobs: [], resource }));
  await openApp(page);

  await page.getByRole('button', { name: 'Browse' }).click();
  await page.getByRole('button', { name: /configs/ }).click();
  await page.getByRole('button', { name: /production.yaml/ }).click();

  await expect(page.getByTitle('/srv/project/configs/production.yaml')).toBeVisible();
});
