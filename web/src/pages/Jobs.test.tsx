import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import * as api from '../api';
import type { JobState } from '../types';
import Jobs from './Jobs';

function job(overrides: Partial<JobState> = {}): JobState {
  return {
    schema_version: 1,
    revision: 1,
    job_id: '12345678-1234-4234-8234-123456789abc',
    mode: 'background',
    config_path: '/srv/project/config.yaml',
    config_sha256: 'a'.repeat(64),
    status: 'failed',
    created_at: 1_700_000_000,
    updated_at: 1_700_000_001,
    started_at: 1_700_000_000,
    finished_at: 1_700_000_001,
    counts: {
      discovered: 12,
      pending: 0,
      in_flight: 0,
      ai_complete: 11,
      persisted: 10,
      unresolved_writes: 0,
      failed: 2,
      cancelled: 0,
      retries: 1,
    },
    resource: {
      effective_max_in_flight: 3,
      pressure: false,
      control_status: 'normal',
    },
    last_error: 'writeback failed',
    ...overrides,
  };
}

describe('Jobs page', () => {
  it('renders persisted/failed/resource state and resumes a failed job', async () => {
    const failed = job();
    vi.spyOn(api, 'fetchJobs').mockResolvedValue({
      jobs: [failed],
      resource: { status: 'normal', pressure: false, cpu_percent: 20, memory_percent: 30 },
    });
    vi.spyOn(api, 'fetchJobEvents').mockResolvedValue({ events: [], next_seq: 0 });
    vi.spyOn(api, 'streamJobEvents').mockReturnValue(() => {});
    const resume = vi.spyOn(api, 'resumeJob').mockResolvedValue(
      job({ status: 'queued', last_error: null, finished_at: null, revision: 2 }),
    );
    const user = userEvent.setup();

    render(
      <Jobs
        selection={{ rootId: 'project', relativePath: 'config.yaml' }}
        language="en"
      />,
    );

    expect(await screen.findByText('writeback failed')).toBeInTheDocument();
    expect(screen.getByText('Persisted')).toBeInTheDocument();
    expect(screen.getByText('max_in_flight')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Resume' }));
    await waitFor(() => expect(resume).toHaveBeenCalledWith(failed.job_id));
    expect((await screen.findAllByText('queued')).length).toBeGreaterThan(0);
  });
});
