import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import * as api from '../api';
import WorkspacePicker from './WorkspacePicker';

describe('WorkspacePicker', () => {
  it('navigates server entries and returns a root-relative YAML selection', async () => {
    vi.spyOn(api, 'fetchWorkspaceRoots').mockResolvedValue({
      roots: [{ id: 'project', path: '/srv/project' }],
    });
    vi.spyOn(api, 'fetchWorkspaceEntries')
      .mockResolvedValueOnce({
        root_id: 'project',
        relative_path: '',
        entries: [
          { name: 'configs', relative_path: 'configs', type: 'directory' },
          { name: 'notes.txt', relative_path: 'notes.txt', type: 'file' },
        ],
      })
      .mockResolvedValueOnce({
        root_id: 'project',
        relative_path: 'configs',
        entries: [
          { name: 'production.yaml', relative_path: 'configs/production.yaml', type: 'file' },
        ],
      });
    const onSelect = vi.fn();
    const user = userEvent.setup();

    render(
      <WorkspacePicker
        selection={{ rootId: 'project', relativePath: 'config.yaml' }}
        onSelect={onSelect}
        language="en"
      />,
    );
    await user.click(screen.getByRole('button', { name: 'Browse' }));
    await user.click(await screen.findByRole('button', { name: /configs/ }));
    await user.click(await screen.findByRole('button', { name: /production.yaml/ }));

    await waitFor(() =>
      expect(onSelect).toHaveBeenCalledWith({
        rootId: 'project',
        relativePath: 'configs/production.yaml',
      }),
    );
    expect(screen.queryByText('notes.txt')).not.toBeInTheDocument();
  });
});
