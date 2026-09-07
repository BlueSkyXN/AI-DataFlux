import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import PromptSection from './PromptSection';

describe('canonical model selection', () => {
  it.each([
    ['auto', { mode: 'auto' }],
    ['strict', { mode: 'strict', route_id: '' }],
    ['fallback_group', { mode: 'fallback_group', group: '' }],
  ])('switching to %s removes incompatible selection keys', (mode, expected) => {
    const updateConfig = vi.fn();
    render(<PromptSection formData={{}} language="en" updateConfig={updateConfig}
      getConfig={(path) => path.join('.') === 'job.model_selection'
        ? { mode: 'strict', route_id: 'old-route' } : undefined} />);
    fireEvent.change(screen.getByRole('combobox', { name: 'Routing mode' }), { target: { value: mode } });
    expect(updateConfig).toHaveBeenCalledWith(['job', 'model_selection'], expected);
  });
});
