import '@testing-library/jest-dom/vitest';
import { afterEach, vi } from 'vitest';

afterEach(() => {
  window.sessionStorage?.clear();
  window.localStorage?.clear();
  window.history.replaceState({}, '', '/');
  vi.restoreAllMocks();
});
