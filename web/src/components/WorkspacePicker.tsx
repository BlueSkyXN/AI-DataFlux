import { useCallback, useEffect, useMemo, useState } from 'react';

import { fetchWorkspaceEntries, fetchWorkspaceRoots } from '../api';
import type { WorkspaceEntry, WorkspaceRoot, WorkspaceSelection } from '../types';
import type { Language } from '../i18n';

interface WorkspacePickerProps {
  selection: WorkspaceSelection;
  onSelect: (selection: WorkspaceSelection) => void;
  language: Language;
}

export default function WorkspacePicker({
  selection,
  onSelect,
  language,
}: WorkspacePickerProps) {
  const zh = language === 'zh';
  const [open, setOpen] = useState(false);
  const [roots, setRoots] = useState<WorkspaceRoot[]>([]);
  const [rootId, setRootId] = useState(selection.rootId);
  const [directory, setDirectory] = useState('');
  const [entries, setEntries] = useState<WorkspaceEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    void fetchWorkspaceRoots()
      .then(({ roots: nextRoots }) => {
        if (!active) return;
        setRoots(nextRoots);
        if (nextRoots.length > 0 && !nextRoots.some((root) => root.id === selection.rootId)) {
          onSelect({ rootId: nextRoots[0].id, relativePath: selection.relativePath });
        }
      })
      .catch((reason) => {
        if (active) setError(reason instanceof Error ? reason.message : String(reason));
      });
    return () => {
      active = false;
    };
  }, [onSelect, selection.relativePath, selection.rootId]);

  const openPicker = () => {
    setRootId(selection.rootId);
    const segments = selection.relativePath.split('/');
    segments.pop();
    setDirectory(segments.join('/'));
    setOpen(true);
  };

  const loadEntries = useCallback(async () => {
    if (!open || !rootId) return;
    setLoading(true);
    setError(null);
    try {
      const response = await fetchWorkspaceEntries(rootId, directory);
      setEntries(response.entries);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setLoading(false);
    }
  }, [directory, open, rootId]);

  useEffect(() => {
    const initial = window.setTimeout(loadEntries, 0);
    return () => window.clearTimeout(initial);
  }, [loadEntries]);

  const root = roots.find((item) => item.id === selection.rootId);
  const displayPath = useMemo(
    () => `${root?.path ?? selection.rootId}/${selection.relativePath}`,
    [root?.path, selection.relativePath, selection.rootId],
  );

  const chooseEntry = (entry: WorkspaceEntry) => {
    if (entry.type === 'directory') {
      setDirectory(entry.relative_path);
      return;
    }
    if (!/\.ya?ml$/i.test(entry.name)) return;
    onSelect({ rootId, relativePath: entry.relative_path });
    setOpen(false);
  };

  const parentDirectory = () => {
    const parts = directory.split('/').filter(Boolean);
    parts.pop();
    setDirectory(parts.join('/'));
  };

  return (
    <>
      <div className="flex min-w-0 items-center gap-2 text-xs">
        <span className="shrink-0 text-gray-500">{zh ? '工作区配置' : 'Workspace config'}:</span>
        <span className="truncate font-mono text-gray-700" title={displayPath}>
          {displayPath}
        </span>
        <button
          type="button"
          onClick={openPicker}
          className="shrink-0 rounded bg-gray-100 px-2 py-0.5 font-medium text-gray-600 hover:bg-gray-200"
        >
          {zh ? '浏览' : 'Browse'}
        </button>
      </div>

      {open && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/35 p-4">
          <div
            role="dialog"
            aria-modal="true"
            aria-label={zh ? '选择工作区配置' : 'Select workspace config'}
            className="flex max-h-[80vh] w-full max-w-2xl flex-col overflow-hidden rounded-2xl bg-white shadow-2xl"
          >
            <div className="flex items-center justify-between border-b border-gray-100 px-5 py-4">
              <div>
                <h2 className="font-semibold text-gray-900">
                  {zh ? '选择 YAML 配置' : 'Select YAML configuration'}
                </h2>
                <p className="mt-1 text-xs text-gray-500">
                  {zh ? '只显示后端授权的 workspace roots。' : 'Only server-authorized workspace roots are available.'}
                </p>
              </div>
              <button
                type="button"
                onClick={() => setOpen(false)}
                className="rounded-lg px-3 py-1.5 text-sm text-gray-500 hover:bg-gray-100"
              >
                {zh ? '关闭' : 'Close'}
              </button>
            </div>

            <div className="flex flex-wrap items-center gap-2 border-b border-gray-100 px-5 py-3">
              <label className="text-xs font-medium text-gray-600" htmlFor="workspace-root">
                Root
              </label>
              <select
                id="workspace-root"
                value={rootId}
                onChange={(event) => {
                  setRootId(event.target.value);
                  setDirectory('');
                }}
                className="rounded-lg border border-gray-200 px-3 py-1.5 text-sm"
              >
                {roots.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.id} — {item.path}
                  </option>
                ))}
              </select>
              <button
                type="button"
                onClick={parentDirectory}
                disabled={!directory}
                className="rounded-lg bg-gray-100 px-3 py-1.5 text-sm text-gray-600 disabled:opacity-40"
              >
                ↑ {zh ? '上一级' : 'Parent'}
              </button>
              <span className="min-w-0 flex-1 truncate font-mono text-xs text-gray-500">
                /{directory}
              </span>
            </div>

            <div className="min-h-64 flex-1 overflow-y-auto p-3">
              {loading && <p className="p-4 text-sm text-gray-500">{zh ? '加载中…' : 'Loading…'}</p>}
              {error && <p className="m-2 rounded-lg bg-red-50 p-3 text-sm text-red-600">{error}</p>}
              {!loading && !error && entries.length === 0 && (
                <p className="p-4 text-sm text-gray-500">{zh ? '此目录为空。' : 'This directory is empty.'}</p>
              )}
              <div className="space-y-1">
                {entries
                  .filter((entry) => entry.type === 'directory' || /\.ya?ml$/i.test(entry.name))
                  .map((entry) => (
                    <button
                      type="button"
                      key={`${entry.type}:${entry.relative_path}`}
                      onClick={() => chooseEntry(entry)}
                      className="flex w-full items-center gap-3 rounded-xl px-3 py-2 text-left hover:bg-cyan-50"
                    >
                      <span aria-hidden="true" className="text-lg">
                        {entry.type === 'directory' ? '▸' : '◇'}
                      </span>
                      <span className="min-w-0 flex-1 truncate text-sm text-gray-800">{entry.name}</span>
                      <span className="text-xs text-gray-400">{entry.type}</span>
                    </button>
                  ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
