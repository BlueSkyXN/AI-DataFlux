import { useCallback, useEffect, useMemo, useState } from 'react';

import {
  cancelJob,
  fetchJobEvents,
  fetchJobs,
  resumeJob,
  streamJobEvents,
  submitJob,
} from '../api';
import type { Language } from '../i18n';
import type {
  JobEvent,
  JobListResponse,
  JobState,
  WorkspaceSelection,
} from '../types';

interface JobsProps {
  selection: WorkspaceSelection;
  language: Language;
}

const CANCELLABLE = new Set(['queued', 'running']);
const RESUMABLE = new Set(['blocked', 'interrupted', 'failed']);

export default function Jobs({ selection, language }: JobsProps) {
  const zh = language === 'zh';
  const [data, setData] = useState<JobListResponse | null>(null);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [events, setEvents] = useState<JobEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [busyJobId, setBusyJobId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [eventError, setEventError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const next = await fetchJobs();
      setData(next);
      setSelectedJobId((current) => current ?? next.jobs[0]?.job_id ?? null);
      setError(null);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const initial = window.setTimeout(refresh, 0);
    const timer = window.setInterval(refresh, 3000);
    return () => {
      window.clearTimeout(initial);
      window.clearInterval(timer);
    };
  }, [refresh]);

  useEffect(() => {
    if (!selectedJobId) {
      return;
    }
    let active = true;
    let closeStream: (() => void) | undefined;
    const initial = window.setTimeout(() => {
      setEventError(null);
      void fetchJobEvents(selectedJobId)
        .then((response) => {
          if (!active) return;
          setEvents(response.events);
          closeStream = streamJobEvents(selectedJobId, {
            afterSeq: response.next_seq,
            onEvent: (event) => {
              setEvents((current) =>
                current.some((item) => item.seq === event.seq) ? current : [...current, event],
              );
            },
            onError: (reason) => setEventError(reason.message),
          });
        })
        .catch((reason) => {
          if (active) setEventError(reason instanceof Error ? reason.message : String(reason));
        });
    }, 0);
    return () => {
      active = false;
      window.clearTimeout(initial);
      closeStream?.();
    };
  }, [selectedJobId]);

  const selected = useMemo(
    () => data?.jobs.find((job) => job.job_id === selectedJobId) ?? null,
    [data?.jobs, selectedJobId],
  );

  const replaceJob = (updated: JobState) => {
    setData((current) =>
      current
        ? {
            ...current,
            jobs: current.jobs.some((job) => job.job_id === updated.job_id)
              ? current.jobs.map((job) => (job.job_id === updated.job_id ? updated : job))
              : [updated, ...current.jobs],
          }
        : current,
    );
  };

  const handleSubmit = async () => {
    setSubmitting(true);
    setError(null);
    try {
      const job = await submitJob(selection);
      replaceJob(job);
      setSelectedJobId(job.job_id);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setSubmitting(false);
    }
  };

  const handleAction = async (job: JobState, action: 'cancel' | 'resume') => {
    setBusyJobId(job.job_id);
    setError(null);
    try {
      replaceJob(action === 'cancel' ? await cancelJob(job.job_id) : await resumeJob(job.job_id));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusyJobId(null);
    }
  };

  return (
    <div className="grid min-h-[calc(100vh-190px)] grid-cols-1 gap-5 p-4 sm:p-6 lg:grid-cols-[minmax(0,1.45fr)_minmax(320px,0.8fr)]">
      <section className="min-w-0 space-y-4">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-600">Durable jobs</p>
            <h2 className="mt-1 text-2xl font-semibold text-gray-900">{zh ? '任务队列' : 'Job queue'}</h2>
            <p className="mt-1 text-sm text-gray-500">
              {selection.rootId}/{selection.relativePath}
            </p>
          </div>
          <button
            type="button"
            onClick={handleSubmit}
            disabled={submitting || !selection.relativePath}
            className="rounded-xl bg-gradient-to-r from-cyan-500 to-blue-500 px-4 py-2.5 text-sm font-semibold text-white shadow-sm hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {submitting ? (zh ? '提交中…' : 'Submitting…') : (zh ? '提交当前配置' : 'Submit current config')}
          </button>
        </div>

        {error && <div role="alert" className="rounded-xl bg-red-50 px-4 py-3 text-sm text-red-600">{error}</div>}

        <div className="flex flex-wrap items-center gap-3 rounded-2xl bg-white px-4 py-3 text-xs text-gray-500 shadow-[0_2px_12px_rgba(0,0,0,0.04)]">
          <span className="font-semibold text-gray-700">{zh ? '资源控制' : 'Resource control'}</span>
          <span className={`rounded-full px-2 py-1 ${data?.resource.pressure ? 'bg-amber-50 text-amber-700' : 'bg-emerald-50 text-emerald-700'}`}>
            {data?.resource.status ?? '-'} · {data?.resource.pressure ? (zh ? '有压力' : 'pressure') : (zh ? '正常' : 'normal')}
          </span>
          {data?.resource.cpu_percent != null && <span>CPU {data.resource.cpu_percent.toFixed(1)}%</span>}
          {data?.resource.memory_percent != null && <span>MEM {data.resource.memory_percent.toFixed(1)}%</span>}
        </div>

        <div className="space-y-3">
          {loading && <div className="rounded-2xl bg-white p-8 text-center text-sm text-gray-500">{zh ? '加载中…' : 'Loading…'}</div>}
          {!loading && (data?.jobs.length ?? 0) === 0 && (
            <div className="rounded-2xl border border-dashed border-gray-200 bg-white p-10 text-center">
              <p className="font-medium text-gray-700">{zh ? '队列为空' : 'The queue is empty'}</p>
              <p className="mt-1 text-sm text-gray-400">{zh ? '选择一个 YAML 配置并提交首个任务。' : 'Select a YAML config and submit the first job.'}</p>
            </div>
          )}
          {(data?.jobs ?? []).map((job) => (
            <JobCard
              key={job.job_id}
              job={job}
              selected={job.job_id === selectedJobId}
              busy={job.job_id === busyJobId}
              zh={zh}
              onSelect={() => setSelectedJobId(job.job_id)}
              onCancel={() => void handleAction(job, 'cancel')}
              onResume={() => void handleAction(job, 'resume')}
            />
          ))}
        </div>
      </section>

      <aside className="min-w-0 rounded-2xl bg-slate-950 p-4 text-slate-200 shadow-xl lg:sticky lg:top-5 lg:max-h-[calc(100vh-220px)]">
        <div className="flex items-center justify-between gap-2">
          <div>
            <p className="text-xs uppercase tracking-[0.18em] text-cyan-300">Event stream</p>
            <h3 className="mt-1 truncate font-mono text-sm text-white">
              {selected ? selected.job_id : (zh ? '选择一个任务' : 'Select a job')}
            </h3>
          </div>
          {selected && <StatusBadge status={selected.status} dark />}
        </div>
        {eventError && <p className="mt-3 rounded-lg bg-red-950/70 p-2 text-xs text-red-200">{eventError}</p>}
        <div className="mt-4 max-h-[calc(100vh-310px)] space-y-2 overflow-y-auto pr-1">
          {events.map((event) => (
            <div key={event.seq} className="rounded-xl border border-slate-800 bg-slate-900/80 p-3">
              <div className="flex items-center justify-between gap-2 text-xs">
                <span className="font-semibold text-cyan-300">#{event.seq} {event.type}</span>
                <time className="text-slate-500">{formatTime(event.ts)}</time>
              </div>
              {(event.record_id != null || event.attempt != null) && (
                <p className="mt-1 text-xs text-slate-400">
                  {event.record_id != null ? `record=${String(event.record_id)} ` : ''}
                  {event.attempt != null ? `attempt=${event.attempt}` : ''}
                </p>
              )}
              {Object.keys(event.payload).length > 0 && (
                <pre className="mt-2 overflow-x-auto whitespace-pre-wrap break-words text-[11px] leading-5 text-slate-400">
                  {JSON.stringify(event.payload, null, 2)}
                </pre>
              )}
            </div>
          ))}
          {selected && events.length === 0 && <p className="py-10 text-center text-sm text-slate-500">{zh ? '等待事件…' : 'Waiting for events…'}</p>}
        </div>
      </aside>
    </div>
  );
}

interface JobCardProps {
  job: JobState;
  selected: boolean;
  busy: boolean;
  zh: boolean;
  onSelect: () => void;
  onCancel: () => void;
  onResume: () => void;
}

function JobCard({ job, selected, busy, zh, onSelect, onCancel, onResume }: JobCardProps) {
  const total = Math.max(job.counts.discovered, 0);
  const progress = total > 0 ? Math.min(100, (job.counts.persisted / total) * 100) : 0;
  return (
    <article className={`rounded-2xl bg-white p-5 shadow-[0_2px_12px_rgba(0,0,0,0.04)] ring-1 transition ${selected ? 'ring-cyan-300' : 'ring-transparent hover:ring-gray-200'}`}>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <button type="button" onClick={onSelect} className="min-w-0 flex-1 text-left">
          <div className="flex flex-wrap items-center gap-2">
            <StatusBadge status={job.status} />
            <span className="font-mono text-xs text-gray-500">{job.job_id}</span>
          </div>
          <p className="mt-2 truncate text-sm font-medium text-gray-800" title={job.config_path}>{job.config_path}</p>
          <p className="mt-1 text-xs text-gray-400">{formatTime(job.created_at)} · rev {job.revision}</p>
        </button>
        <div className="flex gap-2">
          {CANCELLABLE.has(job.status) && (
            <button type="button" onClick={onCancel} disabled={busy} className="rounded-lg bg-rose-50 px-3 py-1.5 text-xs font-medium text-rose-600 hover:bg-rose-100 disabled:opacity-50">
              {zh ? '取消' : 'Cancel'}
            </button>
          )}
          {RESUMABLE.has(job.status) && (
            <button type="button" onClick={onResume} disabled={busy} className="rounded-lg bg-cyan-50 px-3 py-1.5 text-xs font-medium text-cyan-700 hover:bg-cyan-100 disabled:opacity-50">
              {zh ? '恢复' : 'Resume'}
            </button>
          )}
        </div>
      </div>

      <div className="mt-4 h-2 overflow-hidden rounded-full bg-gray-100">
        <div className="h-full rounded-full bg-gradient-to-r from-cyan-400 to-blue-500 transition-all" style={{ width: `${progress}%` }} />
      </div>
      <div className="mt-3 grid grid-cols-2 gap-3 text-xs sm:grid-cols-5">
        <Metric label={zh ? '发现' : 'Discovered'} value={job.counts.discovered} />
        <Metric label="AI complete" value={job.counts.ai_complete} />
        <Metric label={zh ? '持久化' : 'Persisted'} value={job.counts.persisted} />
        <Metric label={zh ? '失败' : 'Failed'} value={job.counts.failed} danger />
        <Metric label="max_in_flight" value={job.resource.effective_max_in_flight} />
      </div>
      <div className="mt-3 flex flex-wrap gap-3 text-xs text-gray-400">
        <span>in_flight={job.counts.in_flight}</span>
        <span>pending={job.counts.pending}</span>
        <span>retries={job.counts.retries}</span>
        <span className={job.resource.pressure ? 'text-amber-600' : ''}>{job.resource.control_status}</span>
      </div>
      {job.last_error && <p className="mt-3 rounded-lg bg-red-50 px-3 py-2 text-xs text-red-600">{job.last_error}</p>}
    </article>
  );
}

function Metric({ label, value, danger = false }: { label: string; value: number; danger?: boolean }) {
  return (
    <div>
      <p className="text-gray-400">{label}</p>
      <p className={`mt-1 font-semibold ${danger && value > 0 ? 'text-rose-600' : 'text-gray-800'}`}>{value.toLocaleString()}</p>
    </div>
  );
}

function StatusBadge({ status, dark = false }: { status: string; dark?: boolean }) {
  const active = ['queued', 'running', 'cancelling'].includes(status);
  const attention = ['failed', 'blocked', 'interrupted'].includes(status);
  const colors = dark
    ? 'bg-slate-800 text-slate-200'
    : active
      ? 'bg-cyan-50 text-cyan-700'
      : attention
        ? 'bg-amber-50 text-amber-700'
        : 'bg-emerald-50 text-emerald-700';
  return <span className={`inline-flex rounded-full px-2.5 py-1 text-xs font-medium ${colors}`}>{status}</span>;
}

function formatTime(timestamp: number): string {
  return new Intl.DateTimeFormat(undefined, {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  }).format(new Date(timestamp * 1000));
}
