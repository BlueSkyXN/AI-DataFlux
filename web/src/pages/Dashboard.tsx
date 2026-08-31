import { useCallback, useEffect, useMemo, useState } from 'react';

import { fetchHealth, fetchJobs } from '../api';
import type { Language } from '../i18n';
import type { HealthResponse, JobListResponse, WorkspaceSelection } from '../types';

interface DashboardProps {
  selection: WorkspaceSelection;
  language: Language;
}

export default function Dashboard({ selection, language }: DashboardProps) {
  const zh = language === 'zh';
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [data, setData] = useState<JobListResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [nextHealth, nextJobs] = await Promise.all([fetchHealth(), fetchJobs()]);
      setHealth(nextHealth);
      setData(nextJobs);
      setError(null);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }, []);

  useEffect(() => {
    const initial = window.setTimeout(refresh, 0);
    const timer = window.setInterval(refresh, 5000);
    return () => {
      window.clearTimeout(initial);
      window.clearInterval(timer);
    };
  }, [refresh]);

  const summary = useMemo(() => {
    const jobs = data?.jobs ?? [];
    return {
      active: jobs.filter((job) => ['queued', 'running', 'cancelling'].includes(job.status)).length,
      complete: jobs.filter((job) => ['completed', 'completed_with_errors'].includes(job.status)).length,
      persisted: jobs.reduce((total, job) => total + job.counts.persisted, 0),
      failed: jobs.reduce((total, job) => total + job.counts.failed, 0),
    };
  }, [data?.jobs]);

  const cards = [
    {
      label: zh ? '控制服务' : 'Controller',
      value: health?.status ?? (zh ? '不可用' : 'Unavailable'),
      hint: health?.version ?? '-',
      accent: 'from-cyan-400 to-blue-500',
    },
    {
      label: zh ? '活动任务' : 'Active jobs',
      value: String(summary.active),
      hint: `${summary.complete} ${zh ? '已完成' : 'completed'}`,
      accent: 'from-violet-400 to-indigo-500',
    },
    {
      label: zh ? '已持久化' : 'Persisted',
      value: summary.persisted.toLocaleString(),
      hint: `${summary.failed} ${zh ? '失败' : 'failed'}`,
      accent: 'from-emerald-400 to-teal-500',
    },
    {
      label: zh ? '资源控制' : 'Resource control',
      value: data?.resource.status ?? '-',
      hint: data?.resource.pressure ? (zh ? '资源压力中' : 'Under pressure') : (zh ? '准入正常' : 'Admission open'),
      accent: data?.resource.pressure ? 'from-amber-400 to-orange-500' : 'from-slate-400 to-slate-500',
    },
  ];

  return (
    <div className="space-y-6 p-4 sm:p-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-600">3.2 Control Plane</p>
          <h2 className="mt-1 text-2xl font-semibold text-gray-900">{zh ? '运行概览' : 'Operations overview'}</h2>
          <p className="mt-1 text-sm text-gray-500">
            {zh ? '状态来源于持久化 Job Repository，不再依赖浏览器本地目录。' : 'Status comes from the durable Job Repository, not browser-local directory state.'}
          </p>
        </div>
        <div className="rounded-xl border border-gray-200 bg-white px-3 py-2 text-xs text-gray-500">
          <span className="font-medium text-gray-700">{selection.rootId}</span> / {selection.relativePath}
        </div>
      </div>

      {error && <div className="rounded-xl bg-red-50 px-4 py-3 text-sm text-red-600">{error}</div>}

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {cards.map((card) => (
          <section key={card.label} className="overflow-hidden rounded-2xl bg-white shadow-[0_2px_12px_rgba(0,0,0,0.04)]">
            <div className={`h-1 bg-gradient-to-r ${card.accent}`} />
            <div className="p-5">
              <p className="text-sm text-gray-500">{card.label}</p>
              <p className="mt-2 truncate text-2xl font-semibold capitalize text-gray-900">{card.value}</p>
              <p className="mt-1 text-xs text-gray-400">{card.hint}</p>
            </div>
          </section>
        ))}
      </div>

      <section className="rounded-2xl bg-white p-5 shadow-[0_2px_12px_rgba(0,0,0,0.04)]">
        <div className="flex items-center justify-between">
          <h3 className="font-semibold text-gray-900">{zh ? '最近任务' : 'Recent jobs'}</h3>
          <button type="button" onClick={() => void refresh()} className="rounded-lg bg-gray-100 px-3 py-1.5 text-xs text-gray-600 hover:bg-gray-200">
            {zh ? '刷新' : 'Refresh'}
          </button>
        </div>
        <div className="mt-4 overflow-x-auto">
          <table className="w-full min-w-[680px] text-left text-sm">
            <thead className="text-xs uppercase tracking-wide text-gray-400">
              <tr>
                <th className="pb-3 font-medium">Job</th>
                <th className="pb-3 font-medium">{zh ? '状态' : 'Status'}</th>
                <th className="pb-3 font-medium">{zh ? '持久化' : 'Persisted'}</th>
                <th className="pb-3 font-medium">{zh ? '失败' : 'Failed'}</th>
                <th className="pb-3 font-medium">max_in_flight</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {(data?.jobs ?? []).slice(0, 5).map((job) => (
                <tr key={job.job_id}>
                  <td className="py-3 font-mono text-xs text-gray-700">{job.job_id.slice(0, 12)}…</td>
                  <td className="py-3"><StatusPill status={job.status} /></td>
                  <td className="py-3 text-gray-700">{job.counts.persisted}</td>
                  <td className="py-3 text-gray-700">{job.counts.failed}</td>
                  <td className="py-3 text-gray-700">{job.resource.effective_max_in_flight}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {(data?.jobs.length ?? 0) === 0 && (
            <p className="py-10 text-center text-sm text-gray-400">{zh ? '尚未提交任务。' : 'No jobs submitted yet.'}</p>
          )}
        </div>
      </section>
    </div>
  );
}

function StatusPill({ status }: { status: string }) {
  const active = ['queued', 'running', 'cancelling'].includes(status);
  const failed = ['failed', 'blocked', 'interrupted'].includes(status);
  return (
    <span className={`inline-flex rounded-full px-2.5 py-1 text-xs font-medium ${
      active ? 'bg-cyan-50 text-cyan-700' : failed ? 'bg-amber-50 text-amber-700' : 'bg-emerald-50 text-emerald-700'
    }`}>
      {status}
    </span>
  );
}
