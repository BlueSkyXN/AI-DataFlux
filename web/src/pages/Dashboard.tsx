import { useEffect, useState, useCallback } from 'react';
import type { StatusResponse } from '../types';
import { fetchStatus, startGateway, stopGateway, startProcess, stopProcess } from '../api';
import { getTranslations, interpolate, type Language } from '../i18n';

interface DashboardProps {
  configPath: string;
  language: Language;
}

// Loading spinner component
function LoadingSpinner() {
  return (
    <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
    </svg>
  );
}

// Confirmation dialog component
function ConfirmDialog({
  open,
  title,
  message,
  onConfirm,
  onCancel,
  cancelText,
  confirmText,
}: {
  open: boolean;
  title: string;
  message: string;
  onConfirm: () => void;
  onCancel: () => void;
  cancelText: string;
  confirmText: string;
}) {
  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/30" onClick={onCancel} />

      {/* Dialog */}
      <div className="relative bg-white rounded-2xl p-6 shadow-lg max-w-md w-full mx-4">
        <h3 className="text-lg font-semibold text-gray-800 mb-2">{title}</h3>
        <p className="text-gray-600 mb-6">{message}</p>
        <div className="flex justify-end gap-3">
          <button
            onClick={onCancel}
            className="px-4 py-2 text-sm font-medium text-gray-600 bg-gray-100 rounded-lg hover:bg-gray-200"
          >
            {cancelText}
          </button>
          <button
            onClick={onConfirm}
            className="px-4 py-2 text-sm font-medium text-white bg-rose-500 rounded-lg hover:bg-rose-600"
          >
            {confirmText}
          </button>
        </div>
      </div>
    </div>
  );
}

function StatusLight({ status }: { status: string }) {
  let colorClass = 'bg-gray-400';
  let glowClass = '';
  
  if (status === 'running') {
    colorClass = 'bg-emerald-400';
    glowClass = 'shadow-[0_0_8px_rgba(94,220,180,0.7)] animate-pulse';
  } else if (status === 'exited') {
    colorClass = 'bg-amber-400';
  } else if (status === 'stopped') {
    colorClass = 'bg-gray-400';
  }
  
  return (
    <span className={`inline-block w-3 h-3 rounded-full ${colorClass} ${glowClass}`} />
  );
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.floor(seconds)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.floor(seconds % 60)}s`;
  return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
}

export default function Dashboard({ configPath, language }: DashboardProps) {
  const t = getTranslations(language);
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [loading, setLoading] = useState<{ gateway: boolean; process: boolean }>({ gateway: false, process: false });
  const [error, setError] = useState<string | null>(null);
  const [confirmDialog, setConfirmDialog] = useState<{
    open: boolean;
    target: 'gateway' | 'process' | null;
  }>({ open: false, target: null });

  // Fetch status periodically
  const refreshStatus = useCallback(async () => {
    try {
      const data = await fetchStatus();
      setStatus(data);
      setError(null);
    } catch {
      setError(t.failedToConnect);
    }
  }, [t]);

  useEffect(() => {
    refreshStatus();
    const interval = setInterval(refreshStatus, 2000);
    return () => clearInterval(interval);
  }, [refreshStatus]);

  // Handlers
  const handleStartGateway = async () => {
    setLoading(prev => ({ ...prev, gateway: true }));
    try {
      await startGateway(configPath);
      await refreshStatus();
    } catch (err) {
      setError(`${t.failedToStartGateway}: ${err}`);
    } finally {
      setLoading(prev => ({ ...prev, gateway: false }));
    }
  };

  const handleStopGateway = async () => {
    setConfirmDialog({ open: false, target: null });
    setLoading(prev => ({ ...prev, gateway: true }));
    try {
      await stopGateway();
      await refreshStatus();
    } catch (err) {
      setError(`${t.failedToStopGateway}: ${err}`);
    } finally {
      setLoading(prev => ({ ...prev, gateway: false }));
    }
  };

  const handleStartProcess = async () => {
    setLoading(prev => ({ ...prev, process: true }));
    try {
      await startProcess(configPath);
      await refreshStatus();
    } catch (err) {
      setError(`${t.failedToStartProcess}: ${err}`);
    } finally {
      setLoading(prev => ({ ...prev, process: false }));
    }
  };

  const handleStopProcess = async () => {
    setConfirmDialog({ open: false, target: null });
    setLoading(prev => ({ ...prev, process: true }));
    try {
      await stopProcess();
      await refreshStatus();
    } catch (err) {
      setError(`${t.failedToStopProcess}: ${err}`);
    } finally {
      setLoading(prev => ({ ...prev, process: false }));
    }
  };

  const gatewayStatus = status?.gateway?.managed?.status || 'stopped';
  const processStatus = status?.process?.managed?.status || 'stopped';
  const gatewayHealth = status?.gateway?.health;
  const processProgressRaw = status?.process?.progress;
  // Keep aligned with backend PROGRESS_TIMEOUT_SECONDS (15s)
  const processProgressIsFresh = Boolean(
    processProgressRaw && (Date.now() / 1000) - processProgressRaw.ts <= 15
  );
  const processProgress = processProgressIsFresh ? processProgressRaw : undefined;

  // Calculate runtime
  const gatewayRuntime = status?.gateway?.managed?.start_time
    ? (Date.now() / 1000) - status.gateway.managed.start_time
    : 0;
  const processRuntime = status?.process?.managed?.start_time
    ? (Date.now() / 1000) - status.process.managed.start_time
    : 0;

  // Show "External" label if not managed but health check passes
  const gatewayIsExternal = gatewayStatus !== 'running' && Boolean(gatewayHealth);
  const processIsExternal = processStatus !== 'running' && Boolean(processProgress);

  return (
    <div className="p-6 space-y-6">
      {/* Confirmation Dialog */}
      <ConfirmDialog
        open={confirmDialog.open}
        title={interpolate(t.stopConfirmTitle, { target: confirmDialog.target === 'gateway' ? t.gateway : t.process })}
        message={`${interpolate(t.stopConfirmMessage, { target: confirmDialog.target === 'gateway' ? t.gateway : t.process })} ${confirmDialog.target === 'process' ? t.stopConfirmMessageProcess : ''}`}
        onConfirm={confirmDialog.target === 'gateway' ? handleStopGateway : handleStopProcess}
        onCancel={() => setConfirmDialog({ open: false, target: null })}
        cancelText={t.cancel}
        confirmText={t.stop}
      />

      {/* Error Message */}
      {error && (
        <div className="bg-red-50 text-red-600 px-4 py-3 rounded-lg">
          {error}
        </div>
      )}

      {/* Gateway Card */}
      <div className="bg-white rounded-2xl p-6 shadow-[0_2px_12px_rgba(0,0,0,0.04)]">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <StatusLight status={gatewayIsExternal ? 'running' : gatewayStatus} />
            <h2 className="text-lg font-semibold text-gray-800">{t.gateway}</h2>
            {gatewayIsExternal && (
              <span className="px-2 py-0.5 text-xs bg-blue-100 text-blue-600 rounded-full">
                {t.external}
              </span>
            )}
            <span className="text-sm text-gray-500 capitalize">
              {gatewayIsExternal ? `${t.running} (${t.external})` : gatewayStatus === 'running' ? t.running : gatewayStatus === 'stopped' ? t.stopped : t.exited}
            </span>
          </div>
          <div className="flex gap-2">
            <button
              onClick={handleStartGateway}
              disabled={loading.gateway || gatewayStatus === 'running' || gatewayIsExternal}
              className="px-4 py-2 text-sm font-medium text-white bg-gradient-to-r from-cyan-400 to-blue-500 rounded-lg hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              {loading.gateway && gatewayStatus !== 'running' && <LoadingSpinner />}
              {t.start}
            </button>
            <button
              onClick={() => setConfirmDialog({ open: true, target: 'gateway' })}
              disabled={loading.gateway || gatewayStatus !== 'running'}
              className="px-4 py-2 text-sm font-medium text-white bg-rose-400 rounded-lg hover:bg-rose-500 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              {loading.gateway && gatewayStatus === 'running' && <LoadingSpinner />}
              {t.stop}
            </button>
          </div>
        </div>

        {/* Gateway Info */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div>
            <span className="text-gray-500">{t.pid}:</span>
            <span className="ml-2 text-gray-800">{status?.gateway?.managed?.pid || '-'}</span>
          </div>
          <div>
            <span className="text-gray-500">{t.port}:</span>
            <span className="ml-2 text-gray-800">{status?.gateway?.managed?.port || '8787'}</span>
          </div>
          <div>
            <span className="text-gray-500">{t.models}:</span>
            <span className="ml-2 text-gray-800">
              {gatewayHealth ? `${gatewayHealth.available_models}/${gatewayHealth.total_models}` : '-'}
            </span>
          </div>
          <div>
            <span className="text-gray-500">{t.runtime}:</span>
            <span className="ml-2 text-gray-800">
              {gatewayRuntime > 0 ? formatDuration(gatewayRuntime) : (gatewayHealth ? formatDuration(gatewayHealth.uptime) : '-')}
            </span>
          </div>
        </div>
      </div>

      {/* Process Card */}
      <div className="bg-white rounded-2xl p-6 shadow-[0_2px_12px_rgba(0,0,0,0.04)]">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <StatusLight status={processIsExternal ? 'running' : processStatus} />
            <h2 className="text-lg font-semibold text-gray-800">{t.process}</h2>
            {processIsExternal && (
              <span className="px-2 py-0.5 text-xs bg-blue-100 text-blue-600 rounded-full">
                {t.external}
              </span>
            )}
            <span className="text-sm text-gray-500 capitalize">
              {processIsExternal ? `${t.running} (${t.external})` : processStatus === 'running' ? t.running : processStatus === 'stopped' ? t.stopped : t.exited}
            </span>
          </div>
          <div className="flex gap-2">
            <button
              onClick={handleStartProcess}
              disabled={loading.process || processStatus === 'running' || processIsExternal}
              className="px-4 py-2 text-sm font-medium text-white bg-gradient-to-r from-cyan-400 to-blue-500 rounded-lg hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              {loading.process && processStatus !== 'running' && <LoadingSpinner />}
              {t.start}
            </button>
            <button
              onClick={() => setConfirmDialog({ open: true, target: 'process' })}
              disabled={loading.process || processStatus !== 'running'}
              className="px-4 py-2 text-sm font-medium text-white bg-rose-400 rounded-lg hover:bg-rose-500 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              {loading.process && processStatus === 'running' && <LoadingSpinner />}
              {t.stop}
            </button>
          </div>
        </div>

        {/* Process Info */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div>
            <span className="text-gray-500">{t.pid}:</span>
            <span className="ml-2 text-gray-800">{status?.process?.managed?.pid || '-'}</span>
          </div>
          <div>
            <span className="text-gray-500">{t.progress}:</span>
            <span className="ml-2 text-gray-800">
              {processProgress ? `${processProgress.processed}/${processProgress.total}` : '-'}
            </span>
          </div>
          <div>
            <span className="text-gray-500">{t.shard}:</span>
            <span className="ml-2 text-gray-800">{processProgress?.shard || '-'}</span>
          </div>
          <div>
            <span className="text-gray-500">{t.errors}:</span>
            <span className="ml-2 text-gray-800">{processProgress?.errors ?? '-'}</span>
          </div>
        </div>

        {/* Progress Bar */}
        {processProgress && processProgress.total > 0 && (
          <div className="mt-4">
            <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-cyan-400 to-blue-500 transition-all duration-300"
                style={{ width: `${(processProgress.processed / processProgress.total) * 100}%` }}
              />
            </div>
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>{Math.round((processProgress.processed / processProgress.total) * 100)}%</span>
              <span>{processIsExternal ? t.external : formatDuration(processRuntime)}</span>
            </div>
          </div>
        )}

        {/* Exit Code */}
        {processStatus === 'exited' && status?.process?.managed?.exit_code !== null && (
          <div className="mt-4 text-sm">
            <span className="text-gray-500">{t.exitCode}:</span>
            <span className={`ml-2 ${status?.process?.managed?.exit_code === 0 ? 'text-green-600' : 'text-red-600'}`}>
              {status?.process?.managed?.exit_code}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
