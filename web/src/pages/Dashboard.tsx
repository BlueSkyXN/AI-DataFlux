import { useEffect, useState, useCallback } from 'react';
import type { StatusResponse } from '../types';
import { fetchStatus, startGateway, stopGateway, startProcess, stopProcess } from '../api';

interface DashboardProps {
  configPath: string;
  onConfigPathChange: (path: string) => void;
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

export default function Dashboard({ configPath, onConfigPathChange }: DashboardProps) {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [loading, setLoading] = useState<{ gateway: boolean; process: boolean }>({ gateway: false, process: false });
  const [error, setError] = useState<string | null>(null);

  // Fetch status periodically
  const refreshStatus = useCallback(async () => {
    try {
      const data = await fetchStatus();
      setStatus(data);
      setError(null);
    } catch {
      setError('Failed to connect to Control Server');
    }
  }, []);

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
      setError(`Failed to start Gateway: ${err}`);
    } finally {
      setLoading(prev => ({ ...prev, gateway: false }));
    }
  };

  const handleStopGateway = async () => {
    setLoading(prev => ({ ...prev, gateway: true }));
    try {
      await stopGateway();
      await refreshStatus();
    } catch (err) {
      setError(`Failed to stop Gateway: ${err}`);
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
      setError(`Failed to start Process: ${err}`);
    } finally {
      setLoading(prev => ({ ...prev, process: false }));
    }
  };

  const handleStopProcess = async () => {
    setLoading(prev => ({ ...prev, process: true }));
    try {
      await stopProcess();
      await refreshStatus();
    } catch (err) {
      setError(`Failed to stop Process: ${err}`);
    } finally {
      setLoading(prev => ({ ...prev, process: false }));
    }
  };

  const gatewayStatus = status?.gateway?.managed?.status || 'stopped';
  const processStatus = status?.process?.managed?.status || 'stopped';
  const gatewayHealth = status?.gateway?.health;
  const processProgress = status?.process?.progress;

  // Calculate runtime
  const gatewayRuntime = status?.gateway?.managed?.start_time
    ? (Date.now() / 1000) - status.gateway.managed.start_time
    : 0;
  const processRuntime = status?.process?.managed?.start_time
    ? (Date.now() / 1000) - status.process.managed.start_time
    : 0;

  // Show "External" label if not managed but health check passes
  const gatewayIsExternal = gatewayStatus !== 'running' && gatewayHealth;

  return (
    <div className="p-6 space-y-6">
      {/* Config Path Input */}
      <div className="bg-white rounded-2xl p-6 shadow-[0_2px_12px_rgba(0,0,0,0.04)]">
        <label className="block text-sm font-medium text-gray-600 mb-2">
          Config File Path
        </label>
        <input
          type="text"
          value={configPath}
          onChange={(e) => onConfigPathChange(e.target.value)}
          className="w-full px-4 py-2 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-cyan-400 focus:border-transparent"
          placeholder="config.yaml"
        />
      </div>

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
            <h2 className="text-lg font-semibold text-gray-800">Gateway</h2>
            {gatewayIsExternal && (
              <span className="px-2 py-0.5 text-xs bg-blue-100 text-blue-600 rounded-full">
                External
              </span>
            )}
            <span className="text-sm text-gray-500 capitalize">
              {gatewayIsExternal ? 'Running (External)' : gatewayStatus}
            </span>
          </div>
          <div className="flex gap-2">
            <button
              onClick={handleStartGateway}
              disabled={loading.gateway || gatewayStatus === 'running'}
              className="px-4 py-2 text-sm font-medium text-white bg-gradient-to-r from-cyan-400 to-blue-500 rounded-lg hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Start
            </button>
            <button
              onClick={handleStopGateway}
              disabled={loading.gateway || gatewayStatus !== 'running'}
              className="px-4 py-2 text-sm font-medium text-white bg-rose-400 rounded-lg hover:bg-rose-500 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Stop
            </button>
          </div>
        </div>

        {/* Gateway Info */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div>
            <span className="text-gray-500">PID:</span>
            <span className="ml-2 text-gray-800">{status?.gateway?.managed?.pid || '-'}</span>
          </div>
          <div>
            <span className="text-gray-500">Port:</span>
            <span className="ml-2 text-gray-800">{status?.gateway?.managed?.port || '8787'}</span>
          </div>
          <div>
            <span className="text-gray-500">Models:</span>
            <span className="ml-2 text-gray-800">
              {gatewayHealth ? `${gatewayHealth.available_models}/${gatewayHealth.total_models}` : '-'}
            </span>
          </div>
          <div>
            <span className="text-gray-500">Runtime:</span>
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
            <StatusLight status={processStatus} />
            <h2 className="text-lg font-semibold text-gray-800">Process</h2>
            <span className="text-sm text-gray-500 capitalize">{processStatus}</span>
          </div>
          <div className="flex gap-2">
            <button
              onClick={handleStartProcess}
              disabled={loading.process || processStatus === 'running'}
              className="px-4 py-2 text-sm font-medium text-white bg-gradient-to-r from-cyan-400 to-blue-500 rounded-lg hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Start
            </button>
            <button
              onClick={handleStopProcess}
              disabled={loading.process || processStatus !== 'running'}
              className="px-4 py-2 text-sm font-medium text-white bg-rose-400 rounded-lg hover:bg-rose-500 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Stop
            </button>
          </div>
        </div>

        {/* Process Info */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div>
            <span className="text-gray-500">PID:</span>
            <span className="ml-2 text-gray-800">{status?.process?.managed?.pid || '-'}</span>
          </div>
          <div>
            <span className="text-gray-500">Progress:</span>
            <span className="ml-2 text-gray-800">
              {processProgress ? `${processProgress.processed}/${processProgress.total}` : '-'}
            </span>
          </div>
          <div>
            <span className="text-gray-500">Shard:</span>
            <span className="ml-2 text-gray-800">{processProgress?.shard || '-'}</span>
          </div>
          <div>
            <span className="text-gray-500">Errors:</span>
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
              <span>{formatDuration(processRuntime)}</span>
            </div>
          </div>
        )}

        {/* Exit Code */}
        {processStatus === 'exited' && status?.process?.managed?.exit_code !== null && (
          <div className="mt-4 text-sm">
            <span className="text-gray-500">Exit Code:</span>
            <span className={`ml-2 ${status?.process?.managed?.exit_code === 0 ? 'text-green-600' : 'text-red-600'}`}>
              {status?.process?.managed?.exit_code}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
