import { useState, useEffect, useRef, useCallback } from 'react';
import { AutoReconnectWebSocket } from '../api';
import type { LogTarget } from '../types';

export default function Logs() {
  const [target, setTarget] = useState<LogTarget>('gateway');
  const [logs, setLogs] = useState<string[]>([]);
  const [connected, setConnected] = useState(false);
  const [reconnecting, setReconnecting] = useState(false);
  const [reconnectInfo, setReconnectInfo] = useState<string | null>(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const logContainerRef = useRef<HTMLPreElement>(null);
  const wsRef = useRef<AutoReconnectWebSocket | null>(null);

  const handleMessage = useCallback((line: string) => {
    if (line === 'pong') return;
    setLogs((prev) => {
      const newLogs = [...prev, line];
      if (newLogs.length > 1000) {
        return newLogs.slice(-1000);
      }
      return newLogs;
    });
  }, []);

  const handleConnect = useCallback(() => {
    setConnected(true);
    setReconnecting(false);
    setReconnectInfo(null);
  }, []);

  const handleDisconnect = useCallback(() => {
    setConnected(false);
  }, []);

  const handleReconnecting = useCallback((attempt: number, delay: number) => {
    setReconnecting(true);
    setReconnectInfo(
      `Reconnecting (attempt ${attempt}, ${Math.round(delay / 1000)}s)...`
    );
  }, []);

  const connectWebSocket = useCallback(
    (nextTarget: LogTarget) => {
      if (wsRef.current) {
        wsRef.current.close();
      }

      wsRef.current = new AutoReconnectWebSocket({
        target: nextTarget,
        onMessage: handleMessage,
        onConnect: handleConnect,
        onDisconnect: handleDisconnect,
        onReconnecting: handleReconnecting,
        maxRetries: -1,
        initialDelay: 1000,
        maxDelay: 30000,
      });
    },
    [handleConnect, handleDisconnect, handleMessage, handleReconnecting]
  );

  const switchTarget = (nextTarget: LogTarget) => {
    if (nextTarget === target) return;
    setLogs([]);
    setConnected(false);
    setReconnecting(false);
    setReconnectInfo(null);
    setTarget(nextTarget);
  };

  useEffect(() => {
    connectWebSocket(target);
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connectWebSocket, target]);

  // Auto-scroll
  useEffect(() => {
    if (autoScroll && logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logs, autoScroll]);

  const handleClear = () => {
    setLogs([]);
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(logs.join('\n'));
  };

  const handleReconnect = () => {
    setLogs([]);
    setConnected(false);
    setReconnecting(false);
    setReconnectInfo(null);
    connectWebSocket(target);
  };

  return (
    <div className="p-6 h-full flex flex-col">
      {/* Toolbar */}
      <div className="bg-white rounded-2xl p-4 shadow-[0_2px_12px_rgba(0,0,0,0.04)] mb-4">
        <div className="flex items-center justify-between">
          {/* Target Tabs */}
          <div className="flex gap-1 bg-gray-100 rounded-lg p-1">
            <button
              onClick={() => switchTarget('gateway')}
              className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${
                target === 'gateway'
                  ? 'bg-white text-gray-800 shadow-sm'
                  : 'text-gray-600 hover:text-gray-800'
              }`}
            >
              Gateway
            </button>
            <button
              onClick={() => switchTarget('process')}
              className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${
                target === 'process'
                  ? 'bg-white text-gray-800 shadow-sm'
                  : 'text-gray-600 hover:text-gray-800'
              }`}
            >
              Process
            </button>
          </div>

          {/* Connection Status & Actions */}
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 text-sm">
              <span
                className={`w-2 h-2 rounded-full ${
                  connected ? 'bg-green-400' : reconnecting ? 'bg-yellow-400 animate-pulse' : 'bg-gray-400'
                }`}
              />
              <span className="text-gray-600">
                {connected ? 'Connected' : reconnecting ? reconnectInfo || 'Reconnecting...' : 'Disconnected'}
              </span>
            </div>
            
            <label className="flex items-center gap-2 text-sm text-gray-600">
              <input
                type="checkbox"
                checked={autoScroll}
                onChange={(e) => setAutoScroll(e.target.checked)}
                className="rounded border-gray-300 text-cyan-500 focus:ring-cyan-400"
              />
              Auto-scroll
            </label>

            <div className="flex gap-2">
              {!connected && !reconnecting && (
                <button
                  onClick={handleReconnect}
                  className="px-3 py-1.5 text-sm font-medium text-gray-600 bg-gray-100 rounded-lg hover:bg-gray-200"
                >
                  Reconnect
                </button>
              )}
              <button
                onClick={handleClear}
                className="px-3 py-1.5 text-sm font-medium text-gray-600 bg-gray-100 rounded-lg hover:bg-gray-200"
              >
                Clear
              </button>
              <button
                onClick={handleCopy}
                className="px-3 py-1.5 text-sm font-medium text-gray-600 bg-gray-100 rounded-lg hover:bg-gray-200"
              >
                Copy
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Log Output */}
      <div className="flex-1 bg-slate-800 rounded-2xl overflow-hidden shadow-[0_2px_12px_rgba(0,0,0,0.04)]">
        <pre
          ref={logContainerRef}
          className="h-full p-4 overflow-auto text-sm text-slate-200 font-mono"
          style={{ fontFamily: "'JetBrains Mono', 'Menlo', 'Monaco', monospace" }}
        >
          {logs.length === 0 ? (
            <span className="text-slate-500">
              {connected ? 'Waiting for logs...' : reconnecting ? 'Reconnecting to server...' : 'Not connected. Click Reconnect to start.'}
            </span>
          ) : (
            logs.map((line, i) => (
              <div key={i} className="hover:bg-slate-700/50">
                {line}
              </div>
            ))
          )}
        </pre>
      </div>
    </div>
  );
}
