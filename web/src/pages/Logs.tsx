import { useState, useEffect, useRef, useCallback } from 'react';
import { connectLogs } from '../api';
import type { LogTarget } from '../types';

export default function Logs() {
  const [target, setTarget] = useState<LogTarget>('gateway');
  const [logs, setLogs] = useState<string[]>([]);
  const [connected, setConnected] = useState(false);
  const [autoScroll, setAutoScroll] = useState(true);
  const logContainerRef = useRef<HTMLPreElement>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // Connect to WebSocket
  const connect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
    }

    setLogs([]);
    setConnected(false);

    const ws = connectLogs(
      target,
      (line) => {
        if (line === 'pong') return;  // Ignore ping responses
        setLogs((prev) => {
          const newLogs = [...prev, line];
          // Keep last 1000 lines
          if (newLogs.length > 1000) {
            return newLogs.slice(-1000);
          }
          return newLogs;
        });
      },
      () => {
        setConnected(false);
      }
    );

    ws.onopen = () => {
      setConnected(true);
    };

    ws.onclose = () => {
      setConnected(false);
    };

    wsRef.current = ws;
  }, [target]);

  useEffect(() => {
    connect();
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connect]);

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
    connect();
  };

  return (
    <div className="p-6 h-full flex flex-col">
      {/* Toolbar */}
      <div className="bg-white rounded-2xl p-4 shadow-[0_2px_12px_rgba(0,0,0,0.04)] mb-4">
        <div className="flex items-center justify-between">
          {/* Target Tabs */}
          <div className="flex gap-1 bg-gray-100 rounded-lg p-1">
            <button
              onClick={() => setTarget('gateway')}
              className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${
                target === 'gateway'
                  ? 'bg-white text-gray-800 shadow-sm'
                  : 'text-gray-600 hover:text-gray-800'
              }`}
            >
              Gateway
            </button>
            <button
              onClick={() => setTarget('process')}
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
                  connected ? 'bg-green-400' : 'bg-gray-400'
                }`}
              />
              <span className="text-gray-600">
                {connected ? 'Connected' : 'Disconnected'}
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
              {!connected && (
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
              {connected ? 'Waiting for logs...' : 'Not connected. Click Reconnect to start.'}
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
