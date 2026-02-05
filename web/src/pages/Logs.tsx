/**
 * Logs 页面 - 实时日志查看器
 *
 * 功能:
 * - 左右分栏布局展示 Gateway 和 Process 日志
 * - WebSocket 自动重连
 * - 自动滚动和手动控制
 * - 日志复制和清空
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { AutoReconnectWebSocket } from '../api';
import type { LogTarget } from '../types';
import { getTranslations, type Language } from '../i18n';

interface LogsProps {
  language: Language;
}

interface LogPanelProps {
  target: LogTarget;
  logs: string[];
  connected: boolean;
  reconnecting: boolean;
  reconnectInfo: string | null;
  autoScroll: boolean;
  wordWrap: boolean;
  onToggleAutoScroll: (enabled: boolean) => void;
  onToggleWordWrap: (enabled: boolean) => void;
  onClear: () => void;
  onCopy: () => void;
  onReconnect: () => void;
  t: ReturnType<typeof getTranslations>;
}

// Individual log panel component
function LogPanel({
  target,
  logs,
  connected,
  reconnecting,
  reconnectInfo,
  autoScroll,
  wordWrap,
  onToggleAutoScroll,
  onToggleWordWrap,
  onClear,
  onCopy,
  onReconnect,
  t,
}: LogPanelProps) {
  const logContainerRef = useRef<HTMLPreElement>(null);

  // Auto-scroll
  useEffect(() => {
    if (autoScroll && logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logs, autoScroll]);

  return (
    <div className="flex flex-col h-full">
      {/* Panel Header */}
      <div className="bg-white rounded-t-2xl p-3 border-b border-gray-200">
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-sm font-semibold text-gray-800">
            {target === 'gateway' ? t.gateway : t.process}
          </h3>
          <div className="flex items-center gap-2 text-xs">
            <span
              className={`w-2 h-2 rounded-full ${
                connected ? 'bg-green-400' : reconnecting ? 'bg-yellow-400 animate-pulse' : 'bg-gray-400'
              }`}
            />
            <span className="text-gray-600">
              {connected ? t.connected : reconnecting ? t.reconnecting : t.disconnected}
            </span>
          </div>
        </div>

        {reconnectInfo && (
          <div className="text-xs text-amber-600 mb-2">{reconnectInfo}</div>
        )}

        {/* Controls */}
        <div className="flex items-center gap-2">
          <label className="flex items-center gap-1.5 text-xs text-gray-600 cursor-pointer">
            <input
              type="checkbox"
              checked={autoScroll}
              onChange={(e) => onToggleAutoScroll(e.target.checked)}
              className="rounded border-gray-300 text-cyan-500 focus:ring-cyan-400 w-3.5 h-3.5"
            />
            {t.autoScroll}
          </label>

          <label className="flex items-center gap-1.5 text-xs text-gray-600 cursor-pointer">
            <input
              type="checkbox"
              checked={wordWrap}
              onChange={(e) => onToggleWordWrap(e.target.checked)}
              className="rounded border-gray-300 text-cyan-500 focus:ring-cyan-400 w-3.5 h-3.5"
            />
            {t.wordWrap}
          </label>

          <div className="flex-1" />

          {!connected && !reconnecting && (
            <button
              onClick={onReconnect}
              className="px-2 py-1 text-xs font-medium text-gray-600 bg-gray-100 rounded hover:bg-gray-200"
            >
              {t.reconnect}
            </button>
          )}
          <button
            onClick={onClear}
            className="px-2 py-1 text-xs font-medium text-gray-600 bg-gray-100 rounded hover:bg-gray-200"
          >
            {t.clear}
          </button>
          <button
            onClick={onCopy}
            className="px-2 py-1 text-xs font-medium text-gray-600 bg-gray-100 rounded hover:bg-gray-200"
          >
            {t.copy}
          </button>
        </div>
      </div>

      {/* Log Output */}
      <div className="flex-1 bg-slate-800 rounded-b-2xl overflow-hidden">
        <pre
          ref={logContainerRef}
          className={`h-full p-3 overflow-auto text-xs text-slate-200 font-mono ${wordWrap ? 'whitespace-pre-wrap break-all' : 'whitespace-pre'}`}
          style={{ fontFamily: "'JetBrains Mono', 'Menlo', 'Monaco', monospace" }}
        >
          {logs.length === 0 ? (
            <span className="text-slate-500">
              {connected ? t.waitingForLogs : reconnecting ? t.reconnectingToServer : t.notConnectedClickReconnect}
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

export default function Logs({ language }: LogsProps) {
  const t = getTranslations(language);

  // Gateway logs state
  const [gatewayLogs, setGatewayLogs] = useState<string[]>([]);
  const [gatewayConnected, setGatewayConnected] = useState(false);
  const [gatewayReconnecting, setGatewayReconnecting] = useState(false);
  const [gatewayReconnectInfo, setGatewayReconnectInfo] = useState<string | null>(null);
  const [gatewayAutoScroll, setGatewayAutoScroll] = useState(true);

  // Process logs state
  const [processLogs, setProcessLogs] = useState<string[]>([]);
  const [processConnected, setProcessConnected] = useState(false);
  const [processReconnecting, setProcessReconnecting] = useState(false);
  const [processReconnectInfo, setProcessReconnectInfo] = useState<string | null>(null);
  const [processAutoScroll, setProcessAutoScroll] = useState(true);

  // Word wrap state
  const [gatewayWordWrap, setGatewayWordWrap] = useState(false);
  const [processWordWrap, setProcessWordWrap] = useState(false);

  const gatewayWsRef = useRef<AutoReconnectWebSocket | null>(null);
  const processWsRef = useRef<AutoReconnectWebSocket | null>(null);

  // Gateway WebSocket handlers
  const handleGatewayMessage = useCallback((line: string) => {
    if (line === 'pong') return;
    setGatewayLogs((prev) => {
      const newLogs = [...prev, line];
      if (newLogs.length > 1000) {
        return newLogs.slice(-1000);
      }
      return newLogs;
    });
  }, []);

  const handleGatewayConnect = useCallback(() => {
    setGatewayConnected(true);
    setGatewayReconnecting(false);
    setGatewayReconnectInfo(null);
  }, []);

  const handleGatewayDisconnect = useCallback(() => {
    setGatewayConnected(false);
  }, []);

  const handleGatewayReconnecting = useCallback(
    (attempt: number, delay: number) => {
      setGatewayReconnecting(true);
      const seconds = Math.round(delay / 1000);
      const msg = language === 'zh'
        ? `重新连接中 (尝试 ${attempt}, ${seconds}秒)...`
        : `Reconnecting (attempt ${attempt}, ${seconds}s)...`;
      setGatewayReconnectInfo(msg);
    },
    [language]
  );

  // Process WebSocket handlers
  const handleProcessMessage = useCallback((line: string) => {
    if (line === 'pong') return;
    setProcessLogs((prev) => {
      const newLogs = [...prev, line];
      if (newLogs.length > 1000) {
        return newLogs.slice(-1000);
      }
      return newLogs;
    });
  }, []);

  const handleProcessConnect = useCallback(() => {
    setProcessConnected(true);
    setProcessReconnecting(false);
    setProcessReconnectInfo(null);
  }, []);

  const handleProcessDisconnect = useCallback(() => {
    setProcessConnected(false);
  }, []);

  const handleProcessReconnecting = useCallback(
    (attempt: number, delay: number) => {
      setProcessReconnecting(true);
      const seconds = Math.round(delay / 1000);
      const msg = language === 'zh'
        ? `重新连接中 (尝试 ${attempt}, ${seconds}秒)...`
        : `Reconnecting (attempt ${attempt}, ${seconds}s)...`;
      setProcessReconnectInfo(msg);
    },
    [language]
  );

  // Connect WebSockets
  const connectGatewayWebSocket = useCallback(() => {
    if (gatewayWsRef.current) {
      gatewayWsRef.current.close();
    }

    gatewayWsRef.current = new AutoReconnectWebSocket({
      target: 'gateway',
      onMessage: handleGatewayMessage,
      onConnect: handleGatewayConnect,
      onDisconnect: handleGatewayDisconnect,
      onReconnecting: handleGatewayReconnecting,
      maxRetries: -1,
      initialDelay: 1000,
      maxDelay: 30000,
    });
  }, [
    handleGatewayConnect,
    handleGatewayDisconnect,
    handleGatewayMessage,
    handleGatewayReconnecting,
  ]);

  const connectProcessWebSocket = useCallback(() => {
    if (processWsRef.current) {
      processWsRef.current.close();
    }

    processWsRef.current = new AutoReconnectWebSocket({
      target: 'process',
      onMessage: handleProcessMessage,
      onConnect: handleProcessConnect,
      onDisconnect: handleProcessDisconnect,
      onReconnecting: handleProcessReconnecting,
      maxRetries: -1,
      initialDelay: 1000,
      maxDelay: 30000,
    });
  }, [
    handleProcessConnect,
    handleProcessDisconnect,
    handleProcessMessage,
    handleProcessReconnecting,
  ]);

  // Initialize WebSockets
  useEffect(() => {
    connectGatewayWebSocket();
    connectProcessWebSocket();

    return () => {
      if (gatewayWsRef.current) {
        gatewayWsRef.current.close();
      }
      if (processWsRef.current) {
        processWsRef.current.close();
      }
    };
  }, [connectGatewayWebSocket, connectProcessWebSocket]);

  const handleGatewayClear = () => setGatewayLogs([]);
  const handleProcessClear = () => setProcessLogs([]);

  const handleGatewayCopy = () => {
    navigator.clipboard.writeText(gatewayLogs.join('\n'));
  };

  const handleProcessCopy = () => {
    navigator.clipboard.writeText(processLogs.join('\n'));
  };

  const handleGatewayReconnect = () => {
    setGatewayLogs([]);
    setGatewayConnected(false);
    setGatewayReconnecting(false);
    setGatewayReconnectInfo(null);
    connectGatewayWebSocket();
  };

  const handleProcessReconnect = () => {
    setProcessLogs([]);
    setProcessConnected(false);
    setProcessReconnecting(false);
    setProcessReconnectInfo(null);
    connectProcessWebSocket();
  };

  return (
    <div className="p-6 h-full flex">
      {/* Side-by-side layout */}
      <div className="flex-1 flex gap-6" style={{ height: 'calc(100vh - 200px)' }}>
        {/* Gateway Panel */}
        <div className="flex-1 flex flex-col">
          <LogPanel
            target="gateway"
            logs={gatewayLogs}
            connected={gatewayConnected}
            reconnecting={gatewayReconnecting}
            reconnectInfo={gatewayReconnectInfo}
            autoScroll={gatewayAutoScroll}
            wordWrap={gatewayWordWrap}
            onToggleAutoScroll={setGatewayAutoScroll}
            onToggleWordWrap={setGatewayWordWrap}
            onClear={handleGatewayClear}
            onCopy={handleGatewayCopy}
            onReconnect={handleGatewayReconnect}
            t={t}
          />
        </div>

        {/* Process Panel */}
        <div className="flex-1 flex flex-col">
          <LogPanel
            target="process"
            logs={processLogs}
            connected={processConnected}
            reconnecting={processReconnecting}
            reconnectInfo={processReconnectInfo}
            autoScroll={processAutoScroll}
            wordWrap={processWordWrap}
            onToggleAutoScroll={setProcessAutoScroll}
            onToggleWordWrap={setProcessWordWrap}
            onClear={handleProcessClear}
            onCopy={handleProcessCopy}
            onReconnect={handleProcessReconnect}
            t={t}
          />
        </div>
      </div>
    </div>
  );
}
