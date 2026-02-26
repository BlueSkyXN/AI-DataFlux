/**
 * Logs 页面 - 实时日志查看器
 *
 * 功能:
 * - 左右分栏布局展示 Gateway 和 Process 日志
 * - WebSocket 自动重连（指数退避策略）
 * - 自动滚动和手动控制
 * - 日志复制和清空
 * - 自动换行开关
 * - 日志条数上限 1000 行（超出自动截断旧日志）
 *
 * 导出：默认导出 Logs 组件
 *
 * 内部组件：
 * - LogPanel — 单个日志面板（含工具栏 + 日志输出区）
 *
 * 依赖模块：
 * - api.AutoReconnectWebSocket — 自动重连 WebSocket 类
 * - types.LogTarget — 日志目标类型
 * - i18n — 国际化翻译
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { AutoReconnectWebSocket } from '../api';
import type { LogTarget } from '../types';
import { getTranslations, type Language } from '../i18n';

/** Logs 页面组件的 Props 类型 */
interface LogsProps {
  language: Language;
}

/** 单个日志面板组件的 Props 类型 */
interface LogPanelProps {
  /** 日志目标：网关或处理器 */
  target: LogTarget;
  /** 日志行数组 */
  logs: string[];
  /** 是否已连接 */
  connected: boolean;
  /** 是否正在重连 */
  reconnecting: boolean;
  /** 重连状态描述信息 */
  reconnectInfo: string | null;
  /** 是否启用自动滚动 */
  autoScroll: boolean;
  /** 是否启用自动换行 */
  wordWrap: boolean;
  /** 切换自动滚动 */
  onToggleAutoScroll: (enabled: boolean) => void;
  /** 切换自动换行 */
  onToggleWordWrap: (enabled: boolean) => void;
  /** 清空日志 */
  onClear: () => void;
  /** 复制日志到剪贴板 */
  onCopy: () => void;
  /** 手动重连 */
  onReconnect: () => void;
  /** 翻译文本对象 */
  t: ReturnType<typeof getTranslations>;
}

/**
 * 单个日志面板组件
 *
 * 包含：顶部工具栏（连接状态、自动滚动、换行、重连/清空/复制按钮）
 * 和底部暗色终端风格日志输出区域。
 */
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

  // Auto-scroll — 日志更新时自动滚动到底部
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

/**
 * Logs 页面主组件
 *
 * 管理 Gateway 和 Process 两个日志面板的：
 * - WebSocket 连接生命周期（创建、重连、关闭）
 * - 日志数据状态（缓冲、清空、复制）
 * - 连接状态指示（已连接 / 重连中 / 已断开）
 */
export default function Logs({ language }: LogsProps) {
  const t = getTranslations(language);

  // Gateway logs state — 网关日志状态
  const [gatewayLogs, setGatewayLogs] = useState<string[]>([]);
  const [gatewayConnected, setGatewayConnected] = useState(false);
  const [gatewayReconnecting, setGatewayReconnecting] = useState(false);
  const [gatewayReconnectInfo, setGatewayReconnectInfo] = useState<string | null>(null);
  const [gatewayAutoScroll, setGatewayAutoScroll] = useState(true);

  // Process logs state — 处理器日志状态
  const [processLogs, setProcessLogs] = useState<string[]>([]);
  const [processConnected, setProcessConnected] = useState(false);
  const [processReconnecting, setProcessReconnecting] = useState(false);
  const [processReconnectInfo, setProcessReconnectInfo] = useState<string | null>(null);
  const [processAutoScroll, setProcessAutoScroll] = useState(true);

  // Word wrap state — 自动换行状态
  const [gatewayWordWrap, setGatewayWordWrap] = useState(false);
  const [processWordWrap, setProcessWordWrap] = useState(false);

  /** WebSocket 实例引用 */
  const gatewayWsRef = useRef<AutoReconnectWebSocket | null>(null);
  const processWsRef = useRef<AutoReconnectWebSocket | null>(null);

  // Gateway WebSocket handlers — 网关 WebSocket 事件处理器
  /** 处理网关日志消息（过滤 pong，限制最大 1000 行） */
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

  // Process WebSocket handlers — 处理器 WebSocket 事件处理器
  /** 处理处理器日志消息（过滤 pong，限制最大 1000 行） */
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

  // Connect WebSockets — 创建 WebSocket 连接
  /** 创建网关日志 WebSocket 连接（关闭旧连接后重建） */
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

  /** 创建处理器日志 WebSocket 连接（关闭旧连接后重建） */
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

  // Initialize WebSockets — 初始化 WebSocket 连接（组件挂载时建立，卸载时关闭）
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

  /** 清空网关日志 */
  const handleGatewayClear = () => setGatewayLogs([]);
  /** 清空处理器日志 */
  const handleProcessClear = () => setProcessLogs([]);

  /** 复制网关日志到剪贴板 */
  const handleGatewayCopy = () => {
    navigator.clipboard.writeText(gatewayLogs.join('\n'));
  };

  /** 复制处理器日志到剪贴板 */
  const handleProcessCopy = () => {
    navigator.clipboard.writeText(processLogs.join('\n'));
  };

  /** 手动重连网关 WebSocket（清空旧日志和状态后重建连接） */
  const handleGatewayReconnect = () => {
    setGatewayLogs([]);
    setGatewayConnected(false);
    setGatewayReconnecting(false);
    setGatewayReconnectInfo(null);
    connectGatewayWebSocket();
  };

  /** 手动重连处理器 WebSocket（清空旧日志和状态后重建连接） */
  const handleProcessReconnect = () => {
    setProcessLogs([]);
    setProcessConnected(false);
    setProcessReconnecting(false);
    setProcessReconnectInfo(null);
    connectProcessWebSocket();
  };

  return (
    <div className="flex flex-col h-full">
      {/* Side-by-side layout with proper flex container */}
      <div className="flex-1 flex gap-4 p-4 min-h-0">
        {/* Gateway Panel */}
        <div className="flex-1 flex flex-col min-w-0">
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
        <div className="flex-1 flex flex-col min-w-0">
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
