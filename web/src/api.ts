/**
 * API 客户端模块 — 与控制服务器 (Control Server) 通信的统一接口
 *
 * 导出函数/类清单：
 * - fetchStatus()         — 获取系统状态（网关 + 处理器）
 * - fetchConfig()         — 读取 YAML 配置文件内容
 * - saveConfig()          — 保存 YAML 配置到文件
 * - validateConfig()      — 校验 YAML 配置语法
 * - startGateway()        — 启动 API 网关进程
 * - stopGateway()         — 停止 API 网关进程
 * - startProcess()        — 启动数据处理进程
 * - stopProcess()         — 停止数据处理进程
 * - testFeishuConnection()— 测试飞书连接
 * - connectLogs()         — 建立日志 WebSocket 连接（基础版）
 * - AutoReconnectWebSocket— 自动重连 WebSocket 类（带指数退避）
 *
 * 依赖模块：
 * - types — API 响应类型定义
 *
 * 鉴权机制：
 * 通过 URL hash 中的 token 参数或 sessionStorage 中缓存的 token 进行 Bearer 认证。
 * WebSocket 连接通过 Sec-WebSocket-Protocol 子协议传递 token（Base64URL 编码）。
 */

// API service for communicating with the Control Server

import type {
  StatusResponse,
  ConfigResponse,
  ConfigWriteResponse,
  ConfigValidateResponse,
  ManagedProcessStatus,
} from './types';

const API_BASE = '';  // Use relative URLs (proxied in dev, served by same host in prod)
/** sessionStorage 中存储控制面板 Token 的键名 */
const CONTROL_TOKEN_SESSION_KEY = 'dataflux-control-token';
/** 内存级 Token 缓存，避免重复读取 sessionStorage */
let controlTokenCache: string | null = null;

/**
 * 获取控制面板鉴权 Token
 *
 * 优先从 URL hash 参数 `#token=xxx` 中提取（首次访问），
 * 提取后存入 sessionStorage 并从 URL 中移除以保持地址栏整洁。
 * 后续请求从 sessionStorage 或内存缓存中读取。
 *
 * @returns 鉴权 Token 字符串，未设置时返回空字符串
 */

function getControlToken(): string {
  if (controlTokenCache !== null) {
    return controlTokenCache;
  }

  // 从 URL hash 中解析 token 参数（格式：#token=xxx）
  const hash = window.location.hash.startsWith('#')
    ? window.location.hash.slice(1)
    : window.location.hash;
  const params = new URLSearchParams(hash);
  const hashToken = (params.get('token') ?? '').trim();
  if (hashToken) {
    controlTokenCache = hashToken;
    window.sessionStorage.setItem(CONTROL_TOKEN_SESSION_KEY, hashToken);
    // 从 URL 中移除 token 参数，防止泄露
    params.delete('token');
    const nextHash = params.toString();
    const nextUrl = `${window.location.pathname}${window.location.search}${nextHash ? `#${nextHash}` : ''}`;
    window.history.replaceState({}, '', nextUrl);
    return controlTokenCache;
  }

  // 回退到 sessionStorage 中缓存的 token
  const sessionToken = (window.sessionStorage.getItem(CONTROL_TOKEN_SESSION_KEY) ?? '').trim();
  controlTokenCache = sessionToken;
  return controlTokenCache;
}

/**
 * 为请求头注入 Bearer 鉴权信息
 * @param headers - 已有的请求头键值对
 * @returns 包含 Authorization 头的新请求头对象
 */
function withAuthHeaders(headers: Record<string, string> = {}): Record<string, string> {
  const token = getControlToken();
  if (!token) {
    return headers;
  }
  return {
    ...headers,
    Authorization: `Bearer ${token}`,
  };
}

/**
 * 构建日志 WebSocket 连接 URL
 * @param target - 日志目标：'gateway'（网关）或 'process'（处理器）
 * @returns 完整的 ws:// 或 wss:// URL
 */
function buildLogsWsUrl(target: 'gateway' | 'process'): string {
  // 根据当前页面协议自动选择 ws 或 wss
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}/api/logs?target=${target}`;
}

/**
 * 将 Token 编码为 Base64URL 格式，用于 WebSocket 子协议传递
 *
 * WebSocket 的 Sec-WebSocket-Protocol 不支持任意字符，
 * 因此将 Token 通过 Base64URL 编码后嵌入子协议名中。
 *
 * @param token - 原始 Token 字符串
 * @returns Base64URL 编码后的字符串（无填充符）
 */

function encodeTokenForWsProtocol(token: string): string {
  const bytes = new TextEncoder().encode(token);
  let binary = '';
  for (const byte of bytes) {
    binary += String.fromCharCode(byte);
  }
  const base64 = window.btoa(binary);
  // 转换为 Base64URL：替换 +/ 并移除尾部 = 填充
  return base64.replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/g, '');
}

/**
 * 构建 WebSocket 子协议数组，用于携带鉴权 Token
 * @returns 包含 Token 子协议的数组；无 Token 时返回 undefined
 */

function buildLogsWsProtocols(): string[] | undefined {
  const token = getControlToken();
  if (!token) {
    return undefined;
  }
  return [`dataflux-token-b64.${encodeTokenForWsProtocol(token)}`];
}

/**
 * 获取系统状态
 * 调用 GET /api/status，返回网关和处理器的运行状态、健康信息及处理进度。
 * @returns 聚合状态响应
 * @throws 请求失败时抛出 Error
 */
export async function fetchStatus(): Promise<StatusResponse> {
  const response = await fetch(`${API_BASE}/api/status`, {
    headers: withAuthHeaders(),
  });
  if (!response.ok) {
    throw new Error(`Failed to fetch status: ${response.status}`);
  }
  return response.json();
}

/**
 * 读取配置文件内容
 * 调用 GET /api/config?path=...，返回文件绝对路径和 YAML 内容。
 * @param path - 配置文件路径（相对或绝对）
 * @returns 配置文件响应（路径 + 内容）
 */
export async function fetchConfig(path: string): Promise<ConfigResponse> {
  const response = await fetch(`${API_BASE}/api/config?path=${encodeURIComponent(path)}`, {
    headers: withAuthHeaders(),
  });
  if (!response.ok) {
    throw new Error(`Failed to fetch config: ${response.status}`);
  }
  return response.json();
}

/**
 * 保存配置文件
 * 调用 PUT /api/config，将 YAML 内容写入指定文件，后端自动创建备份。
 * @param path - 配置文件路径
 * @param content - 要保存的 YAML 内容
 * @returns 保存结果（是否成功、是否有备份）
 */
export async function saveConfig(path: string, content: string): Promise<ConfigWriteResponse> {
  const response = await fetch(`${API_BASE}/api/config`, {
    method: 'PUT',
    headers: withAuthHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ path, content }),
  });
  if (!response.ok) {
    throw new Error(`Failed to save config: ${response.status}`);
  }
  return response.json();
}

/**
 * 校验 YAML 配置语法
 * 调用 POST /api/config/validate，服务端解析并验证 YAML 格式。
 * @param content - 待验证的 YAML 字符串
 * @returns 验证结果（是否合法 + 错误信息）
 */
export async function validateConfig(content: string): Promise<ConfigValidateResponse> {
  const response = await fetch(`${API_BASE}/api/config/validate`, {
    method: 'POST',
    headers: withAuthHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ content }),
  });
  if (!response.ok) {
    throw new Error(`Failed to validate config: ${response.status}`);
  }
  return response.json();
}

/**
 * 启动 API 网关进程
 * 调用 POST /api/gateway/start，以指定配置、端口和 worker 数启动网关。
 * @param configPath - 配置文件路径，默认 'config.yaml'
 * @param port - 监听端口，默认 8787
 * @param workers - worker 进程数，默认 1
 * @returns 网关进程状态
 */
export async function startGateway(
  configPath: string = 'config.yaml',
  port: number = 8787,
  workers: number = 1
): Promise<ManagedProcessStatus> {
  const response = await fetch(`${API_BASE}/api/gateway/start`, {
    method: 'POST',
    headers: withAuthHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ config_path: configPath, port, workers }),
  });
  if (!response.ok) {
    throw new Error(`Failed to start gateway: ${response.status}`);
  }
  return response.json();
}

/**
 * 停止 API 网关进程
 * 调用 POST /api/gateway/stop
 * @returns 停止后的网关进程状态
 */
export async function stopGateway(): Promise<ManagedProcessStatus> {
  const response = await fetch(`${API_BASE}/api/gateway/stop`, {
    method: 'POST',
    headers: withAuthHeaders({ 'Content-Type': 'application/json' }),
    body: '{}',
  });
  if (!response.ok) {
    throw new Error(`Failed to stop gateway: ${response.status}`);
  }
  return response.json();
}

/**
 * 启动数据处理进程
 * 调用 POST /api/process/start
 * @param configPath - 配置文件路径，默认 'config.yaml'
 * @returns 处理器进程状态
 */
export async function startProcess(configPath: string = 'config.yaml'): Promise<ManagedProcessStatus> {
  const response = await fetch(`${API_BASE}/api/process/start`, {
    method: 'POST',
    headers: withAuthHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ config_path: configPath }),
  });
  if (!response.ok) {
    throw new Error(`Failed to start process: ${response.status}`);
  }
  return response.json();
}

/**
 * 停止数据处理进程
 * 调用 POST /api/process/stop
 * @returns 停止后的处理器进程状态
 */
export async function stopProcess(): Promise<ManagedProcessStatus> {
  const response = await fetch(`${API_BASE}/api/process/stop`, {
    method: 'POST',
    headers: withAuthHeaders({ 'Content-Type': 'application/json' }),
    body: '{}',
  });
  if (!response.ok) {
    throw new Error(`Failed to stop process: ${response.status}`);
  }
  return response.json();
}

/**
 * 测试飞书（Lark）API 连接
 * 调用 POST /api/feishu/test_connection，验证 App ID 和 App Secret 是否有效。
 * @param appId - 飞书应用 App ID
 * @param appSecret - 飞书应用 App Secret
 * @returns 连接测试结果（是否成功、消息、Token 预览）
 */
export async function testFeishuConnection(
  appId: string,
  appSecret: string
): Promise<{ success: boolean; message: string; token_preview?: string }> {
  const response = await fetch(`${API_BASE}/api/feishu/test_connection`, {
    method: 'POST',
    headers: withAuthHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ app_id: appId, app_secret: appSecret }),
  });

  if (!response.ok) {
    throw new Error(`Failed to test connection: ${response.status}`);
  }

  return response.json();
}

/**
 * 建立日志 WebSocket 连接（基础版，不自动重连）
 *
 * 内置 30 秒心跳 ping 保活机制，连接关闭时自动清理定时器。
 *
 * @param target - 日志目标：'gateway' 或 'process'
 * @param onMessage - 收到日志行时的回调
 * @param onError - 连接错误时的回调（可选）
 * @returns WebSocket 实例
 */
// WebSocket connection for logs (basic)
export function connectLogs(
  target: 'gateway' | 'process',
  onMessage: (line: string) => void,
  onError?: (error: Event) => void
): WebSocket {
  const wsUrl = buildLogsWsUrl(target);
  const ws = new WebSocket(wsUrl, buildLogsWsProtocols());
  
  ws.onmessage = (event) => {
    onMessage(event.data);
  };
  
  ws.onerror = (event) => {
    if (onError) {
      onError(event);
    }
  };
  
  // Keep-alive ping — 每 30 秒发送心跳保持连接
  const pingInterval = setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send('ping');
    }
  }, 30000);
  
  ws.onclose = () => {
    clearInterval(pingInterval);
  };
  
  return ws;
}

/**
 * 自动重连 WebSocket 连接配置选项
 */
// Auto-reconnecting WebSocket connection with exponential backoff
export interface AutoReconnectOptions {
  /** 日志目标：'gateway' 或 'process' */
  target: 'gateway' | 'process';
  /** 收到日志消息的回调 */
  onMessage: (line: string) => void;
  /** 连接建立成功的回调 */
  onConnect?: () => void;
  /** 连接断开的回调 */
  onDisconnect?: () => void;
  /** 重连中的回调，参数为重试次数和延迟毫秒数 */
  onReconnecting?: (attempt: number, delay: number) => void;
  /** 最大重试次数，-1 表示无限重试 */
  maxRetries?: number;  // -1 for infinite
  /** 初始重连延迟（毫秒） */
  initialDelay?: number;  // ms
  /** 最大重连延迟（毫秒） */
  maxDelay?: number;  // ms
}

/**
 * 自动重连 WebSocket 类
 *
 * 封装 WebSocket 连接管理，提供：
 * - 指数退避重连策略（带随机抖动）
 * - 30 秒心跳保活
 * - 自动过滤 pong 响应消息
 * - 可配置最大重试次数
 *
 * 使用方式：new AutoReconnectWebSocket(options) 后自动连接，
 * 调用 close() 停止连接和重连。
 */
export class AutoReconnectWebSocket {
  /** 当前 WebSocket 实例 */
  private ws: WebSocket | null = null;
  /** 心跳 ping 定时器 */
  private pingInterval: ReturnType<typeof setInterval> | null = null;
  /** 重连延迟定时器 */
  private reconnectTimeout: ReturnType<typeof setTimeout> | null = null;
  /** 当前重试次数 */
  private retryCount = 0;
  /** 是否已被用户主动关闭 */
  private isClosed = false;
  /** 完整配置（已合并默认值） */
  private options: Required<AutoReconnectOptions>;

  constructor(options: AutoReconnectOptions) {
    this.options = {
      maxRetries: -1,
      initialDelay: 1000,
      maxDelay: 30000,
      onConnect: () => {},
      onDisconnect: () => {},
      onReconnecting: () => {},
      ...options,
    };
    this.connect();
  }

  /** 建立 WebSocket 连接 */
  private connect(): void {
    if (this.isClosed) return;

    const wsUrl = buildLogsWsUrl(this.options.target);
    
    try {
      this.ws = new WebSocket(wsUrl, buildLogsWsProtocols());
    } catch {
      this.scheduleReconnect();
      return;
    }

    this.ws.onopen = () => {
      this.retryCount = 0;
      this.options.onConnect?.();
      this.startPing();
    };

    this.ws.onmessage = (event) => {
      // 过滤掉心跳 pong 响应，只传递实际日志消息
      if (event.data !== 'pong') {
        this.options.onMessage(event.data);
      }
    };

    this.ws.onerror = () => {
      // Error will be followed by close, so we handle reconnect there
      // 错误事件之后必然触发 close，因此在 onclose 中统一处理重连
    };

    this.ws.onclose = () => {
      this.stopPing();
      this.options.onDisconnect?.();
      if (!this.isClosed) {
        this.scheduleReconnect();
      }
    };
  }

  /** 计划重连：使用指数退避 + 随机抖动策略 */
  private scheduleReconnect(): void {
    if (this.isClosed) return;
    if (this.options.maxRetries >= 0 && this.retryCount >= this.options.maxRetries) {
      return;
    }

    // Exponential backoff with jitter — 指数退避 + 随机抖动，防止雪崩
    const delay = Math.min(
      this.options.initialDelay * Math.pow(2, this.retryCount) + Math.random() * 1000,
      this.options.maxDelay
    );
    
    this.retryCount++;
    this.options.onReconnecting?.(this.retryCount, delay);

    this.reconnectTimeout = setTimeout(() => {
      this.connect();
    }, delay);
  }

  /** 启动心跳 ping 定时器（30 秒间隔） */
  private startPing(): void {
    this.pingInterval = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.ws.send('ping');
      }
    }, 30000);
  }

  /** 停止心跳 ping 定时器 */
  private stopPing(): void {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
  }

  /** 主动关闭连接并停止所有重连和心跳 */
  public close(): void {
    this.isClosed = true;
    this.stopPing();
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  /** 检查当前 WebSocket 是否处于连接状态 */
  public isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}
