// API service for communicating with the Control Server

import type {
  StatusResponse,
  ConfigResponse,
  ConfigWriteResponse,
  ConfigValidateResponse,
  ManagedProcessStatus,
} from './types';

const API_BASE = '';  // Use relative URLs (proxied in dev, served by same host in prod)
const CONTROL_TOKEN_SESSION_KEY = 'dataflux-control-token';
let controlTokenCache: string | null = null;

function getControlToken(): string {
  if (controlTokenCache !== null) {
    return controlTokenCache;
  }

  const hash = window.location.hash.startsWith('#')
    ? window.location.hash.slice(1)
    : window.location.hash;
  const params = new URLSearchParams(hash);
  const hashToken = (params.get('token') ?? '').trim();
  if (hashToken) {
    controlTokenCache = hashToken;
    window.sessionStorage.setItem(CONTROL_TOKEN_SESSION_KEY, hashToken);
    params.delete('token');
    const nextHash = params.toString();
    const nextUrl = `${window.location.pathname}${window.location.search}${nextHash ? `#${nextHash}` : ''}`;
    window.history.replaceState({}, '', nextUrl);
    return controlTokenCache;
  }

  const sessionToken = (window.sessionStorage.getItem(CONTROL_TOKEN_SESSION_KEY) ?? '').trim();
  controlTokenCache = sessionToken;
  return controlTokenCache;
}

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

function buildLogsWsUrl(target: 'gateway' | 'process'): string {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}/api/logs?target=${target}`;
}

function encodeTokenForWsProtocol(token: string): string {
  const bytes = new TextEncoder().encode(token);
  let binary = '';
  for (const byte of bytes) {
    binary += String.fromCharCode(byte);
  }
  const base64 = window.btoa(binary);
  return base64.replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/g, '');
}

function buildLogsWsProtocols(): string[] | undefined {
  const token = getControlToken();
  if (!token) {
    return undefined;
  }
  return [`dataflux-token-b64.${encodeTokenForWsProtocol(token)}`];
}

export async function fetchStatus(): Promise<StatusResponse> {
  const response = await fetch(`${API_BASE}/api/status`, {
    headers: withAuthHeaders(),
  });
  if (!response.ok) {
    throw new Error(`Failed to fetch status: ${response.status}`);
  }
  return response.json();
}

export async function fetchConfig(path: string): Promise<ConfigResponse> {
  const response = await fetch(`${API_BASE}/api/config?path=${encodeURIComponent(path)}`, {
    headers: withAuthHeaders(),
  });
  if (!response.ok) {
    throw new Error(`Failed to fetch config: ${response.status}`);
  }
  return response.json();
}

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
  
  // Keep-alive ping
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

// Auto-reconnecting WebSocket connection with exponential backoff
export interface AutoReconnectOptions {
  target: 'gateway' | 'process';
  onMessage: (line: string) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
  onReconnecting?: (attempt: number, delay: number) => void;
  maxRetries?: number;  // -1 for infinite
  initialDelay?: number;  // ms
  maxDelay?: number;  // ms
}

export class AutoReconnectWebSocket {
  private ws: WebSocket | null = null;
  private pingInterval: ReturnType<typeof setInterval> | null = null;
  private reconnectTimeout: ReturnType<typeof setTimeout> | null = null;
  private retryCount = 0;
  private isClosed = false;
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
      if (event.data !== 'pong') {
        this.options.onMessage(event.data);
      }
    };

    this.ws.onerror = () => {
      // Error will be followed by close, so we handle reconnect there
    };

    this.ws.onclose = () => {
      this.stopPing();
      this.options.onDisconnect?.();
      if (!this.isClosed) {
        this.scheduleReconnect();
      }
    };
  }

  private scheduleReconnect(): void {
    if (this.isClosed) return;
    if (this.options.maxRetries >= 0 && this.retryCount >= this.options.maxRetries) {
      return;
    }

    // Exponential backoff with jitter
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

  private startPing(): void {
    this.pingInterval = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.ws.send('ping');
      }
    }, 30000);
  }

  private stopPing(): void {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
  }

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

  public isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}
