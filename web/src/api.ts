// API service for communicating with the Control Server

import type { StatusResponse, ConfigResponse, ConfigWriteResponse, ManagedProcessStatus } from './types';

const API_BASE = '';  // Use relative URLs (proxied in dev, served by same host in prod)

export async function fetchStatus(): Promise<StatusResponse> {
  const response = await fetch(`${API_BASE}/api/status`);
  if (!response.ok) {
    throw new Error(`Failed to fetch status: ${response.status}`);
  }
  return response.json();
}

export async function fetchConfig(path: string): Promise<ConfigResponse> {
  const response = await fetch(`${API_BASE}/api/config?path=${encodeURIComponent(path)}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch config: ${response.status}`);
  }
  return response.json();
}

export async function saveConfig(path: string, content: string): Promise<ConfigWriteResponse> {
  const response = await fetch(`${API_BASE}/api/config`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path, content }),
  });
  if (!response.ok) {
    throw new Error(`Failed to save config: ${response.status}`);
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
    headers: { 'Content-Type': 'application/json' },
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
  });
  if (!response.ok) {
    throw new Error(`Failed to stop gateway: ${response.status}`);
  }
  return response.json();
}

export async function startProcess(configPath: string = 'config.yaml'): Promise<ManagedProcessStatus> {
  const response = await fetch(`${API_BASE}/api/process/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
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
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/api/logs?target=${target}`;
  const ws = new WebSocket(wsUrl);
  
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

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/api/logs?target=${this.options.target}`;
    
    try {
      this.ws = new WebSocket(wsUrl);
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
