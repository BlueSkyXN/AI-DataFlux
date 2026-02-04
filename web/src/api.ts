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

// WebSocket connection for logs
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
