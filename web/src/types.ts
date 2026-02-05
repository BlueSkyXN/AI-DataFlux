// API response types

export interface ManagedProcessStatus {
  status: 'stopped' | 'running' | 'exited';
  pid: number | null;
  start_time: number | null;
  exit_code: number | null;
  config_path: string | null;
  port?: number | null;
}

export interface GatewayHealth {
  status: 'healthy' | 'degraded' | 'unhealthy';
  available_models: number;
  total_models: number;
  uptime: number;
}

export interface ProcessProgress {
  total: number;
  processed: number;
  active: number;
  shard: string;
  errors: number;
  ts: number;
}

export interface StatusResponse {
  gateway: {
    managed: ManagedProcessStatus;
    health?: GatewayHealth;
  };
  process: {
    managed: ManagedProcessStatus;
    progress?: ProcessProgress;
  };
}

export interface ConfigResponse {
  path: string;
  content: string;
}

export interface ConfigWriteResponse {
  success: boolean;
  path: string;
  backed_up: boolean;
}

export interface ConfigValidateResponse {
  valid: boolean;
  error?: string;
}

export type TabType = 'dashboard' | 'config' | 'logs';
export type LogTarget = 'gateway' | 'process';
