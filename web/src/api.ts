/** Authenticated client for the Control API v1. */

import type {
  ApiErrorEnvelope,
  ConfigDocument,
  ConfigValidateResponse,
  HealthResponse,
  JobEvent,
  JobEventsResponse,
  JobListResponse,
  JobState,
  WorkspaceEntriesResponse,
  WorkspaceRootsResponse,
  WorkspaceSelection,
} from './types';

const API_V1 = '/api/v1';
const CONTROL_TOKEN_SESSION_KEY = 'dataflux-control-token';
let controlTokenCache: string | null = null;

function getControlToken(): string {
  if (controlTokenCache !== null) return controlTokenCache;
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
    return hashToken;
  }
  controlTokenCache = (
    window.sessionStorage.getItem(CONTROL_TOKEN_SESSION_KEY) ?? ''
  ).trim();
  return controlTokenCache;
}

function withAuthHeaders(headers: HeadersInit = {}): Headers {
  const result = new Headers(headers);
  const token = getControlToken();
  if (token) result.set('Authorization', `Bearer ${token}`);
  return result;
}

async function apiError(response: Response): Promise<Error> {
  let message = `Request failed: ${response.status}`;
  try {
    const envelope = (await response.json()) as ApiErrorEnvelope;
    if (envelope.error?.message) {
      message = envelope.error.request_id
        ? `${envelope.error.message} (${envelope.error.request_id})`
        : envelope.error.message;
    }
  } catch {
    // Keep the status-based fallback when a proxy returns a non-JSON error.
  }
  return new Error(message);
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: withAuthHeaders(init?.headers),
  });
  if (!response.ok) throw await apiError(response);
  return response.json() as Promise<T>;
}

export async function fetchHealth(): Promise<HealthResponse> {
  const response = await fetch('/health');
  if (!response.ok) throw await apiError(response);
  return response.json() as Promise<HealthResponse>;
}

export function fetchWorkspaceRoots(): Promise<WorkspaceRootsResponse> {
  return requestJson(`${API_V1}/workspace/roots`);
}

export function fetchWorkspaceEntries(
  rootId: string,
  relativePath: string = '',
): Promise<WorkspaceEntriesResponse> {
  const params = new URLSearchParams({ root_id: rootId, relative_path: relativePath });
  return requestJson(`${API_V1}/workspace/entries?${params}`);
}

export async function fetchConfig(
  selection: WorkspaceSelection,
): Promise<ConfigDocument> {
  const params = new URLSearchParams({
    root_id: selection.rootId,
    relative_path: selection.relativePath,
  });
  const response = await fetch(`${API_V1}/config?${params}`, {
    headers: withAuthHeaders(),
  });
  if (!response.ok) throw await apiError(response);
  const body = (await response.json()) as Omit<ConfigDocument, 'etag'>;
  return { ...body, etag: response.headers.get('ETag') ?? '' };
}

export async function saveConfig(
  selection: WorkspaceSelection,
  content: string,
  etag: string,
): Promise<ConfigDocument> {
  const response = await fetch(`${API_V1}/config`, {
    method: 'PUT',
    headers: withAuthHeaders({
      'Content-Type': 'application/json',
      'If-Match': etag,
    }),
    body: JSON.stringify({
      root_id: selection.rootId,
      relative_path: selection.relativePath,
      content,
    }),
  });
  if (!response.ok) throw await apiError(response);
  const body = (await response.json()) as Omit<ConfigDocument, 'etag'>;
  return { ...body, etag: response.headers.get('ETag') ?? '' };
}

export function validateConfig(
  content: string,
  selection?: WorkspaceSelection,
): Promise<ConfigValidateResponse> {
  return requestJson(`${API_V1}/config/validate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      content,
      root_id: selection?.rootId,
      relative_path: selection?.relativePath,
    }),
  });
}

export function submitJob(
  selection: WorkspaceSelection,
  options?: Record<string, unknown>,
): Promise<JobState> {
  return requestJson(`${API_V1}/jobs`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      root_id: selection.rootId,
      relative_path: selection.relativePath,
      ...(options ? { options } : {}),
    }),
  });
}

export function fetchJobs(): Promise<JobListResponse> {
  return requestJson(`${API_V1}/jobs`);
}

export function fetchJob(jobId: string): Promise<JobState> {
  return requestJson(`${API_V1}/jobs/${encodeURIComponent(jobId)}`);
}

export function cancelJob(jobId: string): Promise<JobState> {
  return requestJson(`${API_V1}/jobs/${encodeURIComponent(jobId)}/cancel`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: '{}',
  });
}

export function resumeJob(jobId: string): Promise<JobState> {
  return requestJson(`${API_V1}/jobs/${encodeURIComponent(jobId)}/resume`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: '{}',
  });
}

export function fetchJobEvents(
  jobId: string,
  afterSeq: number = 0,
  limit: number = 200,
): Promise<JobEventsResponse> {
  const params = new URLSearchParams({ after_seq: String(afterSeq), limit: String(limit) });
  return requestJson(
    `${API_V1}/jobs/${encodeURIComponent(jobId)}/events?${params}`,
  );
}

export interface JobEventStreamOptions {
  afterSeq?: number;
  onEvent: (event: JobEvent) => void;
  onError?: (error: Error) => void;
}

/** Authenticated SSE over fetch; browser EventSource cannot set Bearer headers. */
export function streamJobEvents(
  jobId: string,
  options: JobEventStreamOptions,
): () => void {
  const controller = new AbortController();
  const params = new URLSearchParams({ after_seq: String(options.afterSeq ?? 0) });
  void (async () => {
    try {
      const response = await fetch(
        `${API_V1}/jobs/${encodeURIComponent(jobId)}/events/stream?${params}`,
        {
          headers: withAuthHeaders({ Accept: 'text/event-stream' }),
          signal: controller.signal,
        },
      );
      if (!response.ok) throw await apiError(response);
      if (!response.body) throw new Error('Event stream is unavailable');
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      while (true) {
        const { value, done } = await reader.read();
        buffer += decoder.decode(value, { stream: !done });
        const blocks = buffer.split(/\r?\n\r?\n/);
        buffer = blocks.pop() ?? '';
        for (const block of blocks) {
          const data = block
            .split(/\r?\n/)
            .filter((line) => line.startsWith('data:'))
            .map((line) => line.slice(5).trimStart())
            .join('\n');
          if (data) options.onEvent(JSON.parse(data) as JobEvent);
        }
        if (done) break;
      }
    } catch (error) {
      if (!controller.signal.aborted) {
        options.onError?.(error instanceof Error ? error : new Error(String(error)));
      }
    }
  })();
  return () => controller.abort();
}

export async function testFeishuConnection(
  appId: string,
  appSecret: string,
): Promise<{ success: boolean; message: string; token_preview?: string }> {
  return requestJson(`${API_V1}/feishu/test-connection`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ app_id: appId, app_secret: appSecret }),
  });
}

function encodeTokenForWsProtocol(token: string): string {
  const bytes = new TextEncoder().encode(token);
  let binary = '';
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return window.btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/g, '');
}

export interface AutoReconnectOptions {
  target: 'gateway' | 'process';
  onMessage: (line: string) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
  onReconnecting?: (attempt: number, delay: number) => void;
  maxRetries?: number;
  initialDelay?: number;
  maxDelay?: number;
}

/** Compatibility log transport, migrated to the versioned route. */
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
    const token = getControlToken();
    const protocols = token ? [`dataflux-token-b64.${encodeTokenForWsProtocol(token)}`] : undefined;
    try {
      this.ws = new WebSocket(
        `${protocol}//${window.location.host}${API_V1}/logs?target=${this.options.target}`,
        protocols,
      );
    } catch {
      this.scheduleReconnect();
      return;
    }
    this.ws.onopen = () => {
      this.retryCount = 0;
      this.options.onConnect();
      this.startPing();
    };
    this.ws.onmessage = (event) => {
      if (event.data !== 'pong') this.options.onMessage(event.data);
    };
    this.ws.onclose = () => {
      this.stopPing();
      this.options.onDisconnect();
      if (!this.isClosed) this.scheduleReconnect();
    };
  }

  private scheduleReconnect(): void {
    if (this.isClosed) return;
    if (this.options.maxRetries >= 0 && this.retryCount >= this.options.maxRetries) return;
    const delay = Math.min(
      this.options.initialDelay * 2 ** this.retryCount + Math.random() * 1000,
      this.options.maxDelay,
    );
    this.retryCount += 1;
    this.options.onReconnecting(this.retryCount, delay);
    this.reconnectTimeout = setTimeout(() => this.connect(), delay);
  }

  private startPing(): void {
    this.pingInterval = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) this.ws.send('ping');
    }, 30000);
  }

  private stopPing(): void {
    if (this.pingInterval) clearInterval(this.pingInterval);
    this.pingInterval = null;
  }

  close(): void {
    this.isClosed = true;
    this.stopPing();
    if (this.reconnectTimeout) clearTimeout(this.reconnectTimeout);
    this.reconnectTimeout = null;
    this.ws?.close();
    this.ws = null;
  }

  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}
