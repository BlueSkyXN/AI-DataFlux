/** Shared frontend contracts for the Control API v1. */

export interface ApiErrorEnvelope {
  error: {
    code: string;
    message: string;
    details?: unknown;
    request_id: string;
  };
}

export interface HealthResponse {
  status: string;
  version: string;
}

export interface WorkspaceRoot {
  id: string;
  path: string;
}

export interface WorkspaceRootsResponse {
  roots: WorkspaceRoot[];
}

export interface WorkspaceEntry {
  name: string;
  relative_path: string;
  type: 'file' | 'directory';
}

export interface WorkspaceEntriesResponse {
  root_id: string;
  relative_path: string;
  entries: WorkspaceEntry[];
}

export interface WorkspaceSelection {
  rootId: string;
  relativePath: string;
}

export interface ConfigDocument {
  root_id: string;
  relative_path: string;
  content: string;
  revision: number;
  etag: string;
}

export interface ConfigValidateResponse {
  valid: boolean;
  errors: string[];
  warnings: string[];
}

export type JobStatus =
  | 'queued'
  | 'running'
  | 'cancelling'
  | 'interrupted'
  | 'blocked'
  | 'completed'
  | 'completed_with_errors'
  | 'completed_with_unresolved_writes'
  | 'failed'
  | 'cancelled';

export interface JobCounts {
  discovered: number;
  pending: number;
  in_flight: number;
  ai_complete: number;
  persisted: number;
  unresolved_writes: number;
  failed: number;
  cancelled: number;
  retries: number;
}

export interface JobResourceState {
  effective_max_in_flight: number;
  pressure: boolean;
  control_status: string;
}

export interface JobState {
  schema_version: number;
  revision: number;
  job_id: string;
  mode: string;
  config_path: string;
  config_sha256: string;
  status: JobStatus;
  created_at: number;
  updated_at: number;
  started_at: number | null;
  finished_at: number | null;
  counts: JobCounts;
  resource: JobResourceState;
  last_error: string | null;
}

export interface ResourceSnapshot {
  sampled_at?: number;
  status: string;
  pressure: boolean;
  cpu_percent?: number | null;
  memory_percent?: number | null;
  available_memory_bytes?: number | null;
  reasons?: string[];
}

export interface JobListResponse {
  jobs: JobState[];
  resource: ResourceSnapshot;
}

export interface JobEvent {
  seq: number;
  ts: number;
  type: string;
  job_id: string;
  record_id: unknown | null;
  attempt: number | null;
  payload: Record<string, unknown>;
}

export interface JobEventsResponse {
  events: JobEvent[];
  next_seq: number;
}

export type TabType = 'dashboard' | 'jobs' | 'config';
export type LogTarget = 'gateway' | 'process';
