"""Injected-runner Job worker with durable state and lease heartbeat."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
import inspect
from typing import Any, Awaitable, Callable, Mapping, Protocol
from uuid import uuid4

from src.config.security import redact_sensitive_text

from .models import (
    JobRequest,
    JobResourceState,
    JobStatus,
    TERMINAL_JOB_STATUSES,
    utc_timestamp,
)
from .repository import (
    DEFAULT_HEARTBEAT_SECONDS,
    FileJobRepository,
    LeaseConflictError,
)

TargetConcurrencyProvider = Callable[[], int]


class AsyncJobRunner(Protocol):
    """Runner contract intentionally independent from core/control/CLI."""

    def __call__(
        self,
        job_id: str,
        request: JobRequest,
        repository: FileJobRepository,
        target_concurrency_provider: TargetConcurrencyProvider,
        cancel_event: asyncio.Event,
    ) -> Awaitable[Any]: ...


@dataclass(frozen=True)
class JobRunResult:
    """Normalized terminal result returned by a runner."""

    status: JobStatus
    summary: Mapping[str, Any]


class JobWorker:
    """Run one durable Job at a time through an injected async callable."""

    def __init__(
        self,
        repository: FileJobRepository,
        runner: AsyncJobRunner,
        *,
        worker_id: str | None = None,
        heartbeat_interval_seconds: float = DEFAULT_HEARTBEAT_SECONDS,
    ):
        if heartbeat_interval_seconds <= 0:
            raise ValueError("heartbeat_interval_seconds must be positive")
        self.repository = repository
        self.runner = runner
        self.worker_id = worker_id or f"worker-{uuid4()}"
        self.heartbeat_interval_seconds = heartbeat_interval_seconds

    async def run(
        self,
        job_id: str,
        *,
        target_concurrency_provider: TargetConcurrencyProvider | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> JobRunResult:
        provider = target_concurrency_provider or (lambda: 1)
        cancellation = cancel_event or asyncio.Event()
        request = self.repository.get_request(job_id)
        initial_state = self.repository.get_state(job_id)
        if initial_state.status in TERMINAL_JOB_STATUSES:
            raise ValueError(f"job is already terminal: {initial_state.status.value}")

        self.repository.acquire_lease(
            job_id,
            self.worker_id,
            heartbeat_interval_seconds=self.heartbeat_interval_seconds,
        )
        heartbeat_task: asyncio.Task[None] | None = None
        try:
            effective_target = max(1, int(provider()))

            def mark_running(state):
                return replace(
                    state,
                    status=JobStatus.RUNNING,
                    started_at=state.started_at or utc_timestamp(),
                    finished_at=None,
                    last_error=None,
                    resource=JobResourceState(
                        effective_max_in_flight=effective_target,
                        pressure=False,
                        control_status="normal",
                    ),
                )

            self.repository.update_state(job_id, mark_running)
            self.repository.append_event(
                job_id,
                "worker_started",
                payload={"worker_id": self.worker_id},
            )
            heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(job_id, cancellation)
            )
            raw_result = self.runner(
                job_id,
                request,
                self.repository,
                provider,
                cancellation,
            )
            if not inspect.isawaitable(raw_result):
                raise TypeError("runner must return an awaitable")
            result = self._normalize_result(await raw_result)
            if cancellation.is_set() and result.status not in {
                JobStatus.CANCELLED,
                JobStatus.FAILED,
            }:
                result = JobRunResult(JobStatus.CANCELLED, result.summary)
            if result.status not in TERMINAL_JOB_STATUSES | {JobStatus.BLOCKED}:
                raise ValueError(
                    f"runner returned invalid exit status: {result.status.value}"
                )
            self.repository.transition(job_id, result.status)
            self.repository.append_event(
                job_id,
                "worker_finished",
                payload={
                    "status": result.status.value,
                    "summary": dict(result.summary),
                },
            )
            return result
        except asyncio.CancelledError:
            self.repository.transition(job_id, JobStatus.INTERRUPTED)
            self.repository.append_event(
                job_id,
                "worker_interrupted",
                payload={"worker_id": self.worker_id},
            )
            raise
        except Exception as error:
            safe_error = redact_sensitive_text(error)
            self.repository.transition(job_id, JobStatus.FAILED, last_error=safe_error)
            self.repository.append_event(
                job_id,
                "worker_failed",
                payload={"error": safe_error, "worker_id": self.worker_id},
            )
            raise
        finally:
            if heartbeat_task is not None:
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    cancellation.set()
            try:
                self.repository.release_lease(job_id, self.worker_id)
            except LeaseConflictError:
                cancellation.set()

    async def _heartbeat_loop(self, job_id: str, cancel_event: asyncio.Event) -> None:
        while not cancel_event.is_set():
            await asyncio.sleep(self.heartbeat_interval_seconds)
            self.repository.heartbeat_lease(job_id, self.worker_id)

    @staticmethod
    def _normalize_result(value: Any) -> JobRunResult:
        if isinstance(value, JobRunResult):
            return value
        if isinstance(value, JobStatus):
            return JobRunResult(value, {})
        if isinstance(value, str):
            return JobRunResult(JobStatus(value), {})
        if isinstance(value, tuple) and len(value) == 2:
            status, summary = value
            return JobRunResult(JobStatus(status), dict(summary or {}))
        if isinstance(value, Mapping):
            status = JobStatus(value["status"])
            summary = value.get("summary") or {
                key: item for key, item in value.items() if key != "status"
            }
            return JobRunResult(status, dict(summary))
        raise TypeError(f"unsupported runner result: {type(value).__name__}")
