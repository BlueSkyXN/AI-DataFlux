"""Adapter that executes UniversalAIProcessor through the durable Job API."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

from ..config import execution_config_hash, load_config
from ..jobs import (
    FileJobRepository,
    JobCounts,
    JobRequest,
    JobRunResult,
    JobStatus,
)
from .processor import UniversalAIProcessor
from .job_tracker import JobRecordTracker


def resolve_terminal_status(
    *,
    cancelled: bool,
    job_failed: bool,
    unresolved_writes: int,
    failed_records: int,
) -> JobStatus:
    """Apply the v4 terminal-state priority without conflating record failures."""

    if cancelled:
        return JobStatus.CANCELLED
    if job_failed:
        return JobStatus.FAILED
    if unresolved_writes:
        return JobStatus.COMPLETED_WITH_UNRESOLVED_WRITES
    if failed_records:
        return JobStatus.COMPLETED_WITH_ERRORS
    return JobStatus.COMPLETED


async def run_processing_job(
    job_id: str,
    request: JobRequest,
    repository: FileJobRepository,
    target_concurrency_provider,
    cancel_event: asyncio.Event,
) -> JobRunResult:
    """Run one processing config and persist trustworthy terminal counters."""

    config_path = Path(request.config_path)
    if not config_path.is_file():
        return JobRunResult(
            JobStatus.BLOCKED,
            {"reason": "config_missing", "config_path": str(config_path)},
        )
    current_hash = execution_config_hash(load_config(config_path), config_path)
    if current_hash != repository.effective_config_hash(job_id):
        return JobRunResult(
            JobStatus.BLOCKED,
            {"reason": "config_changed", "config_path": str(config_path)},
        )

    tracker = JobRecordTracker(repository, job_id)
    pending_commits = tracker.pending_prepared_results()
    initialization = asyncio.create_task(
        asyncio.to_thread(UniversalAIProcessor, str(config_path))
    )
    try:
        processor = await asyncio.shield(initialization)
    except asyncio.CancelledError:
        # 线程不能被安全强杀：等待构造结束并释放 adapter，不遗留后台连接。
        processor = await initialization
        await processor.task_pool.aclose()
        raise
    processor.configure_job_control(
        cancel_event=cancel_event,
        target_concurrency_provider=target_concurrency_provider,
        job_tracker=tracker,
    )
    try:
        if pending_commits:
            repository.append_event(
                job_id,
                "checkpoint_reconciliation_started",
                payload={"record_count": len(pending_commits)},
            )
            await processor.reconcile_checkpoint_results(pending_commits)
        completed = await processor.process_shard_async_continuous()
    finally:
        await processor.task_pool.aclose()
    manager = processor.task_manager
    checkpoint_counts = tracker.counts()
    persisted = checkpoint_counts.persisted
    failed = checkpoint_counts.failed
    unresolved_writes = checkpoint_counts.unresolved_writes
    discovered = max(manager.total_estimated, checkpoint_counts.discovered)
    pending = max(
        checkpoint_counts.pending,
        discovered
        - persisted
        - failed
        - unresolved_writes
        - checkpoint_counts.in_flight,
    )
    retries = checkpoint_counts.retries
    cancelled = pending if cancel_event.is_set() or not completed else 0

    def update_counts(state):
        return replace(
            state,
            counts=JobCounts(
                discovered=discovered,
                pending=pending,
                in_flight=checkpoint_counts.in_flight,
                ai_complete=checkpoint_counts.ai_complete,
                persisted=persisted,
                unresolved_writes=unresolved_writes,
                failed=failed,
                cancelled=cancelled,
                retries=retries,
            ),
        )

    repository.update_state(job_id, update_counts)
    status = resolve_terminal_status(
        cancelled=cancel_event.is_set() or not completed,
        job_failed=False,
        unresolved_writes=unresolved_writes,
        failed_records=failed,
    )
    return JobRunResult(
        status,
        {
            "discovered": discovered,
            "persisted": persisted,
            "unresolved_writes": unresolved_writes,
            "failed": failed,
            "retries": retries,
        },
    )
