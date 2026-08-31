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
    accepted_hashes = {request.config_sha256}
    for command in repository.list_commands(job_id):
        if command.type != "resume":
            continue
        receipt = repository.get_command_receipt(job_id, command.command_id)
        if receipt is None or not receipt.accepted:
            continue
        accepted_hash = command.payload.get("accepted_config_sha256")
        if isinstance(accepted_hash, str):
            accepted_hashes.add(accepted_hash)
    if current_hash not in accepted_hashes:
        return JobRunResult(
            JobStatus.BLOCKED,
            {"reason": "config_changed", "config_path": str(config_path)},
        )

    tracker = JobRecordTracker(repository, job_id)
    processor = UniversalAIProcessor(str(config_path))
    processor.configure_job_control(
        cancel_event=cancel_event,
        target_concurrency_provider=target_concurrency_provider,
        job_tracker=tracker,
    )
    replayable = tracker.replayable_results()
    if replayable:
        repository.append_event(
            job_id,
            "checkpoint_replay_started",
            payload={"record_count": len(replayable)},
        )
        await processor.persist_checkpoint_results(replayable)
    completed = await processor.process_shard_async_continuous()
    manager = processor.task_manager
    checkpoint_counts = tracker.counts()
    persisted = checkpoint_counts.persisted
    failed = checkpoint_counts.failed
    discovered = max(manager.total_estimated, checkpoint_counts.discovered)
    pending = max(
        checkpoint_counts.pending,
        discovered - persisted - failed - checkpoint_counts.in_flight,
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
                failed=failed,
                cancelled=cancelled,
                retries=retries,
            ),
        )

    repository.update_state(job_id, update_counts)
    if cancel_event.is_set() or not completed:
        return JobRunResult(JobStatus.CANCELLED, {"persisted": persisted})
    status = JobStatus.COMPLETED_WITH_ERRORS if failed else JobStatus.COMPLETED
    return JobRunResult(
        status,
        {
            "discovered": discovered,
            "persisted": persisted,
            "failed": failed,
            "retries": retries,
        },
    )
