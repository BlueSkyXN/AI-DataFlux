"""Injected async runner tests for JobWorker."""

import asyncio
from pathlib import Path

import pytest

from src.jobs import (
    FileJobRepository,
    JobRunResult,
    JobStatus,
    JobWorker,
    compute_config_hash,
)


def create_job(repository: FileJobRepository):
    request = repository.new_request(
        mode="process",
        config_path="config.yaml",
        config_sha256=compute_config_hash({"a": 1}),
    )
    repository.create_job(request)
    return request


@pytest.mark.asyncio
async def test_worker_runs_injected_callable_and_persists_terminal_state(
    tmp_path: Path,
):
    repository = FileJobRepository(tmp_path / "jobs")
    request = create_job(repository)
    observed = {}

    async def runner(job_id, actual_request, repo, target_provider, cancel_event):
        observed.update(
            job_id=job_id,
            request=actual_request,
            repository=repo,
            target=target_provider(),
            cancelled=cancel_event.is_set(),
        )
        await asyncio.sleep(0)
        return JobRunResult(JobStatus.COMPLETED, {"persisted": 3})

    result = await JobWorker(
        repository, runner, worker_id="worker-test", heartbeat_interval_seconds=0.01
    ).run(request.job_id, target_concurrency_provider=lambda: 3)

    assert result.status == JobStatus.COMPLETED
    assert repository.get_state(request.job_id).status == JobStatus.COMPLETED
    assert repository.get_lease(request.job_id) is None
    assert observed == {
        "job_id": request.job_id,
        "request": request,
        "repository": repository,
        "target": 3,
        "cancelled": False,
    }


@pytest.mark.asyncio
async def test_worker_failure_is_durable_and_reraised(tmp_path: Path):
    repository = FileJobRepository(tmp_path / "jobs")
    request = create_job(repository)

    async def runner(*args):
        raise RuntimeError("runner failed")

    with pytest.raises(RuntimeError, match="runner failed"):
        await JobWorker(repository, runner, worker_id="worker-test").run(request.job_id)

    state = repository.get_state(request.job_id)
    assert state.status == JobStatus.FAILED
    assert state.last_error == "runner failed"
    assert repository.get_lease(request.job_id) is None


@pytest.mark.asyncio
async def test_worker_redacts_secret_shaped_error_text(tmp_path: Path):
    repository = FileJobRepository(tmp_path / "jobs")
    request = create_job(repository)

    async def runner(*_args):
        raise RuntimeError("request failed api_key=super-secret-value")

    with pytest.raises(RuntimeError):
        await JobWorker(repository, runner, worker_id="worker-test").run(request.job_id)

    state = repository.get_state(request.job_id)
    assert "super-secret-value" not in state.last_error
    assert "[REDACTED_SECRET]" in state.last_error
    serialized_events = str(
        [event.to_dict() for event in repository.list_events(request.job_id)]
    )
    assert "super-secret-value" not in serialized_events


@pytest.mark.parametrize(
    ("value", "status", "summary"),
    [
        (JobStatus.COMPLETED, JobStatus.COMPLETED, {}),
        ("cancelled", JobStatus.CANCELLED, {}),
        (("failed", {"reason": "x"}), JobStatus.FAILED, {"reason": "x"}),
        (
            {"status": "completed_with_errors", "failed": 2},
            JobStatus.COMPLETED_WITH_ERRORS,
            {"failed": 2},
        ),
        (
            {"status": "completed", "summary": {"persisted": 1}},
            JobStatus.COMPLETED,
            {"persisted": 1},
        ),
    ],
)
def test_worker_normalizes_supported_runner_results(value, status, summary):
    result = JobWorker._normalize_result(value)
    assert result.status == status
    assert result.summary == summary


def test_worker_rejects_unsupported_runner_result():
    with pytest.raises(TypeError, match="unsupported"):
        JobWorker._normalize_result(123)


@pytest.mark.asyncio
async def test_worker_cooperative_cancel_overrides_completed_result(tmp_path: Path):
    repository = FileJobRepository(tmp_path / "jobs")
    request = create_job(repository)
    cancel_event = asyncio.Event()
    cancel_event.set()

    async def runner(*_args):
        return JobStatus.COMPLETED

    result = await JobWorker(repository, runner).run(
        request.job_id, cancel_event=cancel_event
    )
    assert result.status == JobStatus.CANCELLED
    assert repository.get_state(request.job_id).status == JobStatus.CANCELLED
    with pytest.raises(ValueError, match="terminal"):
        await JobWorker(repository, runner).run(request.job_id)


@pytest.mark.asyncio
async def test_worker_rejects_sync_and_nonterminal_runners(tmp_path: Path):
    repository = FileJobRepository(tmp_path / "jobs")
    sync_request = create_job(repository)

    def sync_runner(*_args):
        return JobStatus.COMPLETED

    with pytest.raises(TypeError, match="awaitable"):
        await JobWorker(repository, sync_runner).run(sync_request.job_id)
    assert repository.get_state(sync_request.job_id).status == JobStatus.FAILED

    queued_request = create_job(repository)

    async def queued_runner(*_args):
        return JobStatus.QUEUED

    with pytest.raises(ValueError, match="invalid exit status"):
        await JobWorker(repository, queued_runner).run(queued_request.job_id)


@pytest.mark.asyncio
async def test_worker_cancelled_task_becomes_interrupted(tmp_path: Path):
    repository = FileJobRepository(tmp_path / "jobs")
    request = create_job(repository)

    async def runner(*_args):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await JobWorker(repository, runner).run(request.job_id)
    assert repository.get_state(request.job_id).status == JobStatus.INTERRUPTED
