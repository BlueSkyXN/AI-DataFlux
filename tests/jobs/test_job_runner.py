from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest

from src.core import job_runner
from src.core.job_tracker import JobRecordTracker
from src.jobs import (
    CommandReceipt,
    FileJobRepository,
    JobCommand,
    JobStatus,
    hash_config_file,
)


def _repository_with_job(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("version: 1\n", encoding="utf-8")
    repository = FileJobRepository(tmp_path / "jobs")
    request = repository.new_request(
        mode="background",
        config_path=str(config_path),
        config_sha256=hash_config_file(config_path),
    )
    repository.create_job(request)
    return repository, request, config_path


@pytest.mark.asyncio
async def test_job_runner_replays_ai_complete_before_scanning(tmp_path, monkeypatch):
    repository, request, _config_path = _repository_with_job(tmp_path)
    tracker = JobRecordTracker(repository, request.job_id)
    tracker.mark_in_flight("record-a", {"input": "hello"})
    tracker.mark_ai_complete("record-a", {"answer": "checkpointed"})
    replayed = []

    class FakeProcessor:
        def __init__(self, _path):
            self.task_manager = SimpleNamespace(
                total_processed_successfully=0,
                max_retries_exceeded_count=0,
                total_estimated=1,
                retried_tasks_count={},
            )

        def configure_job_control(self, **kwargs):
            self.tracker = kwargs["job_tracker"]

        async def persist_checkpoint_results(self, results):
            replayed.append(results)
            for record_id in results:
                self.tracker.mark_persisted(record_id)
            return set(results)

        async def process_shard_async_continuous(self):
            return True

    monkeypatch.setattr(job_runner, "UniversalAIProcessor", FakeProcessor)
    result = await job_runner.run_processing_job(
        request.job_id,
        request,
        repository,
        lambda: 1,
        asyncio.Event(),
    )

    assert result.status == JobStatus.COMPLETED
    assert replayed == [{"record-a": {"answer": "checkpointed"}}]
    assert repository.get_state(request.job_id).counts.persisted == 1


@pytest.mark.asyncio
async def test_changed_config_requires_explicit_accepted_resume_hash(
    tmp_path, monkeypatch
):
    repository, request, config_path = _repository_with_job(tmp_path)
    config_path.write_text("version: 2\n", encoding="utf-8")

    blocked = await job_runner.run_processing_job(
        request.job_id,
        request,
        repository,
        lambda: 1,
        asyncio.Event(),
    )
    assert blocked.status == JobStatus.BLOCKED

    command = JobCommand(
        command_id=str(uuid4()),
        job_id=request.job_id,
        type="resume",
        payload={"accepted_config_sha256": hash_config_file(config_path)},
    )
    repository.add_command(command)
    repository.save_command_receipt(
        CommandReceipt(command.command_id, request.job_id, accepted=True)
    )

    class EmptyProcessor:
        def __init__(self, _path):
            self.task_manager = SimpleNamespace(
                total_processed_successfully=0,
                max_retries_exceeded_count=0,
                total_estimated=0,
                retried_tasks_count={},
            )

        def configure_job_control(self, **_kwargs):
            return None

        async def process_shard_async_continuous(self):
            return True

    monkeypatch.setattr(job_runner, "UniversalAIProcessor", EmptyProcessor)
    accepted = await job_runner.run_processing_job(
        request.job_id,
        request,
        repository,
        lambda: 1,
        asyncio.Event(),
    )
    assert accepted.status == JobStatus.COMPLETED
