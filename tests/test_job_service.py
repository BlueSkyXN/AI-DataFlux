from __future__ import annotations

import asyncio
import time
from uuid import uuid4

import pytest
import yaml

from src.control.job_service import JobService
from src.jobs import JobCommand, JobStatus, ResourceSnapshot
from src.jobs import SupervisorConflictError


def _config(tmp_path):
    input_path = tmp_path / "input.csv"
    input_path.write_text("input,result\nhello,\n", encoding="utf-8")
    config = {
        "schema_version": 4,
        "runtime": {
            "log": {"level": "error", "format": "text", "output": "console"},
            "auth": {"token": "test-token"},
            "workspace": {
                "roots": {"project": str(tmp_path)},
                "state_dir": ".dataflux/jobs",
            },
            "scheduler": {
                "max_active_jobs": 1,
                "sample_interval_seconds": 0.01,
            },
        },
        "job": {
            "gateway_url": "http://127.0.0.1:8787",
            "datasource": {
                "type": "csv",
                "input_path": str(input_path),
                "output_path": str(input_path),
                "engine": "pandas",
                "require_all_input_fields": True,
            },
            "columns": {"extract": ["input"], "write": {"result": "result"}},
            "prompt": {"template": "{input}"},
            "concurrency": {"batch_size": 1, "max_in_flight": 1},
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_standalone_worker_discovers_jobs_created_after_start(tmp_path):
    config_path = _config(tmp_path)
    worker_service = JobService(str(config_path))
    control_service = JobService(str(config_path))

    state = control_service.submit(root_id="project", relative_path="config.yaml")
    assert state.job_id not in worker_service.scheduler.queued_job_ids

    worker_service._discover_queued_jobs()
    assert worker_service.scheduler.queued_job_ids == (state.job_id,)


def test_invalid_persisted_max_in_flight_blocks_only_that_job(tmp_path):
    config_path = _config(tmp_path)
    service = JobService(str(config_path))
    request = service.repository.new_request(
        mode="background",
        config_path=str(config_path),
        config_sha256="a" * 64,
        options={"max_in_flight": "invalid"},
    )
    state = service.repository.create_job(request)

    service._discover_queued_jobs()

    blocked = service.repository.get_state(state.job_id)
    assert blocked.status == JobStatus.BLOCKED
    assert blocked.last_error == "invalid_job_options"


@pytest.mark.parametrize(
    "options",
    [
        {"max_in_flight": 0},
        {"max_in_flight": 10_001},
        {"max_in_flight": "10"},
        {"unknown": 1},
    ],
)
def test_submit_rejects_invalid_job_options(tmp_path, options):
    service = JobService(str(_config(tmp_path)))

    with pytest.raises(ValueError, match="Job options|max_in_flight"):
        service.submit(
            root_id="project",
            relative_path="config.yaml",
            options=options,
        )


def test_durable_cancel_command_reaches_separate_worker(tmp_path):
    config_path = _config(tmp_path)
    control_service = JobService(str(config_path))
    worker_service = JobService(str(config_path))
    state = control_service.submit(root_id="project", relative_path="config.yaml")
    control_service.repository.transition(state.job_id, JobStatus.RUNNING)

    cancelling = control_service.cancel(state.job_id)
    command = control_service.repository.list_commands(state.job_id)[-1]
    assert cancelling.status == JobStatus.CANCELLING
    assert control_service.repository.get_command_receipt(
        state.job_id, command.command_id
    ).accepted

    worker_service._consume_commands()
    receipt = worker_service.repository.get_command_receipt(
        state.job_id, command.command_id
    )
    assert receipt is not None and receipt.accepted
    assert worker_service._cancel_events[state.job_id].is_set()


@pytest.mark.asyncio
async def test_recovery_blocks_changed_config_and_respects_fresh_lease(tmp_path):
    config_path = _config(tmp_path)
    service = JobService(str(config_path))
    changed = service.submit(root_id="project", relative_path="config.yaml")
    service.repository.transition(changed.job_id, JobStatus.RUNNING)
    changed_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    changed_config["job"]["concurrency"]["batch_size"] = 2
    config_path.write_text(yaml.safe_dump(changed_config), encoding="utf-8")

    await service.recover()
    assert service.repository.get_state(changed.job_id).status == JobStatus.BLOCKED

    fresh = service.submit(root_id="project", relative_path="config.yaml")
    service.repository.transition(fresh.job_id, JobStatus.RUNNING)
    service.repository.acquire_lease(fresh.job_id, "other-worker")
    await service.recover()
    assert service.repository.get_state(fresh.job_id).status == JobStatus.RUNNING


def test_command_consumer_rejects_unknown_and_cancels_discovered_queue(tmp_path):
    config_path = _config(tmp_path)
    service = JobService(str(config_path))
    unknown_job = service.submit(root_id="project", relative_path="config.yaml")
    unknown = JobCommand(
        command_id=str(uuid4()), job_id=unknown_job.job_id, type="unknown"
    )
    service.repository.add_command(unknown)
    service._consume_commands()
    unknown_receipt = service.repository.get_command_receipt(
        unknown_job.job_id, unknown.command_id
    )
    assert unknown_receipt is not None and not unknown_receipt.accepted

    queued_job = service.submit(root_id="project", relative_path="config.yaml")
    cancel = JobCommand(
        command_id=str(uuid4()), job_id=queued_job.job_id, type="cancel"
    )
    service.repository.add_command(cancel)
    service._consume_commands()
    assert service.repository.get_state(queued_job.job_id).status == JobStatus.CANCELLED
    assert service.repository.get_command_receipt(
        queued_job.job_id, cancel.command_id
    ).accepted


def test_job_service_event_limits_and_resource_status(tmp_path):
    config_path = _config(tmp_path)
    service = JobService(str(config_path))
    state = service.submit(root_id="project", relative_path="config.yaml")
    events, cursor = service.list_events(state.job_id, after_seq=0, limit=1)
    assert len(events) == 1 and cursor == events[0]["seq"]
    with pytest.raises(ValueError, match="limit"):
        service.list_events(state.job_id, limit=0)
    resource = service.resource_status()
    assert resource["max_active_jobs"] == 1
    assert state.job_id in resource["queued_jobs"]


@pytest.mark.asyncio
async def test_run_loop_admits_and_releases_job_with_injected_worker(
    tmp_path, monkeypatch
):
    config_path = _config(tmp_path)
    service = JobService(str(config_path))
    state = service.submit(root_id="project", relative_path="config.yaml")
    snapshot = ResourceSnapshot(sampled_at=time.time(), status="normal", pressure=False)
    monkeypatch.setattr(service.scheduler.probe, "sample", lambda **_kwargs: snapshot)

    class FakeWorker:
        def __init__(self, repository, _runner):
            self.repository = repository

        async def run(self, job_id, **_kwargs):
            self.repository.transition(job_id, JobStatus.COMPLETED)
            return None

    monkeypatch.setattr("src.control.job_service.JobWorker", FakeWorker)
    loop_task = asyncio.create_task(service.run_loop())
    for _ in range(50):
        if service.repository.get_state(state.job_id).status == JobStatus.COMPLETED:
            break
        await asyncio.sleep(0.01)
    await service.stop()
    await loop_task

    assert service.repository.get_state(state.job_id).status == JobStatus.COMPLETED
    assert state.job_id not in service.scheduler.active_job_ids


@pytest.mark.asyncio
async def test_supervisor_ownership_is_distinct_from_control_mutations(tmp_path):
    path = _config(tmp_path)
    first, second = JobService(str(path)), JobService(str(path))
    loop = await first.start()
    try:
        with pytest.raises(SupervisorConflictError):
            await second.start()
        state = second.submit(root_id="project", relative_path="config.yaml")
        assert second.cancel(state.job_id).status == JobStatus.CANCELLED
    finally:
        await first.stop()
        await loop
    replacement = JobService(str(path))
    loop = await replacement.start()
    await replacement.stop()
    await loop


@pytest.mark.asyncio
async def test_supervisor_shutdown_interrupts_instead_of_user_cancelling(
    tmp_path, monkeypatch
):
    service = JobService(str(_config(tmp_path)))
    state = service.submit(root_id="project", relative_path="config.yaml")
    started = asyncio.Event()

    async def runner(*_args):
        started.set()
        await asyncio.sleep(10)

    monkeypatch.setattr("src.control.job_service.run_processing_job", runner)
    monkeypatch.setattr(
        service.scheduler.probe,
        "sample",
        lambda **_: ResourceSnapshot(
            sampled_at=time.time(), status="normal", pressure=False
        ),
    )
    loop = await service.start()
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
    finally:
        await service.stop()
        await loop
    assert service.repository.get_state(state.job_id).status == JobStatus.INTERRUPTED
    assert service.repository.get_lease(state.job_id) is None


@pytest.mark.asyncio
async def test_accepted_cancel_converges_after_supervisor_crash(tmp_path):
    path = _config(tmp_path)
    first = JobService(str(path))
    state = first.submit(root_id="project", relative_path="config.yaml")
    first.repository.transition(state.job_id, JobStatus.RUNNING)
    first.cancel(state.job_id)
    recovered = JobService(str(path))
    await recovered.recover()
    assert recovered.repository.get_state(state.job_id).status == JobStatus.CANCELLED
    assert state.job_id not in recovered.scheduler.queued_job_ids


def test_cancel_state_and_receipt_share_one_recovery_commit(tmp_path, monkeypatch):
    import src.jobs.repository as module

    service = JobService(str(_config(tmp_path)))
    state = service.submit(root_id="project", relative_path="config.yaml")
    original = module.atomic_write_json

    def write(path, payload):
        if path == service.repository.state_path(state.job_id):
            raise OSError("command state fault")
        original(path, payload)

    with monkeypatch.context() as patch:
        patch.setattr(module, "atomic_write_json", write)
        with pytest.raises(OSError, match="command state fault"):
            service.cancel(state.job_id)
    command = service.repository.list_commands(state.job_id)[0]
    assert service.repository.get_state(state.job_id).status == JobStatus.QUEUED
    assert (
        service.repository.get_command_receipt(state.job_id, command.command_id) is None
    )
    service._consume_commands()
    assert service.repository.get_state(state.job_id).status == JobStatus.CANCELLED
    assert service.repository.get_command_receipt(
        state.job_id, command.command_id
    ).accepted


@pytest.mark.asyncio
async def test_resumed_config_hash_remains_valid_during_crash_recovery(tmp_path):
    path = _config(tmp_path)
    service = JobService(str(path))
    state = service.submit(root_id="project", relative_path="config.yaml")
    service.repository.transition(state.job_id, JobStatus.FAILED)
    config = yaml.safe_load(path.read_text())
    config["job"]["prompt"]["template"] = "new {record_json}"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    service.resume(state.job_id)
    service.repository.transition(state.job_id, JobStatus.INTERRUPTED)
    recovered = JobService(str(path))
    await recovered.recover()
    assert recovered.repository.get_state(state.job_id).status == JobStatus.QUEUED


def test_rejected_command_does_not_change_terminal_finish_time(tmp_path):
    service = JobService(str(_config(tmp_path)))
    state = service.submit(root_id="project", relative_path="config.yaml")
    completed = service.repository.transition(state.job_id, JobStatus.COMPLETED)
    with pytest.raises(ValueError, match="不能取消"):
        service.cancel(state.job_id)
    actual = service.repository.get_state(state.job_id)
    assert actual.status == JobStatus.COMPLETED
    assert actual.finished_at == completed.finished_at


@pytest.mark.asyncio
async def test_running_supervisor_retries_recovery_when_old_lease_expires(
    tmp_path, monkeypatch
):
    from dataclasses import replace
    from src.jobs import atomic_write_json

    service = JobService(str(_config(tmp_path)))
    state = service.submit(root_id="project", relative_path="config.yaml")
    service.repository.transition(state.job_id, JobStatus.RUNNING)
    lease = service.repository.acquire_lease(state.job_id, "old-supervisor")
    started = asyncio.Event()

    async def runner(*args):
        started.set()
        return JobStatus.COMPLETED

    monkeypatch.setattr("src.control.job_service.run_processing_job", runner)
    monkeypatch.setattr(
        service.scheduler.probe,
        "sample",
        lambda **_: ResourceSnapshot(
            sampled_at=time.time(), status="normal", pressure=False
        ),
    )
    loop = await service.start()
    try:
        assert not started.is_set()
        atomic_write_json(
            service.repository.lease_path(state.job_id),
            replace(lease, heartbeat_at=0).to_dict(),
        )
        await asyncio.wait_for(started.wait(), timeout=2)
        for _ in range(20):
            if service.repository.get_state(state.job_id).status == JobStatus.COMPLETED:
                break
            await asyncio.sleep(0.01)
        assert service.repository.get_state(state.job_id).status == JobStatus.COMPLETED
    finally:
        await service.stop()
        await loop
