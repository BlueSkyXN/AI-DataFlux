from __future__ import annotations

import asyncio
import time
from uuid import uuid4

import pytest
import yaml

from src.control.job_service import JobService
from src.jobs import JobCommand, JobStatus, ResourceSnapshot


def _config(tmp_path):
    input_path = tmp_path / "input.csv"
    input_path.write_text("input,result\nhello,\n", encoding="utf-8")
    config = {
        "global": {
            "log": {"level": "error", "format": "text", "output": "console"},
            "flux_api_url": "http://127.0.0.1:8787",
        },
        "datasource": {
            "type": "csv",
            "engine": "pandas",
            "concurrency": {"batch_size": 1, "max_in_flight": 1},
        },
        "csv": {"input_path": str(input_path), "output_path": str(input_path)},
        "columns_to_extract": ["input"],
        "columns_to_write": {"result": "result"},
        "prompt": {"template": "{input}"},
        "workspace": {
            "roots": {"project": str(tmp_path)},
            "state_dir": ".dataflux/jobs",
        },
        "server": {"token": "test-token"},
        "scheduler": {
            "max_active_jobs": 1,
            "sample_interval_seconds": 0.01,
        },
        "models": [],
        "channels": {},
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
    assert (
        control_service.repository.get_command_receipt(state.job_id, command.command_id)
        is None
    )

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
    config_path.write_text(
        config_path.read_text(encoding="utf-8") + "\n# changed\n",
        encoding="utf-8",
    )

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
