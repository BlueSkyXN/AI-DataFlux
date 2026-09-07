"""H2.1：独立格式版本、原子发布及跨进程 revision 合同。"""

from dataclasses import replace
import json
import multiprocessing
import os
from pathlib import Path
import time

import pytest

from src.jobs import (
    FileJobRepository,
    ImmutableRecordError,
    JobRepositoryError,
    JobRequest,
    JobStatus,
    LeaseConflictError,
    RevisionConflictError,
    compute_config_hash,
)
from src.jobs.repository import _exclusive_file_lock


def _new_request(repository):
    return repository.new_request(
        mode="process", config_path="config.yaml", config_sha256=compute_config_hash({})
    )


@pytest.fixture
def job(tmp_path):
    repository = FileJobRepository(tmp_path / "jobs")
    request = _new_request(repository)
    repository.create_job(request)
    return repository, request


@pytest.mark.parametrize("kind", ["request", "state"])
@pytest.mark.parametrize("version", [None, 0, 2, 4, "1", True, 1.0])
def test_request_and_state_reject_missing_or_unsupported_version(job, kind, version):
    if kind == "state" and version == 2:
        version = 1
    repository, request = job
    path = repository.job_dir(request.job_id) / f"{kind}.json"
    data = json.loads(path.read_text())
    if version is None:
        data.pop("schema_version")
    else:
        data["schema_version"] = version
    path.write_text(json.dumps(data), encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="schema_version"):
        repository.get_state(request.job_id)
    with pytest.raises(ValueError, match="schema_version"):
        repository.transition(request.job_id, JobStatus.RUNNING)
    assert path.read_bytes() == before


@pytest.mark.parametrize("revision", [None, -1, True, "0", 0.5])
def test_corrupt_revision_is_not_coerced_or_overwritten(job, revision):
    repository, request = job
    path = repository.state_path(request.job_id)
    data = json.loads(path.read_text())
    data["revision"] = revision
    path.write_text(json.dumps(data), encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="revision"):
        repository.transition(request.job_id, JobStatus.RUNNING)
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "field", ["mode", "config_path", "config_sha256", "created_at"]
)
def test_state_cannot_rewrite_immutable_request_identity(job, field):
    repository, request = job
    before = repository.state_path(request.job_id).read_bytes()
    value = 123.0 if field == "created_at" else "changed"
    with pytest.raises(JobRepositoryError, match=field):
        repository.update_state(
            request.job_id, lambda state: replace(state, **{field: value})
        )
    assert repository.state_path(request.job_id).read_bytes() == before


@pytest.mark.parametrize("kind", ["request", "state"])
def test_persisted_identity_must_match_directory_and_request(job, kind):
    repository, request = job
    path = repository.job_dir(request.job_id) / f"{kind}.json"
    data = json.loads(path.read_text())
    data["job_id"] = _new_request(repository).job_id
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(JobRepositoryError, match="identity|job_id"):
        repository.get_state(request.job_id)


def test_save_state_checks_snapshot_revision_by_default(job):
    repository, request = job
    stale = repository.get_state(request.job_id)
    repository.transition(request.job_id, JobStatus.RUNNING)
    with pytest.raises(RevisionConflictError):
        repository.save_state(replace(stale, status=JobStatus.FAILED))
    assert repository.get_state(request.job_id).status == JobStatus.RUNNING


def test_updater_cannot_mutate_repository_owned_revision(job):
    repository, request = job

    def update(state):
        state.revision = 100
        return state

    with pytest.raises(RevisionConflictError):
        repository.update_state(request.job_id, update)
    assert repository.get_state(request.job_id).revision == 0


@pytest.mark.parametrize("failure", [False, True])
def test_creation_exposes_only_a_complete_job(tmp_path, monkeypatch, failure):
    import src.jobs.repository as module

    repository = FileJobRepository(tmp_path / "jobs")
    request = _new_request(repository)
    write = module.atomic_write_json

    def write_staged(path, data):
        assert repository.list_job_ids() == []
        assert not repository.job_dir(request.job_id).exists()
        if failure and Path(path).name == "state.json":
            raise OSError("injected write failure")
        write(path, data)

    monkeypatch.setattr(module, "atomic_write_json", write_staged)
    if failure:
        with pytest.raises(OSError, match="injected"):
            repository.create_job(request)
        assert repository.list_job_ids() == []
    else:
        repository.create_job(request)
        assert repository.list_job_ids() == [request.job_id]
        assert repository.get_request(request.job_id) == request
        assert repository.list_events(request.job_id)[0].type == "job_created"
    assert not list(repository.root.glob(".creating-*"))


def test_failed_state_replace_preserves_snapshot_and_releases_lock(job, monkeypatch):
    import src.jobs.io as job_io

    repository, request = job
    path = repository.state_path(request.job_id)
    before = path.read_bytes()
    with monkeypatch.context() as patch:

        def fail_replace(source, target):
            raise OSError("injected replace failure")

        patch.setattr(job_io.os, "replace", fail_replace)
        with pytest.raises(OSError, match="injected"):
            repository.transition(request.job_id, JobStatus.RUNNING)
    assert path.read_bytes() == before
    assert not list(path.parent.glob("*.tmp"))
    assert repository.transition(request.job_id, JobStatus.RUNNING).revision == 1


def _concurrent_operation(root, request_data, barrier, results, operation):
    repository = FileJobRepository(root)
    request = JobRequest.from_dict(request_data)
    snapshot = None if operation == "create" else repository.get_state(request.job_id)
    barrier.wait(timeout=15)
    if operation == "create":
        try:
            repository.create_job(request)
            results.put("created")
        except ImmutableRecordError:
            results.put("duplicate")
    elif operation == "cas":
        try:
            repository.save_state(replace(snapshot, status=JobStatus.RUNNING))
            results.put("saved")
        except RevisionConflictError:
            results.put("conflict")
    elif operation == "claim":
        try:
            repository.claim_job(request.job_id, str(os.getpid()))
            results.put("claimed")
        except LeaseConflictError:
            results.put("conflict")
    else:
        for _ in range(15):
            repository.update_state(
                request.job_id,
                lambda state: replace(
                    state,
                    counts=replace(state.counts, retries=state.counts.retries + 1),
                ),
            )
        results.put("updated")


@pytest.mark.parametrize("operation", ["create", "cas", "update", "claim"])
def test_processes_do_not_publish_duplicates_or_lose_updates(tmp_path, operation):
    repository = FileJobRepository(tmp_path / "jobs")
    request = _new_request(repository)
    if operation != "create":
        repository.create_job(request)
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    barrier = context.Barrier(2)
    processes = [
        context.Process(
            target=_concurrent_operation,
            args=(repository.root, request.to_dict(), barrier, results, operation),
        )
        for _ in range(2)
    ]
    try:
        for process in processes:
            process.start()
        output = sorted(results.get(timeout=20) for _ in processes)
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
        expected = {
            "create": ["created", "duplicate"],
            "cas": ["conflict", "saved"],
            "update": ["updated", "updated"],
            "claim": ["claimed", "conflict"],
        }
        assert output == expected[operation]
        state = repository.get_state(request.job_id)
        assert (
            state.revision
            == {"create": 0, "cas": 1, "update": 30, "claim": 1}[operation]
        )
        if operation == "update":
            assert state.counts.retries == 30
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
        results.close()
        results.join_thread()


def _hold_lock(path, ready):
    with _exclusive_file_lock(path):
        ready.set()
        time.sleep(20)


@pytest.mark.parametrize("lock_name", [".repository.lock", ".supervisor.lock"])
def test_old_lock_is_not_stolen_and_owner_crash_releases_it(tmp_path, lock_name):
    path = tmp_path / lock_name
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    process = context.Process(target=_hold_lock, args=(path, ready))
    process.start()
    try:
        assert ready.wait(timeout=10)
        if lock_name == ".supervisor.lock":
            from src.jobs import SupervisorConflictError

            repository = FileJobRepository(tmp_path)
            with pytest.raises(SupervisorConflictError):
                with repository.supervisor_lock():
                    pytest.fail("second supervisor was admitted")
            repository.create_job(_new_request(repository))
        os.utime(path, (1.0, 1.0))
        with pytest.raises(TimeoutError):
            with _exclusive_file_lock(path, timeout_seconds=0.05):
                pytest.fail("live owner lock was stolen")
        process.terminate()
        process.join(timeout=10)
        assert not process.is_alive()
        with _exclusive_file_lock(path, timeout_seconds=1):
            assert path.exists()
        assert path.exists()  # 保留 inode，避免等待者锁住不同的文件。
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
