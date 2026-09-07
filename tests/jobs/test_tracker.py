from __future__ import annotations

from dataclasses import replace
import json

import pytest

from src.core.contracts import PreparedResult
from src.core.job_tracker import JobRecordTracker
from src.jobs import (
    FileJobRepository,
    ImmutableRecordError,
    JobRepositoryError,
    RecordStatus,
    hash_config_file,
)


@pytest.fixture
def tracker_context(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("version: 1\n", encoding="utf-8")
    repository = FileJobRepository(tmp_path / "jobs")
    request = repository.new_request(
        mode="background",
        config_path=str(config_path),
        config_sha256=hash_config_file(config_path),
    )
    repository.create_job(request)
    return repository, request.job_id


def test_record_transitions_persist_prepared_reference_and_counts(tracker_context):
    repository, job_id = tracker_context
    tracker = JobRecordTracker(repository, job_id)

    tracker.mark_in_flight("row-a", {"input": "hello"})
    tracker.mark_pending(
        "row-a",
        {"input": "hello"},
        error_code="content_error",
        retry_error_type="content_error",
    )
    tracker.mark_in_flight("row-a", {"input": "hello"})
    prepared = PreparedResult.create("row-a", {"answer": "ok"})
    tracker.mark_prepared(prepared)
    tracker.mark_commit_attempt("row-a")
    tracker.mark_persisted("row-a")

    shard = repository.get_shard(job_id, "records")
    checkpoint = shard.records[0]
    assert checkpoint.status == RecordStatus.PERSISTED
    assert checkpoint.attempt == 2
    assert checkpoint.retry_counts == {"content_error": 1}
    assert checkpoint.prepared_ref == f"prepared/{prepared.commit_id}.json"
    assert checkpoint.prepared_hash == prepared.payload_hash
    assert checkpoint.commit_id == prepared.commit_id
    assert checkpoint.commit_attempts == 1
    assert "result" not in checkpoint.to_dict()
    state = repository.get_state(job_id)
    assert state.counts.discovered == 1
    assert state.counts.ai_complete == 1
    assert state.counts.persisted == 1
    assert state.counts.unresolved_writes == 0
    assert state.counts.retries == 1
    events = repository.list_events(job_id)
    assert events[-1].type == "record_persisted"
    assert "answer" not in json.dumps(events[-1].payload)


def test_pending_commit_loads_exact_prepared_result_after_restart(tracker_context):
    repository, job_id = tracker_context
    tracker = JobRecordTracker(repository, job_id)
    tracker.mark_in_flight(42, {"input": "hello"})
    prepared = PreparedResult.create(42, {"answer": "durable"})
    tracker.mark_prepared(prepared)

    recovered = JobRecordTracker(repository, job_id)
    assert recovered.pending_prepared_results() == {42: prepared}
    assert recovered.should_process_scanned(42) is False

    recovered.mark_unresolved(42)
    unresolved = JobRecordTracker(repository, job_id)
    assert unresolved.pending_prepared_results() == {}
    assert unresolved.counts().unresolved_writes == 1
    assert unresolved.should_process_scanned(42) is False


def test_scanning_does_not_repeat_terminal_record_work(tracker_context):
    repository, job_id = tracker_context
    tracker = JobRecordTracker(repository, job_id)
    tracker.mark_in_flight("content", {"input": "invalid"})
    tracker.mark_failed("content", error_code="content_error")
    tracker.mark_in_flight("writeback", {"input": "hello"})
    tracker.mark_prepared(
        PreparedResult.create("writeback", {"answer": "checkpointed"})
    )

    recovered = JobRecordTracker(repository, job_id)
    assert recovered.should_process_scanned("writeback") is False
    assert recovered.should_process_scanned("content") is False
    assert recovered.should_process_scanned("new-record") is True


def test_blob_after_checkpoint_failure_is_orphaned_and_ignored(
    tracker_context,
    monkeypatch,
):
    repository, job_id = tracker_context
    tracker = JobRecordTracker(repository, job_id)
    tracker.mark_in_flight("row-a", {"input": "hello"})
    original_save = repository.save_shard

    def fail_save(_shard):
        raise OSError("fault injection")

    monkeypatch.setattr(repository, "save_shard", fail_save)
    prepared = PreparedResult.create("row-a", {"answer": "not acknowledged"})
    with pytest.raises(OSError, match="fault injection"):
        tracker.mark_prepared(prepared)
    monkeypatch.setattr(repository, "save_shard", original_save)

    assert repository.prepared_result_path(job_id, prepared.commit_id).is_file()
    recovered = JobRecordTracker(repository, job_id)
    assert recovered.get("row-a").status == RecordStatus.IN_FLIGHT
    assert recovered.pending_prepared_results() == {}


def test_failure_before_prepared_blob_keeps_record_in_flight(
    tracker_context,
    monkeypatch,
):
    repository, job_id = tracker_context
    tracker = JobRecordTracker(repository, job_id)
    tracker.mark_in_flight("row-a", {"input": "hello"})

    def fail_blob(*_args, **_kwargs):
        raise OSError("blob fault")

    monkeypatch.setattr(repository, "save_prepared_result", fail_blob)
    with pytest.raises(OSError, match="blob fault"):
        tracker.mark_prepared(PreparedResult.create("row-a", {"answer": "no"}))

    recovered = JobRecordTracker(repository, job_id)
    assert recovered.get("row-a").status == RecordStatus.IN_FLIGHT
    assert list(repository.prepared_dir(job_id).iterdir()) == []


def test_prepared_hash_mismatch_fails_recovery(tracker_context):
    repository, job_id = tracker_context
    tracker = JobRecordTracker(repository, job_id)
    tracker.mark_in_flight("row-a", {"input": "hello"})
    prepared = PreparedResult.create("row-a", {"answer": "ok"})
    tracker.mark_prepared(prepared)
    path = repository.prepared_result_path(job_id, prepared.commit_id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["values"]["answer"] = "tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(JobRepositoryError, match="invalid prepared result blob"):
        JobRecordTracker(repository, job_id).pending_prepared_results()


def test_duplicate_commit_id_with_different_payload_is_rejected(tracker_context):
    repository, job_id = tracker_context
    tracker = JobRecordTracker(repository, job_id)
    first = PreparedResult.create("row-a", {"answer": "one"})
    tracker.mark_in_flight("row-a", {"input": "a"})
    tracker.mark_prepared(first)

    second = PreparedResult.create("row-b", {"answer": "two"})
    second = replace(second, commit_id=first.commit_id)
    tracker.mark_in_flight("row-b", {"input": "b"})
    with pytest.raises(ImmutableRecordError, match="collision"):
        tracker.mark_prepared(second)


def test_scan_pages_are_checkpointed_as_independent_shards(tracker_context):
    repository, job_id = tracker_context
    tracker = JobRecordTracker(repository, job_id)
    first_shard = tracker.new_scan_shard_id()
    second_shard = tracker.new_scan_shard_id()

    tracker.register_scan_batch(
        first_shard,
        [("row-a", {"input": "a"}), ("row-b", {"input": "b"})],
        cursor={"last_id": "row-b"},
    )
    tracker.register_scan_batch(
        second_shard,
        [("row-c", {"input": "c"})],
        cursor=None,
    )
    tracker.mark_in_flight("row-a", {"input": "a"})
    tracker.mark_prepared(PreparedResult.create("row-a", {"answer": "done"}))
    tracker.mark_persisted("row-a")
    tracker.mark_failed("row-b", error_code="content_error")

    shards = {shard.shard_id: shard for shard in repository.list_shards(job_id)}
    assert set(shards) == {first_shard, second_shard}
    assert shards[first_shard].cursor == {"last_id": "row-b"}
    assert shards[first_shard].status == "completed"
    assert shards[second_shard].status == "active"
    assert [record.record_id for record in shards[first_shard].records] == [
        "row-a",
        "row-b",
    ]

    recovered = JobRecordTracker(repository, job_id)
    assert recovered.counts().discovered == 3
    assert recovered.counts().persisted == 1
    assert recovered.get("row-c").status == RecordStatus.PENDING
    assert recovered.new_scan_shard_id() == "scan-000003"


def test_state_not_diagnostic_shard_is_recovery_truth(tracker_context):
    repository, job_id = tracker_context
    tracker = JobRecordTracker(repository, job_id)
    tracker.mark_in_flight("a", {"input": "a"})
    prepared = PreparedResult.create("a", {"answer": "committed reference"})
    tracker.mark_prepared(prepared)
    state = json.loads(repository.state_path(job_id).read_text())
    record = state["checkpoints"]["records"]["records"][0]
    assert record["status"] == "pending_commit"
    assert record["prepared_hash"] == prepared.payload_hash
    assert "checkpoints" not in repository.get_state(job_id).to_dict()
    (repository.shards_dir(job_id) / "records.json").write_text("broken diagnostic")
    assert JobRecordTracker(repository, job_id).pending_prepared_results() == {
        "a": prepared
    }


def test_state_commit_failure_leaves_blob_orphan_and_rolls_back_memory(
    tracker_context, monkeypatch
):
    import src.jobs.repository as module

    repository, job_id = tracker_context
    tracker = JobRecordTracker(repository, job_id)
    tracker.mark_in_flight("a", {"input": "a"})
    before = repository.state_path(job_id).read_bytes()
    original = module.atomic_write_json

    def write(path, payload):
        if path == repository.state_path(job_id):
            raise OSError("state commit failed")
        original(path, payload)

    prepared = PreparedResult.create("a", {"answer": "not committed"})
    with monkeypatch.context() as patch:
        patch.setattr(module, "atomic_write_json", write)
        with pytest.raises(OSError, match="state commit failed"):
            tracker.mark_prepared(prepared)
    assert repository.state_path(job_id).read_bytes() == before
    assert tracker.get("a").status == RecordStatus.IN_FLIGHT
    assert tracker.pending_prepared_results() == {}
    assert JobRecordTracker(repository, job_id).pending_prepared_results() == {}
    orphan = f"prepared/{prepared.commit_id}.json"
    assert repository.prune_orphan_prepared(job_id) == (orphan,)
    assert repository.prepared_result_path(job_id, prepared.commit_id).exists()
    assert repository.prune_orphan_prepared(job_id, confirm=True) == (orphan,)
    assert not repository.prepared_result_path(job_id, prepared.commit_id).exists()


def test_event_failure_cannot_erase_committed_prepared_reference(
    tracker_context, monkeypatch
):
    repository, job_id = tracker_context
    tracker = JobRecordTracker(repository, job_id)
    tracker.mark_in_flight("a", {"input": "a"})
    prepared = PreparedResult.create("a", {"answer": "durable"})

    def fail_event(*args, **kwargs):
        raise OSError("event fault")

    monkeypatch.setattr(repository, "append_event", fail_event)
    with pytest.raises(OSError, match="event fault"):
        tracker.mark_prepared(prepared)
    assert JobRecordTracker(repository, job_id).pending_prepared_results() == {
        "a": prepared
    }
    assert repository.get_state(job_id).counts.ai_complete == 1
    assert repository.prune_orphan_prepared(job_id, confirm=True) == ()


def test_prepared_parent_symlink_cannot_escape_job(tracker_context, tmp_path):
    repository, job_id = tracker_context
    outside = tmp_path / "outside"
    outside.mkdir()
    prepared_dir = repository.prepared_dir(job_id)
    prepared_dir.rmdir()
    try:
        prepared_dir.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink unavailable on this platform")
    prepared = PreparedResult.create("a", {"answer": "never written outside"})
    with pytest.raises(JobRepositoryError, match="symlink"):
        repository.save_prepared_result(job_id, prepared.commit_id, prepared.to_dict())
    with pytest.raises(JobRepositoryError, match="symlink"):
        repository.prune_orphan_prepared(job_id, confirm=True)
    assert list(outside.iterdir()) == []
