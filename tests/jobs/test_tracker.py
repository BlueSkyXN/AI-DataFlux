from __future__ import annotations

import pytest

from src.core.job_tracker import JobRecordTracker
from src.jobs import FileJobRepository, RecordStatus, hash_config_file


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


def test_record_transitions_persist_counts_and_safe_events(tracker_context):
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
    tracker.mark_ai_complete("row-a", {"answer": "ok"})
    tracker.mark_persisted("row-a")

    shard = repository.get_shard(job_id, "records")
    assert shard.records[0].status == RecordStatus.PERSISTED
    assert shard.records[0].attempt == 2
    assert shard.records[0].retry_counts == {"content_error": 1}
    assert shard.records[0].result == {"answer": "ok"}
    state = repository.get_state(job_id)
    assert state.counts.discovered == 1
    assert state.counts.ai_complete == 1
    assert state.counts.persisted == 1
    assert state.counts.retries == 1
    events = repository.list_events(job_id)
    assert events[-1].type == "record_persisted"
    assert "answer" not in events[-1].payload


def test_ai_complete_checkpoint_replays_after_commit_before_ack(tracker_context):
    repository, job_id = tracker_context
    tracker = JobRecordTracker(repository, job_id)
    tracker.mark_in_flight(42, {"input": "hello"})
    tracker.mark_ai_complete(42, {"answer": "durable"})

    # Simulate a datasource commit followed by a process crash before the
    # checkpoint can be promoted to persisted.
    recovered = JobRecordTracker(repository, job_id)
    assert recovered.replayable_results() == {42: {"answer": "durable"}}

    recovered.mark_persisted(42)
    recovered_again = JobRecordTracker(repository, job_id)
    assert recovered_again.replayable_results() == {}
    assert recovered_again.counts().persisted == 1


def test_scanning_skips_outputs_that_must_be_replayed_without_new_ai(
    tracker_context,
):
    repository, job_id = tracker_context
    tracker = JobRecordTracker(repository, job_id)
    tracker.mark_in_flight("writeback", {"input": "hello"})
    tracker.mark_failed(
        "writeback",
        error_code="writeback_failed",
        result={"answer": "checkpointed"},
    )
    tracker.mark_in_flight("content", {"input": "invalid"})
    tracker.mark_failed("content", error_code="content_error")

    recovered = JobRecordTracker(repository, job_id)
    assert recovered.replayable_results() == {"writeback": {"answer": "checkpointed"}}
    assert recovered.should_process_scanned("writeback") is False
    assert recovered.should_process_scanned("content") is True
    assert recovered.should_process_scanned("new-record") is True


def test_checkpoint_write_failure_does_not_publish_ai_complete(
    tracker_context, monkeypatch
):
    repository, job_id = tracker_context
    tracker = JobRecordTracker(repository, job_id)
    tracker.mark_in_flight("row-a", {"input": "hello"})
    original_save = repository.save_shard

    def fail_save(_shard):
        raise OSError("fault injection")

    monkeypatch.setattr(repository, "save_shard", fail_save)
    with pytest.raises(OSError, match="fault injection"):
        tracker.mark_ai_complete("row-a", {"answer": "not acknowledged"})
    monkeypatch.setattr(repository, "save_shard", original_save)

    recovered = JobRecordTracker(repository, job_id)
    assert recovered.get("row-a").status == RecordStatus.IN_FLIGHT
    assert recovered.replayable_results() == {}


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
    tracker.mark_ai_complete("row-a", {"answer": "done"})
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
