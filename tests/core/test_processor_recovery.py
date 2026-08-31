from __future__ import annotations

import asyncio
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.core.processor import UniversalAIProcessor
from src.core.job_tracker import JobRecordTracker
from src.core.retry import RetryStrategy
from src.core.state import TaskStateManager
from src.data.contracts import (
    TaskBatch,
    TaskRecord,
    WriteFailure,
    WritebackReceipt,
)
from src.models.errors import ErrorType
from src.jobs import FileJobRepository, RecordStatus, hash_config_file


class _Pool:
    def __init__(self):
        self.records = [("record-a", {"input": "original"})]
        self.writes = []
        self.scan_cursors = []

    async def scan(self, cursor, limit):
        self.scan_cursors.append(cursor)
        if cursor is not None:
            raise AssertionError(f"unexpected cursor: {cursor}")
        records = tuple(
            TaskRecord(record_id=record_id, data=data)
            for record_id, data in self.records[:limit]
        )
        self.records = self.records[limit:]
        return TaskBatch(records=records, next_cursor=None)

    async def reload(self, record_ids):
        return {record_id: {"input": "reloaded"} for record_id in record_ids}

    async def write_results(self, batch_id, results):
        self.writes.append(dict(results))
        return WritebackReceipt.persisted(batch_id, list(results), atomic=True)


def _processor_for_loop(pool):
    processor = object.__new__(UniversalAIProcessor)
    processor.task_pool = pool
    processor.task_manager = SimpleNamespace(
        load_next_shard=lambda: False,
        total_shards=1,
        current_shard_index=1,
        total_estimated=1,
        total_processed_successfully=0,
        max_retries_exceeded_count=0,
        retried_tasks_count=defaultdict(int),
        monitor_memory_usage=lambda: None,
        progress_percent=100.0,
    )
    processor.state_manager = TaskStateManager()
    processor.retry_strategy = RetryStrategy(
        {
            ErrorType.API: 0,
            ErrorType.CONTENT: 1,
            ErrorType.SYSTEM: 0,
        },
        base_backoff_seconds=0,
    )
    processor.max_in_flight = 1
    processor.batch_size = 1
    processor.write_retry_limit = 0
    processor.columns_to_write = {"answer": "result"}
    processor._job_cancel_event = None
    processor._target_concurrency_provider = None
    processor._job_tracker = None
    processor._write_progress = lambda: None
    return processor


@pytest.mark.asyncio
async def test_content_retry_limit_one_makes_one_real_extra_request():
    pool = _Pool()
    processor = _processor_for_loop(pool)
    calls = []

    async def process_one(_session, _record_id, row_data):
        calls.append(dict(row_data))
        if len(calls) == 1:
            return {
                "_error": "invalid_or_missing_json",
                "_error_type": ErrorType.CONTENT,
            }
        return {"answer": "ok"}

    processor._process_one_record = process_one
    completed = await processor._process_loop(object())

    assert completed is True
    assert calls == [{"input": "original"}, {"input": "original"}]
    assert processor.task_manager.retried_tasks_count[ErrorType.CONTENT] == 1
    assert processor.task_manager.total_processed_successfully == 1
    assert pool.writes == [{"record-a": {"answer": "ok"}}]


@pytest.mark.asyncio
async def test_processor_uses_opaque_adapter_cursor_for_all_pages():
    class CursorPool(_Pool):
        def __init__(self):
            super().__init__()
            self.records = []

        async def scan(self, cursor, limit):
            assert limit == 1
            self.scan_cursors.append(cursor)
            if cursor is None:
                return TaskBatch(
                    records=(TaskRecord("string-a", {"input": "one"}),),
                    next_cursor={"after": "string-a"},
                )
            assert cursor == {"after": "string-a"}
            return TaskBatch(
                records=(TaskRecord("string-z", {"input": "two"}),),
                next_cursor=None,
            )

    pool = CursorPool()
    processor = _processor_for_loop(pool)

    async def process_one(_session, record_id, _row_data):
        return {"answer": record_id}

    processor._process_one_record = process_one
    completed = await processor._process_loop(object())

    assert completed is True
    assert pool.scan_cursors == [None, {"after": "string-a"}]
    assert pool.writes == [
        {"string-a": {"answer": "string-a"}},
        {"string-z": {"answer": "string-z"}},
    ]
    assert processor.task_manager.total_processed_successfully == 2


@pytest.mark.asyncio
async def test_writeback_only_acknowledges_persisted_ids():
    processor = object.__new__(UniversalAIProcessor)
    processor.write_retry_limit = 0

    class PartialPool:
        async def write_results(self, batch_id, _results):
            return WritebackReceipt(
                batch_id=batch_id,
                persisted_ids=("a",),
                failures=(
                    WriteFailure("b", "remote_failure", "failed", retryable=True),
                ),
            )

    processor.task_pool = PartialPool()
    persisted = await processor._persist_results_with_retry(
        {"a": {"answer": "ok"}, "b": {"answer": "not persisted"}}
    )

    assert persisted == {"a"}


def test_retry_after_supports_seconds_and_http_date():
    assert UniversalAIProcessor._parse_retry_after({"Retry-After": "2.5"}) == 2.5
    assert UniversalAIProcessor._parse_retry_after({"Retry-After": "invalid"}) == 0


@pytest.mark.asyncio
async def test_writeback_exception_retries_whole_pending_batch(monkeypatch):
    processor = object.__new__(UniversalAIProcessor)
    processor.write_retry_limit = 1
    calls = 0

    class FlakyPool:
        async def write_results(self, batch_id, results):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("temporary write failure")
            return WritebackReceipt.persisted(batch_id, list(results), atomic=True)

    processor.task_pool = FlakyPool()
    sleep = AsyncMock()
    monkeypatch.setattr("src.core.processor.asyncio.sleep", sleep)

    persisted = await processor._persist_results_with_retry({"a": {"answer": "ok"}})

    assert persisted == {"a"}
    assert calls == 2
    sleep.assert_awaited_once_with(1)


@pytest.mark.asyncio
async def test_permanent_write_failure_is_not_retried(monkeypatch):
    processor = object.__new__(UniversalAIProcessor)
    processor.write_retry_limit = 3
    calls = 0

    class PermanentFailurePool:
        async def write_results(self, batch_id, _results):
            nonlocal calls
            calls += 1
            return WritebackReceipt(
                batch_id=batch_id,
                failures=(
                    WriteFailure(
                        "missing",
                        "record_not_found",
                        "record does not exist",
                        retryable=False,
                    ),
                ),
            )

    processor.task_pool = PermanentFailurePool()
    sleep = AsyncMock()
    monkeypatch.setattr("src.core.processor.asyncio.sleep", sleep)

    persisted = await processor._persist_results_with_retry(
        {"missing": {"answer": "not writable"}}
    )

    assert persisted == set()
    assert calls == 1
    sleep.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("conflicting", [False, True])
async def test_malformed_write_receipt_cannot_create_phantom_success(conflicting):
    processor = object.__new__(UniversalAIProcessor)
    processor.write_retry_limit = 0

    class MalformedReceiptPool:
        async def write_results(self, batch_id, _results):
            return WritebackReceipt(
                batch_id=batch_id,
                persisted_ids=(("record-a",) if conflicting else ("phantom",)),
                failures=(
                    (WriteFailure("record-a", "conflict", "conflicting receipt"),)
                    if conflicting
                    else ()
                ),
            )

    processor.task_pool = MalformedReceiptPool()
    persisted = await processor._persist_results_with_retry(
        {"record-a": {"answer": "value"}}
    )

    assert persisted == set()


@pytest.mark.asyncio
async def test_datasource_count_failure_is_not_treated_as_empty_success():
    processor = object.__new__(UniversalAIProcessor)
    finalized = False

    class BrokenPool:
        def get_total_task_count(self):
            raise RuntimeError("database unavailable")

    def finalize():
        nonlocal finalized
        finalized = True

    processor.task_pool = BrokenPool()
    processor.task_manager = SimpleNamespace(finalize=finalize)
    processor.batch_size = 1

    with pytest.raises(RuntimeError, match="database unavailable"):
        await processor.process_shard_async_continuous()
    assert finalized is True


@pytest.mark.asyncio
async def test_retry_budget_survives_checkpoint_recovery(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("version: 1\n", encoding="utf-8")
    repository = FileJobRepository(tmp_path / "jobs")
    request = repository.new_request(
        mode="background",
        config_path=str(config_path),
        config_sha256=hash_config_file(config_path),
    )
    repository.create_job(request)
    tracker = JobRecordTracker(repository, request.job_id)
    tracker.mark_in_flight("record-a", {"input": "original"})
    tracker.mark_pending(
        "record-a",
        {"input": "original"},
        error_code=ErrorType.CONTENT.value,
        retry_error_type=ErrorType.CONTENT.value,
    )

    recovered = JobRecordTracker(repository, request.job_id)
    assert recovered.retry_counts_for("record-a") == {"content_error": 1}
    pool = _Pool()
    processor = _processor_for_loop(pool)
    processor._job_tracker = recovered
    calls = 0

    async def content_error(*_args):
        nonlocal calls
        calls += 1
        return {
            "_error": "invalid_or_missing_json",
            "_error_type": ErrorType.CONTENT,
        }

    processor._process_one_record = content_error
    completed = await processor._process_loop(object())

    assert completed is True
    assert calls == 1
    checkpoint = JobRecordTracker(repository, request.job_id).get("record-a")
    assert checkpoint is not None
    assert checkpoint.status == RecordStatus.FAILED
    assert checkpoint.retry_counts == {"content_error": 1}


@pytest.mark.asyncio
async def test_checkpoint_replay_tracks_persisted_and_failed_ids():
    processor = object.__new__(UniversalAIProcessor)
    processor.write_retry_limit = 0

    class PartialPool:
        async def write_results(self, batch_id, _results):
            return WritebackReceipt(
                batch_id=batch_id,
                persisted_ids=("a",),
                failures=(WriteFailure("b", "write_failed", "failed"),),
            )

    class Tracker:
        def __init__(self):
            self.persisted = []
            self.failed = []

        def mark_persisted(self, record_id):
            self.persisted.append(record_id)

        def mark_failed(self, record_id, **kwargs):
            self.failed.append((record_id, kwargs))

    tracker = Tracker()
    processor.task_pool = PartialPool()
    processor._job_tracker = tracker

    persisted = await processor.persist_checkpoint_results(
        {"a": {"answer": "ok"}, "b": {"answer": "retry"}}
    )

    assert persisted == {"a"}
    assert tracker.persisted == ["a"]
    assert tracker.failed[0][0] == "b"
    assert tracker.failed[0][1]["error_code"] == "checkpoint_writeback_failed"


@pytest.mark.asyncio
async def test_job_cancellation_cancels_active_request_and_checkpoints_pending(
    monkeypatch,
):
    pool = _Pool()
    processor = _processor_for_loop(pool)
    cancel_event = asyncio.Event()
    checkpointed = []

    class Tracker:
        def should_process_scanned(self, _record_id):
            return True

        def retry_counts_for(self, _record_id):
            return {}

        def new_scan_shard_id(self):
            return "scan-000001"

        def register_scan_batch(self, *_args, **_kwargs):
            return None

        def mark_in_flight(self, *_args):
            return None

        def mark_pending(self, record_id, data, **kwargs):
            checkpointed.append((record_id, data, kwargs))

    async def never_finishes(*_args):
        await asyncio.Event().wait()

    async def wait_and_cancel(tasks, **_kwargs):
        cancel_event.set()
        return set(), set(tasks)

    processor._process_one_record = never_finishes
    processor._job_cancel_event = cancel_event
    processor._job_tracker = Tracker()
    monkeypatch.setattr("src.core.processor.asyncio.wait", wait_and_cancel)

    completed = await processor._process_loop(object())

    assert completed is False
    assert checkpointed == [
        ("record-a", {"input": "original"}, {"error_code": "cancelled"})
    ]
