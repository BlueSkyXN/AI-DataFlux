from __future__ import annotations

import asyncio
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.core.processor import UniversalAIProcessor
from src.core.contracts import (
    FailureStage,
    PreparedResult,
    SourceOperationError,
    TaskFailure,
    TaskSuccess,
    WritebackOutcome,
)
from src.core.job_tracker import JobRecordTracker
from src.core.retry import RetryStrategy
from src.core.state import TaskStateManager
from src.data.contracts import (
    CommitDisposition,
    TaskBatch,
    TaskRecord,
    WritebackContractError,
    WritebackItem,
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
        return WritebackReceipt.committed(batch_id, results, atomic=True)

    async def reconcile_results(self, batch_id, results):
        return WritebackReceipt.committed(batch_id, results, atomic=True)


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
        unresolved_writes_count=0,
        retried_tasks_count=defaultdict(int),
        monitor_memory_usage=lambda: None,
        progress_percent=100.0,
    )
    processor.state_manager = TaskStateManager()
    processor.retry_strategy = RetryStrategy(
        max_attempts={
            ErrorType.API: 1,
            ErrorType.CONTENT: 2,
            ErrorType.SYSTEM: 1,
            ErrorType.SOURCE: 3,
        },
        base_backoff_seconds=0,
    )
    processor.source_max_attempts = 3
    processor.max_in_flight = 1
    processor.batch_size = 1
    processor.commit_max_attempts = 3
    processor.reconciliation_max_attempts = 3
    processor.writeback_backoff_initial_seconds = 1
    processor.writeback_backoff_max_seconds = 30
    processor.columns_to_write = {"answer": "result"}
    processor._job_cancel_event = None
    processor._target_concurrency_provider = None
    processor._job_tracker = None
    processor._write_progress = lambda: None
    return processor


def _writeback_processor(pool, *, commit_attempts=3, reconciliation_attempts=3):
    processor = object.__new__(UniversalAIProcessor)
    processor.task_pool = pool
    processor.commit_max_attempts = commit_attempts
    processor.reconciliation_max_attempts = reconciliation_attempts
    processor.writeback_backoff_initial_seconds = 1
    processor.writeback_backoff_max_seconds = 30
    processor._job_tracker = None
    return processor


@pytest.mark.asyncio
async def test_content_max_attempts_two_makes_one_real_retry():
    pool = _Pool()
    processor = _processor_for_loop(pool)
    calls = []

    async def process_one(_session, _record_id, row_data):
        calls.append(dict(row_data))
        if len(calls) == 1:
            return TaskFailure(
                FailureStage.CONTENT,
                "invalid_or_missing_json",
                "invalid content",
                True,
            )
        return TaskSuccess(PreparedResult.create(_record_id, {"answer": "ok"}))

    processor._process_one_record = process_one
    completed = await processor._process_loop(object())

    assert completed is True
    assert calls == [{"input": "original"}, {"input": "original"}]
    assert processor.task_manager.retried_tasks_count[ErrorType.CONTENT] == 1
    assert processor.task_manager.total_processed_successfully == 1
    assert pool.writes == [{"record-a": {"answer": "ok"}}]


@pytest.mark.asyncio
async def test_commit_retry_does_not_call_model_again(monkeypatch):
    class RetryWritePool(_Pool):
        async def write_results(self, batch_id, results):
            self.writes.append(dict(results))
            if len(self.writes) == 1:
                return WritebackReceipt(
                    batch_id=batch_id,
                    submitted_ids=tuple(results),
                    items=(
                        WritebackItem(
                            "record-a",
                            CommitDisposition.REJECTED,
                            "temporary",
                            "retry",
                            True,
                        ),
                    ),
                    atomic=True,
                )
            return WritebackReceipt.committed(batch_id, results, atomic=True)

    pool = RetryWritePool()
    processor = _processor_for_loop(pool)
    model_calls = 0

    async def process_one(_session, record_id, _row_data):
        nonlocal model_calls
        model_calls += 1
        return TaskSuccess(PreparedResult.create(record_id, {"answer": "ok"}))

    sleep = AsyncMock()
    monkeypatch.setattr("src.core.processor.asyncio.sleep", sleep)
    processor._process_one_record = process_one

    assert await processor._process_loop(object()) is True
    assert model_calls == 1
    assert pool.writes == [
        {"record-a": {"answer": "ok"}},
        {"record-a": {"answer": "ok"}},
    ]
    sleep.assert_awaited_once_with(1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stage", "expected_inputs", "error_type"),
    [
        (
            FailureStage.CONTENT,
            [{"input": "original"}, {"input": "original"}],
            ErrorType.CONTENT,
        ),
        (
            FailureStage.MODEL,
            [{"input": "original"}, {"input": "reloaded"}],
            ErrorType.API,
        ),
        (
            FailureStage.SYSTEM,
            [{"input": "original"}, {"input": "reloaded"}],
            ErrorType.SYSTEM,
        ),
    ],
)
async def test_retry_stage_controls_reload(stage, expected_inputs, error_type):
    pool = _Pool()
    processor = _processor_for_loop(pool)
    processor.retry_strategy.max_attempts[error_type] = 2
    processor.retry_strategy.last_pause_end_time = float("inf")
    calls = []

    async def process_one(_session, record_id, row_data):
        calls.append(dict(row_data))
        if len(calls) == 1:
            return TaskFailure(stage, "failure", "failed", True)
        return TaskSuccess(PreparedResult.create(record_id, {"answer": "ok"}))

    processor._process_one_record = process_one

    assert await processor._process_loop(object()) is True
    assert calls == expected_inputs
    assert processor.task_manager.retried_tasks_count[error_type] == 1


@pytest.mark.asyncio
async def test_nonretryable_failure_stops_after_first_attempt():
    pool = _Pool()
    processor = _processor_for_loop(pool)
    processor.retry_strategy.max_attempts[ErrorType.API] = 4
    calls = 0

    async def process_one(*_args):
        nonlocal calls
        calls += 1
        return TaskFailure(
            FailureStage.MODEL,
            "model_http_400",
            "invalid request",
            False,
        )

    processor._process_one_record = process_one

    assert await processor._process_loop(object()) is True
    assert calls == 1
    assert processor.task_manager.max_retries_exceeded_count == 1
    assert pool.writes == []


@pytest.mark.asyncio
async def test_max_attempts_one_means_no_retry():
    pool = _Pool()
    processor = _processor_for_loop(pool)
    processor.retry_strategy.max_attempts[ErrorType.CONTENT] = 1
    calls = 0

    async def process_one(*_args):
        nonlocal calls
        calls += 1
        return TaskFailure(
            FailureStage.CONTENT,
            "invalid_json",
            "invalid content",
            True,
        )

    processor._process_one_record = process_one

    assert await processor._process_loop(object()) is True
    assert calls == 1
    assert processor.task_manager.retried_tasks_count[ErrorType.CONTENT] == 0


@pytest.mark.asyncio
async def test_model_retry_after_uses_larger_of_server_and_local_delay(monkeypatch):
    pool = _Pool()
    processor = _processor_for_loop(pool)
    processor.retry_strategy.max_attempts[ErrorType.API] = 2
    processor.retry_strategy.base_backoff_seconds = 1
    processor.retry_strategy.api_pause_duration = 2
    calls = 0

    async def process_one(_session, record_id, _row_data):
        nonlocal calls
        calls += 1
        if calls == 1:
            return TaskFailure(
                FailureStage.MODEL,
                "model_http_429",
                "rate limited",
                True,
                retry_after_seconds=3,
            )
        return TaskSuccess(PreparedResult.create(record_id, {"answer": "ok"}))

    sleep = AsyncMock()
    monkeypatch.setattr("src.core.processor.asyncio.sleep", sleep)
    processor._process_one_record = process_one

    assert await processor._process_loop(object()) is True
    sleep.assert_awaited_once_with(3)


@pytest.mark.asyncio
async def test_reload_missing_fails_record_without_reusing_stale_input():
    class MissingReloadPool(_Pool):
        async def reload(self, _record_ids):
            return {}

    pool = MissingReloadPool()
    processor = _processor_for_loop(pool)
    processor.retry_strategy.max_attempts[ErrorType.SYSTEM] = 2
    calls = 0

    async def process_one(*_args):
        nonlocal calls
        calls += 1
        return TaskFailure(FailureStage.SYSTEM, "system", "failed", True)

    processor._process_one_record = process_one

    assert await processor._process_loop(object()) is True
    assert calls == 1
    assert processor.task_manager.max_retries_exceeded_count == 1
    assert pool.writes == []


@pytest.mark.asyncio
async def test_reload_source_failures_use_independent_attempt_budget(monkeypatch):
    class FlakyReloadPool(_Pool):
        def __init__(self):
            super().__init__()
            self.reload_calls = 0

        async def reload(self, record_ids):
            self.reload_calls += 1
            if self.reload_calls < 3:
                raise OSError("source unavailable")
            return {record_id: {"input": "fresh"} for record_id in record_ids}

    pool = FlakyReloadPool()
    processor = _processor_for_loop(pool)
    processor.retry_strategy.max_attempts[ErrorType.SYSTEM] = 2
    processor.source_max_attempts = 3
    calls = []

    async def process_one(_session, record_id, row_data):
        calls.append(dict(row_data))
        if len(calls) == 1:
            return TaskFailure(FailureStage.SYSTEM, "system", "failed", True)
        return TaskSuccess(PreparedResult.create(record_id, {"answer": "ok"}))

    sleep = AsyncMock()
    monkeypatch.setattr("src.core.processor.asyncio.sleep", sleep)
    processor._process_one_record = process_one

    assert await processor._process_loop(object()) is True
    assert pool.reload_calls == 3
    assert calls == [{"input": "original"}, {"input": "fresh"}]
    assert processor.task_manager.retried_tasks_count[ErrorType.SOURCE] == 2
    assert [call.args[0] for call in sleep.await_args_list] == [1.0, 2.0]


@pytest.mark.asyncio
async def test_scan_failure_retries_then_succeeds_without_empty_completion(monkeypatch):
    class FlakyScanPool(_Pool):
        def __init__(self):
            super().__init__()
            self.calls = 0

        async def scan(self, cursor, limit):
            self.calls += 1
            if self.calls == 1:
                raise OSError("scan failed")
            return await super().scan(cursor, limit)

    pool = FlakyScanPool()
    processor = _processor_for_loop(pool)

    async def process_one(_session, record_id, _row_data):
        return TaskSuccess(PreparedResult.create(record_id, {"answer": "ok"}))

    sleep = AsyncMock()
    monkeypatch.setattr("src.core.processor.asyncio.sleep", sleep)
    processor._process_one_record = process_one

    assert await processor._process_loop(object()) is True
    assert pool.calls == 2
    sleep.assert_awaited_once_with(1.0)


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
        return TaskSuccess(PreparedResult.create(record_id, {"answer": record_id}))

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
async def test_persisted_advances_only_for_committed_receipt_items():
    processor = object.__new__(UniversalAIProcessor)
    processor.commit_max_attempts = 1
    processor.reconciliation_max_attempts = 1
    processor.writeback_backoff_initial_seconds = 0
    processor.writeback_backoff_max_seconds = 0
    processor._job_tracker = None

    class PartialPool:
        async def write_results(self, batch_id, results):
            return WritebackReceipt(
                batch_id=batch_id,
                submitted_ids=tuple(results),
                items=(
                    WritebackItem("a", CommitDisposition.COMMITTED),
                    WritebackItem(
                        "b",
                        CommitDisposition.REJECTED,
                        "remote_failure",
                        "failed",
                        False,
                    ),
                ),
                atomic=False,
            )

    processor.task_pool = PartialPool()
    outcome = await processor._commit_prepared_results(
        {
            "a": PreparedResult.create("a", {"answer": "ok"}),
            "b": PreparedResult.create("b", {"answer": "not persisted"}),
        }
    )

    assert outcome == WritebackOutcome(frozenset({"a"}), frozenset({"b"}), frozenset())


def test_retry_after_supports_seconds_and_http_date():
    assert UniversalAIProcessor._parse_retry_after({"Retry-After": "2.5"}) == 2.5
    assert UniversalAIProcessor._parse_retry_after({"Retry-After": "invalid"}) == 0


@pytest.mark.asyncio
async def test_retryable_rejection_reuses_identical_prepared_payload(
    monkeypatch,
):
    calls = []

    class FlakyPool:
        async def write_results(self, batch_id, results):
            calls.append((batch_id, dict(results)))
            if len(calls) == 1:
                return WritebackReceipt(
                    batch_id=batch_id,
                    submitted_ids=tuple(results),
                    items=(
                        WritebackItem(
                            "a",
                            CommitDisposition.REJECTED,
                            "temporary",
                            "retry",
                            True,
                        ),
                    ),
                    atomic=True,
                )
            return WritebackReceipt.committed(batch_id, results, atomic=True)

    processor = _writeback_processor(FlakyPool(), commit_attempts=2)
    sleep = AsyncMock()
    monkeypatch.setattr("src.core.processor.asyncio.sleep", sleep)
    prepared = PreparedResult.create("a", {"answer": "ok"})
    original = prepared.to_dict()

    outcome = await processor._commit_prepared_results({"a": prepared})

    assert outcome.committed == {"a"}
    assert [values for _batch_id, values in calls] == [
        {"a": prepared.values},
        {"a": prepared.values},
    ]
    assert prepared.to_dict() == original
    sleep.assert_awaited_once_with(1)


@pytest.mark.asyncio
async def test_permanent_write_failure_is_not_retried(monkeypatch):
    calls = 0

    class PermanentFailurePool:
        async def write_results(self, batch_id, results):
            nonlocal calls
            calls += 1
            return WritebackReceipt(
                batch_id=batch_id,
                submitted_ids=tuple(results),
                items=(
                    WritebackItem(
                        "missing",
                        CommitDisposition.REJECTED,
                        "record_not_found",
                        "record does not exist",
                        False,
                    ),
                ),
                atomic=True,
            )

    processor = _writeback_processor(PermanentFailurePool())
    sleep = AsyncMock()
    monkeypatch.setattr("src.core.processor.asyncio.sleep", sleep)

    outcome = await processor._commit_prepared_results(
        {
            "missing": PreparedResult.create(
                "missing",
                {"answer": "not writable"},
            )
        }
    )

    assert outcome.failed == {"missing"}
    assert calls == 1
    sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_indeterminate_write_only_reconciles_and_never_blind_retries():
    writes = 0
    reconciliations = 0

    class UncertainPool:
        async def write_results(self, batch_id, results):
            nonlocal writes
            writes += 1
            return WritebackReceipt(
                batch_id=batch_id,
                submitted_ids=tuple(results),
                items=(
                    WritebackItem(
                        "record-a",
                        CommitDisposition.INDETERMINATE,
                        "transport_lost",
                        "unknown commit outcome",
                        False,
                    ),
                ),
                atomic=True,
            )

        async def reconcile_results(self, batch_id, results):
            nonlocal reconciliations
            reconciliations += 1
            return WritebackReceipt.committed(batch_id, results, atomic=True)

    processor = _writeback_processor(UncertainPool())
    outcome = await processor._commit_prepared_results(
        {
            "record-a": PreparedResult.create(
                "record-a",
                {"answer": "value"},
            )
        }
    )

    assert outcome.committed == {"record-a"}
    assert writes == 1
    assert reconciliations == 1


@pytest.mark.asyncio
async def test_write_exception_is_reconciled_without_reissuing_write():
    writes = 0

    class LostReceiptPool:
        async def write_results(self, _batch_id, _results):
            nonlocal writes
            writes += 1
            raise ConnectionError("receipt lost")

        async def reconcile_results(self, batch_id, results):
            return WritebackReceipt.committed(batch_id, results, atomic=True)

    processor = _writeback_processor(LostReceiptPool())
    outcome = await processor._commit_prepared_results(
        {"a": PreparedResult.create("a", {"answer": "ok"})}
    )

    assert outcome.committed == {"a"}
    assert writes == 1


@pytest.mark.asyncio
async def test_reconciliation_budget_exhaustion_becomes_unresolved(monkeypatch):
    reconciliations = 0

    class UnknownPool:
        async def write_results(self, batch_id, results):
            return WritebackReceipt(
                batch_id=batch_id,
                submitted_ids=tuple(results),
                items=tuple(
                    WritebackItem(
                        record_id,
                        CommitDisposition.INDETERMINATE,
                        "unknown",
                        "unknown",
                        False,
                    )
                    for record_id in results
                ),
                atomic=False,
            )

        async def reconcile_results(self, batch_id, results):
            nonlocal reconciliations
            reconciliations += 1
            return WritebackReceipt(
                batch_id=batch_id,
                submitted_ids=tuple(results),
                items=tuple(
                    WritebackItem(
                        record_id,
                        CommitDisposition.INDETERMINATE,
                        "still_unknown",
                        "still unknown",
                        False,
                    )
                    for record_id in results
                ),
                atomic=False,
            )

    processor = _writeback_processor(UnknownPool(), reconciliation_attempts=2)
    sleep = AsyncMock()
    monkeypatch.setattr("src.core.processor.asyncio.sleep", sleep)
    outcome = await processor._commit_prepared_results(
        {"a": PreparedResult.create("a", {"answer": "ok"})}
    )

    assert outcome.unresolved == {"a"}
    assert reconciliations == 2
    sleep.assert_awaited_once_with(1)


@pytest.mark.asyncio
async def test_malformed_write_receipt_fails_closed():

    class MalformedReceiptPool:
        async def write_results(self, batch_id, _results):
            return WritebackReceipt(
                batch_id=f"wrong-{batch_id}",
                submitted_ids=("record-a",),
                items=(WritebackItem("record-a", CommitDisposition.COMMITTED),),
                atomic=True,
            )

    processor = _writeback_processor(MalformedReceiptPool())
    with pytest.raises(WritebackContractError, match="batch_id mismatch"):
        await processor._commit_prepared_results(
            {
                "record-a": PreparedResult.create(
                    "record-a",
                    {"answer": "value"},
                )
            }
        )


@pytest.mark.asyncio
async def test_malformed_receipt_never_promotes_pending_commit(tmp_path):
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
    tracker.mark_in_flight("a", {"input": "original"})
    prepared = PreparedResult.create("a", {"answer": "ok"})
    tracker.mark_prepared(prepared)

    class MissingItemPool:
        async def write_results(self, batch_id, results):
            return WritebackReceipt(
                batch_id=batch_id,
                submitted_ids=tuple(results),
                items=(),
                atomic=True,
            )

    processor = _writeback_processor(MissingItemPool())
    processor._job_tracker = tracker
    with pytest.raises(WritebackContractError, match="item coverage"):
        await processor._commit_prepared_results({"a": prepared})

    recovered = JobRecordTracker(repository, request.job_id)
    assert recovered.get("a").status == RecordStatus.PENDING_COMMIT
    assert recovered.counts().persisted == 0


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
    processor.source_max_attempts = 1

    with pytest.raises(SourceOperationError, match="source count failed"):
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
        return TaskFailure(
            FailureStage.CONTENT,
            "invalid_or_missing_json",
            "invalid content",
            True,
        )

    processor._process_one_record = content_error
    completed = await processor._process_loop(object())

    assert completed is True
    assert calls == 1
    checkpoint = JobRecordTracker(repository, request.job_id).get("record-a")
    assert checkpoint is not None
    assert checkpoint.status == RecordStatus.FAILED
    assert checkpoint.retry_counts == {"content_error": 1}


@pytest.mark.asyncio
async def test_checkpoint_recovery_reconciles_committed_before_any_rewrite(tmp_path):
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
    tracker.mark_in_flight("a", {"input": "original"})
    tracker.mark_prepared(PreparedResult.create("a", {"answer": "ok"}))
    writes = 0

    class AlreadyCommittedPool:
        async def write_results(self, _batch_id, _results):
            nonlocal writes
            writes += 1
            raise AssertionError("recovery must reconcile before writing")

        async def reconcile_results(self, batch_id, results):
            return WritebackReceipt.committed(batch_id, results, atomic=True)

    recovered = JobRecordTracker(repository, request.job_id)
    processor = _writeback_processor(AlreadyCommittedPool())
    processor._job_tracker = recovered

    outcome = await processor.reconcile_checkpoint_results(
        recovered.pending_prepared_results()
    )

    assert outcome.committed == {"a"}
    assert writes == 0
    assert JobRecordTracker(repository, request.job_id).get("a").status == (
        RecordStatus.PERSISTED
    )


@pytest.mark.asyncio
async def test_checkpoint_recovery_writes_only_after_reconciliation_rejects(tmp_path):
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
    tracker.mark_in_flight("a", {"input": "original"})
    tracker.mark_prepared(PreparedResult.create("a", {"answer": "ok"}))
    calls = []

    class NotCommittedPool:
        async def reconcile_results(self, batch_id, results):
            calls.append("reconcile")
            return WritebackReceipt(
                batch_id=batch_id,
                submitted_ids=tuple(results),
                items=(
                    WritebackItem(
                        "a",
                        CommitDisposition.REJECTED,
                        "not_committed",
                        "expected values are absent",
                        True,
                    ),
                ),
                atomic=True,
            )

        async def write_results(self, batch_id, results):
            calls.append("write")
            return WritebackReceipt.committed(batch_id, results, atomic=True)

    recovered = JobRecordTracker(repository, request.job_id)
    processor = _writeback_processor(NotCommittedPool())
    processor._job_tracker = recovered

    outcome = await processor.reconcile_checkpoint_results(
        recovered.pending_prepared_results()
    )

    assert outcome.committed == {"a"}
    assert calls == ["reconcile", "write"]


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
