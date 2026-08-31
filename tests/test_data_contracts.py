from __future__ import annotations

import pytest

from src.data.base import BaseTaskPool
from src.data.contracts import (
    CommitDisposition,
    WritebackContractError,
    WritebackItem,
    WritebackReceipt,
    declared_adapter_capabilities,
    validate_writeback_receipt,
)


class _LegacyPool(BaseTaskPool):
    def __init__(self):
        super().__init__(["input"], {"answer": "output"})
        self.rows = [(1, {"input": "a"}), (2, {"input": "b"})]
        self.return_receipt = False
        self.closed = False

    def get_total_task_count(self):
        return len(self.rows)

    def get_processed_task_count(self):
        return 0

    def get_id_boundaries(self):
        return (1, 2) if self.rows else (None, None)

    def initialize_shard(self, _shard_id, _min_id, _max_id):
        self.tasks = list(self.rows)
        return len(self.tasks)

    def get_task_batch(self, batch_size):
        batch = self.tasks[:batch_size]
        self.tasks = self.tasks[batch_size:]
        return batch

    def update_task_results(self, batch_id, results):
        if self.return_receipt:
            return WritebackReceipt.committed(batch_id, results, atomic=False)
        return None

    def reload_task_data(self, task_id):
        return next((data for rid, data in self.rows if rid == task_id), None)

    def close(self):
        self.closed = True


class _FullScanPool(_LegacyPool):
    def fetch_all_rows(self, columns):
        return [{column: "all" for column in columns}, {columns[0]: "second"}]

    def fetch_all_processed_rows(self, columns):
        return [{column: "processed" for column in columns}]

    def sample_processed_rows(self, sample_size):
        return [{"output": "done"}][:sample_size]


@pytest.mark.asyncio
async def test_legacy_adapter_bridge_scan_reload_write_and_close():
    pool = _LegacyPool()
    pool.add_task_to_front("ignored", None)
    pool.add_task_to_back("ignored", None)
    pool.add_task_to_front(0, {"input": "front"})
    pool.add_task_to_back(3, {"input": "back"})
    assert pool.has_tasks() and pool.get_remaining_count() == 2
    pool.clear_tasks()
    assert not pool.has_tasks()

    first = await pool.scan(None, 1)
    second = await pool.scan(first.next_cursor, 2)
    assert [record.record_id for record in first.records] == [1]
    assert [record.record_id for record in second.records] == [2]
    assert second.next_cursor is None
    with pytest.raises(ValueError):
        await pool.scan({"bad": True}, 1)
    with pytest.raises(ValueError):
        await pool.scan(None, 0)

    assert await pool.reload([1, 999]) == {1: {"input": "a"}}
    with pytest.raises(WritebackContractError, match="must return WritebackReceipt"):
        await pool.write_results(
            "batch-a", {1: {"answer": "ok"}, 2: {"_error": "skip"}}
        )
    pool.return_receipt = True
    receipt = await pool.write_results("batch-b", {1: {"answer": "ok"}})
    assert receipt.batch_id == "batch-b"
    assert receipt.committed_ids == (1,)

    reconciliation = await pool.reconcile_results(
        "batch-c",
        {1: {"answer": "ok"}},
    )
    assert reconciliation.items[0].disposition == CommitDisposition.INDETERMINATE

    await pool.aclose()
    assert pool.closed is True
    pool.closed = False
    pool.close_readonly()
    assert pool.closed is True


@pytest.mark.asyncio
async def test_sampling_full_scan_defaults_and_failure_receipt():
    pool = _FullScanPool()
    assert pool.capabilities.full_scan is True
    assert await pool.sample("unprocessed", 1) == [{"input": "a"}]
    assert await pool.sample("processed", 1) == [{"output": "done"}]
    assert await pool.sample("all", None) == [
        {"input": "all"},
        {"input": "second"},
    ]
    assert await pool.sample("all", 1, ["input"]) == [{"input": "all"}]
    assert await pool.sample("all_processed", 1) == [{"output": "processed"}]
    assert await pool.sample("processed", 0) == []
    with pytest.raises(ValueError, match="unsupported"):
        await pool.sample("bad", 1)

    failure = pool.failed_receipt(
        "batch", {1: {}, 2: {}}, ValueError("invalid"), retryable=False
    )
    assert [item.record_id for item in failure.items] == [1, 2]
    assert all(not item.retryable for item in failure.items)

    legacy = _LegacyPool()
    legacy.rows = []
    assert legacy.sample_unprocessed_rows(2) == []
    assert legacy.sample_processed_rows(2) == []
    assert legacy.fetch_all_rows(["input"]) == []
    assert legacy.fetch_all_processed_rows(["output"]) == []
    assert legacy.capabilities.full_scan is False


def test_declared_capability_registry_and_unknown_type():
    assert declared_adapter_capabilities("SQLite").atomic_batch is True
    assert declared_adapter_capabilities("csv").atomic_batch is False
    with pytest.raises(ValueError, match="unsupported"):
        declared_adapter_capabilities("unknown")


@pytest.mark.parametrize(
    "receipt",
    [
        WritebackReceipt(
            batch_id="wrong",
            submitted_ids=("a", "b"),
            items=(
                WritebackItem("a", CommitDisposition.COMMITTED),
                WritebackItem("b", CommitDisposition.COMMITTED),
            ),
            atomic=True,
        ),
        WritebackReceipt(
            batch_id="batch",
            submitted_ids=("a",),
            items=(WritebackItem("a", CommitDisposition.COMMITTED),),
            atomic=True,
        ),
        WritebackReceipt(
            batch_id="batch",
            submitted_ids=("a", "b", "unknown"),
            items=(
                WritebackItem("a", CommitDisposition.COMMITTED),
                WritebackItem("b", CommitDisposition.COMMITTED),
                WritebackItem("unknown", CommitDisposition.COMMITTED),
            ),
            atomic=True,
        ),
        WritebackReceipt(
            batch_id="batch",
            submitted_ids=("a", "a"),
            items=(
                WritebackItem("a", CommitDisposition.COMMITTED),
                WritebackItem("a", CommitDisposition.REJECTED),
            ),
            atomic=True,
        ),
        WritebackReceipt(
            batch_id="batch",
            submitted_ids=("a", "b"),
            items=(WritebackItem("a", CommitDisposition.COMMITTED),),
            atomic=True,
        ),
        WritebackReceipt(
            batch_id="batch",
            submitted_ids=("a", "b"),
            items=(
                WritebackItem("a", CommitDisposition.COMMITTED),
                WritebackItem("unknown", CommitDisposition.COMMITTED),
            ),
            atomic=True,
        ),
        WritebackReceipt(
            batch_id="batch",
            submitted_ids=("a", "b"),
            items=(
                WritebackItem("a", CommitDisposition.COMMITTED),
                WritebackItem("a", CommitDisposition.REJECTED),
            ),
            atomic=True,
        ),
        WritebackReceipt(
            batch_id="batch",
            submitted_ids=("a", "b"),
            items=(
                WritebackItem("a", CommitDisposition.COMMITTED, retryable=True),
                WritebackItem("b", CommitDisposition.COMMITTED),
            ),
            atomic=True,
        ),
    ],
    ids=[
        "wrong-batch",
        "missing-submitted-id",
        "unknown-submitted-id",
        "duplicate-submitted-id",
        "missing-item",
        "unknown-item",
        "conflicting-duplicate-item",
        "committed-retryable",
    ],
)
def test_receipt_contract_rejects_malformed_coverage(receipt):
    with pytest.raises(WritebackContractError):
        validate_writeback_receipt(
            receipt,
            batch_id="batch",
            submitted_ids=("a", "b"),
        )
