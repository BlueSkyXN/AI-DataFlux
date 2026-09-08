from __future__ import annotations

import asyncio
import threading
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4
from unittest.mock import AsyncMock

import pytest
import yaml

from src.config import execution_config_hash, load_config
from src.core import job_runner
from src.core.contracts import PreparedResult, WritebackOutcome
from src.core.contracts import TaskSuccess
from src.data.excel import ExcelTaskPool
from src.core.job_runner import resolve_terminal_status
from src.core.job_tracker import JobRecordTracker
from src.jobs import (
    CommandReceipt,
    FileJobRepository,
    JobCommand,
    JobStatus,
)


def _repository_with_job(tmp_path, suffix=".csv", in_place=False, datasource=None):
    config_path = tmp_path / "config.yaml"
    config = {
        "schema_version": 4,
        "runtime": {
            "workspace": {
                "roots": {"project": str(tmp_path)},
                "state_dir": ".dataflux/jobs",
            }
        },
        "job": {
            "datasource": {
                "type": "csv" if suffix == ".csv" else "excel",
                "input_path": str(tmp_path / f"input{suffix}"),
                "output_path": str(
                    tmp_path / f"{'input' if in_place else 'output'}{suffix}"
                ),
                "engine": "pandas",
            },
            "columns": {"extract": ["input"], "write": {"answer": "result"}},
            "prompt": {"template": "v1 {record_json}"},
            "concurrency": {"batch_size": 1, "max_in_flight": 1},
        },
    }
    if datasource is not None:
        config["job"]["datasource"] = datasource
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    repository = FileJobRepository(tmp_path / "jobs")
    request = repository.new_request(
        mode="background",
        config_path=str(config_path),
        config_sha256=execution_config_hash(load_config(config_path), config_path),
    )
    repository.create_job(request)
    return repository, request, config_path


@pytest.mark.asyncio
async def test_bitable_resume_replays_native_id_then_processes_multiple_pages(
    tmp_path, monkeypatch
):
    from src.data.feishu.bitable import FeishuBitableTaskPool

    datasource = {
        "type": "feishu_bitable",
        "app_id": "test-app",
        "app_secret": "test-secret",
        "app_token": "test-base",
        "table_id": "test-table",
    }
    repository, request, _ = _repository_with_job(tmp_path, datasource=datasource)
    rows = [
        {"record_id": f"rec-{name}", "fields": {"input": name, "result": ""}}
        for name in ("X", "A", "B", "C")
    ]

    async def update(_app, _table, values):
        for value in values:
            next(row for row in rows if row["record_id"] == value["record_id"])[
                "fields"
            ].update(value["fields"])
        return values

    client = SimpleNamespace(
        bitable_list_records=AsyncMock(return_value=rows),
        bitable_batch_update=AsyncMock(side_effect=update),
        close=AsyncMock(),
    )
    monkeypatch.setattr(
        "src.data.feishu.bitable.FeishuClient", lambda **_kwargs: client
    )
    pool = FeishuBitableTaskPool(
        "test-app",
        "test-secret",
        "test-base",
        "test-table",
        ["input"],
        {"answer": "result"},
    )
    identity = await pool.recovery_identity()
    repository.update_state(
        request.job_id, lambda state: replace(state, source_identity=identity)
    )
    tracker = JobRecordTracker(repository, request.job_id)
    tracker.mark_in_flight("rec-A", {"input": "A"})
    tracker.mark_prepared(PreparedResult.create("rec-A", {"answer": "checkpointed-A"}))
    calls = []

    async def model(_self, _session, record_id, row):
        calls.append(record_id)
        return TaskSuccess(
            PreparedResult.create(record_id, {"answer": f"answer-{row['input']}"})
        )

    monkeypatch.setattr(job_runner.UniversalAIProcessor, "_process_one_record", model)
    result = await job_runner.run_processing_job(
        request.job_id, request, repository, lambda: 1, asyncio.Event()
    )
    assert result.status == JobStatus.COMPLETED
    assert calls == ["rec-X", "rec-B", "rec-C"]
    assert [row["fields"]["result"] for row in rows] == [
        "answer-X",
        "checkpointed-A",
        "answer-B",
        "answer-C",
    ]
    assert repository.get_state(request.job_id).counts.persisted == 4


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("suffix", "mutation", "in_place"),
    [
        (suffix, mutation, False)
        for suffix in (".csv", ".xlsx")
        for mutation in (
            None,
            "reordered",
            "missing_output",
            "changed_output",
            "missing_identity",
        )
    ]
    + [(".csv", None, True), (".xlsx", None, True)],
)
async def test_file_job_resume_preserves_committed_output_or_blocks(
    tmp_path, monkeypatch, suffix, mutation, in_place
):
    import pandas as pd

    repository, request, _ = _repository_with_job(tmp_path, suffix, in_place)
    source, output = tmp_path / f"input{suffix}", tmp_path / f"output{suffix}"
    if in_place:
        output = source
    frame = pd.DataFrame({"input": ["A", "B"], "result": [None, None]})

    def write_frame(data, path):
        if suffix == ".csv":
            data.to_csv(path, index=False)
        else:
            data.to_excel(path, index=False)

    write_frame(frame, source)
    pool = ExcelTaskPool(source, output, ["input"], {"answer": "result"})
    identity = await pool.recovery_identity()
    repository.update_state(
        request.job_id, lambda state: replace(state, source_identity=identity)
    )
    assert repository.get_state(request.job_id).source_identity == identity
    assert "source_identity" not in repository.get_state(request.job_id).to_dict()
    tracker = JobRecordTracker(repository, request.job_id)
    tracker.mark_in_flight(0, {"input": "A"})
    prepared = PreparedResult.create(0, {"answer": "answer-A"})
    tracker.mark_prepared(prepared)
    assert (
        await pool.write_results("first", {0: dict(prepared.values)})
    ).committed_ids == (0,)
    tracker.mark_persisted(0)
    await pool.aclose()
    if mutation == "reordered":
        write_frame(frame.iloc[::-1], source)
    elif mutation == "missing_output":
        output.unlink()
    elif mutation == "changed_output":
        write_frame(frame, output)
    elif mutation == "missing_identity":
        repository.update_state(
            request.job_id, lambda state: replace(state, source_identity=None)
        )

    calls = []

    async def model(_self, _session, record_id, row):
        calls.append(record_id)
        return TaskSuccess(
            PreparedResult.create(record_id, {"answer": f"answer-{row['input']}"})
        )

    monkeypatch.setattr(job_runner.UniversalAIProcessor, "_process_one_record", model)
    result = await job_runner.run_processing_job(
        request.job_id, request, repository, lambda: 1, asyncio.Event()
    )
    if mutation:
        assert result.status == JobStatus.BLOCKED
        assert not calls
        return
    assert result.status == JobStatus.COMPLETED
    assert calls == [1]
    assert repository.get_state(request.job_id).counts.persisted == 2
    actual = pd.read_csv(output) if suffix == ".csv" else pd.read_excel(output)
    assert actual["result"].tolist() == ["answer-A", "answer-B"]


@pytest.mark.asyncio
async def test_job_runner_reconciles_pending_commit_before_scanning(
    tmp_path,
    monkeypatch,
):
    repository, request, _config_path = _repository_with_job(tmp_path)
    tracker = JobRecordTracker(repository, request.job_id)
    tracker.mark_in_flight("record-a", {"input": "hello"})
    prepared = PreparedResult.create("record-a", {"answer": "checkpointed"})
    tracker.mark_prepared(prepared)
    replayed = []

    class FakeProcessor:
        def __init__(self, _path):
            self.task_pool = SimpleNamespace(
                aclose=AsyncMock(),
                recovery_identity=AsyncMock(return_value=None),
                prepare_resume=AsyncMock(),
            )
            self.task_manager = SimpleNamespace(
                total_processed_successfully=0,
                max_retries_exceeded_count=0,
                total_estimated=1,
                retried_tasks_count={},
            )

        def configure_job_control(self, **kwargs):
            self.tracker = kwargs["job_tracker"]

        async def reconcile_checkpoint_results(self, results):
            replayed.append(results)
            for record_id in results:
                self.tracker.mark_persisted(record_id)
            return WritebackOutcome(frozenset(results), frozenset(), frozenset())

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
    assert replayed == [{"record-a": prepared}]
    assert repository.get_state(request.job_id).counts.persisted == 1


@pytest.mark.asyncio
async def test_changed_config_requires_explicit_accepted_resume_hash(
    tmp_path, monkeypatch
):
    repository, request, config_path = _repository_with_job(tmp_path)
    changed = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    changed["job"]["prompt"]["template"] = "v2 {record_json}"
    config_path.write_text(yaml.safe_dump(changed), encoding="utf-8")

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
        payload={
            "accepted_config_sha256": execution_config_hash(
                load_config(config_path), config_path
            )
        },
    )
    repository.add_command(command)
    repository.save_command_receipt(
        CommandReceipt(command.command_id, request.job_id, accepted=True)
    )

    class EmptyProcessor:
        def __init__(self, _path):
            self.task_pool = SimpleNamespace(
                aclose=AsyncMock(),
                recovery_identity=AsyncMock(return_value=None),
                prepare_resume=AsyncMock(),
            )
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


@pytest.mark.parametrize(
    ("cancelled", "job_failed", "unresolved", "failed", "expected"),
    [
        (True, True, 1, 1, JobStatus.CANCELLED),
        (False, True, 1, 1, JobStatus.FAILED),
        (
            False,
            False,
            1,
            1,
            JobStatus.COMPLETED_WITH_UNRESOLVED_WRITES,
        ),
        (False, False, 0, 1, JobStatus.COMPLETED_WITH_ERRORS),
        (False, False, 0, 0, JobStatus.COMPLETED),
    ],
)
def test_terminal_status_priority(
    cancelled,
    job_failed,
    unresolved,
    failed,
    expected,
):
    assert (
        resolve_terminal_status(
            cancelled=cancelled,
            job_failed=job_failed,
            unresolved_writes=unresolved,
            failed_records=failed,
        )
        == expected
    )


@pytest.mark.asyncio
async def test_processor_initialization_does_not_block_control_loop(
    tmp_path, monkeypatch
):
    repository, request, _ = _repository_with_job(tmp_path)
    entered, release = threading.Event(), threading.Event()
    closed = AsyncMock()

    class SlowProcessor:
        def __init__(self, _path):
            entered.set()
            assert release.wait(timeout=2)
            self.task_pool = SimpleNamespace(aclose=closed)

    monkeypatch.setattr(job_runner, "UniversalAIProcessor", SlowProcessor)
    task = asyncio.create_task(
        job_runner.run_processing_job(
            request.job_id, request, repository, lambda: 1, asyncio.Event()
        )
    )
    try:
        for _ in range(50):
            if entered.is_set():
                break
            await asyncio.sleep(0.01)
        assert entered.is_set()
        task.cancel()
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        closed.assert_awaited_once()
    finally:
        release.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
