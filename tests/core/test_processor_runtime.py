from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pandas as pd
import pytest
import yaml

from src.core.processor import UniversalAIProcessor
from src.core.contracts import FailureStage, PreparedResult, TaskFailure, TaskSuccess
from src.core.job_tracker import JobRecordTracker
from src.config import execution_config_hash, load_config
from src.jobs import FileJobRepository, RecordStatus


def _processor_config(tmp_path, *, routing: bool = False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_path / "input.csv"
    pd.DataFrame([{"input": "hello", "category": "special", "result": None}]).to_csv(
        csv_path, index=False
    )
    config = {
        "schema_version": 4,
        "runtime": {
            "log": {"level": "error", "format": "text", "output": "console"},
            "auth": {"token": ""},
            "workspace": {
                "roots": {"project": str(tmp_path)},
                "state_dir": ".dataflux/jobs",
            },
        },
        "job": {
            "gateway_url": "http://127.0.0.1:8787",
            "datasource": {
                "type": "csv",
                "input_path": str(csv_path),
                "output_path": str(csv_path),
                "engine": "pandas",
                "require_all_input_fields": True,
            },
            "columns": {"extract": ["input"], "write": {"answer": "result"}},
            "concurrency": {
                "batch_size": 1,
                "max_in_flight": 1,
            },
            "retry": {
                "task_max_attempts": {
                    "api_error": 2,
                    "content_error": 2,
                    "system_error": 2,
                    "source_error": 2,
                },
            },
            "prompt": {
                "template": "Process {record_json}",
                "required_fields": ["answer"],
                "use_json_schema": True,
                "temperature": 0.2,
                "temperature_override": True,
                "system_prompt": "system",
            },
            "validation": {"enabled": False, "field_rules": {}},
        },
    }
    if routing:
        profile = {
            "prompt": {
                "template": "Routed {record_json}",
                "temperature": 0.1,
                "temperature_override": False,
                "system_prompt": "routed-system",
            }
        }
        (tmp_path / "special.yaml").write_text(
            yaml.safe_dump(profile), encoding="utf-8"
        )
        config["job"]["routing"] = {
            "enabled": True,
            "field": "category",
            "subtasks": [{"match": "special", "profile": "special.yaml"}],
        }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path, csv_path


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "selection,model,group",
    [
        ({"mode": "auto"}, "auto", None),
        ({"mode": "strict", "route_id": "chosen"}, "chosen", None),
        ({"mode": "fallback_group", "group": "primary"}, "auto", "primary"),
    ],
)
async def test_canonical_model_selection_reaches_gateway_payload(
    tmp_path, selection, model, group
):
    path, _ = _processor_config(tmp_path, routing=True)
    config = yaml.safe_load(path.read_text())
    config["job"]["model_selection"] = selection
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    processor = UniversalAIProcessor(str(path))
    processor.client.call = AsyncMock(return_value='{"answer":"ok"}')
    try:
        outcome = await processor._process_one_record(
            object(), "a", {"input": "hello", "category": "special"}
        )
        assert isinstance(outcome, TaskSuccess)
        assert processor.client.call.call_args.kwargs["model"] == model
        assert processor.client.call.call_args.kwargs.get("fallback_group") == group
    finally:
        await processor.task_pool.aclose()


@pytest.mark.asyncio
async def test_processor_initializes_and_persists_successful_csv(tmp_path):
    config_path, csv_path = _processor_config(tmp_path)
    processor = UniversalAIProcessor(str(config_path))
    repository = FileJobRepository(tmp_path / "job-state")
    request = repository.new_request(
        mode="background",
        config_path=str(config_path),
        config_sha256=execution_config_hash(load_config(config_path), config_path),
    )
    repository.create_job(request)
    tracker = JobRecordTracker(repository, request.job_id)
    processor.configure_job_control(job_tracker=tracker)

    async def process_one(_session, _record_id, row_data):
        assert row_data["input"] == "hello"
        return TaskSuccess(PreparedResult.create(_record_id, {"answer": "done"}))

    processor._process_one_record = process_one
    completed = await processor.process_shard_async_continuous()

    assert completed is True
    assert processor.task_manager.total_processed_successfully == 1
    assert pd.read_csv(csv_path).loc[0, "result"] == "done"
    shards = repository.list_shards(request.job_id)
    assert [shard.shard_id for shard in shards] == ["scan-000001"]
    assert shards[0].records[0].record_id == 0
    assert shards[0].records[0].status == RecordStatus.PERSISTED


@pytest.mark.asyncio
async def test_cancelling_processor_drains_tasks_from_later_scan_waves(tmp_path):
    path, csv_path = _processor_config(tmp_path)
    csv_path.write_text("input,result\nfirst,\nsecond,\n", encoding="utf-8")
    processor = UniversalAIProcessor(str(path))
    started, drained = asyncio.Event(), asyncio.Event()

    async def process_one(_session, record_id, _data):
        if record_id == 0:
            return TaskSuccess(PreparedResult.create(record_id, {"answer": "done"}))
        started.set()
        try:
            await asyncio.sleep(10)
        finally:
            drained.set()

    processor._process_one_record = process_one
    task = asyncio.create_task(processor.process_shard_async_continuous())
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert drained.is_set()
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await processor.task_pool.aclose()


@pytest.mark.asyncio
async def test_processor_routing_context_and_ai_error_classification(tmp_path):
    config_path, _csv_path = _processor_config(tmp_path, routing=True)
    processor = UniversalAIProcessor(str(config_path))
    session = object()
    try:
        assert processor.routing_field_is_implicit is True
        assert processor._get_routing_context({}) is None
        assert processor._get_routing_context({"category": "unknown"}) is None
        assert processor._get_routing_context({"category": "special"}) is not None

        processor.client.call = AsyncMock(return_value='{"answer":"routed"}')
        result = await processor._process_one_record(
            session, "record-a", {"input": "hello", "category": "special"}
        )
        assert isinstance(result, TaskSuccess)
        assert result.prepared_result.values == {"answer": "routed"}
        call = processor.client.call.call_args
        assert call.kwargs["temperature"] is None
        assert call.args[1][0] == {"role": "system", "content": "routed-system"}

        response_error = aiohttp.ClientResponseError(
            MagicMock(),
            (),
            status=429,
            message="rate limited",
            headers={"Retry-After": "3"},
        )
        processor.client.call = AsyncMock(side_effect=response_error)
        rate_limited = await processor._process_one_record(
            session, "record-a", {"input": "hello"}
        )
        assert isinstance(rate_limited, TaskFailure)
        assert rate_limited.stage == FailureStage.MODEL
        assert rate_limited.retryable is True
        assert rate_limited.retry_after_seconds == 3

        processor.client.call = AsyncMock(
            side_effect=aiohttp.ClientResponseError(
                MagicMock(), (), status=400, message="invalid", headers={}
            )
        )
        permanent = await processor._process_one_record(
            session, "record-a", {"input": "hello"}
        )
        assert isinstance(permanent, TaskFailure)
        assert permanent.retryable is False

        processor.client.call = AsyncMock(side_effect=TimeoutError("timeout"))
        timeout = await processor._process_one_record(
            session, "record-a", {"input": "hello"}
        )
        assert isinstance(timeout, TaskFailure)
        assert timeout.retryable is True

        processor.client.call = AsyncMock(side_effect=RuntimeError("broken"))
        system = await processor._process_one_record(
            session, "record-a", {"input": "hello"}
        )
        assert isinstance(system, TaskFailure)
        assert system.stage == FailureStage.SYSTEM
    finally:
        processor.task_pool.close()


def test_processor_progress_file_write_cleanup_and_failure(tmp_path, monkeypatch):
    config_path, _csv_path = _processor_config(tmp_path)
    progress_path = tmp_path / "progress.json"
    processor = UniversalAIProcessor(str(config_path), progress_file=str(progress_path))
    try:
        processor.task_manager.total_shards = 2
        processor.task_manager.current_shard_index = 1
        processor.task_manager.total_estimated = 3
        processor.task_manager.total_processed_successfully = 1
        processor._write_progress()
        payload = json.loads(progress_path.read_text(encoding="utf-8"))
        assert payload["shard"] == "1/2"
        assert payload["processed"] == 1

        monkeypatch.setattr(
            "src.core.processor.os.replace", MagicMock(side_effect=OSError)
        )
        processor._write_progress()
        monkeypatch.undo()
        processor._cleanup_progress()
        assert not progress_path.exists()
        processor._cleanup_progress()
    finally:
        processor.task_pool.close()


def test_processor_run_cleans_progress_on_success(tmp_path):
    config_path, _csv_path = _processor_config(tmp_path)
    progress_path = tmp_path / "progress.json"
    processor = UniversalAIProcessor(str(config_path), progress_file=str(progress_path))

    async def process_one(_session, _record_id, _row_data):
        return TaskSuccess(PreparedResult.create(_record_id, {"answer": "sync"}))

    processor._process_one_record = process_one
    progress_path.write_text("{}", encoding="utf-8")
    assert processor.run() is True
    assert not progress_path.exists()


@pytest.mark.asyncio
async def test_processor_empty_datasource_finishes_without_http_session(tmp_path):
    config_path, csv_path = _processor_config(tmp_path)
    csv_path.write_text("input,category,result\n", encoding="utf-8")
    processor = UniversalAIProcessor(str(config_path))

    assert await processor.process_shard_async_continuous() is True


def test_processor_reports_config_datasource_and_routing_initialization_errors(
    tmp_path,
):
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("invalid: yaml: value:", encoding="utf-8")
    with pytest.raises(ValueError, match="无法加载配置文件"):
        UniversalAIProcessor(str(invalid))

    config_path, csv_path = _processor_config(tmp_path)
    csv_path.unlink()
    with pytest.raises(RuntimeError, match="无法初始化数据源任务池"):
        UniversalAIProcessor(str(config_path))

    config_path, _csv_path = _processor_config(tmp_path / "routing")
    processor = UniversalAIProcessor(str(config_path))
    try:
        processor.routing_enabled = True
        processor.config["routing"] = {"subtasks": []}
        with pytest.raises(ValueError, match="缺少 subtasks"):
            processor._init_routing_contexts()

        processor.config["routing"] = {"subtasks": [{"match": "x"}]}
        with pytest.raises(ValueError, match="match 和 profile"):
            processor._init_routing_contexts()
    finally:
        processor.task_pool.close()
