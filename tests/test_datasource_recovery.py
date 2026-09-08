"""数据身份、恢复与分页的组合回归；外部 API 均使用模拟响应。"""

from unittest.mock import AsyncMock

import pytest

from src.data.feishu.bitable import FeishuBitableTaskPool
from src.data.feishu.sheet import FeishuSheetTaskPool
from src.data.excel import ExcelTaskPool


@pytest.mark.parametrize(
    "kind,resource",
    [
        ("feishu_bitable", {"app_token": "test-base", "table_id": "test-table"}),
        ("feishu_sheet", {"spreadsheet_token": "test-sheet", "sheet_id": "0"}),
    ],
)
def test_canonical_feishu_config_reaches_factory(kind, resource):
    from src.config import RootConfig, compile_job_config
    from src.data.factory import create_task_pool

    root = RootConfig.model_validate(
        {
            "schema_version": 4,
            "runtime": {},
            "job": {
                "datasource": {
                    "type": kind,
                    "app_id": "test-app",
                    "app_secret": "test-secret",
                    **resource,
                },
                "columns": {"extract": ["input"], "write": {"answer": "result"}},
                "prompt": {"template": "{record_json}"},
            },
        }
    )
    pool = create_task_pool(compile_job_config(root), ["input"], {"answer": "result"})
    try:
        assert isinstance(
            pool,
            FeishuBitableTaskPool if kind == "feishu_bitable" else FeishuSheetTaskPool,
        )
    finally:
        pool.close()


def bitable(records):
    pool = FeishuBitableTaskPool(
        "test-app",
        "test-secret",
        "test-base",
        "test-table",
        ["input"],
        {"answer": "output"},
    )
    pool.client.bitable_list_records = AsyncMock(return_value=records)
    pool.client.bitable_batch_update = AsyncMock(
        side_effect=lambda app, table, rows: rows
    )
    return pool


def sheet(rows):
    pool = FeishuSheetTaskPool(
        "test-app", "test-secret", "test-sheet", "0", ["input"], {"answer": "output"}
    )
    pool._header_row = ["input", "output"]
    pool._col_name_to_index = {"input": 0, "output": 1}
    pool._data_rows = rows
    pool._snapshot_loaded = True
    return pool


@pytest.mark.asyncio
async def test_sheet_resume_identity_detects_insert_but_ignores_own_output():
    first = sheet([["A", ""], ["B", ""]])
    written = sheet([["A", "answer-A"], ["B", ""]])
    inserted = sheet([["X", ""], ["A", "answer-A"], ["B", ""]])
    assert await first.recovery_identity() == await written.recovery_identity()
    assert await first.recovery_identity() != await inserted.recovery_identity()


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [0, 1, 2, 3, 4, 5])
async def test_sheet_scan_advances_through_all_pages(count):
    pool = sheet([[str(i), ""] for i in range(count)])
    cursor, found = None, []
    while True:
        page = await pool.scan(cursor, 2)
        found.extend(record.record_id for record in page.records)
        if page.next_cursor is None:
            break
        assert page.next_cursor != cursor
        cursor = page.next_cursor
    assert found == list(range(count))


@pytest.mark.asyncio
async def test_same_output_path_cannot_have_two_active_file_writers(tmp_path):
    source, output = tmp_path / "input.csv", tmp_path / "output.csv"
    source.write_text("input,output\nA,\nB,\n", encoding="utf-8")
    a, b = [
        ExcelTaskPool(source, output, ["input"], {"answer": "output"}) for _ in range(2)
    ]
    try:
        await a.scan(None, 1)
        with pytest.raises(TimeoutError):
            await b.scan(None, 1)
        assert not output.exists()
        await a.aclose()
        assert len((await b.scan(None, 1)).records) == 1
    finally:
        await a.aclose()
        await b.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [0, 1, 2, 3, 4, 5])
async def test_bitable_scan_advances_through_all_pages(count):
    records = [
        {"record_id": f"rec-{i}", "fields": {"input": str(i), "output": ""}}
        for i in range(count)
    ]
    pool = bitable(records)
    try:
        cursor, found = None, []
        while True:
            page = await pool.scan(cursor, 2)
            found.extend(record.record_id for record in page.records)
            if page.next_cursor is None:
                break
            assert page.next_cursor != cursor
            cursor = page.next_cursor
        assert found == [record["record_id"] for record in records]
    finally:
        await pool.aclose()


@pytest.mark.asyncio
async def test_bitable_reordered_snapshot_keeps_native_record_identity():
    pool = bitable(
        [
            {"record_id": "rec-B", "fields": {"input": "B", "output": ""}},
            {"record_id": "rec-A", "fields": {"input": "A", "output": ""}},
        ]
    )
    try:
        receipt = await pool.write_results("resume", {"rec-A": {"answer": "answer-A"}})
        assert receipt.committed_ids == ("rec-A",)
        assert pool.client.bitable_batch_update.call_args.args[2] == [
            {"record_id": "rec-A", "fields": {"output": "answer-A"}}
        ]
        assert await pool.reload(["rec-A"]) == {"rec-A": {"input": "A"}}
        rejected = await pool.write_results("old-position", {0: {"answer": "wrong"}})
        assert not rejected.committed_ids
        assert pool.client.bitable_batch_update.await_count == 1
    finally:
        await pool.aclose()
