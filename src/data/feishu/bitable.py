"""
飞书多维表格（Bitable）数据源任务池

本模块实现基于飞书多维表格的 TaskPool，适用于将多维表格作为
AI-DataFlux 的数据源进行批量 AI 处理。

核心设计（对应云端表格与本地文件的差异）:
    1. 快照读取 —— 初始化时一次性拉取全部记录到内存，后续操作基于快照
    2. ID 映射表 —— 连续整数 task_id ↔ 字符串 record_id 的稳定映射
    3. 写入控制 —— 批量更新上限 1000 条/次，自动分块
    4. 部分失败追溯 —— 每条记录的写入状态独立跟踪
    5. Token 自动刷新 —— 由 FeishuClient 透明处理

分片策略:
    与数据库数据源不同，Bitable 的 record_id 是字符串（如 recXXXXXX），
    不是连续数字。因此：
    - get_id_boundaries() 返回 (0, total_records - 1)
    - 使用连续整数 task_id 做分片，通过映射表查找 record_id
    - initialize_shard() 按 task_id 范围从快照中加载

类清单:
    FeishuBitableTaskPool(BaseTaskPool)
        飞书多维表格数据源任务池，继承自 BaseTaskPool

关键变量:
    _snapshot       — list[dict]: 内存快照，元素为 {"record_id": str, "fields": dict}
    _id_map         — dict[int, str]: task_id → record_id 正向映射
    _reverse_map    — dict[str, int]: record_id → task_id 反向映射
    _snapshot_loaded — bool: 快照是否已加载（双重检查锁保护）

方法清单:
    快照管理:
        _load_snapshot_sync()                    — 同步加载快照（线程安全，双重检查锁）
        _load_snapshot()                         — [async] 从飞书拉取全部记录到内存
        _get_fields(record) → dict               — 提取记录的 fields 字典
        _is_unprocessed(fields) → bool            — 判断记录是否未处理（输入有效 & 输出缺失）
        _is_processed(fields) → bool              — 判断记录是否已处理（所有输出列非空）
        _field_not_empty(value) → bool            — [static] 判断字段值是否非空

    BaseTaskPool 接口实现:
        get_total_task_count() → int              — 统计未处理任务总数
        get_processed_task_count() → int          — 统计已处理任务总数
        get_id_boundaries() → (int, int)          — 返回 task_id 边界 (0, N-1)
        initialize_shard(shard_id, min_id, max_id) → int — 加载分片内未处理任务
        get_task_batch(batch_size) → list          — 从内存队列弹出一批任务
        update_task_results(results)               — 批量写回结果到飞书
        reload_task_data(task_id) → dict|None      — 从快照重新加载任务数据
        close()                                    — 关闭飞书客户端连接

    工具方法:
        _convert_field_value(value) → str          — [static] 飞书字段值转字符串
        _batch_update(records)                     — [async] 异步执行批量更新并同步快照

    Token 估算采样:
        sample_unprocessed_rows(sample_size) → list — 采样未处理行（用于 Token 估算）
        sample_processed_rows(sample_size) → list   — 采样已处理行（用于 Token 估算）

模块依赖:
    logging, threading           — 日志与线程安全
    ..base.BaseTaskPool          — 任务池基类
    .run_async                   — 同步-异步桥接
    .client.FeishuClient         — 飞书 HTTP 客户端

使用示例:
    pool = FeishuBitableTaskPool(
        app_id="cli_xxx",
        app_secret="xxx",
        app_token="bascXXX",
        table_id="tblXXX",
        columns_to_extract=["问题", "上下文"],
        columns_to_write={"answer": "AI回答", "category": "分类"},
    )
"""

import asyncio
import logging
import threading
from typing import Any

from ..base import BaseTaskPool
from ..contracts import (
    AdapterCapabilities,
    CommitDisposition,
    WritebackItem,
    WritebackReceipt,
)
from . import run_async
from .client import (
    BITABLE_BATCH_UPDATE_LIMIT,
    FeishuAPIError,
    FeishuClient,
    FeishuRateLimitError,
)


def _bitable_error_item(record_id: str, error: BaseException) -> WritebackItem:
    """将飞书写入异常分类为明确拒绝或结果不明。"""

    message = f"{type(error).__name__}: {error}"
    if (
        isinstance(error, FeishuAPIError)
        and error.code not in {-1}
        and not (500 <= error.code <= 599)
    ):
        return WritebackItem(
            record_id,
            CommitDisposition.REJECTED,
            "bitable_api_rejected",
            message,
            isinstance(error, FeishuRateLimitError) or error.code == 429,
        )
    return WritebackItem(
        record_id,
        CommitDisposition.INDETERMINATE,
        "bitable_write_indeterminate",
        message,
        False,
    )


class FeishuBitableTaskPool(BaseTaskPool):
    """
    飞书多维表格数据源任务池

    从飞书多维表格读取未处理的记录，AI 处理后将结果写回对应字段。

    Attributes:
        client: 飞书异步 HTTP 客户端
        app_token: 多维表格 App Token
        table_id: 数据表 ID
        _snapshot: 内存快照 [{record_id, fields}, ...]
        _id_map: task_id → record_id 映射
        _reverse_map: record_id → task_id 映射
    """

    def __init__(
        self,
        app_id: str,
        app_secret: str,
        app_token: str,
        table_id: str,
        columns_to_extract: list[str],
        columns_to_write: dict[str, str],
        require_all_input_fields: bool = True,
        max_retries: int = 3,
        qps_limit: float = 0,
    ):
        super().__init__(columns_to_extract, columns_to_write, require_all_input_fields)

        self.app_token = app_token
        self.table_id = table_id

        # 飞书客户端
        self.client = FeishuClient(
            app_id=app_id,
            app_secret=app_secret,
            max_retries=max_retries,
            qps_limit=qps_limit,
        )

        # 快照与映射
        self._snapshot: list[dict[str, Any]] = []
        self._id_map: dict[int, str] = {}  # task_id → record_id
        self._reverse_map: dict[str, int] = {}  # record_id → task_id
        self._snapshot_loaded = False

        # 写回列名
        self.write_aliases = list(self.columns_to_write.keys())
        self.write_colnames = list(self.columns_to_write.values())

        # 分片状态
        self.current_shard_id = -1
        self.current_min_id = 0
        self.current_max_id = 0

        self._snapshot_lock = threading.Lock()
        self._logger = logging.getLogger("feishu.bitable_pool")

    # ==================== 快照管理 ====================

    def _load_snapshot_sync(self) -> None:
        """同步方式加载快照（线程安全）"""
        if self._snapshot_loaded:
            return
        with self._snapshot_lock:
            if self._snapshot_loaded:
                return
            run_async(self._load_snapshot())

    async def _load_snapshot(self) -> None:
        """从飞书拉取全部记录到内存快照"""
        if self._snapshot_loaded:
            return

        self._logger.info(
            f"开始拉取多维表格快照: app_token={self.app_token}, table_id={self.table_id}"
        )

        records = await self.client.bitable_list_records(self.app_token, self.table_id)

        self._snapshot = records
        self._id_map = {}
        self._reverse_map = {}

        for idx, rec in enumerate(records):
            record_id = rec.get("record_id", "")
            self._id_map[idx] = record_id
            self._reverse_map[record_id] = idx

        self._snapshot_loaded = True
        self._logger.info(f"快照加载完成，共 {len(records)} 条记录")

    def _get_fields(self, record: dict[str, Any]) -> dict[str, Any]:
        """提取记录的 fields 字典"""
        return record.get("fields", {})

    def _is_unprocessed(self, fields: dict[str, Any]) -> bool:
        """判断记录是否未处理"""
        # 输入有效性检查
        input_valid = False
        if self.require_all_input_fields:
            input_valid = all(
                self._field_not_empty(fields.get(col))
                for col in self.columns_to_extract
            )
        else:
            input_valid = any(
                self._field_not_empty(fields.get(col))
                for col in self.columns_to_extract
            )

        if not input_valid:
            return False

        # 输出列任一为空 → 未处理
        return any(
            not self._field_not_empty(fields.get(col)) for col in self.write_colnames
        )

    def _is_processed(self, fields: dict[str, Any]) -> bool:
        """判断记录是否已处理（所有输出列非空）"""
        return all(
            self._field_not_empty(fields.get(col)) for col in self.write_colnames
        )

    @staticmethod
    def _field_not_empty(value: Any) -> bool:
        """判断字段值是否非空"""
        if value is None:
            return False
        if isinstance(value, str) and value.strip() == "":
            return False
        if isinstance(value, list) and len(value) == 0:
            return False
        return True

    # ==================== BaseTaskPool 抽象方法实现 ====================

    def get_total_task_count(self) -> int:
        """获取未处理任务总数"""
        self._load_snapshot_sync()
        count = sum(
            1 for rec in self._snapshot if self._is_unprocessed(self._get_fields(rec))
        )
        self._logger.info(f"多维表格未处理任务数: {count}")
        return count

    def get_processed_task_count(self) -> int:
        """获取已处理任务总数"""
        self._load_snapshot_sync()
        count = sum(
            1 for rec in self._snapshot if self._is_processed(self._get_fields(rec))
        )
        self._logger.info(f"多维表格已处理任务数: {count}")
        return count

    def get_id_boundaries(self) -> tuple[int, int]:
        """
        获取任务 ID 边界

        Bitable 使用连续整数 task_id 映射 record_id，
        边界为 (0, total_records - 1)。
        """
        self._load_snapshot_sync()
        if not self._snapshot:
            return (0, -1)
        return (0, len(self._snapshot) - 1)

    def initialize_shard(self, shard_id: int, min_id: int, max_id: int) -> int:
        """
        初始化分片，从快照中加载指定 task_id 范围内的未处理任务

        Args:
            shard_id: 分片编号
            min_id: task_id 范围起始（包含）
            max_id: task_id 范围结束（包含）

        Returns:
            加载的任务数量
        """
        self._load_snapshot_sync()

        shard_tasks: list[tuple[int, dict[str, Any]]] = []

        for task_id in range(min_id, min(max_id + 1, len(self._snapshot))):
            rec = self._snapshot[task_id]
            fields = self._get_fields(rec)
            if self._is_unprocessed(fields):
                # 提取输入列
                record_dict = {
                    col: self._convert_field_value(fields.get(col, ""))
                    for col in self.columns_to_extract
                }
                shard_tasks.append((task_id, record_dict))

        with self.lock:
            self.tasks = shard_tasks

        self.current_shard_id = shard_id
        self.current_min_id = min_id
        self.current_max_id = max_id

        self._logger.info(
            f"分片 {shard_id} (task_id: {min_id}-{max_id}) 加载完成，"
            f"任务数: {len(shard_tasks)}"
        )
        return len(shard_tasks)

    def get_task_batch(self, batch_size: int) -> list[tuple[int, dict[str, Any]]]:
        """从内存队列获取一批任务"""
        with self.lock:
            batch = self.tasks[:batch_size]
            self.tasks = self.tasks[batch_size:]
            return batch

    @property
    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            atomic_batch=False,
            idempotent_write=True,
            resumable=True,
            full_scan=True,
        )

    def update_task_results(
        self,
        batch_id: str,
        results: dict[int, dict[str, Any]],
    ) -> WritebackReceipt:
        """
        批量写回任务结果到飞书多维表格

        通过 Bitable batch_update API 批量更新。
        按 record_id 覆盖写入，天然幂等。

        Args:
            results: {task_id: {alias: value, ...}, ...}
        """
        if not results:
            return WritebackReceipt.committed(batch_id, (), atomic=False)

        # 构建待更新记录
        update_records: list[dict[str, Any]] = []
        task_ids_by_record: dict[str, int] = {}
        outcomes: dict[int, WritebackItem] = {}
        for task_id, row_result in results.items():
            if "_error" in row_result:
                outcomes[task_id] = WritebackItem(
                    task_id,
                    CommitDisposition.REJECTED,
                    "invalid_result",
                    "结果包含 _error",
                    False,
                )
                continue

            record_id = self._id_map.get(task_id)
            if not record_id:
                outcomes[task_id] = WritebackItem(
                    record_id=task_id,
                    disposition=CommitDisposition.REJECTED,
                    code="record_not_found",
                    message="task_id 无对应 record_id",
                    retryable=False,
                )
                continue

            fields: dict[str, Any] = {}
            for alias, col_name in self.columns_to_write.items():
                if alias in row_result:
                    fields[col_name] = row_result[alias]

            if fields:
                update_records.append(
                    {
                        "record_id": record_id,
                        "fields": fields,
                    }
                )
                task_ids_by_record[record_id] = task_id
            else:
                outcomes[task_id] = WritebackItem(
                    task_id,
                    CommitDisposition.REJECTED,
                    "no_writable_fields",
                    "结果不包含可写字段",
                    False,
                )

        if not update_records:
            return WritebackReceipt(
                batch_id=batch_id,
                submitted_ids=tuple(results),
                items=tuple(outcomes[task_id] for task_id in results),
                atomic=False,
            )

        record_outcomes = run_async(self._batch_update(update_records))
        for record_id, record_outcome in record_outcomes.items():
            task_id = task_ids_by_record.get(record_id)
            if task_id is None:
                continue
            outcomes[task_id] = WritebackItem(
                record_id=task_id,
                disposition=record_outcome.disposition,
                code=record_outcome.code,
                message=record_outcome.message,
                retryable=record_outcome.retryable,
            )
        for task_id in results:
            outcomes.setdefault(
                task_id,
                WritebackItem(
                    task_id,
                    CommitDisposition.INDETERMINATE,
                    "bitable_result_missing",
                    "Bitable 响应未覆盖该记录",
                    False,
                ),
            )
        return WritebackReceipt(
            batch_id=batch_id,
            submitted_ids=tuple(results),
            items=tuple(outcomes[task_id] for task_id in results),
            atomic=False,
        )

    async def _batch_update(
        self, records: list[dict[str, Any]]
    ) -> dict[str, WritebackItem]:
        """异步执行独立 chunk，并为每个 record 返回精确结果。"""

        chunks = [
            records[index : index + BITABLE_BATCH_UPDATE_LIMIT]
            for index in range(0, len(records), BITABLE_BATCH_UPDATE_LIMIT)
        ]
        outcomes = await asyncio.gather(
            *(
                self.client.bitable_batch_update(self.app_token, self.table_id, chunk)
                for chunk in chunks
            ),
            return_exceptions=True,
        )
        record_outcomes: dict[str, WritebackItem] = {}
        for chunk, outcome in zip(chunks, outcomes):
            if isinstance(outcome, BaseException):
                for record in chunk:
                    record_id = str(record["record_id"])
                    record_outcomes[record_id] = _bitable_error_item(
                        record_id,
                        outcome,
                    )
                continue
            if not isinstance(outcome, list):
                for record in chunk:
                    record_id = str(record["record_id"])
                    record_outcomes[record_id] = WritebackItem(
                        record_id,
                        CommitDisposition.INDETERMINATE,
                        "bitable_response_malformed",
                        "Feishu 批量更新响应不是记录列表",
                        False,
                    )
                continue
            acknowledged = {
                str(item.get("record_id"))
                for item in outcome
                if isinstance(item, dict) and item.get("record_id")
            }
            for record in chunk:
                record_id = str(record["record_id"])
                if record_id in acknowledged:
                    record_outcomes[record_id] = WritebackItem(
                        record_id,
                        CommitDisposition.COMMITTED,
                    )
                else:
                    record_outcomes[record_id] = WritebackItem(
                        record_id,
                        CommitDisposition.INDETERMINATE,
                        "bitable_ack_missing",
                        "Feishu 响应未确认该 record_id",
                        False,
                    )

        self._logger.info(
            "Bitable 批量更新完成，确认提交 %s 条，其他 %s 条",
            sum(
                item.disposition == CommitDisposition.COMMITTED
                for item in record_outcomes.values()
            ),
            sum(
                item.disposition != CommitDisposition.COMMITTED
                for item in record_outcomes.values()
            ),
        )

        # 同步更新内存快照，防止多 shard 重复处理（O(1) 映射查找）
        for rec in records:
            rec_id = str(rec["record_id"])
            outcome = record_outcomes.get(rec_id)
            if outcome is None or outcome.disposition != CommitDisposition.COMMITTED:
                continue
            fields = rec["fields"]
            task_id = self._reverse_map.get(rec_id)
            if task_id is None:
                self._logger.warning(f"record_id={rec_id} 无对应 task_id，跳过快照同步")
                continue
            if task_id < 0 or task_id >= len(self._snapshot):
                self._logger.warning(
                    f"record_id={rec_id} 映射 task_id={task_id} 越界，跳过快照同步"
                )
                continue
            snapshot_rec = self._snapshot[task_id]
            snapshot_fields = snapshot_rec.get("fields")
            if not isinstance(snapshot_fields, dict):
                snapshot_fields = {}
                snapshot_rec["fields"] = snapshot_fields
            snapshot_fields.update(fields)
        return record_outcomes

    def reconcile_task_results(
        self,
        batch_id: str,
        results: dict[int, dict[str, Any]],
    ) -> WritebackReceipt:
        """从 Bitable 重新读取远端字段并逐记录核对。"""

        outcomes: dict[int, WritebackItem] = {}
        expected: dict[int, tuple[str, dict[str, Any]]] = {}
        for task_id, row_result in results.items():
            if "_error" in row_result:
                outcomes[task_id] = WritebackItem(
                    task_id,
                    CommitDisposition.REJECTED,
                    "invalid_result",
                    "结果包含 _error",
                    False,
                )
                continue
            record_id = self._id_map.get(task_id)
            if not record_id:
                outcomes[task_id] = WritebackItem(
                    task_id,
                    CommitDisposition.REJECTED,
                    "record_not_found",
                    "task_id 无对应 record_id",
                    False,
                )
                continue
            fields = {
                col_name: row_result[alias]
                for alias, col_name in self.columns_to_write.items()
                if alias in row_result
            }
            if not fields:
                outcomes[task_id] = WritebackItem(
                    task_id,
                    CommitDisposition.REJECTED,
                    "no_writable_fields",
                    "结果不包含可写字段",
                    False,
                )
                continue
            expected[task_id] = (record_id, fields)

        try:
            remote_records = run_async(
                self.client.bitable_list_records(
                    self.app_token,
                    self.table_id,
                    field_names=sorted(set(self.columns_to_write.values())),
                )
            )
        except Exception as exc:
            for task_id in expected:
                outcomes[task_id] = WritebackItem(
                    task_id,
                    CommitDisposition.INDETERMINATE,
                    "bitable_reconciliation_failed",
                    str(exc),
                    False,
                )
            return self._receipt(batch_id, results, outcomes)

        by_record_id = {
            str(record.get("record_id")): record
            for record in remote_records
            if record.get("record_id")
        }
        for task_id, (record_id, expected_fields) in expected.items():
            remote = by_record_id.get(record_id)
            if remote is None:
                outcomes[task_id] = WritebackItem(
                    task_id,
                    CommitDisposition.REJECTED,
                    "record_not_found_on_reconciliation",
                    f"远端不存在 record_id={record_id}",
                    False,
                )
                continue
            actual_fields = self._get_fields(remote)
            mismatched = [
                field_name
                for field_name, value in expected_fields.items()
                if field_name not in actual_fields
                or not self._field_values_equal(value, actual_fields[field_name])
            ]
            if mismatched:
                outcomes[task_id] = WritebackItem(
                    task_id,
                    CommitDisposition.REJECTED,
                    "bitable_reconciliation_mismatch",
                    f"远端字段不匹配: {mismatched}",
                    True,
                )
                continue
            outcomes[task_id] = WritebackItem(
                task_id,
                CommitDisposition.COMMITTED,
            )
            if 0 <= task_id < len(self._snapshot):
                snapshot_fields = self._snapshot[task_id].setdefault("fields", {})
                if isinstance(snapshot_fields, dict):
                    snapshot_fields.update(actual_fields)

        return self._receipt(batch_id, results, outcomes)

    @staticmethod
    def _receipt(
        batch_id: str,
        results: dict[int, dict[str, Any]],
        outcomes: dict[int, WritebackItem],
    ) -> WritebackReceipt:
        return WritebackReceipt(
            batch_id=batch_id,
            submitted_ids=tuple(results),
            items=tuple(outcomes[task_id] for task_id in results),
            atomic=False,
        )

    @classmethod
    def _field_values_equal(cls, expected: Any, actual: Any) -> bool:
        try:
            if expected == actual:
                return True
        except (TypeError, ValueError):
            pass
        return cls._convert_field_value(expected) == cls._convert_field_value(actual)

    def reload_task_data(self, task_id: int) -> dict[str, Any] | None:
        """重新从快照加载任务数据"""
        if 0 <= task_id < len(self._snapshot):
            fields = self._get_fields(self._snapshot[task_id])
            return {
                col: self._convert_field_value(fields.get(col, ""))
                for col in self.columns_to_extract
            }
        self._logger.warning(f"task_id={task_id} 超出快照范围")
        return None

    def close(self) -> None:
        """关闭飞书客户端"""
        self._logger.info("关闭飞书多维表格任务池 ...")
        run_async(self.client.close())

    # ==================== 工具方法 ====================

    @staticmethod
    def _convert_field_value(value: Any) -> str:
        """将飞书字段值转换为字符串"""
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, list):
            # 多维表格的多选、人员等字段是列表
            parts: list[str] = []
            for item in value:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", item.get("name", str(item)))))
                else:
                    parts.append(str(item))
            return ", ".join(parts)
        if isinstance(value, dict):
            return str(value.get("text", value.get("link", str(value))))
        return str(value)

    # ==================== Token 估算采样 ====================

    def sample_unprocessed_rows(self, sample_size: int) -> list[dict[str, Any]]:
        """采样未处理行"""
        self._load_snapshot_sync()
        samples: list[dict[str, Any]] = []
        for rec in self._snapshot:
            if len(samples) >= sample_size:
                break
            fields = self._get_fields(rec)
            if self._is_unprocessed(fields):
                samples.append(
                    {
                        col: self._convert_field_value(fields.get(col, ""))
                        for col in self.columns_to_extract
                    }
                )
        return samples

    def sample_processed_rows(self, sample_size: int) -> list[dict[str, Any]]:
        """采样已处理行"""
        self._load_snapshot_sync()
        samples: list[dict[str, Any]] = []
        for rec in self._snapshot:
            if len(samples) >= sample_size:
                break
            fields = self._get_fields(rec)
            if self._is_processed(fields):
                samples.append(
                    {
                        col: self._convert_field_value(fields.get(col, ""))
                        for col in self.write_colnames
                    }
                )
        return samples

    def fetch_all_rows(self, columns: list[str]) -> list[dict[str, Any]]:
        self._load_snapshot_sync()
        return [
            {
                col: self._convert_field_value(self._get_fields(record).get(col, ""))
                for col in columns
            }
            for record in self._snapshot
        ]

    def fetch_all_processed_rows(self, columns: list[str]) -> list[dict[str, Any]]:
        self._load_snapshot_sync()
        rows: list[dict[str, Any]] = []
        for record in self._snapshot:
            fields = self._get_fields(record)
            if self._is_processed(fields):
                rows.append(
                    {
                        col: self._convert_field_value(fields.get(col, ""))
                        for col in columns
                    }
                )
        return rows
