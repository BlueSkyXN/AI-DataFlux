"""
飞书多维表格（Bitable）数据源任务池

本模块实现基于飞书多维表格的 TaskPool，适用于将多维表格作为
AI-DataFlux 的数据源进行批量 AI 处理。

核心设计（对应云端表格与本地文件的差异）:
    1. 快照读取 —— 初始化时一次性拉取全部记录到内存，后续操作基于快照
    2. ID 映射表 —— 连续整数 task_id ↔ 字符串 record_id 的稳定映射
    3. 写入控制 —— 批量更新上限 500 条/次，自动分块
    4. 部分失败追溯 —— 每条记录的写入状态独立跟踪
    5. Token 自动刷新 —— 由 FeishuClient 透明处理

分片策略:
    与数据库数据源不同，Bitable 的 record_id 是字符串（如 recXXXXXX），
    不是连续数字。因此：
    - get_id_boundaries() 返回 (0, total_records - 1)
    - 使用连续整数 task_id 做分片，通过映射表查找 record_id
    - initialize_shard() 按 task_id 范围从快照中加载

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

import logging
import threading
from typing import Any

from ..base import BaseTaskPool
from . import run_async
from .client import FeishuClient


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
        self._id_map: dict[int, str] = {}       # task_id → record_id
        self._reverse_map: dict[str, int] = {}   # record_id → task_id
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

        records = await self.client.bitable_list_records(
            self.app_token, self.table_id
        )

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
            not self._field_not_empty(fields.get(col))
            for col in self.write_colnames
        )

    def _is_processed(self, fields: dict[str, Any]) -> bool:
        """判断记录是否已处理（所有输出列非空）"""
        return all(
            self._field_not_empty(fields.get(col))
            for col in self.write_colnames
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
            1 for rec in self._snapshot
            if self._is_unprocessed(self._get_fields(rec))
        )
        self._logger.info(f"多维表格未处理任务数: {count}")
        return count

    def get_processed_task_count(self) -> int:
        """获取已处理任务总数"""
        self._load_snapshot_sync()
        count = sum(
            1 for rec in self._snapshot
            if self._is_processed(self._get_fields(rec))
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

    def update_task_results(self, results: dict[int, dict[str, Any]]) -> None:
        """
        批量写回任务结果到飞书多维表格

        通过 Bitable batch_update API 批量更新。
        按 record_id 覆盖写入，天然幂等。

        Args:
            results: {task_id: {alias: value, ...}, ...}
        """
        if not results:
            return

        # 构建待更新记录
        update_records: list[dict[str, Any]] = []
        for task_id, row_result in results.items():
            if "_error" in row_result:
                continue

            record_id = self._id_map.get(task_id)
            if not record_id:
                self._logger.warning(f"task_id={task_id} 无对应 record_id，跳过")
                continue

            fields: dict[str, Any] = {}
            for alias, col_name in self.columns_to_write.items():
                if alias in row_result:
                    fields[col_name] = row_result[alias]

            if fields:
                update_records.append({
                    "record_id": record_id,
                    "fields": fields,
                })

        if not update_records:
            return

        # 同步调用异步写入
        run_async(self._batch_update(update_records))

    async def _batch_update(self, records: list[dict[str, Any]]) -> None:
        """异步执行批量更新"""
        try:
            result = await self.client.bitable_batch_update(
                self.app_token, self.table_id, records
            )
            self._logger.info(
                f"Bitable 批量更新完成，成功 {len(result)} 条"
            )
        except Exception as e:
            self._logger.error(f"Bitable 批量更新失败: {e}")
            raise

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
            parts = []
            for item in value:
                if isinstance(item, dict):
                    parts.append(item.get("text", item.get("name", str(item))))
                else:
                    parts.append(str(item))
            return ", ".join(parts)
        if isinstance(value, dict):
            return value.get("text", value.get("link", str(value)))
        return str(value)

    # ==================== Token 估算采样 ====================

    def sample_unprocessed_rows(self, sample_size: int) -> list[dict[str, Any]]:
        """采样未处理行"""
        self._load_snapshot_sync()
        samples = []
        for rec in self._snapshot:
            if len(samples) >= sample_size:
                break
            fields = self._get_fields(rec)
            if self._is_unprocessed(fields):
                samples.append({
                    col: self._convert_field_value(fields.get(col, ""))
                    for col in self.columns_to_extract
                })
        return samples

    def sample_processed_rows(self, sample_size: int) -> list[dict[str, Any]]:
        """采样已处理行"""
        self._load_snapshot_sync()
        samples = []
        for rec in self._snapshot:
            if len(samples) >= sample_size:
                break
            fields = self._get_fields(rec)
            if self._is_processed(fields):
                samples.append({
                    col: self._convert_field_value(fields.get(col, ""))
                    for col in self.write_colnames
                })
        return samples
