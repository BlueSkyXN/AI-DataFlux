"""
分片任务调度管理器

本模块实现大规模数据处理的分片调度机制，负责将海量数据切分为可管理的
分片单元，并跟踪处理进度、监控内存使用。

设计目标:
    - 避免一次性加载全部数据导致内存溢出
    - 动态调整分片大小以适应系统资源
    - 实时监控处理进度和内存使用
    - 提供详细的处理统计信息

分片策略:
    ┌──────────────────────────────────────────────────────────────┐
    │                      数据源 (N 条记录)                        │
    └──────────────────────────────────────────────────────────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
              ┌──────────┐ ┌──────────┐ ┌──────────┐
              │ 分片 1   │ │ 分片 2   │ │ 分片 N   │
              │ ID: 0-999│ │ID:1000-  │ │ ...      │
              └──────────┘ └──────────┘ └──────────┘
                    │
                    ▼
              ┌──────────────────────────────────────────┐
              │          逐个分片加载处理                  │
              │  - 基于 ID 范围过滤                       │
              │  - 处理完毕后加载下一分片                  │
              │  - 内存受控，不超出限制                    │
              └──────────────────────────────────────────┘

动态分片大小计算:
    分片大小 = min(内存限制, 时间限制, 配置值)

    - 内存限制: 可用内存 × 30% / 估算记录大小
    - 时间限制: 处理速率 × 目标时长 (15分钟)
    - 配置值: optimal_shard_size (默认 10000)

内存监控阈值:
    - 系统内存 > 85%: 触发 GC
    - 进程内存 > 40GB: 触发 GC

使用示例:
    from src.core.scheduler import ShardedTaskManager

    manager = ShardedTaskManager(task_pool, shard_size=10000)
    if manager.initialize():
        while manager.load_next_shard():
            # 处理当前分片
            pass
        manager.finalize()
"""

import gc
import logging
import time
from typing import Any

try:
    import psutil
except ModuleNotFoundError:  # pragma: no cover - 环境缺少依赖时的兜底
    psutil = None

from ..models.errors import ErrorType
from ..data.base import BaseTaskPool


class ShardedTaskManager:
    """
    分片任务管理器

    负责将大量数据分片加载、跟踪处理进度、监控内存使用。

    核心功能:
        1. 分片计算: 根据 ID 范围和配置计算分片边界
        2. 分片加载: 逐个加载分片到内存
        3. 进度跟踪: 记录已处理数量和成功率
        4. 内存监控: 定期检查内存使用，必要时触发 GC
        5. 统计输出: 完成后输出详细统计信息

    生命周期:
        ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
        │ __init__ │ → │initialize│ → │load_next │ → │ finalize │
        │ 配置初始化│    │ 计算分片 │    │ _shard   │    │ 统计输出 │
        └──────────┘    └──────────┘    └────┬─────┘    └──────────┘
                                            │ ↑
                                            └─┘ (循环直到无更多分片)

    Attributes:
        task_pool: 数据源任务池
        optimal_shard_size: 理想分片大小
        min_shard_size: 最小分片大小
        max_shard_size: 最大分片大小
        max_retry_counts: 各错误类型的最大重试次数
        current_shard_index: 当前分片索引
        total_shards: 总分片数
        shard_boundaries: 分片边界列表 [(min_id, max_id), ...]
        total_processed_successfully: 成功处理的记录数
    """

    def __init__(
        self,
        task_pool: BaseTaskPool,
        optimal_shard_size: int = 10000,
        min_shard_size: int = 1000,
        max_shard_size: int = 50000,
        max_retry_counts: dict[str, int] | None = None,
    ):
        """
        初始化分片任务管理器

        Args:
            task_pool: 数据源任务池实例
            optimal_shard_size: 理想分片大小
            min_shard_size: 最小分片大小
            max_shard_size: 最大分片大小
            max_retry_counts: 各错误类型的最大重试次数
        """
        if not isinstance(task_pool, BaseTaskPool):
            raise TypeError("task_pool 必须是 BaseTaskPool 的实例")

        self.task_pool = task_pool
        self.optimal_shard_size = optimal_shard_size
        self.min_shard_size = min_shard_size
        self.max_shard_size = max_shard_size

        # 重试限制
        self.max_retry_counts = max_retry_counts or {
            ErrorType.API: 3,
            ErrorType.CONTENT: 1,
            ErrorType.SYSTEM: 2,
        }

        # 分片状态
        self.current_shard_index = 0
        self.total_shards = 0
        self.shard_boundaries: list[tuple[Any, Any]] = []

        # 统计信息
        self.total_estimated = 0
        self.total_processed_successfully = 0
        self.start_time = time.time()

        # 处理指标
        self.processing_metrics = {
            "avg_time_per_record": 0.0,
            "records_per_second": 0.0,
        }

        # 重试统计
        self.retried_tasks_count = {
            ErrorType.API: 0,
            ErrorType.CONTENT: 0,
            ErrorType.SYSTEM: 0,
        }
        self.max_retries_exceeded_count = 0

        # 内存监控
        self.memory_tracker = {
            "last_check_time": time.time(),
            "check_interval": 60,
            "peak_memory_usage": 0.0,
            "current_memory_usage": 0.0,
        }

        # 获取进程信息
        self._process_info = None
        if psutil is None:
            logging.warning("psutil 未安装，内存监控将不可用")
        else:
            try:
                self._process_info = psutil.Process()
            except psutil.Error as e:
                logging.warning(f"无法获取当前进程信息 (psutil): {e}，内存监控将不可用")

        logging.info("ShardedTaskManager 初始化完成")

    def calculate_optimal_shard_size(self, total_range: int) -> int:
        """
        动态计算最优分片大小

        基于可用内存和处理速度进行启发式计算，自动适应系统资源。

        计算逻辑:
            1. 内存限制 = 可用内存 × 30% / 记录大小估算 (5KB/条)
            2. 时间限制 = 处理速率 × 目标时长 (15分钟)
            3. 最终大小 = min(内存限制, 时间限制, optimal_shard_size)
            4. 限制在 [min_shard_size, max_shard_size] 范围内

        Args:
            total_range: 总数据范围 (max_id - min_id + 1)

        Returns:
            计算出的分片大小

        Note:
            首次运行时 records_per_second 为 0，时间限制不生效。
            随着处理进行，指标会逐渐准确。
        """
        # 基于内存的限制
        memory_based_limit = self.max_shard_size

        if self._process_info and psutil is not None:
            try:
                mem = psutil.virtual_memory()
                available_mb = mem.available / (1024 * 1024)
                record_size_mb_estimate = 5 / 1024  # 假设每条记录约 5KB
                memory_based_limit = (
                    int((available_mb * 0.3) / record_size_mb_estimate)
                    if record_size_mb_estimate > 0
                    else self.max_shard_size
                )
            except Exception as e:
                logging.warning(f"计算内存限制时出错: {e}")

        # 基于处理速度的限制
        time_based_limit = self.max_shard_size

        if self.processing_metrics["records_per_second"] > 0:
            target_duration_seconds = 15 * 60  # 目标处理时间 15 分钟
            time_based_limit = int(
                self.processing_metrics["records_per_second"] * target_duration_seconds
            )

        # 取各限制中的最小值
        calculated_size = min(
            memory_based_limit, time_based_limit, self.optimal_shard_size
        )

        # 限制在范围内
        shard_size = max(self.min_shard_size, min(calculated_size, self.max_shard_size))

        logging.info(
            f"动态分片大小计算: 内存限制={memory_based_limit}, "
            f"时间限制={time_based_limit}, 最终选择={shard_size}"
        )

        return shard_size

    def initialize(self) -> bool:
        """
        初始化分片

        获取数据边界，计算分片大小，创建分片边界列表。

        初始化流程:
            1. 获取待处理任务总数
            2. 获取 ID/索引边界 (min_id, max_id)
            3. 计算最优分片大小
            4. 创建分片边界列表

        Returns:
            是否成功初始化 (False 表示无任务或初始化失败)

        Raises:
            不抛出异常，错误会被记录并返回 False
        """
        logging.info("正在初始化分片任务管理器...")
        self.start_time = time.time()

        try:
            # 获取未处理任务总数
            self.total_estimated = self.task_pool.get_total_task_count()

            if self.total_estimated <= 0:
                logging.info("数据源中没有需要处理的任务")
                return False

            # 获取 ID 边界
            min_id, max_id = self.task_pool.get_id_boundaries()
            logging.info(
                f"获取到数据源 ID/索引范围: {min_id} - {max_id}，"
                f"预估未处理任务数: {self.total_estimated}"
            )

            # 转换为数值
            try:
                numeric_min = int(min_id)
                numeric_max = int(max_id)
                total_range = numeric_max - numeric_min + 1
            except (ValueError, TypeError):
                logging.error("ID 边界无法转换为整数，无法分片")
                return False

            # 处理边界情况
            if total_range <= 0 and self.total_estimated > 0:
                logging.warning("ID 范围无效但仍有任务，将作为一个分片处理")
                self.total_shards = 1
                self.shard_boundaries = [(min_id, max_id)]
            elif total_range <= 0:
                logging.info("ID 范围无效且无任务")
                return False
            else:
                # 计算分片
                shard_size = self.calculate_optimal_shard_size(total_range)
                self.total_shards = max(1, (total_range + shard_size - 1) // shard_size)

                logging.info(
                    f"数据分为 {self.total_shards} 个分片 (大小约 {shard_size})"
                )

                # 创建分片边界
                self.shard_boundaries = []
                current_start = numeric_min

                for i in range(self.total_shards):
                    current_end = min(current_start + shard_size - 1, numeric_max)
                    self.shard_boundaries.append((current_start, current_end))
                    current_start = current_end + 1

                logging.debug(f"分片边界: {self.shard_boundaries}")

            self.current_shard_index = 0
            logging.info(f"分片任务管理器初始化成功，共 {self.total_shards} 个分片")
            return True

        except Exception as e:
            logging.error(f"初始化分片任务管理器失败: {e}", exc_info=True)
            return False

    def load_next_shard(self) -> bool:
        """
        加载下一个分片

        从数据源加载下一个分片的数据到任务池。
        如果当前分片无任务，会自动递归加载下一个分片。

        Returns:
            是否成功加载 (False 表示没有更多分片)

        Note:
            如果分片加载失败，会自动跳过并尝试下一个分片，
            确保单个分片的问题不会阻塞整个处理流程。
        """
        if self.current_shard_index >= self.total_shards:
            logging.info("所有分片已处理完毕，没有更多分片可加载")
            return False

        min_id, max_id = self.shard_boundaries[self.current_shard_index]
        shard_num = self.current_shard_index + 1

        logging.info(
            f"--- 加载分片 {shard_num}/{self.total_shards} (范围: {min_id}-{max_id}) ---"
        )

        try:
            loaded_count = self.task_pool.initialize_shard(shard_num, min_id, max_id)
            self.current_shard_index += 1

            if loaded_count == 0:
                logging.info(f"分片 {shard_num} 无任务，尝试加载下一个...")
                return self.load_next_shard()
            else:
                logging.info(f"分片 {shard_num} 加载成功 ({loaded_count} 个任务)")
                return True

        except Exception as e:
            logging.error(f"加载分片 {shard_num} 时发生错误: {e}", exc_info=True)
            self.current_shard_index += 1
            logging.warning(f"跳过加载失败的分片 {shard_num}，尝试加载下一个")
            return self.load_next_shard()

    def update_processing_metrics(
        self, batch_success_count: int, batch_processing_time: float
    ) -> None:
        """
        更新处理指标

        使用指数移动平均 (EMA) 更新处理速度指标，平滑因子 α=0.1。
        EMA 公式: new_value = α × current + (1 - α) × old_value

        该算法使得:
            - 新数据占 10% 权重
            - 历史数据占 90% 权重
            - 平滑抖动，避免偶发波动影响分片大小计算

        Args:
            batch_success_count: 本批次成功处理数量
            batch_processing_time: 本批次处理时间 (秒)

        Note:
            batch_success_count <= 0 或 batch_processing_time <= 0 时不更新。
        """
        if batch_processing_time <= 0 or batch_success_count <= 0:
            return

        current_time_per_record = batch_processing_time / batch_success_count
        current_records_per_second = 1.0 / current_time_per_record

        # 平滑因子
        alpha = 0.1

        if self.processing_metrics["avg_time_per_record"] == 0.0:
            self.processing_metrics["avg_time_per_record"] = current_time_per_record
            self.processing_metrics["records_per_second"] = current_records_per_second
        else:
            # 指数移动平均
            self.processing_metrics["avg_time_per_record"] = (
                alpha * current_time_per_record
                + (1 - alpha) * self.processing_metrics["avg_time_per_record"]
            )
            self.processing_metrics["records_per_second"] = 1.0 / max(
                1e-9, self.processing_metrics["avg_time_per_record"]
            )

    def monitor_memory_usage(self) -> None:
        """
        监控内存使用

        定期检查内存使用情况，高内存使用时触发垃圾回收 (GC)。

        检查间隔: 60 秒 (memory_tracker["check_interval"])

        GC 触发条件 (满足任一即触发):
            - 系统内存使用率 > 85%
            - 进程内存使用量 > 40GB

        监控指标:
            - current_memory_usage: 当前进程内存 (MB)
            - peak_memory_usage: 峰值内存使用 (MB)
        """
        if not self._process_info or psutil is None:
            return

        current_time = time.time()
        if (
            current_time - self.memory_tracker["last_check_time"]
            < self.memory_tracker["check_interval"]
        ):
            return

        try:
            # 获取当前内存使用
            current_mem_mb = self._process_info.memory_info().rss / (1024 * 1024)
            self.memory_tracker["current_memory_usage"] = current_mem_mb
            self.memory_tracker["peak_memory_usage"] = max(
                self.memory_tracker["peak_memory_usage"], current_mem_mb
            )
            self.memory_tracker["last_check_time"] = current_time

            logging.debug(
                f"内存监控: 当前={current_mem_mb:.1f}MB, "
                f"峰值={self.memory_tracker['peak_memory_usage']:.1f}MB"
            )

            # 检查是否需要 GC
            system_mem = psutil.virtual_memory()
            if system_mem.percent > 85.0 or current_mem_mb > 40000.0:
                logging.warning(
                    f"高内存使用: 进程={current_mem_mb:.1f}MB, "
                    f"系统={system_mem.percent}%，触发 GC..."
                )
                gc.collect()

                current_mem_after_gc = self._process_info.memory_info().rss / (
                    1024 * 1024
                )
                logging.info(f"GC 后内存: {current_mem_after_gc:.1f}MB")
                self.memory_tracker["current_memory_usage"] = current_mem_after_gc

        except Exception as e:
            logging.warning(f"内存监控检查失败: {e}")

    def finalize(self) -> None:
        """
        完成处理，输出统计信息并关闭资源

        输出内容:
            - 总耗时
            - 预估任务数 vs 成功处理数
            - 各错误类型的重试次数
            - 重试超限任务数
            - 平均处理速率 (条/秒)
            - 峰值内存使用

        资源清理:
            - 关闭任务池 (task_pool.close())
        """
        end_time = time.time()
        total_duration = end_time - self.start_time

        # 输出统计信息
        logging.info("=" * 50)
        logging.info(" 任务处理结束 ".center(50, "="))
        logging.info("=" * 50)
        logging.info(f"总耗时: {total_duration:.2f} 秒")
        logging.info(f"预估总任务数: {self.total_estimated}")
        logging.info(f"成功处理并更新的任务数: {self.total_processed_successfully}")

        # 重试统计
        logging.info("重试统计信息:")
        for error_type, count in self.retried_tasks_count.items():
            max_retries = self.max_retry_counts.get(error_type, 0)
            logging.info(
                f"  - {error_type.value}: {count} 次重试 (最大重试次数: {max_retries})"
            )
        logging.info(f"  - 重试次数超限任务数: {self.max_retries_exceeded_count}")

        # 处理速率
        if self.total_processed_successfully > 0 and total_duration > 0:
            rate = self.total_processed_successfully / total_duration
            logging.info(f"平均处理速率: {rate:.2f} 条记录/秒")

        # 内存使用
        if self.memory_tracker["peak_memory_usage"] > 0:
            logging.info(
                f"峰值内存使用量: {self.memory_tracker['peak_memory_usage']:.1f} MB"
            )

        logging.info("=" * 50)

        # 关闭任务池
        try:
            logging.info("正在关闭任务池资源...")
            self.task_pool.close()
            logging.info("任务池资源已关闭")
        except Exception as e:
            logging.error(f"关闭任务池资源时发生错误: {e}", exc_info=True)

    @property
    def has_more_shards(self) -> bool:
        """是否还有更多分片"""
        return self.current_shard_index < self.total_shards

    @property
    def progress_percent(self) -> float:
        """当前进度百分比"""
        if self.total_estimated <= 0:
            return 100.0
        return (self.total_processed_successfully / self.total_estimated) * 100
