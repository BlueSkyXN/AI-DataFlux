"""
分片调度器测试

测试 src/core/scheduler.py 的分片任务管理功能
"""

import time
import pytest
from unittest.mock import MagicMock


class TestShardedTaskManagerInit:
    """分片任务管理器初始化测试"""

    def test_init_with_valid_task_pool(self):
        """测试使用有效任务池初始化"""
        from src.core.scheduler import ShardedTaskManager
        from src.data.base import BaseTaskPool

        mock_pool = MagicMock(spec=BaseTaskPool)

        manager = ShardedTaskManager(mock_pool)

        assert manager.task_pool is mock_pool
        assert manager.optimal_shard_size == 10000
        assert manager.min_shard_size == 1000
        assert manager.max_shard_size == 50000

    def test_init_with_custom_shard_sizes(self):
        """测试自定义分片大小"""
        from src.core.scheduler import ShardedTaskManager
        from src.data.base import BaseTaskPool

        mock_pool = MagicMock(spec=BaseTaskPool)

        manager = ShardedTaskManager(
            mock_pool,
            optimal_shard_size=5000,
            min_shard_size=500,
            max_shard_size=20000
        )

        assert manager.optimal_shard_size == 5000
        assert manager.min_shard_size == 500
        assert manager.max_shard_size == 20000

    def test_init_with_invalid_task_pool(self):
        """测试无效任务池抛出异常"""
        from src.core.scheduler import ShardedTaskManager

        with pytest.raises(TypeError, match="task_pool 必须是 BaseTaskPool"):
            ShardedTaskManager("not a task pool")

    def test_init_retry_counts(self):
        """测试默认重试次数配置"""
        from src.core.scheduler import ShardedTaskManager
        from src.data.base import BaseTaskPool
        from src.models.errors import ErrorType

        mock_pool = MagicMock(spec=BaseTaskPool)

        manager = ShardedTaskManager(mock_pool)

        assert manager.max_retry_counts[ErrorType.API] == 3
        assert manager.max_retry_counts[ErrorType.CONTENT] == 1
        assert manager.max_retry_counts[ErrorType.SYSTEM] == 2


class TestShardCalculation:
    """分片计算测试"""

    @pytest.fixture
    def manager(self):
        """创建测试用的管理器"""
        from src.core.scheduler import ShardedTaskManager
        from src.data.base import BaseTaskPool

        mock_pool = MagicMock(spec=BaseTaskPool)
        return ShardedTaskManager(mock_pool)

    def test_calculate_optimal_shard_size_basic(self, manager):
        """测试基本分片大小计算"""
        result = manager.calculate_optimal_shard_size(100000)

        assert manager.min_shard_size <= result <= manager.max_shard_size

    def test_calculate_optimal_shard_size_small_range(self, manager):
        """测试小范围数据"""
        manager.min_shard_size = 100
        result = manager.calculate_optimal_shard_size(50)

        # 应该返回最小分片大小
        assert result >= manager.min_shard_size

    def test_calculate_optimal_shard_size_with_processing_metrics(self, manager):
        """测试带处理速度指标的计算"""
        # 模拟高处理速度
        manager.processing_metrics["records_per_second"] = 100.0

        result = manager.calculate_optimal_shard_size(1000000)

        assert result >= manager.min_shard_size


class TestShardInitialization:
    """分片初始化测试"""

    @pytest.fixture
    def manager(self):
        """创建测试用的管理器"""
        from src.core.scheduler import ShardedTaskManager
        from src.data.base import BaseTaskPool

        mock_pool = MagicMock(spec=BaseTaskPool)
        mock_pool.get_total_task_count.return_value = 100
        mock_pool.get_id_boundaries.return_value = (0, 99)

        return ShardedTaskManager(mock_pool, optimal_shard_size=50)

    def test_initialize_success(self, manager):
        """测试成功初始化"""
        result = manager.initialize()

        assert result is True
        assert manager.total_estimated == 100
        assert manager.total_shards >= 1
        assert len(manager.shard_boundaries) == manager.total_shards

    def test_initialize_no_tasks(self, manager):
        """测试无任务时初始化"""
        manager.task_pool.get_total_task_count.return_value = 0

        result = manager.initialize()

        assert result is False

    def test_initialize_shard_boundaries(self, manager):
        """测试分片边界正确性"""
        manager.initialize()

        # 验证边界覆盖完整范围
        if manager.shard_boundaries:
            first_start = manager.shard_boundaries[0][0]
            last_end = manager.shard_boundaries[-1][1]

            assert first_start == 0
            assert last_end == 99


class TestShardLoading:
    """分片加载测试"""

    @pytest.fixture
    def manager(self):
        """创建已初始化的管理器"""
        from src.core.scheduler import ShardedTaskManager
        from src.data.base import BaseTaskPool

        mock_pool = MagicMock(spec=BaseTaskPool)
        mock_pool.get_total_task_count.return_value = 100
        mock_pool.get_id_boundaries.return_value = (0, 99)
        mock_pool.initialize_shard.return_value = 50

        mgr = ShardedTaskManager(mock_pool, optimal_shard_size=50)
        mgr.initialize()
        return mgr

    def test_load_next_shard_success(self, manager):
        """测试成功加载下一个分片"""
        result = manager.load_next_shard()

        assert result is True
        assert manager.current_shard_index == 1
        manager.task_pool.initialize_shard.assert_called()

    def test_load_next_shard_no_more(self, manager):
        """测试没有更多分片"""
        # 加载所有分片
        while manager.load_next_shard():
            pass

        result = manager.load_next_shard()
        assert result is False

    def test_load_next_shard_empty_shard(self, manager):
        """测试空分片时跳过"""
        # 重新初始化 manager 确保有足够的分片
        from src.core.scheduler import ShardedTaskManager
        from src.data.base import BaseTaskPool
        from unittest.mock import MagicMock

        mock_pool = MagicMock(spec=BaseTaskPool)
        mock_pool.get_total_task_count.return_value = 100
        mock_pool.get_id_boundaries.return_value = (0, 99)
        # 第一个分片为空，第二个有数据，后续也有数据
        mock_pool.initialize_shard.side_effect = [0, 50, 30, 20]

        mgr = ShardedTaskManager(mock_pool, optimal_shard_size=25, min_shard_size=10)
        mgr.initialize()

        result = mgr.load_next_shard()

        # 应该跳过空分片并加载下一个有数据的
        assert result is True
        # 应该调用了两次 initialize_shard (第一个为空，加载第二个)
        assert mock_pool.initialize_shard.call_count == 2

    def test_has_more_shards(self, manager):
        """测试 has_more_shards 属性"""
        assert manager.has_more_shards is True

        # 加载所有分片
        while manager.load_next_shard():
            pass

        assert manager.has_more_shards is False


class TestProcessingMetrics:
    """处理指标测试"""

    @pytest.fixture
    def manager(self):
        """创建测试用的管理器"""
        from src.core.scheduler import ShardedTaskManager
        from src.data.base import BaseTaskPool

        mock_pool = MagicMock(spec=BaseTaskPool)
        return ShardedTaskManager(mock_pool)

    def test_update_processing_metrics_initial(self, manager):
        """测试首次更新处理指标"""
        manager.update_processing_metrics(10, 2.0)

        assert manager.processing_metrics["avg_time_per_record"] == 0.2
        assert manager.processing_metrics["records_per_second"] == 5.0

    def test_update_processing_metrics_smoothing(self, manager):
        """测试指标平滑更新"""
        # 首次更新
        manager.update_processing_metrics(10, 2.0)  # 0.2s per record

        # 第二次更新 (更快)
        manager.update_processing_metrics(10, 1.0)  # 0.1s per record

        # 应该使用指数移动平均，不会直接变成 0.1
        assert manager.processing_metrics["avg_time_per_record"] < 0.2
        assert manager.processing_metrics["avg_time_per_record"] > 0.1

    def test_update_processing_metrics_zero_time(self, manager):
        """测试零时间不更新"""
        initial_avg = manager.processing_metrics["avg_time_per_record"]

        manager.update_processing_metrics(10, 0.0)

        assert manager.processing_metrics["avg_time_per_record"] == initial_avg

    def test_update_processing_metrics_zero_count(self, manager):
        """测试零数量不更新"""
        initial_avg = manager.processing_metrics["avg_time_per_record"]

        manager.update_processing_metrics(0, 2.0)

        assert manager.processing_metrics["avg_time_per_record"] == initial_avg


class TestProgressTracking:
    """进度跟踪测试"""

    @pytest.fixture
    def manager(self):
        """创建测试用的管理器"""
        from src.core.scheduler import ShardedTaskManager
        from src.data.base import BaseTaskPool

        mock_pool = MagicMock(spec=BaseTaskPool)
        mock_pool.get_total_task_count.return_value = 100
        mock_pool.get_id_boundaries.return_value = (0, 99)

        mgr = ShardedTaskManager(mock_pool)
        mgr.initialize()
        return mgr

    def test_progress_percent_zero(self, manager):
        """测试初始进度为 0"""
        assert manager.progress_percent == 0.0

    def test_progress_percent_partial(self, manager):
        """测试部分进度"""
        manager.total_processed_successfully = 50

        assert manager.progress_percent == 50.0

    def test_progress_percent_complete(self, manager):
        """测试完成进度"""
        manager.total_processed_successfully = 100

        assert manager.progress_percent == 100.0

    def test_progress_percent_no_tasks(self, manager):
        """测试无任务时进度"""
        manager.total_estimated = 0

        assert manager.progress_percent == 100.0


class TestMemoryMonitoring:
    """内存监控测试"""

    @pytest.fixture
    def manager(self):
        """创建测试用的管理器"""
        from src.core.scheduler import ShardedTaskManager
        from src.data.base import BaseTaskPool

        mock_pool = MagicMock(spec=BaseTaskPool)
        return ShardedTaskManager(mock_pool)

    def test_monitor_memory_usage_updates_tracker(self, manager):
        """测试内存监控更新跟踪器"""
        # 设置检查时间为很久以前
        manager.memory_tracker["last_check_time"] = 0
        manager.memory_tracker["check_interval"] = 0

        manager.monitor_memory_usage()

        # 应该更新了内存使用信息
        if manager._process_info:
            assert manager.memory_tracker["current_memory_usage"] > 0

    def test_monitor_memory_usage_respects_interval(self, manager):
        """测试内存监控遵守检查间隔"""
        # 设置最近刚检查过
        manager.memory_tracker["last_check_time"] = time.time()
        manager.memory_tracker["check_interval"] = 3600  # 1 小时

        initial_usage = manager.memory_tracker["current_memory_usage"]
        manager.monitor_memory_usage()

        # 不应该更新
        assert manager.memory_tracker["current_memory_usage"] == initial_usage


class TestFinalize:
    """结束处理测试"""

    @pytest.fixture
    def manager(self):
        """创建测试用的管理器"""
        from src.core.scheduler import ShardedTaskManager
        from src.data.base import BaseTaskPool

        mock_pool = MagicMock(spec=BaseTaskPool)
        mock_pool.get_total_task_count.return_value = 100
        mock_pool.get_id_boundaries.return_value = (0, 99)

        mgr = ShardedTaskManager(mock_pool)
        mgr.initialize()
        mgr.total_processed_successfully = 80
        return mgr

    def test_finalize_closes_task_pool(self, manager):
        """测试 finalize 关闭任务池"""
        manager.finalize()

        manager.task_pool.close.assert_called_once()

    def test_finalize_handles_close_error(self, manager):
        """测试 finalize 处理关闭错误"""
        manager.task_pool.close.side_effect = Exception("关闭失败")

        # 不应抛出异常
        manager.finalize()
