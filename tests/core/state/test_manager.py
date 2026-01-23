"""
任务状态管理器单元测试

测试 src/core/state/manager.py 的 TaskStateManager 类功能，包括：
- 任务状态追踪 (开始/完成/查询)
- 元数据管理 (创建/获取/删除)
- 过期数据清理
- 并发安全性
"""

import time
from concurrent.futures import ThreadPoolExecutor
import pytest
from src.core.state.manager import TaskStateManager
from src.models.task import TaskMetadata

class TestTaskStateManager:

    @pytest.fixture
    def manager(self):
        return TaskStateManager()

    def test_try_start_task(self, manager):
        task_id = "task_1"
        assert manager.try_start_task(task_id) is True
        assert manager.try_start_task(task_id) is False  # 重复启动
        assert manager.is_task_in_progress(task_id) is True

    def test_complete_task(self, manager):
        task_id = "task_1"
        manager.try_start_task(task_id)
        manager.complete_task(task_id)
        assert manager.is_task_in_progress(task_id) is False
        assert manager.try_start_task(task_id) is True  # 可以再次启动

    def test_get_active_count(self, manager):
        manager.try_start_task("t1")
        manager.try_start_task("t2")
        assert manager.get_active_count() == 2
        manager.complete_task("t1")
        assert manager.get_active_count() == 1

    def test_metadata_lifecycle(self, manager):
        task_id = "meta_task"
        meta1 = manager.get_metadata(task_id)
        assert isinstance(meta1, TaskMetadata)

        # 再次获取应该是同一个对象
        meta2 = manager.get_metadata(task_id)
        assert meta1 is meta2

        manager.remove_metadata(task_id)
        # 删除后再次获取应该是新对象
        meta3 = manager.get_metadata(task_id)
        assert meta3 is not meta1

    def test_cleanup_expired(self, manager):
        # 模拟旧元数据
        old_id = "old_task"
        manager.get_metadata(old_id)

        # 修改创建时间来模拟过期（Hack）
        with manager._metadata_lock:
             manager._task_metadata[old_id].created_at = time.time() - (25 * 3600)

        # 新元数据
        new_id = "new_task"
        manager.get_metadata(new_id)

        removed_count = manager.cleanup_expired(max_age_hours=24)
        assert removed_count == 1

        with manager._metadata_lock:
            assert old_id not in manager._task_metadata
            assert new_id in manager._task_metadata

    def test_concurrency(self, manager):
        # 并发测试：100个线程尝试启动同一个任务，应该只有一个成功
        task_id = "race_task"
        success_count = 0

        def attempt_start():
            nonlocal success_count
            if manager.try_start_task(task_id):
                # 模拟一点工作
                time.sleep(0.001)
                return True
            return False

        with ThreadPoolExecutor(max_workers=20) as executor:
            results = list(executor.map(lambda _: attempt_start(), range(100)))

        successes = [r for r in results if r]
        assert len(successes) == 1
        assert manager.is_task_in_progress(task_id) is True
