import logging
import threading
import time
from typing import Any, Dict, Set

from ...models.task import TaskMetadata

class TaskStateManager:
    """
    任务状态管理器
    负责线程安全地管理：
    1. 正在进行的任务集合
    2. 任务元数据（重试次数、错误历史）
    """

    def __init__(self):
        # 任务状态追踪
        self._tasks_in_progress: Set[Any] = set()
        self._progress_lock = threading.Lock()

        # 任务元数据管理
        self._task_metadata: Dict[Any, TaskMetadata] = {}
        self._metadata_lock = threading.Lock()

    def try_start_task(self, task_id: Any) -> bool:
        """
        尝试标记任务为处理中

        Returns:
            bool: 如果成功标记为处理中返回 True，如果已经在处理中返回 False
        """
        with self._progress_lock:
            if task_id in self._tasks_in_progress:
                return False
            self._tasks_in_progress.add(task_id)
            return True

    def complete_task(self, task_id: Any) -> None:
        """标记任务处理完成（无论成功失败）"""
        with self._progress_lock:
            self._tasks_in_progress.discard(task_id)

    def is_task_in_progress(self, task_id: Any) -> bool:
        """检查任务是否处于处理中"""
        with self._progress_lock:
            return task_id in self._tasks_in_progress

    def get_active_count(self) -> int:
        """获取当前活动任务数"""
        with self._progress_lock:
            return len(self._tasks_in_progress)

    def get_metadata(self, task_id: Any) -> TaskMetadata:
        """获取或创建任务元数据"""
        with self._metadata_lock:
            if task_id not in self._task_metadata:
                self._task_metadata[task_id] = TaskMetadata(task_id)
            return self._task_metadata[task_id]

    def remove_metadata(self, task_id: Any) -> None:
        """移除任务元数据"""
        with self._metadata_lock:
            self._task_metadata.pop(task_id, None)

    def cleanup_expired(self, max_age_hours: int = 24) -> int:
        """
        清理过期的任务元数据

        Args:
            max_age_hours: 最大保留小时数

        Returns:
            int: 清理的数量
        """
        cutoff_time = time.time() - (max_age_hours * 3600)

        with self._metadata_lock:
            to_remove = [
                task_id
                for task_id, meta in self._task_metadata.items()
                if meta.created_at < cutoff_time
            ]

            for task_id in to_remove:
                del self._task_metadata[task_id]

            if to_remove:
                logging.info(f"[元数据清理] 清理了 {len(to_remove)} 个过期的任务元数据")

            return len(to_remove)
