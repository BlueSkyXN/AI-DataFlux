"""
任务状态管理模块

本模块提供任务处理状态的集中管理，确保并发环境下的
状态一致性，是连续任务流模式的关键组件。

模块职责:
    - TaskStateManager: 线程安全的任务状态管理器

核心功能:
    1. 防止重复处理: 同一任务不会被同时处理
    2. 元数据分离: 将任务元数据与业务数据解耦
    3. 生命周期管理: 自动清理过期元数据，防止内存泄漏

使用示例:
    from src.core.state import TaskStateManager
    
    manager = TaskStateManager()
    
    # 尝试开始处理任务
    if manager.try_start_task(task_id):
        # 处理任务...
        manager.complete_task(task_id)
    else:
        # 任务已在处理中，跳过
        pass
"""

from .manager import TaskStateManager

__all__ = ["TaskStateManager"]
