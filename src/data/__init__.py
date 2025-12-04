"""数据源抽象层"""

from .base import BaseTaskPool
from .factory import create_task_pool, MYSQL_AVAILABLE, EXCEL_ENABLED

__all__ = [
    "BaseTaskPool",
    "create_task_pool",
    "MYSQL_AVAILABLE",
    "EXCEL_ENABLED",
]
