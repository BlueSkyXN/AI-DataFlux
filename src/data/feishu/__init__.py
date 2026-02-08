"""
飞书数据源模块

提供飞书电子表格（Sheet）和飞书多维表格（Bitable）两种云端数据源的支持。
参考 XTF 项目经验，在 AI-DataFlux 中实现原生异步飞书客户端，
不使用共享 SDK，不做跨项目耦合。

模块结构:
    - client.py: 原生异步飞书 HTTP 客户端（Token 管理、频控、重试）
    - bitable.py: 飞书多维表格 TaskPool 实现
    - sheet.py: 飞书电子表格 TaskPool 实现

核心设计:
    1. 快照读取 —— 一次性拉取全部数据缓存到内存，后续操作基于快照
    2. ID 映射表 —— task_id ↔ record_id 映射在任务生命周期内不可变
    3. 写入队列 —— 控制写入速率，Bitable 批量上限 500 条/次
    4. 部分失败可追溯 —— 每条记录的写入结果都有记录
    5. Token 自动刷新 —— 请求前检查过期时间，提前 5 分钟刷新
    6. 断点续传 —— 处理进度持久化到本地 JSON 文件
"""

from .client import FeishuClient

__all__ = ["FeishuClient"]
