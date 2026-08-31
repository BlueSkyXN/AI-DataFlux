"""
通用 AI 数据处理器

本模块实现 AI-DataFlux 的核心处理引擎，采用组件化架构设计。
作为系统的协调者 (Coordinator)，负责编排各个组件完成数据处理工作流。

架构设计:
    ┌─────────────────────────────────────────────────────────────────┐
    │              UniversalAIProcessor (协调者/Coordinator)           │
    ├─────────────────────────────────────────────────────────────────┤
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
    │  │ TaskPool    │  │ Scheduler   │  │ FluxAIClient            │ │
    │  │ 数据源读写  │  │ 分片调度    │  │ API 通信/超时管理       │ │
    │  └──────┬──────┘  └──────┬──────┘  └────────────┬────────────┘ │
    │         │                │                      │              │
    │         ▼                ▼                      ▼              │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
    │  │ Content     │  │ TaskState   │  │ RetryStrategy           │ │
    │  │ Processor   │  │ Manager     │  │ 错误分类/重试决策       │ │
    │  │ Prompt/解析 │  │ 状态追踪    │  │ API熔断机制             │ │
    │  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
    └─────────────────────────────────────────────────────────────────┘

核心处理模式 - 连续任务流 (Continuous Task Flow):
    传统批处理: 加载批次 → 并发处理 → 等待全部完成 → 写入结果 → 下一批次
    连续任务流: 动态填充任务池 → 任一完成即处理 → 实时写入 → 持续补充新任务

    优势:
    - 无需等待整批完成，响应更快
    - 更灵活的错误处理和重试机制
    - 资源利用率更高

错误处理策略:
    ┌──────────────┬──────────────────┬────────────┬─────────────────┐
    │ 错误类型      │ 处理动作          │ 数据重载   │ 是否暂停        │
    ├──────────────┼──────────────────┼────────────┼─────────────────┤
    │ API_ERROR    │ PAUSE_THEN_RETRY │ ✓          │ ✓ (api_pause)   │
    │ CONTENT_ERROR│ RETRY            │ ✗          │ ✗               │
    │ SYSTEM_ERROR │ RETRY            │ ✓          │ ✗               │
    │ SOURCE_ERROR │ RETRY            │ ✓          │ ✗               │
    └──────────────┴──────────────────┴────────────┴─────────────────┘

类清单:
    UniversalAIProcessor
        - 功能: 协调各组件完成「加载→API调用→解析→写回」的完整数据处理流
        - 关键属性:
            config (dict): 完整配置
            flux_api_url (str): API 网关 URL
            batch_size (int): 最大并发任务数
            client (FluxAIClient): API 客户端
            content_processor (ContentProcessor): 内容处理器
            state_manager (TaskStateManager): 任务状态管理器
            retry_strategy (RetryStrategy): 重试策略
            task_pool (BaseTaskPool): 数据源任务池
            task_manager (ShardedTaskManager): 分片调度器

        方法:
        ├── __init__(config_path, progress_file=None)
        │     功能: 加载配置 → 初始化 6 个核心组件 → 创建任务池和分片管理器
        │     输入: config_path (str), progress_file (str|None)
        │     异常: ValueError (配置缺失), RuntimeError (组件初始化失败)
        │
        ├── run() -> None
        │     功能: 同步入口，内部调用 asyncio.run() 启动异步处理
        │
        ├── process_shard_async_continuous() -> None  [async]
        │     功能: 异步处理入口 — 初始化分片 → 创建连接池 → 执行主循环 → 清理资源
        │
        ├── _process_loop(session) -> None  [async, 核心]
        │     功能: 连续任务流主循环，包含 8 个阶段:
        │       1.分片轮转 2.填充任务 3.等待完成 4.处理结果
        │       5.API熔断 6.重试入队 7.批量写回 8.监控日志
        │
        ├── _process_one_record(session, record_id, row_data) -> TaskSuccess | TaskFailure  [async]
        │     功能: 单条记录处理（Prompt生成→API调用→响应解析）
        │     输出: 成功时返回 PreparedResult; 失败时返回显式失败阶段和重试信息
        │
        ├── _init_routing_contexts() -> None
        │     功能: 加载路由子配置，为每个规则创建独立的 ContentProcessor 和 Validator
        │
        ├── _load_routing_profile(profile_path) -> dict
        │     功能: 加载路由子配置文件（支持绝对/相对路径）
        │
        ├── _get_routing_context(row_data) -> dict | None
        │     功能: 根据记录的路由字段值匹配路由上下文
        │
        ├── _write_progress() -> None
        │     功能: 原子写入进度 JSON 文件供 GUI 控制面板读取
        │
        └── _cleanup_progress() -> None
              功能: 正常结束时删除进度文件

模块依赖:
    - asyncio, aiohttp: 异步 HTTP 通信
    - ..config.settings: 配置加载与合并
    - ..models.errors.ErrorType: 错误类型枚举
    - ..data: 任务池工厂
    - .scheduler.ShardedTaskManager: 分片调度
    - .validator.JsonValidator: 字段验证
    - .content.ContentProcessor: Prompt 生成与响应解析
    - .clients.FluxAIClient: API 客户端
    - .state.TaskStateManager: 任务状态管理
    - .retry.RetryStrategy: 重试决策

使用示例:
    # 基本使用
    processor = UniversalAIProcessor("config.yaml")
    processor.run()  # 同步运行

    # 异步使用
    await processor.process_shard_async_continuous()

配置要点:
    - job.gateway_url: API 网关地址
    - job.concurrency.batch_size: 最大并发任务数
    - job.retry.api_pause_duration_seconds: API 熔断暂停时长
    - job.retry.task_max_attempts: 各类错误的最大总尝试次数

重构历史:
    2026-01-22: 采用组件化架构重构，拆分为独立的处理组件
"""

import asyncio
from collections import deque
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
from collections.abc import Callable

import aiohttp

from ..config.settings import (
    compile_job_config,
    init_logging,
    load_config,
    load_routing_profile,
)
from ..models.errors import ErrorType
from ..data import create_task_pool
from .scheduler import ShardedTaskManager
from .validator import JsonValidator

# 新组件
from .content import ContentProcessor
from .clients import FluxAIClient
from .contracts import (
    FailureStage,
    PreparedResult,
    SourceOperationError,
    TaskFailure,
    TaskSuccess,
)
from .state import TaskStateManager
from .retry import RetryAction, RetryDecision, RetryStrategy


class UniversalAIProcessor:
    """
    通用 AI 数据处理器

    作为协调者 (Coordinator)，负责编排各组件完成数据处理工作流：
    1. TaskPool: 数据源读写
    2. ShardedTaskManager: 分片管理
    3. ContentProcessor: Prompt生成与结果解析
    4. FluxAIClient: API 通信
    5. TaskStateManager: 状态管理
    6. RetryStrategy: 错误重试决策

    组件职责分离:
        - 本类只负责编排协调，不直接实现具体逻辑
        - 各组件独立可测试，便于单元测试和功能扩展
        - 组件之间通过明确接口通信，降低耦合度

    生命周期:
        __init__ → run() → process_shard_async_continuous() → finalize()
              ↓           ↓
         配置加载     异步处理循环
         组件初始化   分片轮转
                     任务处理
                     结果写入

    Attributes:
        config: 配置字典
        flux_api_url: API 网关 URL
        batch_size: 最大并发任务数
        client: FluxAIClient 实例
        content_processor: ContentProcessor 实例
        state_manager: TaskStateManager 实例
        retry_strategy: RetryStrategy 实例
        task_pool: BaseTaskPool 实例
        task_manager: ShardedTaskManager 实例
    """

    def __init__(self, config_path: str, progress_file: str | None = None):
        """
        初始化处理器

        Args:
            config_path: 配置文件路径
            progress_file: 进度文件路径 (可选，用于 GUI 控制面板)

        Raises:
            ValueError: 配置加载失败
        """
        # 保存进度文件路径
        self.progress_file = progress_file

        # 1. 加载配置
        try:
            config_path_obj = Path(config_path)
            self.config_base_dir = config_path_obj.parent
            self.root_config = load_config(config_path)
            self.config = compile_job_config(self.root_config, config_path_obj)
        except Exception as e:
            raise ValueError(f"无法加载配置文件: {e}") from e

        # 初始化日志
        global_cfg = self.config.get("global", {})
        init_logging(global_cfg.get("log"))

        # 2. 读取关键配置
        # API 配置
        self.flux_api_url = global_cfg.get("flux_api_url")
        if not self.flux_api_url:
            raise ValueError("配置文件 [global] 部分缺少 'flux_api_url'")

        # 数据源配置
        datasource_cfg = self.config.get("datasource", {})

        # 并发配置
        concurrency_cfg = datasource_cfg.get("concurrency", {})
        self.batch_size = concurrency_cfg.get("batch_size", 100)
        self.max_in_flight = concurrency_cfg.get("max_in_flight", self.batch_size)
        self.max_connections = concurrency_cfg.get("max_connections", 1000)
        self.max_connections_per_host = concurrency_cfg.get(
            "max_connections_per_host", 0
        )
        retry_cfg = self.config.get("retry", {})
        writeback_cfg = self.config.get("writeback", {})
        self.write_retry_limit = int(writeback_cfg.get("commit_max_attempts", 3)) - 1
        self._job_cancel_event: asyncio.Event | None = None
        self._target_concurrency_provider: Callable[[], int] | None = None
        self._job_tracker: Any | None = None

        api_pause_duration = float(retry_cfg.get("api_pause_duration_seconds", 2.0))
        api_error_trigger_window = float(
            retry_cfg.get("api_error_trigger_window_seconds", 2.0)
        )

        task_attempts_cfg = retry_cfg.get("task_max_attempts", {})
        max_attempts = {
            ErrorType.API: int(task_attempts_cfg.get("api_error", 4)),
            ErrorType.CONTENT: int(task_attempts_cfg.get("content_error", 2)),
            ErrorType.SYSTEM: int(task_attempts_cfg.get("system_error", 3)),
            ErrorType.SOURCE: int(task_attempts_cfg.get("source_error", 3)),
        }
        self.source_max_attempts = max_attempts[ErrorType.SOURCE]

        # 3. 初始化各组件

        # API 客户端
        logging.info(f"API 端点: {self.flux_api_url}")
        gateway_token = (
            os.getenv("DATAFLUX_TOKEN", "").strip()
            or str(self.config.get("server", {}).get("token", "")).strip()
        )
        self.client = FluxAIClient(self.flux_api_url, api_token=gateway_token)

        # 默认验证器
        self.validator = JsonValidator()
        self.validator.configure(self.config.get("validation"))

        # 状态管理器
        self.state_manager = TaskStateManager()

        # 重试策略
        self.retry_strategy = RetryStrategy(
            max_attempts=max_attempts,
            api_pause_duration=api_pause_duration,
            api_error_trigger_window=api_error_trigger_window,
            base_backoff_seconds=1,
            max_backoff_seconds=30,
        )

        # 4. 初始化数据源和分片管理器
        # 注意：需要在路由初始化之前处理 columns_to_extract
        self.columns_to_extract = self.config.get("columns_to_extract", [])
        self.columns_to_write = self.config.get("columns_to_write", {})

        if not self.columns_to_extract or not self.columns_to_write:
            raise ValueError("缺少 columns_to_extract 或 columns_to_write 配置")

        # 处理路由字段
        # - 如果用户显式声明 → 作为业务字段，提供给 AI
        # - 如果用户未声明 → 自动追加但排除出 Prompt（仅用于路由决策）
        self.routing_enabled = False
        self.routing_field = None
        self.routing_field_is_implicit = False  # 是否为隐式路由字段
        self.routing_contexts: dict[str, dict[str, Any]] = {}

        routing_cfg = self.config.get("routing", {})
        if routing_cfg.get("enabled", False):
            routing_field = routing_cfg.get("field")
            if routing_field:
                self.routing_field = routing_field
                self.routing_enabled = True

                if routing_field not in self.columns_to_extract:
                    # 隐式路由字段：自动追加，标记为排除
                    logging.info(
                        f"路由字段 '{routing_field}' 自动追加到 columns_to_extract（不发给 AI）"
                    )
                    self.columns_to_extract.append(routing_field)
                    self.routing_field_is_implicit = True
                else:
                    # 显式路由字段：用户明确声明，作为业务字段
                    logging.info(
                        f"路由字段 '{routing_field}' 为显式声明的业务字段（会发给 AI）"
                    )
                    self.routing_field_is_implicit = False

        # 提前读取 prompt 配置（_init_routing_contexts 需要这些属性）
        prompt_cfg = self.config.get("prompt", {})
        self.ai_model = prompt_cfg.get("model", "auto")
        self.ai_temperature = prompt_cfg.get("temperature", 0.7)
        self.ai_temperature_override = prompt_cfg.get("temperature_override", True)
        self.ai_system_prompt = prompt_cfg.get("system_prompt")
        self.ai_use_json_schema = prompt_cfg.get("use_json_schema", False)

        # 初始化规则路由上下文（需要在 routing_field_is_implicit 和 ai_* 属性设置后调用）
        self._init_routing_contexts()

        # 创建默认内容处理器（需要在路由初始化后创建，以确定 exclude_fields）
        # 只有隐式路由字段才排除（用户未显式声明的字段）
        exclude_fields = []
        if (
            self.routing_enabled
            and self.routing_field
            and self.routing_field_is_implicit
        ):
            exclude_fields = [self.routing_field]

        self.content_processor = ContentProcessor(
            prompt_template=prompt_cfg.get("template", ""),
            required_fields=prompt_cfg.get("required_fields", []),
            validator=self.validator,
            use_json_schema=self.ai_use_json_schema,
            exclude_fields=exclude_fields,
        )

        try:
            self.task_pool = create_task_pool(
                self.config,
                self.columns_to_extract,
                self.columns_to_write,
            )
        except Exception as e:
            raise RuntimeError(f"无法初始化数据源任务池: {e}") from e

        shard_size = concurrency_cfg.get("shard_size", 10000)
        min_shard_size = concurrency_cfg.get("min_shard_size", 1000)
        max_shard_size = concurrency_cfg.get("max_shard_size", 50000)

        try:
            self.task_manager = ShardedTaskManager(
                self.task_pool,
                shard_size,
                min_shard_size,
                max_shard_size,
                max_attempts,
            )
        except Exception as e:
            raise RuntimeError(f"无法初始化分片任务管理器: {e}") from e

        logging.info("UniversalAIProcessor 初始化完成 (组件化版本)")

    def _write_progress(self) -> None:
        """
        写入进度文件 (GUI 控制面板读取)

        仅当 progress_file 参数被设置时才写入。
        使用原子写入 (写临时文件后 rename) 保证数据一致性。
        """
        if not self.progress_file:
            return

        try:
            # 计算当前分片
            current_shard = min(
                self.task_manager.current_shard_index, self.task_manager.total_shards
            )

            data = {
                "total": self.task_manager.total_estimated,
                "processed": self.task_manager.total_processed_successfully,
                "active": self.state_manager.get_active_count(),
                "shard": f"{current_shard}/{self.task_manager.total_shards}",
                "errors": self.task_manager.max_retries_exceeded_count,
                "ts": time.time(),
            }

            tmp_path = self.progress_file + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp_path, self.progress_file)
        except Exception as e:
            logging.debug(f"写入进度文件失败: {e}")

    def _cleanup_progress(self) -> None:
        """
        清理进度文件

        在处理正常结束时调用，删除进度文件。
        """
        if not self.progress_file:
            return

        try:
            if os.path.exists(self.progress_file):
                os.remove(self.progress_file)
        except OSError:
            pass

    def _init_routing_contexts(self) -> None:
        """
        初始化规则路由上下文

        加载所有子配置文件，为每个路由规则创建独立的 ContentProcessor 和 Validator。

        前提条件:
            - self.routing_enabled 已设置
            - self.routing_field 已设置
            - self.routing_field_is_implicit 已设置
        """
        if not self.routing_enabled:
            return

        routing_cfg = self.config.get("routing", {})
        subtasks = routing_cfg.get("subtasks")
        if not isinstance(subtasks, list) or not subtasks:
            raise ValueError("routing 配置缺少 subtasks")

        for idx, subtask in enumerate(subtasks):
            if "match" not in subtask or "profile" not in subtask:
                raise ValueError(f"routing.subtasks[{idx}] 必须包含 match 和 profile")

            match_value = str(subtask["match"])
            profile_path = subtask["profile"]
            profile_config = self._load_routing_profile(profile_path)

            allowed_keys = {"prompt", "validation"}
            unknown_keys = set(profile_config.keys()) - allowed_keys
            if unknown_keys:
                raise ValueError(
                    f"routing 子配置仅允许 prompt/validation，发现非法键: {sorted(unknown_keys)}"
                )

            merged_config = dict(self.config)
            for section in ("prompt", "validation"):
                if section in profile_config:
                    merged_config[section] = {
                        **self.config.get(section, {}),
                        **profile_config[section],
                    }
            prompt_cfg = merged_config.get("prompt", {})
            validator = JsonValidator()
            validator.configure(merged_config.get("validation"))

            # 路由上下文的 ContentProcessor 也遵循隐式/显式规则
            exclude_fields = []
            if self.routing_field and self.routing_field_is_implicit:
                exclude_fields = [self.routing_field]

            processor = ContentProcessor(
                prompt_template=prompt_cfg.get("template", ""),
                required_fields=prompt_cfg.get("required_fields", []),
                validator=validator,
                use_json_schema=prompt_cfg.get("use_json_schema", False),
                exclude_fields=exclude_fields,
            )

            self.routing_contexts[match_value] = {
                "content_processor": processor,
                "validator": validator,
                "model": prompt_cfg.get("model", self.ai_model),
                "temperature": prompt_cfg.get("temperature", self.ai_temperature),
                "temperature_override": prompt_cfg.get(
                    "temperature_override", self.ai_temperature_override
                ),
                "system_prompt": prompt_cfg.get("system_prompt"),
                "use_json_schema": prompt_cfg.get(
                    "use_json_schema", self.ai_use_json_schema
                ),
            }

    def _load_routing_profile(self, profile_path: str) -> dict[str, Any]:
        """
        加载规则路由的子配置文件

        支持绝对路径和相对路径（相对于主配置文件目录）

        Args:
            profile_path: 子配置文件路径

        Returns:
            子配置字典
        """
        profile_path_obj = Path(profile_path)
        if not profile_path_obj.is_absolute():
            profile_path_obj = self.config_base_dir / profile_path_obj

        return load_routing_profile(profile_path_obj)

    def _get_routing_context(self, row_data: Dict[str, Any]) -> dict[str, Any] | None:
        """
        根据记录数据获取路由上下文

        根据 routing.field 指定的字段值，查找匹配的路由上下文。

        行为说明:
            - 路由未启用时返回 None（使用默认配置）
            - 路由字段不存在于记录中时返回 None（使用默认配置）
            - 路由字段值没有匹配规则时返回 None（使用默认配置）
            - 匹配成功时返回对应的路由上下文

        Args:
            row_data: 记录数据字典

        Returns:
            匹配的路由上下文或 None
        """
        if not self.routing_enabled or not self.routing_field:
            return None

        # 字段不存在时使用默认配置
        if self.routing_field not in row_data:
            logging.debug(
                f"routing.field '{self.routing_field}' 不存在于记录中，使用默认配置"
            )
            return None

        match_value = str(row_data.get(self.routing_field))
        context = self.routing_contexts.get(match_value)

        if context is None:
            logging.debug(
                f"routing 未找到匹配规则: {self.routing_field}='{match_value}'，使用默认配置"
            )

        return context

    def configure_job_control(
        self,
        *,
        cancel_event: asyncio.Event | None = None,
        target_concurrency_provider: Callable[[], int] | None = None,
        job_tracker: Any | None = None,
    ) -> None:
        self._job_cancel_event = cancel_event
        self._target_concurrency_provider = target_concurrency_provider
        self._job_tracker = job_tracker

    async def process_shard_async_continuous(self) -> bool:
        """
        连续任务流模式的异步处理

        核心处理入口，实现连续任务流 (Continuous Task Flow) 模式:
        1. 初始化分片管理器
        2. 创建 HTTP 连接池
        3. 进入主处理循环
        4. 完成后关闭资源

        连续任务流的关键特性:
            - 动态任务填充: 任务完成后立即补充新任务，保持并发度
            - 实时结果处理: 无需等待整批完成，单个任务完成即处理结果
            - 灵活错误处理: 可针对单个任务进行重试，不影响其他任务

        HTTP 连接池配置:
            - limit: 总连接数上限 (max_connections)
            - limit_per_host: 单主机连接数 (max_connections_per_host)

        Raises:
            Exception: 处理过程中的异常会被记录，但不会中断整体流程
        """
        try:
            self.task_manager.start_time = time.time()
            self.task_manager.total_estimated = await self._run_source_operation(
                "count",
                lambda: asyncio.to_thread(self.task_pool.get_total_task_count),
            )
            if self.task_manager.total_estimated <= 0:
                logging.info("数据源中没有需要处理的任务")
                return True
            self.task_manager.current_shard_index = 0
            self.task_manager.total_shards = max(
                1,
                (self.task_manager.total_estimated + max(1, self.batch_size) - 1)
                // max(1, self.batch_size),
            )

            # 创建连接池
            connector = aiohttp.TCPConnector(
                limit=self.max_connections, limit_per_host=self.max_connections_per_host
            )
            async with aiohttp.ClientSession(connector=connector) as session:
                return await self._process_loop(session)
        finally:
            self.task_manager.finalize()

    async def _run_source_operation(
        self,
        operation: str,
        invoke: Callable[[], Any],
    ) -> Any:
        """Retry count/scan source operations without involving API pause."""

        last_failure: TaskFailure | None = None
        for attempt in range(1, self.source_max_attempts + 1):
            try:
                value = invoke()
                if hasattr(value, "__await__"):
                    value = await value
                return value
            except Exception as exc:
                last_failure = TaskFailure(
                    stage=FailureStage.SOURCE,
                    code=f"source_{operation}_failed",
                    message=f"{type(exc).__name__}: {exc}",
                    retryable=True,
                )
                if attempt >= self.source_max_attempts:
                    break
                delay = min(30.0, float(2 ** (attempt - 1)))
                logging.warning(
                    "datasource %s 第 %s/%s 次尝试失败，%.2fs 后重试: %s",
                    operation,
                    attempt,
                    self.source_max_attempts,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)
        assert last_failure is not None
        raise SourceOperationError(operation, last_failure)

    async def _reload_record_for_retry(
        self,
        record_id: Any,
        checkpoint_data: dict[str, Any],
        metadata: Any,
    ) -> dict[str, Any] | TaskFailure:
        """Reload one record with its independent durable source budget."""

        while True:
            attempt = metadata.get_retry_count(ErrorType.SOURCE) + 1
            try:
                reloaded = await self.task_pool.reload([record_id])
            except Exception as exc:
                failure = TaskFailure(
                    stage=FailureStage.SOURCE,
                    code="source_reload_failed",
                    message=f"{type(exc).__name__}: {exc}",
                    retryable=True,
                )
            else:
                data = reloaded.get(record_id)
                if data is not None:
                    return data
                return TaskFailure(
                    stage=FailureStage.SOURCE,
                    code="source_record_missing",
                    message="record was not returned by datasource reload",
                    retryable=False,
                )

            if attempt >= self.source_max_attempts:
                return failure
            metadata.increment_retry(ErrorType.SOURCE)
            self.task_manager.retried_tasks_count[ErrorType.SOURCE] += 1
            if self._job_tracker is not None:
                self._job_tracker.mark_pending(
                    record_id,
                    checkpoint_data,
                    error_code=failure.code,
                    retry_error_type=ErrorType.SOURCE.value,
                )
            delay = min(30.0, float(2 ** (attempt - 1)))
            logging.warning(
                "记录[%s] datasource reload 第 %s/%s 次尝试失败，%.2fs 后重试",
                record_id,
                attempt,
                self.source_max_attempts,
                delay,
            )
            await asyncio.sleep(delay)

    async def _process_loop(self, session: aiohttp.ClientSession) -> bool:
        """
        主处理循环

        实现连续任务流的核心循环逻辑，包含以下 8 个阶段:

        处理流程:
            ┌─────────────────────────────────────────────────────────┐
            │                      主循环开始                          │
            └─────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   │
            ┌───────────────┐                          │
            │ 1. 分片轮转检查│ ─────── 无更多分片 ─────→ 结束
            └───────┬───────┘
                    ▼
            ┌───────────────┐
            │ 2. 填充任务池  │ ← Backpressure 控制
            └───────┬───────┘   (保持 batch_size 并发度)
                    ▼
            ┌───────────────┐
            │ 3. 等待任务完成│ ← asyncio.wait(FIRST_COMPLETED)
            └───────┬───────┘
                    ▼
            ┌───────────────┐
            │ 4. 处理完成任务│ → 成功: 存入结果缓冲
            └───────┬───────┘   失败: 重试决策
                    ▼
            ┌───────────────┐
            │ 5. API 熔断检查│ → 触发: 暂停 api_pause_duration 秒
            └───────┬───────┘
                    ▼
            ┌───────────────┐
            │ 6. 重试任务入队│ → 加到队首优先处理
            └───────┬───────┘
                    ▼
            ┌───────────────┐
            │ 7. 批量写回结果│
            └───────┬───────┘
                    ▼
            ┌───────────────┐
            │ 8. 监控与日志  │ ← 每 5 秒输出进度
            └───────┬───────┘   清理过期元数据
                    │
                    └─────────────────→ 返回循环开始

        Args:
            session: aiohttp 客户端会话
        """
        current_shard_num = 0
        active_tasks: Set[asyncio.Task] = set()
        task_id_map: Dict[asyncio.Task, Tuple[Any, Dict[str, Any]]] = {}
        source_queue: deque[Tuple[Any, Dict[str, Any]]] = deque()
        retry_queue: deque[Tuple[Any, Dict[str, Any]]] = deque()
        scan_cursor: Any | None = None
        scan_exhausted = False
        results_buffer: Dict[Any, Dict[str, Any]] = {}
        successful_result_ids: set[Any] = set()

        last_progress_time = time.time()

        while True:
            if self._job_cancel_event is not None and self._job_cancel_event.is_set():
                if self._job_tracker is not None:
                    for record_id, data in task_id_map.values():
                        self._job_tracker.mark_pending(
                            record_id,
                            data,
                            error_code="cancelled",
                        )
                for task in active_tasks:
                    task.cancel()
                if active_tasks:
                    await asyncio.gather(*active_tasks, return_exceptions=True)
                logging.info("收到 Job 取消请求，停止调度新任务")
                return False

            if self._target_concurrency_provider is not None:
                try:
                    self.max_in_flight = max(
                        1, int(self._target_concurrency_provider())
                    )
                except Exception:
                    logging.exception("读取动态并发目标失败，保留当前值")

            if (
                scan_exhausted
                and not source_queue
                and not retry_queue
                and not active_tasks
            ):
                logging.info("所有 datasource cursor 页面处理完毕")
                return True

            # 1. 通过 datasource-owned cursor 填充连续任务流。
            space_available = self.max_in_flight - len(active_tasks)
            tasks_batch: list[Tuple[Any, Dict[str, Any]]] = []
            while space_available > len(tasks_batch):
                if retry_queue:
                    tasks_batch.append(retry_queue.popleft())
                    continue
                if source_queue:
                    tasks_batch.append(source_queue.popleft())
                    continue
                if scan_exhausted:
                    break

                previous_cursor = scan_cursor
                page = await self._run_source_operation(
                    "scan",
                    lambda: self.task_pool.scan(scan_cursor, self.batch_size),
                )
                scan_cursor = page.next_cursor
                if scan_cursor is None:
                    scan_exhausted = True
                elif scan_cursor == previous_cursor:
                    raise RuntimeError("datasource scan cursor did not advance")

                if page.records:
                    records_to_process = page.records
                    if self._job_tracker is not None:
                        records_to_process = tuple(
                            record
                            for record in page.records
                            if self._job_tracker.should_process_scanned(
                                record.record_id
                            )
                        )
                    current_shard_num += 1
                    self.task_manager.current_shard_index = current_shard_num
                    self.task_manager.total_shards = max(
                        self.task_manager.total_shards,
                        current_shard_num,
                    )
                    self.task_manager.total_estimated = max(
                        self.task_manager.total_estimated,
                        self.task_manager.total_processed_successfully
                        + len(active_tasks)
                        + len(source_queue)
                        + len(retry_queue)
                        + len(records_to_process),
                    )
                    logging.info(
                        "--- 扫描 datasource 页面 %s/%s (%s 条记录) ---",
                        current_shard_num,
                        self.task_manager.total_shards,
                        len(page.records),
                    )
                    if self._job_tracker is not None and records_to_process:
                        self._job_tracker.register_scan_batch(
                            self._job_tracker.new_scan_shard_id(),
                            (
                                (record.record_id, record.data)
                                for record in records_to_process
                            ),
                            cursor=page.next_cursor,
                        )
                    source_queue.extend(
                        (record.record_id, record.data) for record in records_to_process
                    )

            for record_id, data in tasks_batch:
                if self.state_manager.try_start_task(record_id):
                    metadata = self.state_manager.get_metadata(record_id)
                    if self._job_tracker is not None:
                        durable_retry_counts = self._job_tracker.retry_counts_for(
                            record_id
                        )
                        for error_type in ErrorType:
                            metadata.retry_counts[error_type] = (
                                durable_retry_counts.get(
                                    error_type.value,
                                    0,
                                )
                            )
                        self._job_tracker.mark_in_flight(record_id, data)

                    task = asyncio.create_task(
                        self._process_one_record(session, record_id, data)
                    )
                    task_id_map[task] = (record_id, data)
                    active_tasks.add(task)
                else:
                    retry_queue.append((record_id, data))

            if not active_tasks:
                if scan_exhausted and not source_queue and not retry_queue:
                    logging.info("所有 datasource cursor 页面处理完毕")
                    return True
                await asyncio.sleep(0.1)
                continue

            # 2. 等待任一任务完成
            done, pending = await asyncio.wait(
                active_tasks, timeout=1.0, return_when=asyncio.FIRST_COMPLETED
            )
            active_tasks = pending

            # 3. 处理完成的任务
            tasks_to_retry: List[Tuple[Any, Dict[str, Any]]] = []
            should_pause_api = False
            pause_duration = 0.0
            retry_delay = 0.0

            for completed_task in done:
                record_id, original_data = task_id_map.pop(completed_task)
                self.state_manager.complete_task(record_id)

                try:
                    result = completed_task.result()
                    if isinstance(result, TaskFailure):
                        error_type = result.error_type
                        metadata = self.state_manager.get_metadata(record_id)
                        metadata.add_error(error_type, result.message)

                        if not result.retryable:
                            decision = RetryDecision(action=RetryAction.FAIL)
                        else:
                            decision = self.retry_strategy.decide(error_type, metadata)

                        if decision.action in [
                            RetryAction.RETRY,
                            RetryAction.PAUSE_THEN_RETRY,
                        ]:
                            metadata.increment_retry(error_type)
                            self.task_manager.retried_tasks_count[error_type] += 1
                            logging.warning(
                                "记录[%s] %s [%s]: %s -> 重试",
                                record_id,
                                result.stage.value,
                                result.code,
                                result.message,
                            )

                            if self._job_tracker is not None:
                                self._job_tracker.mark_pending(
                                    record_id,
                                    original_data,
                                    error_code=result.code,
                                    retry_error_type=error_type.value,
                                )

                            retry_data: dict[str, Any] | None = original_data
                            if decision.reload_data:
                                reloaded = await self._reload_record_for_retry(
                                    record_id,
                                    original_data,
                                    metadata,
                                )
                                if isinstance(reloaded, TaskFailure):
                                    metadata.add_error(
                                        ErrorType.SOURCE,
                                        reloaded.message,
                                    )
                                    logging.error(
                                        "记录[%s] datasource reload 失败 [%s]: %s",
                                        record_id,
                                        reloaded.code,
                                        reloaded.message,
                                    )
                                    self.task_manager.max_retries_exceeded_count += 1
                                    if self._job_tracker is not None:
                                        self._job_tracker.mark_failed(
                                            record_id,
                                            error_code=reloaded.code,
                                        )
                                    self.state_manager.remove_metadata(record_id)
                                    continue
                                retry_data = reloaded

                                if self._job_tracker is not None:
                                    self._job_tracker.mark_pending(
                                        record_id,
                                        retry_data,
                                        error_code=result.code,
                                    )

                            tasks_to_retry.append((record_id, retry_data))

                            if decision.action == RetryAction.PAUSE_THEN_RETRY:
                                should_pause_api = True
                                pause_duration = decision.pause_duration
                            retry_delay = max(
                                retry_delay,
                                decision.retry_delay,
                                result.retry_after_seconds,
                            )

                        else:  # FAIL
                            logging.error(
                                "记录[%s] %s 终止 [%s]: %s",
                                record_id,
                                result.stage.value,
                                result.code,
                                result.message,
                            )
                            self.task_manager.max_retries_exceeded_count += 1

                            if self._job_tracker is not None:
                                self._job_tracker.mark_failed(
                                    record_id,
                                    error_code=result.code,
                                )
                            self.state_manager.remove_metadata(record_id)
                    elif isinstance(result, TaskSuccess):
                        prepared = result.prepared_result
                        if prepared.record_id != record_id:
                            raise ValueError(
                                "TaskSuccess record_id does not match scheduled record"
                            )
                        if self._job_tracker is not None:
                            self._job_tracker.mark_ai_complete(
                                record_id,
                                prepared.values,
                            )
                        results_buffer[record_id] = prepared.values
                        successful_result_ids.add(record_id)
                    else:
                        raise TypeError(
                            "_process_one_record must return TaskSuccess or TaskFailure"
                        )

                except Exception as e:
                    logging.error(f"处理结果时发生未捕获异常: {e}")
                    self.state_manager.remove_metadata(record_id)
                    raise

            # 4. 执行 API 暂停
            sleep_duration = max(pause_duration, retry_delay)
            if sleep_duration > 0:
                logging.warning("重试退避 %.2fs...", sleep_duration)
                await asyncio.sleep(sleep_duration)
                if should_pause_api:
                    self.retry_strategy.record_pause()

            # 5. 重新入队重试任务
            for r_id, r_data in reversed(tasks_to_retry):
                retry_queue.appendleft((r_id, r_data))

            # 6. 批量回写结果
            if results_buffer:
                persisted_ids = await self._persist_results_with_retry(results_buffer)
                for record_id in persisted_ids:
                    if record_id in successful_result_ids:
                        self.task_manager.total_processed_successfully += 1
                        if self._job_tracker is not None:
                            self._job_tracker.mark_persisted(record_id)
                    self.state_manager.remove_metadata(record_id)
                failed_write_ids = set(results_buffer) - persisted_ids
                for record_id in failed_write_ids:
                    logging.error(
                        "记录[%s] AI 处理完成但结果未持久化，不能计为成功",
                        record_id,
                    )
                    self.task_manager.max_retries_exceeded_count += 1
                    if self._job_tracker is not None:
                        self._job_tracker.mark_failed(
                            record_id,
                            error_code="writeback_failed",
                            result=(
                                results_buffer[record_id]
                                if record_id in successful_result_ids
                                else None
                            ),
                        )
                    self.state_manager.remove_metadata(record_id)
                results_buffer.clear()
                successful_result_ids.clear()

            # 7. 监控与日志
            current_time = time.time()
            if current_time - last_progress_time >= 5.0:
                self.task_manager.monitor_memory_usage()
                logging.info(
                    f"进度: {self.task_manager.progress_percent:.1f}% | "
                    f"成功: {self.task_manager.total_processed_successfully} | "
                    f"活动: {self.state_manager.get_active_count()}"
                )
                last_progress_time = current_time
                # 清理过期元数据
                self.state_manager.cleanup_expired()
                # 写入进度文件 (GUI 控制面板读取)
                self._write_progress()

    async def _persist_results_with_retry(
        self, results: dict[Any, dict[str, Any]]
    ) -> set[Any]:
        """Persist result batches and only acknowledge datasource receipts."""

        pending = dict(results)
        persisted: set[Any] = set()
        attempts = self.write_retry_limit + 1

        for attempt in range(attempts):
            if not pending:
                break
            batch_id = uuid.uuid4().hex
            try:
                receipt = await self.task_pool.write_results(batch_id, pending)
                pending_ids = set(pending)
                acknowledged = set(receipt.persisted_ids)
                receipt_failure_ids = {
                    failure.record_id for failure in receipt.failures
                }
                unknown_ids = (acknowledged | receipt_failure_ids) - pending_ids
                conflicting_ids = acknowledged & receipt_failure_ids
                if unknown_ids or conflicting_ids:
                    raise ValueError(
                        "invalid writeback receipt: "
                        f"unknown_ids={sorted(map(str, unknown_ids))}, "
                        f"conflicting_ids={sorted(map(str, conflicting_ids))}"
                    )
                persisted.update(acknowledged)
                retryable_failed_ids = {
                    failure.record_id
                    for failure in receipt.failures
                    if failure.retryable and failure.record_id not in acknowledged
                }
                permanent_failed_ids = {
                    failure.record_id
                    for failure in receipt.failures
                    if not failure.retryable and failure.record_id not in acknowledged
                }
                failed_ids = retryable_failed_ids | permanent_failed_ids
                unacknowledged = set(pending) - acknowledged - failed_ids
                retry_ids = retryable_failed_ids | unacknowledged
                pending = {
                    record_id: value
                    for record_id, value in pending.items()
                    if record_id in retry_ids
                }
            except Exception as exc:
                logging.error(
                    "结果写回批次 %s 失败 (%s/%s): %s",
                    batch_id,
                    attempt + 1,
                    attempts,
                    exc,
                    exc_info=True,
                )

            if pending and attempt + 1 < attempts:
                await asyncio.sleep(min(2**attempt, 5))

        return persisted

    async def persist_checkpoint_results(
        self, results: dict[Any, dict[str, Any]]
    ) -> set[Any]:
        """Replay AI-complete outputs before datasource scanning resumes."""

        persisted = await self._persist_results_with_retry(results)
        if self._job_tracker is not None:
            for record_id in persisted:
                self._job_tracker.mark_persisted(record_id)
            for record_id in set(results) - persisted:
                self._job_tracker.mark_failed(
                    record_id,
                    error_code="checkpoint_writeback_failed",
                    result=results[record_id],
                )
        return persisted

    async def _process_one_record(
        self, session: aiohttp.ClientSession, record_id: Any, row_data: Dict[str, Any]
    ) -> TaskSuccess | TaskFailure:
        """
        处理单条记录

        完成单条数据的完整处理流程:
        1. 生成 Prompt - 使用 ContentProcessor 将原始数据转换为 AI 输入
        2. 调用 API - 通过 FluxAIClient 发送请求并获取响应
        3. 解析结果 - 提取 AI 返回的结构化数据

        异常处理策略:
            - aiohttp 异常 (连接失败、超时): 返回 API_ERROR
            - Prompt 生成失败: 返回 SYSTEM_ERROR
            - 其他未知异常: 返回 SYSTEM_ERROR 并记录详情

        Args:
            session: aiohttp 客户端会话
            record_id: 记录唯一标识符
            row_data: 原始记录数据字典

        Returns:
            TaskSuccess: 包含已校验的 PreparedResult。
            TaskFailure: 包含失败阶段、稳定错误码、消息和重试属性。

        Example:
            TaskSuccess(PreparedResult.create(record_id, {"field1": "value1"}))
            TaskFailure(FailureStage.MODEL, "model_transport_error", "timeout", True)
        """
        try:
            routing_context = self._get_routing_context(row_data)
            if routing_context:
                content_processor = routing_context["content_processor"]
                model = routing_context["model"]
                temperature = routing_context["temperature"]
                temperature_override = routing_context["temperature_override"]
                system_prompt = routing_context["system_prompt"]
                use_schema = routing_context["use_json_schema"]
            else:
                content_processor = self.content_processor
                model = self.ai_model
                temperature = self.ai_temperature
                temperature_override = self.ai_temperature_override
                system_prompt = self.ai_system_prompt
                use_schema = self.ai_use_json_schema

            # 1. 生成 Prompt
            prompt = content_processor.create_prompt(row_data)
            if not prompt:
                return TaskFailure(
                    stage=FailureStage.SYSTEM,
                    code="prompt_generation_failed",
                    message="prompt generation returned empty content",
                    retryable=True,
                )

            # 2. 调用 API
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response_content = await self.client.call(
                session,
                messages,
                model=model,
                temperature=temperature if temperature_override else None,
                use_json_schema=use_schema,
                json_schema=(content_processor.build_schema() if use_schema else None),
            )

            # 3. 解析结果
            result = content_processor.parse_response(response_content)
            if "_error" in result:
                validation_errors = result.get("_validation_errors")
                message = str(result["_error"])
                if validation_errors:
                    message = f"{message}: {validation_errors}"
                return TaskFailure(
                    stage=FailureStage.CONTENT,
                    code=str(result["_error"]),
                    message=message,
                    retryable=True,
                )
            return TaskSuccess(PreparedResult.create(record_id, result))

        except aiohttp.ClientResponseError as e:
            retryable = e.status in {408, 429} or 500 <= e.status <= 599
            return TaskFailure(
                stage=FailureStage.MODEL,
                code=f"model_http_{e.status}",
                message=f"model request failed with HTTP {e.status}",
                retryable=retryable,
                retry_after_seconds=self._parse_retry_after(e.headers),
            )
        except (TimeoutError, aiohttp.ClientError) as e:
            return TaskFailure(
                stage=FailureStage.MODEL,
                code="model_transport_error",
                message=f"{type(e).__name__}: {e}",
                retryable=True,
            )
        except Exception as e:
            logging.exception(f"记录[{record_id}] 处理异常: {e}")
            return TaskFailure(
                stage=FailureStage.SYSTEM,
                code="unexpected_error",
                message=f"{type(e).__name__}: {e}",
                retryable=True,
            )

    @staticmethod
    def _parse_retry_after(headers: Any) -> float:
        if not headers:
            return 0.0
        raw = headers.get("Retry-After") or headers.get("retry-after")
        if raw is None:
            return 0.0
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            try:
                target = parsedate_to_datetime(str(raw))
                if target.tzinfo is None:
                    target = target.replace(tzinfo=timezone.utc)
                return max(
                    0.0,
                    (target - datetime.now(timezone.utc)).total_seconds(),
                )
            except (TypeError, ValueError, OverflowError):
                return 0.0

    def run(self) -> bool:
        """
        运行处理器（同步入口）

        为异步处理提供同步包装，适用于命令行直接调用。
        内部使用 asyncio.run() 启动事件循环。

        使用示例:
            processor = UniversalAIProcessor("config.yaml")
            processor.run()  # 阻塞直到处理完成
        """
        logging.info("启动 AI 数据处理引擎...")
        try:
            completed = asyncio.run(self.process_shard_async_continuous())
        except Exception:
            # 异常退出时保留进度文件，便于 GUI 侧基于 ts 判断超时/残留
            raise
        else:
            # 正常结束才清理进度文件
            self._cleanup_progress()
        logging.info("AI 数据处理引擎已停止")
        return completed
