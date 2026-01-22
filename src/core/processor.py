"""
通用 AI 数据处理器

核心处理引擎，负责任务调度、API 调用、结果处理等功能。
"""

import asyncio
import json
import logging
import re
import threading
import time
from typing import Any

import aiohttp

from ..config.settings import load_config, init_logging
from ..models.errors import ErrorType
from ..models.task import TaskMetadata
from ..data import create_task_pool, BaseTaskPool
from .scheduler import ShardedTaskManager
from .validator import JsonValidator


class UniversalAIProcessor:
    """
    通用 AI 数据处理器

    编排整个 AI 数据处理工作流：
    1. 加载配置和初始化组件
    2. 分片加载任务
    3. 并发调用 AI API
    4. 处理结果并写回数据源

    采用连续任务流模式，比传统批处理更高效。
    """

    def __init__(self, config_path: str):
        """
        初始化处理器

        Args:
            config_path: 配置文件路径

        Raises:
            ValueError: 配置无效
            RuntimeError: 组件初始化失败
        """
        # 加载配置
        try:
            self.config = load_config(config_path)
        except Exception as e:
            raise ValueError(f"无法加载配置文件: {e}") from e

        # 初始化日志
        global_cfg = self.config.get("global", {})
        init_logging(global_cfg.get("log"))

        # API 端点配置
        self.flux_api_url = global_cfg.get("flux_api_url")
        if not self.flux_api_url:
            raise ValueError("配置文件 [global] 部分缺少 'flux_api_url'")

        if not self.flux_api_url.startswith(("http://", "https://")):
            logging.warning(f"Flux API URL '{self.flux_api_url}' 格式可能不正确")

        if "/v1/chat/completions" not in self.flux_api_url:
            self.flux_api_url = self.flux_api_url.rstrip("/") + "/v1/chat/completions"

        logging.info(f"将使用的 Flux API 端点: {self.flux_api_url}")

        # 数据源配置
        datasource_cfg = self.config.get("datasource", {})
        self.datasource_type = datasource_cfg.get("type", "excel").lower()

        if self.datasource_type not in ["excel", "mysql"]:
            raise ValueError(f"不支持的数据源类型: {self.datasource_type}")

        logging.info(f"数据源类型: {self.datasource_type}")

        # 并发配置
        concurrency_cfg = datasource_cfg.get("concurrency", {})
        self.batch_size = concurrency_cfg.get("batch_size", 100)
        self.api_pause_duration = float(concurrency_cfg.get("api_pause_duration", 2.0))
        self.api_error_trigger_window = float(
            concurrency_cfg.get("api_error_trigger_window", 2.0)
        )
        self.last_api_pause_end_time = 0.0
        self.max_connections = concurrency_cfg.get("max_connections", 1000)
        self.max_connections_per_host = concurrency_cfg.get(
            "max_connections_per_host", 0
        )

        logging.info(
            f"并发设置: 批次大小={self.batch_size}, "
            f"API 暂停={self.api_pause_duration}s (窗口={self.api_error_trigger_window}s)"
        )

        # 重试限制
        retry_limits_cfg = concurrency_cfg.get("retry_limits", {})
        self.max_retry_counts = {
            ErrorType.API: retry_limits_cfg.get("api_error", 3),
            ErrorType.CONTENT: retry_limits_cfg.get("content_error", 1),
            ErrorType.SYSTEM: retry_limits_cfg.get("system_error", 2),
        }

        logging.info(
            f"重试限制: API={self.max_retry_counts[ErrorType.API]}, "
            f"内容={self.max_retry_counts[ErrorType.CONTENT]}, "
            f"系统={self.max_retry_counts[ErrorType.SYSTEM]}"
        )

        # 分片配置
        shard_size = concurrency_cfg.get("shard_size", 10000)
        min_shard_size = concurrency_cfg.get("min_shard_size", 1000)
        max_shard_size = concurrency_cfg.get("max_shard_size", 50000)

        # 提示词配置
        prompt_cfg = self.config.get("prompt", {})
        self.prompt_template = prompt_cfg.get("template", "")
        self.system_prompt = prompt_cfg.get("system_prompt", "")
        self.required_fields = prompt_cfg.get("required_fields", [])
        self.use_json_schema = prompt_cfg.get("use_json_schema", False)
        self.ai_model_override = prompt_cfg.get("model", "auto")
        self.ai_temperature = prompt_cfg.get("temperature", 0.7)

        if not self.prompt_template:
            logging.warning("提示模板 'template' 为空")

        if "{record_json}" not in self.prompt_template:
            logging.warning("提示模板不含 '{record_json}' 占位符")

        # 验证器
        self.validator = JsonValidator()
        self.validator.configure(self.config.get("validation"))

        # 列配置
        self.columns_to_extract = self.config.get("columns_to_extract", [])
        self.columns_to_write = self.config.get("columns_to_write", {})

        if not self.columns_to_extract:
            raise ValueError("缺少 columns_to_extract 配置")
        if not self.columns_to_write:
            raise ValueError("缺少 columns_to_write 配置")

        logging.info(f"输入列: {self.columns_to_extract}")
        logging.info(f"输出映射: {self.columns_to_write}")

        # 初始化任务池
        try:
            self.task_pool: BaseTaskPool = create_task_pool(
                self.config, self.columns_to_extract, self.columns_to_write
            )
        except Exception as e:
            raise RuntimeError(f"无法初始化数据源任务池: {e}") from e

        # 初始化分片管理器
        try:
            self.task_manager = ShardedTaskManager(
                self.task_pool,
                shard_size,
                min_shard_size,
                max_shard_size,
                self.max_retry_counts,
            )
        except Exception as e:
            raise RuntimeError(f"无法初始化分片任务管理器: {e}") from e

        # 任务状态追踪
        self.tasks_in_progress: set[Any] = set()
        self.tasks_progress_lock = threading.Lock()

        # 任务元数据管理
        self.task_metadata: dict[Any, TaskMetadata] = {}
        self.metadata_lock = threading.Lock()

        logging.info("UniversalAIProcessor 初始化完成")

    # ==================== 任务状态管理 ====================

    def mark_task_in_progress(self, record_id: Any) -> bool:
        """标记任务为处理中"""
        with self.tasks_progress_lock:
            if record_id in self.tasks_in_progress:
                return False
            self.tasks_in_progress.add(record_id)
            return True

    def mark_task_completed(self, record_id: Any) -> None:
        """标记任务处理完成"""
        with self.tasks_progress_lock:
            self.tasks_in_progress.discard(record_id)

    def is_task_in_progress(self, record_id: Any) -> bool:
        """检查任务是否处于处理中"""
        with self.tasks_progress_lock:
            return record_id in self.tasks_in_progress

    def get_task_metadata(self, task_id: Any) -> TaskMetadata:
        """获取或创建任务元数据"""
        with self.metadata_lock:
            if task_id not in self.task_metadata:
                self.task_metadata[task_id] = TaskMetadata(task_id)
            return self.task_metadata[task_id]

    def remove_task_metadata(self, task_id: Any) -> None:
        """移除任务元数据"""
        with self.metadata_lock:
            self.task_metadata.pop(task_id, None)

    def cleanup_old_metadata(self, max_age_hours: int = 24) -> None:
        """清理过期的任务元数据"""
        cutoff_time = time.time() - (max_age_hours * 3600)

        with self.metadata_lock:
            to_remove = [
                task_id
                for task_id, meta in self.task_metadata.items()
                if meta.created_at < cutoff_time
            ]

            for task_id in to_remove:
                del self.task_metadata[task_id]

            if to_remove:
                logging.info(f"[元数据清理] 清理了 {len(to_remove)} 个过期的任务元数据")

    # ==================== 提示词处理 ====================

    def create_prompt(self, record_data: dict[str, Any]) -> str:
        """创建提示词"""
        if not self.prompt_template:
            return ""

        try:
            # 过滤内部字段
            filtered_data = {
                k: v
                for k, v in record_data.items()
                if v is not None and not k.startswith("_")
            }
            record_json_str = json.dumps(
                filtered_data, ensure_ascii=False, separators=(",", ":")
            )
        except (TypeError, ValueError) as e:
            logging.error(f"记录数据无法序列化为 JSON: {e}")
            return self.prompt_template.replace(
                "{record_json}", '{"error": "无法序列化数据"}'
            )

        return self.prompt_template.replace("{record_json}", record_json_str)

    def extract_json_from_response(self, content: str | None) -> dict[str, Any]:
        """从 AI 响应中提取 JSON"""
        if not content:
            return {"_error": "empty_ai_response", "_error_type": ErrorType.CONTENT}

        content = content.strip()
        parse_content = content

        # 尝试提取 Markdown JSON 代码块
        code_block_match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL | re.IGNORECASE
        )
        if code_block_match:
            parse_content = code_block_match.group(1).strip()

        def check_required(data_dict: dict[str, Any]) -> bool:
            missing = [k for k in self.required_fields if k not in data_dict]
            if missing:
                logging.warning(f"JSON 缺少必需字段: {missing}")
                return False
            return True

        # 尝试直接解析
        try:
            parse_content_cleaned = re.sub(r",\s*([}\]])", r"\1", parse_content)
            data = json.loads(parse_content_cleaned)

            if isinstance(data, dict):
                if not self.required_fields or check_required(data):
                    is_valid, errors = self.validator.validate(data)
                    if is_valid:
                        return data
                    else:
                        return {
                            "_error": "invalid_field_values",
                            "_error_type": ErrorType.CONTENT,
                            "_validation_errors": errors,
                        }
        except json.JSONDecodeError:
            pass

        # 尝试正则提取
        pattern = r"(\{.*?\})"
        for match in re.finditer(pattern, content, re.DOTALL):
            match_str = match.group(1)
            try:
                match_str_cleaned = re.sub(r",\s*([}\]])", r"\1", match_str.strip())
                candidate = json.loads(match_str_cleaned)

                if isinstance(candidate, dict):
                    if not self.required_fields or check_required(candidate):
                        is_valid, errors = self.validator.validate(candidate)
                        if is_valid:
                            return candidate
            except json.JSONDecodeError:
                continue

        logging.error(f"无法提取有效 JSON (必需字段: {self.required_fields})")
        return {"_error": "invalid_or_missing_json", "_error_type": ErrorType.CONTENT}

    def build_json_schema(self) -> dict[str, Any] | None:
        """构建 JSON Schema"""
        if not self.use_json_schema:
            return None

        if not self.required_fields:
            logging.warning("JSON Schema 已启用但 required_fields 为空")
            return None

        schema: dict[str, Any] = {
            "type": "object",
            "properties": {},
            "required": list(self.required_fields),
        }

        validation_rules = self.validator.field_rules if self.validator.enabled else {}

        for field in self.required_fields:
            prop_def: dict[str, Any] = {}
            allowed_values = validation_rules.get(field)

            if allowed_values:
                first_val = allowed_values[0]
                if isinstance(first_val, bool):
                    prop_def["type"] = "boolean"
                elif isinstance(first_val, int):
                    prop_def["type"] = "integer"
                elif isinstance(first_val, float):
                    prop_def["type"] = "number"
                else:
                    prop_def["type"] = "string"
                prop_def["enum"] = allowed_values
            else:
                prop_def["type"] = "string"

            schema["properties"][field] = prop_def

        return schema

    # ==================== API 调用 ====================

    async def call_ai_api_async(
        self, session: aiohttp.ClientSession, prompt: str
    ) -> str:
        """调用 AI API"""
        headers = {"Content-Type": "application/json"}

        # 构建消息
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        # 构建请求体
        payload: dict[str, Any] = {
            "model": self.ai_model_override,
            "messages": messages,
            "temperature": self.ai_temperature,
            "stream": False,
        }

        if self.use_json_schema:
            payload["response_format"] = {"type": "json_object"}

        request_timeout = aiohttp.ClientTimeout(connect=20, total=600)

        logging.debug(f"向 Flux API ({self.flux_api_url}) 发送请求...")
        start_time = time.time()

        try:
            async with session.post(
                self.flux_api_url,
                headers=headers,
                json=payload,
                timeout=request_timeout,
            ) as resp:
                resp_status = resp.status
                response_text = await resp.text()
                elapsed = time.time() - start_time

                logging.debug(f"Flux API 响应状态: {resp_status} in {elapsed:.2f}s")

                if resp_status == 200:
                    try:
                        data = json.loads(response_text)
                        choices = data.get("choices")

                        if choices and isinstance(choices, list) and choices[0]:
                            message = choices[0].get("message")
                            content = message.get("content") if message else None

                            if content is not None and isinstance(content, str):
                                return content

                        raise ValueError("响应结构无效或缺少内容")

                    except (json.JSONDecodeError, ValueError) as e:
                        logging.error(f"Flux API 返回 200 OK 但响应无效: {e}")
                        raise aiohttp.ClientResponseError(
                            resp.request_info,
                            resp.history,
                            status=resp_status,
                            message=f"Invalid 200 OK response: {e}",
                            headers=resp.headers,
                        ) from e
                else:
                    logging.warning(f"Flux API 调用失败: HTTP {resp_status}")
                    raise aiohttp.ClientResponseError(
                        resp.request_info,
                        resp.history,
                        status=resp_status,
                        message=f"Flux API Error: {response_text[:500]}",
                        headers=resp.headers,
                    )

        except asyncio.TimeoutError as e:
            logging.error(f"调用 Flux API 超时 (>{request_timeout.total}s)")
            raise TimeoutError(
                f"Flux API call timed out after {time.time() - start_time:.2f}s"
            ) from e

        except aiohttp.ClientError as e:
            logging.error(f"调用 Flux API 时网络/客户端错误: {e}")
            raise

    async def process_one_record_async(
        self, session: aiohttp.ClientSession, record_id: Any, row_data: dict[str, Any]
    ) -> dict[str, Any]:
        """处理单条记录"""
        log_label = f"记录[{record_id}]"
        logging.debug(f"{log_label}: 开始处理...")

        prompt = self.create_prompt(row_data)
        if not prompt:
            return {
                "_error": "prompt_generation_failed",
                "_error_type": ErrorType.SYSTEM,
            }

        try:
            # 调用 API
            ai_response_content = await self.call_ai_api_async(session, prompt)

            # 提取 JSON
            parsed_result = self.extract_json_from_response(ai_response_content)

            if parsed_result.get("_error_type") == ErrorType.CONTENT:
                logging.warning(
                    f"{log_label}: 内容处理失败 ({parsed_result.get('_error')})"
                )
                return parsed_result
            else:
                logging.info(f"{log_label}: 处理成功完成")
                return parsed_result

        except (aiohttp.ClientResponseError, TimeoutError, aiohttp.ClientError) as e:
            status_code = getattr(e, "status", None)
            logging.warning(
                f"{log_label}: API 调用失败. 类型: {type(e).__name__}, "
                f"状态: {status_code or 'N/A'}"
            )
            return {
                "_error": f"api_call_failed: {type(e).__name__}",
                "_error_type": ErrorType.API,
                "_details": str(e)[:200],
            }

        except Exception as e:
            logging.exception(f"{log_label}: 处理中意外错误: {e}")
            return {
                "_error": f"unexpected_error: {str(e)}",
                "_error_type": ErrorType.SYSTEM,
            }

    # ==================== 主处理流程 ====================

    async def process_shard_async_continuous(self) -> None:
        """连续任务流模式的异步处理"""
        if not self.task_manager.initialize():
            logging.info("无任务或初始化失败")
            return

        try:
            # 创建连接器
            connector = aiohttp.TCPConnector(
                limit=self.max_connections, limit_per_host=self.max_connections_per_host
            )

            async with aiohttp.ClientSession(connector=connector) as session:
                await self._process_loop(session)

        finally:
            self.task_manager.finalize()

    async def _process_loop(self, session: aiohttp.ClientSession) -> None:
        """主处理循环"""
        current_shard_num = 0
        active_tasks: set[asyncio.Task] = set()
        max_concurrent = self.batch_size
        task_id_map: dict[asyncio.Task, tuple[Any, dict[str, Any]]] = {}
        results_buffer: dict[Any, dict[str, Any]] = {}

        processed_in_shard = 0
        last_progress_time = time.time()
        progress_interval = 5.0

        while True:
            # 检查是否需要加载新分片
            if not self.task_pool.has_tasks() and len(active_tasks) == 0:
                if not self.task_manager.load_next_shard():
                    logging.info("所有分片加载完毕")
                    break
                current_shard_num += 1
                processed_in_shard = 0
                logging.info(
                    f"--- 开始处理分片 {current_shard_num}/{self.task_manager.total_shards} ---"
                )

            # 填充任务池
            space_available = max_concurrent - len(active_tasks)
            if space_available > 0 and self.task_pool.has_tasks():
                fetch_count = min(space_available, self.task_pool.get_remaining_count())
                tasks_batch = self.task_pool.get_task_batch(fetch_count)

                for record_id, data in tasks_batch:
                    if not self.is_task_in_progress(record_id):
                        self.mark_task_in_progress(record_id)
                        self.get_task_metadata(record_id)

                        task = asyncio.create_task(
                            self.process_one_record_async(session, record_id, data)
                        )
                        task_id_map[task] = (record_id, data)
                        active_tasks.add(task)
                    else:
                        self.task_pool.add_task_to_front(record_id, data)

            if not active_tasks:
                await asyncio.sleep(0.1)
                continue

            # 等待任务完成
            done, pending = await asyncio.wait(
                active_tasks, timeout=1.0, return_when=asyncio.FIRST_COMPLETED
            )

            active_tasks = pending
            api_error_in_batch = False
            tasks_to_retry: list[tuple[Any, dict[str, Any]]] = []

            # 处理完成的任务
            for completed_task in done:
                try:
                    result = completed_task.result()
                    record_id, _ = task_id_map.pop(completed_task)
                    self.mark_task_completed(record_id)

                    metadata = self.get_task_metadata(record_id)

                    if isinstance(result, dict):
                        error_type = result.get("_error_type")

                        if error_type:
                            # 处理错误
                            api_error_in_batch = api_error_in_batch or (
                                error_type == ErrorType.API
                            )
                            retry_result = self._handle_error_result(
                                record_id, result, metadata, error_type
                            )

                            if retry_result:
                                tasks_to_retry.append(retry_result)
                            else:
                                results_buffer[record_id] = result
                        else:
                            # 成功
                            results_buffer[record_id] = result
                            processed_in_shard += 1
                            self.task_manager.total_processed_successfully += 1
                            self.remove_task_metadata(record_id)
                    else:
                        results_buffer[record_id] = {
                            "_error": f"unknown_result_type: {type(result).__name__}",
                            "_error_type": ErrorType.SYSTEM,
                        }
                        self.remove_task_metadata(record_id)

                except Exception as e:
                    record_id, _ = task_id_map.pop(completed_task, (None, None))
                    if record_id is not None:
                        self.mark_task_completed(record_id)
                        logging.error(f"记录[{record_id}] 处理时发生异常: {e}")
                        self.remove_task_metadata(record_id)

            # API 错误暂停
            if api_error_in_batch:
                current_time = time.time()
                if (
                    current_time - self.last_api_pause_end_time
                    > self.api_error_trigger_window
                ):
                    logging.warning(
                        f"检测到 API 错误，暂停 {self.api_pause_duration}s..."
                    )
                    await asyncio.sleep(self.api_pause_duration)
                    self.last_api_pause_end_time = time.time()

            # 重试任务
            for record_id, clean_data in tasks_to_retry:
                self.task_pool.add_task_to_front(record_id, clean_data)

            # 批量更新结果
            if results_buffer:
                self.task_pool.update_task_results(results_buffer)
                results_buffer.clear()

            # 进度日志
            current_time = time.time()
            if current_time - last_progress_time >= progress_interval:
                self.task_manager.monitor_memory_usage()
                logging.info(
                    f"进度: {self.task_manager.progress_percent:.1f}% | "
                    f"成功: {self.task_manager.total_processed_successfully} | "
                    f"活动任务: {len(active_tasks)}"
                )
                last_progress_time = current_time

    def _handle_error_result(
        self,
        record_id: Any,
        result: dict[str, Any],
        metadata: TaskMetadata,
        error_type: ErrorType,
    ) -> tuple[Any, dict[str, Any]] | None:
        """处理错误结果，返回需要重试的任务或 None"""
        current_retries = metadata.get_retry_count(error_type)
        max_retries = self.max_retry_counts.get(error_type, 1)

        if current_retries < max_retries:
            metadata.increment_retry(error_type)
            metadata.add_error(error_type, result.get("_error", ""))
            self.task_manager.retried_tasks_count[error_type] += 1

            logging.warning(
                f"记录[{record_id}] {error_type.value}: {result.get('_error')}，"
                f"重试 {current_retries + 1}/{max_retries}"
            )

            clean_data = self.task_pool.reload_task_data(record_id)
            if clean_data:
                return (record_id, clean_data)
            else:
                logging.error(f"记录[{record_id}] 从数据源重新加载数据失败")
                self.task_manager.max_retries_exceeded_count += 1
                self.remove_task_metadata(record_id)
        else:
            logging.error(
                f"记录[{record_id}] {error_type.value}，"
                f"已达最大重试次数 ({max_retries})"
            )
            self.task_manager.max_retries_exceeded_count += 1
            self.remove_task_metadata(record_id)

        return None

    def run(self) -> None:
        """运行处理器（同步入口）"""
        logging.info("启动 AI 数据处理引擎...")
        asyncio.run(self.process_shard_async_continuous())
        logging.info("AI 数据处理引擎已停止")
