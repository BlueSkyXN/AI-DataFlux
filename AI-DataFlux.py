# AI_DataFlux.py (Main Orchestrator)

# --- Standard Imports ---
import yaml
import aiohttp
import asyncio
import logging
import json
import re
import time
import os
import sys
import gc
import psutil
import argparse
import threading
from datetime import datetime
from typing import Dict, Any, Optional, List, Set, Tuple # Keep ABC if type hinting BaseTaskPool
# from abc import ABC, abstractmethod # ABC definition now in Flux_Data

# --- Import from Flux_Data.py ---
try:
    # Import the factory function and availability flags
    from Flux_Data import create_task_pool, MYSQL_AVAILABLE, EXCEL_ENABLED
    # Optionally import BaseTaskPool for type hinting
    from Flux_Data import BaseTaskPool
except ImportError as e:
     print(f"[致命错误] 无法从 Flux_Data.py 导入必要组件: {e}")
     print("请确保 Flux_Data.py 文件存在于同一目录且没有语法错误。")
     sys.exit(1)


# --- ErrorType Enum (Keep Here) ---
class ErrorType:
    """Defines categories for errors encountered during processing."""
    API_ERROR = "api_error"       # Error calling Flux-Api.py OR Flux-Api returned non-200 OR Timeout
    CONTENT_ERROR = "content_error" # AI response content parsing/validation failed
    SYSTEM_ERROR = "system_error"   # Internal errors (e.g., data reload failure, unexpected exceptions)

# --- BaseTaskPool ABC ---
# --- REMOVED - Definition is now in Flux_Data.py ---


# --- JsonValidator Class ---
class JsonValidator:
    """Validates specific field values within a parsed JSON object against configured rules."""
    def __init__(self):
        self.enabled = False
        self.field_rules: Dict[str, List[Any]] = {}
        logging.info("JsonValidator 初始化。")

    def configure(self, validation_config: Optional[Dict[str, Any]]):
        """Loads validation rules from the configuration."""
        if not validation_config:
            self.enabled = False
            logging.info("JSON 字段值验证配置未找到或为空，验证已禁用。")
            return

        self.enabled = validation_config.get("enabled", False)
        if not self.enabled:
            logging.info("JSON 字段值验证功能已在配置中禁用。")
            return

        rules = validation_config.get("field_rules", {})
        if not isinstance(rules, dict):
             logging.warning("validation.field_rules 配置格式错误，应为字典。验证已禁用。")
             self.enabled = False
             return

        self.field_rules = {} # Reset rules
        loaded_rule_count = 0
        for field, values in rules.items():
            if isinstance(values, list):
                self.field_rules[field] = values
                logging.debug(f"加载字段验证规则: '{field}' -> {len(values)} 个允许值: {values[:5]}...")
                loaded_rule_count += 1
            else:
                logging.warning(f"字段 '{field}' 的验证规则格式错误，应为列表，已忽略此规则。")

        if not self.field_rules:
             logging.warning("JSON 字段值验证已启用，但未加载任何有效的字段规则。")
             self.enabled = False
        else:
             logging.info(f"JSON 字段值验证功能已启用，共加载 {loaded_rule_count} 个字段的规则。")

    def validate(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validates the data against the loaded rules."""
        if not self.enabled or not self.field_rules:
            return True, []

        errors = []
        for field, allowed_values in self.field_rules.items():
            if field in data:
                value = data[field]
                if value not in allowed_values:
                    errors.append(f"字段 '{field}' 的值 '{value}' (类型: {type(value).__name__}) 不在允许的范围内: {allowed_values[:10]}...")

        is_valid = len(errors) == 0
        if not is_valid:
            logging.debug(f"JSON 字段值验证失败: {errors}")
        return is_valid, errors

# --- ShardedTaskManager Class ---
class ShardedTaskManager:
    """Manages the process of loading and tracking progress across data shards."""
    # Add type hint using the imported BaseTaskPool
    def __init__(
        self,
        task_pool: BaseTaskPool, # Use imported BaseTaskPool for type hint
        optimal_shard_size=10000,
        min_shard_size=1000,
        max_shard_size=50000
    ):
        if not isinstance(task_pool, BaseTaskPool): # Check instance against imported type
             raise TypeError("task_pool 必须是 BaseTaskPool 的实例。")
        self.task_pool = task_pool
        self.optimal_shard_size = optimal_shard_size
        self.min_shard_size = min_shard_size
        self.max_shard_size = max_shard_size

        self.current_shard_index = 0
        self.total_shards = 0
        self.shard_boundaries: List[Tuple[Any, Any]] = []

        self.total_estimated = 0
        self.total_processed_successfully = 0
        self.start_time = time.time()
        self.processing_metrics = {
            'avg_time_per_record': 0.0,
            'records_per_second': 0.0,
        }
        self.memory_tracker = {
            'last_check_time': time.time(),
            'check_interval': 60,
            'peak_memory_usage': 0.0,
            'current_memory_usage': 0.0,
        }
        try: # Handle case where psutil might fail
            self._process_info = psutil.Process()
        except psutil.Error as e:
             logging.warning(f"无法获取当前进程信息 (psutil): {e}. 内存监控将不可用。")
             self._process_info = None
        logging.info("ShardedTaskManager 初始化完成。")

    def calculate_optimal_shard_size(self, total_range: int) -> int:
        """Dynamically calculates a suitable shard size based on heuristics."""
        memory_based_limit = self.max_shard_size
        if self._process_info: # Check if psutil is available
            try:
                mem = psutil.virtual_memory()
                available_mb = mem.available / (1024 * 1024)
                record_size_mb_estimate = 5 / 1024 # ~5KB per record estimate
                memory_based_limit = int((available_mb * 0.3) / record_size_mb_estimate) if record_size_mb_estimate > 0 else self.max_shard_size
            except psutil.Error as e:
                 logging.warning(f"计算内存限制时出错: {e}")

        time_based_limit = self.max_shard_size
        if self.processing_metrics['records_per_second'] > 0:
            target_duration_seconds = 15 * 60 # Aim for 15 min shards
            time_based_limit = int(self.processing_metrics['records_per_second'] * target_duration_seconds)

        calculated_size = min(memory_based_limit, time_based_limit, self.optimal_shard_size)
        shard_size = max(self.min_shard_size, min(calculated_size, self.max_shard_size))

        logging.info(
            f"动态分片大小计算: 内存限制={memory_based_limit}, "
            f"时间限制={time_based_limit}. 最终选择: {shard_size}"
        )
        return shard_size

    def initialize(self) -> bool:
        """Initializes sharding."""
        logging.info("正在初始化分片任务管理器...")
        self.start_time = time.time()
        try:
            self.total_estimated = self.task_pool.get_total_task_count()
            if self.total_estimated <= 0:
                logging.info("数据源中没有需要处理的任务。")
                return False

            min_id, max_id = self.task_pool.get_id_boundaries()
            logging.info(f"获取到数据源 ID/索引范围: {min_id} - {max_id}。预估未处理任务数: {self.total_estimated}")

            # Simple range check for sharding logic
            try:
                 # Check if both are usable as numbers for range calculation
                 numeric_min = int(min_id)
                 numeric_max = int(max_id)
                 total_range = numeric_max - numeric_min + 1
            except (ValueError, TypeError):
                 logging.error("获取到的 ID 边界无法转换为整数进行范围计算，无法分片。")
                 # Maybe handle as single shard? For now, fail initialization.
                 return False

            if total_range <= 0 and self.total_estimated > 0:
                logging.warning("ID 范围无效但仍有任务，将尝试将所有任务作为一个分片处理。")
                self.total_shards = 1
                self.shard_boundaries = [(min_id, max_id)] # Use original boundaries
            elif total_range <= 0:
                logging.info("ID 范围无效且无任务。")
                return False
            else:
                shard_size = self.calculate_optimal_shard_size(total_range)
                self.total_shards = max(1, (total_range + shard_size - 1) // shard_size)
                logging.info(f"基于范围 {total_range} 和大小 {shard_size}，将数据分为 {self.total_shards} 个分片。")

                self.shard_boundaries = []
                current_start = numeric_min
                for i in range(self.total_shards):
                    # Use numeric boundaries for calculation, store original type if needed?
                    # Let's assume int boundaries are expected by task pools for now.
                    current_end = min(current_start + shard_size - 1, numeric_max)
                    self.shard_boundaries.append((current_start, current_end))
                    current_start = current_end + 1
                logging.debug(f"计算出的分片边界: {self.shard_boundaries}")

            self.current_shard_index = 0
            logging.info(f"分片任务管理器初始化成功。共 {self.total_shards} 个分片。")
            return True

        except Exception as e:
            logging.error(f"初始化分片任务管理器失败: {e}", exc_info=True)
            return False

    def load_next_shard(self) -> bool:
        """Loads the next available shard into the task pool."""
        if self.current_shard_index >= self.total_shards:
            logging.info("所有分片已处理完毕，没有更多分片可加载。")
            return False

        min_id, max_id = self.shard_boundaries[self.current_shard_index]
        shard_num = self.current_shard_index + 1
        logging.info(f"--- Loading Shard {shard_num}/{self.total_shards} (Range: {min_id}-{max_id}) ---")

        try:
            loaded_count = self.task_pool.initialize_shard(shard_num, min_id, max_id)
            self.current_shard_index += 1

            if loaded_count == 0:
                logging.info(f"分片 {shard_num} 加载完成，但无任务。尝试加载下一个分片...")
                return self.load_next_shard()
            else:
                logging.info(f"分片 {shard_num} 加载成功，包含 {loaded_count} 个任务。")
                return True

        except Exception as e:
            logging.error(f"加载分片 {shard_num} 时发生错误: {e}", exc_info=True)
            self.current_shard_index += 1
            logging.warning(f"跳过加载失败的分片 {shard_num}，尝试加载下一个。")
            return self.load_next_shard()

    def update_processing_metrics(self, batch_success_count: int, batch_processing_time: float):
        """Updates average processing time and records per second."""
        if batch_processing_time <= 0 or batch_success_count <= 0: return

        current_time_per_record = batch_processing_time / batch_success_count
        current_records_per_second = 1.0 / current_time_per_record
        alpha = 0.1 # Smoothing factor

        if self.processing_metrics['avg_time_per_record'] == 0.0:
            self.processing_metrics['avg_time_per_record'] = current_time_per_record
            self.processing_metrics['records_per_second'] = current_records_per_second
        else:
            self.processing_metrics['avg_time_per_record'] = (
                alpha * current_time_per_record +
                (1 - alpha) * self.processing_metrics['avg_time_per_record']
            )
            self.processing_metrics['records_per_second'] = 1.0 / max(1e-9, self.processing_metrics['avg_time_per_record'])

    def monitor_memory_usage(self):
        """Checks current memory usage."""
        if not self._process_info: return # psutil failed
        current_time = time.time()
        if current_time - self.memory_tracker['last_check_time'] < self.memory_tracker['check_interval']:
            return

        try:
            current_mem_mb = self._process_info.memory_info().rss / (1024 * 1024)
            self.memory_tracker['current_memory_usage'] = current_mem_mb
            self.memory_tracker['peak_memory_usage'] = max(self.memory_tracker['peak_memory_usage'], current_mem_mb)
            self.memory_tracker['last_check_time'] = current_time
            logging.debug(f"内存监控: 当前={current_mem_mb:.1f}MB, 峰值={self.memory_tracker['peak_memory_usage']:.1f}MB")

            system_mem = psutil.virtual_memory()
            if system_mem.percent > 85.0 or current_mem_mb > 1536.0: # System > 85% or Process > 1.5GB
                logging.warning(
                    f"高内存使用: Process={current_mem_mb:.1f}MB, System={system_mem.percent}%. 触发 GC..."
                )
                gc.collect()
                current_mem_after_gc = self._process_info.memory_info().rss / (1024 * 1024)
                logging.info(f"GC 后内存: {current_mem_after_gc:.1f}MB")
                self.memory_tracker['current_memory_usage'] = current_mem_after_gc

        except psutil.Error as e:
             logging.warning(f"内存监控检查失败: {e}")

    def finalize(self):
        """Performs final cleanup and logging."""
        end_time = time.time()
        total_duration = end_time - self.start_time
        logging.info("=" * 50)
        logging.info("任务处理结束")
        logging.info(f"总耗时: {total_duration:.2f} 秒")
        logging.info(f"预估总任务数: {self.total_estimated}")
        logging.info(f"成功处理并更新的任务数: {self.total_processed_successfully}")
        if self.total_processed_successfully > 0 and total_duration > 0:
             logging.info(f"平均处理速率: {self.total_processed_successfully / total_duration:.2f} 条记录/秒")
        if self.memory_tracker['peak_memory_usage'] > 0:
             logging.info(f"峰值内存使用量: {self.memory_tracker['peak_memory_usage']:.1f} MB")
        logging.info("=" * 50)

        try:
            logging.info("正在关闭任务池资源...")
            self.task_pool.close()
            logging.info("任务池资源已关闭。")
        except Exception as e:
            logging.error(f"关闭任务池资源时发生错误: {e}", exc_info=True)


# --- Utility Functions ---
def load_config(config_path: str) -> Dict[str, Any]:
    """Loads the YAML configuration file."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict): raise ValueError("配置文件格式错误，顶层必须是字典。")
        logging.info(f"配置文件 '{config_path}' 加载成功。")
        return config
    except FileNotFoundError:
        logging.error(f"配置文件 '{config_path}' 不存在！")
        raise
    except yaml.YAMLError as e:
        raise ValueError(f"配置文件 YAML 格式错误: {e}") from e
    except Exception as e:
        logging.error(f"加载配置文件 '{config_path}' 时发生未知错误: {e}")
        raise

def init_logging(log_config: Optional[Dict[str, Any]]):
    """Initializes logging based on the configuration."""
    if log_config is None: log_config = {}
    level_str = log_config.get("level", "info").upper()
    level = getattr(logging, level_str, logging.INFO)
    # Use text format string if format is 'text' or not specified correctly
    log_format_str = log_config.get("format", "text")
    if log_format_str == "json":
        log_format = '{"time": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", "message": "%(message)s"}'
    else: # Default to text format
        log_format = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"

    date_format = log_config.get("date_format", "%Y-%m-%d %H:%M:%S")
    output_type = log_config.get("output", "console")
    log_handlers = []

    if output_type == "file":
        file_path = log_config.get("file_path", "./ai_dataflux.log")
        try:
            log_dir = os.path.dirname(file_path)
            if log_dir and not os.path.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(file_path, encoding='utf-8')
            log_handlers.append(file_handler)
            print(f"日志将输出到文件: {file_path}")
        except Exception as e:
            print(f"创建日志文件失败: {e}. 回退到控制台。", file=sys.stderr)
            output_type = "console"

    if output_type == "console" or not log_handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        log_handlers.append(console_handler)
        if output_type != "console": print("同时将日志输出到控制台。")

    # Configure root logger - set force=True to allow reconfiguration if run multiple times (e.g., in tests)
    logging.basicConfig(level=level, format=log_format, datefmt=date_format, handlers=log_handlers, force=True)

    # Set library levels
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    if MYSQL_AVAILABLE: logging.getLogger("mysql.connector").setLevel(logging.WARNING)

    logging.info(f"日志系统初始化完成。级别: {level_str}, 输出: {output_type}")


# --- Main Processor Class ---
class UniversalAIProcessor:
    """Orchestrates the entire AI data processing workflow."""
    def __init__(self, config_path: str):
        """Loads config, initializes components."""
        try:
            self.config = load_config(config_path)
        except Exception as e:
            raise ValueError(f"无法加载配置文件: {e}") from e

        global_cfg = self.config.get("global", {})
        init_logging(global_cfg.get("log")) # Init logging early

        self.flux_api_url = global_cfg.get("flux_api_url")
        if not self.flux_api_url:
            raise ValueError("配置文件 [global] 部分缺少 'flux_api_url'")
        if not self.flux_api_url.startswith(("http://", "https://")):
             logging.warning(f"Flux API URL '{self.flux_api_url}' 格式可能不正确。")
        if "/v1/chat/completions" not in self.flux_api_url:
             self.flux_api_url = self.flux_api_url.rstrip('/') + "/v1/chat/completions"
        logging.info(f"将使用的 Flux API 端点: {self.flux_api_url}")

        self.datasource_type = self.config.get("datasource", {}).get("type", "excel").lower()
        logging.info(f"数据源类型配置为: {self.datasource_type}")
        if self.datasource_type not in ["excel", "mysql"]:
             raise ValueError(f"不支持的数据源类型: {self.datasource_type}")

        concurrency_cfg = self.config.get("datasource", {}).get("concurrency", {})
        self.batch_size = concurrency_cfg.get("batch_size", 100)
        self.api_retry_times = concurrency_cfg.get("api_retry_times", 3)
        self.api_retry_delay = concurrency_cfg.get("api_retry_delay", 5)
        shard_size = concurrency_cfg.get("shard_size", 10000)
        min_shard_size = concurrency_cfg.get("min_shard_size", 1000)
        max_shard_size = concurrency_cfg.get("max_shard_size", 50000)
        logging.info(f"并发设置: 批次大小={self.batch_size}, API重试次数={self.api_retry_times}, API重试延迟={self.api_retry_delay}s")
        logging.info(f"分片设置: 建议大小={shard_size}, 最小={min_shard_size}, 最大={max_shard_size}")

        prompt_cfg = self.config.get("prompt", {})
        self.prompt_template = prompt_cfg.get("template", "")
        if not self.prompt_template: logging.warning("提示模板 'template' 为空。")
        if "{record_json}" not in self.prompt_template: logging.warning("提示模板不含 '{record_json}' 占位符。")

        self.required_fields = prompt_cfg.get("required_fields", [])
        self.use_json_schema = prompt_cfg.get("use_json_schema", False)
        self.ai_model_override = prompt_cfg.get("model", "auto")
        self.ai_temperature = prompt_cfg.get("temperature", 0.7)
        logging.info(f"提示配置: 必需字段={self.required_fields}, 使用Schema={self.use_json_schema}, 模型='{self.ai_model_override}', 温度={self.ai_temperature}")

        self.validator = JsonValidator()
        self.validator.configure(self.config.get("validation"))

        self.columns_to_extract = self.config.get("columns_to_extract", [])
        self.columns_to_write = self.config.get("columns_to_write", {})
        if not self.columns_to_extract: raise ValueError("缺少 columns_to_extract 配置")
        if not self.columns_to_write: raise ValueError("缺少 columns_to_write 配置")
        logging.info(f"输入列: {self.columns_to_extract}")
        logging.info(f"输出映射: {self.columns_to_write}")

        # --- Create Data Source Task Pool using the factory from Flux_Data ---
        try:
            # Pass necessary parts of the config to the factory
            self.task_pool: BaseTaskPool = create_task_pool(
                self.config,
                self.columns_to_extract,
                self.columns_to_write
            )
        except (ImportError, ValueError, RuntimeError, FileNotFoundError, IOError) as e:
            logging.critical(f"创建任务池失败: {e}", exc_info=True)
            raise RuntimeError(f"无法初始化数据源任务池: {e}") from e

        # --- Create Sharded Task Manager ---
        try:
            self.task_manager = ShardedTaskManager(
                task_pool=self.task_pool,
                optimal_shard_size=shard_size,
                min_shard_size=min_shard_size,
                max_shard_size=max_shard_size
            )
        except Exception as e:
            logging.critical(f"创建分片任务管理器失败: {e}", exc_info=True)
            raise RuntimeError(f"无法初始化分片任务管理器: {e}") from e

        logging.info("UniversalAIProcessor 初始化完成。")

    # Remove _create_task_pool method, use factory directly in __init__
    # def _create_task_pool(self) -> BaseTaskPool: ... REMOVED ...

    def create_prompt(self, record_data: Dict[str, Any]) -> str:
        """Creates the full prompt string."""
        if not self.prompt_template: return ""
        try:
            filtered_data = {k: v for k, v in record_data.items() if v is not None}
            # Using separators=(',', ':') for compact JSON in prompt
            record_json_str = json.dumps(filtered_data, ensure_ascii=False, separators=(',', ':'))
        except (TypeError, ValueError) as e:
            logging.error(f"记录数据无法序列化为JSON: {e}. 数据: {record_data}")
            return self.prompt_template.replace("{record_json}", '{"error": "无法序列化数据"}') if "{record_json}" in self.prompt_template else ""
        return self.prompt_template.replace("{record_json}", record_json_str)

    def extract_json_from_response(self, content: Optional[str]) -> Dict[str, Any]:
        """Attempts to extract a valid JSON object from the AI's response string."""
        if not content: return {"_error": "empty_ai_response", "_error_type": ErrorType.CONTENT_ERROR}
        content = content.strip()
        parse_content = content
        parsed_json = None

        code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL | re.IGNORECASE)
        if code_block_match:
            parse_content = code_block_match.group(1).strip()
            logging.debug("检测到 Markdown JSON 代码块。")
        elif not (content.startswith('{') and content.endswith('}')):
             logging.debug("响应非严格 JSON 对象格式，将尝试正则提取。")

        def check_required(data_dict: Dict[str, Any], required: List[str]) -> bool:
             missing = [k for k in required if k not in data_dict]
             if missing: logging.warning(f"解析的 JSON 缺少必需字段: {missing}"); return False
             return True

        try: # Attempt direct parse
            parse_content_cleaned = re.sub(r",\s*([}\]])", r"\1", parse_content)
            data = json.loads(parse_content_cleaned)
            if isinstance(data, dict):
                if not self.required_fields or check_required(data, self.required_fields):
                    is_valid, errors = self.validator.validate(data)
                    if is_valid:
                        parsed_json = data
                        logging.debug("直接解析并验证 JSON 成功。")
                    else: return {"_error": "invalid_field_values", "_error_type": ErrorType.CONTENT_ERROR, "_validation_errors": errors}
            else: logging.warning(f"解析内容非字典类型: {type(data).__name__}。尝试正则。")
        except json.JSONDecodeError:
            logging.debug("直接解析 JSON 失败。尝试正则。")

        if parsed_json is None: # Attempt regex parse
             logging.debug("尝试正则提取 JSON 对象...")
             pattern = r'(\{.*?\})'
             matches = re.finditer(pattern, content, re.DOTALL)
             for match in matches:
                 match_str = match.group(1)
                 try:
                     match_str_cleaned = re.sub(r",\s*([}\]])", r"\1", match_str.strip())
                     candidate = json.loads(match_str_cleaned)
                     if isinstance(candidate, dict):
                         if not self.required_fields or check_required(candidate, self.required_fields):
                             is_valid, errors = self.validator.validate(candidate)
                             if is_valid:
                                 parsed_json = candidate; logging.debug("正则提取并验证 JSON 成功。"); break
                             else: logging.debug(f"正则提取 JSON 验证失败: {errors}")
                 except json.JSONDecodeError: continue

        if parsed_json is not None: return parsed_json
        else:
            logging.error(f"无法提取有效 JSON。响应前 500 字符: {content[:500]}...")
            return {"_error": "invalid_or_missing_json", "_error_type": ErrorType.CONTENT_ERROR}

    def build_json_schema(self) -> Optional[Dict[str, Any]]:
        """Builds a JSON Schema dictionary, if enabled."""
        if not self.use_json_schema: return None
        fields_for_schema = self.required_fields
        if not fields_for_schema:
             logging.warning("JSON Schema 已启用但 required_fields 为空，无法构建。")
             return None

        schema: Dict[str, Any] = {"type": "object", "properties": {}, "required": list(fields_for_schema)}
        validation_rules = self.validator.field_rules if self.validator.enabled else {}

        for field in fields_for_schema:
            prop_def = {}
            if field in validation_rules and validation_rules[field]:
                allowed_values = validation_rules[field]; first_val = allowed_values[0]
                if isinstance(first_val, bool): prop_def["type"] = "boolean"
                elif isinstance(first_val, int): prop_def["type"] = "integer"
                elif isinstance(first_val, float): prop_def["type"] = "number"
                else: prop_def["type"] = "string"
                prop_def["enum"] = allowed_values
            else: prop_def["type"] = "string"
            schema["properties"][field] = prop_def
        logging.debug(f"构建的 JSON Schema: {json.dumps(schema)}")
        return schema

    async def call_ai_api_async(self, session: aiohttp.ClientSession, prompt: str) -> str:
        """Calls the configured Flux API endpoint."""
        headers = {"Content-Type": "application/json"}
        payload: Dict[str, Any] = {
            "model": self.ai_model_override, "messages": [{"role": "user", "content": prompt}],
            "temperature": self.ai_temperature, "stream": False }
        json_schema = self.build_json_schema()
        if json_schema: payload["response_format"] = {"type": "json_object"}

        request_timeout = aiohttp.ClientTimeout(connect=20, total=600)
        logging.debug(f"向 Flux API ({self.flux_api_url}) 发送请求...")
        start_time = time.time()

        try:
            async with session.post(self.flux_api_url, headers=headers, json=payload, timeout=request_timeout) as resp:
                resp_status = resp.status; response_text = await resp.text()
                elapsed = time.time() - start_time; logging.debug(f"Flux API 响应状态: {resp_status} in {elapsed:.2f}s")

                if resp_status == 200:
                    try:
                        data = json.loads(response_text); choices = data.get("choices")
                        if choices and isinstance(choices, list) and choices[0]:
                            message = choices[0].get("message"); content = message.get("content") if message else None
                            if content is not None and isinstance(content, str):
                                return content # Success
                        raise ValueError("响应结构无效或缺少内容") # Trigger exception if structure bad
                    except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as e:
                         logging.error(f"Flux API 返回 200 OK 但响应无效: {e}. 响应: {response_text[:500]}...")
                         raise aiohttp.ClientResponseError(resp.request_info, resp.history, status=resp_status, message=f"Invalid 200 OK response: {e}", headers=resp.headers) from e
                else: # Non-200 status
                    logging.warning(f"Flux API 调用失败: HTTP {resp_status}. 响应: {response_text[:500]}...")
                    raise aiohttp.ClientResponseError(resp.request_info, resp.history, status=resp_status, message=f"Flux API Error: {response_text}", headers=resp.headers)

        except asyncio.TimeoutError as e:
            elapsed = time.time() - start_time
            logging.error(f"调用 Flux API 超时 (>{request_timeout.total}s)。")
            raise TimeoutError(f"Flux API call timed out after {elapsed:.2f}s") from e
        except aiohttp.ClientError as e:
            logging.error(f"调用 Flux API 时网络/客户端错误: {e}")
            raise

    async def process_one_record_async(self, session: aiohttp.ClientSession, record_id: Any, row_data: Dict[str, Any]) -> Dict[str, Any]:
        """Processes a single record with API calls and retries."""
        log_label = f"记录[{record_id}]"
        logging.debug(f"{log_label}: 开始处理...")
        prompt = self.create_prompt(row_data)
        if not prompt: return {"_error": "prompt_generation_failed", "_error_type": ErrorType.SYSTEM_ERROR}

        last_api_error: Optional[Exception] = None
        for attempt in range(self.api_retry_times):
            logging.debug(f"{log_label}: 尝试调用 Flux API (第 {attempt + 1}/{self.api_retry_times} 次)...")
            try:
                ai_response_content = await self.call_ai_api_async(session, prompt)
                logging.debug(f"{log_label}: Flux API 调用成功，提取/验证 JSON...")
                parsed_result = self.extract_json_from_response(ai_response_content)

                if parsed_result.get("_error_type") == ErrorType.CONTENT_ERROR:
                    logging.warning(f"{log_label}: 内容处理失败 ({parsed_result.get('_error')})。")
                    parsed_result["_original_response_excerpt"] = (ai_response_content[:150] + "...") if ai_response_content and len(ai_response_content) > 150 else ai_response_content
                    return parsed_result
                else:
                    logging.info(f"{log_label}: 处理成功完成。")
                    return parsed_result

            except (aiohttp.ClientResponseError, TimeoutError, aiohttp.ClientError) as e:
                 last_api_error = e; error_type_name = type(e).__name__
                 status_code = getattr(e, 'status', None)
                 log_message = f"{log_label}: API 调用失败 ({attempt + 1}/{self.api_retry_times}). {error_type_name}, Status: {status_code or 'N/A'}, Err: {str(e)[:200]}..."
                 should_retry = True; wait_time = self.api_retry_delay * (attempt + 1)

                 if isinstance(e, asyncio.TimeoutError): log_message += " (超时)"
                 elif status_code:
                      if 400 <= status_code < 500 and status_code not in [408, 429]: should_retry = False
                      elif status_code == 429: wait_time *= 2
                 elif isinstance(e, aiohttp.ClientConnectionError): log_message += " (连接错误)"

                 logging.warning(log_message)
                 if not should_retry or (attempt + 1) >= self.api_retry_times: break
                 logging.info(f"{log_label}: 等待 {wait_time:.1f} 秒后重试...")
                 await asyncio.sleep(wait_time)

            except Exception as e:
                logging.exception(f"{log_label}: 处理中意外错误: {e}")
                return {"_error": f"unexpected_error: {str(e)}", "_error_type": ErrorType.SYSTEM_ERROR}

        logging.error(f"{log_label}: 经 {self.api_retry_times} 次尝试后 API 调用仍失败。")
        return {"_error": "flux_api_failed_after_retries", "_error_type": ErrorType.API_ERROR,
                "_last_api_error": f"{type(last_api_error).__name__}: {str(last_api_error)}" if last_api_error else "N/A"}

    async def process_shard_async(self):
        """Manages the asynchronous processing of all tasks, shard by shard."""
        if not self.task_manager.initialize():
            logging.info("任务管理器初始化失败或无任务。")
            return

        async with aiohttp.ClientSession() as session:
            current_shard_num = 0
            while True:
                if not self.task_manager.load_next_shard():
                    logging.info("所有分片已加载完毕。")
                    break
                current_shard_num += 1
                processed_in_shard_successfully = 0

                while self.task_pool.has_tasks():
                    batch_size = min(self.batch_size, self.task_pool.get_remaining_count())
                    tasks_batch = self.task_pool.get_task_batch(batch_size)
                    if not tasks_batch: break

                    logging.info(f"分片 {current_shard_num}: 获取到 {len(tasks_batch)} 个任务并发处理...")
                    batch_start_time = time.time()
                    coros = [self.process_one_record_async(session, rid, data) for rid, data in tasks_batch]
                    results = await asyncio.gather(*coros, return_exceptions=True)

                    batch_results_to_update: Dict[Any, Dict[str, Any]] = {}
                    tasks_to_retry: List[Tuple[Any, Dict[str, Any]]] = []
                    retried_count = 0; failed_count = 0; success_batch_count = 0

                    for i, result in enumerate(results):
                        record_id, _ = tasks_batch[i]
                        if isinstance(result, Exception):
                            logging.error(f"记录[{record_id}] 异常: {result}", exc_info=result)
                            reloaded = self.task_pool.reload_task_data(record_id)
                            if reloaded: tasks_to_retry.append((record_id, reloaded)); retried_count += 1
                            else: batch_results_to_update[record_id] = {"_error": f"system_exception_reload_failed: {result}", "_error_type": ErrorType.SYSTEM_ERROR}; failed_count += 1
                        elif isinstance(result, dict):
                            etype = result.get("_error_type")
                            if etype in [ErrorType.SYSTEM_ERROR, ErrorType.API_ERROR]:
                                logging.log(logging.ERROR if etype == ErrorType.API_ERROR else logging.WARNING, f"记录[{record_id}] 失败 ({etype}): {result.get('_error')}. 重试。")
                                reloaded = self.task_pool.reload_task_data(record_id)
                                if reloaded: tasks_to_retry.append((record_id, reloaded)); retried_count += 1
                                else: batch_results_to_update[record_id] = result; failed_count += 1
                            elif etype == ErrorType.CONTENT_ERROR:
                                logging.warning(f"记录[{record_id}] 内容错误: {result.get('_error')}")
                                batch_results_to_update[record_id] = result; failed_count += 1
                            else: # Success
                                batch_results_to_update[record_id] = result; success_batch_count += 1
                        else: # Unknown result
                             batch_results_to_update[record_id] = {"_error": "unknown_result_type", "_error_type": ErrorType.SYSTEM_ERROR}; failed_count += 1

                    if tasks_to_retry:
                        logging.info(f"分片 {current_shard_num}: 将 {len(tasks_to_retry)} 个任务放回队列重试...")
                        for rid, rdata in reversed(tasks_to_retry): self.task_pool.add_task_to_front(rid, rdata)

                    if batch_results_to_update:
                        logging.info(f"分片 {current_shard_num}: 更新 {len(batch_results_to_update)} 条记录结果 ({success_batch_count} 成功)...")
                        try:
                            self.task_pool.update_task_results(batch_results_to_update)
                            processed_in_shard_successfully += success_batch_count
                            self.task_manager.total_processed_successfully += success_batch_count
                        except Exception as update_e: logging.error(f"分片 {current_shard_num}: 更新结果时错误: {update_e}", exc_info=True)

                    batch_time = time.time() - batch_start_time
                    self.task_manager.update_processing_metrics(success_batch_count, batch_time)
                    remaining = self.task_pool.get_remaining_count()
                    logging.info(f"分片 {current_shard_num} 批处理: 成={success_batch_count}, 错={failed_count}, 重={retried_count}. "
                                 f"耗时={batch_time:.2f}s ({self.task_manager.processing_metrics['records_per_second']:.2f} rec/s). 剩={remaining}. "
                                 f"总={self.task_manager.total_processed_successfully}/{self.task_manager.total_estimated}.")
                    self.task_manager.monitor_memory_usage()
                    # await asyncio.sleep(0.01) # Optional yield

                logging.info(f"--- 分片 {current_shard_num} 处理完毕 ---")

        logging.info("所有分片处理循环结束。")
        self.task_manager.finalize()

    def process_tasks(self):
        """Entry point to start the asynchronous processing workflow."""
        proc_start_time = time.time()
        logging.info("开始执行 AI 数据处理任务...")
        try:
            asyncio.run(self.process_shard_async())
        except KeyboardInterrupt:
            logging.warning("检测到用户中断。尝试优雅退出...")
            if hasattr(self, 'task_manager') and self.task_manager: self.task_manager.finalize()
            logging.info("程序已中断。")
        except Exception as e:
            logging.critical(f"执行过程中发生未处理异常: {e}", exc_info=True)
            if hasattr(self, 'task_manager') and self.task_manager: self.task_manager.finalize()
            raise
        finally:
            logging.info(f"任务执行结束。总耗时: {time.time() - proc_start_time:.2f} 秒。")


# --- Command Line Interface ---
def validate_config_file(config_path: str) -> bool:
    """Performs basic validation of the configuration file."""
    print(f"正在验证配置文件: {config_path}")
    if not os.path.exists(config_path): print(f"[错误] 配置文件路径不存在: {config_path}"); return False
    try:
        with open(config_path, "r", encoding="utf-8") as f: config = yaml.safe_load(f)
        if not isinstance(config, dict): print("[错误] 配置文件格式错误。"); return False
        errors = []
        if "global" not in config or "flux_api_url" not in config["global"]: errors.append("缺少 global.flux_api_url")
        ds_type = config.get("datasource", {}).get("type", "excel").lower()
        if ds_type not in ["mysql", "excel"]: errors.append("不支持的 datasource.type")
        if ds_type == "excel":
            if not EXCEL_ENABLED: errors.append("Excel 已配置但库未加载")
            if "excel" not in config or "input_path" not in config.get("excel", {}): errors.append("缺少 excel.input_path")
        if ds_type == "mysql":
            if not MYSQL_AVAILABLE: errors.append("MySQL 已配置但库未加载")
            if "mysql" not in config: errors.append("缺少 mysql 配置段")
            else:
                req = ["host", "user", "password", "database", "table_name"]
                missing = [k for k in req if k not in config["mysql"] or not config["mysql"][k]]
                if missing: errors.append(f"mysql 配置缺少: {missing}")
        if "columns_to_extract" not in config or not config.get("columns_to_extract"): errors.append("缺少 columns_to_extract")
        if "columns_to_write" not in config or not config.get("columns_to_write"): errors.append("缺少 columns_to_write")
        if "prompt" not in config or "template" not in config.get("prompt", {}): errors.append("缺少 prompt.template")

        if errors: print("[错误] 配置验证失败:"); [print(f"  - {e}") for e in errors]; return False
        else: print("配置文件验证通过。"); return True
    except Exception as e: print(f"[错误] 验证配置文件时出错: {e}"); return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='AI DataFlux - Batch Processor')
    parser.add_argument('--config', '-c', default='./config.yaml', help='Config file path')
    args = parser.parse_args()

    print("=" * 60 + f"\n{' AI DataFlux Initializing '.center(60, '=')}\n" + "=" * 60)
    print(f"使用配置文件: {os.path.abspath(args.config)}")
    if not validate_config_file(args.config): sys.exit(1)

    try:
        print("正在初始化 AI 处理器..."); processor = UniversalAIProcessor(args.config)
        print("开始处理任务..."); processor.process_tasks()
        print("=" * 60 + f"\n{' AI DataFlux Finished '.center(60, '=')}\n" + "=" * 60)
        sys.exit(0)
    except (ValueError, ImportError, RuntimeError, FileNotFoundError, IOError) as e:
         logging.critical(f"初始化错误: {e}", exc_info=True); print(f"[致命错误] 初始化失败: {e}"); sys.exit(1)
    except KeyboardInterrupt: print("\n[信息] 程序被中断 (main)。"); sys.exit(130)
    except Exception as e:
        logging.critical(f"运行时致命错误: {e}", exc_info=True); print(f"[致命错误] 运行时错误: {e}"); sys.exit(1)

if __name__ == "__main__":
    main()