# AI_DataFlux.py (Main Orchestrator - Modified for Continuous Task Flow)

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
    CONTENT_ERROR = "content_error" # AI response content parsing/validation failed (Now triggers retry)
    SYSTEM_ERROR = "system_error"   # Internal errors (e.g., data reload failure, unexpected exceptions)


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
             self.enabled = False # Disable if no rules loaded
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
    def __init__(
        self,
        task_pool: BaseTaskPool,
        optimal_shard_size=10000,
        min_shard_size=1000,
        max_shard_size=50000
    ):
        if not isinstance(task_pool, BaseTaskPool):
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
        try:
            self._process_info = psutil.Process()
        except psutil.Error as e:
             logging.warning(f"无法获取当前进程信息 (psutil): {e}. 内存监控将不可用。")
             self._process_info = None
        logging.info("ShardedTaskManager 初始化完成。")

    def calculate_optimal_shard_size(self, total_range: int) -> int:
        """Dynamically calculates a suitable shard size based on heuristics."""
        memory_based_limit = self.max_shard_size
        if self._process_info:
            try:
                mem = psutil.virtual_memory()
                available_mb = mem.available / (1024 * 1024)
                record_size_mb_estimate = 5 / 1024
                memory_based_limit = int((available_mb * 0.3) / record_size_mb_estimate) if record_size_mb_estimate > 0 else self.max_shard_size
            except psutil.Error as e:
                 logging.warning(f"计算内存限制时出错: {e}")

        time_based_limit = self.max_shard_size
        if self.processing_metrics['records_per_second'] > 0:
            target_duration_seconds = 15 * 60
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

            try:
                 numeric_min = int(min_id); numeric_max = int(max_id)
                 total_range = numeric_max - numeric_min + 1
            except (ValueError, TypeError):
                 logging.error("ID 边界无法转换为整数，无法分片。")
                 return False

            if total_range <= 0 and self.total_estimated > 0:
                logging.warning("ID 范围无效但仍有任务，将作为一个分片处理。")
                self.total_shards = 1
                self.shard_boundaries = [(min_id, max_id)]
            elif total_range <= 0:
                logging.info("ID 范围无效且无任务。")
                return False
            else:
                shard_size = self.calculate_optimal_shard_size(total_range)
                self.total_shards = max(1, (total_range + shard_size - 1) // shard_size)
                logging.info(f"数据分为 {self.total_shards} 个分片 (大小约 {shard_size})。")
                self.shard_boundaries = []
                current_start = numeric_min
                for i in range(self.total_shards):
                    current_end = min(current_start + shard_size - 1, numeric_max)
                    self.shard_boundaries.append((current_start, current_end))
                    current_start = current_end + 1
                logging.debug(f"分片边界: {self.shard_boundaries}")

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
                logging.info(f"分片 {shard_num} 无任务。尝试加载下一个...")
                return self.load_next_shard()
            else:
                logging.info(f"分片 {shard_num} 加载成功 ({loaded_count} 个任务)。")
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
            self.processing_metrics['avg_time_per_record'] = (alpha * current_time_per_record + (1 - alpha) * self.processing_metrics['avg_time_per_record'])
            self.processing_metrics['records_per_second'] = 1.0 / max(1e-9, self.processing_metrics['avg_time_per_record'])

    def monitor_memory_usage(self):
        """Checks current memory usage."""
        if not self._process_info: return
        current_time = time.time()
        if current_time - self.memory_tracker['last_check_time'] < self.memory_tracker['check_interval']: return

        try:
            current_mem_mb = self._process_info.memory_info().rss / (1024 * 1024)
            self.memory_tracker['current_memory_usage'] = current_mem_mb
            self.memory_tracker['peak_memory_usage'] = max(self.memory_tracker['peak_memory_usage'], current_mem_mb)
            self.memory_tracker['last_check_time'] = current_time
            logging.debug(f"内存监控: 当前={current_mem_mb:.1f}MB, 峰值={self.memory_tracker['peak_memory_usage']:.1f}MB")

            system_mem = psutil.virtual_memory()
            if system_mem.percent > 85.0 or current_mem_mb > 40000.0:
                logging.warning(f"高内存使用: Process={current_mem_mb:.1f}MB, System={system_mem.percent}%. 触发 GC...")
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
        logging.info("=" * 50 + "\n" + " 任务处理结束 ".center(50, "=") + "\n" + "=" * 50)
        logging.info(f"总耗时: {total_duration:.2f} 秒")
        logging.info(f"预估总任务数: {self.total_estimated}")
        logging.info(f"成功处理并更新的任务数: {self.total_processed_successfully}")
        if self.total_processed_successfully > 0 and total_duration > 0:
             logging.info(f"平均处理速率: {self.total_processed_successfully / total_duration:.2f} 条记录/秒")
        if self.memory_tracker['peak_memory_usage'] > 0:
             logging.info(f"峰值内存使用量: {self.memory_tracker['peak_memory_usage']:.1f} MB")
        logging.info("=" * 50)
        try:
            logging.info("正在关闭任务池资源..."); self.task_pool.close()
            logging.info("任务池资源已关闭。")
        except Exception as e: logging.error(f"关闭任务池资源时发生错误: {e}", exc_info=True)


# --- Utility Functions ---
def load_config(config_path: str) -> Dict[str, Any]:
    """Loads the YAML configuration file."""
    try:
        with open(config_path, "r", encoding="utf-8") as f: config = yaml.safe_load(f)
        if not isinstance(config, dict): raise ValueError("配置文件格式错误")
        logging.info(f"配置文件 '{config_path}' 加载成功。")
        return config
    except FileNotFoundError: logging.error(f"配置文件 '{config_path}' 不存在！"); raise
    except yaml.YAMLError as e: raise ValueError(f"配置文件 YAML 格式错误: {e}") from e
    except Exception as e: logging.error(f"加载配置文件 '{config_path}' 异常: {e}"); raise

def init_logging(log_config: Optional[Dict[str, Any]]):
    """Initializes logging based on the configuration."""
    if log_config is None: log_config = {}
    level_str = log_config.get("level", "info").upper()
    level = getattr(logging, level_str, logging.INFO)
    log_format_str = log_config.get("format", "text")
    if log_format_str == "json":
        log_format = '{"time": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", "message": "%(message)s"}'
    else: log_format = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
    date_format = log_config.get("date_format", "%Y-%m-%d %H:%M:%S")
    output_type = log_config.get("output", "console")
    log_handlers = []
    if output_type == "file":
        file_path = log_config.get("file_path", "./ai_dataflux.log")
        try:
            log_dir = os.path.dirname(file_path);
            if log_dir and not os.path.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(file_path, encoding='utf-8'); log_handlers.append(file_handler)
            print(f"日志将输出到文件: {file_path}")
        except Exception as e: print(f"创建日志文件失败: {e}. 回退到控制台。", file=sys.stderr); output_type = "console"
    if output_type == "console" or not log_handlers:
        console_handler = logging.StreamHandler(sys.stdout); log_handlers.append(console_handler)
        if output_type != "console": print("同时将日志输出到控制台。")
    logging.basicConfig(level=level, format=log_format, datefmt=date_format, handlers=log_handlers, force=True)
    logging.getLogger("aiohttp").setLevel(logging.WARNING); logging.getLogger("asyncio").setLevel(logging.WARNING)
    if MYSQL_AVAILABLE: logging.getLogger("mysql.connector").setLevel(logging.WARNING)
    logging.info(f"日志系统初始化完成。级别: {level_str}, 输出: {output_type}")


# --- Main Processor Class ---
class UniversalAIProcessor:
    """Orchestrates the entire AI data processing workflow."""
    def __init__(self, config_path: str):
        """Loads config, initializes components."""
        try: self.config = load_config(config_path)
        except Exception as e: raise ValueError(f"无法加载配置文件: {e}") from e
        global_cfg = self.config.get("global", {})
        init_logging(global_cfg.get("log"))

        self.flux_api_url = global_cfg.get("flux_api_url")
        if not self.flux_api_url: raise ValueError("配置文件 [global] 部分缺少 'flux_api_url'")
        if not self.flux_api_url.startswith(("http://", "https://")): logging.warning(f"Flux API URL '{self.flux_api_url}' 格式可能不正确。")
        if "/v1/chat/completions" not in self.flux_api_url: self.flux_api_url = self.flux_api_url.rstrip('/') + "/v1/chat/completions"
        logging.info(f"将使用的 Flux API 端点: {self.flux_api_url}")

        self.datasource_type = self.config.get("datasource", {}).get("type", "excel").lower()
        logging.info(f"数据源类型配置为: {self.datasource_type}")
        if self.datasource_type not in ["excel", "mysql"]: raise ValueError(f"不支持的数据源类型: {self.datasource_type}")

        concurrency_cfg = self.config.get("datasource", {}).get("concurrency", {})
        self.batch_size = concurrency_cfg.get("batch_size", 100)
        # --- NEW: Pause mechanism config ---
        self.api_pause_duration = float(concurrency_cfg.get("api_pause_duration", 2.0)) # 暂停时长（秒）
        self.api_error_trigger_window = float(concurrency_cfg.get("api_error_trigger_window", 2.0)) # 错误触发窗口（秒）
        self.last_api_pause_end_time = 0.0 # 上次 API 暂停结束的时间戳
        # --- NEW: aiohttp connection limits ---
        self.max_connections = concurrency_cfg.get("max_connections", 1000)
        self.max_connections_per_host = concurrency_cfg.get("max_connections_per_host", 0)
        logging.info(f"并发设置: 批次大小={self.batch_size}, API错误暂停={self.api_pause_duration}s (触发窗口={self.api_error_trigger_window}s)")
        logging.info(f"aiohttp设置: 最大连接数={self.max_connections}, 每主机最大连接数={self.max_connections_per_host}")
        # --- REMOVED retry config ---
        # self.api_retry_times = concurrency_cfg.get("api_retry_times", 3)
        # self.api_retry_delay = concurrency_cfg.get("api_retry_delay", 5)
        shard_size = concurrency_cfg.get("shard_size", 10000); min_shard_size = concurrency_cfg.get("min_shard_size", 1000); max_shard_size = concurrency_cfg.get("max_shard_size", 50000)
        logging.info(f"并发设置: 批次大小={self.batch_size}, API错误暂停={self.api_pause_duration}s (触发窗口={self.api_error_trigger_window}s)")
        logging.info(f"分片设置: 建议大小={shard_size}, 最小={min_shard_size}, 最大={max_shard_size}")

        prompt_cfg = self.config.get("prompt", {}); self.prompt_template = prompt_cfg.get("template", "")
        if not self.prompt_template: logging.warning("提示模板 'template' 为空。")
        if "{record_json}" not in self.prompt_template: logging.warning("提示模板不含 '{record_json}' 占位符。")
        self.required_fields = prompt_cfg.get("required_fields", []); self.use_json_schema = prompt_cfg.get("use_json_schema", False)
        self.ai_model_override = prompt_cfg.get("model", "auto"); self.ai_temperature = prompt_cfg.get("temperature", 0.7)
        logging.info(f"提示配置: 必需字段={self.required_fields}, 使用Schema={self.use_json_schema}, 模型='{self.ai_model_override}', 温度={self.ai_temperature}")

        self.validator = JsonValidator(); self.validator.configure(self.config.get("validation"))
        self.columns_to_extract = self.config.get("columns_to_extract", []); self.columns_to_write = self.config.get("columns_to_write", {})
        if not self.columns_to_extract: raise ValueError("缺少 columns_to_extract 配置")
        if not self.columns_to_write: raise ValueError("缺少 columns_to_write 配置")
        logging.info(f"输入列: {self.columns_to_extract}"); logging.info(f"输出映射: {self.columns_to_write}")

        try:
            self.task_pool: BaseTaskPool = create_task_pool(self.config, self.columns_to_extract, self.columns_to_write)
        except (ImportError, ValueError, RuntimeError, FileNotFoundError, IOError) as e:
            raise RuntimeError(f"无法初始化数据源任务池: {e}") from e
        try:
            self.task_manager = ShardedTaskManager(self.task_pool, shard_size, min_shard_size, max_shard_size)
        except Exception as e: raise RuntimeError(f"无法初始化分片任务管理器: {e}") from e
        logging.info("UniversalAIProcessor 初始化完成。")

    def create_prompt(self, record_data: Dict[str, Any]) -> str:
        """Creates the full prompt string."""
        if not self.prompt_template: return ""
        try:
            filtered_data = {k: v for k, v in record_data.items() if v is not None}
            record_json_str = json.dumps(filtered_data, ensure_ascii=False, separators=(',', ':'))
        except (TypeError, ValueError) as e:
            logging.error(f"记录数据无法序列化为JSON: {e}. Data: {record_data}")
            return self.prompt_template.replace("{record_json}", '{"error": "无法序列化数据"}') if "{record_json}" in self.prompt_template else ""
        return self.prompt_template.replace("{record_json}", record_json_str)

    def extract_json_from_response(self, content: Optional[str]) -> Dict[str, Any]:
        """Attempts to extract a valid JSON object from the AI's response string."""
        if not content: return {"_error": "empty_ai_response", "_error_type": ErrorType.CONTENT_ERROR}
        content = content.strip(); parse_content = content; parsed_json = None
        code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL | re.IGNORECASE)
        if code_block_match: parse_content = code_block_match.group(1).strip(); logging.debug("检测到 Markdown JSON 代码块。")
        elif not (content.startswith('{') and content.endswith('}')): logging.debug("响应非严格 JSON 对象格式，尝试正则。")

        def check_required(data_dict: Dict[str, Any], required: List[str]) -> bool:
             missing = [k for k in required if k not in data_dict];
             if missing: logging.warning(f"JSON 缺少必需字段: {missing}"); return False
             return True

        try: # Attempt direct parse
            parse_content_cleaned = re.sub(r",\s*([}\]])", r"\1", parse_content)
            data = json.loads(parse_content_cleaned)
            if isinstance(data, dict):
                if not self.required_fields or check_required(data, self.required_fields):
                    is_valid, errors = self.validator.validate(data)
                    if is_valid: parsed_json = data; logging.debug("直接解析并验证 JSON 成功。")
                    else: return {"_error": "invalid_field_values", "_error_type": ErrorType.CONTENT_ERROR, "_validation_errors": errors}
            else: logging.warning(f"解析内容非字典类型: {type(data).__name__}。尝试正则。")
        except json.JSONDecodeError: logging.debug("直接解析 JSON 失败。尝试正则。")

        if parsed_json is None: # Attempt regex parse
             logging.debug("尝试正则提取 JSON 对象...")
             pattern = r'(\{.*?\})'; matches = re.finditer(pattern, content, re.DOTALL)
             for match in matches:
                 match_str = match.group(1)
                 try:
                     match_str_cleaned = re.sub(r",\s*([}\]])", r"\1", match_str.strip())
                     candidate = json.loads(match_str_cleaned)
                     if isinstance(candidate, dict):
                         if not self.required_fields or check_required(candidate, self.required_fields):
                             is_valid, errors = self.validator.validate(candidate)
                             if is_valid: parsed_json = candidate; logging.debug("正则提取并验证 JSON 成功。"); break
                             else: logging.debug(f"正则提取 JSON 验证失败: {errors}")
                 except json.JSONDecodeError: continue

        if parsed_json is not None: return parsed_json
        else:
            # --- MODIFIED: Reduced log verbosity on final failure ---
            logging.error(f"无法提取有效 JSON (必需字段: {self.required_fields})。")
            return {"_error": "invalid_or_missing_json", "_error_type": ErrorType.CONTENT_ERROR}

    def build_json_schema(self) -> Optional[Dict[str, Any]]:
        """Builds a JSON Schema dictionary, if enabled."""
        if not self.use_json_schema: return None
        fields_for_schema = self.required_fields
        if not fields_for_schema: logging.warning("JSON Schema 已启用但 required_fields 为空。"); return None
        schema: Dict[str, Any] = {"type": "object", "properties": {}, "required": list(fields_for_schema)}
        validation_rules = self.validator.field_rules if self.validator.enabled else {}
        for field in fields_for_schema:
            prop_def = {}; allowed_values = validation_rules.get(field)
            if allowed_values:
                first_val = allowed_values[0]
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
        
        # 构建消息数组，包含system提示词和user提示词
        messages = []
        
        # 添加system提示词(如果配置中存在)
        system_prompt = self.config.get("prompt", {}).get("system_prompt", "")
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # 添加用户提示词
        messages.append({"role": "user", "content": prompt})
        
        # 使用修改后的messages数组构建payload
        payload: Dict[str, Any] = {
            "model": self.ai_model_override, 
            "messages": messages,
            "temperature": self.ai_temperature, 
            "stream": False 
        }
        
        if self.use_json_schema: 
            payload["response_format"] = {"type": "json_object"}

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
                            if content is not None and isinstance(content, str): return content
                        raise ValueError("响应结构无效或缺少内容")
                    except (json.JSONDecodeError, ValueError) as e:
                         logging.error(f"Flux API 返回 200 OK 但响应无效: {e}. 响应: {response_text[:500]}...")
                         raise aiohttp.ClientResponseError(resp.request_info, resp.history, status=resp_status, message=f"Invalid 200 OK response: {e}", headers=resp.headers) from e
                else:
                    logging.warning(f"Flux API 调用失败: HTTP {resp_status}. 响应: {response_text[:500]}...")
                    raise aiohttp.ClientResponseError(resp.request_info, resp.history, status=resp_status, message=f"Flux API Error: {response_text}", headers=resp.headers)
        except asyncio.TimeoutError as e:
            logging.error(f"调用 Flux API 超时 (>{request_timeout.total}s)。")
            raise TimeoutError(f"Flux API call timed out after {time.time() - start_time:.2f}s") from e
        except aiohttp.ClientError as e: logging.error(f"调用 Flux API 时网络/客户端错误: {e}"); raise

    # --- MODIFIED: process_one_record_async - Removed internal retry loop ---
    async def process_one_record_async(self, session: aiohttp.ClientSession, record_id: Any, row_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a single record: creates prompt, calls API ONCE, extracts JSON.
        Returns success dict or error dict indicating API_ERROR, CONTENT_ERROR, or SYSTEM_ERROR.
        """
        log_label = f"记录[{record_id}]"
        logging.debug(f"{log_label}: 开始处理...")
        prompt = self.create_prompt(row_data)
        if not prompt: return {"_error": "prompt_generation_failed", "_error_type": ErrorType.SYSTEM_ERROR}

        try:
            # --- Call API only ONCE ---
            logging.debug(f"{log_label}: 调用 Flux API...")
            ai_response_content = await self.call_ai_api_async(session, prompt)
            logging.debug(f"{log_label}: Flux API 调用成功，提取/验证 JSON...")

            # --- Extract and Validate JSON ---
            parsed_result = self.extract_json_from_response(ai_response_content)

            # --- Handle Result ---
            if parsed_result.get("_error_type") == ErrorType.CONTENT_ERROR:
                # Content error occurred during extraction/validation
                logging.warning(f"{log_label}: 内容处理失败 ({parsed_result.get('_error')})，将重试任务。")
                # Return the error dict,上层将据此重试
                return parsed_result
            else:
                # Success! JSON extracted and validated.
                logging.info(f"{log_label}: 处理成功完成。")
                return parsed_result

        except (aiohttp.ClientResponseError, TimeoutError, aiohttp.ClientError) as e:
            # API call failed (Network, Timeout, HTTP Error >= 400)
            status_code = getattr(e, 'status', None)
            logging.warning(f"{log_label}: API调用失败. 类型: {type(e).__name__}, Status: {status_code or 'N/A'}, Err: {str(e)[:200]}... 将重试任务并触发暂停。")
            # Return dict indicating API error
            return {
                "_error": f"api_call_failed: {type(e).__name__}",
                "_error_type": ErrorType.API_ERROR,
                "_details": str(e)
            }
        except Exception as e:
            # Catch any other unexpected error during the process
            logging.exception(f"{log_label}: 处理中意外错误: {e}")
            # Mark as system error
            return {"_error": f"unexpected_error: {str(e)}", "_error_type": ErrorType.SYSTEM_ERROR}
        # --- END MODIFIED METHOD ---

    # --- 替换为新的连续任务流处理方法 ---
    async def process_shard_async_continuous(self):
        """连续任务流模式的异步处理方法，避免批次锁定"""
        if not self.task_manager.initialize():
            logging.info("无任务或初始化失败。")
            return

        try:
            # 创建 TCPConnector 并配置连接限制
            connector = aiohttp.TCPConnector(
                limit=self.max_connections,
                limit_per_host=self.max_connections_per_host
            )
            logging.debug(f"创建 aiohttp TCP连接器: limit={self.max_connections}, limit_per_host={self.max_connections_per_host}")
            
            # 使用配置好的连接器创建 ClientSession
            async with aiohttp.ClientSession(connector=connector) as session:
                current_shard_num = 0
                
                # 任务池跟踪和管理变量
                active_tasks = set()  # 当前活动任务集合
                max_concurrent = self.batch_size  # 最大并发数，复用原有的 batch_size 配置
                task_id_map = {}  # 映射 task -> (record_id, data) 用于结果处理
                results_buffer = {}  # 暂存处理结果，稍后批量更新
                
                # 记录处理统计信息
                processed_in_shard_successfully = 0
                last_progress_log_time = time.time()  # 上次进度日志时间
                progress_log_interval = 5.0  # 进度日志间隔（秒）
                
                while True:
                    # 检查是否需要加载新分片
                    if not self.task_pool.has_tasks() and len(active_tasks) == 0:
                        if not self.task_manager.load_next_shard():
                            logging.info("所有分片加载完毕。")
                            break
                        current_shard_num += 1
                        processed_in_shard_successfully = 0
                        logging.info(f"--- 开始处理分片 {current_shard_num}/{self.task_manager.total_shards} ---")
                    
                    # 填充任务池至最大并发数
                    space_available = max_concurrent - len(active_tasks)
                    if space_available > 0 and self.task_pool.has_tasks():
                        # 获取任务，一次最多获取可用空间数量
                        fetch_count = min(space_available, self.task_pool.get_remaining_count())
                        tasks_batch = self.task_pool.get_task_batch(fetch_count)
                        
                        if tasks_batch:
                            for record_id, data in tasks_batch:
                                # 为每个任务创建异步任务并添加到活动集合
                                task = asyncio.create_task(
                                    self.process_one_record_async(session, record_id, data)
                                )
                                # 保存任务ID映射关系用于后续处理
                                task_id_map[task] = (record_id, data)
                                active_tasks.add(task)
                            
                            logging.debug(f"分片 {current_shard_num}: 填充 {len(tasks_batch)} 个新任务到任务池，当前活动任务：{len(active_tasks)}")
                    
                    # 如果没有活动任务，则可能是等待加载新分片或已经处理完成
                    if not active_tasks:
                        await asyncio.sleep(0.1)  # 短暂等待
                        continue
                    
                    # 等待任一任务完成或短暂超时
                    done, pending = await asyncio.wait(
                        active_tasks,
                        timeout=1.0,  # 短暂超时以便定期检查和日志
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    # 更新活动任务集合
                    active_tasks = pending
                    
                    # 处理已完成任务
                    api_error_in_batch = False  # 跟踪是否有API错误需要暂停
                    tasks_to_retry = []  # 需要重试的任务
                    
                    for completed_task in done:
                        try:
                            result = completed_task.result()
                            record_id, original_data = task_id_map.pop(completed_task)
                            
                            # 根据结果类型处理
                            if isinstance(result, dict):
                                error_type = result.get("_error_type")
                                
                                if error_type == ErrorType.API_ERROR:
                                    api_error_in_batch = True
                                    logging.warning(f"记录[{record_id}] API错误: {result.get('_error')}，将重试")
                                    tasks_to_retry.append((record_id, None))  # 需要重载数据
                                
                                elif error_type == ErrorType.CONTENT_ERROR:
                                    logging.warning(f"记录[{record_id}] 内容错误: {result.get('_error')}，将重试")
                                    tasks_to_retry.append((record_id, None))  # 需要重载数据
                                
                                elif error_type == ErrorType.SYSTEM_ERROR:
                                    logging.error(f"记录[{record_id}] 系统错误: {result.get('_error')}，将重试")
                                    tasks_to_retry.append((record_id, None))  # 需要重载数据
                                
                                else:
                                    # 成功处理，将结果添加到更新缓冲区
                                    results_buffer[record_id] = result
                                    if "_error" not in result:
                                        processed_in_shard_successfully += 1
                                        self.task_manager.total_processed_successfully += 1
                                        logging.debug(f"记录[{record_id}] 处理成功")
                            
                            else:
                                # 未知结果类型
                                logging.error(f"记录[{record_id}] 返回未知类型结果: {type(result)}，标记为系统错误")
                                results_buffer[record_id] = {
                                    "_error": f"unknown_result_type: {type(result).__name__}",
                                    "_error_type": ErrorType.SYSTEM_ERROR
                                }
                        
                        except Exception as e:
                            # 任务执行过程中发生异常
                            try:
                                record_id, original_data = task_id_map.pop(completed_task, (None, None))
                                if record_id is not None:
                                    logging.error(f"记录[{record_id}] 处理时发生异常: {e}", exc_info=True)
                                    tasks_to_retry.append((record_id, None))
                                else:
                                    logging.error(f"任务执行异常，无法确定记录ID: {e}", exc_info=True)
                            except Exception as inner_e:
                                logging.error(f"处理任务异常时又发生异常: {inner_e}", exc_info=True)
                    
                    # 处理 API 错误暂停逻辑（如果有API错误）
                    if api_error_in_batch:
                        now = time.time()
                        if now >= self.last_api_pause_end_time:
                            logging.warning(f"检测到API错误，全局暂停 {self.api_pause_duration} 秒...")
                            await asyncio.sleep(self.api_pause_duration)
                            self.last_api_pause_end_time = time.time() + self.api_error_trigger_window
                            logging.info(f"全局暂停结束，错误触发窗口将持续到 {datetime.fromtimestamp(self.last_api_pause_end_time).strftime('%H:%M:%S')}")
                        else:
                            logging.info(f"检测到API错误，但当前仍在错误触发窗口内（直到 {datetime.fromtimestamp(self.last_api_pause_end_time).strftime('%H:%M:%S')}），本次不暂停。")
                    
                    # 重新加载需要重试的任务数据（使用线程池以避免阻塞）
                    if tasks_to_retry:
                        logging.info(f"重新加载 {len(tasks_to_retry)} 个任务的数据...")
                        loop = asyncio.get_running_loop()
                        reload_futures = [
                            loop.run_in_executor(None, lambda rid=rid: self.task_pool.reload_task_data(rid))
                            for rid, _ in tasks_to_retry
                        ]
                        reloaded_datas = await asyncio.gather(*reload_futures, return_exceptions=True)
                        
                        retry_success_count = 0
                        for i, reloaded_result in enumerate(reloaded_datas):
                            original_rid, _ = tasks_to_retry[i]
                            
                            if isinstance(reloaded_result, Exception):
                                logging.error(f"记录[{original_rid}] 数据重载失败: {reloaded_result}")
                                results_buffer[original_rid] = {
                                    "_error": f"reload_failed: {reloaded_result}",
                                    "_error_type": ErrorType.SYSTEM_ERROR
                                }
                            elif reloaded_result is None:
                                logging.error(f"记录[{original_rid}] 数据重载返回None")
                                results_buffer[original_rid] = {
                                    "_error": "reload_returned_none",
                                    "_error_type": ErrorType.SYSTEM_ERROR
                                }
                            else:
                                self.task_pool.add_task_to_front(original_rid, reloaded_result)
                                retry_success_count += 1
                        
                        if retry_success_count > 0:
                            logging.info(f"成功将 {retry_success_count} 个任务放回队列头部重试")
                    
                    # 批量更新结果（如果有）
                    if results_buffer:
                        results_count = len(results_buffer)
                        success_count = sum(1 for r in results_buffer.values() if "_error" not in r)
                        
                        logging.info(f"更新 {results_count} 条记录结果 ({success_count} 成功, {results_count - success_count} 失败)...")
                        try:
                            # 调用任务池的更新方法 (Excel会根据save_interval决定是否写入文件)
                            self.task_pool.update_task_results(results_buffer)
                            results_buffer = {}  # 清空缓冲区
                        except Exception as update_e:
                            logging.error(f"更新结果时发生错误: {update_e}", exc_info=True)
                    
                    # 定期记录进度
                    now = time.time()
                    if now - last_progress_log_time >= progress_log_interval:
                        remaining = self.task_pool.get_remaining_count()
                        active_count = len(active_tasks)
                        
                        logging.info(
                            f"分片 {current_shard_num} 进度: 已处理={processed_in_shard_successfully}, "
                            f"活动任务={active_count}, 待处理={remaining}, "
                            f"总进度={self.task_manager.total_processed_successfully}/{self.task_manager.total_estimated}"
                        )
                        
                        # 更新性能指标和监控内存
                        if processed_in_shard_successfully > 0:
                            elapsed = time.time() - self.task_manager.start_time
                            if elapsed > 0:
                                rate = processed_in_shard_successfully / elapsed
                                self.task_manager.processing_metrics['records_per_second'] = rate
                                logging.debug(f"当前处理速率: {rate:.2f} 记录/秒")
                        
                        last_progress_log_time = now
                    
                    # 监控内存使用情况
                    self.task_manager.monitor_memory_usage()
                
                # 处理最后的结果缓冲区（如果有）
                if results_buffer:
                    try:
                        logging.info(f"更新最后 {len(results_buffer)} 条记录结果...")
                        self.task_pool.update_task_results(results_buffer)
                    except Exception as final_e:
                        logging.error(f"更新最终结果时发生错误: {final_e}", exc_info=True)
                
                # 所有分片处理完成
                logging.info("所有分片处理循环结束。")
                
        except Exception as e:
            logging.error(f"处理过程中发生未捕获异常: {e}", exc_info=True)
            raise
        finally:
            # 确保最终清理
            self.task_manager.finalize()

    # --- 更新原始的批处理方法（保留用于兼容）---
    async def process_shard_async(self):
        """原始的基于批次的异步处理方法（保留用于兼容）。"""
        logging.warning("使用旧版基于批次的处理方法 process_shard_async()，建议使用新版的连续流处理方法。")
        
        if not self.task_manager.initialize():
            logging.info("无任务或初始化失败。")
            return

        try:
            # 创建 TCPConnector 并配置连接限制
            connector = aiohttp.TCPConnector(
                limit=self.max_connections,
                limit_per_host=self.max_connections_per_host
            )
            logging.debug(f"创建 aiohttp TCP连接器: limit={self.max_connections}, limit_per_host={self.max_connections_per_host}")
            
            # 使用配置好的连接器创建 ClientSession
            async with aiohttp.ClientSession(connector=connector) as session:
                current_shard_num = 0
                while True:
                    if not self.task_manager.load_next_shard():
                        logging.info("所有分片加载完毕。")
                        break
                    current_shard_num += 1
                    processed_in_shard_successfully = 0

                    while self.task_pool.has_tasks():
                        batch_size = min(self.batch_size, self.task_pool.get_remaining_count())
                        tasks_batch = self.task_pool.get_task_batch(batch_size)
                        if not tasks_batch: break

                        logging.info(f"分片 {current_shard_num}: 处理批次 {len(tasks_batch)} 个任务...")
                        batch_start_time = time.time()
                        # 创建并发执行 API 调用的协程列表
                        coros = [self.process_one_record_async(session, rid, data) for rid, data in tasks_batch]
                        # 并发执行并收集结果（包括异常）
                        results = await asyncio.gather(*coros, return_exceptions=True)

                        # --- 处理批次结果 ---
                        batch_results_to_update: Dict[Any, Dict[str, Any]] = {}
                        tasks_to_retry: List[Tuple[Any, Optional[Dict[str, Any]]]] = [] # 存储需要重试的任务ID (稍后重载数据)
                        api_error_in_batch = False # 标记本批次是否有 API 错误

                        for i, result in enumerate(results):
                            record_id, _ = tasks_batch[i] # 获取对应的记录ID
                            task_result_status = "UNKNOWN" # 用于调试日志

                            if isinstance(result, Exception):
                                # 捕获到 gather 或 process_one_record_async 中的未处理异常
                                logging.error(f"记录[{record_id}] 处理时捕获到意外异常: {result}", exc_info=result)
                                tasks_to_retry.append((record_id, None)) # 标记需要重载数据并重试
                                task_result_status = "SYSTEM_RETRY"
                            elif isinstance(result, dict):
                                etype = result.get("_error_type")
                                if etype == ErrorType.API_ERROR:
                                    api_error_in_batch = True # 标记发生了 API 错误，可能需要暂停
                                    logging.warning(f"记录[{record_id}] API错误: {result.get('_error')}. 重试并可能暂停。")
                                    tasks_to_retry.append((record_id, None))
                                    task_result_status = "API_RETRY"
                                elif etype == ErrorType.CONTENT_ERROR:
                                    # 内容错误现在也触发重试
                                    logging.warning(f"记录[{record_id}] 内容错误: {result.get('_error')}. 重试。")
                                    tasks_to_retry.append((record_id, None))
                                    task_result_status = "CONTENT_RETRY"
                                elif etype == ErrorType.SYSTEM_ERROR:
                                    logging.error(f"记录[{record_id}] 系统错误: {result.get('_error')}. 重试。")
                                    tasks_to_retry.append((record_id, None))
                                    task_result_status = "SYSTEM_RETRY"
                                else: # Success (没有 _error_type 或为 None)
                                    batch_results_to_update[record_id] = result
                                    task_result_status = "SUCCESS"
                            else: # 返回了未知类型的结果
                                batch_results_to_update[record_id] = {"_error": "unknown_result_type", "_error_type": ErrorType.SYSTEM_ERROR}
                                logging.error(f"记录[{record_id}] 返回未知类型结果: {type(result)}. 标记失败。")
                                task_result_status = "FAILED_UNKNOWN"

                            logging.debug(f"记录[{record_id}] 状态: {task_result_status}")

                        # --- 处理 API 错误暂停逻辑 (如果本批次有API错误) ---
                        if api_error_in_batch:
                            now = time.time()
                            # 检查当前时间是否已经超过了上次暂停应该结束的时间点
                            if now >= self.last_api_pause_end_time:
                                logging.warning(f"检测到API错误，全局暂停 {self.api_pause_duration} 秒...")
                                await asyncio.sleep(self.api_pause_duration)
                                # 更新下一次可以触发暂停的时间点 (当前时间 + 触发窗口)
                                self.last_api_pause_end_time = time.time() + self.api_error_trigger_window
                                logging.info(f"全局暂停结束，错误触发窗口将持续到 {datetime.fromtimestamp(self.last_api_pause_end_time).strftime('%H:%M:%S')}")
                            else:
                                # 如果当前时间仍在上次触发的窗口期内，则不再次暂停
                                logging.info(f"检测到API错误，但当前仍在错误触发窗口内（直到 {datetime.fromtimestamp(self.last_api_pause_end_time).strftime('%H:%M:%S')}），本次不暂停。")

                        # --- 重新加载需要重试的任务数据 ---
                        if tasks_to_retry:
                            logging.info(f"分片 {current_shard_num}: 重新加载 {len(tasks_to_retry)} 个任务的数据...")
                            # --- 使用 run_in_executor 和 lambda 执行同步的 reload_task_data ---
                            loop = asyncio.get_running_loop()
                            reload_futures = [
                                # 对于列表中的每个 rid，创建一个在线程池中运行 reload_task_data(rid) 的任务
                                # lambda rid=rid: ... 使用默认参数捕获当前的 rid 值
                                loop.run_in_executor(None, lambda rid=rid: self.task_pool.reload_task_data(rid))
                                for rid, _ in tasks_to_retry # 从 tasks_to_retry 中获取 rid
                            ]
                            # 并发等待所有重载任务完成
                            reloaded_datas = await asyncio.gather(*reload_futures, return_exceptions=True)
                            # ---

                            final_retry_queue: List[Tuple[Any, Dict[str, Any]]] = []
                            for i, reloaded_result in enumerate(reloaded_datas):
                                original_rid, _ = tasks_to_retry[i]
                                # 检查重载结果是否是异常或 None
                                if isinstance(reloaded_result, Exception) or reloaded_result is None:
                                    error_info = str(reloaded_result) if isinstance(reloaded_result, Exception) else "返回None"
                                    logging.error(f"记录[{original_rid}] 数据重载失败，无法重试。错误: {error_info}")
                                    # 将这个记录标记为最终失败，以便更新其错误状态
                                    batch_results_to_update[original_rid] = {"_error": f"reload_failed: {error_info}", "_error_type": ErrorType.SYSTEM_ERROR}
                                elif isinstance(reloaded_result, dict): # 确认重载结果是字典
                                    final_retry_queue.append((original_rid, reloaded_result))
                                else: # 如果返回了其他非字典类型
                                    logging.error(f"记录[{original_rid}] 数据重载返回意外类型: {type(reloaded_result)}。无法重试。")
                                    batch_results_to_update[original_rid] = {"_error": f"reload_returned_unexpected_type: {type(reloaded_result).__name__}", "_error_type": ErrorType.SYSTEM_ERROR}

                            # 将成功重载的任务数据放回任务池队列头部
                            if final_retry_queue:
                                logging.info(f"分片 {current_shard_num}: 将 {len(final_retry_queue)} 个重载成功的任务放回队列...")
                                # 反向插入以尽量保持原始顺序（相对于其他重试任务）
                                for rid, rdata in reversed(final_retry_queue):
                                    self.task_pool.add_task_to_front(rid, rdata)

                        # --- 更新成功或最终失败的任务结果 ---
                        if batch_results_to_update:
                            # 计算本次更新中成功的数量和最终失败的数量
                            success_count = sum(1 for r in batch_results_to_update.values() if "_error" not in r)
                            final_fail_count = len(batch_results_to_update) - success_count
                            logging.info(f"分片 {current_shard_num}: 更新 {len(batch_results_to_update)} 条记录结果 ({success_count} 成功, {final_fail_count} 最终失败)...")
                            try:
                                # 调用任务池的更新方法
                                self.task_pool.update_task_results(batch_results_to_update)
                                # 更新统计信息
                                processed_in_shard_successfully += success_count
                                self.task_manager.total_processed_successfully += success_count
                            except Exception as update_e:
                                # 记录更新过程中的错误，这可能表示数据写入失败
                                logging.error(f"分片 {current_shard_num}: 更新结果时发生错误: {update_e}", exc_info=True)

                        # --- 记录批处理进度 ---
                        batch_time = time.time() - batch_start_time
                        retry_count_this_batch = len(tasks_to_retry) # 本批次尝试重试的任务总数
                        # 使用本次批处理成功写入的数量来更新性能指标
                        self.task_manager.update_processing_metrics(success_count, batch_time)
                        # 获取当前分片在内存中剩余的任务数
                        remaining = self.task_pool.get_remaining_count()
                        logging.info(
                            f"分片 {current_shard_num} 批处理: 成={success_count}, 终败={final_fail_count}, 待重试={retry_count_this_batch}. "
                            f"耗时={batch_time:.2f}s ({self.task_manager.processing_metrics['records_per_second']:.2f} rec/s). 剩={remaining}. "
                            f"总={self.task_manager.total_processed_successfully}/{self.task_manager.total_estimated}."
                        )
                        # 监控内存使用情况
                        self.task_manager.monitor_memory_usage()

                    # 当前分片的所有任务（初始加载的）都已尝试处理（可能部分被放回队列）
                    logging.info(f"--- 分片 {current_shard_num} 处理完毕 ---")

                # 所有分片都已加载和处理
                logging.info("所有分片处理循环结束。")
                # 执行最终的清理工作（例如，保存最后的 Excel 文件）
                self.task_manager.finalize()
        except Exception as e:
            logging.error(f"创建 aiohttp.ClientSession 或处理过程中出错: {e}", exc_info=True)
            # 确保在发生异常时也执行清理
            self.task_manager.finalize()
            raise
            
    # --- 更新主处理方法，使用新的连续流处理模式 ---
    def process_tasks(self):
        """入口点函数，启动异步处理工作流。"""
        proc_start_time = time.time()
        logging.info("开始执行 AI 数据处理任务...")
        try:
            # 使用新的连续流处理方法替代原有的批处理方法
            asyncio.run(self.process_shard_async_continuous())
        except KeyboardInterrupt:
            logging.warning("检测到用户中断。尝试优雅退出...")
            if hasattr(self, 'task_manager') and self.task_manager:
                self.task_manager.finalize()
            logging.info("程序已中断。")
        except Exception as e:
            logging.critical(f"执行过程中发生未处理异常: {e}", exc_info=True)
            if hasattr(self, 'task_manager') and self.task_manager:
                self.task_manager.finalize()
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