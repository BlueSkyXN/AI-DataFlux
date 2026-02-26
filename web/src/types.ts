/**
 * 全局类型定义文件
 *
 * 定义控制面板前端与后端 API 交互所使用的所有 TypeScript 类型接口。
 *
 * 导出类型清单：
 * - ManagedProcessStatus — 受控进程（Gateway / Process）的运行状态
 * - GatewayHealth — 网关健康检查响应
 * - ProcessProgress — 数据处理进度信息
 * - StatusResponse — /api/status 聚合状态响应
 * - ConfigResponse — /api/config 配置读取响应
 * - ConfigWriteResponse — /api/config PUT 配置保存响应
 * - ConfigValidateResponse — /api/config/validate 验证响应
 * - TabType — 顶部导航标签页类型
 * - LogTarget — 日志目标类型（网关 / 处理器）
 */

// API response types

/** 受控进程运行状态 */
export interface ManagedProcessStatus {
  /** 进程状态：已停止 / 运行中 / 已退出 */
  status: 'stopped' | 'running' | 'exited';
  /** 进程 ID，未运行时为 null */
  pid: number | null;
  /** 启动时间戳（Unix 秒），未运行时为 null */
  start_time: number | null;
  /** 退出代码，未退出时为 null */
  exit_code: number | null;
  /** 配置文件路径 */
  config_path: string | null;
  /** 监听端口（仅 Gateway） */
  port?: number | null;
}

/** 网关健康检查信息 */
export interface GatewayHealth {
  /** 健康状态：健康 / 降级 / 不健康 */
  status: 'healthy' | 'degraded' | 'unhealthy';
  /** 可用模型数量 */
  available_models: number;
  /** 总模型数量 */
  total_models: number;
  /** 运行时长（秒） */
  uptime: number;
}

/** 数据处理进度信息 */
export interface ProcessProgress {
  /** 任务总数 */
  total: number;
  /** 已处理任务数 */
  processed: number;
  /** 当前活跃（正在处理）任务数 */
  active: number;
  /** 当前分片标识 */
  shard: string;
  /** 累计错误数 */
  errors: number;
  /** 进度上报时间戳（Unix 秒） */
  ts: number;
}

/** /api/status 聚合状态响应 */
export interface StatusResponse {
  /** 网关状态 */
  gateway: {
    managed: ManagedProcessStatus;
    health?: GatewayHealth;
  };
  /** 处理器状态 */
  process: {
    managed: ManagedProcessStatus;
    progress?: ProcessProgress;
  };
  /** 服务器工作目录 */
  working_directory?: string;
}

/** 配置文件读取响应 */
export interface ConfigResponse {
  /** 配置文件绝对路径 */
  path: string;
  /** 配置文件 YAML 内容 */
  content: string;
}

/** 配置文件保存响应 */
export interface ConfigWriteResponse {
  /** 是否保存成功 */
  success: boolean;
  /** 保存后的文件路径 */
  path: string;
  /** 是否创建了备份 */
  backed_up: boolean;
}

/** 配置文件验证响应 */
export interface ConfigValidateResponse {
  /** YAML 语法是否合法 */
  valid: boolean;
  /** 验证错误信息（仅在 valid=false 时存在） */
  error?: string;
}

/** 顶部导航标签页类型：仪表盘 / 配置 / 监控 */
export type TabType = 'dashboard' | 'config' | 'monitor';
/** 日志目标类型：网关 / 处理器 */
export type LogTarget = 'gateway' | 'process';
