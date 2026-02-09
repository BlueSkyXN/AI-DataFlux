// i18n - Internationalization support for AI-DataFlux Control Panel
// Supports English and Chinese languages

export type Language = 'en' | 'zh';

export interface Translations {
  // Header
  controlPanel: string;

  // Tabs
  dashboard: string;
  config: string;
  monitor: string;

  // Dashboard
  configFilePath: string;
  gateway: string;
  process: string;
  external: string;
  running: string;
  stopped: string;
  exited: string;
  start: string;
  stop: string;
  stopConfirmTitle: string;
  stopConfirmMessage: string;
  stopConfirmMessageProcess: string;
  cancel: string;
  pid: string;
  port: string;
  models: string;
  runtime: string;
  progress: string;
  shard: string;
  errors: string;
  exitCode: string;
  workingDirectory: string;

  // Config Editor
  configFile: string;
  reload: string;
  loading: string;
  validate: string;
  save: string;
  saving: string;
  unsavedChanges: string;
  saved: string;
  savedWithBackup: string;
  enterYamlConfig: string;
  yamlSyntaxValid: string;
  yamlSyntaxError: string;
  yamlNoTabs: string;
  failedToLoad: string;
  failedToSave: string;
  failedToValidate: string;
  discardAndReload: string;
  discardAndLoadNew: string;
  rawYamlEditor: string;
  yamlParseError: string;
  commentLossWarning: string;

  // Config Sidebar Sections
  cfgGlobal: string;
  cfgDatasource: string;
  cfgConcurrency: string;
  cfgColumns: string;
  cfgValidation: string;
  cfgModels: string;
  cfgChannels: string;
  cfgPrompt: string;
  cfgRouting: string;

  // Global Section
  cfgApiGatewayUrl: string;
  cfgApiGatewayUrlDesc: string;
  cfgLogSettings: string;
  cfgLogLevel: string;
  cfgLogFormat: string;
  cfgLogOutput: string;
  cfgLogFilePath: string;
  cfgGatewayConnection: string;
  cfgGatewayConnectionDesc: string;
  cfgMaxConnections: string;
  cfgMaxConnectionsPerHost: string;

  // Datasource Section
  cfgDatasourceType: string;
  cfgType: string;
  cfgEngineSettings: string;
  cfgEngineSettingsDesc: string;
  cfgEngine: string;
  cfgExcelReader: string;
  cfgExcelWriter: string;
  cfgRequireAllFields: string;
  cfgRequireAllFieldsDesc: string;
  cfgConnectionSettings: string;
  cfgInputPath: string;
  cfgOutputPath: string;
  cfgOutputPathDefault: string;
  cfgUser: string;
  cfgPassword: string;
  cfgDatabase: string;
  cfgTableName: string;
  cfgPoolSize: string;
  cfgDbPath: string;
  cfgFeishuBitable: string;
  cfgFeishuSheet: string;
  cfgFeishuDesc: string;
  cfgFeishuAppId: string;
  cfgFeishuAppSecret: string;
  cfgFeishuAppToken: string;
  cfgFeishuTableId: string;
  cfgFeishuSpreadsheetToken: string;
  cfgFeishuSheetId: string;
  cfgFeishuMaxRetries: string;
  cfgFeishuQpsLimit: string;
  cfgFeishuQpsLimitDesc: string;

  // Concurrency Section
  cfgBasicConcurrency: string;
  cfgBatchSize: string;
  cfgBatchSizeDesc: string;
  cfgSaveInterval: string;
  cfgSaveIntervalDesc: string;
  cfgMaxConnPerHostDesc: string;
  cfgShardSettings: string;
  cfgShardSize: string;
  cfgMinShardSize: string;
  cfgMaxShardSize: string;
  cfgCircuitBreaker: string;
  cfgCircuitBreakerDesc: string;
  cfgApiPauseDuration: string;
  cfgApiPauseDurationDesc: string;
  cfgApiErrorWindow: string;
  cfgApiErrorWindowDesc: string;
  cfgRetryLimits: string;
  cfgRetryLimitsDesc: string;
  cfgApiError: string;
  cfgContentError: string;
  cfgSystemError: string;

  // Columns Section
  cfgColumnsToExtract: string;
  cfgColumnsToExtractDesc: string;
  cfgColumnsToWrite: string;
  cfgColumnsToWriteDesc: string;
  cfgAddColumn: string;
  cfgAlias: string;
  cfgColumnName: string;
  cfgAdd: string;

  // Validation Section
  cfgValidationSettings: string;
  cfgEnableValidation: string;
  cfgFieldRules: string;
  cfgFieldRulesDesc: string;
  cfgAddAllowedValue: string;
  cfgNewFieldName: string;
  cfgAddField: string;

  // Models Section
  cfgModelsTitle: string;
  cfgModelsDesc: string;
  cfgModelName: string;
  cfgModelId: string;
  cfgChannelId: string;
  cfgTimeout: string;
  cfgWeight: string;
  cfgWeightDesc: string;
  cfgTemperature: string;
  cfgSafeRps: string;
  cfgSafeRpsDesc: string;
  cfgAdvancedParams: string;
  cfgAddModel: string;

  // Channels Section
  cfgChannelsTitle: string;
  cfgChannelsDesc: string;
  cfgChannelName: string;
  cfgChannelIdPlaceholder: string;
  cfgAddChannel: string;
  cfgChannelInUse: string;
  cfgIpPoolDesc: string;

  // Prompt Section
  cfgPromptSettings: string;
  cfgRequiredFields: string;
  cfgRequiredFieldsDesc: string;
  cfgTempOverride: string;
  cfgSystemPrompt: string;
  cfgSystemPromptPlaceholder: string;
  cfgTemplate: string;
  cfgTemplateDesc: string;
  cfgTemplatePlaceholder: string;
  cfgTokenEstimation: string;
  cfgTokenEstimationDesc: string;
  cfgTokenMode: string;
  cfgSampleSize: string;
  cfgSampleSizeDesc: string;
  cfgEncoding: string;

  // Routing Section
  cfgRoutingSettings: string;
  cfgRoutingDesc: string;
  cfgEnableRouting: string;
  cfgRoutingField: string;
  cfgRoutingFieldDesc: string;
  cfgSubtasks: string;
  cfgMatchValue: string;
  cfgProfilePath: string;
  cfgAddSubtask: string;

  // Logs
  connected: string;
  disconnected: string;
  reconnecting: string;
  autoScroll: string;
  reconnect: string;
  clear: string;
  copy: string;
  waitingForLogs: string;
  reconnectingToServer: string;
  notConnectedClickReconnect: string;
  wordWrap: string;

  // Footer
  footerText: string;

  // Errors
  failedToConnect: string;
  failedToStartGateway: string;
  failedToStopGateway: string;
  failedToStartProcess: string;
  failedToStopProcess: string;

  // New features
  edit: string;
  browse: string;
  chooseFile: string;
  chooseFolder: string;
  controllerStatus: string;
  controllerConnected: string;
  controllerDisconnected: string;
}

const en: Translations = {
  // Header
  controlPanel: 'Control Panel',

  // Tabs
  dashboard: 'Dashboard',
  config: 'Config',
  monitor: 'Monitor',

  // Dashboard
  configFilePath: 'Config File Path',
  gateway: 'Gateway',
  process: 'Process',
  external: 'External',
  running: 'Running',
  stopped: 'Stopped',
  exited: 'Exited',
  start: 'Start',
  stop: 'Stop',
  stopConfirmTitle: 'Stop {target}?',
  stopConfirmMessage: 'Are you sure you want to stop the {target}?',
  stopConfirmMessageProcess: 'Any ongoing processing will be interrupted.',
  cancel: 'Cancel',
  pid: 'PID',
  port: 'Port',
  models: 'Models',
  runtime: 'Runtime',
  progress: 'Progress',
  shard: 'Shard',
  errors: 'Errors',
  exitCode: 'Exit Code',
  workingDirectory: 'Working Directory',

  // Config Editor
  configFile: 'Config File',
  reload: 'Reload',
  loading: 'Loading...',
  validate: 'Validate',
  save: 'Save',
  saving: 'Saving...',
  unsavedChanges: '● Unsaved changes',
  saved: 'Saved',
  savedWithBackup: 'Saved (backup created)',
  enterYamlConfig: 'Enter YAML configuration...',
  yamlSyntaxValid: 'YAML syntax is valid',
  yamlSyntaxError: 'YAML syntax error',
  yamlNoTabs: 'YAML does not allow tabs. Please use spaces for indentation.',
  failedToLoad: 'Failed to load config',
  failedToSave: 'Failed to save config',
  failedToValidate: 'Failed to validate YAML',
  discardAndReload: 'Discard unsaved changes and reload from disk?',
  discardAndLoadNew: 'You have unsaved changes. Discard them and load the new config?',
  rawYamlEditor: 'Raw YAML',
  yamlParseError: 'Failed to parse YAML. Switching to Raw YAML mode.',
  commentLossWarning: 'Visual editor will remove YAML comments on save.',

  // Config Sidebar Sections
  cfgGlobal: 'Global',
  cfgDatasource: 'Datasource',
  cfgConcurrency: 'Concurrency',
  cfgColumns: 'Columns',
  cfgValidation: 'Validation',
  cfgModels: 'Models',
  cfgChannels: 'Channels',
  cfgPrompt: 'Prompt',
  cfgRouting: 'Routing',

  // Global Section
  cfgApiGatewayUrl: 'API Gateway',
  cfgApiGatewayUrlDesc: 'URL auto-appends /v1/chat/completions',
  cfgLogSettings: 'Log Settings',
  cfgLogLevel: 'Level',
  cfgLogFormat: 'Format',
  cfgLogOutput: 'Output',
  cfgLogFilePath: 'File Path',
  cfgGatewayConnection: 'Gateway Connection',
  cfgGatewayConnectionDesc: 'Connection limits from gateway to upstream AI providers',
  cfgMaxConnections: 'Max Connections',
  cfgMaxConnectionsPerHost: 'Max Per Host',

  // Datasource Section
  cfgDatasourceType: 'Datasource Type',
  cfgType: 'Type',
  cfgEngineSettings: 'Engine Settings',
  cfgEngineSettingsDesc: 'DataFrame engine and reader/writer selection',
  cfgEngine: 'Engine',
  cfgExcelReader: 'Excel Reader',
  cfgExcelWriter: 'Excel Writer',
  cfgRequireAllFields: 'Require All Fields',
  cfgRequireAllFieldsDesc: 'true = all fields must be non-empty, false = at least one',
  cfgConnectionSettings: 'Connection Settings',
  cfgInputPath: 'Input Path',
  cfgOutputPath: 'Output Path',
  cfgOutputPathDefault: 'Same as input path',
  cfgUser: 'User',
  cfgPassword: 'Password',
  cfgDatabase: 'Database',
  cfgTableName: 'Table Name',
  cfgPoolSize: 'Pool Size',
  cfgDbPath: 'DB Path',
  cfgFeishuBitable: 'Feishu Bitable',
  cfgFeishuSheet: 'Feishu Sheet',
  cfgFeishuDesc: 'Feishu (Lark) cloud spreadsheet. Requires app_id and app_secret from Feishu Open Platform.',
  cfgFeishuAppId: 'App ID',
  cfgFeishuAppSecret: 'App Secret',
  cfgFeishuAppToken: 'App Token',
  cfgFeishuTableId: 'Table ID',
  cfgFeishuSpreadsheetToken: 'Spreadsheet Token',
  cfgFeishuSheetId: 'Sheet ID',
  cfgFeishuMaxRetries: 'Max Retries',
  cfgFeishuQpsLimit: 'QPS Limit',
  cfgFeishuQpsLimitDesc: '0 = unlimited',

  // Concurrency Section
  cfgBasicConcurrency: 'Basic Concurrency',
  cfgBatchSize: 'Batch Size',
  cfgBatchSizeDesc: 'Max concurrent tasks',
  cfgSaveInterval: 'Save Interval',
  cfgSaveIntervalDesc: 'Excel/CSV save interval (seconds)',
  cfgMaxConnPerHostDesc: '0 = unlimited',
  cfgShardSettings: 'Shard Settings',
  cfgShardSize: 'Shard Size',
  cfgMinShardSize: 'Min Shard Size',
  cfgMaxShardSize: 'Max Shard Size',
  cfgCircuitBreaker: 'Circuit Breaker',
  cfgCircuitBreakerDesc: 'Auto-pause on consecutive API errors',
  cfgApiPauseDuration: 'Pause Duration (s)',
  cfgApiPauseDurationDesc: 'Seconds to pause on API error',
  cfgApiErrorWindow: 'Error Window (s)',
  cfgApiErrorWindowDesc: 'Time window for error trigger',
  cfgRetryLimits: 'Retry Limits',
  cfgRetryLimitsDesc: 'Max retries by error type',
  cfgApiError: 'API Error',
  cfgContentError: 'Content Error',
  cfgSystemError: 'System Error',

  // Columns Section
  cfgColumnsToExtract: 'Columns to Extract',
  cfgColumnsToExtractDesc: 'Fields to read from data source and send to AI',
  cfgColumnsToWrite: 'Columns to Write',
  cfgColumnsToWriteDesc: 'Map AI output field aliases to actual column names',
  cfgAddColumn: 'Add column name...',
  cfgAlias: 'alias',
  cfgColumnName: 'column name',
  cfgAdd: 'Add',

  // Validation Section
  cfgValidationSettings: 'Validation Settings',
  cfgEnableValidation: 'Enable Validation',
  cfgFieldRules: 'Field Rules',
  cfgFieldRulesDesc: 'Define allowed values for each AI output field',
  cfgAddAllowedValue: 'Add allowed value...',
  cfgNewFieldName: 'Field name...',
  cfgAddField: 'Add Field',

  // Models Section
  cfgModelsTitle: 'Model Configuration',
  cfgModelsDesc: 'AI models available for processing. Weighted random selection.',
  cfgModelName: 'Display Name',
  cfgModelId: 'Model ID',
  cfgChannelId: 'Channel',
  cfgTimeout: 'Timeout (s)',
  cfgWeight: 'Weight',
  cfgWeightDesc: '0 = disabled',
  cfgTemperature: 'Temperature',
  cfgSafeRps: 'Safe RPS',
  cfgSafeRpsDesc: 'Token bucket: capacity = rps * 2',
  cfgAdvancedParams: 'Advanced Params',
  cfgAddModel: 'Add Model',

  // Channels Section
  cfgChannelsTitle: 'Channel Configuration',
  cfgChannelsDesc: 'API endpoints for connecting to AI providers',
  cfgChannelName: 'Name',
  cfgChannelIdPlaceholder: 'ID...',
  cfgAddChannel: 'Add Channel',
  cfgChannelInUse: 'This channel is referenced by models. Delete anyway?',
  cfgIpPoolDesc: 'IP addresses for round-robin DNS. Ignored when proxy is set.',

  // Prompt Section
  cfgPromptSettings: 'Prompt Settings',
  cfgRequiredFields: 'Required Fields',
  cfgRequiredFieldsDesc: 'Fields that AI must include in response',
  cfgTempOverride: 'Override Model Temp',
  cfgSystemPrompt: 'System Prompt',
  cfgSystemPromptPlaceholder: 'Enter system prompt...',
  cfgTemplate: 'Template',
  cfgTemplateDesc: 'Use {record_json} as data placeholder',
  cfgTemplatePlaceholder: 'Enter prompt template...',
  cfgTokenEstimation: 'Token Estimation',
  cfgTokenEstimationDesc: 'Settings for token cost estimation',
  cfgTokenMode: 'Mode',
  cfgSampleSize: 'Sample Size',
  cfgSampleSizeDesc: '-1 = all records',
  cfgEncoding: 'Encoding',

  // Routing Section
  cfgRoutingSettings: 'Rule Routing',
  cfgRoutingDesc: 'Route records to different prompts based on field value',
  cfgEnableRouting: 'Enable Routing',
  cfgRoutingField: 'Routing Field',
  cfgRoutingFieldDesc: 'Field name used for routing decision',
  cfgSubtasks: 'Subtasks',
  cfgMatchValue: 'match value',
  cfgProfilePath: 'profile path',
  cfgAddSubtask: 'Add Subtask',

  // Logs
  connected: 'Connected',
  disconnected: 'Disconnected',
  reconnecting: 'Reconnecting...',
  autoScroll: 'Auto-scroll',
  reconnect: 'Reconnect',
  clear: 'Clear',
  copy: 'Copy',
  waitingForLogs: 'Waiting for logs...',
  reconnectingToServer: 'Reconnecting to server...',
  notConnectedClickReconnect: 'Not connected. Click Reconnect to start.',
  wordWrap: 'Word wrap',

  // Footer
  footerText: 'AI-DataFlux Control Panel',

  // Errors
  failedToConnect: 'Failed to connect to Control Server',
  failedToStartGateway: 'Failed to start Gateway',
  failedToStopGateway: 'Failed to stop Gateway',
  failedToStartProcess: 'Failed to start Process',
  failedToStopProcess: 'Failed to stop Process',

  // New features
  edit: 'Edit',
  browse: 'Browse',
  chooseFile: 'Choose File',
  chooseFolder: 'Choose Folder',
  controllerStatus: 'Controller',
  controllerConnected: 'Connected',
  controllerDisconnected: 'Disconnected',
};

const zh: Translations = {
  // Header
  controlPanel: '控制面板',

  // Tabs
  dashboard: '仪表盘',
  config: '配置',
  monitor: '监控',

  // Dashboard
  configFilePath: '配置文件路径',
  gateway: '网关',
  process: '处理器',
  external: '外部',
  running: '运行中',
  stopped: '已停止',
  exited: '已退出',
  start: '启动',
  stop: '停止',
  stopConfirmTitle: '停止{target}？',
  stopConfirmMessage: '确定要停止{target}吗？',
  stopConfirmMessageProcess: '正在进行的处理将被中断。',
  cancel: '取消',
  pid: '进程ID',
  port: '端口',
  models: '模型',
  runtime: '运行时间',
  progress: '进度',
  shard: '分片',
  errors: '错误',
  exitCode: '退出代码',
  workingDirectory: '工作目录',

  // Config Editor
  configFile: '配置文件',
  reload: '重新加载',
  loading: '加载中...',
  validate: '验证',
  save: '保存',
  saving: '保存中...',
  unsavedChanges: '● 有未保存的更改',
  saved: '已保存',
  savedWithBackup: '已保存（已创建备份）',
  enterYamlConfig: '输入 YAML 配置...',
  yamlSyntaxValid: 'YAML 语法有效',
  yamlSyntaxError: 'YAML 语法错误',
  yamlNoTabs: 'YAML 不允许使用制表符。请使用空格缩进。',
  failedToLoad: '加载配置失败',
  failedToSave: '保存配置失败',
  failedToValidate: '验证 YAML 失败',
  discardAndReload: '放弃未保存的更改并从磁盘重新加载？',
  discardAndLoadNew: '您有未保存的更改。放弃它们并加载新配置？',
  rawYamlEditor: '原始 YAML',
  yamlParseError: 'YAML 解析失败，已切换到原始 YAML 模式。',
  commentLossWarning: '可视化编辑器保存时将移除 YAML 注释。',

  // Config Sidebar Sections
  cfgGlobal: '全局设置',
  cfgDatasource: '数据源',
  cfgConcurrency: '并发控制',
  cfgColumns: '字段配置',
  cfgValidation: '验证规则',
  cfgModels: '模型配置',
  cfgChannels: '通道配置',
  cfgPrompt: '提示词',
  cfgRouting: '规则路由',

  // Global Section
  cfgApiGatewayUrl: 'API 网关',
  cfgApiGatewayUrlDesc: 'URL 自动补全 /v1/chat/completions',
  cfgLogSettings: '日志设置',
  cfgLogLevel: '级别',
  cfgLogFormat: '格式',
  cfgLogOutput: '输出',
  cfgLogFilePath: '文件路径',
  cfgGatewayConnection: '网关连接',
  cfgGatewayConnectionDesc: '网关到上游 AI 服务的连接限制',
  cfgMaxConnections: '最大连接数',
  cfgMaxConnectionsPerHost: '单主机最大连接',

  // Datasource Section
  cfgDatasourceType: '数据源类型',
  cfgType: '类型',
  cfgEngineSettings: '引擎设置',
  cfgEngineSettingsDesc: 'DataFrame 引擎和读写器选择',
  cfgEngine: '引擎',
  cfgExcelReader: 'Excel 读取器',
  cfgExcelWriter: 'Excel 写入器',
  cfgRequireAllFields: '要求所有字段',
  cfgRequireAllFieldsDesc: '开启 = 全部非空才处理，关闭 = 至少一个非空',
  cfgConnectionSettings: '连接配置',
  cfgInputPath: '输入路径',
  cfgOutputPath: '输出路径',
  cfgOutputPathDefault: '与输入路径相同',
  cfgUser: '用户名',
  cfgPassword: '密码',
  cfgDatabase: '数据库',
  cfgTableName: '表名',
  cfgPoolSize: '连接池大小',
  cfgDbPath: '数据库路径',
  cfgFeishuBitable: '飞书多维表格',
  cfgFeishuSheet: '飞书电子表格',
  cfgFeishuDesc: '飞书云端表格数据源，需要在飞书开放平台创建自建应用获取 App ID 和 App Secret。',
  cfgFeishuAppId: '应用 ID',
  cfgFeishuAppSecret: '应用密钥',
  cfgFeishuAppToken: '多维表格 Token',
  cfgFeishuTableId: '数据表 ID',
  cfgFeishuSpreadsheetToken: '电子表格 Token',
  cfgFeishuSheetId: '工作表 ID',
  cfgFeishuMaxRetries: '最大重试次数',
  cfgFeishuQpsLimit: 'QPS 限制',
  cfgFeishuQpsLimitDesc: '0 = 不限制',

  // Concurrency Section
  cfgBasicConcurrency: '基本并发',
  cfgBatchSize: '批大小',
  cfgBatchSizeDesc: '最大并发任务数',
  cfgSaveInterval: '保存间隔',
  cfgSaveIntervalDesc: 'Excel/CSV 保存间隔（秒）',
  cfgMaxConnPerHostDesc: '0 = 不限制',
  cfgShardSettings: '分片设置',
  cfgShardSize: '分片大小',
  cfgMinShardSize: '最小分片',
  cfgMaxShardSize: '最大分片',
  cfgCircuitBreaker: '熔断机制',
  cfgCircuitBreakerDesc: '连续 API 错误时自动暂停',
  cfgApiPauseDuration: '暂停时长（秒）',
  cfgApiPauseDurationDesc: 'API 错误时暂停秒数',
  cfgApiErrorWindow: '错误窗口（秒）',
  cfgApiErrorWindowDesc: '触发熔断的时间窗口',
  cfgRetryLimits: '重试限制',
  cfgRetryLimitsDesc: '按错误类型设置最大重试次数',
  cfgApiError: 'API 错误',
  cfgContentError: '内容错误',
  cfgSystemError: '系统错误',

  // Columns Section
  cfgColumnsToExtract: '提取字段',
  cfgColumnsToExtractDesc: '从数据源读取并发送给 AI 的字段',
  cfgColumnsToWrite: '写回字段',
  cfgColumnsToWriteDesc: 'AI 输出字段别名 → 实际列名的映射',
  cfgAddColumn: '添加字段名...',
  cfgAlias: '别名',
  cfgColumnName: '列名',
  cfgAdd: '添加',

  // Validation Section
  cfgValidationSettings: '验证设置',
  cfgEnableValidation: '启用验证',
  cfgFieldRules: '字段规则',
  cfgFieldRulesDesc: '定义每个 AI 输出字段的允许值',
  cfgAddAllowedValue: '添加允许值...',
  cfgNewFieldName: '字段名...',
  cfgAddField: '添加字段',

  // Models Section
  cfgModelsTitle: '模型配置',
  cfgModelsDesc: '可用于处理的 AI 模型，加权随机选择调度',
  cfgModelName: '显示名称',
  cfgModelId: '模型标识',
  cfgChannelId: '通道',
  cfgTimeout: '超时（秒）',
  cfgWeight: '权重',
  cfgWeightDesc: '0 = 禁用',
  cfgTemperature: '温度',
  cfgSafeRps: '安全 RPS',
  cfgSafeRpsDesc: '令牌桶容量 = rps * 2',
  cfgAdvancedParams: '高级参数',
  cfgAddModel: '添加模型',

  // Channels Section
  cfgChannelsTitle: '通道配置',
  cfgChannelsDesc: '连接 AI 服务商的 API 端点',
  cfgChannelName: '名称',
  cfgChannelIdPlaceholder: 'ID...',
  cfgAddChannel: '添加通道',
  cfgChannelInUse: '此通道被模型引用，确定删除？',
  cfgIpPoolDesc: 'IP 轮询池，设置代理时被忽略',

  // Prompt Section
  cfgPromptSettings: '提示词设置',
  cfgRequiredFields: '必需字段',
  cfgRequiredFieldsDesc: 'AI 响应中必须包含的字段',
  cfgTempOverride: '覆盖模型温度',
  cfgSystemPrompt: '系统提示词',
  cfgSystemPromptPlaceholder: '输入系统提示词...',
  cfgTemplate: '模板',
  cfgTemplateDesc: '使用 {record_json} 作为数据占位符',
  cfgTemplatePlaceholder: '输入提示词模板...',
  cfgTokenEstimation: 'Token 估算',
  cfgTokenEstimationDesc: 'Token 成本估算设置',
  cfgTokenMode: '模式',
  cfgSampleSize: '采样数量',
  cfgSampleSizeDesc: '-1 = 全量计算',
  cfgEncoding: '编码器',

  // Routing Section
  cfgRoutingSettings: '规则路由',
  cfgRoutingDesc: '根据字段值将记录路由到不同的提示词配置',
  cfgEnableRouting: '启用路由',
  cfgRoutingField: '路由字段',
  cfgRoutingFieldDesc: '用于路由决策的字段名',
  cfgSubtasks: '子任务',
  cfgMatchValue: '匹配值',
  cfgProfilePath: '配置文件路径',
  cfgAddSubtask: '添加子任务',

  // Logs
  connected: '已连接',
  disconnected: '已断开',
  reconnecting: '重新连接中...',
  autoScroll: '自动滚动',
  reconnect: '重新连接',
  clear: '清空',
  copy: '复制',
  waitingForLogs: '等待日志...',
  reconnectingToServer: '正在重新连接到服务器...',
  notConnectedClickReconnect: '未连接。点击重新连接开始。',
  wordWrap: '自动换行',

  // Footer
  footerText: 'AI-DataFlux 控制面板',

  // Errors
  failedToConnect: '连接到控制服务器失败',
  failedToStartGateway: '启动网关失败',
  failedToStopGateway: '停止网关失败',
  failedToStartProcess: '启动处理器失败',
  failedToStopProcess: '停止处理器失败',

  // New features
  edit: '编辑',
  browse: '浏览',
  chooseFile: '选择文件',
  chooseFolder: '选择文件夹',
  controllerStatus: '控制器',
  controllerConnected: '已连接',
  controllerDisconnected: '未连接',
};

const translations: Record<Language, Translations> = {
  en,
  zh,
};

// Get browser language preference
function getBrowserLanguage(): Language {
  const lang = navigator.language.toLowerCase();
  if (lang.startsWith('zh')) {
    return 'zh';
  }
  return 'en';
}

// Get stored language preference or browser default
export function getInitialLanguage(): Language {
  const stored = localStorage.getItem('language') as Language | null;
  if (stored && (stored === 'en' || stored === 'zh')) {
    return stored;
  }
  return getBrowserLanguage();
}

// Save language preference
export function saveLanguagePreference(lang: Language): void {
  localStorage.setItem('language', lang);
}

// Get translations for a language
export function getTranslations(lang: Language): Translations {
  return translations[lang];
}

// Interpolate variables in translation strings
export function interpolate(text: string, vars: Record<string, string>): string {
  return text.replace(/\{(\w+)\}/g, (match, key) => vars[key] || match);
}
