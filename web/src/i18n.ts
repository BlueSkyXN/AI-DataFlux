// i18n - Internationalization support for AI-DataFlux Control Panel
// Supports English and Chinese languages

export type Language = 'en' | 'zh';

export interface Translations {
  // Header
  controlPanel: string;

  // Tabs
  dashboard: string;
  config: string;
  logs: string;

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
  logs: 'Logs',

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
  rawYamlEditor: 'Raw YAML Editor',

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
  logs: '日志',

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
  rawYamlEditor: '原始 YAML 编辑器',

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
