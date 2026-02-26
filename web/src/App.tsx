/**
 * 主应用组件 — AI-DataFlux 控制面板入口
 *
 * 功能职责：
 * - 顶部导航栏（Logo、标签页切换、语言切换、控制器连接状态）
 * - 配置文件路径和工作目录的显示与编辑
 * - 根据标签页渲染 Dashboard / ConfigEditor / Logs 子页面
 * - 定时轮询控制服务器连接状态（5 秒间隔）
 *
 * 导出：默认导出 App 组件
 *
 * 依赖模块：
 * - pages/Dashboard — 仪表盘页面
 * - pages/ConfigEditor — 配置编辑页面
 * - pages/Logs — 日志监控页面
 * - i18n — 国际化翻译
 * - api — 后端 API 调用
 */
import { useState, useEffect, useCallback } from 'react';
import type { TabType } from './types';
import Dashboard from './pages/Dashboard';
import ConfigEditor from './pages/ConfigEditor';
import Logs from './pages/Logs';
import { getInitialLanguage, saveLanguagePreference, getTranslations, type Language } from './i18n';
import { fetchStatus, fetchConfig } from './api';

/**
 * 主应用组件
 *
 * 管理全局状态：当前标签页、配置路径、工作目录、语言偏好、控制器连接状态。
 * 渲染顶部导航栏 + 主内容区 + 底部页脚。
 */
function App() {
  // --- 全局状态 ---
  /** 当前激活的标签页 */
  const [activeTab, setActiveTab] = useState<TabType>('dashboard');
  /** 配置文件相对路径 */
  const [configPath, setConfigPath] = useState('config.yaml');
  /** 配置文件绝对路径（从服务端获取） */
  const [absoluteConfigPath, setAbsoluteConfigPath] = useState<string>('');
  /** 当前界面语言 */
  const [language, setLanguage] = useState<Language>(getInitialLanguage());
  /** 服务器工作目录 */
  const [workingDir, setWorkingDir] = useState<string>('');
  /** 是否正在编辑配置路径 */
  const [isEditingConfigPath, setIsEditingConfigPath] = useState(false);
  /** 是否正在编辑工作目录 */
  const [isEditingWorkingDir, setIsEditingWorkingDir] = useState(false);
  /** 编辑中的临时配置路径 */
  const [tempConfigPath, setTempConfigPath] = useState('');
  /** 编辑中的临时工作目录 */
  const [tempWorkingDir, setTempWorkingDir] = useState('');
  /** 控制服务器是否连接 */
  const [controllerConnected, setControllerConnected] = useState<boolean>(false);
  /** 当前页面主机地址 */
  const host = window.location.host;

  /** 获取当前语言的翻译文本 */
  const t = getTranslations(language);

  /** 导航标签页定义 */
  const tabs: { id: TabType; label: string }[] = [
    { id: 'dashboard', label: t.dashboard },
    { id: 'config', label: t.config },
    { id: 'monitor', label: t.monitor },
  ];

  // 初始化：获取工作目录和配置文件绝对路径，并每 5 秒轮询连接状态
  // Fetch working directory and absolute config path from status
  useEffect(() => {
    const getInitialData = async () => {
      try {
        const status = await fetchStatus();
        setControllerConnected(true);
        if (status.working_directory) {
          setWorkingDir(status.working_directory);
        }
        // Fetch config to get absolute path
        try {
          const configData = await fetchConfig(configPath);
          setAbsoluteConfigPath(configData.path);
        } catch {
          // If config fetch fails, just use the relative path
          setAbsoluteConfigPath(configPath);
        }
      } catch {
        setControllerConnected(false);
      }
    };
    getInitialData();
    // Poll every 5 seconds to check connection
    const interval = setInterval(getInitialData, 5000);
    return () => clearInterval(interval);
  }, [configPath]);

  /** 切换语言并持久化偏好到 localStorage */
  const handleLanguageChange = (lang: Language) => {
    setLanguage(lang);
    saveLanguagePreference(lang);
  };

  /** 进入配置路径编辑模式 */
  const handleStartEditConfigPath = () => {
    setTempConfigPath(configPath);
    setIsEditingConfigPath(true);
  };

  /** 确认保存配置路径 */
  const handleSaveConfigPath = () => {
    setConfigPath(tempConfigPath);
    setIsEditingConfigPath(false);
  };

  /** 取消配置路径编辑 */
  const handleCancelEditConfigPath = () => {
    setIsEditingConfigPath(false);
  };

  /** 进入工作目录编辑模式 */
  const handleStartEditWorkingDir = () => {
    setTempWorkingDir(workingDir);
    setIsEditingWorkingDir(true);
  };

  /** 确认保存工作目录 */
  const handleSaveWorkingDir = () => {
    setWorkingDir(tempWorkingDir);
    setIsEditingWorkingDir(false);
  };

  /** 取消工作目录编辑 */
  const handleCancelEditWorkingDir = () => {
    setIsEditingWorkingDir(false);
  };

  /** 弹出文件选择对话框，选择 YAML 配置文件 */
  const handleChooseFile = useCallback(() => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.yaml,.yml';
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (file) {
        // Use relative path from file name
        setTempConfigPath(file.name);
      }
    };
    input.click();
  }, []);

  /** 弹出文件夹选择对话框，选择工作目录 */
  const handleChooseFolder = useCallback(() => {
    const input = document.createElement('input');
    input.type = 'file';
    input.setAttribute('webkitdirectory', '');
    input.setAttribute('directory', '');
    input.onchange = (e) => {
      const files = (e.target as HTMLInputElement).files;
      if (files && files.length > 0) {
        // Get the directory path from the first file
        const filePath = files[0].webkitRelativePath || files[0].name;
        const dirPath = filePath.split('/')[0];
        setTempWorkingDir(dirPath);
      }
    };
    input.click();
  }, []);

  return (
    <div className="min-h-screen bg-[#F8FAFB] flex flex-col">
      {/* Header */}
      <header className="bg-white shadow-[0_1px_3px_rgba(0,0,0,0.05)]">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            {/* Logo */}
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-gradient-to-br from-cyan-400 to-blue-500 rounded-lg flex items-center justify-center">
                <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
              <h1 className="text-lg font-semibold text-gray-800">AI-DataFlux</h1>
              <span className="text-xs text-gray-400 bg-gray-100 px-2 py-0.5 rounded-full">{t.controlPanel}</span>
              {/* Controller Status */}
              <div className="flex items-center gap-1.5 text-xs ml-2">
                <span
                  className={`w-2 h-2 rounded-full ${
                    controllerConnected ? 'bg-green-400' : 'bg-red-400'
                  }`}
                />
                <span className="text-gray-600">
                  {t.controllerStatus}: {controllerConnected ? t.controllerConnected : t.controllerDisconnected}
                </span>
              </div>
            </div>

            {/* Right side: Tabs + Language Selector */}
            <div className="flex items-center gap-4">
              {/* Tabs */}
              <nav className="flex gap-1">
                {tabs.map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
                      activeTab === tab.id
                        ? 'text-cyan-600 bg-cyan-50'
                        : 'text-gray-600 hover:text-gray-800 hover:bg-gray-50'
                    }`}
                  >
                    {tab.label}
                  </button>
                ))}
              </nav>

              {/* Language Selector */}
              <div className="flex gap-1 bg-gray-100 rounded-lg p-1">
                <button
                  onClick={() => handleLanguageChange('en')}
                  className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
                    language === 'en'
                      ? 'bg-white text-gray-800 shadow-sm'
                      : 'text-gray-600 hover:text-gray-800'
                  }`}
                >
                  EN
                </button>
                <button
                  onClick={() => handleLanguageChange('zh')}
                  className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
                    language === 'zh'
                      ? 'bg-white text-gray-800 shadow-sm'
                      : 'text-gray-600 hover:text-gray-800'
                  }`}
                >
                  中文
                </button>
              </div>
            </div>
          </div>

          {/* Config Path and Working Directory Section */}
          <div className="mt-3 space-y-2">
            {/* Working Directory */}
            <div className="flex items-center gap-2 text-xs">
              <svg className="w-3.5 h-3.5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
              </svg>
              <span className="text-gray-500">{t.workingDirectory}:</span>

              {!isEditingWorkingDir ? (
                <>
                  <span className="font-mono text-gray-700">{workingDir}</span>
                  <button
                    onClick={handleStartEditWorkingDir}
                    className="ml-1 px-2 py-0.5 text-xs font-medium text-gray-600 bg-gray-100 rounded hover:bg-gray-200"
                  >
                    {t.edit}
                  </button>
                </>
              ) : (
                <>
                  <input
                    type="text"
                    value={tempWorkingDir}
                    onChange={(e) => setTempWorkingDir(e.target.value)}
                    className="flex-1 max-w-md px-2 py-0.5 font-mono text-xs border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-cyan-400"
                    placeholder={t.workingDirectory}
                  />
                  <button
                    onClick={handleChooseFolder}
                    className="px-2 py-0.5 text-xs font-medium text-gray-600 bg-gray-100 rounded hover:bg-gray-200"
                    title={t.chooseFolder}
                  >
                    {t.browse}
                  </button>
                  <button
                    onClick={handleSaveWorkingDir}
                    className="px-2 py-0.5 text-xs font-medium text-white bg-cyan-500 rounded hover:bg-cyan-600"
                  >
                    {t.save}
                  </button>
                  <button
                    onClick={handleCancelEditWorkingDir}
                    className="px-2 py-0.5 text-xs font-medium text-gray-600 bg-gray-100 rounded hover:bg-gray-200"
                  >
                    {t.cancel}
                  </button>
                </>
              )}
            </div>

            {/* Config File Path */}
            <div className="flex items-center gap-2 text-xs">
              <svg className="w-3.5 h-3.5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              <span className="text-gray-500">{t.configFilePath}:</span>

              {!isEditingConfigPath ? (
                <>
                  <span className="font-mono text-gray-700">{absoluteConfigPath || configPath}</span>
                  <button
                    onClick={handleStartEditConfigPath}
                    className="ml-1 px-2 py-0.5 text-xs font-medium text-gray-600 bg-gray-100 rounded hover:bg-gray-200"
                  >
                    {t.edit}
                  </button>
                </>
              ) : (
                <>
                  <input
                    type="text"
                    value={tempConfigPath}
                    onChange={(e) => setTempConfigPath(e.target.value)}
                    className="flex-1 max-w-md px-2 py-0.5 font-mono text-xs border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-cyan-400"
                    autoFocus
                    placeholder="config.yaml"
                  />
                  <button
                    onClick={handleChooseFile}
                    className="px-2 py-0.5 text-xs font-medium text-gray-600 bg-gray-100 rounded hover:bg-gray-200"
                    title={t.chooseFile}
                  >
                    {t.browse}
                  </button>
                  <button
                    onClick={handleSaveConfigPath}
                    className="px-2 py-0.5 text-xs font-medium text-white bg-cyan-500 rounded hover:bg-cyan-600"
                  >
                    {t.save}
                  </button>
                  <button
                    onClick={handleCancelEditConfigPath}
                    className="px-2 py-0.5 text-xs font-medium text-gray-600 bg-gray-100 rounded hover:bg-gray-200"
                  >
                    {t.cancel}
                  </button>
                </>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 max-w-7xl mx-auto w-full">
        {activeTab === 'dashboard' && (
          <Dashboard configPath={configPath} language={language} />
        )}
        {activeTab === 'config' && (
          <ConfigEditor configPath={configPath} onConfigPathChange={setConfigPath} language={language} />
        )}
        {activeTab === 'monitor' && <Logs language={language} />}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-100 py-4">
        <div className="max-w-7xl mx-auto px-6 text-center text-sm text-gray-400">
          {t.footerText} • {host}
        </div>
      </footer>
    </div>
  );
}

export default App;
