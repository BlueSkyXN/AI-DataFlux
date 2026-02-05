import { useState, useEffect, useCallback } from 'react';
import type { TabType } from './types';
import Dashboard from './pages/Dashboard';
import ConfigEditor from './pages/ConfigEditor';
import Logs from './pages/Logs';
import { getInitialLanguage, saveLanguagePreference, getTranslations, type Language } from './i18n';
import { fetchStatus, fetchConfig } from './api';

function App() {
  const [activeTab, setActiveTab] = useState<TabType>('dashboard');
  const [configPath, setConfigPath] = useState('config.yaml');
  const [absoluteConfigPath, setAbsoluteConfigPath] = useState<string>('');
  const [language, setLanguage] = useState<Language>(getInitialLanguage());
  const [workingDir, setWorkingDir] = useState<string>('');
  const [isEditingConfigPath, setIsEditingConfigPath] = useState(false);
  const [tempConfigPath, setTempConfigPath] = useState('');
  const host = window.location.host;

  const t = getTranslations(language);

  const tabs: { id: TabType; label: string }[] = [
    { id: 'dashboard', label: t.dashboard },
    { id: 'config', label: t.config },
    { id: 'logs', label: t.logs },
  ];

  // Fetch working directory and absolute config path from status
  useEffect(() => {
    const getInitialData = async () => {
      try {
        const status = await fetchStatus();
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
        // Ignore error
      }
    };
    getInitialData();
  }, [configPath]);

  const handleLanguageChange = (lang: Language) => {
    setLanguage(lang);
    saveLanguagePreference(lang);
  };

  const handleStartEditConfigPath = () => {
    setTempConfigPath(configPath);
    setIsEditingConfigPath(true);
  };

  const handleSaveConfigPath = () => {
    setConfigPath(tempConfigPath);
    setIsEditingConfigPath(false);
  };

  const handleCancelEditConfigPath = () => {
    setIsEditingConfigPath(false);
  };

  const handleOpenFolder = useCallback((path: string) => {
    // Try to open folder in file explorer
    // This uses platform-specific protocols
    const isWin = navigator.platform.toLowerCase().includes('win');
    const isMac = navigator.platform.toLowerCase().includes('mac');

    if (isWin) {
      // Windows: use file:// protocol to folder
      window.open(`file:///${path.replace(/\\/g, '/')}`, '_blank');
    } else if (isMac) {
      // macOS: use file:// protocol
      window.open(`file://${path}`, '_blank');
    } else {
      // Linux/other: use file:// protocol
      window.open(`file://${path}`, '_blank');
    }
  }, []);

  const handleOpenFile = useCallback((path: string) => {
    // Try to open file in default application
    window.open(`file://${path}`, '_blank');
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
            {workingDir && (
              <div className="flex items-center gap-2 text-xs">
                <svg className="w-3.5 h-3.5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
                </svg>
                <span className="text-gray-500">{t.workingDirectory}:</span>
                <span className="font-mono text-gray-700">{workingDir}</span>
                <button
                  onClick={() => handleOpenFolder(workingDir)}
                  className="ml-1 px-2 py-0.5 text-xs font-medium text-cyan-600 bg-cyan-50 rounded hover:bg-cyan-100"
                  title={t.openFolder}
                >
                  {t.openFolder}
                </button>
              </div>
            )}

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
                  {absoluteConfigPath && (
                    <button
                      onClick={() => handleOpenFile(absoluteConfigPath)}
                      className="px-2 py-0.5 text-xs font-medium text-cyan-600 bg-cyan-50 rounded hover:bg-cyan-100"
                      title={t.openFile}
                    >
                      {t.openFile}
                    </button>
                  )}
                </>
              ) : (
                <>
                  <input
                    type="text"
                    value={tempConfigPath}
                    onChange={(e) => setTempConfigPath(e.target.value)}
                    className="flex-1 max-w-md px-2 py-0.5 font-mono text-xs border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-cyan-400"
                    autoFocus
                  />
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
          <Dashboard configPath={configPath} onConfigPathChange={setConfigPath} language={language} />
        )}
        {activeTab === 'config' && (
          <ConfigEditor configPath={configPath} onConfigPathChange={setConfigPath} language={language} />
        )}
        {activeTab === 'logs' && <Logs language={language} />}
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
