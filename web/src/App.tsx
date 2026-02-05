import { useState, useEffect } from 'react';
import type { TabType } from './types';
import Dashboard from './pages/Dashboard';
import ConfigEditor from './pages/ConfigEditor';
import Logs from './pages/Logs';
import { getInitialLanguage, saveLanguagePreference, getTranslations, type Language } from './i18n';
import { fetchStatus } from './api';

function App() {
  const [activeTab, setActiveTab] = useState<TabType>('dashboard');
  const [configPath, setConfigPath] = useState('config.yaml');
  const [language, setLanguage] = useState<Language>(getInitialLanguage());
  const [workingDir, setWorkingDir] = useState<string>('');
  const host = window.location.host;

  const t = getTranslations(language);

  const tabs: { id: TabType; label: string }[] = [
    { id: 'dashboard', label: t.dashboard },
    { id: 'config', label: t.config },
    { id: 'logs', label: t.logs },
  ];

  // Fetch working directory from status
  useEffect(() => {
    const getWorkingDir = async () => {
      try {
        const status = await fetchStatus();
        if (status.working_directory) {
          setWorkingDir(status.working_directory);
        }
      } catch {
        // Ignore error
      }
    };
    getWorkingDir();
  }, []);

  const handleLanguageChange = (lang: Language) => {
    setLanguage(lang);
    saveLanguagePreference(lang);
  };

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

          {/* Working Directory */}
          {workingDir && (
            <div className="mt-2 text-xs text-gray-500 flex items-center gap-2">
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
              </svg>
              <span>{t.workingDirectory}:</span>
              <span className="font-mono">{workingDir}</span>
            </div>
          )}
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
