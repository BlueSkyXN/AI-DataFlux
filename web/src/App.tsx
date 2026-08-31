import { useCallback, useEffect, useState } from 'react';

import { fetchHealth } from './api';
import WorkspacePicker from './components/WorkspacePicker';
import { getInitialLanguage, getTranslations, saveLanguagePreference } from './i18n';
import type { Language } from './i18n';
import ConfigEditor from './pages/ConfigEditor';
import Dashboard from './pages/Dashboard';
import Jobs from './pages/Jobs';
import type { TabType, WorkspaceSelection } from './types';

export default function App() {
  const [activeTab, setActiveTab] = useState<TabType>('dashboard');
  const [language, setLanguage] = useState<Language>(getInitialLanguage());
  const [selection, setSelection] = useState<WorkspaceSelection>({
    rootId: 'project',
    relativePath: 'config.yaml',
  });
  const [controllerConnected, setControllerConnected] = useState(false);
  const [version, setVersion] = useState('');
  const t = getTranslations(language);

  const updateSelection = useCallback((next: WorkspaceSelection) => {
    setSelection(next);
  }, []);

  useEffect(() => {
    let active = true;
    const check = async () => {
      try {
        const health = await fetchHealth();
        if (!active) return;
        setControllerConnected(health.status === 'ok' || health.status === 'healthy');
        setVersion(health.version);
      } catch {
        if (active) setControllerConnected(false);
      }
    };
    void check();
    const timer = window.setInterval(check, 5000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

  const tabs: Array<{ id: TabType; label: string }> = [
    { id: 'dashboard', label: t.dashboard },
    { id: 'jobs', label: language === 'zh' ? '任务' : 'Jobs' },
    { id: 'config', label: t.config },
  ];

  const handleLanguageChange = (next: Language) => {
    setLanguage(next);
    saveLanguagePreference(next);
  };

  return (
    <div className="flex min-h-screen flex-col bg-[#F8FAFB]">
      <header className="bg-white shadow-[0_1px_3px_rgba(0,0,0,0.05)]">
        <div className="mx-auto max-w-7xl px-4 py-4 sm:px-6">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex min-w-0 items-center gap-3">
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-cyan-400 to-blue-500">
                <span className="font-bold text-white" aria-hidden="true">↯</span>
              </div>
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <h1 className="font-semibold text-gray-800">AI-DataFlux</h1>
                  <span className="rounded-full bg-gray-100 px-2 py-0.5 text-xs text-gray-500">
                    {version || '3.2.0'}
                  </span>
                </div>
                <div className="mt-0.5 flex items-center gap-1.5 text-xs text-gray-500">
                  <span className={`h-2 w-2 rounded-full ${controllerConnected ? 'bg-green-400' : 'bg-red-400'}`} />
                  {controllerConnected ? t.controllerConnected : t.controllerDisconnected}
                </div>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-3">
              <nav aria-label="Primary" className="flex gap-1 rounded-xl bg-gray-50 p-1">
                {tabs.map((tab) => (
                  <button
                    type="button"
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    aria-current={activeTab === tab.id ? 'page' : undefined}
                    className={`rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${
                      activeTab === tab.id
                        ? 'bg-white text-cyan-600 shadow-sm'
                        : 'text-gray-600 hover:text-gray-900'
                    }`}
                  >
                    {tab.label}
                  </button>
                ))}
              </nav>
              <div className="flex rounded-lg bg-gray-100 p-1">
                {(['en', 'zh'] as const).map((item) => (
                  <button
                    type="button"
                    key={item}
                    onClick={() => handleLanguageChange(item)}
                    className={`rounded-md px-2.5 py-1 text-xs ${
                      language === item ? 'bg-white text-gray-800 shadow-sm' : 'text-gray-500'
                    }`}
                  >
                    {item === 'en' ? 'EN' : '中文'}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="mt-3 border-t border-gray-100 pt-3">
            <WorkspacePicker selection={selection} onSelect={updateSelection} language={language} />
          </div>
        </div>
      </header>

      <main className="mx-auto w-full max-w-7xl flex-1">
        {activeTab === 'dashboard' && <Dashboard selection={selection} language={language} />}
        {activeTab === 'jobs' && <Jobs selection={selection} language={language} />}
        {activeTab === 'config' && (
          <ConfigEditor
            selection={selection}
            onSelectionChange={updateSelection}
            language={language}
          />
        )}
      </main>

      <footer className="border-t border-gray-100 bg-white py-4 text-center text-sm text-gray-400">
        {t.footerText} · {window.location.host}
      </footer>
    </div>
  );
}
