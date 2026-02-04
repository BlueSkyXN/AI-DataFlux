import { useState } from 'react';
import type { TabType } from './types';
import Dashboard from './pages/Dashboard';
import ConfigEditor from './pages/ConfigEditor';
import Logs from './pages/Logs';

function App() {
  const [activeTab, setActiveTab] = useState<TabType>('dashboard');
  const [configPath, setConfigPath] = useState('config.yaml');

  const tabs: { id: TabType; label: string }[] = [
    { id: 'dashboard', label: 'Dashboard' },
    { id: 'config', label: 'Config' },
    { id: 'logs', label: 'Logs' },
  ];

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
              <span className="text-xs text-gray-400 bg-gray-100 px-2 py-0.5 rounded-full">Control Panel</span>
            </div>

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
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 max-w-7xl mx-auto w-full">
        {activeTab === 'dashboard' && (
          <Dashboard configPath={configPath} onConfigPathChange={setConfigPath} />
        )}
        {activeTab === 'config' && (
          <ConfigEditor configPath={configPath} onConfigPathChange={setConfigPath} />
        )}
        {activeTab === 'logs' && <Logs />}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-100 py-4">
        <div className="max-w-7xl mx-auto px-6 text-center text-sm text-gray-400">
          AI-DataFlux Control Panel • Listening on 127.0.0.1:8790
        </div>
      </footer>
    </div>
  );
}

export default App;

