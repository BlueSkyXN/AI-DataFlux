import { useState, useEffect, useCallback, useRef } from 'react';
import { fetchConfig, saveConfig, validateConfig } from '../api';
import { getTranslations, type Language } from '../i18n';

interface ConfigEditorProps {
  configPath: string;
  onConfigPathChange: (path: string) => void;
  language: Language;
}

export default function ConfigEditor({ configPath, onConfigPathChange, language }: ConfigEditorProps) {
  const t = getTranslations(language);
  const [content, setContent] = useState('');
  const [originalContent, setOriginalContent] = useState('');
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const lastLoadedPathRef = useRef<string>('');

  const hasChanges = content !== originalContent;

  const loadConfig = useCallback(async (path: string) => {
    if (!path) return;

    setLoading(true);
    setError(null);
    try {
      const data = await fetchConfig(path);
      setContent(data.content);
      setOriginalContent(data.content);
      lastLoadedPathRef.current = path;
    } catch (err) {
      setError(`${t.failedToLoad}: ${err}`);
    } finally {
      setLoading(false);
    }
  }, [t.failedToLoad]);

  useEffect(() => {
    if (!configPath) return;
    if (configPath === lastLoadedPathRef.current) return;

    // Debounce auto-load to avoid firing on every keystroke while editing path.
    const timer = setTimeout(() => {
      if (configPath === lastLoadedPathRef.current) return;

      if (lastLoadedPathRef.current && hasChanges) {
        const ok = window.confirm(
          t.discardAndLoadNew
        );
        if (!ok) {
          onConfigPathChange(lastLoadedPathRef.current);
          return;
        }
      }

      void loadConfig(configPath);
    }, 400);

    return () => clearTimeout(timer);
  }, [configPath, hasChanges, loadConfig, onConfigPathChange]);

  const handleSave = async () => {
    if (!configPath) return;
    
    setSaving(true);
    setError(null);
    setSuccess(null);
    try {
      const result = await saveConfig(configPath, content);
      setOriginalContent(content);
      setSuccess(result.backed_up ? t.savedWithBackup : t.saved);
      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      setError(`${t.failedToSave}: ${err}`);
    } finally {
      setSaving(false);
    }
  };
  const handleReload = async () => {
    if (!configPath) return;
    if (hasChanges) {
      const ok = window.confirm(t.discardAndReload);
      if (!ok) return;
    }
    await loadConfig(configPath);
  };

  const validateYaml = async () => {
    // Quick check: YAML doesn't allow tabs for indentation
    if (content.includes('\t')) {
      setError(t.yamlNoTabs);
      setSuccess(null);
      return;
    }

    setError(null);
    setSuccess(null);
    try {
      const result = await validateConfig(content);
      if (result.valid) {
        setSuccess(t.yamlSyntaxValid);
        setTimeout(() => setSuccess(null), 3000);
      } else {
        setError(result.error ? `${t.yamlSyntaxError}: ${result.error}` : t.yamlSyntaxError);
      }
    } catch (err) {
      setError(`${t.failedToValidate}: ${err}`);
    }
  };

  return (
    <div className="p-6 h-full flex flex-col">
      {/* Toolbar */}
      <div className="bg-white rounded-2xl p-4 shadow-[0_2px_12px_rgba(0,0,0,0.04)] mb-4">
        <div className="flex items-center gap-4">
          <div className="flex-1">
            <label className="block text-sm font-medium text-gray-600 mb-1">
              {t.configFile}
            </label>
            <input
              type="text"
              value={configPath}
              onChange={(e) => onConfigPathChange(e.target.value)}
              className="w-full px-3 py-2 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-cyan-400 focus:border-transparent text-sm"
              placeholder="config.yaml"
            />
          </div>
          <div className="flex gap-2 items-end">
            <button
              onClick={handleReload}
              disabled={loading}
              className="px-4 py-2 text-sm font-medium text-gray-600 bg-gray-100 rounded-lg hover:bg-gray-200 disabled:opacity-50"
            >
              {loading ? t.loading : t.reload}
            </button>
            <button
              onClick={validateYaml}
              disabled={loading}
              className="px-4 py-2 text-sm font-medium text-gray-600 bg-gray-100 rounded-lg hover:bg-gray-200"
            >
              {t.validate}
            </button>
            <button
              onClick={handleSave}
              disabled={saving || !hasChanges}
              className="px-4 py-2 text-sm font-medium text-white bg-gradient-to-r from-cyan-400 to-blue-500 rounded-lg hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {saving ? t.saving : t.save}
            </button>
          </div>
        </div>
        {hasChanges && (
          <div className="mt-2 text-sm text-amber-600">
            {t.unsavedChanges}
          </div>
        )}
      </div>

      {/* Messages */}
      {error && (
        <div className="bg-red-50 text-red-600 px-4 py-3 rounded-lg mb-4">
          {error}
        </div>
      )}
      {success && (
        <div className="bg-green-50 text-green-600 px-4 py-3 rounded-lg mb-4">
          {success}
        </div>
      )}

      {/* Editor */}
      <div className="flex-1 bg-white rounded-2xl shadow-[0_2px_12px_rgba(0,0,0,0.04)] overflow-hidden">
        <textarea
          value={content}
          onChange={(e) => setContent(e.target.value)}
          className="w-full h-full p-4 font-mono text-sm resize-none focus:outline-none bg-slate-50"
          style={{ fontFamily: "'JetBrains Mono', 'Menlo', 'Monaco', monospace" }}
          placeholder={loading ? t.loading : t.enterYamlConfig}
          spellCheck={false}
        />
      </div>
    </div>
  );
}
