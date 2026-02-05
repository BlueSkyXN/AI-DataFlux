import { useState, useEffect, useCallback, useRef } from 'react';
import yaml from 'js-yaml';
import { fetchConfig, saveConfig, validateConfig } from '../api';
import { getTranslations, type Language } from '../i18n';
import ConfigSidebar, { type ConfigSectionId } from '../components/config/ConfigSidebar';
import SectionRenderer from '../components/config/SectionRenderer';
import RawYamlEditor from '../components/config/RawYamlEditor';

interface ConfigEditorProps {
  configPath: string;
  onConfigPathChange: (path: string) => void;
  language: Language;
}

function parseYaml(content: string): Record<string, unknown> {
  const result = yaml.load(content);
  if (typeof result !== 'object' || result === null) {
    return {};
  }
  return result as Record<string, unknown>;
}

function serializeYaml(data: Record<string, unknown>): string {
  return yaml.dump(data, {
    indent: 2,
    lineWidth: 120,
    noRefs: true,
    sortKeys: false,
    quotingType: '"',
    forceQuotes: false,
  });
}

export default function ConfigEditor({ configPath, onConfigPathChange, language }: ConfigEditorProps) {
  const t = getTranslations(language);

  // Editor mode: visual sections or raw YAML
  const [activeSection, setActiveSection] = useState<ConfigSectionId>('global');

  // Raw YAML state
  const [rawContent, setRawContent] = useState('');
  const [originalRawContent, setOriginalRawContent] = useState('');

  // Parsed form data state
  const [formData, setFormData] = useState<Record<string, unknown>>({});
  const [originalFormData, setOriginalFormData] = useState<Record<string, unknown>>({});

  // UI state
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const lastLoadedPathRef = useRef<string>('');

  const isRawMode = activeSection === 'raw';

  // Change detection
  const hasChanges = isRawMode
    ? rawContent !== originalRawContent
    : JSON.stringify(formData) !== JSON.stringify(originalFormData);

  // --- Config read/write helpers ---
  const getConfig = useCallback((path: string[]): unknown => {
    let target: unknown = formData;
    for (const key of path) {
      if (target && typeof target === 'object' && key in (target as Record<string, unknown>)) {
        target = (target as Record<string, unknown>)[key];
      } else {
        return undefined;
      }
    }
    return target;
  }, [formData]);

  const updateConfig = useCallback((path: string[], value: unknown) => {
    setFormData(prev => {
      const next = structuredClone(prev);
      let target = next as Record<string, unknown>;
      for (let i = 0; i < path.length - 1; i++) {
        if (target[path[i]] === undefined || target[path[i]] === null || typeof target[path[i]] !== 'object') {
          target[path[i]] = {};
        }
        target = target[path[i]] as Record<string, unknown>;
      }
      target[path[path.length - 1]] = value;
      return next;
    });
  }, []);

  // --- Load config ---
  const loadConfig = useCallback(async (path: string) => {
    if (!path) return;
    setLoading(true);
    setError(null);
    try {
      const data = await fetchConfig(path);
      const content = data.content;
      setRawContent(content);
      setOriginalRawContent(content);

      try {
        const parsed = parseYaml(content);
        setFormData(parsed);
        setOriginalFormData(structuredClone(parsed));
      } catch {
        // YAML parse failed — start in raw mode
        setFormData({});
        setOriginalFormData({});
        if (activeSection !== 'raw') {
          setActiveSection('raw');
          setError(t.yamlParseError);
        }
      }

      lastLoadedPathRef.current = path;
    } catch (err) {
      setError(`${t.failedToLoad}: ${err}`);
    } finally {
      setLoading(false);
    }
  }, [t.failedToLoad, t.yamlParseError, activeSection]);

  // Auto-load on path change
  useEffect(() => {
    if (!configPath) return;
    if (configPath === lastLoadedPathRef.current) return;

    const timer = setTimeout(() => {
      if (configPath === lastLoadedPathRef.current) return;

      if (lastLoadedPathRef.current && hasChanges) {
        const ok = window.confirm(t.discardAndLoadNew);
        if (!ok) {
          onConfigPathChange(lastLoadedPathRef.current);
          return;
        }
      }

      void loadConfig(configPath);
    }, 400);

    return () => clearTimeout(timer);
  }, [configPath, hasChanges, loadConfig, onConfigPathChange, t.discardAndLoadNew]);

  // --- Section switching: sync data between modes ---
  const handleSectionChange = useCallback((section: ConfigSectionId) => {
    if (section === activeSection) return;

    // Switching FROM raw to visual: parse raw content
    if (activeSection === 'raw' && section !== 'raw') {
      try {
        const parsed = parseYaml(rawContent);
        setFormData(parsed);
      } catch {
        setError(t.yamlParseError);
        return; // Stay in raw mode
      }
    }

    // Switching FROM visual to raw: serialize form data
    if (activeSection !== 'raw' && section === 'raw') {
      const serialized = serializeYaml(formData);
      setRawContent(serialized);
    }

    setError(null);
    setActiveSection(section);
  }, [activeSection, rawContent, formData, t.yamlParseError]);

  // --- Save ---
  const handleSave = async () => {
    if (!configPath) return;

    let contentToSave: string;
    if (isRawMode) {
      contentToSave = rawContent;
    } else {
      contentToSave = serializeYaml(formData);
    }

    setSaving(true);
    setError(null);
    setSuccess(null);
    try {
      const result = await saveConfig(configPath, contentToSave);

      // Update both states to reflect saved content
      setOriginalRawContent(contentToSave);
      setRawContent(contentToSave);
      try {
        const parsed = parseYaml(contentToSave);
        setOriginalFormData(structuredClone(parsed));
        setFormData(parsed);
      } catch {
        // If serialized YAML can't be re-parsed, just keep current form data as original
        setOriginalFormData(structuredClone(formData));
      }

      setSuccess(result.backed_up ? t.savedWithBackup : t.saved);
      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      setError(`${t.failedToSave}: ${err}`);
    } finally {
      setSaving(false);
    }
  };

  // --- Reload ---
  const handleReload = async () => {
    if (!configPath) return;
    if (hasChanges) {
      const ok = window.confirm(t.discardAndReload);
      if (!ok) return;
    }
    await loadConfig(configPath);
  };

  // --- Validate ---
  const validateYaml = async () => {
    let contentToValidate: string;
    if (isRawMode) {
      if (rawContent.includes('\t')) {
        setError(t.yamlNoTabs);
        setSuccess(null);
        return;
      }
      contentToValidate = rawContent;
    } else {
      contentToValidate = serializeYaml(formData);
    }

    setError(null);
    setSuccess(null);
    try {
      const result = await validateConfig(contentToValidate);
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
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            {hasChanges && (
              <span className="text-sm text-amber-600">{t.unsavedChanges}</span>
            )}
            {!isRawMode && hasChanges && (
              <span className="text-xs text-gray-400">({t.commentLossWarning})</span>
            )}
          </div>
          <div className="flex gap-2">
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
      </div>

      {/* Messages */}
      {error && (
        <div className="bg-red-50 text-red-600 px-4 py-3 rounded-lg mb-4 text-sm">
          {error}
        </div>
      )}
      {success && (
        <div className="bg-green-50 text-green-600 px-4 py-3 rounded-lg mb-4 text-sm">
          {success}
        </div>
      )}

      {/* Main area: sidebar + content */}
      <div className="flex-1 flex gap-4 min-h-0">
        {/* Sidebar */}
        <div className="w-44 shrink-0">
          <ConfigSidebar
            activeSection={activeSection}
            onSelect={handleSectionChange}
            language={language}
          />
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto min-h-0">
          {isRawMode ? (
            <RawYamlEditor
              value={rawContent}
              onChange={setRawContent}
              loading={loading}
              placeholder={t.enterYamlConfig}
            />
          ) : (
            <SectionRenderer
              section={activeSection}
              formData={formData}
              updateConfig={updateConfig}
              getConfig={getConfig}
              language={language}
            />
          )}
        </div>
      </div>
    </div>
  );
}
