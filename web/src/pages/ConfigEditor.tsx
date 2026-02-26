/**
 * 配置编辑页面组件
 *
 * 功能职责：
 * - 支持可视化表单编辑和原始 YAML 两种模式
 * - 自动加载/保存/验证 YAML 配置文件
 * - 编辑变更检测，带未保存提醒
 * - 侧边栏导航各配置分区（Global / Datasource / Models 等）
 *
 * 导出：默认导出 ConfigEditor 组件
 *
 * Props：
 * - configPath — 配置文件路径
 * - onConfigPathChange — 路径变更回调
 * - language — 当前界面语言
 *
 * 依赖模块：
 * - js-yaml — YAML 解析与序列化
 * - api — fetchConfig / saveConfig / validateConfig
 * - i18n — 国际化翻译
 * - components/config/* — 侧边栏、分区渲染器、原始编辑器子组件
 */
import { useState, useEffect, useCallback, useRef } from 'react';

import yaml from 'js-yaml';
import { fetchConfig, saveConfig, validateConfig } from '../api';
import { getTranslations, type Language } from '../i18n';
import ConfigSidebar, { type ConfigSectionId } from '../components/config/ConfigSidebar';
import SectionRenderer from '../components/config/SectionRenderer';
import RawYamlEditor from '../components/config/RawYamlEditor';

/** ConfigEditor 组件的 Props 类型 */
interface ConfigEditorProps {
  /** 配置文件路径 */
  configPath: string;
  /** 路径变更回调（通知父组件更新） */
  onConfigPathChange: (path: string) => void;
  /** 当前界面语言 */
  language: Language;
}

/**
 * 解析 YAML 字符串为 JavaScript 对象
 * @param content - YAML 字符串
 * @returns 解析后的配置对象，解析失败返回空对象
 */
function parseYaml(content: string): Record<string, unknown> {
  const result = yaml.load(content);
  if (typeof result !== 'object' || result === null) {
    return {};
  }
  return result as Record<string, unknown>;
}

/**
 * 将 JavaScript 对象序列化为 YAML 字符串
 * @param data - 配置数据对象
 * @returns 格式化的 YAML 字符串（2 空格缩进，120 字符行宽）
 */
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

/**
 * 配置编辑器主组件
 *
 * 支持两种编辑模式：
 * 1. 可视化模式 — 通过侧边栏分区表单编辑各配置项
 * 2. 原始 YAML 模式 — 直接编辑 YAML 文本
 *
 * 模式切换时自动同步数据（可视化 ↔ 原始 YAML 互转）。
 */
export default function ConfigEditor({ configPath, onConfigPathChange, language }: ConfigEditorProps) {
  const t = getTranslations(language);

  // Editor mode: visual sections or raw YAML
  // 编辑器模式：可视化分区 或 原始 YAML
  const [activeSection, setActiveSection] = useState<ConfigSectionId>('global');

  // Raw YAML state — 原始 YAML 编辑器状态
  const [rawContent, setRawContent] = useState('');
  /** 上次从服务端加载的原始内容（用于变更检测） */
  const [originalRawContent, setOriginalRawContent] = useState('');

  // Parsed form data state — 解析后的表单数据状态
  const [formData, setFormData] = useState<Record<string, unknown>>({});
  /** 上次从服务端加载的表单数据（用于变更检测） */
  const [originalFormData, setOriginalFormData] = useState<Record<string, unknown>>({});

  // UI state — UI 加载/保存状态
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  /** 最近一次成功加载的配置路径（防止重复加载） */
  const lastLoadedPathRef = useRef<string>('');

  /** 是否处于原始 YAML 编辑模式 */
  const isRawMode = activeSection === 'raw';

  // Change detection — 变更检测：比较当前值与原始值
  const hasChanges = isRawMode
    ? rawContent !== originalRawContent
    : JSON.stringify(formData) !== JSON.stringify(originalFormData);

  // --- Config read/write helpers --- 配置读写辅助函数
  /**
   * 按路径读取配置值
   * @param path - 键路径数组，如 ['global', 'api_gateway']
   * @returns 对应路径的值，不存在时返回 undefined
   */
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

  /**
   * 按路径更新配置值（深拷贝后修改，保证不可变性）
   * @param path - 键路径数组
   * @param value - 新值
   */
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

  // --- Load config --- 从服务端加载配置文件
  /**
   * 加载指定路径的配置文件
   * 同时更新原始 YAML 和解析后的表单数据。YAML 解析失败时自动切换到原始模式。
   */
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

  // Auto-load on path change — 配置路径变化时自动加载（带 400ms 防抖）
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

  // --- Section switching: sync data between modes --- 模式切换时同步数据
  /**
   * 处理侧边栏分区切换
   * - 从原始模式切换到可视化：解析 YAML → 表单数据
   * - 从可视化切换到原始模式：序列化表单数据 → YAML
   */
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

  // --- Save --- 保存配置到服务端
  /** 保存当前配置内容（根据模式选择原始 YAML 或序列化表单数据） */
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

  // --- Reload --- 重新从磁盘加载配置
  /** 重新加载配置（有未保存更改时弹出确认对话框） */
  const handleReload = async () => {
    if (!configPath) return;
    if (hasChanges) {
      const ok = window.confirm(t.discardAndReload);
      if (!ok) return;
    }
    await loadConfig(configPath);
  };

  // --- Validate --- 验证 YAML 语法
  /** 调用服务端验证当前 YAML 内容的合法性 */
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
