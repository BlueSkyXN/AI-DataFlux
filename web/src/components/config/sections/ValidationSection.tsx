/**
 * 字段校验规则配置分区组件
 *
 * 用途：配置输出字段的枚举校验规则，启用后可为每个字段
 *       定义允许的值列表，AI 返回的值不在列表中则视为错误
 *
 * 导出：ValidationSection（默认导出）
 *   Props: SectionProps
 */
import { useState } from 'react';
import type { SectionProps } from '../SectionRenderer';
import { getTranslations } from '../../../i18n';
import SectionCard from '../shared/SectionCard';
import FormField from '../shared/FormField';
import ToggleSwitch from '../shared/ToggleSwitch';
import StringListEditor from '../shared/StringListEditor';

/**
 * 字段校验配置组件
 * 提供启用开关及字段规则编辑器（每个字段可配置允许的值列表）
 */
export default function ValidationSection({ updateConfig, getConfig, language }: SectionProps) {
  const t = getTranslations(language);
  const [newFieldName, setNewFieldName] = useState('');

  const enabled = (getConfig(['validation', 'enabled']) as boolean) ?? false;
  const fieldRules = (getConfig(['validation', 'field_rules']) as Record<string, string[]>) ?? {};

  /** 添加新的校验字段 */
  const handleAddField = () => {
    const name = newFieldName.trim();
    if (!name || name in fieldRules) return;
    updateConfig(['validation', 'field_rules'], { ...fieldRules, [name]: [] });
    setNewFieldName('');
  };

  /** 删除指定校验字段及其规则 */
  const handleRemoveField = (fieldName: string) => {
    const next = { ...fieldRules };
    delete next[fieldName];
    updateConfig(['validation', 'field_rules'], next);
  };

  /** 更新指定字段的允许值列表 */
  const handleUpdateValues = (fieldName: string, values: string[]) => {
    updateConfig(['validation', 'field_rules'], { ...fieldRules, [fieldName]: values });
  };

  return (
    <div className="space-y-4">
      <SectionCard title={t.cfgValidationSettings}>
        <FormField label={t.cfgEnableValidation} horizontal>
          <ToggleSwitch
            checked={enabled}
            onChange={(v) => updateConfig(['validation', 'enabled'], v)}
          />
        </FormField>
      </SectionCard>

      {enabled && (
        <SectionCard title={t.cfgFieldRules} description={t.cfgFieldRulesDesc}>
          <div className="space-y-4">
            {Object.entries(fieldRules).map(([fieldName, values]) => (
              <div key={fieldName} className="border border-gray-200 rounded-xl p-4">
                <div className="flex items-center justify-between mb-3">
                  <span className="text-sm font-medium text-gray-700 font-mono">{fieldName}</span>
                  <button
                    type="button"
                    onClick={() => handleRemoveField(fieldName)}
                    className="text-gray-400 hover:text-red-500 p-1"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
                <StringListEditor
                  value={values}
                  onChange={(v) => handleUpdateValues(fieldName, v)}
                  placeholder={t.cfgAddAllowedValue}
                  addLabel={t.cfgAdd}
                />
              </div>
            ))}

            {/* Add new field */}
            {/* 添加新的校验字段 */}
            <div className="flex gap-2">
              <input
                type="text"
                value={newFieldName}
                onChange={(e) => setNewFieldName(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault();
                    handleAddField();
                  }
                }}
                placeholder={t.cfgNewFieldName}
                className="flex-1 px-3 py-1.5 text-sm font-mono border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-cyan-400 focus:border-transparent"
              />
              <button
                type="button"
                onClick={handleAddField}
                disabled={!newFieldName.trim()}
                className="px-4 py-1.5 text-sm font-medium text-cyan-600 bg-cyan-50 rounded-lg hover:bg-cyan-100 disabled:opacity-40 disabled:cursor-not-allowed"
              >
                {t.cfgAddField}
              </button>
            </div>
          </div>
        </SectionCard>
      )}
    </div>
  );
}
