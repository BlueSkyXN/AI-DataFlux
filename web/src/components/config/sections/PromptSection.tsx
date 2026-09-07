/**
 * Prompt 提示词配置分区组件
 *
 * 用途：配置 AI 请求的提示词相关参数，包括必填字段列表、温度、
 *       JSON Schema 开关、系统提示词、用户提示模板
 *
 * 导出：PromptSection（默认导出）
 *   Props: SectionProps
 */
import type { SectionProps } from '../SectionRenderer';
import { getTranslations } from '../../../i18n';
import SectionCard from '../shared/SectionCard';
import FormField from '../shared/FormField';
import NumberInput from '../shared/NumberInput';
import ToggleSwitch from '../shared/ToggleSwitch';
import TextareaField from '../shared/TextareaField';
import StringListEditor from '../shared/StringListEditor';
import SelectDropdown from '../shared/SelectDropdown';
import TextInput from '../shared/TextInput';

/**
 * 提示词配置组件
 * 包含基本设置、系统提示词、模板三个卡片区域
 */
export default function PromptSection({ updateConfig, getConfig, language }: SectionProps) {
  const t = getTranslations(language);

  const requiredFields = (getConfig(['job', 'prompt', 'required_fields']) as string[]) ?? [];
  const useJsonSchema = (getConfig(['job', 'prompt', 'use_json_schema']) as boolean) ?? false;
  const temperature = (getConfig(['job', 'prompt', 'temperature']) as number) ?? 0.7;
  const temperatureOverride = (getConfig(['job', 'prompt', 'temperature_override']) as boolean) ?? true;
  const systemPrompt = (getConfig(['job', 'prompt', 'system_prompt']) as string) ?? '';
  const template = (getConfig(['job', 'prompt', 'template']) as string) ?? '';
  const selection = (getConfig(['job', 'model_selection']) as { mode: string; route_id?: string; group?: string }) ?? { mode: 'auto' };
  const zh = language === 'zh';

  return (
    <div className="space-y-4">
      <SectionCard title={zh ? '模型选择' : 'Model selection'}
        description={zh ? 'strict 只使用指定 route；fallback_group 只按已配置组的顺序切换。' : 'Strict stays on one route; fallback follows only the configured group order.'}>
        <FormField label={zh ? '路由模式' : 'Routing mode'}>
          <SelectDropdown ariaLabel="Routing mode" value={selection.mode}
            options={['auto', 'strict', 'fallback_group'].map((mode) => ({ value: mode, label: mode }))}
            onChange={(mode) => updateConfig(['job', 'model_selection'], mode === 'strict'
              ? { mode, route_id: '' } : mode === 'fallback_group' ? { mode, group: '' } : { mode })} />
        </FormField>
        {selection.mode === 'strict' && <FormField label="Route ID" required>
          <TextInput value={selection.route_id ?? ''} placeholder="route-a" monospace
            onChange={(route_id) => updateConfig(['job', 'model_selection'], { mode: 'strict', route_id })} />
        </FormField>}
        {selection.mode === 'fallback_group' && <FormField label="Fallback group" required
          description={zh ? '填写 gateway.fallback_groups 中的组名，可在 YAML 视图编辑组成员。' : 'Use a group from gateway.fallback_groups; edit its members in the YAML view.'}>
          <TextInput value={selection.group ?? ''} placeholder="primary" monospace
            onChange={(group) => updateConfig(['job', 'model_selection'], { mode: 'fallback_group', group })} />
        </FormField>}
      </SectionCard>
      {/* Basic Settings */}
      {/* 基本设置：必填字段、温度、开关选项 */}
      <SectionCard title={t.cfgPromptSettings}>
        <FormField label={t.cfgRequiredFields} description={t.cfgRequiredFieldsDesc}>
          <StringListEditor
            value={requiredFields}
            onChange={(v) => updateConfig(['job', 'prompt', 'required_fields'], v)}
            placeholder={t.cfgAddField}
            addLabel={t.cfgAdd}
          />
        </FormField>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <FormField label={t.cfgTemperature}>
            <NumberInput
              value={temperature}
              onChange={(v) => updateConfig(['job', 'prompt', 'temperature'], v)}
              min={0} max={2} step={0.1}
            />
          </FormField>
        </div>
        <div className="flex gap-6">
          <FormField label="JSON Schema" horizontal>
            <ToggleSwitch
              checked={useJsonSchema}
              onChange={(v) => updateConfig(['job', 'prompt', 'use_json_schema'], v)}
            />
          </FormField>
          <FormField label={t.cfgTempOverride} horizontal>
            <ToggleSwitch
              checked={temperatureOverride}
              onChange={(v) => updateConfig(['job', 'prompt', 'temperature_override'], v)}
            />
          </FormField>
        </div>
      </SectionCard>

      {/* System Prompt */}
      {/* 系统提示词：设定 AI 的角色和行为 */}
      <SectionCard title={t.cfgSystemPrompt}>
        <TextareaField
          value={systemPrompt}
          onChange={(v) => updateConfig(['job', 'prompt', 'system_prompt'], v)}
          rows={6}
          monospace={false}
          placeholder={t.cfgSystemPromptPlaceholder}
        />
      </SectionCard>

      {/* Template */}
      {/* 用户模板：使用 {record_json} 等占位符构建提示词 */}
      <SectionCard title={t.cfgTemplate} description={t.cfgTemplateDesc}>
        <TextareaField
          value={template}
          onChange={(v) => updateConfig(['job', 'prompt', 'template'], v)}
          rows={12}
          monospace={false}
          placeholder={t.cfgTemplatePlaceholder}
        />
      </SectionCard>
    </div>
  );
}
