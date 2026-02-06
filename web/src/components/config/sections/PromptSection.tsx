import type { SectionProps } from '../SectionRenderer';
import { getTranslations } from '../../../i18n';
import SectionCard from '../shared/SectionCard';
import FormField from '../shared/FormField';
import NumberInput from '../shared/NumberInput';
import ToggleSwitch from '../shared/ToggleSwitch';
import TextareaField from '../shared/TextareaField';
import StringListEditor from '../shared/StringListEditor';

export default function PromptSection({ updateConfig, getConfig, language }: SectionProps) {
  const t = getTranslations(language);

  const requiredFields = (getConfig(['prompt', 'required_fields']) as string[]) ?? [];
  const useJsonSchema = (getConfig(['prompt', 'use_json_schema']) as boolean) ?? false;
  const temperature = (getConfig(['prompt', 'temperature']) as number) ?? 0.7;
  const temperatureOverride = (getConfig(['prompt', 'temperature_override']) as boolean) ?? true;
  const systemPrompt = (getConfig(['prompt', 'system_prompt']) as string) ?? '';
  const template = (getConfig(['prompt', 'template']) as string) ?? '';

  return (
    <div className="space-y-4">
      {/* Basic Settings */}
      <SectionCard title={t.cfgPromptSettings}>
        <FormField label={t.cfgRequiredFields} description={t.cfgRequiredFieldsDesc}>
          <StringListEditor
            value={requiredFields}
            onChange={(v) => updateConfig(['prompt', 'required_fields'], v)}
            placeholder={t.cfgAddField}
            addLabel={t.cfgAdd}
          />
        </FormField>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <FormField label={t.cfgTemperature}>
            <NumberInput
              value={temperature}
              onChange={(v) => updateConfig(['prompt', 'temperature'], v)}
              min={0} max={2} step={0.1}
            />
          </FormField>
        </div>
        <div className="flex gap-6">
          <FormField label="JSON Schema" horizontal>
            <ToggleSwitch
              checked={useJsonSchema}
              onChange={(v) => updateConfig(['prompt', 'use_json_schema'], v)}
            />
          </FormField>
          <FormField label={t.cfgTempOverride} horizontal>
            <ToggleSwitch
              checked={temperatureOverride}
              onChange={(v) => updateConfig(['prompt', 'temperature_override'], v)}
            />
          </FormField>
        </div>
      </SectionCard>

      {/* System Prompt */}
      <SectionCard title={t.cfgSystemPrompt}>
        <TextareaField
          value={systemPrompt}
          onChange={(v) => updateConfig(['prompt', 'system_prompt'], v)}
          rows={6}
          monospace={false}
          placeholder={t.cfgSystemPromptPlaceholder}
        />
      </SectionCard>

      {/* Template */}
      <SectionCard title={t.cfgTemplate} description={t.cfgTemplateDesc}>
        <TextareaField
          value={template}
          onChange={(v) => updateConfig(['prompt', 'template'], v)}
          rows={12}
          monospace={false}
          placeholder={t.cfgTemplatePlaceholder}
        />
      </SectionCard>
    </div>
  );
}
