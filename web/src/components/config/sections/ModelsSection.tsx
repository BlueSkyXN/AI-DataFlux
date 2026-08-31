/**
 * AI 模型配置分区组件
 *
 * 用途：管理 AI 模型列表，支持新增、删除、复制模型条目
 *       每个模型可配置 ID、名称、模型标识、渠道绑定、API Key、超时、
 *       权重、温度、安全 RPS、JSON Schema 支持等参数
 *
 * 导出：ModelsSection（默认导出）
 *   Props: SectionProps
 */
import type { SectionProps } from '../SectionRenderer';
import { getTranslations } from '../../../i18n';
import SectionCard from '../shared/SectionCard';
import FormField from '../shared/FormField';
import TextInput from '../shared/TextInput';
import NumberInput from '../shared/NumberInput';
import SelectDropdown from '../shared/SelectDropdown';
import ToggleSwitch from '../shared/ToggleSwitch';
import ArrayItemCard from '../shared/ArrayItemCard';
import StringListEditor from '../shared/StringListEditor';

/** 单个模型的配置数据结构 */
interface ModelConfig {
  id: string;
  display_name: string;
  aliases: string[];
  upstream_model: string;
  channel_id: string;
  api_key: string;
  timeout_seconds: number;
  weight: number;
  temperature: number;
  safe_rps: number;
  capabilities: string[];
}

/** 新模型的默认配置值 */
const defaultModel: ModelConfig = {
  id: 'model-1',
  display_name: '',
  aliases: [],
  upstream_model: '',
  channel_id: '1',
  api_key: '',
  timeout_seconds: 300,
  weight: 10,
  temperature: 0.3,
  safe_rps: 5,
  capabilities: ['chat_completions', 'stream', 'json_schema'],
};

const CAPABILITIES = [
  'chat_completions',
  'responses',
  'stream',
  'multimodal',
  'tools',
  'n',
  'json_schema',
  'logprobs',
  'previous_response_id',
] as const;

/**
 * 模型配置组件
 * 以可折叠卡片列表展示所有模型，支持增删改复制操作
 */
export default function ModelsSection({ updateConfig, getConfig, language }: SectionProps) {
  const t = getTranslations(language);

  const models = (getConfig(['gateway', 'routes']) as ModelConfig[]) ?? [];
  const channels = (getConfig(['gateway', 'channels']) as Record<string, unknown>) ?? {};

  // Build channel options for dropdown
  // 构建渠道下拉选项列表
  const channelOptions = Object.keys(channels).map((id) => ({
    value: id,
    label: id,
  }));
  if (channelOptions.length === 0) {
    channelOptions.push({ value: '1', label: '1' });
  }

  /** 更新指定模型的某个字段 */
  const handleUpdate = (index: number, field: keyof ModelConfig, value: unknown) => {
    const next = [...models];
    next[index] = { ...next[index], [field]: value };
    updateConfig(['gateway', 'routes'], next);
  };

  /** 新增模型，自动分配递增 ID */
  const handleAdd = () => {
    let suffix = models.length + 1;
    while (models.some((model) => model.id === `model-${suffix}`)) suffix += 1;
    updateConfig(['gateway', 'routes'], [...models, { ...defaultModel, id: `model-${suffix}` }]);
  };

  /** 删除指定索引的模型 */
  const handleRemove = (index: number) => {
    updateConfig(['gateway', 'routes'], models.filter((_, i) => i !== index));
  };

  /** 复制指定模型，插入到原模型下方 */
  const handleDuplicate = (index: number) => {
    let suffix = models.length + 1;
    while (models.some((model) => model.id === `model-${suffix}`)) suffix += 1;
    const copy = {
      ...models[index],
      id: `model-${suffix}`,
      display_name: `${models[index].display_name}-copy`,
      aliases: [...(models[index].aliases ?? [])],
      capabilities: [...(models[index].capabilities ?? [])],
    };
    const next = [...models];
    next.splice(index + 1, 0, copy);
    updateConfig(['gateway', 'routes'], next);
  };

  return (
    <div className="space-y-4">
      <SectionCard title={t.cfgModelsTitle} description={t.cfgModelsDesc}>
        <div className="space-y-3">
          {models.map((m, i) => (
            <ArrayItemCard
              key={`${m.id}-${i}`}
              title={`#${m.id} ${m.display_name || m.upstream_model || '(unnamed)'}`}
              subtitle={`weight: ${m.weight ?? 0}`}
              onRemove={() => handleRemove(i)}
              onDuplicate={() => handleDuplicate(i)}
            >
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <FormField label="ID" required>
                  <TextInput value={m.id ?? ''} onChange={(v) => handleUpdate(i, 'id', v)} placeholder="model-1" monospace />
                </FormField>
                <FormField label={t.cfgModelName}>
                  <TextInput value={m.display_name ?? ''} onChange={(v) => handleUpdate(i, 'display_name', v)} placeholder="model-1" />
                </FormField>
                <FormField label={t.cfgModelId}>
                  <TextInput value={m.upstream_model ?? ''} onChange={(v) => handleUpdate(i, 'upstream_model', v)} placeholder="gpt-4-turbo" monospace />
                </FormField>
                <FormField label={t.cfgChannelId}>
                  <SelectDropdown
                    value={String(m.channel_id ?? '1')}
                    onChange={(v) => handleUpdate(i, 'channel_id', v)}
                    options={channelOptions}
                  />
                </FormField>
                <FormField label="API Key">
                  <TextInput value={m.api_key ?? ''} onChange={(v) => handleUpdate(i, 'api_key', v)} type="password" monospace />
                </FormField>
                <FormField label={t.cfgTimeout}>
                  <NumberInput value={m.timeout_seconds ?? 300} onChange={(v) => handleUpdate(i, 'timeout_seconds', v)} min={1} />
                </FormField>
                <FormField label={t.cfgWeight} description={t.cfgWeightDesc}>
                  <NumberInput value={m.weight ?? 10} onChange={(v) => handleUpdate(i, 'weight', v)} min={0} />
                </FormField>
                <FormField label={t.cfgTemperature}>
                  <NumberInput value={m.temperature ?? 0.3} onChange={(v) => handleUpdate(i, 'temperature', v)} min={0} max={2} step={0.1} />
                </FormField>
                <FormField label={t.cfgSafeRps} description={t.cfgSafeRpsDesc}>
                  <NumberInput value={m.safe_rps ?? 5} onChange={(v) => handleUpdate(i, 'safe_rps', v)} min={1} />
                </FormField>
              </div>
              <FormField label="Aliases">
                <StringListEditor
                  value={m.aliases ?? []}
                  onChange={(v) => handleUpdate(i, 'aliases', v)}
                  placeholder="gpt-4"
                  addLabel={t.cfgAdd}
                />
              </FormField>
              <FormField label="Capabilities" description="Explicit protocol and payload features exposed by this model.">
                <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
                  {CAPABILITIES.map((capability) => {
                    const checked = (m.capabilities ?? []).includes(capability);
                    return (
                      <label key={capability} className="flex items-center gap-2 rounded-lg border border-gray-100 px-2 py-2 text-xs text-gray-700">
                        <ToggleSwitch
                          checked={checked}
                          onChange={(enabled) =>
                            handleUpdate(
                              i,
                              'capabilities',
                              enabled
                                ? [...(m.capabilities ?? []), capability]
                                : (m.capabilities ?? []).filter((item) => item !== capability),
                            )
                          }
                        />
                        <span className="break-all font-mono">{capability}</span>
                      </label>
                    );
                  })}
                </div>
              </FormField>
            </ArrayItemCard>
          ))}
        </div>

        <button
          type="button"
          onClick={handleAdd}
          className="mt-3 w-full py-2.5 text-sm font-medium text-cyan-600 bg-cyan-50 rounded-xl hover:bg-cyan-100 border border-dashed border-cyan-200 transition-colors"
        >
          + {t.cfgAddModel}
        </button>
      </SectionCard>
    </div>
  );
}
