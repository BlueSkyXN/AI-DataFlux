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

/** 单个模型的配置数据结构 */
interface ModelConfig {
  id: string;
  name: string;
  model: string;
  channel_id: string;
  api_key: string;
  timeout: number;
  weight: number;
  temperature: number;
  safe_rps: number;
  capabilities: string[];
}

/** 新模型的默认配置值 */
const defaultModel: ModelConfig = {
  id: 'model-1',
  name: '',
  model: '',
  channel_id: '1',
  api_key: '',
  timeout: 300,
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

  const models = (getConfig(['models']) as ModelConfig[]) ?? [];
  const channels = (getConfig(['channels']) as Record<string, { name?: string }>) ?? {};

  // Build channel options for dropdown
  // 构建渠道下拉选项列表
  const channelOptions = Object.entries(channels).map(([id, ch]) => ({
    value: id,
    label: ch.name ? `${id} - ${ch.name}` : id,
  }));
  if (channelOptions.length === 0) {
    channelOptions.push({ value: '1', label: '1' });
  }

  /** 更新指定模型的某个字段 */
  const handleUpdate = (index: number, field: keyof ModelConfig, value: unknown) => {
    const next = [...models];
    next[index] = { ...next[index], [field]: value };
    updateConfig(['models'], next);
  };

  /** 新增模型，自动分配递增 ID */
  const handleAdd = () => {
    let suffix = models.length + 1;
    while (models.some((model) => model.id === `model-${suffix}`)) suffix += 1;
    updateConfig(['models'], [...models, { ...defaultModel, id: `model-${suffix}` }]);
  };

  /** 删除指定索引的模型 */
  const handleRemove = (index: number) => {
    updateConfig(['models'], models.filter((_, i) => i !== index));
  };

  /** 复制指定模型，插入到原模型下方 */
  const handleDuplicate = (index: number) => {
    let suffix = models.length + 1;
    while (models.some((model) => model.id === `model-${suffix}`)) suffix += 1;
    const copy = {
      ...models[index],
      id: `model-${suffix}`,
      name: `${models[index].name}-copy`,
      capabilities: [...(models[index].capabilities ?? [])],
    };
    const next = [...models];
    next.splice(index + 1, 0, copy);
    updateConfig(['models'], next);
  };

  return (
    <div className="space-y-4">
      <SectionCard title={t.cfgModelsTitle} description={t.cfgModelsDesc}>
        <div className="space-y-3">
          {models.map((m, i) => (
            <ArrayItemCard
              key={`${m.id}-${i}`}
              title={`#${m.id} ${m.name || m.model || '(unnamed)'}`}
              subtitle={`weight: ${m.weight ?? 0}`}
              onRemove={() => handleRemove(i)}
              onDuplicate={() => handleDuplicate(i)}
            >
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <FormField label="ID" required>
                  <TextInput value={m.id ?? ''} onChange={(v) => handleUpdate(i, 'id', v)} placeholder="model-1" monospace />
                </FormField>
                <FormField label={t.cfgModelName}>
                  <TextInput value={m.name ?? ''} onChange={(v) => handleUpdate(i, 'name', v)} placeholder="model-1" />
                </FormField>
                <FormField label={t.cfgModelId}>
                  <TextInput value={m.model ?? ''} onChange={(v) => handleUpdate(i, 'model', v)} placeholder="gpt-4-turbo" monospace />
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
                  <NumberInput value={m.timeout ?? 300} onChange={(v) => handleUpdate(i, 'timeout', v)} min={1} />
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
