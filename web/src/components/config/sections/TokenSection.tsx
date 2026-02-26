/**
 * Token 估算配置分区组件
 *
 * 用途：配置 Token 用量估算的模式（输入/输出/输入+输出）、
 *       采样大小和编码方式
 *
 * 导出：TokenSection（默认导出）
 *   Props: SectionProps
 */
import type { SectionProps } from '../SectionRenderer';
import { getTranslations } from '../../../i18n';
import SectionCard from '../shared/SectionCard';
import FormField from '../shared/FormField';
import TextInput from '../shared/TextInput';
import NumberInput from '../shared/NumberInput';
import SelectDropdown from '../shared/SelectDropdown';

/** Token 估算配置组件 */
export default function TokenSection({ updateConfig, getConfig, language }: SectionProps) {
  const t = getTranslations(language);

  // Token estimation
  // Token 估算参数
  const tokenMode = (getConfig(['token_estimation', 'mode']) as string) ?? 'io';
  const sampleSize = (getConfig(['token_estimation', 'sample_size']) as number) ?? -1;
  const encoding = (getConfig(['token_estimation', 'encoding']) as string) ?? 'o200k_base';

  return (
    <div className="space-y-4">
      {/* Token Estimation */}
      {/* Token 估算设置 */}
      <SectionCard title={t.cfgTokenEstimation} description={t.cfgTokenEstimationDesc}>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <FormField label={t.cfgTokenMode}>
            <SelectDropdown
              value={tokenMode}
              onChange={(v) => updateConfig(['token_estimation', 'mode'], v)}
              options={[
                { value: 'in', label: 'Input' },
                { value: 'out', label: 'Output' },
                { value: 'io', label: 'Input + Output' },
              ]}
            />
          </FormField>
          <FormField label={t.cfgSampleSize}>
            <NumberInput
              value={sampleSize}
              onChange={(v) => updateConfig(['token_estimation', 'sample_size'], v)}
              min={-1}
              placeholder="-1"
            />
          </FormField>
          <FormField label={t.cfgEncoding}>
            <TextInput
              value={encoding}
              onChange={(v) => updateConfig(['token_estimation', 'encoding'], v)}
              placeholder="o200k_base"
              monospace
            />
          </FormField>
        </div>
        <p className="text-xs text-gray-400 mt-2">{t.cfgSampleSizeDesc}</p>
      </SectionCard>
    </div>
  );
}
