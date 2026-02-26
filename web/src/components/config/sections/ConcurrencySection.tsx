/**
 * 并发与调度配置分区组件
 *
 * 用途：配置批处理并发参数，包括批大小、保存间隔、连接数限制、
 *       分片策略、熔断器参数、各类错误重试次数上限
 *
 * 导出：ConcurrencySection（默认导出）
 *   Props: SectionProps
 */
import type { SectionProps } from '../SectionRenderer';
import { getTranslations } from '../../../i18n';
import SectionCard from '../shared/SectionCard';
import FormField from '../shared/FormField';
import NumberInput from '../shared/NumberInput';

/**
 * 并发配置组件
 * 包含四个卡片区域：基础并发、分片设置、熔断器、重试限制
 */
export default function ConcurrencySection({ updateConfig, getConfig, language }: SectionProps) {
  const t = getTranslations(language);

  // 读取并发配置的辅助函数
  const get = (key: string) => getConfig(['datasource', 'concurrency', key]) as number | undefined;
  // 更新并发配置的辅助函数
  const set = (key: string, v: number) => updateConfig(['datasource', 'concurrency', key], v);

  return (
    <div className="space-y-4">
      {/* Basic Concurrency */}
      {/* 基础并发：批大小、保存间隔、最大连接数 */}
      <SectionCard title={t.cfgBasicConcurrency}>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <FormField label={t.cfgBatchSize} description={t.cfgBatchSizeDesc}>
            <NumberInput value={get('batch_size') ?? 100} onChange={(v) => set('batch_size', v)} min={1} />
          </FormField>
          <FormField label={t.cfgSaveInterval} description={t.cfgSaveIntervalDesc}>
            <NumberInput value={get('save_interval') ?? 300} onChange={(v) => set('save_interval', v)} min={1} />
          </FormField>
          <FormField label={t.cfgMaxConnections}>
            <NumberInput value={get('max_connections') ?? 1000} onChange={(v) => set('max_connections', v)} min={1} />
          </FormField>
          <FormField label={t.cfgMaxConnectionsPerHost} description={t.cfgMaxConnPerHostDesc}>
            <NumberInput value={get('max_connections_per_host') ?? 0} onChange={(v) => set('max_connections_per_host', v)} min={0} />
          </FormField>
        </div>
      </SectionCard>

      {/* Shard Settings */}
      {/* 分片设置：控制数据分片的大小范围 */}
      <SectionCard title={t.cfgShardSettings}>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <FormField label={t.cfgShardSize}>
            <NumberInput value={get('shard_size') ?? 10000} onChange={(v) => set('shard_size', v)} min={1} />
          </FormField>
          <FormField label={t.cfgMinShardSize}>
            <NumberInput value={get('min_shard_size') ?? 1000} onChange={(v) => set('min_shard_size', v)} min={1} />
          </FormField>
          <FormField label={t.cfgMaxShardSize}>
            <NumberInput value={get('max_shard_size') ?? 50000} onChange={(v) => set('max_shard_size', v)} min={1} />
          </FormField>
        </div>
      </SectionCard>

      {/* Circuit Breaker */}
      {/* 熔断器：连续 API 错误时的全局暂停策略 */}
      <SectionCard title={t.cfgCircuitBreaker} description={t.cfgCircuitBreakerDesc}>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <FormField label={t.cfgApiPauseDuration} description={t.cfgApiPauseDurationDesc}>
            <NumberInput value={get('api_pause_duration') ?? 2.0} onChange={(v) => set('api_pause_duration', v)} min={0} step={0.1} />
          </FormField>
          <FormField label={t.cfgApiErrorWindow} description={t.cfgApiErrorWindowDesc}>
            <NumberInput value={get('api_error_trigger_window') ?? 2.0} onChange={(v) => set('api_error_trigger_window', v)} min={0} step={0.1} />
          </FormField>
        </div>
      </SectionCard>

      {/* Retry Limits */}
      {/* 重试限制：各类错误的最大重试次数 */}
      <SectionCard title={t.cfgRetryLimits} description={t.cfgRetryLimitsDesc}>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <FormField label={t.cfgApiError}>
            <NumberInput
              value={(getConfig(['datasource', 'concurrency', 'retry_limits', 'api_error']) as number) ?? 3}
              onChange={(v) => updateConfig(['datasource', 'concurrency', 'retry_limits', 'api_error'], v)}
              min={0}
            />
          </FormField>
          <FormField label={t.cfgContentError}>
            <NumberInput
              value={(getConfig(['datasource', 'concurrency', 'retry_limits', 'content_error']) as number) ?? 1}
              onChange={(v) => updateConfig(['datasource', 'concurrency', 'retry_limits', 'content_error'], v)}
              min={0}
            />
          </FormField>
          <FormField label={t.cfgSystemError}>
            <NumberInput
              value={(getConfig(['datasource', 'concurrency', 'retry_limits', 'system_error']) as number) ?? 2}
              onChange={(v) => updateConfig(['datasource', 'concurrency', 'retry_limits', 'system_error'], v)}
              min={0}
            />
          </FormField>
        </div>
      </SectionCard>
    </div>
  );
}
