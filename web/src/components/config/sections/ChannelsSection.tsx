/**
 * API 渠道（Channels）配置分区组件
 *
 * 用途：管理 API 渠道列表，支持添加、删除、编辑渠道的连接参数
 *       每个渠道包含名称、Base URL、API Path、超时、代理、SSL 验证、IP 池等配置
 *
 * 导出：ChannelsSection（默认导出）
 *   Props: SectionProps（formData, updateConfig, getConfig, language）
 *
 * 依赖：../shared/*（表单控件）、../../../i18n（国际化）
 */
import { useState } from 'react';
import type { SectionProps } from '../SectionRenderer';
import { getTranslations } from '../../../i18n';
import SectionCard from '../shared/SectionCard';
import ArrayItemCard from '../shared/ArrayItemCard';
import FormField from '../shared/FormField';
import TextInput from '../shared/TextInput';
import NumberInput from '../shared/NumberInput';
import ToggleSwitch from '../shared/ToggleSwitch';
import StringListEditor from '../shared/StringListEditor';

/** 单个渠道的配置数据结构 */
interface ChannelConfig {
  name: string;
  base_url: string;
  api_path: string;
  timeout: number;
  proxy: string;
  ssl_verify: boolean;
  ip_pool?: string[];
}

/** 新建渠道时的默认值 */
const defaultChannel: ChannelConfig = {
  name: '',
  base_url: '',
  api_path: '/v1/chat/completions',
  timeout: 300,
  proxy: '',
  ssl_verify: true,
};

/**
 * 渠道配置分区组件
 * 以卡片列表形式展示和编辑所有 API 渠道，底部提供新增渠道输入框
 */
export default function ChannelsSection({ updateConfig, getConfig, language }: SectionProps) {
  const t = getTranslations(language);
  const [newChannelId, setNewChannelId] = useState('');

  const channels = (getConfig(['channels']) as Record<string, ChannelConfig>) ?? {};
  const models = (getConfig(['models']) as Array<{ channel_id?: string }>) ?? [];

  const channelEntries = Object.entries(channels);

  /** 更新指定渠道的某个字段 */
  const handleUpdate = (id: string, field: keyof ChannelConfig, value: unknown) => {
    const next = { ...channels, [id]: { ...channels[id], [field]: value } };
    updateConfig(['channels'], next);
  };

  /** 添加新渠道，ID 不能为空且不能重复 */
  const handleAdd = () => {
    const id = newChannelId.trim();
    if (!id || id in channels) return;
    updateConfig(['channels'], { ...channels, [id]: { ...defaultChannel } });
    setNewChannelId('');
  };

  /** 删除渠道，若有模型引用该渠道则弹出确认提示 */
  const handleRemove = (id: string) => {
    // Check if any model references this channel
    // 检查是否有模型引用了此渠道
    const referencedBy = models.filter((m) => String(m.channel_id) === id);
    if (referencedBy.length > 0) {
      const ok = window.confirm(
        `${t.cfgChannelInUse} (${referencedBy.length} ${t.cfgModels.toLowerCase()})`
      );
      if (!ok) return;
    }
    const next = { ...channels };
    delete next[id];
    updateConfig(['channels'], next);
  };

  return (
    <div className="space-y-4">
      <SectionCard title={t.cfgChannelsTitle} description={t.cfgChannelsDesc}>
        <div className="space-y-3">
          {channelEntries.map(([id, ch]) => (
            <ArrayItemCard
              key={id}
              title={`Channel "${id}"`}
              subtitle={ch.name || undefined}
              onRemove={() => handleRemove(id)}
            >
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <FormField label={t.cfgChannelName}>
                  <TextInput value={ch.name ?? ''} onChange={(v) => handleUpdate(id, 'name', v)} placeholder="openai-api" />
                </FormField>
                <FormField label="Base URL" required>
                  <TextInput
                    value={ch.base_url ?? ''}
                    onChange={(v) => handleUpdate(id, 'base_url', v)}
                    placeholder="https://api.openai.com"
                    monospace
                  />
                </FormField>
                <FormField label="API Path" required>
                  <TextInput
                    value={ch.api_path ?? '/v1/chat/completions'}
                    onChange={(v) => handleUpdate(id, 'api_path', v)}
                    placeholder="/v1/chat/completions"
                    monospace
                  />
                </FormField>
                <FormField label={t.cfgTimeout}>
                  <NumberInput value={ch.timeout ?? 300} onChange={(v) => handleUpdate(id, 'timeout', v)} min={1} />
                </FormField>
                <FormField label="Proxy">
                  <TextInput
                    value={ch.proxy ?? ''}
                    onChange={(v) => handleUpdate(id, 'proxy', v)}
                    placeholder="http://127.0.0.1:7890"
                    monospace
                  />
                </FormField>
              </div>
              <FormField label="SSL Verify" horizontal>
                <ToggleSwitch
                  checked={ch.ssl_verify ?? true}
                  onChange={(v) => handleUpdate(id, 'ssl_verify', v)}
                />
              </FormField>
              <FormField label="IP Pool" description={t.cfgIpPoolDesc}>
                <StringListEditor
                  value={ch.ip_pool ?? []}
                  onChange={(v) => handleUpdate(id, 'ip_pool', v)}
                  placeholder="1.2.3.4"
                  addLabel={t.cfgAdd}
                />
              </FormField>
            </ArrayItemCard>
          ))}
        </div>

        {/* Add channel */}
        <div className="flex gap-2 mt-3">
          <input
            type="text"
            value={newChannelId}
            onChange={(e) => setNewChannelId(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault();
                handleAdd();
              }
            }}
            placeholder={t.cfgChannelIdPlaceholder}
            className="w-24 px-3 py-2 text-sm font-mono border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-cyan-400 focus:border-transparent"
          />
          <button
            type="button"
            onClick={handleAdd}
            disabled={!newChannelId.trim() || newChannelId.trim() in channels}
            className="flex-1 py-2.5 text-sm font-medium text-cyan-600 bg-cyan-50 rounded-xl hover:bg-cyan-100 border border-dashed border-cyan-200 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          >
            + {t.cfgAddChannel}
          </button>
        </div>
      </SectionCard>
    </div>
  );
}
