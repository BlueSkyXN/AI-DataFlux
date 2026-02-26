/**
 * 路由（子任务分发）配置分区组件
 *
 * 用途：配置基于字段值的任务路由功能，可将不同类别的数据
 *       分发到不同的处理配置文件（profile）
 *
 * 导出：RoutingSection（默认导出）
 *   Props: SectionProps
 */
import type { SectionProps } from '../SectionRenderer';
import { getTranslations } from '../../../i18n';
import SectionCard from '../shared/SectionCard';
import FormField from '../shared/FormField';
import TextInput from '../shared/TextInput';
import ToggleSwitch from '../shared/ToggleSwitch';

/** 单条子任务路由规则 */
interface Subtask {
  match: string;
  profile: string;
}

/**
 * 路由配置组件
 * 启用后显示路由字段和子任务匹配规则列表
 */
export default function RoutingSection({ updateConfig, getConfig, language }: SectionProps) {
  const t = getTranslations(language);

  const enabled = (getConfig(['routing', 'enabled']) as boolean) ?? false;
  const field = (getConfig(['routing', 'field']) as string) ?? '';
  const subtasks = (getConfig(['routing', 'subtasks']) as Subtask[]) ?? [];

  /** 更新指定子任务规则的字段值 */
  const handleSubtaskUpdate = (index: number, key: keyof Subtask, value: string) => {
    const next = [...subtasks];
    next[index] = { ...next[index], [key]: value };
    updateConfig(['routing', 'subtasks'], next);
  };

  const handleAddSubtask = () => {
    updateConfig(['routing', 'subtasks'], [...subtasks, { match: '', profile: '' }]);
  };

  const handleRemoveSubtask = (index: number) => {
    updateConfig(['routing', 'subtasks'], subtasks.filter((_, i) => i !== index));
  };

  return (
    <div className="space-y-4">
      <SectionCard title={t.cfgRoutingSettings} description={t.cfgRoutingDesc}>
        <FormField label={t.cfgEnableRouting} horizontal>
          <ToggleSwitch
            checked={enabled}
            onChange={(v) => updateConfig(['routing', 'enabled'], v)}
          />
        </FormField>

        {enabled && (
          <>
            <FormField label={t.cfgRoutingField} description={t.cfgRoutingFieldDesc}>
              <TextInput
                value={field}
                onChange={(v) => updateConfig(['routing', 'field'], v)}
                placeholder="category"
                monospace
              />
            </FormField>

            {/* Subtasks */}
            {/* 子任务规则列表：match（匹配值）→ profile（配置文件路径） */}
            <div className="mt-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">{t.cfgSubtasks}</label>
              <div className="space-y-2">
                {subtasks.map((st, i) => (
                  <div key={i} className="flex items-center gap-2">
                    <input
                      type="text"
                      value={st.match}
                      onChange={(e) => handleSubtaskUpdate(i, 'match', e.target.value)}
                      placeholder={t.cfgMatchValue}
                      className="w-[140px] px-2.5 py-1.5 text-sm font-mono border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-cyan-400 focus:border-transparent"
                    />
                    <svg className="w-4 h-4 text-gray-300 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                    </svg>
                    <input
                      type="text"
                      value={st.profile}
                      onChange={(e) => handleSubtaskUpdate(i, 'profile', e.target.value)}
                      placeholder={t.cfgProfilePath}
                      className="flex-1 px-2.5 py-1.5 text-sm font-mono border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-cyan-400 focus:border-transparent"
                    />
                    <button
                      type="button"
                      onClick={() => handleRemoveSubtask(i)}
                      className="text-gray-400 hover:text-red-500 p-1"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  </div>
                ))}
              </div>

              <button
                type="button"
                onClick={handleAddSubtask}
                className="mt-2 w-full py-2 text-sm font-medium text-cyan-600 bg-cyan-50 rounded-xl hover:bg-cyan-100 border border-dashed border-cyan-200 transition-colors"
              >
                + {t.cfgAddSubtask}
              </button>
            </div>
          </>
        )}
      </SectionCard>
    </div>
  );
}
