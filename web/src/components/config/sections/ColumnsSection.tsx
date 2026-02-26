/**
 * 列映射配置分区组件
 *
 * 用途：配置从 AI 响应中提取的字段列表（columns_to_extract）
 *       以及写回数据源时的列名映射（columns_to_write）
 *
 * 导出：ColumnsSection（默认导出）
 *   Props: SectionProps
 */
import type { SectionProps } from '../SectionRenderer';
import { getTranslations } from '../../../i18n';
import SectionCard from '../shared/SectionCard';
import StringListEditor from '../shared/StringListEditor';
import KeyValueEditor from '../shared/KeyValueEditor';

/**
 * 列映射配置组件
 * 上半部分管理待提取列（字符串列表），下半部分管理写回列的别名映射（键值对）
 */
export default function ColumnsSection({ updateConfig, getConfig, language }: SectionProps) {
  const t = getTranslations(language);

  const columnsToExtract = (getConfig(['columns_to_extract']) as string[]) ?? [];
  const columnsToWrite = (getConfig(['columns_to_write']) as Record<string, string>) ?? {};

  return (
    <div className="space-y-4">
      {/* Columns to Extract */}
      {/* 待提取列：AI 响应中需要提取的字段名列表 */}
      <SectionCard title={t.cfgColumnsToExtract} description={t.cfgColumnsToExtractDesc}>
        <StringListEditor
          value={columnsToExtract}
          onChange={(v) => updateConfig(['columns_to_extract'], v)}
          placeholder={t.cfgAddColumn}
          addLabel={t.cfgAdd}
        />
      </SectionCard>

      {/* Columns to Write */}
      {/* 写回列映射：别名 → 数据源列名 */}
      <SectionCard title={t.cfgColumnsToWrite} description={t.cfgColumnsToWriteDesc}>
        <KeyValueEditor
          value={columnsToWrite}
          onChange={(v) => updateConfig(['columns_to_write'], v)}
          keyPlaceholder={t.cfgAlias}
          valuePlaceholder={t.cfgColumnName}
          addLabel={t.cfgAdd}
        />
      </SectionCard>
    </div>
  );
}
