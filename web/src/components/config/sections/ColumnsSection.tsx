import type { SectionProps } from '../SectionRenderer';
import { getTranslations } from '../../../i18n';
import SectionCard from '../shared/SectionCard';

import StringListEditor from '../shared/StringListEditor';
import KeyValueEditor from '../shared/KeyValueEditor';

export default function ColumnsSection({ updateConfig, getConfig, language }: SectionProps) {
  const t = getTranslations(language);

  const columnsToExtract = (getConfig(['columns_to_extract']) as string[]) ?? [];
  const columnsToWrite = (getConfig(['columns_to_write']) as Record<string, string>) ?? {};

  return (
    <div className="space-y-4">
      {/* Columns to Extract */}
      <SectionCard title={t.cfgColumnsToExtract} description={t.cfgColumnsToExtractDesc}>
        <StringListEditor
          value={columnsToExtract}
          onChange={(v) => updateConfig(['columns_to_extract'], v)}
          placeholder={t.cfgAddColumn}
          addLabel={t.cfgAdd}
        />
      </SectionCard>

      {/* Columns to Write */}
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
