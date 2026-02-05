import type { SectionProps } from '../SectionRenderer';
import { getTranslations } from '../../../i18n';
import SectionCard from '../shared/SectionCard';
import FormField from '../shared/FormField';
import TextInput from '../shared/TextInput';
import NumberInput from '../shared/NumberInput';
import SelectDropdown from '../shared/SelectDropdown';
import ToggleSwitch from '../shared/ToggleSwitch';

export default function DatasourceSection({ updateConfig, getConfig, language }: SectionProps) {
  const t = getTranslations(language);

  const dsType = (getConfig(['datasource', 'type']) as string) ?? 'excel';
  const engine = (getConfig(['datasource', 'engine']) as string) ?? 'auto';
  const excelReader = (getConfig(['datasource', 'excel_reader']) as string) ?? 'auto';
  const excelWriter = (getConfig(['datasource', 'excel_writer']) as string) ?? 'auto';
  const requireAll = (getConfig(['datasource', 'require_all_input_fields']) as boolean) ?? true;

  return (
    <div className="space-y-4">
      {/* Datasource Type */}
      <SectionCard title={t.cfgDatasourceType}>
        <FormField label={t.cfgType} required>
          <SelectDropdown
            value={dsType}
            onChange={(v) => updateConfig(['datasource', 'type'], v)}
            options={[
              { value: 'excel', label: 'Excel' },
              { value: 'csv', label: 'CSV' },
              { value: 'mysql', label: 'MySQL' },
              { value: 'postgresql', label: 'PostgreSQL' },
              { value: 'sqlite', label: 'SQLite' },
            ]}
          />
        </FormField>
      </SectionCard>

      {/* Engine Settings */}
      <SectionCard title={t.cfgEngineSettings} description={t.cfgEngineSettingsDesc}>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <FormField label={t.cfgEngine}>
            <SelectDropdown
              value={engine}
              onChange={(v) => updateConfig(['datasource', 'engine'], v)}
              options={[
                { value: 'auto', label: 'Auto' },
                { value: 'pandas', label: 'Pandas' },
                { value: 'polars', label: 'Polars' },
              ]}
            />
          </FormField>
          {dsType === 'excel' && (
            <>
              <FormField label={t.cfgExcelReader}>
                <SelectDropdown
                  value={excelReader}
                  onChange={(v) => updateConfig(['datasource', 'excel_reader'], v)}
                  options={[
                    { value: 'auto', label: 'Auto' },
                    { value: 'openpyxl', label: 'openpyxl' },
                    { value: 'calamine', label: 'calamine (10x)' },
                  ]}
                />
              </FormField>
              <FormField label={t.cfgExcelWriter}>
                <SelectDropdown
                  value={excelWriter}
                  onChange={(v) => updateConfig(['datasource', 'excel_writer'], v)}
                  options={[
                    { value: 'auto', label: 'Auto' },
                    { value: 'openpyxl', label: 'openpyxl' },
                    { value: 'xlsxwriter', label: 'xlsxwriter (3x)' },
                  ]}
                />
              </FormField>
            </>
          )}
        </div>
        <FormField label={t.cfgRequireAllFields} description={t.cfgRequireAllFieldsDesc} horizontal>
          <ToggleSwitch
            checked={requireAll}
            onChange={(v) => updateConfig(['datasource', 'require_all_input_fields'], v)}
          />
        </FormField>
      </SectionCard>

      {/* Connection Config */}
      {dsType === 'excel' && <ExcelConnection getConfig={getConfig} updateConfig={updateConfig} t={t} />}
      {dsType === 'csv' && <CsvConnection getConfig={getConfig} updateConfig={updateConfig} t={t} />}
      {dsType === 'mysql' && <MysqlConnection getConfig={getConfig} updateConfig={updateConfig} t={t} />}
      {dsType === 'postgresql' && <PostgresqlConnection getConfig={getConfig} updateConfig={updateConfig} t={t} />}
      {dsType === 'sqlite' && <SqliteConnection getConfig={getConfig} updateConfig={updateConfig} t={t} />}
    </div>
  );
}

// --- Connection sub-components ---

interface ConnProps {
  getConfig: (path: string[]) => unknown;
  updateConfig: (path: string[], value: unknown) => void;
  t: ReturnType<typeof getTranslations>;
}

function ExcelConnection({ getConfig, updateConfig, t }: ConnProps) {
  return (
    <SectionCard title={t.cfgConnectionSettings}>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <FormField label={t.cfgInputPath} required>
          <TextInput
            value={(getConfig(['excel', 'input_path']) as string) ?? ''}
            onChange={(v) => updateConfig(['excel', 'input_path'], v)}
            placeholder="./data/input.xlsx"
            monospace
          />
        </FormField>
        <FormField label={t.cfgOutputPath}>
          <TextInput
            value={(getConfig(['excel', 'output_path']) as string) ?? ''}
            onChange={(v) => updateConfig(['excel', 'output_path'], v)}
            placeholder={t.cfgOutputPathDefault}
            monospace
          />
        </FormField>
      </div>
    </SectionCard>
  );
}

function CsvConnection({ getConfig, updateConfig, t }: ConnProps) {
  return (
    <SectionCard title={t.cfgConnectionSettings}>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <FormField label={t.cfgInputPath} required>
          <TextInput
            value={(getConfig(['csv', 'input_path']) as string) ?? ''}
            onChange={(v) => updateConfig(['csv', 'input_path'], v)}
            placeholder="./data/input.csv"
            monospace
          />
        </FormField>
        <FormField label={t.cfgOutputPath}>
          <TextInput
            value={(getConfig(['csv', 'output_path']) as string) ?? ''}
            onChange={(v) => updateConfig(['csv', 'output_path'], v)}
            placeholder={t.cfgOutputPathDefault}
            monospace
          />
        </FormField>
      </div>
    </SectionCard>
  );
}

function MysqlConnection({ getConfig, updateConfig, t }: ConnProps) {
  return (
    <SectionCard title={t.cfgConnectionSettings}>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <FormField label="Host" required>
          <TextInput
            value={(getConfig(['mysql', 'host']) as string) ?? 'localhost'}
            onChange={(v) => updateConfig(['mysql', 'host'], v)}
            placeholder="localhost"
          />
        </FormField>
        <FormField label="Port" required>
          <NumberInput
            value={(getConfig(['mysql', 'port']) as number) ?? 3306}
            onChange={(v) => updateConfig(['mysql', 'port'], v)}
            min={1} max={65535}
          />
        </FormField>
        <FormField label={t.cfgUser} required>
          <TextInput
            value={(getConfig(['mysql', 'user']) as string) ?? ''}
            onChange={(v) => updateConfig(['mysql', 'user'], v)}
            placeholder="root"
          />
        </FormField>
        <FormField label={t.cfgPassword} required>
          <TextInput
            value={(getConfig(['mysql', 'password']) as string) ?? ''}
            onChange={(v) => updateConfig(['mysql', 'password'], v)}
            type="password"
          />
        </FormField>
        <FormField label={t.cfgDatabase} required>
          <TextInput
            value={(getConfig(['mysql', 'database']) as string) ?? ''}
            onChange={(v) => updateConfig(['mysql', 'database'], v)}
          />
        </FormField>
        <FormField label={t.cfgTableName} required>
          <TextInput
            value={(getConfig(['mysql', 'table_name']) as string) ?? ''}
            onChange={(v) => updateConfig(['mysql', 'table_name'], v)}
          />
        </FormField>
        <FormField label={t.cfgPoolSize}>
          <NumberInput
            value={(getConfig(['mysql', 'pool_size']) as number) ?? undefined}
            onChange={(v) => updateConfig(['mysql', 'pool_size'], v)}
            min={1}
            placeholder="auto (batch_size / 10)"
          />
        </FormField>
      </div>
    </SectionCard>
  );
}

function PostgresqlConnection({ getConfig, updateConfig, t }: ConnProps) {
  return (
    <SectionCard title={t.cfgConnectionSettings}>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <FormField label="Host" required>
          <TextInput
            value={(getConfig(['postgresql', 'host']) as string) ?? 'localhost'}
            onChange={(v) => updateConfig(['postgresql', 'host'], v)}
            placeholder="localhost"
          />
        </FormField>
        <FormField label="Port" required>
          <NumberInput
            value={(getConfig(['postgresql', 'port']) as number) ?? 5432}
            onChange={(v) => updateConfig(['postgresql', 'port'], v)}
            min={1} max={65535}
          />
        </FormField>
        <FormField label={t.cfgUser} required>
          <TextInput
            value={(getConfig(['postgresql', 'user']) as string) ?? ''}
            onChange={(v) => updateConfig(['postgresql', 'user'], v)}
            placeholder="postgres"
          />
        </FormField>
        <FormField label={t.cfgPassword} required>
          <TextInput
            value={(getConfig(['postgresql', 'password']) as string) ?? ''}
            onChange={(v) => updateConfig(['postgresql', 'password'], v)}
            type="password"
          />
        </FormField>
        <FormField label={t.cfgDatabase} required>
          <TextInput
            value={(getConfig(['postgresql', 'database']) as string) ?? ''}
            onChange={(v) => updateConfig(['postgresql', 'database'], v)}
          />
        </FormField>
        <FormField label={t.cfgTableName} required>
          <TextInput
            value={(getConfig(['postgresql', 'table_name']) as string) ?? ''}
            onChange={(v) => updateConfig(['postgresql', 'table_name'], v)}
          />
        </FormField>
        <FormField label="Schema">
          <TextInput
            value={(getConfig(['postgresql', 'schema_name']) as string) ?? 'public'}
            onChange={(v) => updateConfig(['postgresql', 'schema_name'], v)}
            placeholder="public"
          />
        </FormField>
        <FormField label={t.cfgPoolSize}>
          <NumberInput
            value={(getConfig(['postgresql', 'pool_size']) as number) ?? undefined}
            onChange={(v) => updateConfig(['postgresql', 'pool_size'], v)}
            min={1}
            placeholder="auto (batch_size / 10)"
          />
        </FormField>
      </div>
    </SectionCard>
  );
}

function SqliteConnection({ getConfig, updateConfig, t }: ConnProps) {
  return (
    <SectionCard title={t.cfgConnectionSettings}>
      <div className="grid grid-cols-1 gap-4">
        <FormField label={t.cfgDbPath} required>
          <TextInput
            value={(getConfig(['sqlite', 'db_path']) as string) ?? ''}
            onChange={(v) => updateConfig(['sqlite', 'db_path'], v)}
            placeholder="./data/tasks.db"
            monospace
          />
        </FormField>
        <FormField label={t.cfgTableName} required>
          <TextInput
            value={(getConfig(['sqlite', 'table_name']) as string) ?? ''}
            onChange={(v) => updateConfig(['sqlite', 'table_name'], v)}
          />
        </FormField>
      </div>
    </SectionCard>
  );
}
