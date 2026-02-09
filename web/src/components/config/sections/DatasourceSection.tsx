import { useState } from 'react';
import type { SectionProps } from '../SectionRenderer';
import { getTranslations } from '../../../i18n';
import SectionCard from '../shared/SectionCard';
import FormField from '../shared/FormField';
import TextInput from '../shared/TextInput';
import NumberInput from '../shared/NumberInput';
import SelectDropdown from '../shared/SelectDropdown';
import ToggleSwitch from '../shared/ToggleSwitch';
import { testFeishuConnection } from '../../../api';

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
              { value: 'feishu_bitable', label: t.cfgFeishuBitable },
              { value: 'feishu_sheet', label: t.cfgFeishuSheet },
            ]}
          />
        </FormField>
      </SectionCard>

      {/* Engine Settings (only for file-based datasources) */}
      {(dsType === 'excel' || dsType === 'csv') && (
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
        </SectionCard>
      )}

      {/* Input field validation */}
      <SectionCard title={t.cfgValidation || 'Validation'}>
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
      {dsType === 'feishu_bitable' && <FeishuBitableConnection getConfig={getConfig} updateConfig={updateConfig} t={t} />}
      {dsType === 'feishu_sheet' && <FeishuSheetConnection getConfig={getConfig} updateConfig={updateConfig} t={t} />}
    </div>
  );
}

// --- Connection sub-components ---

interface ConnProps {
  getConfig: (path: string[]) => unknown;
  updateConfig: (path: string[], value: unknown) => void;
  t: ReturnType<typeof getTranslations>;
}

function FeishuTestButton({ appId, appSecret, t }: { appId: string; appSecret: string; t: ReturnType<typeof getTranslations> }) {
  const [testing, setTesting] = useState(false);
  const [result, setResult] = useState<{ success: boolean; message: string } | null>(null);

  const handleTest = async () => {
    if (!appId || !appSecret) {
      setResult({ success: false, message: 'App ID and App Secret are required' });
      return;
    }

    setTesting(true);
    setResult(null);

    try {
      const res = await testFeishuConnection(appId, appSecret);
      setResult(res);
    } catch (err) {
      setResult({ success: false, message: err instanceof Error ? err.message : 'Unknown error' });
    } finally {
      setTesting(false);
    }
  };

  return (
    <div className="mt-2">
      <div className="flex items-center space-x-2">
        <button
          onClick={handleTest}
          disabled={testing || !appId || !appSecret}
          className={`px-3 py-1.5 text-sm font-medium rounded-md transition-colors ${
            testing || !appId || !appSecret
              ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
              : 'bg-cyan-50 text-cyan-600 hover:bg-cyan-100 border border-cyan-200'
          }`}
        >
          {testing ? t.loading : 'Test Connection'}
        </button>

        {result && (
          <span className={`text-sm ${result.success ? 'text-green-600' : 'text-red-500'}`}>
            {result.success ? '✓ ' : '✗ '} {result.message}
          </span>
        )}
      </div>
    </div>
  );
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

function FeishuBitableConnection({ getConfig, updateConfig, t }: ConnProps) {
  const appId = (getConfig(['feishu', 'app_id']) as string) ?? '';
  const appSecret = (getConfig(['feishu', 'app_secret']) as string) ?? '';

  return (
    <SectionCard title={t.cfgConnectionSettings} description={t.cfgFeishuDesc}>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <FormField label={t.cfgFeishuAppId} required>
          <TextInput
            value={appId}
            onChange={(v) => updateConfig(['feishu', 'app_id'], v)}
            placeholder="cli_xxxxxxxxxxxxx"
            monospace
          />
        </FormField>
        <FormField label={t.cfgFeishuAppSecret} required>
          <TextInput
            value={appSecret}
            onChange={(v) => updateConfig(['feishu', 'app_secret'], v)}
            type="password"
          />
        </FormField>
        <div className="sm:col-span-2">
            <FeishuTestButton appId={appId} appSecret={appSecret} t={t} />
        </div>
        <FormField label={t.cfgFeishuAppToken} required>
          <TextInput
            value={(getConfig(['feishu', 'app_token']) as string) ?? ''}
            onChange={(v) => updateConfig(['feishu', 'app_token'], v)}
            placeholder="bascxxxxxxxxxxxxx"
            monospace
          />
        </FormField>
        <FormField label={t.cfgFeishuTableId} required>
          <TextInput
            value={(getConfig(['feishu', 'table_id']) as string) ?? ''}
            onChange={(v) => updateConfig(['feishu', 'table_id'], v)}
            placeholder="tblxxxxxxxxxxxxx"
            monospace
          />
        </FormField>
        <FormField label={t.cfgFeishuMaxRetries}>
          <NumberInput
            value={(getConfig(['feishu', 'max_retries']) as number) ?? 3}
            onChange={(v) => updateConfig(['feishu', 'max_retries'], v)}
            min={0} max={10}
          />
        </FormField>
        <FormField label={t.cfgFeishuQpsLimit} description={t.cfgFeishuQpsLimitDesc}>
          <NumberInput
            value={(getConfig(['feishu', 'qps_limit']) as number) ?? 5}
            onChange={(v) => updateConfig(['feishu', 'qps_limit'], v)}
            min={0}
          />
        </FormField>
      </div>
    </SectionCard>
  );
}

function FeishuSheetConnection({ getConfig, updateConfig, t }: ConnProps) {
  const appId = (getConfig(['feishu', 'app_id']) as string) ?? '';
  const appSecret = (getConfig(['feishu', 'app_secret']) as string) ?? '';

  return (
    <SectionCard title={t.cfgConnectionSettings} description={t.cfgFeishuDesc}>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <FormField label={t.cfgFeishuAppId} required>
          <TextInput
            value={appId}
            onChange={(v) => updateConfig(['feishu', 'app_id'], v)}
            placeholder="cli_xxxxxxxxxxxxx"
            monospace
          />
        </FormField>
        <FormField label={t.cfgFeishuAppSecret} required>
          <TextInput
            value={appSecret}
            onChange={(v) => updateConfig(['feishu', 'app_secret'], v)}
            type="password"
          />
        </FormField>
        <div className="sm:col-span-2">
            <FeishuTestButton appId={appId} appSecret={appSecret} t={t} />
        </div>
        <FormField label={t.cfgFeishuSpreadsheetToken} required>
          <TextInput
            value={(getConfig(['feishu', 'spreadsheet_token']) as string) ?? ''}
            onChange={(v) => updateConfig(['feishu', 'spreadsheet_token'], v)}
            placeholder="shtcnxxxxxxxxxxxxx"
            monospace
          />
        </FormField>
        <FormField label={t.cfgFeishuSheetId} required>
          <TextInput
            value={(getConfig(['feishu', 'sheet_id']) as string) ?? ''}
            onChange={(v) => updateConfig(['feishu', 'sheet_id'], v)}
            placeholder="0"
            monospace
          />
        </FormField>
        <FormField label={t.cfgFeishuMaxRetries}>
          <NumberInput
            value={(getConfig(['feishu', 'max_retries']) as number) ?? 3}
            onChange={(v) => updateConfig(['feishu', 'max_retries'], v)}
            min={0} max={10}
          />
        </FormField>
        <FormField label={t.cfgFeishuQpsLimit} description={t.cfgFeishuQpsLimitDesc}>
          <NumberInput
            value={(getConfig(['feishu', 'qps_limit']) as number) ?? 5}
            onChange={(v) => updateConfig(['feishu', 'qps_limit'], v)}
            min={0}
          />
        </FormField>
      </div>
    </SectionCard>
  );
}
