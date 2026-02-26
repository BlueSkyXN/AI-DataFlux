/**
 * 数据源配置分区组件
 *
 * 用途：配置数据源类型及其连接参数，支持 Excel、CSV、MySQL、PostgreSQL、
 *       SQLite、飞书多维表格、飞书电子表格等多种数据源
 *       包含引擎选择（Pandas/Polars）、输入字段校验等设置
 *
 * 导出：DatasourceSection（默认导出）
 *   Props: SectionProps
 *
 * 内部子组件：ExcelConnection、CsvConnection、MysqlConnection、
 *             PostgresqlConnection、SqliteConnection、FeishuBitableConnection、
 *             FeishuSheetConnection、FeishuTestButton
 *
 * 依赖：../shared/*（表单控件）、../../../api（飞书连接测试 API）、../../../i18n
 */
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

/**
 * 数据源配置主组件
 * 顶部选择数据源类型，根据类型动态渲染对应的连接配置子组件
 */
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
      {/* 数据源类型选择 */}
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
      {/* 引擎设置：仅文件类数据源（Excel/CSV）显示 */}
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
      {/* 输入字段校验：是否要求所有输入字段非空 */}
      <SectionCard title={t.cfgValidation || 'Validation'}>
        <FormField label={t.cfgRequireAllFields} description={t.cfgRequireAllFieldsDesc} horizontal>
          <ToggleSwitch
            checked={requireAll}
            onChange={(v) => updateConfig(['datasource', 'require_all_input_fields'], v)}
          />
        </FormField>
      </SectionCard>

      {/* Connection Config */}
      {/* 根据数据源类型渲染对应的连接配置子组件 */}
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
// --- 数据源连接配置子组件 ---

/** 连接配置子组件的通用 Props */
interface ConnProps {
  getConfig: (path: string[]) => unknown;
  updateConfig: (path: string[], value: unknown) => void;
  t: ReturnType<typeof getTranslations>;
}

function getStringWithDefinedFallback(primary: unknown, fallback: unknown): string {
  if (primary !== null && primary !== undefined) return String(primary);
  if (fallback !== null && fallback !== undefined) return String(fallback);
  return '';
}

/** 飞书连接测试按钮组件，调用后端 API 验证 App 凭证 */
function FeishuTestButton({ appId, appSecret, t }: { appId: string; appSecret: string; t: ReturnType<typeof getTranslations> }) {
  const [testing, setTesting] = useState(false);
  const [result, setResult] = useState<{ success: boolean; message: string } | null>(null);

  const handleTest = async () => {
    if (!appId || !appSecret) {
      setResult({ success: false, message: t.cfgFeishuCredentialRequired });
      return;
    }

    setTesting(true);
    setResult(null);

    try {
      const res = await testFeishuConnection(appId, appSecret);
      setResult({ success: res.success, message: res.message });
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
          {testing ? t.loading : t.cfgFeishuTestConnection}
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

/** Excel 数据源连接配置（输入/输出文件路径） */
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

/** CSV 数据源连接配置（输入/输出文件路径） */
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

/** MySQL 数据源连接配置（主机、端口、用户、密码、数据库、表名、连接池） */
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

/** PostgreSQL 数据源连接配置（主机、端口、用户、密码、数据库、表名、Schema、连接池） */
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

/** SQLite 数据源连接配置（数据库路径、表名） */
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

/** 飞书多维表格数据源连接配置（App 凭证、应用令牌、表 ID、重试与限流） */
function FeishuBitableConnection({ getConfig, updateConfig, t }: ConnProps) {
  const appId = (getConfig(['feishu', 'app_id']) as string) ?? '';
  const appSecret = (getConfig(['feishu', 'app_secret']) as string) ?? '';
  const appToken = getStringWithDefinedFallback(
    getConfig(['feishu', 'app_token']),
    getConfig(['datasource', 'app_token'])
  );
  const tableId = getStringWithDefinedFallback(
    getConfig(['feishu', 'table_id']),
    getConfig(['datasource', 'table_id'])
  );

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
            value={appToken}
            onChange={(v) => updateConfig(['feishu', 'app_token'], v)}
            placeholder="bascxxxxxxxxxxxxx"
            monospace
          />
        </FormField>
        <FormField label={t.cfgFeishuTableId} required>
          <TextInput
            value={tableId}
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

/** 飞书电子表格数据源连接配置（App 凭证、电子表格令牌、工作表 ID、重试与限流） */
function FeishuSheetConnection({ getConfig, updateConfig, t }: ConnProps) {
  const appId = (getConfig(['feishu', 'app_id']) as string) ?? '';
  const appSecret = (getConfig(['feishu', 'app_secret']) as string) ?? '';
  const spreadsheetToken = getStringWithDefinedFallback(
    getConfig(['feishu', 'spreadsheet_token']),
    getConfig(['datasource', 'spreadsheet_token'])
  );
  const sheetId = getStringWithDefinedFallback(
    getConfig(['feishu', 'sheet_id']),
    getConfig(['datasource', 'sheet_id'])
  );

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
            value={spreadsheetToken}
            onChange={(v) => updateConfig(['feishu', 'spreadsheet_token'], v)}
            placeholder="shtcnxxxxxxxxxxxxx"
            monospace
          />
        </FormField>
        <FormField label={t.cfgFeishuSheetId} required>
          <TextInput
            value={sheetId}
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
