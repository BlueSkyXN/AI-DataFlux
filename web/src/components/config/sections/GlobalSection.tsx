import type { SectionProps } from '../SectionRenderer';
import { getTranslations } from '../../../i18n';
import SectionCard from '../shared/SectionCard';
import FormField from '../shared/FormField';
import TextInput from '../shared/TextInput';
import NumberInput from '../shared/NumberInput';
import SelectDropdown from '../shared/SelectDropdown';

export default function GlobalSection({ updateConfig, getConfig, language }: SectionProps) {
  const t = getTranslations(language);

  const logLevel = (getConfig(['global', 'log', 'level']) as string) ?? 'info';
  const logFormat = (getConfig(['global', 'log', 'format']) as string) ?? 'text';
  const logOutput = (getConfig(['global', 'log', 'output']) as string) ?? 'console';
  const logFilePath = (getConfig(['global', 'log', 'file_path']) as string) ?? '';
  const fluxApiUrl = (getConfig(['global', 'flux_api_url']) as string) ?? '';
  const gwMaxConn = (getConfig(['gateway', 'max_connections']) as number) ?? 1000;
  const gwMaxConnPerHost = (getConfig(['gateway', 'max_connections_per_host']) as number) ?? 1000;

  return (
    <div className="space-y-4">
      {/* API URL */}
      <SectionCard title={t.cfgApiGatewayUrl} description={t.cfgApiGatewayUrlDesc}>
        <FormField label="Flux API URL" required>
          <TextInput
            value={fluxApiUrl}
            onChange={(v) => updateConfig(['global', 'flux_api_url'], v)}
            placeholder="http://127.0.0.1:8787"
            monospace
          />
        </FormField>
      </SectionCard>

      {/* Log Settings */}
      <SectionCard title={t.cfgLogSettings}>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <FormField label={t.cfgLogLevel}>
            <SelectDropdown
              value={logLevel}
              onChange={(v) => updateConfig(['global', 'log', 'level'], v)}
              options={[
                { value: 'debug', label: 'Debug' },
                { value: 'info', label: 'Info' },
                { value: 'warning', label: 'Warning' },
                { value: 'error', label: 'Error' },
              ]}
            />
          </FormField>
          <FormField label={t.cfgLogFormat}>
            <SelectDropdown
              value={logFormat}
              onChange={(v) => updateConfig(['global', 'log', 'format'], v)}
              options={[
                { value: 'text', label: 'Text' },
                { value: 'json', label: 'JSON' },
              ]}
            />
          </FormField>
          <FormField label={t.cfgLogOutput}>
            <SelectDropdown
              value={logOutput}
              onChange={(v) => updateConfig(['global', 'log', 'output'], v)}
              options={[
                { value: 'console', label: 'Console' },
                { value: 'file', label: 'File' },
              ]}
            />
          </FormField>
          {logOutput === 'file' && (
            <FormField label={t.cfgLogFilePath}>
              <TextInput
                value={logFilePath}
                onChange={(v) => updateConfig(['global', 'log', 'file_path'], v)}
                placeholder="./logs/ai_dataflux.log"
                monospace
              />
            </FormField>
          )}
        </div>
      </SectionCard>

      {/* Gateway Connection */}
      <SectionCard title={t.cfgGatewayConnection} description={t.cfgGatewayConnectionDesc}>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <FormField label={t.cfgMaxConnections}>
            <NumberInput
              value={gwMaxConn}
              onChange={(v) => updateConfig(['gateway', 'max_connections'], v)}
              min={1}
            />
          </FormField>
          <FormField label={t.cfgMaxConnectionsPerHost}>
            <NumberInput
              value={gwMaxConnPerHost}
              onChange={(v) => updateConfig(['gateway', 'max_connections_per_host'], v)}
              min={0}
            />
          </FormField>
        </div>
      </SectionCard>
    </div>
  );
}
