import type { ConfigSectionId } from './ConfigSidebar';
import type { Language } from '../../i18n';
import GlobalSection from './sections/GlobalSection';
import DatasourceSection from './sections/DatasourceSection';
import ConcurrencySection from './sections/ConcurrencySection';
import ColumnsSection from './sections/ColumnsSection';
import ValidationSection from './sections/ValidationSection';
import ModelsSection from './sections/ModelsSection';
import ChannelsSection from './sections/ChannelsSection';
import PromptSection from './sections/PromptSection';
import TokenSection from './sections/TokenSection';
import RoutingSection from './sections/RoutingSection';

export interface SectionProps {
  formData: Record<string, unknown>;
  updateConfig: (path: string[], value: unknown) => void;
  getConfig: (path: string[]) => unknown;
  language: Language;
}

interface SectionRendererProps extends SectionProps {
  section: ConfigSectionId;
}

export default function SectionRenderer({ section, ...props }: SectionRendererProps) {
  switch (section) {
    case 'global':
      return <GlobalSection {...props} />;
    case 'datasource':
      return <DatasourceSection {...props} />;
    case 'concurrency':
      return <ConcurrencySection {...props} />;
    case 'columns':
      return <ColumnsSection {...props} />;
    case 'validation':
      return <ValidationSection {...props} />;
    case 'models':
      return <ModelsSection {...props} />;
    case 'channels':
      return <ChannelsSection {...props} />;
    case 'prompt':
      return <PromptSection {...props} />;
    case 'token':
      return <TokenSection {...props} />;
    case 'routing':
      return <RoutingSection {...props} />;
    default:
      return null;
  }
}
