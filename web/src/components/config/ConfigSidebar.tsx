import type { Language } from '../../i18n';
import { getTranslations } from '../../i18n';

export type ConfigSectionId =
  | 'global'
  | 'datasource'
  | 'concurrency'
  | 'columns'
  | 'validation'
  | 'models'
  | 'channels'
  | 'prompt'
  | 'routing'
  | 'raw';

interface ConfigSidebarProps {
  activeSection: ConfigSectionId;
  onSelect: (section: ConfigSectionId) => void;
  language: Language;
}

interface SidebarItem {
  id: ConfigSectionId;
  icon: React.ReactNode;
}

const sectionIcons: Record<ConfigSectionId, React.ReactNode> = {
  global: (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.066 2.573c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.573 1.066c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.066-2.573c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
    </svg>
  ),
  datasource: (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
    </svg>
  ),
  concurrency: (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
    </svg>
  ),
  columns: (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
    </svg>
  ),
  validation: (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  ),
  models: (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
    </svg>
  ),
  channels: (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8.111 16.404a5.5 5.5 0 017.778 0M12 20h.01m-7.08-7.071c3.904-3.905 10.236-3.905 14.141 0M1.394 9.393c5.857-5.858 15.355-5.858 21.213 0" />
    </svg>
  ),
  prompt: (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
    </svg>
  ),
  routing: (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
    </svg>
  ),
  raw: (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
    </svg>
  ),
};

const sectionIds: ConfigSectionId[] = [
  'global', 'datasource', 'concurrency', 'columns', 'validation',
  'models', 'channels', 'prompt', 'routing',
];

export default function ConfigSidebar({ activeSection, onSelect, language }: ConfigSidebarProps) {
  const t = getTranslations(language);

  const sectionLabels: Record<ConfigSectionId, string> = {
    global: t.cfgGlobal,
    datasource: t.cfgDatasource,
    concurrency: t.cfgConcurrency,
    columns: t.cfgColumns,
    validation: t.cfgValidation,
    models: t.cfgModels,
    channels: t.cfgChannels,
    prompt: t.cfgPrompt,
    routing: t.cfgRouting,
    raw: t.rawYamlEditor,
  };

  const items: SidebarItem[] = sectionIds.map(id => ({
    id,
    icon: sectionIcons[id],
  }));

  return (
    <nav className="bg-white rounded-2xl shadow-[0_2px_12px_rgba(0,0,0,0.04)] p-2 sticky top-6">
      <div className="space-y-0.5">
        {items.map((item) => (
          <button
            key={item.id}
            onClick={() => onSelect(item.id)}
            className={`w-full flex items-center gap-2.5 px-3 py-2 text-sm rounded-lg transition-colors text-left ${
              activeSection === item.id
                ? 'text-cyan-600 bg-cyan-50 font-medium'
                : 'text-gray-600 hover:text-gray-800 hover:bg-gray-50'
            }`}
          >
            {item.icon}
            {sectionLabels[item.id]}
          </button>
        ))}
      </div>

      {/* Divider */}
      <div className="my-2 border-t border-gray-100" />

      {/* Raw YAML */}
      <button
        onClick={() => onSelect('raw')}
        className={`w-full flex items-center gap-2.5 px-3 py-2 text-sm rounded-lg transition-colors text-left ${
          activeSection === 'raw'
            ? 'text-cyan-600 bg-cyan-50 font-medium'
            : 'text-gray-600 hover:text-gray-800 hover:bg-gray-50'
        }`}
      >
        {sectionIcons.raw}
        {sectionLabels.raw}
      </button>
    </nav>
  );
}
